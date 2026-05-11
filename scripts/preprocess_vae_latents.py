"""
Offline VAE latent extraction — adds z_vae to existing Qwen embedding .pt files.

For each {name}.pt in the embedding directory:
  • Loads the raw image from data["_abs_image_path"] (written by preprocess_embeddings.py)
  • Encodes it with the frozen VAE: z_vae = vae.encode(image) → (4, H/8, W/8)
  • Writes z_vae back into the same .pt file (in-place update)

Safe to re-run: any .pt that already contains a "z_vae" key is skipped.

Motivation:
  Storing the latent (4×56×56 = 12.5 K values) instead of loading a JPEG image
  (3×448×448 = 602 K values) during training gives:
    • 48× smaller I/O per sample from the DataLoader workers
    • Removes vae.encode() (~25 ms/step) from the training loop
    • Removes qwen_enc and VAE from the training GPU (saves ~5 GB VRAM)
    • Allows batch_size_per_gpu to grow ~2–4× at the same memory budget

Usage:
    torchrun --nproc_per_node=8 scripts/preprocess_vae_latents.py \\
        --embedding_dir /share/project/congsheng/robobrain-dex-qwen-embedding \\
        --unet_name Manojb/stable-diffusion-2-1-base \\
        --resolution 448 \\
        --batch_size 64

    # To process multiple directories:
    torchrun --nproc_per_node=8 scripts/preprocess_vae_latents.py \\
        --config scripts/preprocess_config.yaml \\
        --unet_name Manojb/stable-diffusion-2-1-base

Config YAML format (same preprocess_config.yaml used for Qwen embeddings):
    output_root: /share/project/congsheng/WMDEC_qwen
    datasets:
      - name: robobrain-dex
        image_dir: ...   # not needed here (image path is stored in .pt)
        output_dir: /share/project/congsheng/robobrain-dex-qwen-embedding
"""

import os
import sys
import glob
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.vae import VAEWrapper


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _image_transform(resolution: int):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def process_embedding_dir(
    embedding_dir: str,
    vae: VAEWrapper,
    device: torch.device,
    rank: int,
    world_size: int,
    batch_size: int,
    resolution: int,
):
    """Add z_vae to every .pt file in embedding_dir that doesn't already have it."""

    # Collect all .pt files (excluding stats.json)
    all_pts = sorted(
        p for p in glob.glob(os.path.join(embedding_dir, "**", "*.pt"), recursive=True)
        if os.path.basename(p) not in ("stats.json",)
    )

    # Shard across ranks (simple strided split — no communication needed)
    my_pts = all_pts[rank::world_size]

    transform = _image_transform(resolution)

    pending_paths = []
    pending_images = []
    skipped = 0
    written = 0
    errors = 0

    pbar = tqdm(
        total=len(my_pts),
        desc=f"[GPU {rank}] VAE encode",
        disable=(rank != 0),
        dynamic_ncols=True,
        unit="files",
    )

    def flush():
        nonlocal written, errors
        if not pending_paths:
            return

        # Stack images and encode in one VAE forward pass
        batch_tensor = torch.stack(pending_images).to(device)  # (N, 3, H, W)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z_vae_batch = vae.encode(batch_tensor)          # (N, 4, H/8, W/8)
        except Exception as e:
            if rank == 0:
                print(f"  VAE encode error on batch of {len(pending_paths)}: {e}")
            errors += len(pending_paths)
            pending_paths.clear()
            pending_images.clear()
            return

        z_vae_batch = z_vae_batch.cpu().to(torch.float16)

        for pt_path, z_vae in zip(pending_paths, z_vae_batch):
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                data["z_vae"] = z_vae
                torch.save(data, pt_path)
                written += 1
            except Exception as e:
                if rank == 0:
                    print(f"  Failed to save {pt_path}: {e}")
                errors += 1

        pending_paths.clear()
        pending_images.clear()
        pbar.update(len(pending_paths) + len(pending_images))  # already cleared, update below

    count = 0
    for pt_path in my_pts:
        pbar.update(1)
        count += 1

        # Quick check: is z_vae already present?
        try:
            probe = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            errors += 1
            continue

        if "z_vae" in probe:
            skipped += 1
            continue

        # Load image
        img_path = probe.get("_abs_image_path")
        if img_path is None or not os.path.exists(img_path):
            errors += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
        except Exception:
            errors += 1
            continue

        pending_paths.append(pt_path)
        pending_images.append(img_tensor)

        if len(pending_paths) >= batch_size:
            flush()

    flush()  # remaining partial batch
    pbar.close()

    # Gather summary from all ranks
    t_written = torch.tensor(written,  dtype=torch.long, device=device)
    t_skipped = torch.tensor(skipped,  dtype=torch.long, device=device)
    t_errors  = torch.tensor(errors,   dtype=torch.long, device=device)
    dist.all_reduce(t_written, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_skipped, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_errors,  op=dist.ReduceOp.SUM)

    if rank == 0:
        print(
            f"  [{embedding_dir}]  "
            f"written={t_written.item():,}  skipped(existing)={t_skipped.item():,}  "
            f"errors={t_errors.item():,}  total={len(all_pts):,}"
        )


def main():
    parser = argparse.ArgumentParser(description="Add z_vae latents to existing embedding .pt files")
    parser.add_argument("--unet_name", type=str, required=True,
                        help="HF model name, e.g. Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="VAE encode batch size per GPU")

    # Single-directory mode
    parser.add_argument("--embedding_dir", type=str, default=None,
                        help="Directory containing .pt embedding files (recursive)")

    # Multi-directory mode
    parser.add_argument("--config", type=str, default=None,
                        help="Same YAML used by preprocess_embeddings.py (reads embedding_dir per dataset)")

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if is_main:
        print(f"Loading VAE from {args.unet_name} ...")

    vae = VAEWrapper(args.unet_name).to(device)
    vae.eval()

    # Collect directories to process
    dirs_to_process = []
    if args.embedding_dir:
        dirs_to_process.append(args.embedding_dir)
    elif args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for d in cfg.get("datasets", []):
            # Support both 'embedding_dir' and 'output_dir' keys
            emb_dir = d.get("embedding_dir") or d.get("output_dir")
            if emb_dir:
                dirs_to_process.append(emb_dir)
            else:
                # Derive from output_root + dataset name
                output_root = cfg.get("output_root", "")
                dirs_to_process.append(os.path.join(output_root, d["name"]))
    else:
        parser.error("Provide either --embedding_dir or --config")

    for emb_dir in dirs_to_process:
        if is_main:
            print(f"\nProcessing: {emb_dir}")
        dist.barrier()
        process_embedding_dir(
            emb_dir, vae, device, rank, world_size, args.batch_size, args.resolution
        )

    dist.barrier()
    dist.destroy_process_group()
    if is_main:
        print("\nDone.")


if __name__ == "__main__":
    main()
