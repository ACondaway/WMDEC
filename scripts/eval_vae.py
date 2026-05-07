"""
VAE Reconstruction Quality Evaluation
======================================

Measures how well the project's pretrained VAE (encode → decode round-trip)
preserves image content, using PSNR (↑ better) and LPIPS (↓ better).

This is a pre-training sanity check: if the VAE itself cannot reconstruct
images faithfully there is a hard upper bound on what the full diffusion
decoder can achieve.

Usage
-----
    # Evaluate using datasets from config
    python scripts/eval_vae.py --config configs/base.yaml

    # Evaluate a specific image directory instead
    python scripts/eval_vae.py --config configs/base.yaml \
        --image_dir /path/to/images --num_samples 500

    # Multi-GPU (all ranks cooperate, rank-0 prints final summary)
    torchrun --nproc_per_node=4 scripts/eval_vae.py --config configs/base.yaml

Output
------
    Per-batch progress is printed to stdout.
    A final JSON summary is saved to --output_json (default: vae_eval_results.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Resolve project root so imports work regardless of cwd
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.vae import VAEWrapper
from evaluation.metrics import LPIPSMetric, MetricAccumulator, compute_psnr


# ---------------------------------------------------------------------------
# Image dataset  (raw images, no embeddings needed)
# ---------------------------------------------------------------------------

class ImageFolderFlat(Dataset):
    """
    Walks one or more directories recursively and returns all JPEG/PNG images.

    Args:
        roots:       List of (dataset_name, root_path) tuples.
        resolution:  Images are resized and centre-cropped to this size.
        max_samples: Cap on total images (random subset, deterministic seed).
    """

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        roots: List[Tuple[str, str]],
        resolution: int = 512,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        # Collect (dataset_name, abs_path) pairs
        all_samples: List[Tuple[str, str]] = []
        for ds_name, root in roots:
            for dirpath, _, fnames in os.walk(root):
                for fname in fnames:
                    if Path(fname).suffix.lower() in self.EXTS:
                        all_samples.append((ds_name, os.path.join(dirpath, fname)))

        if not all_samples:
            raise ValueError(f"No images found under: {[r for _, r in roots]}")

        if max_samples and max_samples < len(all_samples):
            rng = random.Random(seed)
            all_samples = rng.sample(all_samples, max_samples)

        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ds_name, path = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return {
            "image":        self.transform(img),
            "dataset_name": ds_name,
        }


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _all_reduce_scalar(value: float, device: torch.device) -> float:
    if not _is_dist():
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def _all_reduce_int(value: int, device: torch.device) -> int:
    return int(_all_reduce_scalar(float(value), device))


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vae(
    vae: VAEWrapper,
    lpips_fn: LPIPSMetric,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    log_every: int = 10,
) -> MetricAccumulator:
    """
    Run the VAE encode → decode round-trip on every batch and accumulate metrics.

    Returns a MetricAccumulator populated with PSNR and LPIPS per dataset.
    """
    acc = MetricAccumulator()
    vae.eval()

    for batch_idx, batch in enumerate(loader):
        images: torch.Tensor = batch["image"].to(device)     # (B, 3, H, W) in [-1, 1]
        ds_names: List[str]  = batch["dataset_name"]

        # VAE round-trip
        latents = vae.encode(images)
        recon   = vae.decode(latents)                         # (B, 3, H, W) in [-1, 1]

        # Per-image metrics
        psnr_vals  = compute_psnr(recon, images)             # (B,)
        lpips_vals = lpips_fn(recon, images)                 # (B,)

        # Accumulate — group by dataset name
        for i in range(len(images)):
            acc.update(
                psnr=psnr_vals[i].item(),
                lpips_val=lpips_vals[i].item(),
                n=1,
                key=ds_names[i],
            )

        if rank == 0 and (batch_idx + 1) % log_every == 0:
            print(
                f"  batch {batch_idx+1:4d}/{len(loader)}  "
                f"PSNR={acc.mean_psnr:.2f} dB  LPIPS={acc.mean_lpips:.4f}  "
                f"n={acc.count:,}",
                flush=True,
            )

    return acc


# ---------------------------------------------------------------------------
# Distributed aggregation of MetricAccumulator
# ---------------------------------------------------------------------------

def _gather_accumulator(local: MetricAccumulator, device: torch.device) -> MetricAccumulator:
    """
    Merge per-rank MetricAccumulators into one global accumulator on rank 0.
    All ranks must call this simultaneously.
    """
    all_dicts: List[Optional[dict]] = [None] * _world_size()
    dist.all_gather_object(all_dicts, local.to_dict())

    if _rank() != 0:
        return local  # non-zero ranks don't need the merged result

    merged = MetricAccumulator()
    for rank_dict in all_dicts:
        # Reconstruct a dummy accumulator from the serialised dict
        n = rank_dict["n_images"]
        if n == 0:
            continue
        merged.update(rank_dict["psnr_db"], rank_dict["lpips"], n=n)
        for ds_name, ds_dict in rank_dict.get("per_dataset", {}).items():
            ds_n = ds_dict["n_images"]
            if ds_n > 0:
                merged.update(ds_dict["psnr_db"], ds_dict["lpips"], n=ds_n, key=ds_name)
    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction quality (PSNR + LPIPS)."
    )
    p.add_argument("--config", required=True, help="Path to configs/base.yaml")
    p.add_argument(
        "--image_dir", default=None,
        help="Override: evaluate a single flat image directory instead of config datasets.",
    )
    p.add_argument(
        "--dataset_name", default="custom",
        help="Dataset name used when --image_dir is specified.",
    )
    p.add_argument(
        "--num_samples", type=int, default=None,
        help="Max number of images to evaluate (random subset).  Default: all.",
    )
    p.add_argument("--resolution",  type=int, default=None,
                   help="Override resolution (default: from config).")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lpips_net",   default="vgg",
                   choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone network.")
    p.add_argument("--log_every",   type=int, default=10,
                   help="Print progress every N batches.")
    p.add_argument("--output_json", default="vae_eval_results.json",
                   help="Path to write JSON results (rank-0 only).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Distributed init ────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank   = _rank()
    world  = _world_size()

    # ── Config ──────────────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    resolution = args.resolution or config["data"]["resolution"]

    # ── Dataset ─────────────────────────────────────────────────────────────
    if args.image_dir:
        roots = [(args.dataset_name, args.image_dir)]
    else:
        # Pull image_dirs from the config datasets section
        roots = [
            (ds["name"], ds["image_dir"])
            for ds in config["data"]["datasets"]
            if "image_dir" in ds
        ]
        if not roots:
            raise ValueError(
                "No image_dir entries found in config datasets. "
                "Specify --image_dir explicitly."
            )

    if rank == 0:
        print(f"[eval_vae] resolution={resolution}  datasets={[r[0] for r in roots]}")

    dataset = ImageFolderFlat(
        roots=roots,
        resolution=resolution,
        max_samples=args.num_samples,
        seed=args.seed,
    )

    # Each rank evaluates a disjoint slice
    if world > 1:
        # Simple index-based shard (no sampler needed — evaluation is deterministic)
        indices = list(range(rank, len(dataset), world))
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if rank == 0:
        total = len(dataset) * world if world > 1 else len(dataset)
        print(
            f"[eval_vae] {total:,} images total  "
            f"(~{math.ceil(total/world):,}/rank)  "
            f"batch={args.batch_size}  gpus={world}"
        )

    # ── Models ──────────────────────────────────────────────────────────────
    unet_name = config["model"]["unet_name"]
    if rank == 0:
        print(f"[eval_vae] Loading VAE from {unet_name} ...")

    vae      = VAEWrapper(model_name=unet_name).to(device)
    lpips_fn = LPIPSMetric(net=args.lpips_net, device=device)

    if rank == 0:
        print(f"[eval_vae] Running evaluation ...")

    # ── Evaluate ────────────────────────────────────────────────────────────
    local_acc = evaluate_vae(
        vae=vae,
        lpips_fn=lpips_fn,
        loader=loader,
        device=device,
        rank=rank,
        log_every=args.log_every,
    )

    # ── Aggregate across ranks ───────────────────────────────────────────────
    if world > 1:
        global_acc = _gather_accumulator(local_acc, device)
    else:
        global_acc = local_acc

    # ── Report (rank 0 only) ─────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "─" * 60)
        print("[eval_vae] Results:")
        print(global_acc)
        print("─" * 60)

        result_dict = global_acc.to_dict()
        result_dict["config"]     = args.config
        result_dict["resolution"] = resolution
        result_dict["lpips_net"]  = args.lpips_net
        result_dict["backbone"]   = config["model"].get("backbone", "unknown")

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"[eval_vae] Results saved to {out_path}")

    if _is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
