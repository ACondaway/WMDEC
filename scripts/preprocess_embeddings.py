"""
Multi-GPU offline preprocessing script to extract SigLIP and T5-XXL embeddings
from the robobrain-dex dataset and store them as .pt files.

Each GPU gets an equal shard of the dataset. Uses torchrun for launching.

Usage:
    torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \
        --image_dir /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex \
        --output_dir /share/project/congsheng/robobrain-dex-siglip-embedding \
        --batch_size 64
"""

import os
import argparse
import glob

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models.siglip_encoder import SigLIPEncoder
from models.text_encoder import T5TextEncoder


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def find_all_images(image_dir: str) -> list[dict]:
    """
    Find all images and extract task names.

    Structure:
        image_dir/{Task_name}/videos/chunk-000/observation.images.image_top/episode_XXXXXX/image_X.0.jpg
    """
    pattern = os.path.join(
        image_dir, "*", "videos", "chunk-000",
        "observation.images.image_top", "episode_*", "*.jpg"
    )
    image_paths = sorted(glob.glob(pattern))

    samples = []
    for path in image_paths:
        rel = os.path.relpath(path, image_dir)
        parts = rel.split(os.sep)
        task_name = parts[0]
        episode = parts[-2]  # episode_XXXXXX
        filename = os.path.splitext(parts[-1])[0]  # image_X.0

        samples.append({
            "image_path": path,
            "task_name": task_name,
            "episode": episode,
            "filename": filename,
            "text": task_name.replace("_", " "),
        })

    return samples


def shard_samples(samples: list[dict], rank: int, world_size: int) -> list[dict]:
    """Split samples evenly across ranks."""
    return samples[rank::world_size]


def process_batch(
    batch_samples: list[dict],
    siglip: SigLIPEncoder,
    text_encoder: T5TextEncoder,
    output_dir: str,
):
    """Process a batch of images and save embeddings."""
    images = []
    for s in batch_samples:
        img = Image.open(s["image_path"]).convert("RGB")
        images.append(img)

    texts = [s["text"] for s in batch_samples]

    # Encode
    z_img = siglip.encode_image_from_raw(images)  # (B, D)
    z_txt = text_encoder.encode(texts)             # (B, T, C)

    # Save individually
    for i, s in enumerate(batch_samples):
        out_dir = os.path.join(output_dir, s["task_name"], s["episode"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{s['filename']}.pt")

        torch.save({
            "z_img": z_img[i].cpu(),
            "z_txt": z_txt[i].cpu(),
            "task_name": s["task_name"],
            "text": s["text"],
        }, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Root image directory (robobrain-dex)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--siglip_model", type=str,
                        default="google/siglip-large-patch16-384")
    parser.add_argument("--t5_model", type=str,
                        default="google/t5-xxl-lm-adapt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Running on {world_size} GPUs")

    # Each GPU loads its own copy of the encoders
    if rank == 0:
        print("Loading SigLIP...")
    siglip = SigLIPEncoder(args.siglip_model).to(device)

    if rank == 0:
        print("Loading T5-XXL...")
    text_encoder = T5TextEncoder(args.t5_model).to(device)

    # Find and shard data
    if rank == 0:
        print("Finding images...")
    samples = find_all_images(args.image_dir)
    if rank == 0:
        print(f"Found {len(samples)} total images")

    my_samples = shard_samples(samples, rank, world_size)
    if rank == 0:
        print(f"~{len(my_samples)} images per GPU")

    # Synchronize before starting
    dist.barrier()

    # Process in batches
    pbar = tqdm(
        range(0, len(my_samples), args.batch_size),
        desc=f"GPU {rank}",
        disable=(rank != 0),
    )
    for i in pbar:
        batch = my_samples[i:i + args.batch_size]
        process_batch(batch, siglip, text_encoder, args.output_dir)

    # Wait for all ranks to finish
    dist.barrier()
    if rank == 0:
        print(f"Done! Embeddings saved to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
