"""
Multi-GPU offline preprocessing orchestrator.

Uses the abstract BaseDatasetPreprocessor API so any registered dataset can be
processed with the same script. After processing, per-dataset and combined
statistics are printed and saved.

Usage — single dataset:
    torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \\
        --dataset robobrain-dex \\
        --image_dir /share/project/hotel/.../robobrain-dex \\
        --output_dir /share/project/congsheng/robobrain-dex-qwen-embedding \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \\
        --batch_size 16

Usage — multiple datasets in one launch (sequential, each on all GPUs):
    torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt

preprocess_config.yaml format:
    output_root: /share/project/congsheng/all-embeddings
    datasets:
      - name: robobrain-dex           # dataset name (used as output subdir)
        image_dir: /share/project/hotel/.../robobrain-dex
        type: robobrain-dex           # schema type (registry key); omit if same as name
      - name: your-actual-dataset-a   # any name you choose
        image_dir: /share/.../dataset-a
        type: lerobot                 # schema type shared by all LeRobot datasets
        camera_key: image             # optional; default "image"
      - name: your-actual-dataset-b
        image_dir: /share/.../dataset-b
        type: lerobot

Supported schema types (PREPROCESSORS registry keys)
-----------------------------------------------------
robobrain-dex:
    {image_root}/{task_name}/videos/chunk-000/observation.images.image_top/episode_*/image_*.jpg
    Text: task_name folder name.  No meta file needed.

lerobot:
    {image_root}/meta/episodes.jsonl   ← task per episode
    {image_root}/videos/chunk-*/observation.images.{camera_key}/episode_*/image_*.jpg
    Supports any dataset name; add new datasets only in the YAML, no code change.
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from tqdm import tqdm

from models.qwen_visual_encoder import QwenVisualEncoder
from data.preprocessors import PREPROCESSORS, BaseDatasetPreprocessor, SampleMeta
from data.stats import (
    DatasetStats,
    compute_output_size,
    print_dataset_stats,
    print_combined_stats,
    save_stats,
    save_combined_stats,
)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def shard(samples, rank, world_size):
    return samples[rank::world_size]


def process_dataset(
    preprocessor: BaseDatasetPreprocessor,
    encoder: QwenVisualEncoder,
    device: torch.device,
    batch_size: int,
    rank: int,
    world_size: int,
    is_main: bool,
) -> DatasetStats:
    """Run preprocessing for one dataset across all ranks. Returns stats (rank 0 only)."""

    if is_main:
        print(f"\n[{preprocessor.dataset_name}] Finding samples ...")
    dist.barrier()

    all_samples = preprocessor.find_samples()

    if is_main:
        print(f"[{preprocessor.dataset_name}] Found {len(all_samples):,} samples")

    my_samples = shard(all_samples, rank, world_size)

    # Scan the output directory ONCE per rank to build a set of existing .pt paths.
    # This replaces N individual stat() calls (one per sample) with a single
    # os.walk pass, which is orders of magnitude faster for millions of files.
    if is_main:
        print(f"[{preprocessor.dataset_name}] Scanning existing embeddings ...")
    processed_set = preprocessor.build_processed_set()

    # Count already-done samples via O(1) set lookup instead of per-file stat().
    already_done = sum(1 for s in my_samples if preprocessor.is_processed(s, processed_set))

    t0 = time.time()
    written_local = 0
    key_counts_local: dict[str, int] = {}

    pbar = tqdm(
        range(0, len(my_samples), batch_size),
        desc=f"[GPU {rank}] {preprocessor.dataset_name}",
        disable=(rank != 0),
        dynamic_ncols=True,
    )

    for i in pbar:
        batch = my_samples[i : i + batch_size]
        n_written = preprocessor.process_batch(batch, encoder, device, processed_set=processed_set)
        written_local += n_written

        # Accumulate group counts (all samples, not just written)
        for s in batch:
            key = preprocessor.stats_key(s)
            key_counts_local[key] = key_counts_local.get(key, 0) + 1

    dist.barrier()
    elapsed = time.time() - t0

    # Aggregate stats on rank 0 via all_reduce / gather
    stats = DatasetStats(dataset_name=preprocessor.dataset_name)

    if dist.is_available() and dist.is_initialized():
        # Reduce scalar counts
        def reduce_int(val: int) -> int:
            t = torch.tensor(val, dtype=torch.long, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        total_samples = reduce_int(len(my_samples))
        total_written = reduce_int(written_local)
        total_skipped = reduce_int(already_done)

        if is_main:
            stats.total_frames = total_samples
            stats.written_frames_override = total_written  # not a real field, just for display
            stats.skipped_frames = total_skipped
    else:
        stats.total_frames = len(my_samples)
        stats.skipped_frames = already_done

    # Gather per-group counts to rank 0
    all_key_counts = [None] * world_size
    dist.all_gather_object(all_key_counts, key_counts_local)

    if is_main:
        merged: dict[str, int] = {}
        for kc in all_key_counts:
            for k, v in kc.items():
                merged[k] = merged.get(k, 0) + v
        stats.frames_per_group = merged
        stats.processing_time_seconds = elapsed
        stats.output_size_bytes = compute_output_size(
            preprocessor.output_root, preprocessor.dataset_name
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Offline Qwen visual embedding extraction")
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="Path to standalone Qwen visual encoder .pt")
    parser.add_argument("--batch_size", type=int, default=16)

    # Single-dataset mode
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset name. Available: {list(PREPROCESSORS.keys())}")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # Multi-dataset mode
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config for multi-dataset preprocessing")

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # Build job list — each entry is a dict with name, image_dir, output_root,
    # schema_type, and any extra kwargs forwarded to the preprocessor constructor.
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        output_root = Path(cfg["output_root"])
        jobs = []
        for d in cfg["datasets"]:
            schema_type = d.get("type", d["name"])  # fall back to name if type omitted
            extra = {k: v for k, v in d.items()
                     if k not in ("name", "image_dir", "type")}
            jobs.append({
                "name":        d["name"],
                "image_dir":   d["image_dir"],
                "output_root": output_root,
                "type":        schema_type,
                "extra":       extra,
            })
    elif args.dataset and args.image_dir and args.output_dir:
        jobs = [{
            "name":        args.dataset,
            "image_dir":   args.image_dir,
            "output_root": Path(args.output_dir),
            "type":        args.dataset,
            "extra":       {},
        }]
    else:
        parser.error("Provide either --config or (--dataset + --image_dir + --output_dir)")

    # Validate schema types
    for job in jobs:
        if job["type"] not in PREPROCESSORS:
            raise ValueError(
                f"Unknown schema type '{job['type']}' for dataset '{job['name']}'. "
                f"Available types: {list(PREPROCESSORS.keys())}. "
                f"Register new schemas in data/preprocessors/__init__.py."
            )

    if is_main:
        print(f"Running on {world_size} GPU(s)")
        print(f"Loading Qwen visual encoder from {args.encoder_ckpt} ...")

    encoder = QwenVisualEncoder.from_standalone(args.encoder_ckpt).to(device)
    encoder.eval()

    all_stats = []

    for job in jobs:
        preprocessor_cls = PREPROCESSORS[job["type"]]
        # Pass name and any extra YAML keys (e.g. camera_key) to the constructor.
        # RoboBrainDexPreprocessor ignores unknown kwargs via **extra (see below).
        try:
            preprocessor = preprocessor_cls(
                job["image_dir"], str(job["output_root"]),
                name=job["name"], **job["extra"],
            )
        except TypeError:
            # Schema class (e.g. RoboBrainDexPreprocessor) has a fixed name —
            # fall back to positional-only construction.
            preprocessor = preprocessor_cls(job["image_dir"], str(job["output_root"]))

        stats = process_dataset(
            preprocessor, encoder, device,
            args.batch_size, rank, world_size, is_main,
        )

        if is_main:
            print_dataset_stats(stats)
            save_stats(stats, job["output_root"])
            all_stats.append(stats)

    if is_main and len(all_stats) > 0:
        if len(all_stats) > 1:
            print_combined_stats(all_stats)
        combined_root = jobs[0]["output_root"]
        save_combined_stats(all_stats, combined_root)
        print(f"\nStats saved to {combined_root}/combined_stats.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
