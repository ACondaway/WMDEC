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

Usage — limit frames per cycle (recommended for millions of files):
    torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt ... \\
        --max_frames_per_cycle 50000

Cycle mode (--max_frames_per_cycle > 0):
    Unprocessed samples are split into chunks of `max_frames_per_cycle` total
    frames (divided evenly across GPUs).  Each cycle is processed sequentially
    with a progress line printed between cycles.  All cycle stats are aggregated
    into the final per-dataset report.  The processed_set from the previous
    cycle is updated in-memory so no re-scanning is needed between cycles.

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
    {image_root}/meta/episodes.jsonl   <- task per episode
    {image_root}/videos/chunk-*/observation.images.{camera_key}/episode_*/image_*.jpg
    Supports any dataset name; add new datasets only in the YAML, no code change.

lerobot_without_text:
    Same layout as lerobot but skips meta/episodes.jsonl entirely.
    task_name is always "" — no text encoder pass triggered.
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
    format_time,
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


def _chunk(lst, n):
    """Split list into chunks of at most n elements."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_dataset(
    preprocessor: BaseDatasetPreprocessor,
    encoder: QwenVisualEncoder,
    device: torch.device,
    batch_size: int,
    rank: int,
    world_size: int,
    is_main: bool,
    max_frames_per_cycle: int = 0,
) -> DatasetStats:
    """
    Run preprocessing for one dataset across all ranks.

    If max_frames_per_cycle > 0, unprocessed samples are split into cycles
    of at most max_frames_per_cycle total frames (across all GPUs).  Each
    cycle is processed sequentially; per-cycle stats are accumulated and
    reported at the end.

    Returns aggregated DatasetStats (populated on rank 0).
    """

    if is_main:
        print(f"\n[{preprocessor.dataset_name}] Finding samples ...")
    dist.barrier()

    all_samples = preprocessor.find_samples()

    if is_main:
        print(f"[{preprocessor.dataset_name}] Found {len(all_samples):,} total samples")

    my_samples = shard(all_samples, rank, world_size)

    # Scan output directory ONCE — replaces N individual stat() calls.
    if is_main:
        print(f"[{preprocessor.dataset_name}] Scanning existing embeddings ...")
    processed_set = set(preprocessor.build_processed_set())  # mutable set; updated each cycle

    # Separate already-done from pending so cycles only touch new work.
    already_done = [s for s in my_samples if preprocessor.is_processed(s, processed_set)]
    pending      = [s for s in my_samples if not preprocessor.is_processed(s, processed_set)]

    if is_main:
        total_pending_global = _reduce_int(len(pending), device)
        total_done_global    = _reduce_int(len(already_done), device)
        print(
            f"[{preprocessor.dataset_name}]  "
            f"pending={total_pending_global:,}  cached={total_done_global:,}"
        )
    else:
        dist.barrier(); dist.barrier()   # match the two reduce calls on main

    # --- Split pending into cycles -------------------------------------------
    # Per-rank limit: max_frames_per_cycle is the *total* frame budget per cycle
    # across all GPUs, so each rank handles max_frames_per_cycle // world_size.
    if max_frames_per_cycle > 0 and len(pending) > 0:
        per_rank_limit = max(1, max_frames_per_cycle // world_size)
        cycles = list(_chunk(pending, per_rank_limit))
    else:
        cycles = [pending] if pending else []

    n_cycles = len(cycles)

    # ---- Accumulate stats across cycles ----
    t0 = time.time()
    written_local_total  = 0
    key_counts_local: dict[str, int] = {}

    for cycle_idx, cycle_samples in enumerate(cycles):
        cycle_t0 = time.time()

        if is_main and n_cycles > 1:
            start_frame = cycle_idx * (max_frames_per_cycle // world_size)
            print(
                f"\n  [{preprocessor.dataset_name}] "
                f"Cycle {cycle_idx + 1}/{n_cycles} — "
                f"~{len(cycle_samples) * world_size:,} frames this cycle"
            )

        pbar = tqdm(
            range(0, len(cycle_samples), batch_size),
            desc=(
                f"[GPU {rank}] {preprocessor.dataset_name}"
                + (f" [{cycle_idx+1}/{n_cycles}]" if n_cycles > 1 else "")
            ),
            disable=(rank != 0),
            dynamic_ncols=True,
        )

        written_cycle = 0
        for i in pbar:
            batch = cycle_samples[i : i + batch_size]
            n_written = preprocessor.process_batch(
                batch, encoder, device, processed_set=processed_set
            )
            written_cycle += n_written

            # Track newly written paths so next cycle's processed_set is current.
            if n_written:
                for s in batch:
                    processed_set.add(str(preprocessor.output_path(s)))

            # Accumulate group counts for this cycle.
            for s in batch:
                key = preprocessor.stats_key(s)
                key_counts_local[key] = key_counts_local.get(key, 0) + 1

        written_local_total += written_cycle

        dist.barrier()

        if is_main and n_cycles > 1:
            cycle_written_global = _reduce_int(written_cycle, device)
            print(
                f"  [{preprocessor.dataset_name}] "
                f"Cycle {cycle_idx + 1}/{n_cycles} done — "
                f"{cycle_written_global:,} written  "
                f"({format_time(time.time() - cycle_t0)})"
            )
        else:
            if not is_main:
                _reduce_int(written_cycle, device)   # participate in reduce

    elapsed = time.time() - t0

    # ---- Aggregate final stats across all ranks ----
    stats = DatasetStats(dataset_name=preprocessor.dataset_name)

    total_samples = _reduce_int(len(my_samples), device)
    total_written = _reduce_int(written_local_total, device)
    total_skipped = _reduce_int(len(already_done), device)

    if is_main:
        stats.total_frames   = total_samples
        stats.skipped_frames = total_skipped

        all_key_counts = [None] * world_size
        dist.all_gather_object(all_key_counts, key_counts_local)
        merged: dict[str, int] = {}
        for kc in all_key_counts:
            for k, v in kc.items():
                merged[k] = merged.get(k, 0) + v
        stats.frames_per_group       = merged
        stats.processing_time_seconds = elapsed
        stats.output_size_bytes = compute_output_size(
            preprocessor.output_root, preprocessor.dataset_name
        )
    else:
        dist.all_gather_object([None] * world_size, key_counts_local)

    return stats


def _reduce_int(val: int, device: torch.device) -> int:
    """All-reduce a single integer across ranks; returns the sum on all ranks."""
    t = torch.tensor(val, dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.barrier()
    return t.item()


def main():
    parser = argparse.ArgumentParser(description="Offline Qwen visual embedding extraction")
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="Path to standalone Qwen visual encoder .pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_frames_per_cycle", type=int, default=0,
        help=(
            "Maximum total frames to process per cycle across all GPUs. "
            "0 = no limit (process everything in one pass). "
            "Recommended: 50000 for datasets with millions of files."
        ),
    )

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

    # Build job list
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        output_root = Path(cfg["output_root"])
        jobs = []
        for d in cfg["datasets"]:
            schema_type = d.get("type", d["name"])
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

    for job in jobs:
        if job["type"] not in PREPROCESSORS:
            raise ValueError(
                f"Unknown schema type '{job['type']}' for dataset '{job['name']}'. "
                f"Available types: {list(PREPROCESSORS.keys())}. "
                f"Register new schemas in data/preprocessors/__init__.py."
            )

    if is_main:
        print(f"Running on {world_size} GPU(s)")
        if args.max_frames_per_cycle > 0:
            print(
                f"Cycle mode: {args.max_frames_per_cycle:,} frames/cycle total  "
                f"({args.max_frames_per_cycle // world_size:,} per GPU)"
            )
        print(f"Loading Qwen visual encoder from {args.encoder_ckpt} ...")

    encoder = QwenVisualEncoder.from_standalone(args.encoder_ckpt).to(device)
    encoder.eval()

    all_stats = []

    for job in jobs:
        preprocessor_cls = PREPROCESSORS[job["type"]]
        try:
            preprocessor = preprocessor_cls(
                job["image_dir"], str(job["output_root"]),
                name=job["name"], **job["extra"],
            )
        except TypeError:
            preprocessor = preprocessor_cls(job["image_dir"], str(job["output_root"]))

        stats = process_dataset(
            preprocessor, encoder, device,
            args.batch_size, rank, world_size, is_main,
            max_frames_per_cycle=args.max_frames_per_cycle,
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
