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

Usage — limit frames per run (recommended for millions of files):
    torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt ... \\
        --max_frames_per_cycle 50000

Streaming design:
    Samples are consumed one at a time from iter_samples_for_rank() — no
    upfront list is built in memory.  If --max_frames_per_cycle N is set,
    each run stops after writing N total new frames across all GPUs, then
    exits.  Re-running the same command resumes automatically: already-written
    .pt files are detected via a single os.walk of the output dir (processed_set)
    and skipped in O(1) per sample.

Supported schema types (PREPROCESSORS registry keys)
-----------------------------------------------------
robobrain-dex:
    {image_root}/{task_name}/videos/chunk-000/observation.images.image_top/episode_*/image_*.jpg
    Text: task_name folder name.  No meta file needed.

lerobot:
    {image_root}/meta/episodes.jsonl   <- task per episode
    {image_root}/videos/chunk-*/observation.images.{camera_key}/episode_*/image_*.jpg

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
from data.preprocessors import PREPROCESSORS, BaseDatasetPreprocessor
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


def _reduce_int(val: int, device: torch.device) -> int:
    """All-reduce a single integer across ALL ranks (collective — every rank must call)."""
    t = torch.tensor(val, dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


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
    Run preprocessing for one dataset across all ranks using streaming iteration.

    Samples are consumed one at a time from iter_samples_for_rank() — no
    upfront list is built in memory.  If max_frames_per_cycle > 0, the run
    stops after writing that many total new frames across all GPUs.
    Re-running resumes automatically via the processed_set.

    All dist.all_reduce / dist.all_gather_object calls are unconditional so
    every rank participates in every collective at the same point.

    Returns aggregated DatasetStats (rank-0 fields populated).
    """
    if is_main:
        print(f"\n[{preprocessor.dataset_name}] Scanning existing embeddings ...")

    # One os.walk of the output dir — replaces N individual stat() calls.
    processed_set: set[str] = set(preprocessor.build_processed_set())

    # Per-rank new-frame budget for this run (0 = no limit).
    per_rank_limit = max(1, max_frames_per_cycle // world_size) if max_frames_per_cycle > 0 else 0

    t0 = time.time()
    total_seen_local   = 0  # all samples visited by this rank (new + cached)
    already_done_local = 0  # samples skipped because already in processed_set
    written_local      = 0  # samples newly embedded this run
    key_counts_local: dict[str, int] = {}

    pbar = tqdm(
        desc=f"[GPU {rank}] {preprocessor.dataset_name}",
        disable=(rank != 0),
        dynamic_ncols=True,
        unit="frames",
    )

    batch: list = []

    def flush() -> None:
        nonlocal written_local
        if not batch:
            return
        n = preprocessor.process_batch(batch, encoder, device, processed_set=processed_set)
        written_local += n
        # Update processed_set in-memory so later batches skip these files.
        for s in batch:
            processed_set.add(str(preprocessor.output_path(s)))
        pbar.update(len(batch))
        batch.clear()

    for sample in preprocessor.iter_samples_for_rank(rank, world_size):
        total_seen_local += 1
        key_counts_local[preprocessor.stats_key(sample)] = (
            key_counts_local.get(preprocessor.stats_key(sample), 0) + 1
        )

        if preprocessor.is_processed(sample, processed_set):
            already_done_local += 1
            continue

        batch.append(sample)
        if len(batch) >= batch_size:
            flush()
            if per_rank_limit > 0 and written_local >= per_rank_limit:
                break

    flush()  # remaining partial batch
    pbar.close()

    dist.barrier()
    elapsed = time.time() - t0

    # ---- Aggregate stats — every rank participates in every collective ----
    total_seen_g    = _reduce_int(total_seen_local,   device)
    total_written_g = _reduce_int(written_local,      device)
    total_skipped_g = _reduce_int(already_done_local, device)

    if is_main:
        print(
            f"[{preprocessor.dataset_name}]  "
            f"seen={total_seen_g:,}  written={total_written_g:,}  "
            f"cached={total_skipped_g:,}  ({format_time(elapsed)})"
        )

    all_key_counts = [None] * world_size
    dist.all_gather_object(all_key_counts, key_counts_local)

    stats = DatasetStats(dataset_name=preprocessor.dataset_name)
    if is_main:
        stats.total_frames            = total_seen_g
        stats.skipped_frames          = total_skipped_g
        merged: dict[str, int] = {}
        for kc in all_key_counts:
            for k, v in kc.items():
                merged[k] = merged.get(k, 0) + v
        stats.frames_per_group        = merged
        stats.processing_time_seconds = elapsed
        stats.output_size_bytes       = compute_output_size(
            preprocessor.output_root, preprocessor.dataset_name
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Offline Qwen visual embedding extraction")
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="Path to standalone Qwen visual encoder .pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_frames_per_cycle", type=int, default=0,
        help=(
            "Maximum total new frames to write per run across all GPUs. "
            "0 = no limit (process everything). "
            "Re-run the same command to continue — already-written files are skipped. "
            "Recommended: 50000 for datasets with millions of files."
        ),
    )

    # Single-dataset mode
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset schema type. Available: {list(PREPROCESSORS.keys())}")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # Multi-dataset mode
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config for multi-dataset preprocessing")

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        output_root = Path(cfg["output_root"])
        jobs = []
        for d in cfg["datasets"]:
            schema_type = d.get("type", d["name"])
            extra = {k: v for k, v in d.items() if k not in ("name", "image_dir", "type")}
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
                f"Available: {list(PREPROCESSORS.keys())}. "
                f"Register new schemas in data/preprocessors/__init__.py."
            )

    if is_main:
        print(f"Running on {world_size} GPU(s)")
        if args.max_frames_per_cycle > 0:
            print(
                f"Cycle mode: up to {args.max_frames_per_cycle:,} new frames/run total  "
                f"({args.max_frames_per_cycle // world_size:,} per GPU)  "
                f"— re-run to continue"
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
