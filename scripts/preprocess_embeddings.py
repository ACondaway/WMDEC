"""
Multi-GPU / Multi-Node offline preprocessing orchestrator.

Uses the abstract BaseDatasetPreprocessor API so any registered dataset can be
processed with the same script. After processing, per-dataset and combined
statistics are printed and saved.

Usage — single dataset (single node):
    torchrun --nproc_per_node=8 scripts/preprocess_embeddings.py \\
        --dataset lerobot_without_text \\
        --image_dir /share/project/hotel/.../my-dataset \\
        --output_dir /share/project/congsheng/my-dataset-qwen-embedding \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \\
        --batch_size 256

Usage — multiple datasets via config (single node):
    torchrun --nproc_per_node=8 scripts/preprocess_embeddings.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt

Usage — multiple nodes (run on EACH node, see scripts/launch_multinode.sh):
    torchrun --nnodes=4 --nproc_per_node=8 \\
        --node_rank=<RANK> --master_addr=<MASTER_IP> --master_port=29500 \\
        scripts/preprocess_embeddings.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \\
        --num_io_workers 48 --prefetch_batches 6

Streaming + prefetch design:
    A background producer thread walks the dataset, skips already-written frames
    (detected via a single os.walk → processed_set), loads image batches in parallel
    using a long-lived ThreadPoolExecutor(num_io_workers), and fills a bounded queue.
    The GPU consumer thread pulls pre-loaded batches and encodes without waiting for
    disk I/O — CPU image loading and GPU encoding overlap continuously.

    Queue depth (prefetch_batches) controls lookahead:
      - Each slot holds one batch of PIL images already in RAM.
      - 4–8 slots typically saturate H100 encode throughput with zero GPU stall.

    Re-running resumes automatically: already-written .pt files are detected via a
    single os.walk of the output dir (processed_set) and skipped in the producer.

Supported schema types (PREPROCESSORS registry keys)
-----------------------------------------------------
lerobot_without_text:
    {image_root}/videos/chunk-*/observation.images.{camera_key}/episode_*/image_*.jpg
    task_name is always "" — no text encoder pass triggered.

lerobot:
    Same layout but reads meta/episodes.jsonl for task text per episode.

robobrain-dex:
    {image_root}/{task_name}/videos/chunk-000/observation.images.image_top/episode_*/image_*.jpg
"""

import os
import sys
import queue
import threading
import argparse
import time
import yaml
from concurrent.futures import ThreadPoolExecutor
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


# ---------------------------------------------------------------------------
# Prefetch producer
# ---------------------------------------------------------------------------

def _prefetch_worker(
    preprocessor: BaseDatasetPreprocessor,
    rank: int,
    world_size: int,
    batch_size: int,
    num_io_workers: int,
    processed_set: set,
    per_rank_limit: int,
    out_q: "queue.Queue",
) -> None:
    """
    Background producer thread.

    Iterates this rank's share of the dataset, filters already-processed samples,
    loads image batches in parallel with a long-lived ThreadPoolExecutor, and puts
    ('batch', samples, images) items into ``out_q``.

    Terminates with a ('done', stats_dict) sentinel so the consumer can collect
    per-rank counts without any shared mutable state or locks.

    Parameters
    ----------
    per_rank_limit : int
        Stop after enqueuing this many new frames (0 = no limit).
    num_io_workers : int
        Thread-pool size for parallel JPEG decoding.  On 192-core H100 nodes
        32–64 workers saturate disk/NFS bandwidth without excessive context switching.
    """
    total_seen: int = 0
    already_done: int = 0
    key_counts: dict[str, int] = {}
    enqueued_frames: int = 0
    raw_batch: list = []

    def _flush(pool: ThreadPoolExecutor) -> None:
        nonlocal raw_batch, enqueued_frames
        if not raw_batch:
            return
        images = list(pool.map(preprocessor.load_image, raw_batch))
        out_q.put(('batch', list(raw_batch), images))
        enqueued_frames += len(raw_batch)
        raw_batch = []

    try:
        with ThreadPoolExecutor(max_workers=num_io_workers) as pool:
            for sample in preprocessor.iter_samples_for_rank(rank, world_size):
                total_seen += 1
                key = preprocessor.stats_key(sample)
                key_counts[key] = key_counts.get(key, 0) + 1

                if preprocessor.is_processed(sample, processed_set):
                    already_done += 1
                    continue

                raw_batch.append(sample)
                if len(raw_batch) >= batch_size:
                    _flush(pool)
                    if per_rank_limit > 0 and enqueued_frames >= per_rank_limit:
                        break

            _flush(pool)  # remaining partial batch

    except Exception as exc:  # noqa: BLE001
        out_q.put(('error', str(exc)))
        return

    out_q.put(('done', {
        'total_seen':  total_seen,
        'already_done': already_done,
        'key_counts':  key_counts,
    }))


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def process_dataset(
    preprocessor: BaseDatasetPreprocessor,
    encoder: QwenVisualEncoder,
    device: torch.device,
    batch_size: int,
    rank: int,
    world_size: int,
    is_main: bool,
    max_frames_per_cycle: int = 0,
    num_io_workers: int = 32,
    prefetch_batches: int = 4,
) -> DatasetStats:
    """
    Run preprocessing for one dataset across all ranks using a prefetch pipeline.

    Design
    ------
    A background producer thread (``_prefetch_worker``) fills a bounded queue with
    pre-loaded image batches.  The GPU consumer (this function's main loop) encodes
    and saves without blocking on image I/O, so CPU loading and GPU encoding run
    concurrently at all times.

    All dist.all_reduce / dist.all_gather_object calls are unconditional so every
    rank participates in every collective at the same point.

    Returns aggregated DatasetStats (rank-0 fields populated).
    """
    if is_main:
        print(f"\n[{preprocessor.dataset_name}] Scanning existing embeddings ...")

    # One os.walk of the output dir — replaces N individual stat() calls.
    processed_set: set[str] = set(preprocessor.build_processed_set())

    # Per-rank new-frame budget for this run (0 = no limit).
    per_rank_limit = (
        max(1, max_frames_per_cycle // world_size) if max_frames_per_cycle > 0 else 0
    )

    t0 = time.time()
    written_local: int = 0

    pbar = tqdm(
        desc=f"[GPU {rank}] {preprocessor.dataset_name}",
        disable=(rank != 0),
        dynamic_ncols=True,
        unit="frames",
    )

    # ---- Launch prefetch producer ----------------------------------------
    prefetch_q: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    producer = threading.Thread(
        target=_prefetch_worker,
        args=(
            preprocessor, rank, world_size, batch_size,
            num_io_workers, processed_set, per_rank_limit, prefetch_q,
        ),
        daemon=True,
        name=f"prefetch-rank{rank}",
    )
    producer.start()

    # ---- GPU consumer loop -----------------------------------------------
    producer_stats: dict | None = None

    while True:
        item = prefetch_q.get()
        kind = item[0]

        if kind == 'batch':
            _, batch_samples, images = item

            # Encode on GPU
            patch_embeds = encoder.encode_images(images, device=device).to(torch.bfloat16)

            # Save embeddings and update processed_set in-memory
            for i, sample in enumerate(batch_samples):
                preprocessor.save_embedding(sample, patch_embeds[i])
                processed_set.add(str(preprocessor.output_path(sample)))

            written_local += len(batch_samples)
            pbar.update(len(batch_samples))

        elif kind == 'done':
            producer_stats = item[1]
            break

        elif kind == 'error':
            producer.join(timeout=5)
            raise RuntimeError(
                f"Prefetch worker on rank {rank} raised an exception: {item[1]}"
            )

    producer.join()
    pbar.close()

    dist.barrier()
    elapsed = time.time() - t0

    # Recover per-rank counts from producer's bookkeeping
    total_seen_local  = producer_stats['total_seen']
    already_done_local = producer_stats['already_done']
    key_counts_local  = producer_stats['key_counts']

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
        stats.total_frames             = total_seen_g
        stats.skipped_frames           = total_skipped_g
        merged: dict[str, int] = {}
        for kc in all_key_counts:
            for k, v in kc.items():
                merged[k] = merged.get(k, 0) + v
        stats.frames_per_group         = merged
        stats.processing_time_seconds  = elapsed
        stats.output_size_bytes        = compute_output_size(
            preprocessor.output_root, preprocessor.dataset_name
        )
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline Qwen visual embedding extraction — multi-GPU / multi-node"
    )
    parser.add_argument("--encoder_ckpt", type=str, required=True,
                        help="Path to standalone Qwen visual encoder .pt")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Images per GPU batch (default: 256)")
    parser.add_argument(
        "--num_io_workers", type=int, default=32,
        help=(
            "ThreadPoolExecutor size for parallel image loading per GPU rank. "
            "Higher values saturate NFS/SSD bandwidth on many-core nodes. "
            "Recommended: 32–64 on 192-core H100 nodes (default: 32)."
        ),
    )
    parser.add_argument(
        "--prefetch_batches", type=int, default=4,
        help=(
            "Number of pre-loaded batches buffered in the queue ahead of the GPU. "
            "Larger values hide I/O latency spikes at the cost of more RAM. "
            "Recommended: 4–8 (default: 4)."
        ),
    )
    parser.add_argument(
        "--max_frames_per_cycle", type=int, default=0,
        help=(
            "Maximum total new frames to write per run across all GPUs. "
            "0 = no limit (process everything). "
            "Re-run the same command to continue — already-written files are skipped."
        ),
    )

    parser.add_argument(
        "--compile", action="store_true", default=False,
        help=(
            "JIT-compile the visual encoder with torch.compile. "
            "Adds ~60 s warm-up on first batch but gives 30-50%% faster encode on H100."
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
        compile_tag = "torch.compile=ON" if args.compile else "torch.compile=OFF"
        print(
            f"Running on {world_size} GPU(s) | io_workers/rank={args.num_io_workers} | "
            f"prefetch_batches={args.prefetch_batches} | batch_size={args.batch_size} | {compile_tag}"
        )
        if args.max_frames_per_cycle > 0:
            print(
                f"Cycle mode: up to {args.max_frames_per_cycle:,} new frames/run total  "
                f"({args.max_frames_per_cycle // world_size:,} per GPU)  — re-run to continue"
            )
        print(f"Loading Qwen visual encoder from {args.encoder_ckpt} ...")

    encoder = QwenVisualEncoder.from_standalone(args.encoder_ckpt).to(device)
    encoder.eval()

    if args.compile:
        if is_main:
            print("Compiling visual encoder with torch.compile(mode='reduce-overhead') ...")
            print("  (first batch incurs warm-up; all subsequent batches are faster)")
        try:
            encoder.visual = torch.compile(
                encoder.visual,
                mode="reduce-overhead",
                dynamic=True,
                fullgraph=False,
            )
            if is_main:
                print("  torch.compile OK")
        except Exception as e:
            if is_main:
                print(f"  torch.compile failed ({e}), continuing without compilation.")

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
            num_io_workers=args.num_io_workers,
            prefetch_batches=args.prefetch_batches,
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
