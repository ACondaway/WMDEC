"""
Single-node multi-GPU offline preprocessing.

Replaces torchrun / torch.distributed entirely. Uses torch.multiprocessing.spawn
to launch one worker process per GPU. Workers are fully independent — no NCCL,
no rendezvous, no collective ops. Each GPU processes its own shard of the dataset
(modulo interleaving via iter_samples_for_rank) and writes embeddings directly to
disk. Stats are returned to the main process via a multiprocessing.Queue.

Usage — single dataset:
    python scripts/preprocess_single_node.py \\
        --dataset lerobot_without_text \\
        --image_dir /share/project/hotel/.../my-dataset \\
        --output_dir /share/project/congsheng/my-dataset-qwen-embedding \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \\
        --num_gpus 8 --batch_size 256

Usage — multiple datasets via config:
    python scripts/preprocess_single_node.py \\
        --config scripts/preprocess_config.yaml \\
        --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \\
        --num_gpus 8

Resume: already-written .pt files are detected and skipped automatically.
"""

import os
import sys
import queue
import threading
import argparse
import time
import yaml
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.multiprocessing as tmp
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


# ---------------------------------------------------------------------------
# Prefetch producer (identical logic to multi-node version)
# ---------------------------------------------------------------------------

def _prefetch_worker(
    preprocessor: BaseDatasetPreprocessor,
    rank: int,
    world_size: int,
    batch_size: int,
    num_io_workers: int,
    processed_set: set,
    out_q: "queue.Queue",
) -> None:
    total_seen = 0
    already_done = 0
    key_counts: dict[str, int] = {}
    raw_batch = []

    def _flush(pool):
        nonlocal raw_batch
        if not raw_batch:
            return
        images = list(pool.map(preprocessor.load_image, raw_batch))
        out_q.put(('batch', list(raw_batch), images))
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

            _flush(pool)

    except Exception as exc:  # noqa: BLE001
        out_q.put(('error', str(exc)))
        return

    out_q.put(('done', {
        'total_seen': total_seen,
        'already_done': already_done,
        'key_counts': key_counts,
    }))


# ---------------------------------------------------------------------------
# Per-GPU worker (spawned by torch.multiprocessing.spawn)
# ---------------------------------------------------------------------------

def _gpu_worker(
    rank: int,
    world_size: int,
    jobs: list,
    encoder_ckpt: str,
    batch_size: int,
    num_io_workers: int,
    prefetch_batches: int,
    result_queue: mp.Queue,
    compile_encoder: bool,
) -> None:
    """
    Runs in a subprocess — one per GPU.
    Processes all jobs sequentially, then puts per-job stats into result_queue.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    encoder = QwenVisualEncoder.from_standalone(encoder_ckpt).to(device)
    encoder.eval()

    if compile_encoder:
        try:
            encoder.visual = torch.compile(
                encoder.visual,
                mode="reduce-overhead",
                dynamic=True,
                fullgraph=False,
            )
        except Exception:
            pass  # fall back silently

    job_results = []

    for job in jobs:
        preprocessor_cls = PREPROCESSORS[job["type"]]
        try:
            preprocessor = preprocessor_cls(
                job["image_dir"], str(job["output_root"]),
                name=job["name"], **job["extra"],
            )
        except TypeError:
            preprocessor = preprocessor_cls(job["image_dir"], str(job["output_root"]))

        processed_set: set[str] = set(preprocessor.build_processed_set())

        t0 = time.time()
        written_local = 0

        pbar = tqdm(
            desc=f"[GPU {rank}] {preprocessor.dataset_name}",
            disable=(rank != 0),
            dynamic_ncols=True,
            unit="frames",
        )

        prefetch_q: queue.Queue = queue.Queue(maxsize=prefetch_batches)
        producer = threading.Thread(
            target=_prefetch_worker,
            args=(preprocessor, rank, world_size, batch_size,
                  num_io_workers, processed_set, prefetch_q),
            daemon=True,
        )
        producer.start()

        producer_stats = None
        while True:
            item = prefetch_q.get()
            kind = item[0]

            if kind == 'batch':
                _, batch_samples, images = item
                patch_embeds = encoder.encode_images(images, device=device).to(torch.bfloat16)
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
                    f"Prefetch worker on GPU {rank} raised: {item[1]}"
                )

        producer.join()
        pbar.close()

        elapsed = time.time() - t0
        job_results.append({
            'dataset_name': preprocessor.dataset_name,
            'output_root': str(job["output_root"]),
            'total_seen': producer_stats['total_seen'],
            'already_done': producer_stats['already_done'],
            'key_counts': producer_stats['key_counts'],
            'written': written_local,
            'elapsed': elapsed,
        })

    result_queue.put((rank, job_results))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Single-node multi-GPU offline Qwen visual embedding extraction"
    )
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_io_workers", type=int, default=32)
    parser.add_argument("--prefetch_batches", type=int, default=4)
    parser.add_argument("--compile", action="store_true", default=False)

    # Single-dataset mode
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Available: {list(PREPROCESSORS.keys())}")
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # Multi-dataset mode
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")

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
                f"Unknown schema type '{job['type']}'. "
                f"Available: {list(PREPROCESSORS.keys())}"
            )

    compile_tag = "torch.compile=ON" if args.compile else "torch.compile=OFF"
    print(
        f"Single-node preprocessing | GPUs={num_gpus} | "
        f"batch={args.batch_size} | io_workers/gpu={args.num_io_workers} | "
        f"prefetch={args.prefetch_batches} | {compile_tag}"
    )

    # Use 'spawn' context — required for CUDA multiprocessing
    ctx = tmp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(num_gpus):
        p = ctx.Process(
            target=_gpu_worker,
            args=(rank, num_gpus, jobs, args.encoder_ckpt,
                  args.batch_size, args.num_io_workers, args.prefetch_batches,
                  result_queue, args.compile),
            daemon=False,
        )
        p.start()
        processes.append(p)

    # Collect results from all workers
    all_rank_results = {}
    for _ in range(num_gpus):
        rank, job_results = result_queue.get()
        all_rank_results[rank] = job_results

    for p in processes:
        p.join()

    # Aggregate stats across ranks per job
    num_jobs = len(jobs)
    all_stats = []

    for job_idx in range(num_jobs):
        job = jobs[job_idx]
        dataset_name = None
        output_root = None
        total_seen = 0
        total_written = 0
        total_skipped = 0
        merged_key_counts: dict[str, int] = {}
        max_elapsed = 0.0

        for rank in range(num_gpus):
            r = all_rank_results[rank][job_idx]
            dataset_name = r['dataset_name']
            output_root = Path(r['output_root'])
            total_seen += r['total_seen']
            total_written += r['written']
            total_skipped += r['already_done']
            max_elapsed = max(max_elapsed, r['elapsed'])
            for k, v in r['key_counts'].items():
                merged_key_counts[k] = merged_key_counts.get(k, 0) + v

        print(
            f"[{dataset_name}]  seen={total_seen:,}  "
            f"written={total_written:,}  cached={total_skipped:,}  "
            f"({format_time(max_elapsed)})"
        )

        stats = DatasetStats(dataset_name=dataset_name)
        stats.total_frames = total_seen
        stats.skipped_frames = total_skipped
        stats.frames_per_group = merged_key_counts
        stats.processing_time_seconds = max_elapsed
        stats.output_size_bytes = compute_output_size(output_root, dataset_name)

        print_dataset_stats(stats)
        save_stats(stats, output_root)
        all_stats.append(stats)

    if len(all_stats) > 1:
        print_combined_stats(all_stats)
    combined_root = jobs[0]["output_root"]
    save_combined_stats(all_stats, combined_root)
    print(f"\nStats saved to {combined_root}/combined_stats.json")


if __name__ == "__main__":
    main()
