"""
Single-node multi-GPU offline preprocessing.

Streaming / chunked version with:
    - constant RAM usage
    - streaming dataset scan
    - ONE-TIME processed_set scan
    - incremental processed_set updates
    - no torch.distributed / NCCL

Pipeline
--------
dataset start:
    scan existing embeddings ONCE

loop:
    scan next chunk
    process chunk
    update processed_set incrementally
"""

import os
import sys
import queue
import threading
import argparse
import time
import yaml
import itertools
import multiprocessing as mp

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
# Prefetch producer
# ---------------------------------------------------------------------------

def _prefetch_worker(
    preprocessor: BaseDatasetPreprocessor,
    shard: list,
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

        images = list(
            pool.map(
                preprocessor.load_image,
                raw_batch,
            )
        )

        out_q.put(
            (
                "batch",
                list(raw_batch),
                images,
            )
        )

        raw_batch = []

    try:

        with ThreadPoolExecutor(
            max_workers=num_io_workers
        ) as pool:

            for sample in shard:

                total_seen += 1

                key = preprocessor.stats_key(sample)

                key_counts[key] = (
                    key_counts.get(key, 0) + 1
                )

                if preprocessor.is_processed(
                    sample,
                    processed_set,
                ):
                    already_done += 1
                    continue

                raw_batch.append(sample)

                if len(raw_batch) >= batch_size:
                    _flush(pool)

            _flush(pool)

    except Exception as exc:

        out_q.put(
            (
                "error",
                str(exc),
            )
        )

        return

    out_q.put(
        (
            "done",
            {
                "total_seen": total_seen,
                "already_done": already_done,
                "key_counts": key_counts,
            },
        )
    )


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def _gpu_worker(
    rank: int,
    jobs: list,
    pre_scanned: list,
    encoder_ckpt: str,
    batch_size: int,
    num_io_workers: int,
    prefetch_batches: int,
    result_queue: mp.Queue,
    compile_encoder: bool,
) -> None:

    device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)

    encoder = QwenVisualEncoder.from_standalone(
        encoder_ckpt
    ).to(device)

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
            pass

    job_results = []

    for job, scan in zip(jobs, pre_scanned):

        preprocessor_cls = PREPROCESSORS[job["type"]]

        try:

            preprocessor = preprocessor_cls(
                job["image_dir"],
                str(job["output_root"]),
                name=job["name"],
                **job["extra"],
            )

        except TypeError:

            preprocessor = preprocessor_cls(
                job["image_dir"],
                str(job["output_root"]),
            )

        shard = scan["shard"]

        processed_set: set[str] = set(
            scan["processed_set"]
        )

        t0 = time.time()

        written_local = 0

        new_paths = []

        pbar = tqdm(
            desc=f"[GPU {rank}] {preprocessor.dataset_name}",
            disable=(rank != 0),
            dynamic_ncols=True,
            unit="frames",
            total=len(shard),
        )

        prefetch_q: queue.Queue = queue.Queue(
            maxsize=prefetch_batches
        )

        producer = threading.Thread(
            target=_prefetch_worker,
            args=(
                preprocessor,
                shard,
                batch_size,
                num_io_workers,
                processed_set,
                prefetch_q,
            ),
            daemon=True,
        )

        producer.start()

        producer_stats = None

        while True:

            item = prefetch_q.get()

            kind = item[0]

            if kind == "batch":

                _, batch_samples, images = item

                patch_embeds = encoder.encode_images(
                    images,
                    device=device,
                ).to(torch.bfloat16)

                for i, sample in enumerate(batch_samples):

                    preprocessor.save_embedding(
                        sample,
                        patch_embeds[i],
                    )

                    path_str = str(
                        preprocessor.output_path(sample)
                    )

                    processed_set.add(path_str)

                    new_paths.append(path_str)

                written_local += len(batch_samples)

                pbar.update(len(batch_samples))

            elif kind == "done":

                producer_stats = item[1]

                break

            elif kind == "error":

                producer.join(timeout=5)

                raise RuntimeError(
                    f"Prefetch worker on GPU {rank} raised: {item[1]}"
                )

        producer.join()

        pbar.close()

        elapsed = time.time() - t0

        job_results.append(
            {
                "dataset_name": scan["dataset_name"],
                "output_root": scan["output_root"],
                "total_seen": producer_stats["total_seen"],
                "already_done": producer_stats["already_done"],
                "key_counts": producer_stats["key_counts"],
                "written": written_local,
                "elapsed": elapsed,
                "new_paths": new_paths,
            }
        )

    result_queue.put(
        (
            rank,
            job_results,
        )
    )


# ---------------------------------------------------------------------------
# Streaming prescan
# ---------------------------------------------------------------------------

def _prescan_job(
    job: dict,
    num_gpus: int,
    sample_iter,
    scan_chunk_size: int,
    processed_set: set,
):

    preprocessor_cls = PREPROCESSORS[job["type"]]

    try:

        preprocessor = preprocessor_cls(
            job["image_dir"],
            str(job["output_root"]),
            name=job["name"],
            **job["extra"],
        )

    except TypeError:

        preprocessor = preprocessor_cls(
            job["image_dir"],
            str(job["output_root"]),
        )

    print(
        f"[{preprocessor.dataset_name}] "
        f"Scanning next chunk "
        f"(max {scan_chunk_size:,} samples)..."
    )

    chunk_samples = list(
        itertools.islice(
            sample_iter,
            scan_chunk_size,
        )
    )

    exhausted = (
        len(chunk_samples) < scan_chunk_size
    )

    if len(chunk_samples) == 0:
        return None, True

    print(
        f"[{preprocessor.dataset_name}] "
        f"Loaded chunk with "
        f"{len(chunk_samples):,} frames"
    )

    print(
        f"[{preprocessor.dataset_name}] "
        f"Using cached processed_set "
        f"({len(processed_set):,} entries)"
    )

    shards = [
        chunk_samples[i::num_gpus]
        for i in range(num_gpus)
    ]

    return (
        [
            {
                "shard": shards[rank],
                "processed_set": processed_set,
                "dataset_name": preprocessor.dataset_name,
                "output_root": str(job["output_root"]),
            }
            for rank in range(num_gpus)
        ],
        exhausted,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Streaming multi-GPU embedding extraction"
    )

    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_io_workers",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--scan_chunk_size",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
    )

    # single dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    # multi dataset
    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    num_gpus = (
        args.num_gpus
        or torch.cuda.device_count()
    )

    if num_gpus == 0:
        raise RuntimeError(
            "No CUDA GPUs found."
        )

    # -----------------------------------------------------------------------
    # Build jobs
    # -----------------------------------------------------------------------

    if args.config:

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        output_root = Path(cfg["output_root"])

        jobs = []

        for d in cfg["datasets"]:

            schema_type = d.get(
                "type",
                d["name"],
            )

            extra = {
                k: v
                for k, v in d.items()
                if k not in (
                    "name",
                    "image_dir",
                    "type",
                )
            }

            jobs.append(
                {
                    "name": d["name"],
                    "image_dir": d["image_dir"],
                    "output_root": output_root,
                    "type": schema_type,
                    "extra": extra,
                }
            )

    elif (
        args.dataset
        and args.image_dir
        and args.output_dir
    ):

        jobs = [
            {
                "name": args.dataset,
                "image_dir": args.image_dir,
                "output_root": Path(args.output_dir),
                "type": args.dataset,
                "extra": {},
            }
        ]

    else:

        parser.error(
            "Provide either --config "
            "or "
            "(--dataset + --image_dir + --output_dir)"
        )

    # -----------------------------------------------------------------------
    # Validate preprocessors
    # -----------------------------------------------------------------------

    for job in jobs:

        if job["type"] not in PREPROCESSORS:

            raise ValueError(
                f"Unknown schema type '{job['type']}'. "
                f"Available: {list(PREPROCESSORS.keys())}"
            )

    compile_tag = (
        "torch.compile=ON"
        if args.compile
        else "torch.compile=OFF"
    )

    print(
        f"Streaming preprocessing | "
        f"GPUs={num_gpus} | "
        f"batch={args.batch_size} | "
        f"io_workers/gpu={args.num_io_workers} | "
        f"prefetch={args.prefetch_batches} | "
        f"scan_chunk={args.scan_chunk_size:,} | "
        f"{compile_tag}"
    )

    # -----------------------------------------------------------------------
    # Process datasets sequentially
    # -----------------------------------------------------------------------

    all_stats = []

    for job in jobs:

        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {job['name']}")
        print(f"{'=' * 80}")

        preprocessor_cls = PREPROCESSORS[job["type"]]

        try:

            preprocessor = preprocessor_cls(
                job["image_dir"],
                str(job["output_root"]),
                name=job["name"],
                **job["extra"],
            )

        except TypeError:

            preprocessor = preprocessor_cls(
                job["image_dir"],
                str(job["output_root"]),
            )

        # -------------------------------------------------------------------
        # Build sample iterator
        # -------------------------------------------------------------------

        sample_iter = iter(
            preprocessor.iter_samples()
        )

        # -------------------------------------------------------------------
        # Scan processed embeddings ONCE
        # -------------------------------------------------------------------

        print(
            f"[{preprocessor.dataset_name}] "
            f"Scanning existing embeddings ONCE ..."
        )

        processed_set = set(
            preprocessor.build_processed_set()
        )

        print(
            f"[{preprocessor.dataset_name}] "
            f"Found "
            f"{len(processed_set):,} "
            f"cached embeddings"
        )

        # -------------------------------------------------------------------
        # Dataset stats
        # -------------------------------------------------------------------

        dataset_total_seen = 0
        dataset_total_written = 0
        dataset_total_skipped = 0

        dataset_key_counts = {}

        dataset_max_elapsed = 0.0

        dataset_name = None
        output_root = None

        dataset_done = False

        # -------------------------------------------------------------------
        # Chunk loop
        # -------------------------------------------------------------------

        while not dataset_done:

            pre_scanned, exhausted = _prescan_job(
                job=job,
                num_gpus=num_gpus,
                sample_iter=sample_iter,
                scan_chunk_size=args.scan_chunk_size,
                processed_set=processed_set,
            )

            if pre_scanned is None:
                break

            pre_scanned_by_rank = [
                pre_scanned[rank]
                for rank in range(num_gpus)
            ]

            # ---------------------------------------------------------------
            # Spawn workers
            # ---------------------------------------------------------------

            ctx = tmp.get_context("spawn")

            result_queue = ctx.Queue()

            processes = []

            for rank in range(num_gpus):

                p = ctx.Process(
                    target=_gpu_worker,
                    args=(
                        rank,
                        [job],
                        [pre_scanned_by_rank[rank]],
                        args.encoder_ckpt,
                        args.batch_size,
                        args.num_io_workers,
                        args.prefetch_batches,
                        result_queue,
                        args.compile,
                    ),
                    daemon=False,
                )

                p.start()

                processes.append(p)

            # ---------------------------------------------------------------
            # Collect results
            # ---------------------------------------------------------------

            all_rank_results = {}

            for _ in range(num_gpus):

                rank, job_results = result_queue.get()

                all_rank_results[rank] = job_results[0]

            for p in processes:
                p.join()

            # ---------------------------------------------------------------
            # Aggregate stats
            # ---------------------------------------------------------------

            for rank in range(num_gpus):

                r = all_rank_results[rank]

                dataset_name = r["dataset_name"]

                output_root = Path(r["output_root"])

                dataset_total_seen += r["total_seen"]

                dataset_total_written += r["written"]

                dataset_total_skipped += r["already_done"]

                dataset_max_elapsed = max(
                    dataset_max_elapsed,
                    r["elapsed"],
                )

                # incremental processed_set update
                processed_set.update(
                    r["new_paths"]
                )

                for k, v in r["key_counts"].items():

                    dataset_key_counts[k] = (
                        dataset_key_counts.get(k, 0) + v
                    )

            print(
                f"[Chunk Complete] "
                f"seen={dataset_total_seen:,} "
                f"written={dataset_total_written:,} "
                f"cached={dataset_total_skipped:,} "
                f"processed_set={len(processed_set):,}"
            )

            # ---------------------------------------------------------------
            # Cleanup
            # ---------------------------------------------------------------

            del pre_scanned
            del pre_scanned_by_rank
            del all_rank_results

            torch.cuda.empty_cache()

            dataset_done = exhausted

        # -------------------------------------------------------------------
        # Final dataset stats
        # -------------------------------------------------------------------

        stats = DatasetStats(
            dataset_name=dataset_name
        )

        stats.total_frames = dataset_total_seen

        stats.skipped_frames = dataset_total_skipped

        stats.frames_per_group = dataset_key_counts

        stats.processing_time_seconds = dataset_max_elapsed

        stats.output_size_bytes = compute_output_size(
            output_root,
            dataset_name,
        )

        print_dataset_stats(stats)

        save_stats(stats, output_root)

        all_stats.append(stats)

    # -----------------------------------------------------------------------
    # Combined stats
    # -----------------------------------------------------------------------

    if len(all_stats) > 1:
        print_combined_stats(all_stats)

    combined_root = jobs[0]["output_root"]

    save_combined_stats(
        all_stats,
        combined_root,
    )

    print(
        f"\nStats saved to "
        f"{combined_root}/combined_stats.json"
    )


if __name__ == "__main__":
    main()