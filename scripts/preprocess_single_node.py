"""
Single-node multi-GPU offline preprocessing.

Streaming / chunked version with:
    - constant RAM usage
    - streaming dataset scan
    - ONE-TIME processed_set scan
    - incremental processed_set updates
    - no torch.distributed / NCCL
    - persistent GPU workers (encoder loaded once per GPU, never reloaded)

Pipeline
--------
startup:
    spawn N GPU workers (one per GPU)
    each worker loads encoder ONCE and signals ready

dataset loop:
    scan existing embeddings ONCE
    chunk loop:
        scan next chunk
        dispatch chunk shard to each persistent worker via task queue
        collect results from result queue
        update processed_set incrementally

shutdown:
    send poison pill to each worker
    join all worker processes
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
# Persistent per-GPU worker
#
# Loads the encoder ONCE at startup, signals "ready", then loops on
# task_queue processing one chunk shard at a time.  Each task is a
# (job, scan) tuple; None is the poison pill that causes the worker to exit.
# ---------------------------------------------------------------------------

def _gpu_worker(
    rank: int,
    encoder_ckpt: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    compile_encoder: bool,
    batch_size: int,
    num_io_workers: int,
    prefetch_batches: int,
) -> None:

    device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # Load encoder once — stays resident for all chunks / datasets
    # ------------------------------------------------------------------

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

    # Signal that this worker is ready
    result_queue.put((rank, "ready", None))

    # ------------------------------------------------------------------
    # Task loop — process chunks until poison pill
    # ------------------------------------------------------------------

    while True:

        task = task_queue.get()

        if task is None:
            # Poison pill — shut down cleanly
            break

        job, scan = task

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
        producer_error = None

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

                producer_error = item[1]

                break

        producer.join()

        pbar.close()

        elapsed = time.time() - t0

        if producer_error is not None:

            result_queue.put(
                (
                    rank,
                    "error",
                    f"Prefetch worker on GPU {rank} raised: {producer_error}",
                )
            )

        else:

            result_queue.put(
                (
                    rank,
                    "chunk_done",
                    {
                        "dataset_name": scan["dataset_name"],
                        "output_root": scan["output_root"],
                        "total_seen": producer_stats["total_seen"],
                        "already_done": producer_stats["already_done"],
                        "key_counts": producer_stats["key_counts"],
                        "written": written_local,
                        "elapsed": elapsed,
                        "new_paths": new_paths,
                    },
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
    # Spawn persistent GPU workers (encoder loaded ONCE per GPU)
    # -----------------------------------------------------------------------

    ctx = tmp.get_context("spawn")

    result_queue = ctx.Queue()

    task_queues = [ctx.Queue() for _ in range(num_gpus)]

    processes = []

    print(f"Spawning {num_gpus} persistent GPU worker(s) and loading encoder ...")

    for rank in range(num_gpus):

        p = ctx.Process(
            target=_gpu_worker,
            args=(
                rank,
                args.encoder_ckpt,
                task_queues[rank],
                result_queue,
                args.compile,
                args.batch_size,
                args.num_io_workers,
                args.prefetch_batches,
            ),
            daemon=False,
        )

        p.start()

        processes.append(p)

    # Wait until all workers have loaded the encoder and signalled ready
    for _ in range(num_gpus):

        rank, msg, _ = result_queue.get()

        if msg != "ready":
            raise RuntimeError(
                f"GPU {rank} sent unexpected startup message: {msg}"
            )

    print(f"All {num_gpus} GPU worker(s) ready — encoder resident on each GPU.")

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
        # Chunk loop — dispatch to persistent workers via task queues
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

            # Dispatch one shard to each persistent worker
            for rank in range(num_gpus):

                task_queues[rank].put(
                    (job, pre_scanned[rank])
                )

            # Collect results from all workers
            all_rank_results = {}

            for _ in range(num_gpus):

                rank, msg, payload = result_queue.get()

                if msg == "error":

                    # Shut down remaining workers before raising
                    for tq in task_queues:
                        tq.put(None)

                    for p in processes:
                        p.join(timeout=30)

                    raise RuntimeError(payload)

                all_rank_results[rank] = payload

            # Aggregate stats
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

                # Incremental processed_set update
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

            del pre_scanned
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
    # Shut down persistent workers
    # -----------------------------------------------------------------------

    for tq in task_queues:
        tq.put(None)

    for p in processes:
        p.join()

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
