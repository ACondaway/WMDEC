"""
VAE Reconstruction Quality Evaluation
======================================

Measures how well the project's pretrained VAE (encode → decode round-trip)
preserves image content, using PSNR (↑ better) and LPIPS (↓ better).

Image loading reuses the same preprocessor registry and preprocess_config.yaml
used by the offline embedding pipeline, so directory-walking logic is never
duplicated.

Usage
-----
    # Evaluate all datasets declared in preprocess_config.yaml
    python scripts/eval_vae.py \
        --config configs/base.yaml \
        --preprocess_config scripts/preprocess_config.yaml

    # Cap to 500 random images, save grid
    python scripts/eval_vae.py \
        --config configs/base.yaml \
        --preprocess_config scripts/preprocess_config.yaml \
        --num_samples 500 --vis_samples 32

    # 4-GPU
    torchrun --nproc_per_node=4 scripts/eval_vae.py \
        --config configs/base.yaml \
        --preprocess_config scripts/preprocess_config.yaml

Output
------
    Per-batch progress printed to stdout.
    JSON summary  → --output_json  (default: vae_eval_results.json)
    Comparison grid → --vis_output (default: vae_eval_grid.png)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.vae import VAEWrapper
from evaluation.metrics import LPIPSMetric, MetricAccumulator, compute_psnr
from evaluation.visualize import save_comparison_grid
from data.preprocessors import PREPROCESSORS
from data.preprocessors.base import BaseDatasetPreprocessor, SampleMeta


# ---------------------------------------------------------------------------
# Dataset backed by preprocessors
# ---------------------------------------------------------------------------

@dataclass
class _IndexedSample:
    preprocessor: BaseDatasetPreprocessor
    sample: SampleMeta
    ds_name: str


class PreprocessorDataset(Dataset):
    """
    Map-style Dataset that loads raw images using the same preprocessor
    registry and load_image() logic as the offline embedding pipeline.

    Args:
        preprocessors: list of (dataset_name, preprocessor) pairs.
        resolution:    images are resized + centre-cropped to this size.
        max_samples:   optional cap; a random subset is drawn deterministically.
        seed:          RNG seed for the random subset.
    """

    def __init__(
        self,
        preprocessors: List[Tuple[str, BaseDatasetPreprocessor]],
        resolution: int = 512,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.transform = transforms.Compose([
            transforms.Resize(
                resolution,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

        # Collect sample index — just SampleMeta descriptors, no images yet.
        # iter_samples() is a generator so this is memory-efficient even for
        # millions of files; we stop early once max_samples is reached.
        all_samples: List[_IndexedSample] = []
        for ds_name, prep in preprocessors:
            for sample in prep.iter_samples():
                all_samples.append(_IndexedSample(prep, sample, ds_name))
                # Early exit when we already have enough (avoids walking the
                # remainder of large datasets when max_samples is small).
                if max_samples and len(all_samples) >= max_samples * 4:
                    break   # over-collect then subsample for better coverage

        if not all_samples:
            raise ValueError("No images found via the provided preprocessors.")

        if max_samples and len(all_samples) > max_samples:
            rng = random.Random(seed)
            all_samples = rng.sample(all_samples, max_samples)

        self._samples = all_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        entry = self._samples[idx]
        img = entry.preprocessor.load_image(entry.sample)
        return {
            "image":        self.transform(img),
            "dataset_name": entry.ds_name,
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


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class VisSample:
    original: torch.Tensor   # (3, H, W) cpu, [-1, 1]
    recon:    torch.Tensor   # (3, H, W) cpu, [-1, 1]
    psnr:     float
    lpips:    float
    label:    str


@torch.no_grad()
def evaluate_vae(
    vae: VAEWrapper,
    lpips_fn: LPIPSMetric,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    log_every: int = 10,
    vis_max: int = 0,
) -> Tuple[MetricAccumulator, List[VisSample]]:
    """
    VAE encode → decode round-trip over all batches.

    Returns:
        (accumulator, vis_samples)
        vis_samples is populated only on rank 0, up to vis_max images.
    """
    acc = MetricAccumulator()
    vis_buf: List[VisSample] = []
    vae.eval()

    for batch_idx, batch in enumerate(loader):
        images: torch.Tensor = batch["image"].to(device)   # (B, 3, H, W) in [-1, 1]
        ds_names: List[str]  = batch["dataset_name"]

        latents = vae.encode(images)
        recon   = vae.decode(latents)                       # (B, 3, H, W) in [-1, 1]

        psnr_vals  = compute_psnr(recon, images)            # (B,)
        lpips_vals = lpips_fn(recon, images)                # (B,)

        for i in range(len(images)):
            psnr_i  = psnr_vals[i].item()
            lpips_i = lpips_vals[i].item()
            acc.update(psnr=psnr_i, lpips_val=lpips_i, n=1, key=ds_names[i])

            if rank == 0 and len(vis_buf) < vis_max:
                vis_buf.append(VisSample(
                    original=images[i].cpu(),
                    recon=recon[i].cpu(),
                    psnr=psnr_i,
                    lpips=lpips_i,
                    label=ds_names[i],
                ))

        if rank == 0 and (batch_idx + 1) % log_every == 0:
            print(
                f"  batch {batch_idx+1:4d}/{len(loader)}  "
                f"PSNR={acc.mean_psnr:.2f} dB  LPIPS={acc.mean_lpips:.4f}  "
                f"n={acc.count:,}",
                flush=True,
            )

    return acc, vis_buf


# ---------------------------------------------------------------------------
# Distributed accumulator merge
# ---------------------------------------------------------------------------

def _gather_accumulator(local: MetricAccumulator, device: torch.device) -> MetricAccumulator:
    """Merge per-rank accumulators into one on rank 0. All ranks must call."""
    all_dicts: List[Optional[dict]] = [None] * _world_size()
    dist.all_gather_object(all_dicts, local.to_dict())

    if _rank() != 0:
        return local

    merged = MetricAccumulator()
    for d in all_dicts:
        n = d["n_images"]
        if n == 0:
            continue
        merged.update(d["psnr_db"], d["lpips"], n=n)
        for ds_name, ds_d in d.get("per_dataset", {}).items():
            ds_n = ds_d["n_images"]
            if ds_n > 0:
                merged.update(ds_d["psnr_db"], ds_d["lpips"], n=ds_n, key=ds_name)
    return merged


# ---------------------------------------------------------------------------
# Preprocessor instantiation from preprocess_config.yaml
# ---------------------------------------------------------------------------

def _build_preprocessors(
    preprocess_cfg: dict,
) -> List[Tuple[str, BaseDatasetPreprocessor]]:
    """
    Instantiate preprocessors from the preprocess_config.yaml structure.

    Each dataset entry needs:
        name      — used as dataset_name and output subdir
        image_dir — passed as image_root
        type      — registry key in PREPROCESSORS (defaults to name if omitted)

    output_root is set to a scratch path since eval never writes embeddings.
    """
    result = []
    dummy_output = "/tmp/_eval_vae_dummy"
    for ds in preprocess_cfg["datasets"]:
        ds_name   = ds["name"]
        image_dir = ds["image_dir"]
        ds_type   = ds.get("type", ds_name)   # type defaults to name

        if ds_type not in PREPROCESSORS:
            raise ValueError(
                f"Unknown preprocessor type '{ds_type}' for dataset '{ds_name}'. "
                f"Registered types: {list(PREPROCESSORS)}"
            )

        import inspect
        sig = inspect.signature(PREPROCESSORS[ds_type].__init__)
        kwargs: dict = {"image_root": image_dir, "output_root": dummy_output}
        if "name" in sig.parameters:
            kwargs["name"] = ds_name
        if "camera_key" in sig.parameters and "camera_key" in ds:
            kwargs["camera_key"] = ds["camera_key"]
        prep = PREPROCESSORS[ds_type](**kwargs)
        result.append((ds_name, prep))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate VAE reconstruction quality (PSNR + LPIPS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to configs/base.yaml (provides VAE model name + resolution).")
    p.add_argument("--preprocess_config",
                   default="scripts/preprocess_config.yaml",
                   help="Path to preprocess_config.yaml (provides dataset type + image_dir).")
    p.add_argument("--num_samples", type=int, default=None,
                   help="Max images to evaluate (random subset). Default: all.")
    p.add_argument("--resolution",  type=int, default=None,
                   help="Override image resolution (default: from base.yaml).")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lpips_net",   default="vgg",
                   choices=["vgg", "alex", "squeeze"],
                   help="LPIPS backbone network.")
    p.add_argument("--log_every",   type=int, default=10,
                   help="Print progress every N batches.")
    p.add_argument("--output_json", default="vae_eval_results.json")
    p.add_argument("--vis_samples", type=int, default=16,
                   help="Images to include in the comparison grid (0 = skip).")
    p.add_argument("--vis_nrow",    type=int, default=4,
                   help="(original | reconstructed) pairs per grid row.")
    p.add_argument("--vis_output",  default="vae_eval_grid.png")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Distributed init ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank  = _rank()
    world = _world_size()

    # ── Configs ──────────────────────────────────────────────────────────────
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.preprocess_config) as f:
        preprocess_cfg = yaml.safe_load(f)

    resolution = args.resolution or config["data"]["resolution"]

    # ── Preprocessors → dataset ──────────────────────────────────────────────
    preprocessors = _build_preprocessors(preprocess_cfg)
    ds_names = [n for n, _ in preprocessors]

    if rank == 0:
        print(
            f"[eval_vae] resolution={resolution}  "
            f"datasets={ds_names}  "
            f"num_samples={args.num_samples or 'all'}"
        )

    dataset = PreprocessorDataset(
        preprocessors=preprocessors,
        resolution=resolution,
        max_samples=args.num_samples,
        seed=args.seed,
    )

    # Shard across ranks — simple index interleaving, no sampler needed
    if world > 1:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(rank, len(dataset), world)))

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
            f"[eval_vae] {total:,} images  "
            f"(~{math.ceil(total / world):,}/rank)  "
            f"batch={args.batch_size}  gpus={world}"
        )

    # ── Models ───────────────────────────────────────────────────────────────
    unet_name = config["model"]["unet_name"]
    if rank == 0:
        print(f"[eval_vae] Loading VAE from {unet_name} ...")

    vae      = VAEWrapper(model_name=unet_name).to(device)
    lpips_fn = LPIPSMetric(net=args.lpips_net, device=device)

    if rank == 0:
        print("[eval_vae] Running evaluation ...")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    local_acc, vis_buf = evaluate_vae(
        vae=vae,
        lpips_fn=lpips_fn,
        loader=loader,
        device=device,
        rank=rank,
        log_every=args.log_every,
        vis_max=args.vis_samples if rank == 0 else 0,
    )

    # ── Aggregate ────────────────────────────────────────────────────────────
    global_acc = _gather_accumulator(local_acc, device) if world > 1 else local_acc

    # ── Report (rank 0) ───────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "─" * 60)
        print("[eval_vae] Results:")
        print(global_acc)
        print("─" * 60)

        result_dict = global_acc.to_dict()
        result_dict.update({
            "config":           args.config,
            "preprocess_config": args.preprocess_config,
            "resolution":       resolution,
            "lpips_net":        args.lpips_net,
            "backbone":         config["model"].get("backbone", "unknown"),
        })

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"[eval_vae] Results saved to {out_path}")

        if vis_buf:
            save_comparison_grid(
                originals       =[s.original for s in vis_buf],
                reconstructions =[s.recon    for s in vis_buf],
                psnr_values     =[s.psnr     for s in vis_buf],
                lpips_values    =[s.lpips    for s in vis_buf],
                labels          =[s.label    for s in vis_buf],
                output_path     =args.vis_output,
                nrow            =args.vis_nrow,
                title           =(
                    f"VAE Reconstruction  "
                    f"(PSNR={global_acc.mean_psnr:.2f} dB  "
                    f"LPIPS={global_acc.mean_lpips:.4f})"
                ),
            )
            print(f"[eval_vae] Comparison grid saved to {args.vis_output}")

    if _is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
