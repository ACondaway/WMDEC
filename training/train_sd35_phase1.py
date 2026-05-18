# SD3.5 Phase 1 — Semantic Alignment Training
# Trains: PerceiverResampler + ResidualMLPAdaptor
# Frozen: SD3.5 MMDiT, VAE, Qwen Visual Encoder, TextureControlBranch (disabled)
#
# Usage (single node, 8×H100):
#   torchrun --nproc_per_node=8 training/train_sd35_phase1.py --config configs/sd35_phase1.yaml
#
# Usage (multi-node, 4 nodes × 8 GPUs):
#   torchrun --nnodes=4 --nproc_per_node=8 --node_rank=<RANK> \
#       --master_addr=<MASTER_IP> --master_port=29500 \
#       training/train_sd35_phase1.py --config configs/sd35_phase1.yaml

from __future__ import annotations

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.sd35_model import SD35SemanticModel
from data.dataset_sd35 import MultiDatasetSD35Dataset, DistributedWeightedSampler
from training.cfg import apply_condition_dropout


# ---------------------------------------------------------------------------
# CFG dropout for SD3.5 semantic tokens
# ---------------------------------------------------------------------------

def _apply_cfg_dropout(
    qwen_features: torch.Tensor,
    p_keep: float = 0.75,
    p_drop_img: float = 0.10,
    p_drop_txt: float = 0.10,
    # p_drop_all: 0.05 (remainder)
) -> torch.Tensor:
    """
    Per-sample condition dropout for classifier-free guidance training.

    Drop probabilities (from CLAUDE.md):
        keep all   : 75%
        drop image : 10%   → zero qwen_features
        drop text  : 10%   → noop in Phase 1 (no text branch)
        drop all   :  5%   → zero qwen_features

    Returns dropped qwen_features (B, 64, 2560).
    """
    B = qwen_features.shape[0]
    r = torch.rand(B, device=qwen_features.device)

    # drop image (and drop all) → zero the visual features
    drop_img_mask = r >= p_keep
    drop_img_mask = drop_img_mask & (r < p_keep + p_drop_img + (1.0 - p_keep - p_drop_img - p_drop_txt))

    out = qwen_features.clone()
    out[drop_img_mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    step: int,
    model: SD35SemanticModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    output_dir: str,
    is_main: bool,
) -> None:
    if not is_main:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "step": step,
        "resampler": model.resampler.state_dict(),
        "adaptor":   model.adaptor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
    }
    path = os.path.join(output_dir, f"phase1_step{step:07d}.pt")
    torch.save(ckpt, path)
    # Keep latest symlink
    latest = os.path.join(output_dir, "phase1_latest.pt")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(path), latest)
    print(f"[step {step}] Checkpoint saved → {path}")


def load_checkpoint(
    path: str,
    model: SD35SemanticModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.resampler.load_state_dict(ckpt["resampler"])
    model.adaptor.load_state_dict(ckpt["adaptor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SD3.5 Phase 1 — Semantic Alignment")
    parser.add_argument("--config", required=True, help="Path to sd35_phase1.yaml")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = dist.get_rank() == 0
    world_size = dist.get_world_size()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = SD35SemanticModel(
        transformer_path=cfg["transformer_path"],
        vae_path=cfg["vae_path"],
        qwen_encoder_ckpt=cfg["qwen_encoder_ckpt"],
        resampler_cfg=cfg.get("resampler", {}),
        adaptor_cfg=cfg.get("adaptor", {}),
        control_cfg=None,                         # Phase 1: no texture control
        sem_loss_weight=cfg.get("sem_loss_weight", 0.0),
    ).to(device)

    model.set_phase1()

    if is_main:
        trainable = model.trainable_parameter_count()
        frozen = model.frozen_parameter_count()
        print(f"Trainable params : {trainable / 1e6:.1f}M")
        print(f"Frozen params    : {frozen / 1e6:.1f}M")

    # Wrap trainable submodules in DDP (not the whole model, which has frozen parts)
    resampler_ddp = DDP(model.resampler, device_ids=[local_rank])
    adaptor_ddp   = DDP(model.adaptor,   device_ids=[local_rank])

    # Point model attributes to DDP wrappers for gradient flow
    model.resampler = resampler_ddp
    model.adaptor   = adaptor_ddp

    # ------------------------------------------------------------------
    # Dataset + dataloader
    # ------------------------------------------------------------------

    dataset = MultiDatasetSD35Dataset(
        datasets_config=cfg["datasets"],
        resolution=cfg.get("resolution", 768),
        rebalance_alpha=cfg.get("rebalance_alpha", 0.7),
    )

    if is_main:
        print(dataset.dataset_summary())

    sampler = DistributedWeightedSampler(
        weights=dataset.get_sample_weights(),
        num_samples_per_epoch=len(dataset),
        num_replicas=world_size,
        rank=dist.get_rank(),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("per_gpu_batch", 8),
        sampler=sampler,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimiser + scaler
    # ------------------------------------------------------------------

    trainable_params = (
        list(model.resampler.parameters())
        + list(model.adaptor.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-2),
    )

    scaler = torch.cuda.amp.GradScaler()

    total_steps = cfg.get("total_steps", 100_000)
    grad_accum  = cfg.get("grad_accum", 1)
    log_every   = cfg.get("log_every", 50)
    save_every  = cfg.get("save_every", 2000)
    output_dir  = cfg.get("output_dir", "outputs/sd35_phase1")

    # LR warmup + cosine decay
    warmup_steps = cfg.get("warmup_steps", 1000)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.get("lr", 1e-4),
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    step = 0
    if args.resume:
        step = load_checkpoint(args.resume, model, optimizer, scaler)
        if is_main:
            print(f"Resumed from step {step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    model.train()
    model.transformer.eval()
    model.vae.eval()
    model.qwen.eval()

    accum_loss = 0.0
    accum_loss_diff = 0.0
    accum_loss_sem  = 0.0
    t0 = time.time()

    while step < total_steps:

        sampler.set_epoch(step // len(loader))

        for batch in loader:

            if step >= total_steps:
                break

            image        = batch["image"].to(device, non_blocking=True)
            qwen_features = batch["z_img"].to(device, non_blocking=True)

            # CFG condition dropout
            qwen_features = _apply_cfg_dropout(qwen_features)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                losses = model(image, qwen_features)

            loss = losses["loss"] / grad_accum
            scaler.scale(loss).backward()

            accum_loss      += losses["loss"].item()
            accum_loss_diff += losses["loss_diff"].item()
            accum_loss_sem  += losses["loss_sem"].item()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1

            if is_main and step % log_every == 0:
                elapsed = time.time() - t0
                avg_loss      = accum_loss      / log_every
                avg_loss_diff = accum_loss_diff / log_every
                avg_loss_sem  = accum_loss_sem  / log_every
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {step:7d}/{total_steps}]  "
                    f"loss={avg_loss:.4f}  "
                    f"diff={avg_loss_diff:.4f}  "
                    f"sem={avg_loss_sem:.4f}  "
                    f"lr={lr_now:.2e}  "
                    f"elapsed={elapsed:.0f}s"
                )
                accum_loss = accum_loss_diff = accum_loss_sem = 0.0
                t0 = time.time()

            if is_main and step % save_every == 0:
                save_checkpoint(step, model, optimizer, scaler, output_dir, is_main)

    # Final checkpoint
    save_checkpoint(step, model, optimizer, scaler, output_dir, is_main)

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()
