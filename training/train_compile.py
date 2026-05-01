"""
DDP training with torch.compile() for ~15-30% throughput improvement on H100.

Identical to train.py with three key structural differences:

  1. Checkpoint weights are loaded into the RAW model BEFORE torch.compile(),
     so state_dict keys always match and no compiled-model loading quirks arise.

  2. torch.compile() is applied AFTER weight loading and LoRA setup, BEFORE
     DDP wrapping.  This gives the compiler full visibility of the model graph
     per-rank.

  3. gradient_checkpointing defaults to False.  torch.utils.checkpoint creates
     Python-level control-flow that forces graph breaks, fragmenting the fused
     kernel and erasing most of compile's benefit.  With SD 2.1-base on 8×H100
     (80 GB each) you have ample VRAM headroom without it.

Config keys added under training:
  compile_mode: "default"       # "default" | "reduce-overhead" | "max-autotune"
  compile_adapter: true         # also compile the ImageAdapter (default true)

Usage:
  torchrun --nproc_per_node=8 training/train_compile.py --config configs/base.yaml
"""

import os
import argparse
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adapter import ImageAdapter
from models.unet import build_unet
from models.vae import VAEWrapper
from models.qwen_visual_encoder import QwenVisualEncoder
from diffusion.scheduler import DDPMScheduler
from training.loss import DiffusionLoss
from training.cfg import apply_condition_dropout, build_uncond_context
from training.validate import (
    build_val_loader, run_fast_validation, run_full_validation, BestCheckpointTracker,
)
from data.dataset import MultiDatasetEmbeddingDataset, DistributedWeightedSampler


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


def make_time_ids(batch_size: int, resolution: int, device: torch.device) -> torch.Tensor:
    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=torch.float32, device=device,
    )
    return ids.unsqueeze(0).expand(batch_size, -1)


_METRIC_KEYS = {"val/cosine_sim", "val/lpips"}


def save_loss_plot(loss_history: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    for key, values in loss_history.items():
        if not values or key in _METRIC_KEYS:
            continue
        steps, vals = zip(*values)
        if key.startswith("val/"):
            ax.plot(steps, vals, label=key, linestyle="--", marker="o", markersize=3)
        else:
            ax.plot(steps, vals, label=key, linestyle="-")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend(); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    for key in _METRIC_KEYS:
        values = loss_history.get(key, [])
        if values:
            steps, vals = zip(*values)
            ax2.plot(steps, vals, label=key, marker="o", markersize=3)
    ax2.set_xlabel("Step"); ax2.set_ylabel("Metric")
    ax2.set_title("Validation Metrics"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def train(config: dict, resume_path: str = None):
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    backbone = config["model"].get("backbone", "sdxl")
    compile_mode    = config["training"].get("compile_mode", "default")
    compile_adapter = config["training"].get("compile_adapter", True)

    # ---- Step 1: build raw models (no compile yet) ----
    # gradient_checkpointing defaults False here: torch.utils.checkpoint creates
    # graph breaks that fragment the compiled kernels and erase the speedup.
    # SD 2.1-base on H100-80GB does not need it — ~20 GB used out of 80 GB.
    gc = config["training"].get("gradient_checkpointing", False)

    raw_unet = build_unet(config, gradient_checkpointing=gc).to(device)

    raw_adapter = ImageAdapter(
        qwen_dim=config["model"]["qwen_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        pooled_proj_dim=config["model"].get("pooled_proj_dim", 1280),
        num_tokens=config["model"]["num_img_tokens"],
        num_heads=config["model"]["num_heads"],
        backbone=backbone,
    ).to(device)

    vae = VAEWrapper(config["model"]["unet_name"]).to(device)

    qwen_enc = None
    if config["training"]["lambda_sem"] > 0:
        qwen_enc = QwenVisualEncoder.from_standalone(
            config["model"]["qwen_encoder_ckpt"]
        ).to(device)
        qwen_enc.eval()

    training_mode = config["training"].get("training_mode", "full")
    if training_mode == "lora":
        raw_unet.setup_lora(
            rank=config["training"].get("lora_rank", 64),
            alpha=config["training"].get("lora_alpha", 64),
            target_modules=config["training"].get("lora_target_modules", None),
        )

    # ---- Step 2: load checkpoint weights into RAW models (before compile) ----
    # Compiled OptimizedModule proxies state_dict correctly, but loading into
    # the raw model before compile is simpler and avoids any version quirks.
    start_step = 0
    loss_history = None
    optimizer_state = lr_sched_state = scaler_state = None

    if resume_path is not None:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        raw_unet.load_state_dict(ckpt["unet"])
        raw_adapter.load_state_dict(ckpt["img_adapter"])
        optimizer_state  = ckpt.get("optimizer")
        lr_sched_state   = ckpt.get("lr_scheduler")
        scaler_state     = ckpt.get("scaler")
        start_step       = ckpt.get("step", 0)
        if config["training"].get("save_loss_history", False):
            loss_history = ckpt.get("loss_history", None)
        del ckpt

    # ---- Step 3: torch.compile() — AFTER weight load, BEFORE DDP ----
    if is_main:
        print(f"Compiling UNet (mode='{compile_mode}') ...")
    unet = torch.compile(raw_unet, mode=compile_mode)

    if compile_adapter:
        if is_main:
            print(f"Compiling ImageAdapter (mode='{compile_mode}') ...")
        img_adapter = torch.compile(raw_adapter, mode=compile_mode)
    else:
        img_adapter = raw_adapter

    # ---- Step 4: DDP wrap compiled models ----
    img_adapter = DDP(img_adapter, device_ids=[local_rank], find_unused_parameters=False)
    unet        = DDP(unet,        device_ids=[local_rank], find_unused_parameters=False)

    # ---- Scheduler & Loss ----
    scheduler = DDPMScheduler(model_name=config["model"]["unet_name"])
    criterion = DiffusionLoss(lambda_sem=config["training"]["lambda_sem"])

    # ---- Optimizer (built after DDP so param hooks are in place) ----
    params = list(img_adapter.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_steps"], eta_min=config["training"]["lr"] * 0.01
    )
    scaler = GradScaler()

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if lr_sched_state is not None:
        lr_scheduler.load_state_dict(lr_sched_state)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    # ---- Dataset + Rebalanced Sampler ----
    data_cfg = config["data"]
    dataset = MultiDatasetEmbeddingDataset(
        datasets_config=data_cfg["datasets"],
        resolution=data_cfg["resolution"],
        rebalance_alpha=data_cfg.get("rebalance_alpha", 0.7),
    )
    if is_main:
        print(dataset.dataset_summary())

    val_loader = None
    best_tracker = None
    if is_main and config.get("validation"):
        val_loader = build_val_loader(config, train_dataset=dataset)
        best_tracker = BestCheckpointTracker(
            metric=config["validation"].get("best_metric", "val/cosine_sim")
        )
        print(f"Validation loader: {len(val_loader.dataset)} samples")

    num_samples_per_epoch = data_cfg.get("num_samples_per_epoch") or len(dataset)
    sampler = DistributedWeightedSampler(
        weights=dataset.get_sample_weights(),
        num_samples_per_epoch=num_samples_per_epoch,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        sampler=sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    resolution = data_cfg["resolution"]

    # ---- Training Loop ----
    global_step = start_step
    max_steps = config["training"]["max_steps"]
    save_loss_history = config["training"].get("save_loss_history", False)
    if loss_history is None:
        loss_history = {
            "total": [], "diffusion": [], "semantic": [],
            "val/loss_diffusion": [], "val/cosine_sim": [], "val/lpips": [],
        }

    if is_main:
        print(f"Training from step {global_step} / {max_steps}")
        print(f"Trainable params: {sum(p.numel() for p in params if p.requires_grad) / 1e6:.1f}M")
        print(f"compile_mode={compile_mode}  compile_adapter={compile_adapter}  gc={gc}")
        print("Note: first training step will be slow (~1–3 min) while kernels are compiled.")
        pbar = tqdm(total=max_steps, initial=global_step, desc="Training",
                    dynamic_ncols=True, miniters=100)

    epoch = 0
    while global_step < max_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            if global_step >= max_steps:
                break

            z_img_emb = batch["z_img"].to(device)
            images    = batch["image"].to(device)
            B = images.shape[0]

            emb_noise_std = config["training"].get("embedding_noise_std", 0.0)
            if emb_noise_std > 0.0:
                z_img_emb = z_img_emb + torch.randn_like(z_img_emb) * emb_noise_std

            with torch.no_grad():
                latent = vae.encode(images)

            tokens, pooled = img_adapter(z_img_emb)
            tokens, pooled = apply_condition_dropout(
                tokens, pooled, p_drop=config["training"]["cfg_drop_prob"]
            )

            noise = torch.randn_like(latent)
            t = scheduler.sample_timesteps(B, device)
            x_t = scheduler.q_sample(latent, t, noise)

            if backbone == "sd21":
                target = scheduler.get_v_target(latent, noise, t)
            else:
                target = noise

            with autocast():
                if backbone == "sd21":
                    pred = unet(x_t, t, tokens)
                else:
                    time_ids = make_time_ids(B, resolution, device)
                    pred = unet(x_t, t, tokens, pooled, time_ids)
                losses = criterion(pred, target)

            sem_every = config["training"].get("sem_loss_every", 10)
            if qwen_enc is not None and global_step % sem_every == 0:
                with torch.no_grad():
                    if backbone == "sd21":
                        x_0_pred = scheduler.predict_x0_from_v(pred.detach(), x_t, t)
                    else:
                        x_0_pred = scheduler.predict_x0_from_eps(pred.detach(), x_t, t)
                    x_0_pred = torch.clamp(x_0_pred, -1, 1)
                    img_pred = vae.decode(x_0_pred)
                    img_pred_448 = torch.nn.functional.interpolate(
                        img_pred, size=(448, 448), mode="bilinear", align_corners=False
                    )
                    pil_list = [
                        to_pil_image(((img_pred_448[i] + 1) / 2).clamp(0, 1).cpu())
                        for i in range(B)
                    ]
                    patch_pred = qwen_enc.encode_images(pil_list, device=device)
                sem_loss = criterion.semantic_loss(patch_pred, z_img_emb)
                losses["semantic"] = sem_loss
                losses["total"] = losses["total"] + config["training"]["lambda_sem"] * sem_loss

            optimizer.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1

            if is_main:
                postfix = {
                    "loss": f"{losses['total'].item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "epoch": f"{epoch}",
                }
                if "semantic" in losses:
                    postfix["sem"] = f"{losses['semantic'].item():.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

                if global_step % config["training"]["log_every"] == 0:
                    loss_history["total"].append((global_step, losses["total"].item()))
                    loss_history["diffusion"].append((global_step, losses["diffusion"].item()))
                    if "semantic" in losses:
                        loss_history["semantic"].append((global_step, losses["semantic"].item()))

                if global_step % config["training"]["plot_every"] == 0:
                    save_loss_plot(loss_history, os.path.join(config["training"]["output_dir"], "plots"))

                if global_step % config["training"]["visualize_every"] == 0:
                    vis_dir = os.path.join(config["training"]["output_dir"], "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    # .module on DDP gives the compiled OptimizedModule;
                    # eval()/train() and forward() work correctly through it.
                    _adapter = img_adapter.module
                    _unet    = unet.module
                    _adapter.eval()
                    _unet.eval()
                    with torch.no_grad():
                        vis_n = min(4, B)
                        gt = (images[:vis_n] * 0.5 + 0.5).clamp(0, 1).cpu()
                        vis_tokens, vis_pooled = _adapter(z_img_emb[:vis_n])
                        from diffusers import DDIMScheduler as _DDIMSched
                        _ddim = _DDIMSched.from_pretrained(
                            config["model"]["unet_name"], subfolder="scheduler"
                        )
                        _ddim.set_timesteps(10)
                        x = torch.randn(vis_n, 4, resolution // 8, resolution // 8, device=device)
                        for _t in _ddim.timesteps:
                            _t_b = torch.full((vis_n,), _t, device=device, dtype=torch.long)
                            if backbone == "sd21":
                                _out = _unet(x, _t_b, vis_tokens)
                            else:
                                _vis_time_ids = make_time_ids(vis_n, resolution, device)
                                _out = _unet(x, _t_b, vis_tokens, vis_pooled, _vis_time_ids)
                            x = _ddim.step(_out, _t, x).prev_sample
                        vis_pred = (vae.decode(x) * 0.5 + 0.5).clamp(0, 1).cpu()
                        grid = make_grid(torch.cat([gt, vis_pred], dim=0), nrow=vis_n)
                        save_image(grid, os.path.join(vis_dir, f"step_{global_step}.png"))
                    _adapter.train()
                    _unet.train()

            # ---- Validation ----
            val_cfg = config.get("validation", {})
            if is_main and val_loader is not None:
                val_every      = val_cfg.get("val_every", 5000)
                val_full_every = val_cfg.get("val_full_every", 0)

                if global_step % val_every == 0:
                    val_metrics = run_fast_validation(
                        img_adapter=img_adapter.module,
                        unet=unet.module,
                        vae=vae,
                        qwen_enc=qwen_enc,
                        scheduler=scheduler,
                        val_loader=val_loader,
                        device=device,
                        lambda_sem=config["training"]["lambda_sem"],
                        output_dir=config["training"]["output_dir"],
                        global_step=global_step,
                        backbone=backbone,
                        resolution=resolution,
                    )
                    loss_history["val/loss_diffusion"].append((global_step, val_metrics["val/loss_diffusion"]))
                    loss_history["val/cosine_sim"].append((global_step, val_metrics["val/cosine_sim"]))
                    print(
                        f"\n[Val step {global_step}]  "
                        f"diff_loss={val_metrics['val/loss_diffusion']:.4f}  "
                        f"cosine_sim={val_metrics['val/cosine_sim']:.4f}  "
                        f"total={val_metrics['val/loss_total']:.4f}"
                    )
                    if best_tracker.update(val_metrics, global_step):
                        ckpt_dir = os.path.join(config["training"]["output_dir"], "checkpoints")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        torch.save({
                            "step": global_step,
                            "unet": unet.module.state_dict(),
                            "img_adapter": img_adapter.module.state_dict(),
                            "val_metrics": val_metrics,
                        }, os.path.join(ckpt_dir, "best.pt"))
                        print(
                            f"  → New best ({val_cfg.get('best_metric', 'val/cosine_sim')}="
                            f"{best_tracker.best_value:.4f})  saved best.pt"
                        )

                if val_full_every > 0 and global_step % val_full_every == 0:
                    full_metrics = run_full_validation(
                        img_adapter=img_adapter.module,
                        unet=unet.module,
                        vae=vae,
                        val_loader=val_loader,
                        device=device,
                        output_dir=config["training"]["output_dir"],
                        global_step=global_step,
                        ddim_steps=val_cfg.get("val_ddim_steps", 20),
                        cfg_scale=val_cfg.get("val_cfg_scale", 2.0),
                        max_batches=val_cfg.get("val_full_max_batches", 4),
                        backbone=backbone,
                        model_name=config["model"]["unet_name"],
                        resolution=resolution,
                    )
                    if full_metrics["val/lpips"] >= 0:
                        loss_history["val/lpips"].append((global_step, full_metrics["val/lpips"]))
                        print(f"  [Full val]  LPIPS={full_metrics['val/lpips']:.4f}")

            # ---- Checkpoint ----
            # unet.module.state_dict() works correctly on compiled OptimizedModule —
            # it proxies through to the underlying model's parameter tensors.
            if is_main and global_step % config["training"]["save_every"] == 0:
                ckpt_dir = os.path.join(config["training"]["output_dir"], "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_payload = {
                    "step": global_step,
                    "img_adapter": img_adapter.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if save_loss_history:
                    ckpt_payload["loss_history"] = loss_history
                if training_mode == "lora":
                    unet.module.save_lora(os.path.join(ckpt_dir, f"lora_step_{global_step}"))
                else:
                    ckpt_payload["unet"] = unet.module.state_dict()
                torch.save(ckpt_payload, os.path.join(ckpt_dir, f"step_{global_step}.pt"))

    if is_main:
        pbar.close()
        save_loss_plot(loss_history, os.path.join(config["training"]["output_dir"], "plots"))
        if training_mode == "lora":
            unet.module.save_lora(os.path.join(config["training"]["output_dir"], "lora_final"))
            torch.save({
                "step": global_step,
                "img_adapter": img_adapter.module.state_dict(),
            }, os.path.join(config["training"]["output_dir"], "final.pt"))
        else:
            torch.save({
                "step": global_step,
                "unet": unet.module.state_dict(),
                "img_adapter": img_adapter.module.state_dict(),
            }, os.path.join(config["training"]["output_dir"], "final.pt"))
        print("Training complete!")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, resume_path=args.resume)
