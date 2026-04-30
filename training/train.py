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
from models.unet import SDXLUNet
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
    """SDXL added conditioning: [orig_h, orig_w, crop_y, crop_x, target_h, target_w]."""
    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=torch.float32, device=device,
    )
    return ids.unsqueeze(0).expand(batch_size, -1)


def save_loss_plot(loss_history: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: train + val losses
    ax = axes[0]
    for key, values in loss_history.items():
        if not values or key in ("val/cosine_sim", "val/lpips"):
            continue
        steps, vals = zip(*values)
        ls = "--" if key.startswith("val/") else "-"
        ax.plot(steps, vals, label=key, linestyle=ls)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curves"); ax.legend(); ax.grid(True, alpha=0.3)

    # Right: val cosine similarity + LPIPS
    ax2 = axes[1]
    for key in ("val/cosine_sim", "val/lpips"):
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

    # ---- Models ----
    img_adapter = ImageAdapter(
        qwen_dim=config["model"]["qwen_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        pooled_proj_dim=config["model"]["pooled_proj_dim"],
        num_tokens=config["model"]["num_img_tokens"],
        num_heads=config["model"]["num_heads"],
    ).to(device)

    unet = SDXLUNet(
        model_name=config["model"]["unet_name"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
    ).to(device)

    vae = VAEWrapper(config["model"]["unet_name"]).to(device)

    qwen_enc = None
    if config["training"]["lambda_sem"] > 0:
        qwen_enc = QwenVisualEncoder.from_standalone(
            config["model"]["qwen_encoder_ckpt"]
        ).to(device)
        qwen_enc.eval()

    # ---- Training mode: full fine-tune or LoRA ----
    training_mode = config["training"].get("training_mode", "full")
    if training_mode == "lora":
        unet.setup_lora(
            rank=config["training"].get("lora_rank", 64),
            alpha=config["training"].get("lora_alpha", 64),
            target_modules=config["training"].get("lora_target_modules", None),
        )

    img_adapter = DDP(img_adapter, device_ids=[local_rank], find_unused_parameters=False)
    unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Scheduler & Loss ----
    scheduler = DDPMScheduler(model_name=config["model"]["unet_name"])
    criterion = DiffusionLoss(lambda_sem=config["training"]["lambda_sem"])

    # ---- Optimizer ----
    # Full mode: train all UNet + adapter weights.
    # LoRA mode: only LoRA delta weights + adapter (base UNet is frozen).
    params = list(img_adapter.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_steps"], eta_min=config["training"]["lr"] * 0.01
    )
    scaler = GradScaler()

    # ---- Resume ----
    start_step = 0
    loss_history = None
    if resume_path is not None:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        unet.module.load_state_dict(ckpt["unet"])
        img_adapter.module.load_state_dict(ckpt["img_adapter"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        loss_history = ckpt.get("loss_history", None)
        del ckpt

    # ---- Dataset + Rebalanced Sampler ----
    data_cfg = config["data"]
    dataset = MultiDatasetEmbeddingDataset(
        datasets_config=data_cfg["datasets"],
        resolution=data_cfg["resolution"],
        rebalance_alpha=data_cfg.get("rebalance_alpha", 0.7),
    )

    if is_main:
        print(dataset.dataset_summary())

    # ---- Validation loader (rank-0 only) ----
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
    if loss_history is None:
        loss_history = {"total": [], "diffusion": [], "semantic": []}

    if is_main:
        print(f"Training from step {global_step} / {max_steps}")
        print(f"Trainable params: {sum(p.numel() for p in params if p.requires_grad) / 1e6:.1f}M")
        pbar = tqdm(total=max_steps, initial=global_step, desc="Training",
                    dynamic_ncols=True, miniters=100)

    epoch = 0
    while global_step < max_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            if global_step >= max_steps:
                break

            z_img_emb = batch["z_img"].to(device)   # (B, N_patches, D)
            images = batch["image"].to(device)        # (B, 3, H, W)
            B = images.shape[0]

            with torch.no_grad():
                latent = vae.encode(images)

            tokens, pooled = img_adapter(z_img_emb)
            tokens, pooled = apply_condition_dropout(
                tokens, pooled, p_drop=config["training"]["cfg_drop_prob"]
            )

            time_ids = make_time_ids(B, resolution, device)
            noise = torch.randn_like(latent)
            t = scheduler.sample_timesteps(B, device)
            x_t = scheduler.q_sample(latent, t, noise)

            with autocast():
                eps_pred = unet(x_t, t, tokens, pooled, time_ids)
                losses = criterion(eps_pred, noise)

            sem_every = config["training"].get("sem_loss_every", 10)
            if qwen_enc is not None and global_step % sem_every == 0:
                with torch.no_grad():
                    alpha_t = scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
                    x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
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
                    with torch.no_grad():
                        vis_n = min(4, B)

                        # GT: true images from the batch (loaded from image_dir via _abs_image_path).
                        gt = (images[:vis_n] * 0.5 + 0.5).clamp(0, 1).cpu()

                        # Recompute conditioning without CFG dropout for a clean sample.
                        _adapter = img_adapter.module if hasattr(img_adapter, "module") else img_adapter
                        _unet    = unet.module    if hasattr(unet,    "module") else unet
                        vis_tokens, vis_pooled = _adapter(z_img_emb[:vis_n])
                        vis_time_ids = time_ids[:vis_n]

                        # 10-step DDIM — shows what the model has learned.
                        from diffusers import DDIMScheduler as _DDIMSched
                        _ddim = _DDIMSched(
                            num_train_timesteps=1000,
                            beta_start=0.00085,
                            beta_end=0.012,
                            beta_schedule="scaled_linear",
                            clip_sample=False,
                            set_alpha_to_one=False,
                        )
                        _ddim.set_timesteps(10)
                        x = torch.randn(vis_n, 4, resolution // 8, resolution // 8, device=device)
                        for _t in _ddim.timesteps:
                            _t_b = torch.full((vis_n,), _t, device=device, dtype=torch.long)
                            _eps = _unet(x, _t_b, vis_tokens, vis_pooled, vis_time_ids)
                            x = _ddim.step(_eps, _t, x).prev_sample
                        pred = (vae.decode(x) * 0.5 + 0.5).clamp(0, 1).cpu()

                        grid = make_grid(torch.cat([gt, pred], dim=0), nrow=vis_n)
                        save_image(grid, os.path.join(vis_dir, f"step_{global_step}.png"))

            # ---- Validation ----
            val_cfg = config.get("validation", {})
            if is_main and val_loader is not None:
                val_every = val_cfg.get("val_every", 5000)
                val_full_every = val_cfg.get("val_full_every", 0)  # 0 = disabled

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
                    )
                    loss_history.setdefault("val/loss_diffusion", [])
                    loss_history.setdefault("val/cosine_sim", [])
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
                    )
                    if full_metrics["val/lpips"] >= 0:
                        loss_history.setdefault("val/lpips", [])
                        loss_history["val/lpips"].append((global_step, full_metrics["val/lpips"]))
                        print(f"  [Full val]  LPIPS={full_metrics['val/lpips']:.4f}")

            if is_main and global_step % config["training"]["save_every"] == 0:
                ckpt_dir = os.path.join(config["training"]["output_dir"], "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                if training_mode == "lora":
                    # Save only LoRA delta weights + adapter; base UNet weights unchanged.
                    lora_dir = os.path.join(ckpt_dir, f"lora_step_{global_step}")
                    unet.module.save_lora(lora_dir)
                    torch.save({
                        "step": global_step,
                        "img_adapter": img_adapter.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "loss_history": loss_history,
                    }, os.path.join(ckpt_dir, f"step_{global_step}.pt"))
                else:
                    torch.save({
                        "step": global_step,
                        "unet": unet.module.state_dict(),
                        "img_adapter": img_adapter.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "loss_history": loss_history,
                    }, os.path.join(ckpt_dir, f"step_{global_step}.pt"))

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
