import os
import argparse
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adapter import ImageAdapter, TextAdapter
from models.unet import ConditionalUNet
from models.vae import VAEWrapper
from models.siglip_encoder import SigLIPEncoder
from diffusion.scheduler import DDPMScheduler
from training.loss import DiffusionLoss
from training.cfg import apply_condition_dropout
from data.dataset import EmbeddingDataset


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


def save_loss_plot(loss_history: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, values in loss_history.items():
        if values:
            steps, losses_val = zip(*values)
            ax.plot(steps, losses_val, label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)


def train(config: dict, resume_path: str = None):
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    # ---- Models ----
    img_adapter = ImageAdapter(
        siglip_dim=config["model"]["siglip_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        num_tokens=config["model"]["num_img_tokens"],
        num_layers=config["model"]["adapter_layers"],
    ).to(device)

    txt_adapter = TextAdapter(
        t5_dim=config["model"]["t5_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
    ).to(device)

    unet = ConditionalUNet(
        in_channels=4,
        model_channels=config["model"]["model_channels"],
        channel_mult=tuple(config["model"]["channel_mult"]),
        context_dim=config["model"]["cross_attn_dim"],
        num_heads=config["model"]["num_heads"],
    ).to(device)

    vae = VAEWrapper(config["model"]["vae_name"]).to(device)

    # SigLIP for semantic loss (only on main GPU to save memory)
    siglip = None
    if config["training"]["lambda_sem"] > 0:
        siglip = SigLIPEncoder(config["model"]["siglip_model"]).to(device)

    # DDP wrapping (only trainable modules)
    img_adapter = DDP(img_adapter, device_ids=[local_rank], find_unused_parameters=True)
    txt_adapter = DDP(txt_adapter, device_ids=[local_rank], find_unused_parameters=True)
    unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Scheduler & Loss ----
    scheduler = DDPMScheduler(
        num_timesteps=config["diffusion"]["num_timesteps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
        schedule=config["diffusion"]["schedule"],
    )
    criterion = DiffusionLoss(lambda_sem=config["training"]["lambda_sem"])

    # ---- Optimizer ----
    params = list(img_adapter.parameters()) + list(txt_adapter.parameters()) + list(unet.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["max_steps"],
        eta_min=config["training"]["lr"] * 0.01,
    )

    scaler = GradScaler()

    # ---- Resume from checkpoint ----
    start_step = 0
    if resume_path is not None:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        unet.module.load_state_dict(ckpt["unet"])
        img_adapter.module.load_state_dict(ckpt["img_adapter"])
        txt_adapter.module.load_state_dict(ckpt["txt_adapter"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        del ckpt

    # ---- Dataset ----
    dataset = EmbeddingDataset(
        data_dir=config["data"]["embedding_dir"],
        image_dir=config["data"]["image_dir"],
        resolution=config["data"]["resolution"],
    )
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size_per_gpu"],
        sampler=sampler,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # ---- Training Loop ----
    global_step = start_step
    max_steps = config["training"]["max_steps"]

    # Loss history for plotting (main process only)
    loss_history = {"total": [], "diffusion": [], "semantic": []}

    if is_main:
        print(f"Starting training from step {global_step} for {max_steps} steps")
        print(f"Trainable params: {sum(p.numel() for p in params if p.requires_grad) / 1e6:.1f}M")
        pbar = tqdm(total=max_steps, initial=global_step, desc="Training", dynamic_ncols=True, miniters=100)

    while global_step < max_steps:
        sampler.set_epoch(global_step)
        for batch in dataloader:
            if global_step >= max_steps:
                break

            z_img_emb = batch["z_img"].to(device)       # (B, D)
            z_txt_emb = batch["z_txt"].to(device)       # (B, T, C)
            images = batch["image"].to(device)           # (B, 3, H, W)

            # Encode image to latent
            with torch.no_grad():
                latent = vae.encode(images)

            # Adapters
            tokens_img = img_adapter(z_img_emb)  # (B, N_img, C)
            tokens_txt = txt_adapter(z_txt_emb)  # (B, T, C)

            # CFG dropout
            tokens_img, tokens_txt = apply_condition_dropout(tokens_img, tokens_txt)

            # Concat conditioning
            context = torch.cat([tokens_img, tokens_txt], dim=1)  # (B, N_img+T, C)

            # Diffusion forward
            noise = torch.randn_like(latent)
            t = scheduler.sample_timesteps(latent.shape[0], device)
            x_t = scheduler.q_sample(latent, t, noise)

            with autocast():
                eps_pred = unet(x_t, t, context)
                losses = criterion(eps_pred, noise)

            # Semantic loss (optional, computed less frequently for efficiency)
            if siglip is not None and global_step % config["training"].get("sem_loss_every", 1) == 0:
                with torch.no_grad():
                    # Approximate x_0 from prediction
                    alpha_t = scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
                    x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
                    x_0_pred = torch.clamp(x_0_pred, -1, 1)
                    img_pred = vae.decode(x_0_pred)
                    # Resize for SigLIP
                    img_pred_resized = torch.nn.functional.interpolate(
                        img_pred, size=(384, 384), mode="bilinear", align_corners=False
                    )
                    z_pred = siglip(img_pred_resized)

                sem_loss = criterion.semantic_loss(z_pred, z_img_emb)
                losses["semantic"] = sem_loss
                losses["total"] = losses["total"] + config["training"]["lambda_sem"] * sem_loss

            # Backward
            optimizer.zero_grad()
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1

            # Logging
            if is_main:
                # Update tqdm
                postfix = {"loss": f"{losses['total'].item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                if "semantic" in losses:
                    postfix["sem"] = f"{losses['semantic'].item():.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

                # Record loss history
                if global_step % config["training"]["log_every"] == 0:
                    loss_history["total"].append((global_step, losses["total"].item()))
                    loss_history["diffusion"].append((global_step, losses["diffusion"].item()))
                    if "semantic" in losses:
                        loss_history["semantic"].append((global_step, losses["semantic"].item()))

                # Update loss plot every 100 steps
                if global_step % config["training"]["ploy_every"] == 0:
                    plot_dir = os.path.join(config["training"]["output_dir"], "plots")
                    save_loss_plot(loss_history, plot_dir)

                if global_step % config["training"]["visualize_every"] == 0:
                    vis_dir = os.path.join(config["training"]["output_dir"], "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)

                    with torch.no_grad():
                        vis_n = min(4, images.shape[0])
                        vis_images = images[:vis_n]
                        vis_latent = vae.encode(vis_images)
                        vis_ctx = context[:vis_n]

                        # Use a low timestep for a cleaner x_0 estimate
                        vis_t = torch.full((vis_n,), 50, device=device, dtype=torch.long)
                        vis_noise = torch.randn_like(vis_latent)
                        vis_x_t = scheduler.q_sample(vis_latent, vis_t, vis_noise)

                        vis_eps_pred = unet(vis_x_t, vis_t, vis_ctx)
                        alpha_t = scheduler.alphas_cumprod.to(device)[vis_t].view(-1, 1, 1, 1)
                        vis_x0 = (vis_x_t - torch.sqrt(1 - alpha_t) * vis_eps_pred) / torch.sqrt(alpha_t)
                        vis_x0 = torch.clamp(vis_x0, -1, 1)

                        pred = vae.decode(vis_x0)

                        # Consistent normalization: VAE output is [-1, 1] -> [0, 1]
                        gt = (vis_images * 0.5 + 0.5).clamp(0, 1).cpu()
                        pred = (pred * 0.5 + 0.5).clamp(0, 1).cpu()

                        grid = torch.cat([gt, pred], dim=0)
                        grid_img = make_grid(grid, nrow=vis_n)

                        save_path = os.path.join(vis_dir, f"step_{global_step}.png")
                        save_image(grid_img, save_path)
                    
                    

            # Save checkpoint
            if is_main and global_step % config["training"]["save_every"] == 0:
                ckpt_dir = os.path.join(config["training"]["output_dir"], "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    "step": global_step,
                    "unet": unet.module.state_dict(),
                    "img_adapter": img_adapter.module.state_dict(),
                    "txt_adapter": txt_adapter.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }, os.path.join(ckpt_dir, f"step_{global_step}.pt"))
                print(f"Saved checkpoint at step {global_step}")

    # Final save
    if is_main:
        pbar.close()

        # Final loss plot
        plot_dir = os.path.join(config["training"]["output_dir"], "plots")
        save_loss_plot(loss_history, plot_dir)
        print(f"Loss curve saved to {plot_dir}/loss_curve.png")

        ckpt_dir = os.path.join(config["training"]["output_dir"])
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            "step": global_step,
            "unet": unet.module.state_dict(),
            "img_adapter": img_adapter.module.state_dict(),
            "txt_adapter": txt_adapter.module.state_dict(),
        }, os.path.join(ckpt_dir, "final.pt"))
        print("Training complete!")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_path=args.resume)
