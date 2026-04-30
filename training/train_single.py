"""
Single-process multi-GPU training via model parallelism.

The UNet is automatically sharded across all visible GPUs using
device_map="auto" (HuggingFace Accelerate).  The VAE and ImageAdapter
live on GPU 0.  A single Python process owns all GPUs — no torchrun needed.

Usage:
    python training/train_single.py --config configs/base.yaml
    python training/train_single.py --config configs/base.yaml --resume /path/to/ckpt.pt

When to prefer this over DDP (train.py):
    - Model is too large for one GPU even in bfloat16
    - You want a simpler single-process debugging experience
    - Fewer GPUs than dataset shards needed

When to prefer DDP (train.py):
    - Maximising throughput with data parallelism (each GPU has a full model copy)
    - Large batch sizes spread across ranks

device_map="auto" places the UNet layers on GPUs in order of VRAM availability.
Activations are moved between devices transparently during the forward pass.
"""

import os
import sys
import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from diffusers import UNet2DConditionModel
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.adapter import ImageAdapter
from models.vae import VAEWrapper
from models.qwen_visual_encoder import QwenVisualEncoder
from diffusion.scheduler import DDPMScheduler
from training.loss import DiffusionLoss
from training.cfg import apply_condition_dropout
from data.dataset import MultiDatasetEmbeddingDataset

# -------------------------------------------------------------------------
# UNet with device_map
# -------------------------------------------------------------------------

class SDXLUNetSharded(nn.Module):
    """
    SDXL UNet loaded with device_map="auto" so its layers are spread
    across all available GPUs.  The first (input) device is accessible
    via self.first_device.
    """

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        n_gpus = torch.cuda.device_count()
        print(f"Loading UNet with device_map='auto' across {n_gpus} GPU(s) ...")
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        # The first device is where inputs should be sent.
        self.first_device = next(iter(self.unet.hf_device_map.values()))
        print(f"UNet device map: {self.unet.hf_device_map}")

    def setup_lora(self, rank: int = 64, alpha: int = 64, target_modules: list = None) -> None:
        from peft import LoraConfig, get_peft_model
        _DEFAULT = ["to_q", "to_k", "to_v", "to_out.0", "to_add_out"]
        for p in self.unet.parameters():
            p.requires_grad = False
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=alpha,
            target_modules=target_modules or _DEFAULT,
            lora_dropout=0.0, bias="none",
        )
        self.unet = get_peft_model(self.unet, lora_cfg)
        trainable, total = self.unet.get_nb_trainable_parameters()
        print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total UNet params")

    def save_lora(self, output_dir: str) -> None:
        self.unet.save_pretrained(output_dir)

    def forward(self, x, t, encoder_hidden_states, pooled_proj, time_ids):
        d = self.first_device
        out = self.unet(
            x.to(d), t.to(d),
            encoder_hidden_states=encoder_hidden_states.to(d),
            added_cond_kwargs={
                "text_embeds": pooled_proj.to(d),
                "time_ids":    time_ids.to(d),
            },
        )
        return out.sample


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_time_ids(batch_size: int, resolution: int, device) -> torch.Tensor:
    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=torch.float32, device=device,
    )
    return ids.unsqueeze(0).expand(batch_size, -1)


_METRIC_KEYS = {"val/cosine_sim", "val/lpips"}


def save_loss_plot(loss_history: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: train losses (solid, logged every log_every steps) +
    #       val losses   (dashed, logged every val_every steps).
    # Each series uses its own recorded step coordinates — no sync required.
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

    # Right: val metrics (cosine sim + LPIPS), logged every val_every steps.
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


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

def train(config: dict, resume_path: str = None):
    # All non-UNet models live on GPU 0 (or CPU if no GPU available).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Models ----
    img_adapter = ImageAdapter(
        qwen_dim=config["model"]["qwen_dim"],
        cross_attn_dim=config["model"]["cross_attn_dim"],
        pooled_proj_dim=config["model"]["pooled_proj_dim"],
        num_tokens=config["model"]["num_img_tokens"],
        num_heads=config["model"]["num_heads"],
    ).to(device)

    unet = SDXLUNetSharded(
        model_name=config["model"]["unet_name"],
        gradient_checkpointing=config["training"].get("gradient_checkpointing", True),
    )

    vae = VAEWrapper(config["model"]["unet_name"]).to(device)

    qwen_enc = None
    if config["training"]["lambda_sem"] > 0:
        qwen_enc = QwenVisualEncoder.from_standalone(
            config["model"]["qwen_encoder_ckpt"]
        ).to(device)
        qwen_enc.eval()

    # ---- Training mode ----
    training_mode = config["training"].get("training_mode", "full")
    if training_mode == "lora":
        unet.setup_lora(
            rank=config["training"].get("lora_rank", 64),
            alpha=config["training"].get("lora_alpha", 64),
            target_modules=config["training"].get("lora_target_modules", None),
        )

    # ---- Optimizer ----
    params = list(img_adapter.parameters()) + [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["max_steps"], eta_min=config["training"]["lr"] * 0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    # ---- Scheduler & Loss ----
    scheduler = DDPMScheduler(model_name=config["model"]["unet_name"])
    criterion = DiffusionLoss(lambda_sem=config["training"]["lambda_sem"])

    # ---- Resume ----
    start_step = 0
    loss_history = None
    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if "unet" in ckpt:
            unet.unet.load_state_dict(ckpt["unet"])
        img_adapter.load_state_dict(ckpt["img_adapter"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_scheduler" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        save_loss_history = config["training"].get("save_loss_history", False)
        if save_loss_history:
            loss_history = ckpt.get("loss_history", None)
        del ckpt

    # ---- Dataset — single-process WeightedRandomSampler ----
    data_cfg = config["data"]
    dataset = MultiDatasetEmbeddingDataset(
        datasets_config=data_cfg["datasets"],
        resolution=data_cfg["resolution"],
        rebalance_alpha=data_cfg.get("rebalance_alpha", 0.7),
    )
    print(dataset.dataset_summary())

    num_samples = data_cfg.get("num_samples_per_epoch") or len(dataset)
    sampler = WeightedRandomSampler(
        weights=dataset.get_sample_weights(),
        num_samples=num_samples,
        replacement=True,
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
    unet_device = unet.first_device  # where UNet inputs must land

    # ---- Training loop ----
    global_step = start_step
    max_steps = config["training"]["max_steps"]
    save_loss_history = config["training"].get("save_loss_history", False)
    if loss_history is None:
        # Train keys update every log_every steps; val keys update every val_every steps.
        # They are logged independently — pre-create all keys so iteration is stable.
        loss_history = {
            "total": [], "diffusion": [], "semantic": [],
            "val/loss_diffusion": [], "val/cosine_sim": [], "val/lpips": [],
        }

    print(f"Training from step {global_step} / {max_steps}")
    print(f"Trainable params: {sum(p.numel() for p in params if p.requires_grad) / 1e6:.1f}M")
    print(f"GPUs: {torch.cuda.device_count()}  |  mode: {training_mode}")
    pbar = tqdm(total=max_steps, initial=global_step, desc="Training", dynamic_ncols=True, miniters=100)

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            z_img_emb = batch["z_img"].to(device)   # (B, N_patches, D) — on GPU 0
            images    = batch["image"].to(device)    # (B, 3, H, W)
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

            # UNet forward — inputs are moved to unet_device inside SDXLUNetSharded
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                eps_pred = unet(x_t, t, tokens, pooled, time_ids)
                eps_pred = eps_pred.to(device)  # bring back to GPU 0 for loss
                losses = criterion(eps_pred, noise)

            # Semantic loss
            sem_every = config["training"].get("sem_loss_every", 10)
            if qwen_enc is not None and global_step % sem_every == 0:
                with torch.no_grad():
                    alpha_t = scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
                    x0_pred = (x_t - (1 - alpha_t).sqrt() * eps_pred) / alpha_t.sqrt()
                    img_pred = vae.decode(x0_pred.clamp(-1, 1))
                    from torchvision.transforms.functional import to_pil_image
                    img_pred_448 = F.interpolate(img_pred, size=(448, 448), mode="bilinear", align_corners=False)
                    pil_list = [to_pil_image(((img_pred_448[i] + 1) / 2).clamp(0, 1).cpu()) for i in range(B)]
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

            # ---- Logging ----
            postfix = {"loss": f"{losses['total'].item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
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

            # ---- Visualization ----
            if global_step % config["training"]["visualize_every"] == 0:
                vis_dir = os.path.join(config["training"]["output_dir"], "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                img_adapter.eval()
                unet.eval()
                with torch.no_grad():
                    vis_n = min(4, B)
                    gt = (images[:vis_n] * 0.5 + 0.5).clamp(0, 1).cpu()

                    vis_tokens, vis_pooled = img_adapter(z_img_emb[:vis_n])
                    vis_time_ids = make_time_ids(vis_n, resolution, device)

                    # Full DDIM from pure noise — matches inference exactly.
                    from diffusers import DDIMScheduler as _DDIMSched
                    _ddim = _DDIMSched(
                        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False,
                    )
                    _ddim.set_timesteps(10)
                    x = torch.randn(vis_n, 4, resolution // 8, resolution // 8, device=device)
                    for _t in _ddim.timesteps:
                        _t_b = torch.full((vis_n,), _t, device=device, dtype=torch.long)
                        _eps = unet(x, _t_b, vis_tokens, vis_pooled, vis_time_ids).to(device)
                        x = _ddim.step(_eps, _t, x).prev_sample
                    pred = (vae.decode(x) * 0.5 + 0.5).clamp(0, 1).cpu()

                    grid = make_grid(torch.cat([gt, pred], dim=0), nrow=vis_n)
                    save_image(grid, os.path.join(vis_dir, f"step_{global_step}.png"))
                img_adapter.train()
                unet.train()

            # ---- Checkpoint ----
            if global_step % config["training"]["save_every"] == 0:
                ckpt_dir = os.path.join(config["training"]["output_dir"], "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_payload = {
                    "step": global_step,
                    "img_adapter": img_adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if save_loss_history:
                    ckpt_payload["loss_history"] = loss_history
                if training_mode == "lora":
                    unet.save_lora(os.path.join(ckpt_dir, f"lora_step_{global_step}"))
                else:
                    ckpt_payload["unet"] = unet.unet.state_dict()
                torch.save(ckpt_payload, os.path.join(ckpt_dir, f"step_{global_step}.pt"))

    pbar.close()
    save_loss_plot(loss_history, os.path.join(config["training"]["output_dir"], "plots"))
    if training_mode == "lora":
        unet.save_lora(os.path.join(config["training"]["output_dir"], "lora_final"))
        torch.save({"step": global_step, "img_adapter": img_adapter.state_dict()},
                   os.path.join(config["training"]["output_dir"], "final.pt"))
    else:
        torch.save({"step": global_step, "unet": unet.unet.state_dict(),
                    "img_adapter": img_adapter.state_dict()},
                   os.path.join(config["training"]["output_dir"], "final.pt"))
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, resume_path=args.resume)
