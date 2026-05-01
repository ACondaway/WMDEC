"""
Validation utilities for the SigLIP/Qwen → diffusion decoder.

Two validation modes:
  fast  — diffusion loss + cosine similarity in Qwen patch space.
          No image generation; runs in seconds.
          Trigger: every `val_every` training steps.

  full  — DDIM reconstruction → LPIPS vs ground truth + image grid saved.
          Slower (proportional to DDIM steps × val_num_samples).
          Trigger: every `val_full_every` training steps.

Both modes run on rank-0 only.  Results are returned as a plain dict so the
caller can log / compare / decide whether to save a "best" checkpoint.

Backbone support:
  backbone="sdxl"  — ε-prediction; UNet takes (x, t, tokens, pooled, time_ids)
  backbone="sd21"  — v-prediction; UNet takes (x, t, tokens)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

try:
    import lpips as lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def _make_time_ids(batch_size: int, resolution: int, device: torch.device) -> torch.Tensor:
    """SDXL added conditioning: [orig_h, orig_w, crop_y, crop_x, target_h, target_w]."""
    ids = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        dtype=torch.float32, device=device,
    )
    return ids.unsqueeze(0).expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Val dataloader helper
# ---------------------------------------------------------------------------

def build_val_loader(
    config: dict,
    train_dataset=None,
) -> DataLoader:
    """
    Build a small, fixed-seed validation dataloader.

    Priority:
      1. config["data"]["val_datasets"] — separate val dataset configs.
      2. Fall back to a random fixed-seed subset of `train_dataset`.

    Returns a single-process DataLoader (no sampler needed; rank-0 only).
    """
    from data.dataset import MultiDatasetEmbeddingDataset

    val_cfg = config["data"].get("val_datasets")
    num_val = config["validation"].get("val_num_samples", 200)

    if val_cfg:
        dataset = MultiDatasetEmbeddingDataset(
            datasets_config=val_cfg,
            resolution=config["data"]["resolution"],
            rebalance_alpha=0.0,   # equal weight across val datasets
        )
    else:
        # Carve a fixed-seed subset out of the training dataset.
        assert train_dataset is not None, \
            "Either val_datasets or train_dataset must be provided."
        g = torch.Generator()
        g.manual_seed(42)
        indices = torch.randperm(len(train_dataset), generator=g)[:num_val].tolist()
        dataset = Subset(train_dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=config["validation"].get("val_batch_size", 4),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    return loader


# ---------------------------------------------------------------------------
# Fast validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_fast_validation(
    img_adapter,
    unet,
    vae,
    qwen_enc,
    scheduler,
    val_loader: DataLoader,
    device: torch.device,
    lambda_sem: float,
    output_dir: str = None,
    global_step: int = 0,
    n_vis: int = 4,
    backbone: str = "sdxl",
    resolution: int = 512,
) -> dict:
    """
    Compute diffusion loss and cosine similarity over the val set.
    No image generation; uses single-step x_0 prediction for cosine sim.

    Returns:
        {
          "val/loss_diffusion": float,
          "val/cosine_sim":     float,   # 1 - semantic_loss; higher = better
          "val/loss_total":     float,
        }
    """
    img_adapter.eval()
    unet.eval()

    total_diff_loss = 0.0
    total_cos_sim = 0.0
    n_batches = 0
    vis_saved = False

    for batch in val_loader:
        z_img_emb = batch["z_img"].to(device)   # (B, N_patches, D)
        images = batch["image"].to(device)        # (B, 3, H, W)
        B = images.shape[0]

        latent = vae.encode(images)

        # No condition dropout during validation.
        tokens, pooled = img_adapter(z_img_emb)

        noise = torch.randn_like(latent)
        t = scheduler.sample_timesteps(B, device)
        x_t = scheduler.q_sample(latent, t, noise)

        # Backbone-dependent target and UNet call.
        if backbone == "sd21":
            target = scheduler.get_v_target(latent, noise, t)
            pred = unet(x_t, t, tokens)
        else:
            target = noise
            time_ids = _make_time_ids(B, resolution, device)
            pred = unet(x_t, t, tokens, pooled, time_ids)

        diff_loss = F.mse_loss(pred, target).item()

        # Single-step x_0 estimate → cosine sim in Qwen space.
        x0_pred = None
        if qwen_enc is not None:
            if backbone == "sd21":
                x0_pred = scheduler.predict_x0_from_v(pred, x_t, t).clamp(-1, 1)
            else:
                x0_pred = scheduler.predict_x0_from_eps(pred, x_t, t).clamp(-1, 1)

            img_pred = vae.decode(x0_pred)
            img_pred_448 = F.interpolate(
                img_pred, size=(448, 448), mode="bilinear", align_corners=False
            )
            pil_list = [
                to_pil_image(((img_pred_448[i] + 1) / 2).clamp(0, 1).cpu())
                for i in range(B)
            ]
            patch_pred = qwen_enc.encode_images(pil_list, device=device)
            z_pred = patch_pred.mean(dim=1)
            z_target = z_img_emb.mean(dim=1)
            cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean().item()
        else:
            cos_sim = 0.0

        # Save GT vs x0 reconstruction from the first val batch.
        if not vis_saved and output_dir is not None and x0_pred is not None:
            _n = min(n_vis, B)
            gt_vis   = (images[:_n] * 0.5 + 0.5).clamp(0, 1).cpu()
            pred_vis = (vae.decode(x0_pred[:_n]) * 0.5 + 0.5).clamp(0, 1).cpu()
            grid = make_grid(torch.cat([gt_vis, pred_vis], dim=0), nrow=_n)
            val_vis_dir = os.path.join(output_dir, "val_visualizations")
            os.makedirs(val_vis_dir, exist_ok=True)
            save_image(grid, os.path.join(val_vis_dir, f"step_{global_step}.png"))
            vis_saved = True

        total_diff_loss += diff_loss
        total_cos_sim += cos_sim
        n_batches += 1

    img_adapter.train()
    unet.train()

    avg_diff = total_diff_loss / max(n_batches, 1)
    avg_cos = total_cos_sim / max(n_batches, 1)
    sem_loss = 1.0 - avg_cos
    total = avg_diff + lambda_sem * sem_loss

    return {
        "val/loss_diffusion": avg_diff,
        "val/cosine_sim": avg_cos,        # higher is better
        "val/loss_total": total,
    }


# ---------------------------------------------------------------------------
# Full validation (DDIM + LPIPS)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_full_validation(
    img_adapter,
    unet,
    vae,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    global_step: int,
    ddim_steps: int = 20,
    cfg_scale: float = 2.0,
    max_batches: int = 4,
    backbone: str = "sdxl",
    model_name: str = None,
    resolution: int = 512,
) -> dict:
    """
    Generate images with DDIM and compute LPIPS.

    Saves a side-by-side GT | Reconstruction grid to
      {output_dir}/val_visualizations/full/step_{global_step}.png

    Returns:
        {
          "val/lpips": float,   # lower is better; -1.0 if lpips not installed
        }
    """
    from diffusers import DDIMScheduler as HFDDIMScheduler
    from training.cfg import build_uncond_context

    # Load DDIM scheduler — from_pretrained picks up the correct prediction_type.
    if model_name is not None:
        ddim_sched = HFDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
    else:
        ddim_sched = HFDDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
    ddim_sched.set_timesteps(ddim_steps)

    lpips_fn = None
    if _LPIPS_AVAILABLE:
        lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
        lpips_fn.eval()

    img_adapter.eval()
    unet.eval()

    all_gt, all_pred = [], []
    total_lpips = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break

        z_img_emb = batch["z_img"].to(device)
        images = batch["image"].to(device)
        B = images.shape[0]

        tokens, pooled = img_adapter(z_img_emb)

        # Build unconditional context for CFG.
        pooled_dim = pooled.shape[1] if pooled is not None else 0
        uncond_tokens, uncond_pooled = build_uncond_context(
            B, tokens.shape[1], tokens.shape[2], device,
            pooled_proj_dim=pooled_dim,
        )

        latent_h = latent_w = resolution // 8

        # DDIM denoising loop.
        x = torch.randn(B, 4, latent_h, latent_w, device=device)
        for t in ddim_sched.timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            if cfg_scale > 1.0:
                if backbone == "sd21":
                    pred_cond   = unet(x, t_batch, tokens)
                    pred_uncond = unet(x, t_batch, uncond_tokens)
                else:
                    _time_ids = _make_time_ids(B, resolution, device)
                    pred_cond   = unet(x, t_batch, tokens,       pooled,       _time_ids)
                    pred_uncond = unet(x, t_batch, uncond_tokens, uncond_pooled, _time_ids)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                if backbone == "sd21":
                    pred = unet(x, t_batch, tokens)
                else:
                    _time_ids = _make_time_ids(B, resolution, device)
                    pred = unet(x, t_batch, tokens, pooled, _time_ids)
            x = ddim_sched.step(pred, t, x).prev_sample

        pred_images = vae.decode(x)                             # (B, 3, H, W) in [-1, 1]
        pred_01 = (pred_images * 0.5 + 0.5).clamp(0, 1)
        gt_01   = (images       * 0.5 + 0.5).clamp(0, 1)

        all_gt.append(gt_01.cpu())
        all_pred.append(pred_01.cpu())

        if lpips_fn is not None:
            # LPIPS expects [-1, 1].
            lp = lpips_fn(pred_images.clamp(-1, 1), images.clamp(-1, 1)).mean().item()
            total_lpips += lp

        n_batches += 1

    img_adapter.train()
    unet.train()

    # Save comparison grid.
    if all_gt:
        gt_cat   = torch.cat(all_gt,   dim=0)
        pred_cat = torch.cat(all_pred, dim=0)
        n_show   = min(8, gt_cat.shape[0])
        grid = make_grid(
            torch.cat([gt_cat[:n_show], pred_cat[:n_show]], dim=0),
            nrow=n_show,
        )
        vis_dir = os.path.join(output_dir, "val_visualizations", "full")
        os.makedirs(vis_dir, exist_ok=True)
        save_image(grid, os.path.join(vis_dir, f"step_{global_step}.png"))

    avg_lpips = total_lpips / max(n_batches, 1) if lpips_fn is not None else -1.0
    return {"val/lpips": avg_lpips}


# ---------------------------------------------------------------------------
# Best-checkpoint tracker
# ---------------------------------------------------------------------------

class BestCheckpointTracker:
    """
    Tracks the best validation metric and tells the caller whether to save
    a "best" checkpoint.

    Lower-is-better metrics: loss_total, lpips
    Higher-is-better metrics: cosine_sim
    """

    def __init__(self, metric: str = "val/cosine_sim"):
        self.metric = metric
        self._higher_is_better = "cosine_sim" in metric
        self.best_value = float("-inf") if self._higher_is_better else float("inf")
        self.best_step = -1

    def is_new_best(self, metrics: dict) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self._higher_is_better:
            improved = value > self.best_value
        else:
            improved = value < self.best_value
        if improved:
            self.best_value = value
            return True
        return False

    def update(self, metrics: dict, step: int) -> bool:
        """Returns True if this is a new best."""
        if self.is_new_best(metrics):
            self.best_step = step
            return True
        return False
