# Qwen Visual Embedding Decoder

Reconstructs high-resolution images from frozen **Qwen3-VL-4B** visual backbone embeddings
using a **fine-tuned Stable Diffusion XL** latent diffusion model.
Supports multiple heterogeneous datasets with automatic stats-based rebalancing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Encoding (frozen, offline)                                         │
│                                                                     │
│  image ──► Qwen3-VL-4B Visual Encoder ──► (196, 2560) patches      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ stored as .pt  (bfloat16)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Training  (trainable: ImageAdapter + SDXL UNet)                    │
│                                                                     │
│  (B, 196, 2560) patches                                             │
│       │                                                             │
│       ▼  ImageAdapter                                               │
│       │  ├─ CrossAttentionPooler  ──► tokens      (B, 16, 2048)     │
│       │  └─ Pooled MLP            ──► pooled_proj  (B, 1280)        │
│       │                                                             │
│       ▼  SDXL UNet2DConditionModel (full fine-tune)                 │
│       │  encoder_hidden_states = tokens                             │
│       │  added_cond_kwargs["text_embeds"] = pooled_proj             │
│       │  added_cond_kwargs["time_ids"]    = [H,W,0,0,H,W]          │
│       │                                                             │
│  noise latent + timestep ──► denoised latent (B, 4, 128, 128)      │
│                                                   │                 │
│                              SDXL VAE decoder ◄──┘                 │
│                                   │                                 │
│                         reconstructed image (B, 3, 1024, 1024)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Visual encoder | Qwen3.5-VL-4B (frozen) | Rich spatial patch features (196 tokens × 2560 dim; ViT hidden=1024, projected to out_hidden_size=2560) |
| Adapter | Perceiver cross-attn pooler | Compresses variable-length patches to fixed 16 tokens |
| Diffusion backbone | SDXL UNet (full fine-tune) | Pretrained priors → high-quality textures immediately |
| VAE | SDXL VAE (frozen, scale=0.13025) | Matches the SDXL latent space |
| Conditioning | Dual: tokens + pooled\_proj | Required by SDXL's `addition_embed_type="text_time"` |
| Noise schedule | Loaded from SDXL pretrained | Exact match to UNet training distribution |

---

## Project Structure

```
project/
├── models/
│   ├── qwen_visual_encoder.py   # Frozen Qwen3-VL-4B visual backbone
│   ├── adapter.py               # CrossAttentionPooler → (tokens, pooled_proj)
│   ├── unet.py                  # SDXLUNet wrapper (UNet2DConditionModel)
│   └── vae.py                   # SDXL VAE wrapper (scaling_factor auto-read)
│
├── diffusion/
│   ├── scheduler.py             # DDPMScheduler loaded from SDXL pretrained
│   └── sampler.py               # DDIMSampler using diffusers DDIMScheduler
│
├── training/
│   ├── train.py                 # DDP training with stats-based rebalancing
│   ├── loss.py                  # Diffusion loss + Qwen semantic consistency
│   └── cfg.py                   # Image-only CFG dropout (tokens + pooled)
│
├── data/
│   ├── dataset.py               # MultiDatasetEmbeddingDataset + DistributedWeightedSampler
│   ├── stats.py                 # Statistics collection, printing, JSON persistence
│   └── preprocessors/
│       ├── __init__.py          # PREPROCESSORS registry
│       ├── base.py              # BaseDatasetPreprocessor (abstract API)
│       └── robobrain_dex.py     # RoboBrain-Dex implementation
│
├── configs/
│   └── base.yaml                # All hyperparameters
│
├── inference/
│   └── sample.py                # Reconstruct image from .pt embedding
│
└── scripts/
    ├── extract_qwen_visual_encoder.py   # Step 0: save standalone visual backbone
    ├── preprocess_embeddings.py         # Step 1: offline embedding extraction
    └── preprocess_config.yaml           # Multi-dataset preprocessing config
```

---

## Environment Setup

```bash
conda create -n qwen_sdxl_decoder python=3.10 -y
conda activate qwen_sdxl_decoder

# PyTorch (CUDA 12.1)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Core dependencies
pip install "diffusers>=0.28.0" accelerate "transformers>=4.45.0"
pip install pillow tqdm pyyaml lpips matplotlib qwen-vl-utils peft
```

> **Version notes:**
> - `diffusers >= 0.28` — SDXL `UNet2DConditionModel` and `DDIMScheduler`
> - `transformers >= 4.57.0.dev0` — `Qwen3_5ForConditionalGeneration` (Qwen3.5-VL)

---

## Step 0 — Extract the Qwen Visual Backbone (once)

Extract only the ViT tower from the full Qwen3.5-VL-4B model (~8 GB) and save it as a
standalone checkpoint (~900 MB). Run once; the full model can be deleted after.

```bash
python scripts/extract_qwen_visual_encoder.py \
    --model_name Qwen/Qwen3.5-4B \
    --output /share/project/congsheng/qwen_visual/qwen3_5_visual_encoder_4b.pt
```

---

## Step 1 — Preprocess Datasets (Offline, Multi-GPU)

Encodes all images through the Qwen visual encoder and stores patch embeddings on disk.
After processing, per-dataset and combined **statistics** are printed and saved as JSON.

### Single dataset

```bash
torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \
    --dataset robobrain-dex \
    --image_dir /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex \
    --output_dir /share/project/congsheng/WMDEC_qwen/robobrain-dex-qwen-embedding \
    --encoder_ckpt /share/project/congsheng/qwen_visual/qwen3_5_visual_encoder_4b.pt \
    --batch_size 16
```

### Multiple datasets (via config)

```bash
torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \
    --config scripts/preprocess_config.yaml \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt
```

`scripts/preprocess_config.yaml`:
```yaml
output_root: /share/project/congsheng/all-qwen-embeddings

datasets:
  - name: robobrain-dex
    image_dir: /share/project/hotel/.../robobrain-dex
  - name: my-new-dataset
    image_dir: /share/.../my_new_dataset
```

### Output layout

```
{output_root}/{dataset_name}/
    stats.json                ← frame counts, per-task breakdown, timing, disk size
    {subdir}/.../{frame}.pt   ← {"z_img": Tensor(1024,1152,bfloat16), "dataset_name":str, ...}
```

Each `.pt` file: `z_img` has shape `(196, 2560)` in `bfloat16` (~1.0 MB/file).

### Statistics output

After each dataset finishes, the script prints:

```
────────────────────────────────────────────────────────────
  Dataset : robobrain-dex
────────────────────────────────────────────────────────────
  Total frames    :    1,234,567
  Written         :      987,654
  Skipped (cached):      246,913
  Groups          :           42
  Processing time :      2h 34m
  Output size     :       2.31 GB

  Per-group breakdown (top 20):
    pick_and_place                           45,678  ( 3.7%)
    open_drawer                              32,456  ( 2.6%)
    ...
════════════════════════════════════════════════════════════
  COMBINED STATISTICS  (2 dataset(s))
════════════════════════════════════════════════════════════
  robobrain-dex                      1,234,567  ( 72.3%)
  my-new-dataset                       472,890  ( 27.7%)
  ─────────────────────────────────────────────
  TOTAL                              1,707,457  (100.0%)
  ...
```

And saves `{output_root}/combined_stats.json` + `{output_root}/{dataset_name}/stats.json`.

The processing is **resumable**: already-written `.pt` files are skipped (atomic write via
`.tmp` → rename prevents partial files on interruption).

---

## Step 2 — Training (4×H100)

```bash
torchrun --nproc_per_node=4 training/train.py --config configs/base.yaml
```

### Multi-dataset rebalancing

The training dataloader uses `DistributedWeightedSampler` with temperature-scaled
probabilities derived from the **verified frame counts in `stats.json`**:

```
p_i  =  n_i^α  /  Σ_j  n_j^α
```

where `n_i` is read from `{embedding_dir}/stats.json` (set during preprocessing),
not counted from disk — so sampling weights are stable even during a partial run.

| α value | Effect |
|---------|--------|
| `1.0` | Proportional — large datasets dominate exactly as in their raw ratio |
| `0.7` | Recommended — moderate smoothing, standard in multilingual NLP |
| `0.5` | Square-root smoothing — noticeable uplift for small datasets |
| `0.0` | Fully balanced — equal probability per dataset regardless of size |

At training startup, the composition table is printed:

```
Rebalancing  α = 0.70   (0=balanced, 1=proportional)
Frame counts from: robobrain-dex(stats.json), my-new-dataset(stats.json)
──────────────────────────────────────────────────────────────────────────────────────────
  Dataset                       Stats frames    Disk frames    Raw %   Effective %
──────────────────────────────────────────────────────────────────────────────────────────
  robobrain-dex                    1,234,567      1,234,567    72.3%        65.1%
  my-new-dataset                     472,890        472,890    27.7%        34.9%
──────────────────────────────────────────────────────────────────────────────────────────
  TOTAL                            1,707,457      1,707,457   100.0%       100.0%
```

Configure in `configs/base.yaml`:

```yaml
data:
  rebalance_alpha: 0.7     # tune this
  datasets:
    - name: "robobrain-dex"
      embedding_dir: "/share/project/congsheng/robobrain-dex-qwen-embedding"
      image_dir: "/share/project/hotel/.../rodobrain-dex"
    - name: "my-new-dataset"
      embedding_dir: "/share/project/congsheng/my-new-dataset-qwen-embedding"
      image_dir: "/share/.../my_new_dataset"
```

### Default training config

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL UNet + VAE + scheduler |
| Resolution | 1024×1024 | Latent 128×128 |
| Per-GPU batch | 2 | 2×4 GPUs = global 8 |
| Max steps | 300k | |
| Learning rate | 5e-5 | Lower than SD scratch due to fine-tune |
| Grad checkpointing | enabled | Required for 1024px on H100 |
| CFG drop prob | 10% | Both `tokens` and `pooled_proj` zeroed together |
| λ (semantic loss) | 0.1 | `1 - cosine(Qwen(x̂), z_img)` |
| α (rebalancing) | 0.7 | Stats-based temperature sampling |

**Expected time:** ~18–36 hours on 4×H100 for 300k steps at 1024px.

> **VRAM note:** SDXL full fine-tune (~2.6B) + gradient checkpointing ≈ 65 GB/GPU.
> Reduce `batch_size_per_gpu` to 1 if OOM.

### Resume from checkpoint

```bash
torchrun --nproc_per_node=4 training/train.py \
    --config configs/base.yaml \
    --resume /path/to/checkpoints/step_50000.pt
```

### Loss function

```
L = L_diff + λ · L_sem

L_diff  =  MSE(ε_pred, ε)                                        [diffusion]
L_sem   =  1 − cosine(mean_pool(Qwen(x̂₀)), mean_pool(z_img))   [semantic]
```

`L_sem` is computed every `sem_loss_every` steps (default 10) to amortise the cost of
re-encoding the reconstructed image through the frozen Qwen encoder.

### Outputs

```
{output_dir}/
├── checkpoints/step_XXXXX.pt   ← periodic checkpoints (unet, img_adapter, optimizer, ...)
├── visualizations/step_XXXXX.png  ← GT vs reconstruction side-by-side grids
├── plots/loss_curve.png         ← total / diffusion / semantic loss curves
└── final.pt                    ← final weights (unet + img_adapter only)
```

---

## Step 3 — Inference

```bash
python inference/sample.py \
    --config configs/base.yaml \
    --checkpoint /path/to/final.pt \
    --embedding /path/to/frame.pt \
    --output reconstructed.png \
    --cfg_scale 2.0 \
    --steps 50
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding` | required | Pre-extracted `.pt` embedding file |
| `--cfg_scale` | `2.0` | CFG guidance scale (1.5–3.0 recommended) |
| `--steps` | `50` | DDIM steps (20 for faster, 100 for higher quality) |

---

## Adding a New Dataset

1. **Implement the preprocessor** (`data/preprocessors/my_dataset.py`):

```python
from data.preprocessors.base import BaseDatasetPreprocessor, SampleMeta
from pathlib import Path
from PIL import Image

class MyDatasetPreprocessor(BaseDatasetPreprocessor):

    @property
    def dataset_name(self) -> str:
        return "my-dataset"           # must match the name in PREPROCESSORS

    def find_samples(self):
        # Return a sorted list of SampleMeta covering all images
        samples = []
        for img_path in sorted(self.image_root.rglob("*.jpg")):
            rel = img_path.relative_to(self.image_root)
            samples.append(SampleMeta(
                rel_path=rel,
                extra_meta={"split": rel.parts[0]},   # any extra info
            ))
        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(self.image_root / sample.rel_path).convert("RGB")

    def stats_key(self, sample: SampleMeta) -> str:
        # Group key for per-group statistics (e.g. split, task, scene)
        return sample.extra_meta["split"]
```

2. **Register it** (`data/preprocessors/__init__.py`):

```python
from .my_dataset import MyDatasetPreprocessor

PREPROCESSORS = {
    "robobrain-dex": RoboBrainDexPreprocessor,
    "my-dataset":    MyDatasetPreprocessor,     # ← add this line
}
```

3. **Add to preprocessing config** (`scripts/preprocess_config.yaml`):

```yaml
datasets:
  - name: my-dataset
    image_dir: /path/to/my/images
```

4. **Add to training config** (`configs/base.yaml`):

```yaml
data:
  datasets:
    - name: my-dataset
      embedding_dir: /path/to/my-dataset-qwen-embedding
      image_dir: /path/to/my/images
```

The abstract base class handles output path derivation, atomic file writes,
skip-if-exists logic, and statistics collection automatically.

---

## Evaluation

**Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Cosine similarity (Qwen) | `cosine(mean_pool(Qwen(x̂)), mean_pool(z_img))` | ↑ higher |
| LPIPS | Perceptual similarity | ↓ lower |
| FID | Distribution-level quality (optional) | ↓ lower |

---

## Development Roadmap

1. Extract Qwen visual backbone → `scripts/extract_qwen_visual_encoder.py`
2. Preprocess datasets → `scripts/preprocess_embeddings.py` (generates `stats.json`)
3. Fine-tune SDXL UNet with rebalanced multi-dataset loading → `training/train.py`
4. Enable semantic loss → `lambda_sem > 0` in config
5. Evaluate reconstruction quality (LPIPS, cosine sim, FID)
6. Scale to additional datasets by implementing `BaseDatasetPreprocessor`
