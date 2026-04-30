# Qwen Visual Embedding Decoder

Reconstructs high-resolution images from frozen **Qwen3.5-4B** visual backbone embeddings
using a **fine-tuned Stable Diffusion XL** latent diffusion model.
Supports multiple heterogeneous datasets with automatic stats-based rebalancing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Encoding (frozen, offline)                                         │
│                                                                     │
│  image ──► Qwen3.5-4B Visual Encoder ──► (196, 2560) patches       │
│            ViT hidden=1024, out_hidden=2560, patch=16, merge=2×2    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ stored as .pt  (bfloat16, ~1 MB/file)
                                 │ includes _abs_image_path for training
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
│       ▼  SDXL UNet2DConditionModel (full fine-tune or LoRA)         │
│       │  encoder_hidden_states = tokens                             │
│       │  added_cond_kwargs["text_embeds"] = pooled_proj             │
│       │  added_cond_kwargs["time_ids"]    = [H,W,0,0,H,W]          │
│       │                                                             │
│  noise latent + timestep ──► denoised latent (B, 4, 128, 128)      │
│                                                   │                 │
│                              SDXL VAE decoder ◄──┘  (frozen)       │
│                                   │                                 │
│                         reconstructed image (B, 3, 1024, 1024)     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Visual encoder | Qwen3.5-4B (frozen) | 196 spatial patch tokens × 2560 dim; ViT hidden=1024 projected via merger to 2560 |
| Adapter | Perceiver cross-attn pooler | Compresses variable-length patches to fixed 16 tokens |
| Diffusion backbone | SDXL UNet (full fine-tune or LoRA) | Pretrained priors → high-quality textures immediately |
| VAE | SDXL VAE (**frozen**, scale=0.13025) | VAE training requires separate GAN pipeline; not trained here |
| Conditioning | Dual: tokens + pooled\_proj | Required by SDXL `addition_embed_type="text_time"` |
| Noise schedule | Loaded from SDXL pretrained | Exact match to UNet training distribution |

---

## Project Structure

```
project/
├── models/
│   ├── qwen_visual_encoder.py   # Frozen Qwen3.5-4B visual backbone
│   │                            #   - from_standalone(): loads .pt checkpoint
│   │                            #   - encode_images(): strict (B,196,2560) contract
│   ├── adapter.py               # CrossAttentionPooler → (tokens, pooled_proj)
│   ├── unet.py                  # SDXLUNet wrapper; setup_lora() for LoRA mode
│   └── vae.py                   # SDXL VAE wrapper (frozen, scaling_factor auto-read)
│
├── diffusion/
│   ├── scheduler.py             # DDPMScheduler loaded from SDXL pretrained
│   └── sampler.py               # DDIMSampler for inference
│
├── training/
│   ├── train.py                 # Multi-process DDP training (torchrun)
│   ├── train_single.py          # Single-process multi-GPU via device_map="auto"
│   ├── validate.py              # Fast val (loss+cosine) + full val (DDIM+LPIPS)
│   ├── loss.py                  # Diffusion loss + Qwen semantic consistency
│   └── cfg.py                   # CFG dropout (tokens + pooled zeroed atomically)
│
├── data/
│   ├── dataset.py               # MultiDatasetEmbeddingDataset + samplers
│   │                            #   - reads _abs_image_path from .pt for true GT
│   ├── stats.py                 # Statistics collection, printing, JSON persistence
│   └── preprocessors/
│       ├── __init__.py          # PREPROCESSORS registry
│       ├── base.py              # BaseDatasetPreprocessor (abstract API)
│       │                        #   - saves _abs_image_path in every .pt
│       └── robobrain_dex.py     # RoboBrain-Dex implementation (flattened rel_path)
│
├── configs/
│   └── base.yaml                # All hyperparameters incl. training_mode
│
├── inference/
│   └── sample.py                # Reconstruct image from .pt embedding
│
└── scripts/
    ├── extract_qwen_visual_encoder.py   # Step 0: save standalone visual backbone
    │                                    #   saves visual_cls_name for exact reconstruction
    ├── preprocess_embeddings.py         # Step 1: offline embedding extraction
    └── preprocess_config.yaml          # Multi-dataset preprocessing config
```

---

## Environment Setup

```bash
conda create -n qwen_sdxl_decoder python=3.10 -y
conda activate qwen_sdxl_decoder

# PyTorch (CUDA 12.1)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Core dependencies
pip install "diffusers==0.30.3"          # pin — newer versions break with PyTorch <2.4
pip install accelerate "transformers>=4.57.0.dev0"
pip install pillow tqdm pyyaml lpips matplotlib qwen-vl-utils
pip install peft                         # required for LoRA mode only
```

> **Version notes:**
> - `diffusers==0.30.3` — newer versions (0.31+) use flash_attn_3 annotations incompatible with PyTorch <2.4
> - `transformers>=4.57.0.dev0` — required for `Qwen3_5ForConditionalGeneration`
> - `peft` — only needed when `training_mode: "lora"` in config

---

## Step 0 — Extract the Qwen Visual Backbone (once)

Extract the ViT tower from the full Qwen3.5-4B model (~9 GB) and save it as a
standalone `.pt` checkpoint. Run once; the full model can be deleted after.

```bash
python scripts/extract_qwen_visual_encoder.py \
    --model_name Qwen/Qwen3.5-4B \
    --output /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt
```

The checkpoint stores:
- `state_dict` — visual tower weights (`model.model.visual`)
- `vision_config` — config dict for reconstruction
- `hidden_size` — `out_hidden_size = 2560`
- `visual_cls_name` / `visual_cls_module` — exact class used (e.g. `Qwen3_5VisionTransformer`)

> **Important:** The visual tower is at `model.model.visual` (not `model.visual`).
> The class saved in `visual_cls_name` is used at load time to ensure the merger
> (1024→2560 projection) runs correctly. Re-extract if upgrading transformers.

---

## Step 1 — Preprocess Datasets (Offline, Multi-GPU)

Encodes all images through the frozen Qwen visual encoder and stores `(196, 2560)`
patch embeddings on disk. Each `.pt` file also stores `_abs_image_path` so the
training dataloader can load the true GT image without relying on path reconstruction.

### Single dataset

```bash
torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \
    --dataset robobrain-dex \
    --image_dir /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex \
    --output_dir /share/project/congsheng/robobrain-dex-qwen-embedding \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \
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
    stats.json                  ← frame counts, per-task breakdown, timing, disk size
    {task}/{episode}/{frame}.pt ← {"z_img": (196,2560), "_abs_image_path": str, ...}
```

> **embedding_dir in training config must point to `{output_root}/{dataset_name}/`**,
> not `{output_root}/`. The preprocessor nests files under the dataset name subdirectory.

### Statistics output

```
────────────────────────────────────────────────────────────
  Dataset : robobrain-dex
────────────────────────────────────────────────────────────
  Total frames    :    1,234,567
  Written         :      987,654
  Skipped (cached):      246,913
  Processing time :      2h 34m
  Output size     :       2.31 GB
  Per-group breakdown: pick_and_place 45,678 (3.7%) ...
════════════════════════════════════════════════════════════
  COMBINED STATISTICS  (2 dataset(s))
════════════════════════════════════════════════════════════
  robobrain-dex       1,234,567  (72.3%)
  my-new-dataset        472,890  (27.7%)
  TOTAL               1,707,457
```

Processing is **resumable**: existing `.pt` files are skipped (atomic `.tmp` → rename).

---

## Step 2 — Training

Two launch modes are available. Both support `full` fine-tune and `LoRA` via
`training_mode` in the config.

### Mode A — Multi-process DDP (recommended for throughput)

Each GPU holds a full model copy; gradients are averaged across ranks.

```bash
torchrun --nproc_per_node=4 training/train.py --config configs/base.yaml
```

### Mode B — Single-process model parallelism (no torchrun needed)

The UNet is automatically sharded across all GPUs via `device_map="auto"`.
One Python process owns all GPUs — useful when the model is too large for one GPU
or for simpler single-process debugging.

```bash
python training/train_single.py --config configs/base.yaml
```

| | DDP (`train.py`) | Single-process (`train_single.py`) |
|---|---|---|
| Launch | `torchrun --nproc_per_node=N` | `python training/train_single.py` |
| Model placement | 1 full copy per GPU | UNet sharded via `device_map="auto"` |
| Effective batch | `batch_per_gpu × N_gpus` | `batch_per_gpu` |
| Best for | Max throughput | Large model / debugging |

### Training mode (full vs LoRA)

Set in `configs/base.yaml`:

```yaml
training:
  training_mode: "full"   # or "lora"
  lora_rank: 64
  lora_alpha: 64
  # lora_target_modules: null  # null = ["to_q","to_k","to_v","to_out.0","to_add_out"]
```

| Mode | Trainable params | Checkpoint |
|------|-----------------|------------|
| `full` | All UNet weights + ImageAdapter (~2.6B) | `step_N.pt` (full state_dict) |
| `lora` | LoRA deltas + ImageAdapter (~50–100M) | `lora_step_N/` dir + `step_N.pt` (adapter only) |

LoRA mode requires `pip install peft`.

### Multi-dataset rebalancing

Sampling probabilities are derived from verified frame counts in `stats.json`:

```
p_i  =  n_i^α  /  Σ_j  n_j^α
```

| α | Effect |
|---|--------|
| `1.0` | Proportional — large datasets dominate |
| `0.7` | Recommended — moderate smoothing |
| `0.5` | Square-root smoothing |
| `0.0` | Fully balanced — equal probability per dataset |

Configure datasets in `configs/base.yaml`:

```yaml
data:
  rebalance_alpha: 0.7
  datasets:
    - name: "robobrain-dex"
      # Point to the dataset_name subdirectory, not the output_dir root
      embedding_dir: "/share/project/congsheng/robobrain-dex-qwen-embedding/robobrain-dex"
      image_dir: "/share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex"
    - name: "my-new-dataset"
      embedding_dir: "/share/project/congsheng/my-new-dataset-qwen-embedding/my-new-dataset"
      image_dir: "/share/.../my_new_dataset"
```

### Training config reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL UNet + VAE + scheduler |
| `training_mode` | `"full"` | `"full"` or `"lora"` |
| Resolution | 1024×1024 | Latent 128×128 |
| Per-GPU batch | 1 | ×4 GPUs = global 4 in DDP |
| Max steps | 300k | |
| Learning rate | 5e-5 | |
| Grad checkpointing | enabled | Required for 1024px on H100 |
| CFG drop prob | 10% | `tokens` and `pooled_proj` zeroed atomically |
| λ semantic loss | 0.1 | `1 − cosine(Qwen(x̂), z_img)` every 10 steps |
| `rebalance_alpha` | 0.7 | Stats-based temperature sampling |

> **VAE is frozen** — SDXL VAE is used only as a fixed encoder/decoder.
> VAE fine-tuning requires a separate GAN training pipeline and is not performed here.

> **VRAM note:** Full fine-tune SDXL (~2.6B) + grad checkpointing ≈ 40–65 GB/GPU.
> Use LoRA mode or `train_single.py` with `device_map` to reduce per-GPU footprint.

### Loss function

```
L = L_diff + λ · L_sem

L_diff  =  MSE(ε_pred, ε)
L_sem   =  1 − cosine(mean_pool(Qwen(x̂₀)), mean_pool(z_img))
```

`L_sem` is computed every `sem_loss_every` steps (default 10) to amortise
re-encoding cost through the frozen Qwen encoder.

### Validation

Automatically runs on rank 0 during training:

```yaml
validation:
  val_every: 5000        # fast val: diffusion loss + cosine similarity
  val_full_every: 20000  # full val: 20-step DDIM + LPIPS + saves image grid
  val_num_samples: 200   # samples (subset of train data if val_datasets not set)
  best_metric: "val/cosine_sim"  # saves best.pt when this improves
```

Best checkpoint is saved to `{output_dir}/checkpoints/best.pt`.

### Visualization

Every `visualize_every` steps, a GT vs reconstruction grid is saved to
`{output_dir}/visualizations/step_N.png`.

- **GT** — true images loaded from `image_dir` via `_abs_image_path` in the `.pt` file
- **Pred** — 10-step DDIM sample conditioned on the Qwen embedding (no CFG dropout)

### Outputs

```
{output_dir}/
├── checkpoints/
│   ├── step_N.pt            ← periodic checkpoint
│   ├── lora_step_N/         ← LoRA delta weights (lora mode only)
│   └── best.pt              ← best val checkpoint
├── visualizations/step_N.png
├── plots/loss_curve.png
├── val_images/step_N.png    ← full-val GT vs DDIM reconstruction
└── final.pt
```

### Resume from checkpoint

```bash
# DDP
torchrun --nproc_per_node=4 training/train.py \
    --config configs/base.yaml --resume /path/to/checkpoints/step_50000.pt

# Single-process
python training/train_single.py \
    --config configs/base.yaml --resume /path/to/checkpoints/step_50000.pt
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
        return "my-dataset"

    def find_samples(self):
        samples = []
        for img_path in sorted(self.image_root.rglob("*.jpg")):
            rel = img_path.relative_to(self.image_root)
            samples.append(SampleMeta(
                rel_path=rel,
                extra_meta={
                    "split": rel.parts[0],
                    "_abs_image_path": str(img_path),  # required for GT loading
                },
            ))
        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")
```

2. **Register it** (`data/preprocessors/__init__.py`):

```python
PREPROCESSORS = {
    "robobrain-dex": RoboBrainDexPreprocessor,
    "my-dataset":    MyDatasetPreprocessor,
}
```

3. **Add to configs** (preprocessing + training):

```yaml
# scripts/preprocess_config.yaml
datasets:
  - name: my-dataset
    image_dir: /path/to/my/images

# configs/base.yaml
data:
  datasets:
    - name: my-dataset
      embedding_dir: /path/to/my-dataset-qwen-embedding/my-dataset
      image_dir: /path/to/my/images
```

> **Note:** always include `_abs_image_path` in `extra_meta` so the training
> dataloader can load true GT images regardless of the stored `rel_path` structure.

---

## Evaluation

| Metric | Description | Target |
|--------|-------------|--------|
| Cosine similarity (Qwen) | `cosine(mean_pool(Qwen(x̂)), mean_pool(z_img))` | ↑ higher |
| LPIPS | Perceptual similarity (VGG) | ↓ lower |
| FID | Distribution-level quality (optional) | ↓ lower |

---

## Development Roadmap

1. Extract Qwen visual backbone → `scripts/extract_qwen_visual_encoder.py`
2. Preprocess datasets → `scripts/preprocess_embeddings.py`
3. Train with DDP or single-process → `training/train.py` / `training/train_single.py`
4. Monitor via validation metrics and `val_images/` grids
5. Evaluate reconstruction quality (LPIPS, cosine sim)
6. Scale to additional datasets by implementing `BaseDatasetPreprocessor`
