# Qwen Visual Embedding Decoder

Reconstructs images from frozen **Qwen3.5-4B** visual patch embeddings using a
**Stable Diffusion 2.1** latent diffusion model with gated decoupled image
cross-attention — keeping SD's generative prior intact while injecting visual
conditioning through a separate learnable branch.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Offline preprocessing (run once)                                    │
│                                                                      │
│  image ──► Qwen3.5-4B Visual Encoder ──► (196, 2560) z_img patches  │
│  image ──► SD 2.1 VAE Encoder        ──► (4, 56, 56) z_vae latent   │
│                                                                      │
│  Both saved in one .pt file per frame — no images needed at runtime  │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  .pt  {"z_img": (196,2560), "z_vae": (4,56,56)}
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Training  (frozen: base UNet + text encoder + Qwen + VAE)           │
│  Trainable: ImageAdapter + GatedImageCrossAttention layers + gates   │
│                                                                      │
│  z_img (B, 196, 2560)                                                │
│       │                                                              │
│       ▼  PatchProjector (RMSNorm → MLP → scale)                     │
│       ▼  ResidualPooler  (N×cross-attn + optional self-attn + FFN)   │
│       │                                                              │
│       image_tokens  (B, 16, 1024)                                    │
│       │  ↓ image dropout 15% → null_image_tokens  (CFG training)     │
│       │                                                              │
│  z_vae (B, 4, 56, 56)  ── noise → x_t ──► SD 2.1 UNet               │
│                                              │                       │
│  Per transformer block:                      │                       │
│    hidden = hidden                           │                       │
│           + TextCrossAttn(hidden, empty_text)│  ← SD prior anchor    │
│           + tanh(gate) × ImageCrossAttn(hidden, image_tokens)        │
│                                              │  ← starts at 0        │
│                                              ▼                       │
│                                         v_pred                       │
│                                              │                       │
│                             SD 2.1 VAE decoder (frozen)              │
│                                              │                       │
│                                  reconstructed image                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Backbone | SD 2.1-base (v-prediction) | 865M params, 56×56 latent at 448px; smaller than SDXL |
| Conditioning | **Gated decoupled image cross-attn** | Preserves SD generative prior; gates = 0 → starts as vanilla SD |
| Empty text prior | `CLIPTextModel("")` cached once | Stable anchor for SD's existing cross-attn (attn2) |
| ImageAdapter | Multi-layer ResidualPooler | Deeper compression: RMSNorm→MLP→scale, 2-layer cross-attn pooler |
| Null image tokens | Learnable `nn.Parameter(zeros)` | Enables image-CFG at inference; trained with 15% dropout |
| Training mode | `frozen` | Only ~50–200M params trained; saves ~5 GB VRAM per GPU |
| VAE latents | Pre-computed offline | 48× smaller I/O than loading JPEGs; removes VAE from training GPU |

---

## Project Structure

```
project/
├── models/
│   ├── qwen_visual_encoder.py      # Frozen Qwen3.5-4B visual backbone (196×2560)
│   ├── image_cross_attention.py    # GatedImageCrossAttention + UNetImageConditioner
│   │                               #   - hooks on every BasicTransformerBlock
│   │                               #   - gate initialised to 0 → pure SD at step 0
│   ├── adapter.py                  # ImageAdapter: PatchProjector + ResidualPooler
│   │                               #   - null_image_tokens for image-CFG training
│   ├── unet.py                     # LDMUNet (SD 2.1): frozen CLIP text encoder,
│   │                               #   empty_text_emb buffer, conditioner injection
│   └── vae.py                      # VAE wrapper (frozen; decode only during training)
│
├── diffusion/
│   ├── scheduler.py                # DDPMScheduler from SD 2.1 pretrained
│   └── sampler.py                  # DDIMSampler for inference
│
├── training/
│   ├── train.py                    # Multi-GPU DDP (torchrun)
│   ├── train_compile.py            # DDP + torch.compile (~15–30% faster on H100)
│   ├── validate.py                 # Fast val (diffusion loss) + full val (DDIM+LPIPS)
│   ├── loss.py                     # DiffusionLoss (MSE only; semantic loss removed)
│   └── cfg.py                      # apply_condition_dropout + apply_image_dropout
│
├── data/
│   ├── dataset.py                  # MultiDatasetEmbeddingDataset + samplers
│   │                               #   - returns z_vae from .pt if present (fast path)
│   │                               #   - falls back to image loading if z_vae absent
│   ├── stats.py                    # Statistics collection and JSON persistence
│   └── preprocessors/
│       ├── __init__.py             # PREPROCESSORS registry
│       ├── base.py                 # BaseDatasetPreprocessor (streaming iter_samples)
│       ├── robobrain_dex.py        # RoboBrain-Dex layout
│       └── lerobot.py              # LeRobot layout (with + without text)
│
├── configs/
│   └── base.yaml                   # All hyperparameters
│
├── evaluation/
│   ├── metrics.py                  # PSNR, LPIPSMetric, MetricAccumulator
│   └── visualize.py                # save_comparison_grid (GT | reconstruction)
│
├── scripts/
│   ├── extract_qwen_visual_encoder.py  # Step 0: save standalone visual backbone .pt
│   ├── preprocess_embeddings.py        # Step 1: offline Qwen embedding extraction
│   ├── preprocess_vae_latents.py       # Step 2: add z_vae to existing .pt files
│   ├── eval_vae.py                     # Evaluate VAE reconstruction quality
│   └── preprocess_config.yaml         # Multi-dataset preprocessing config
│
└── inference/
    └── sample.py                   # Reconstruct image from .pt embedding
```

---

## Environment Setup

```bash
conda create -n qwen_decoder python=3.10 -y
conda activate qwen_decoder

# PyTorch (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install "diffusers==0.30.3" accelerate "transformers>=4.57.0.dev0"
pip install pillow tqdm pyyaml lpips matplotlib qwen-vl-utils
pip install peft   # only needed for training_mode: "lora"
```

> `diffusers==0.30.3` pinned — newer versions use flash_attn_3 annotations incompatible with PyTorch < 2.4.
> `transformers>=4.57.0.dev0` required for `Qwen3_5ForConditionalGeneration`.

---

## Step 0 — Extract Qwen Visual Backbone (once)

```bash
python scripts/extract_qwen_visual_encoder.py \
    --model_name Qwen/Qwen3.5-4B \
    --output /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt
```

Saves: `state_dict`, `vision_config`, `hidden_size=2560`, `visual_cls_name` / `visual_cls_module`
(required for the merger to run correctly at load time).

---

## Step 1 — Extract Qwen Embeddings (multi-GPU, resumable)

```bash
# Single dataset
torchrun --nproc_per_node=8 scripts/preprocess_embeddings.py \
    --dataset robobrain-dex \
    --image_dir /share/project/hotel/.../robobrain-dex \
    --output_dir /share/project/congsheng/robobrain-dex-qwen-embedding \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \
    --batch_size 16

# Multiple datasets + cycle limit (recommended for millions of files)
torchrun --nproc_per_node=8 scripts/preprocess_embeddings.py \
    --config scripts/preprocess_config.yaml \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \
    --max_frames_per_cycle 50000
```

Output per frame: `{"z_img": (196, 2560), "_abs_image_path": str}` + `stats.json`.
Re-run is safe — existing `.pt` files are skipped via in-memory set.

---

## Step 2 — Add VAE Latents (multi-GPU, resumable)

Adds `z_vae: (4, 56, 56) float16` to each existing `.pt` file.
Run this once after Step 1 before training — it **removes the VAE and image I/O
entirely from the training loop** (48× smaller DataLoader reads).

```bash
torchrun --nproc_per_node=8 scripts/preprocess_vae_latents.py \
    --embedding_dir /share/project/congsheng/robobrain-dex-qwen-embedding/robobrain-dex \
    --unet_name Manojb/stable-diffusion-2-1-base \
    --resolution 448 \
    --batch_size 64
```

Re-run is safe — `.pt` files that already have `z_vae` are skipped.

After this step, set `image_dir: null` in `configs/base.yaml` — the dataset will
load only `z_img + z_vae` with no image I/O.

---

## Step 3 — Training

### Multi-GPU DDP (recommended)

```bash
torchrun --nproc_per_node=8 training/train.py --config configs/base.yaml
```

### Multi-GPU DDP + torch.compile (~15–30% faster on H100, non-LoRA/frozen modes)

```bash
torchrun --nproc_per_node=8 training/train_compile.py --config configs/base.yaml
```

### Resume from checkpoint

```bash
torchrun --nproc_per_node=8 training/train.py \
    --config configs/base.yaml \
    --resume /path/to/checkpoints/step_50000.pt
```

### Training modes

Set `training_mode` in `configs/base.yaml`:

| Mode | Trainable | VRAM/GPU | Checkpoint |
|------|-----------|----------|------------|
| `frozen` | ImageAdapter + GatedImageCrossAttention layers + gates (~50–200M) | ~20 GB | `img_adapter` + `conditioner` |
| `lora` | LoRA deltas + ImageAdapter (~50–100M) | ~30 GB | `lora_step_N/` + `img_adapter` |
| `full` | All UNet weights + ImageAdapter (~900M) | ~40–65 GB | full `unet` state dict |

**`frozen` is the recommended mode** — the gated image cross-attention starts at zero
(pure SD 2.1 at step 0) and gradually opens as the model learns the image signal,
preserving SD's generative prior throughout training.

### Key config parameters

```yaml
model:
  backbone: "sd21"
  unet_name: "Manojb/stable-diffusion-2-1-base"
  qwen_dim: 2560
  cross_attn_dim: 1024          # image cross-attn token dim
  num_img_tokens: 16
  num_heads: 8
  num_pooler_layers: 2          # ResidualPooler depth
  pooler_self_attn: false

training:
  training_mode: "frozen"
  batch_size_per_gpu: 96        # safe at 80 GB H100 with pre-computed latents
  image_dropout_prob: 0.15      # fraction replaced by null_image_tokens (image CFG)
  embedding_noise_std: 0.0      # Gaussian noise on z_img for robustness (~0.01–0.05)
  lambda_sem: 0.0               # semantic loss disabled (Qwen not loaded at train time)
  gradient_checkpointing: false # not needed in frozen mode; UNet is frozen

data:
  resolution: 448               # latent 56×56×4
  datasets:
    - name: "robobrain-dex"
      embedding_dir: "/share/project/congsheng/WMDEC_qwen/.../robobrain-dex"
      # image_dir: omit once preprocess_vae_latents.py has run
```

### Checkpoint format (frozen mode)

```
step_N.pt:
  step:         int
  img_adapter:  state_dict   ← ImageAdapter weights
  conditioner:  state_dict   ← GatedImageCrossAttention layers + gates
  optimizer:    state_dict
  lr_scheduler: state_dict
  scaler:       state_dict
```

### Efficiency summary

| Optimization | Saving |
|---|---|
| Pre-computed VAE latents | 48× smaller DataLoader I/O; removes VAE from training GPU |
| Frozen UNet (no backprop through 865M params) | 2–4× larger batch at same VRAM |
| No semantic loss | Removes Qwen4B (~4.5 GB) + VAE decode from training GPU |
| Gradient checkpointing OFF | Not needed with frozen UNet |
| torch.compile (non-frozen) | +15–30% throughput on H100 |

### Validation

```yaml
validation:
  val_every: 20000         # fast: diffusion loss + cosine similarity
  val_full_every: 50000    # full: DDIM generation + LPIPS + saves grid
  best_metric: "val/cosine_sim"
```

Best checkpoint saved to `{output_dir}/checkpoints/best.pt`.

### Multi-dataset rebalancing

```
p_i  =  n_i^α  /  Σ_j  n_j^α
```

α=0.7 recommended: moderate smoothing, prevents large datasets from fully
dominating. Probabilities derived from verified `stats.json` frame counts.

---

## Step 4 — Inference

```bash
python inference/sample.py \
    --config configs/base.yaml \
    --checkpoint /path/to/final.pt \
    --embedding /path/to/frame.pt \
    --output reconstructed.png \
    --cfg_scale 2.0 \
    --steps 50
```

Image CFG at inference:
```
ε_final = ε_uncond + scale × (ε_cond − ε_uncond)
```
where `ε_uncond` uses `null_image_tokens` (learned during training).

---

## Training Budget (64 × H100 80 GB)

Assumptions: `frozen` mode, SD 2.1, pre-computed latents, `batch_size_per_gpu=128`.

| | |
|---|---|
| Global batch size | 64 × 128 = **8,192** |
| Steps per epoch (32M frames) | 32M / 8192 ≈ 3,900 steps |
| Estimated step time | ~120–150 ms |
| Time per epoch | ~8–10 min |

| Steps | Effective passes | Wall time | Cost @ $3.50/GPU/hr |
|-------|-----------------|-----------|---------------------|
| 200K | 51× | ~8 h | ~$1,800 |
| **300K** | **77×** | **~12 h** | **~$2,700** ← recommended |
| 500K | 128× | ~20 h | ~$4,500 |

Start evaluating at step 50K–100K (`val/cosine_sim` should be rising).
Convergence typically plateaus at 200K–300K steps for this data scale.

---

## Evaluation

```bash
# VAE reconstruction quality (baseline ceiling)
torchrun --nproc_per_node=4 scripts/eval_vae.py \
    --config configs/base.yaml \
    --num_samples 1000 \
    --vis_samples 16 \
    --vis_output vae_reconstruction.png
```

| Metric | Description | Direction |
|--------|-------------|-----------|
| Cosine similarity (Qwen) | `cosine(mean_pool(Qwen(x̂)), mean_pool(z_img))` | ↑ higher |
| LPIPS (VGG) | Perceptual distance from GT | ↓ lower |
| PSNR | Pixel-level reconstruction | ↑ higher |

---

## Adding a New Dataset

1. **Implement preprocessor** (`data/preprocessors/my_dataset.py`):

```python
from data.preprocessors.base import BaseDatasetPreprocessor, SampleMeta
from PIL import Image

class MyDatasetPreprocessor(BaseDatasetPreprocessor):
    @property
    def dataset_name(self) -> str:
        return "my-dataset"

    def iter_samples(self):
        for img_path in sorted(self.image_root.rglob("*.jpg")):
            yield SampleMeta(
                rel_path=img_path.relative_to(self.image_root),
                extra_meta={"_abs_image_path": str(img_path)},
            )

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")
```

2. **Register** in `data/preprocessors/__init__.py`:

```python
PREPROCESSORS = {
    "robobrain-dex": RoboBrainDexPreprocessor,
    "my-dataset":    MyDatasetPreprocessor,
}
```

3. **Add to both configs**:

```yaml
# scripts/preprocess_config.yaml
datasets:
  - name: my-dataset
    image_dir: /path/to/images

# configs/base.yaml
data:
  datasets:
    - name: my-dataset
      embedding_dir: /path/to/my-dataset-qwen-embedding/my-dataset
```

Always include `_abs_image_path` in `extra_meta` — it is stored in the `.pt`
and used by `preprocess_vae_latents.py` to find the image for VAE encoding.

---

## Development Roadmap

1. `scripts/extract_qwen_visual_encoder.py` — extract Qwen visual tower
2. `scripts/preprocess_embeddings.py` — offline Qwen embedding extraction
3. `scripts/preprocess_vae_latents.py` — add VAE latents to .pt files
4. `training/train.py` — DDP training with frozen SD 2.1 + gated image cross-attn
5. Monitor `val/cosine_sim` and visualization grids; save best checkpoint
6. `inference/sample.py` — image-CFG inference with null tokens
