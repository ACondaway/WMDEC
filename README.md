# SD3.5 Medium + Qwen Semantic Condition Generation System

Semantic-conditioned image generation using a frozen **Stable Diffusion 3.5 Medium** backbone
conditioned on **Qwen3.5-4B** visual embeddings. The system replaces traditional text conditioning
with visual semantic conditioning while preserving SD3.5's native generation quality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Encoding  (frozen, offline)                                                 │
│                                                                              │
│  image ──► Qwen3.5-4B Visual Encoder ──► (64, 2560) patch features          │
│            ViT hidden=1024, out_hidden=2560, merge=2×2 (256×256 input)       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  stored as .pt  (bfloat16, offline)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Semantic Condition Branch  (trainable — Phase 1)                            │
│                                                                              │
│  (B, 64, 2560)                                                               │
│       │                                                                      │
│       ▼  PerceiverResampler  (cross-attn → self-attn → FFN, 4 layers)        │
│       │  learned queries (32) attend over 64 patch tokens                    │
│       ▼  (B, 32, 1024)  semantic latent tokens                               │
│       │                                                                      │
│       ▼  ResidualMLPAdaptor  (6 residual blocks, LayerNorm + GELU)           │
│       ├──► pseudo_text_tokens  (B, 32, 4096)  → encoder_hidden_states        │
│       └──► pooled_proj         (B, 2048)      → pooled_projections           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Texture Control Branch  (trainable — Phase 2 only)                          │
│                                                                              │
│  image ──► SD3.5 VAE Encoder ──► latent prior  (B, 16, H/8, W/8)            │
│                │                                                             │
│                ▼  LightControlEncoder  (4-stage CNN + global avg pool)       │
│                ▼  (B, 512)  control features                                 │
│                ▼  ctrl_to_pooled  ──► (B, 2048)  control residual            │
│                                                                              │
│  pooled_proj_conditioned = pooled_proj + strength(t) × ctrl_residual         │
│  strength(t) = t   (strong at noisy early timesteps, weak at late)           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│  SD3.5 Medium MMDiT  (frozen)                                                │
│                                                                              │
│  noisy_latent + timestep                                                     │
│       │  encoder_hidden_states = pseudo_text_tokens  (B, 32, 4096)           │
│       │  pooled_projections    = pooled_proj_cond    (B, 2048)               │
│       ▼  Joint Attention (24 blocks, hidden=1536)  ──► velocity pred         │
│                                                                              │
│  flow matching:  x_t = (1−t)·x₀ + t·ε                                       │
│  loss:           L = MSE(v_pred, ε − x₀)                                     │
│                                                                              │
│  VAE Decoder (frozen)  ──► reconstructed image  (B, 3, H, W)                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Visual encoder | Qwen3.5-4B (frozen) | 64 spatial tokens × 2560 dim after 2×2 spatial merge |
| Token compression | PerceiverResampler (32 queries) | Reduces attention complexity; 64→32 semantic tokens |
| Manifold alignment | ResidualMLPAdaptor (6 blocks) | Bridges Qwen semantic space → SD3.5 text manifold (4096-dim) |
| Texture control | Lightweight CNN + pooled_projections augmentation | Clean injection into frozen MMDiT without hooks |
| Diffusion backbone | SD3.5 Medium MMDiT (frozen) | 2B param pretrained prior; MMDiT joint attention |
| Noise schedule | Flow matching (FlowMatchEulerDiscreteScheduler) | SD3.5 native; velocity targets |
| VAE | SD3.5 VAE (frozen, 16-ch latents) | Preserves texture quality; not trained here |

---

## Project Structure

```
project/
├── models/
│   ├── qwen_visual_encoder.py    # Frozen Qwen3.5-4B visual backbone
│   │                             #   encode_images(): (B, 64, 2560) contract
│   ├── perceiver_resampler.py    # Light Perceiver: (B,64,2560)→(B,32,1024)
│   ├── residual_mlp_adaptor.py   # Residual MLP: (B,32,1024)→(B,32,4096)+pooled
│   ├── texture_control.py        # CNN control encoder + AdaLN residual injection
│   ├── sd35_model.py             # SD35SemanticModel: full pipeline + training forward
│   ├── adapter.py                # Legacy SDXL ImageAdapter (CrossAttentionPooler)
│   ├── unet.py                   # Legacy SDXL UNet wrapper
│   └── vae.py                    # Legacy SDXL VAE wrapper
│
├── training/
│   ├── train_sd35_phase1.py      # Phase 1: train Perceiver + Adaptor (torchrun)
│   ├── train_sd35_phase2.py      # Phase 2: train TextureControl + continue Phase 1
│   ├── train.py                  # Legacy SDXL DDP training
│   ├── train_single.py           # Legacy SDXL single-process training
│   ├── validate.py               # Fast + full validation helpers
│   ├── loss.py                   # Diffusion + semantic consistency losses
│   └── cfg.py                    # CFG dropout utilities
│
├── data/
│   ├── dataset_sd35.py           # SD35EmbeddingDataset (resolution=768)
│   ├── dataset.py                # Legacy MultiDatasetEmbeddingDataset + samplers
│   ├── stats.py                  # Statistics collection and persistence
│   └── preprocessors/
│       ├── __init__.py           # PREPROCESSORS registry
│       ├── base.py               # BaseDatasetPreprocessor (abstract API)
│       ├── robobrain_dex.py      # RoboBrain-Dex layout
│       └── lerobot.py            # LeRobot layout
│
├── configs/
│   ├── sd35_phase1.yaml          # Phase 1 training config
│   ├── sd35_phase2.yaml          # Phase 2 training config
│   └── base.yaml                 # Legacy SDXL config
│
├── inference/
│   ├── sample_sd35.py            # SD3.5 inference: image → reconstructed image
│   └── sample.py                 # Legacy SDXL inference
│
├── diffusion/
│   ├── scheduler.py              # Legacy DDPM scheduler wrapper
│   └── sampler.py                # Legacy DDIM sampler
│
└── scripts/
    ├── extract_qwen_visual_encoder.py   # Step 0: save standalone Qwen visual backbone
    ├── preprocess_single_node.py        # Step 1a: single-node multi-GPU preprocessing
    │                                    #   (persistent GPU workers — encoder loaded once)
    ├── preprocess_embeddings.py         # Step 1b: multi-node preprocessing (torchrun/NCCL)
    └── preprocess_config.yaml           # Multi-dataset preprocessing config
```

---

## Environment Setup

```bash
conda create -n sd35_qwen python=3.10 -y
conda activate sd35_qwen

# PyTorch (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install "diffusers>=0.31.0"
pip install accelerate "transformers>=4.57.0.dev0"
pip install pillow tqdm pyyaml lpips matplotlib qwen-vl-utils
```

> **Version notes:**
> - `transformers>=4.57.0.dev0` — required for `Qwen3_5ForConditionalGeneration`
> - `diffusers>=0.31.0` — required for `SD3Transformer2DModel` and `FlowMatchEulerDiscreteScheduler`

---

## Step 0 — Extract the Qwen Visual Backbone (once)

Extract the ViT tower from the full Qwen3.5-4B model and save it as a standalone `.pt`.
Run once; the full model can be deleted after extraction.

```bash
python scripts/extract_qwen_visual_encoder.py \
    --model_name /path/to/Qwen3.5-4B \
    --output /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt
```

The checkpoint stores `state_dict`, `vision_config`, `hidden_size=2560`,
`visual_cls_name` / `visual_cls_module` for exact class reconstruction.

> **Important:** Re-extract if upgrading `transformers` — the merger projection
> class name is saved and must match the installed version.

---

## Step 1 — Preprocess Datasets (Offline, Multi-GPU)

Encodes all images through the frozen Qwen visual encoder and stores `(64, 2560)`
patch embeddings on disk. Each `.pt` file stores `_abs_image_path` for GT image loading.

### Single node (no torchrun / no NCCL required)

```bash
python scripts/preprocess_single_node.py \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \
    --dataset robobrain-dex \
    --image_dir /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex \
    --output_dir /share/project/congsheng/robobrain-dex-qwen-embedding \
    --batch_size 256 \
    --num_gpus 8
```

GPU workers are **persistent** — the encoder is loaded once per GPU at startup and
reused across all chunks. This avoids repeated model loading overhead between chunks.

### Multi-node (torchrun / NCCL)

```bash
# Run on each node:
torchrun --nnodes=4 --nproc_per_node=8 \
    --node_rank=<RANK> --master_addr=<MASTER_IP> --master_port=29500 \
    scripts/preprocess_embeddings.py \
    --config scripts/preprocess_config.yaml \
    --encoder_ckpt /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt \
    --batch_size 256 --num_io_workers 48
```

### Multi-dataset config

`scripts/preprocess_config.yaml`:
```yaml
output_root: /share/project/congsheng/robobrain-dex-qwen-embedding

datasets:
  - name: robobrain-dex
    image_dir: /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex
```

### Output layout

```
{output_root}/{dataset_name}/
    stats.json                   ← frame counts, per-task breakdown, timing
    {task}/{episode}/{frame}.pt  ← {"z_img": (64, 2560), "_abs_image_path": str}
```

Processing is **resumable** — existing `.pt` files are detected via a single `os.walk`
and skipped automatically on re-run.

---

## Step 2 — Phase 1 Training: Semantic Alignment

Trains only the **PerceiverResampler** and **ResidualMLPAdaptor**.
Everything else (SD3.5 MMDiT, VAE, Qwen) remains frozen.

### Classifier-free guidance dropout (training)

| Condition | Probability |
|-----------|-------------|
| Keep all  | 75% |
| Drop image | 10% → zero Qwen features |
| Drop text  | 10% → no-op in Phase 1 |
| Drop all   | 5%  → zero Qwen features |

### Launch

```bash
# Single node, 8×H100
torchrun --nproc_per_node=8 training/train_sd35_phase1.py \
    --config configs/sd35_phase1.yaml

# Multi-node, 4 nodes × 8 GPUs
torchrun --nnodes=4 --nproc_per_node=8 \
    --node_rank=<RANK> --master_addr=<MASTER_IP> --master_port=29500 \
    training/train_sd35_phase1.py --config configs/sd35_phase1.yaml
```

### Resume

```bash
torchrun --nproc_per_node=8 training/train_sd35_phase1.py \
    --config configs/sd35_phase1.yaml \
    --resume outputs/sd35_phase1/phase1_latest.pt
```

### Key config knobs (`configs/sd35_phase1.yaml`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `resolution` | 768 | Output image resolution |
| `per_gpu_batch` | 8 | × 8 GPUs = 64 global batch |
| `total_steps` | 100 000 | Minimal viable experiment |
| `lr` | 1e-4 | With cosine decay + warmup |
| `sem_loss_weight` | 0.0 | Set > 0 to enable `L_align = 1 − cos(z_pred, z_orig)` |
| `resampler.num_queries` | 32 | Semantic token count |
| `resampler.num_layers` | 4 | Perceiver depth |
| `adaptor.num_blocks` | 6 | Residual MLP depth |

### Loss

```
L_diff  =  MSE(v_pred, v_target)        flow-matching velocity loss
L_align =  1 − cos(z_qwen_pred, z_img)  semantic consistency (optional, λ=0–0.5)
L       =  L_diff + λ · L_align
```

### Outputs

```
outputs/sd35_phase1/
├── phase1_step0001000.pt     ← periodic checkpoint
├── phase1_step0002000.pt
└── phase1_latest.pt          ← symlink to latest
```

Checkpoint contains: `resampler`, `adaptor`, `optimizer`, `scaler`, `step`.

---

## Step 3 — Phase 2 Training: Texture Control

Trains the **TextureControlBranch** while continuing to fine-tune
the semantic branch from Phase 1. Initialises from a Phase 1 checkpoint.

### Stochastic control dropout

Randomly zeroes the control features with `drop_prob=0.5` during training to
prevent over-reliance on the latent prior and preserve semantic branch effectiveness.

### Timestep-aware control strength

| Diffusion Stage | `t` value | Control Strength |
|---|---|---|
| Early (noisy) | `t → 1` | Strong — controls geometry/layout |
| Late (clean) | `t → 0` | Weak — generative flexibility preserved |

`strength(t) = t` (linear; no learned parameters).

### Launch

```bash
torchrun --nproc_per_node=8 training/train_sd35_phase2.py \
    --config configs/sd35_phase2.yaml \
    --phase1_ckpt outputs/sd35_phase1/phase1_latest.pt
```

### Resume Phase 2

```bash
torchrun --nproc_per_node=8 training/train_sd35_phase2.py \
    --config configs/sd35_phase2.yaml \
    --phase1_ckpt outputs/sd35_phase1/phase1_latest.pt \
    --resume outputs/sd35_phase2/phase2_latest.pt
```

### Key config knobs (`configs/sd35_phase2.yaml`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | 5e-5 | Lower than Phase 1 (fine-tuning regime) |
| `total_steps` | 200 000 | Extended training |
| `sem_loss_weight` | 0.1 | Semantic loss active in Phase 2 |
| `control.ctrl_dim` | 512 | CNN control feature dimension |
| `control.drop_prob` | 0.5 | Stochastic control dropout |

### Outputs

```
outputs/sd35_phase2/
├── phase2_step0001000.pt     ← resampler + adaptor + control + optimizer
└── phase2_latest.pt
```

---

## Step 4 — Inference

### Phase 1 (semantic conditioning only)

```bash
python inference/sample_sd35.py \
    --config     configs/sd35_phase1.yaml \
    --phase1_ckpt outputs/sd35_phase1/phase1_latest.pt \
    --input_dir  /path/to/input/images \
    --output_dir /path/to/outputs \
    --steps 28 --guidance 5.0
```

### Phase 2 (semantic + texture control)

```bash
python inference/sample_sd35.py \
    --config      configs/sd35_phase2.yaml \
    --phase1_ckpt outputs/sd35_phase2/phase2_latest.pt \
    --use_control \
    --input_dir   /path/to/input/images \
    --output_dir  /path/to/outputs \
    --steps 28 --guidance 5.0
```

Output: side-by-side comparison `input | reconstruction` saved per image.

### Inference parameters

| Argument | Default | Notes |
|----------|---------|-------|
| `--steps` | 28 | Denoising steps (FlowMatch Euler) |
| `--guidance` | 5.0 | CFG scale |
| `--resolution` | 768 | Output height × width |
| `--use_control` | off | Enable Phase 2 texture control branch |
| `--seed` | 42 | RNG seed for reproducibility |

---

## Evaluation

| Metric | Description | Direction |
|--------|-------------|-----------|
| Cosine similarity (Qwen) | `cos(mean_pool(Qwen(x̂)), mean_pool(z_img))` | ↑ higher |
| LPIPS | Perceptual similarity (VGG) | ↓ lower |

---

## Development Roadmap

1. Extract Qwen visual backbone → `scripts/extract_qwen_visual_encoder.py`
2. Preprocess datasets offline → `scripts/preprocess_single_node.py` or `preprocess_embeddings.py`
3. Phase 1 — semantic alignment → `training/train_sd35_phase1.py`
4. Evaluate Phase 1 (cosine similarity, sample quality)
5. Phase 2 — texture control → `training/train_sd35_phase2.py`
6. Evaluate Phase 2 (LPIPS, texture fidelity)
7. Scale to additional datasets via `BaseDatasetPreprocessor`

---

## Adding a New Dataset

1. **Implement a preprocessor** (`data/preprocessors/my_dataset.py`):

```python
from data.preprocessors.base import BaseDatasetPreprocessor, SampleMeta
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
                extra_meta={"_abs_image_path": str(img_path)},
            ))
        return samples

    def load_image(self, sample: SampleMeta) -> Image.Image:
        return Image.open(sample.extra_meta["_abs_image_path"]).convert("RGB")
```

2. **Register it** in `data/preprocessors/__init__.py`:

```python
PREPROCESSORS = {
    "robobrain-dex": RoboBrainDexPreprocessor,
    "my-dataset":    MyDatasetPreprocessor,
}
```

3. **Add to configs**:

```yaml
# scripts/preprocess_config.yaml
datasets:
  - name: my-dataset
    image_dir: /path/to/my/images

# configs/sd35_phase1.yaml
datasets:
  - name: my-dataset
    embedding_dir: /path/to/my-dataset-qwen-embedding
    image_dir: /path/to/my/images
```

---

## Legacy: SDXL Decoder

The original SDXL-based decoder (Qwen → ImageAdapter → SDXL UNet fine-tune) remains in the
codebase for reference. See `models/adapter.py`, `training/train.py`, `configs/base.yaml`.
