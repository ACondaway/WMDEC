# SigLIP Embedding Decoder Reconstruction

Reconstructs images from frozen SigLIP embeddings using a latent diffusion decoder with cross-attention conditioning.

## Architecture

```
image -> SigLIP (frozen) -> z_img -> MLP Adapter -> tokens_img \
                                                                 -> cross-attention -> UNet -> denoised latent -> VAE decoder -> image
text  -> T5-XXL (frozen) -> z_txt -> Linear Proj  -> tokens_txt /

noise latent + timestep -> UNet
```

---

## 1. Environment Setup

```bash
# Create conda environment
conda create -n siglip_decoder python=3.10 -y
conda activate siglip_decoder

# Install PyTorch (CUDA 12.1)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install dependencies
pip install diffusers accelerate sentencepiece protobuf transformers==4.44.2
pip install pillow tqdm pyyaml lpips matplotlib
```

---

## 2. Project Structure

```
project/
├── models/
│   ├── siglip_encoder.py    # Frozen SigLIP image encoder
│   ├── text_encoder.py      # Frozen T5-XXL text encoder
│   ├── adapter.py           # Image MLP adapter + Text linear projection
│   ├── unet.py              # Conditional UNet with cross-attention
│   └── vae.py               # Pretrained VAE wrapper
├── diffusion/
│   ├── scheduler.py         # DDPM noise scheduler
│   └── sampler.py           # DDIM sampler for inference
├── training/
│   ├── train.py             # Main distributed training script
│   ├── loss.py              # Diffusion + semantic consistency loss
│   └── cfg.py               # Classifier-free guidance dropout
├── data/
│   └── dataset.py           # Dataset for pre-extracted embeddings
├── configs/
│   └── base.yaml            # Training configuration
├── inference/
│   └── sample.py            # Image reconstruction from embeddings
└── scripts/
    └── preprocess_embeddings.py  # Offline embedding extraction
```

---

## 3. Step 1: Extract Embeddings (Offline)

Extract SigLIP and T5-XXL embeddings from the raw dataset before training.

**Input:** Images at `/share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex/{Task_name}/videos/chunk-000/observation.images.image_top/episode_XXXXXX/image_X.0.jpg`

**Output:** Embeddings at `/share/project/congsheng/robobrain-dex-siglip-embedding/{Task_name}/episode_XXXXXX/image_X.0.pt`

```bash
torchrun --nproc_per_node=4 scripts/preprocess_embeddings.py \
        --image_dir /share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex \
        --output_dir /share/project/congsheng/robobrain-dex-siglip-embedding \
        --batch_size 64
```

Each `.pt` file contains:
- `z_img`: SigLIP image embedding `(D,)` — normalized
- `z_txt`: T5-XXL text embedding `(T, C)` — derived from task name
- `task_name`: string
- `text`: task name with underscores replaced by spaces

**Note:** This requires ~40GB GPU memory for T5-XXL. Use a single GPU with large VRAM, or reduce batch size.

---

## 4. Step 2: Training

### Minimal Viable Experiment (4x H100)

```bash
torchrun --nproc_per_node=4 training/train.py --config configs/base.yaml
```

Default config (`configs/base.yaml`):

| Parameter | Value |
|-----------|-------|
| Resolution | 256 |
| Latent size | 32x32 |
| Per-GPU batch | 8 |
| Global batch | 32 |
| Max steps | 200k |
| Learning rate | 1e-4 |
| Lambda (semantic) | 0.1 |

**Expected time:** 6-12 hours on 4x H100.

### Extended Training (512 resolution)

Edit `configs/base.yaml`:

```yaml
data:
  resolution: 512

training:
  batch_size_per_gpu: 4   # reduce for 512
  max_steps: 500000
```

```bash
torchrun --nproc_per_node=4 training/train.py --config configs/base.yaml
```

**Expected time:** 1-3 days on 4x H100.

### Key Training Details

**Trainable modules:** Image Adapter, Text Adapter, UNet

**Frozen modules:** SigLIP encoder, T5-XXL, VAE

**Loss function:**
- `L = L_diff + lambda * L_sem`
- `L_diff = MSE(eps_pred, eps)` — standard diffusion loss
- `L_sem = 1 - cosine(SigLIP(x_pred), z_img)` — semantic consistency

**Classifier-Free Guidance (CFG) dropout during training:**

| Condition | Probability |
|-----------|-------------|
| Keep all  | 75% |
| Drop text | 10% |
| Drop image | 10% |
| Drop all  | 5% |

**Checkpoints** saved every 10k steps to `outputs/checkpoints/`.

---

## 5. Step 3: Inference

Reconstruct an image from a source image's SigLIP embedding:

```bash
python inference/sample.py \
    --config configs/base.yaml \
    --checkpoint outputs/checkpoints/final.pt \
    --image /path/to/input_image.jpg \
    --text "optional text description" \
    --output reconstructed.png \
    --cfg_scale 2.0 \
    --steps 50
```

**Arguments:**
- `--image`: Source image (will be encoded by SigLIP)
- `--text`: Optional text description (empty string = padding)
- `--cfg_scale`: CFG scale (recommended 1.5-3.0 for image, 0.5-2.0 for text)
- `--steps`: DDIM sampling steps (50 is a good default)

---

## 6. Evaluation

**Metrics to track:**

1. **Cosine similarity (SigLIP space):** Measures semantic consistency between input and reconstructed images
2. **LPIPS:** Perceptual similarity score

**What to evaluate:**
- Semantic consistency with input image
- Reduction of category mismatch
- Sensitivity to text conditioning

---

## 7. Configuration Reference

All settings in `configs/base.yaml`:

```yaml
model:
  siglip_model: "google/siglip-large-patch16-384"
  siglip_dim: 1152          # SigLIP embedding dimension
  t5_dim: 4096              # T5-XXL embedding dimension
  cross_attn_dim: 1024      # UNet cross-attention dimension
  num_img_tokens: 8         # Number of image condition tokens
  adapter_layers: 3         # MLP adapter depth
  model_channels: 256       # Base UNet channels
  channel_mult: [1,2,4,4]   # Channel multipliers per level
  num_heads: 8              # Attention heads

diffusion:
  num_timesteps: 1000
  beta_start: 0.00085
  beta_end: 0.012
  schedule: "scaled_linear"

training:
  batch_size_per_gpu: 8
  max_steps: 200000
  lr: 1.0e-4
  weight_decay: 0.01
  grad_clip: 1.0
  lambda_sem: 0.1           # Semantic loss weight (0.1-0.5)
  sem_loss_every: 10        # Compute semantic loss every N steps
  log_every: 100
  save_every: 10000
  output_dir: "outputs"
```

---

## 8. Development Roadmap

1. **Extract SigLIP embeddings** (offline) — `scripts/preprocess_embeddings.py`
2. **Train diffusion with image condition only** — set text dropout to 100%
3. **Add text conditioning** — default config
4. **Add CFG** — already integrated in training
5. **Add semantic loss** — controlled by `lambda_sem`
