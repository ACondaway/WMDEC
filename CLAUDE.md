# SigLIP Embedding Decoder Reconstruction

## 1. Objective

This project aims to train a **decoder model** that reconstructs images from frozen SigLIP embeddings. Use conda to manage the developing environment. You should know that the code is now under the local environment, so you don't need to run virtually, for example downloading checkpoint and configurate environment. Just write a detailed implementation README.md to show me how to do it and I will manually run on a remote cluster.

### Inputs

* Image SigLIP embedding `z_img`
* Optional: text description `text`, if not used just padding.

### Output

* Reconstructed image `x_hat`

---

## 2. Overall Architecture

We use a **Latent Diffusion Decoder** (similar to Stable Diffusion):

```id="arch001"
image → SigLIP (frozen) → z_img
text → text encoder → z_txt

z_img → MLP → tokens_img
z_txt → projection → tokens_txt

tokens_img + tokens_txt → cross-attention → UNet

noise latent → UNet → denoised latent → VAE decoder → image
```

---

## 3. Modules

### 3.1 SigLIP Encoder (Frozen)

```python id="sig001"
z_img = siglip.encode_image(image)  # shape: (B, D)
z_img = normalize(z_img)
```

---

### 3.2 Text Encoder

Model

* T5-XXL

```python id="txt001"
z_txt = text_encoder(text)  # (B, T, C)
```

---

### 3.3 Adapter (Critical)

#### Image Adapter

```python id="adp001"
tokens_img = MLP(z_img)  # (B, N_img, C)
```

Recommended:

* N_img = 4–8 tokens
* 2–4 layer MLP
* LayerNorm

---

#### Text Adapter

```python id="adp002"
tokens_txt = Linear(z_txt)
```

---

### 3.4 Conditioning (Must use Cross-Attention)

Inside UNet:

```python id="attn001"
Q = latent features
K, V = concat(tokens_img, tokens_txt)
```

---

## 4. Diffusion Model

### Inputs

* Noisy latent `x_t`
* Timestep `t`
* Conditioning tokens

### Output

* Predicted noise `ε_θ`

---

### Diffusion Loss

```python id="loss001"
L_diff = MSE(ε_pred, ε)
```

---

### Semantic Consistency Loss (Strongly Recommended)

```python id="loss002"
L_sem = 1 - cosine(siglip(x_pred), z_img)
```

---

### Total Loss

```python id="loss003"
L = L_diff + λ * L_sem
```

Recommended:

```id="cfg001"
λ = 0.1 – 0.5
```

---

## 5. Classifier-Free Guidance (CFG)

### Training

Randomly drop conditions:

| Condition  | Probability |
| ---------- | ----------- |
| keep all   | 75%         |
| drop text  | 10%         |
| drop image | 10%         |
| drop all   | 5%          |

---

### Inference

```python id="cfg002"
eps = eps_uncond + s * (eps_cond - eps_uncond)
```

Recommended:

```id="cfg003"
s_img = 1.5 – 3.0
s_txt = 0.5 – 2.0
```

---

## 6. Dataset

The data are structured as the image directory and the correspondent `{Task name}`.
`/share/project/hotel/lerobot30_multiimage_data_1fps/robobrain-dex/{Task_name}/videos/chunk-000/observation.images.image_top/episode_000000/image_x.0.jpg`

First, use SigLIP and T5-XXL to process these data offline and store them together in 
`/share/project/congsheng/robobrain-dex-siglip-embedding`



Usage:

```python id="data001"
z_img = SigLIP(image)
z_txt = TextEncoder(text)
```

SigLIP and the text encoder are always frozen.

---

## 7. Training Setup (4×H100)

### Minimal Viable Experiment

```id="train001"
resolution: 256
latent size: 32×32

per GPU batch: 8
global batch: 32

steps: 100k – 200k
time: 6 – 12 hours
```

---

### Extended Training

```id="train002"
resolution: 512
global batch: 64

steps: 300k – 500k
time: 1 – 3 days
```

---

## 8. Training Loop

```python id="trainloop001"
for batch in dataloader:

    image, text = batch

    # encode
    z_img = siglip(image)
    z_txt = text_encoder(text)

    # condition dropout
    z_img, z_txt = random_drop(z_img, z_txt)

    # diffusion forward
    noise = torch.randn_like(x)
    x_t = q_sample(x, t, noise)

    # predict noise
    eps_pred = unet(x_t, t, cond=(z_img, z_txt))

    # diffusion loss
    loss = mse(eps_pred, noise)

    # semantic loss
    x_pred = reconstruct(eps_pred)
    loss += λ * (1 - cosine(siglip(x_pred), z_img))

    loss.backward()
```

---

## 9. Project Structure

```id="struct001"
project/
├── models/
│   ├── siglip_encoder.py
│   ├── text_encoder.py
│   ├── adapter.py
│   ├── unet.py
│   ├── vae.py
│
├── diffusion/
│   ├── scheduler.py
│   ├── sampler.py
│
├── training/
│   ├── train.py
│   ├── loss.py
│   ├── cfg.py
│
├── data/
│   ├── dataset.py
│
├── configs/
│   ├── base.yaml
│
└── inference/
    ├── sample.py
```

---

## 10. Key Design Decisions

* SigLIP is **always frozen**
* Use **cross-attention conditioning**
* Use **dual conditioning (image + text)**
* CFG improves condition fidelity
* Semantic loss prevents drift

---

## 11. Evaluation

Focus on:

* Semantic consistency with input image
* Reduction of category mismatch
* Sensitivity to text conditioning

Metrics:

* Cosine similarity (SigLIP space)
* LPIPS (perceptual similarity)

---

## 12. Development Roadmap

1. Extract SigLIP embeddings (offline)
2. Train diffusion with image condition only
3. Add text conditioning
4. Add CFG
5. Add sementic loss