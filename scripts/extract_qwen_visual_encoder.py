"""
Extract the visual encoder (ViT backbone) from the full Qwen3.5-VL-4B model and save it
as a lightweight standalone checkpoint (~900 MB).

Run ONCE. The resulting checkpoint is used by preprocess_embeddings.py and
QwenVisualEncoder.from_standalone().

Model facts (from config.json):
  model_type           : qwen3_5
  ViT hidden_size      : 1024   (internal transformer hidden dim)
  out_hidden_size      : 2560   (projected output = text hidden_size)
  depth                : 24
  patch_size           : 16
  spatial_merge_size   : 2×2  →  196 tokens per 448×448 image

Usage:
    python scripts/extract_qwen_visual_encoder.py \\
        --model_name Qwen/Qwen3.5-4B \\
        --output /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b.pt
"""

import argparse
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor


def extract(model_name: str, output_path: str):
    print(f"Loading {model_name} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # The outer Qwen3_5ForConditionalGeneration wraps an inner `model` submodule.
    # Weight keys are prefixed "model.visual.*", so the attribute path is model.model.visual.
    inner = model.model
    visual = inner.visual
    vision_config = model.config.vision_config

    # Qwen3.5-VL projects ViT features (hidden_size=1024) to out_hidden_size=2560.
    # We store out_hidden_size as "hidden_size" since that's the embedding dimension
    # consumers (adapter, dataset) will see.
    out_hidden_size = getattr(vision_config, "out_hidden_size", 2560)
    print(f"ViT internal hidden_size : {getattr(vision_config, 'hidden_size', 1024)}")
    print(f"Projected out_hidden_size: {out_hidden_size}  (stored as hidden_size)")
    print(f"Visual encoder params    : {sum(p.numel() for p in visual.parameters()) / 1e6:.1f}M")

    torch.save({
        "state_dict": visual.state_dict(),
        "vision_config": vision_config.to_dict(),
        "hidden_size": out_hidden_size,   # 2560 — dimension of patch embeddings on disk
        "processor_name": model_name,
    }, output_path)

    print(f"Saved standalone visual encoder → {output_path}")
    print("The full model weights can now be deleted to free disk space.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3.5-4B",
        help="HuggingFace model ID of the full Qwen3.5-VL model",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the standalone .pt checkpoint",
    )
    args = parser.parse_args()
    extract(args.model_name, args.output)
