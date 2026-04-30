"""
Extract the visual encoder (ViT backbone) from the full Qwen3.5-VL-4B model and save it
as a HuggingFace-compatible checkpoint directory.

The output directory contains:
  config.json          ← vision model config (loadable by AutoModel.from_pretrained)
  model.safetensors    ← visual model weights
  preprocessor_config.json  ← processor config
  tokenizer* files     ← processor tokenizer files
  metadata.json        ← out_hidden_size and other facts about the stored features

Run ONCE. The resulting directory is used by preprocess_embeddings.py and
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
        --output /share/project/congsheng/checkpoints/qwen3_5_visual_encoder_4b
"""

import argparse
import json
import os
import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor


def extract(model_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {model_name} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # The outer Qwen3_5ForConditionalGeneration wraps an inner `model` submodule.
    # Weight keys are prefixed "model.visual.*", so the attribute path is model.model.visual.
    visual = model.model.visual
    vision_config = model.config.vision_config

    out_hidden_size = getattr(vision_config, "out_hidden_size", 2560)
    vit_hidden_size = getattr(vision_config, "hidden_size", 1024)
    print(f"ViT internal hidden_size : {vit_hidden_size}")
    print(f"Projected out_hidden_size: {out_hidden_size}  (stored as hidden_size)")
    print(f"Visual encoder params    : {sum(p.numel() for p in visual.parameters()) / 1e6:.1f}M")

    # Save as a proper HuggingFace checkpoint so AutoModel.from_pretrained() can load it.
    print(f"Saving visual model to {output_dir} ...")
    visual.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save extra metadata that consumers need (out_hidden_size, tokens/image).
    metadata = {
        "out_hidden_size": out_hidden_size,
        "vit_hidden_size": vit_hidden_size,
        "tokens_per_image_448": 196,
        "source_model": model_name,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved standalone visual encoder → {output_dir}")
    print("Contents:", os.listdir(output_dir))
    print("The full model weights can now be deleted to free disk space.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3.5-4B",
        help="HuggingFace model ID of the full Qwen3.5-VL model",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for the HuggingFace-compatible visual encoder checkpoint",
    )
    args = parser.parse_args()
    extract(args.model_name, args.output)
