"""
Extract the visual encoder (ViT backbone) from a full Qwen2.5-VL model and save it
as a lightweight standalone checkpoint (~600 MB for 3B model visual tower).

This only needs to be run ONCE on any machine that has the full model downloaded.
The resulting checkpoint is then used for offline preprocessing.

Usage:
    python scripts/extract_qwen_visual_encoder.py \
        --model_name Qwen/Qwen2.5-VL-3B-Instruct \
        --output /share/project/congsheng/checkpoints/qwen_visual_encoder_3b.pt
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def extract(model_name: str, output_path: str):
    print(f"Loading {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    visual = model.visual
    vision_config = model.config.vision_config

    # Resolve hidden size
    hidden_size = getattr(vision_config, "hidden_size", 1152)
    print(f"Visual encoder hidden_size: {hidden_size}")
    print(f"Visual encoder params: {sum(p.numel() for p in visual.parameters()) / 1e6:.1f}M")

    payload = {
        "state_dict": visual.state_dict(),
        "vision_config": vision_config.to_dict(),
        "hidden_size": hidden_size,
        "processor_name": model_name,   # used by QwenVisualEncoder.from_standalone
    }

    torch.save(payload, output_path)
    print(f"Saved standalone visual encoder to {output_path}")
    print("You can now delete the full model weights to free disk space.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HuggingFace model ID of the full Qwen2.5-VL model",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the standalone .pt checkpoint",
    )
    args = parser.parse_args()
    extract(args.model_name, args.output)
