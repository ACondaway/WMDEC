import os
import sys
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models.siglip_encoder import SigLIPEncoder
from models.text_encoder import T5TextEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siglip_model", type=str,
                        default="google/siglip-large-patch16-384")
    parser.add_argument("--t5_model", type=str,
                        default="google/t5-xxl-lm-adapt")
    args = parser.parse_args()


    print("Loading SigLIP...")
    siglip = SigLIPEncoder(args.siglip_model)

    print("Loading T5-XXL...")
    text_encoder = T5TextEncoder(args.t5_model)




if __name__ == "__main__":
    main()