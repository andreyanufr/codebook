import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
from pathlib import Path
from all_values_tuning import dequantize_from_dict
from pack_unpack import unpack_2bit, unpack_4bit


def get_codebook_info(state_dict):
    codebook = state_dict["codebook"]
    scale = state_dict["scale"]
    shape = state_dict["shape"]
    indexes = state_dict["indexes"]
    
    codebook = codebook / codebook.abs().max()  # Normalize codebook to [-1, 1]

    if indexes.dtype == torch.uint8 and codebook.shape[0] == 4:
        indexes = unpack_2bit(indexes)
    elif indexes.dtype == torch.uint8 and codebook.shape[0] == 16:
        indexes = unpack_4bit(indexes)
    
    return codebook, scale, shape, indexes
    

def main(argv):
    codebooks_path = "/home/aanuf/proj/learnable_codebooks/3bit/qwen3_4B/full_lr_001_hard_all_clip_last/codebook_layers.pth"
    codebooks_path = "/home/aanuf/proj/learnable_codebooks/3bit/first_block_inputs.pt"
    codebooks = torch.load(codebooks_path, map_location="cpu") if codebooks_path and Path(codebooks_path).is_file() else {}
    
    for name, state_dict in codebooks.items():
        if not 'codebook' in name:
            continue
        codebook = state_dict
        print(f"Layer: {name}")
        print(f"Codebook: {codebook}")
        print("-" * 50)

if __name__ == "__main__":
    main(sys.argv[1:])
