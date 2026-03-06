import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
from pathlib import Path
from all_values_tuning import dequantize_from_dict
from pack_unpack import unpack_2bit, unpack_4bit


# model_id = "meta-llama/Llama-3.2-1B-Instruct"
# codebooks_path = "qwen3_8B/STE_LORA_64_300_samples_40_epochs_torch_compile_seqlen_1024_last/codebook_layers.pth"


# codebooks = torch.load(codebooks_path, map_location="cpu")


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=Path,
    #     default="output",
    #     help="Path to the directory for storing converted models.",
    # )
    parser.add_argument(
        "--codebooks_path",
        type=str,
        default=None,
        help="Path to the previously saved codebooks. If not specified or file does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    
    return parser


def weights_to_codebook(data, group_size):
    out_features, in_features = data.shape
    data = data.reshape(out_features, in_features // group_size, group_size)  # Flatten the weight matrix to 1D
    #scale = data.abs().max(dim=-1, keepdim=True)[0]
    scale = data.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5)
    
    normalized = data / scale  # Normalize to [-1, 1]
    unique_values, inverse_indices = torch.unique(normalized, return_inverse=True)

    return unique_values, inverse_indices.view(data.shape)

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
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    codebooks = torch.load(args.codebooks_path, map_location="cpu") if args.codebooks_path and Path(args.codebooks_path).is_file() else {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in codebooks:
            wcodebook = dequantize_from_dict(codebooks[name], device='cpu')
            wcur = module.weight.data.cpu()
            diff = (wcodebook - wcur).abs().mean().item()
            print(f"Layer: {name}, Mean Absolute Difference: {diff:.6f}")
            print()
            # codebook, scale, shape, indexes = get_codebook_info(codebooks[name])
            # codebook_w, indexes_w = weights_to_codebook(module.weight.data.to('cuda:1'), group_size=indexes.shape[-1])
            
            # print(f"Layer: {name}")
            # print(f"  Original codebook: {codebook.cpu()}")
            # #print(f"  Original scale: {scale.cpu().numpy()}")
            # #print(f"  Original shape: {shape}")
            # #print(f"  Original indexes shape: {indexes.shape}")
            # print(f"  New codebook: {codebook_w.cpu()}")
            # #print(f"  New indexes shape: {indexes_w.shape}")
            
            
    
    # Save the model with the applied codebooks
    # model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
