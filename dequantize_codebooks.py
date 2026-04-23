import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
from pathlib import Path
from pack_unpack import pack_4bit, pack_2bit
from pack_unpack import unpack_4bit, unpack_2bit
import torch.nn.functional as F
import math

def dequantize_from_dict(state_dict: dict, device: torch.device):
    codebook = state_dict["codebook"].to(device)
    scale = state_dict["scale"].to(device)
    shape = state_dict["shape"]
    indexes = state_dict["indexes"].to(device)


    if indexes.dtype == torch.uint8 and (codebook.shape[0] == 16 or codebook.shape[0] == 8): # 4 and 3 bit
        indexes = unpack_4bit(indexes)
    elif indexes.dtype == torch.uint8 and codebook.shape[0] == 4:
        indexes = unpack_2bit(indexes)

    state_dict["indexes"] = indexes

    codebook = codebook / codebook.abs().max().clamp(min=1e-8)
    weight = codebook[indexes.long()]
    weight = weight * scale

    out_features, in_features = shape
    return weight.view(out_features, in_features)


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "--pretrained",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="The model id or path of a pretrained HF model configuration.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="output",
        help="Path to the directory for storing converted models.",
    )
    parser.add_argument(
        "--codebooks_path",
        type=str,
        default=None,
        help="Path to the previously saved codebooks. If not specified or file does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    
    parser.add_argument(
        "--n_layers",
        type=int,
        default=None,
        help="Number of layers in the model. If not specified, it will be inferred from the model.",
    )
    
    return parser


def fake_quantize_int8_per_channel(tensor):
    # Get the maximum absolute value for each output channel (dim=1)
    max_abs_per_channel = tensor.abs().max(dim=1, keepdim=True).values
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    scale = max_abs_per_channel / 127.0 + epsilon
    # Quantize the tensor to int8
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return (quantized * scale).to(tensor.dtype).to(tensor.device)

# main.py --pretrained Qwen/Qwen3-4B --codebooks_path /home/aanuf/proj/learnable_codebooks/3bit/qwen3_4B/STE_LORA_512_1024_samples_20_plus_epoch_90bs_adam_no_exp_scale_diff_lr_last/codebook_layers.pth --output_dir /home/aanuf/proj/learnable_codebooks/3bit/qwen3_4B/STE_LORA_512_1024_samples_20_plus_epoch_90bs_adam_no_exp_scale_diff_lr_last/tmp/
def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    codebooks = torch.load(args.codebooks_path, map_location="cpu") if args.codebooks_path and Path(args.codebooks_path).is_file() else {}
    
    if len(codebooks.keys()) == 0:
        raise NotImplementedError("No codebooks found at the specified path. Please provide a valid path to the codebooks or ensure that the file exists.")  
    
    keys = list(codebooks.keys())
    
    for k in keys:
        if '_orig_mod.' in k:
            codebook_key = k.replace('_orig_mod.', '')
            if not codebook_key in codebooks:
                codebooks[codebook_key] = codebooks[k]
                del codebooks[k]

    layer_counter = 0
    mean_diff = 0.0
    sz_in_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in codebooks:
            layer_counter += 1
            state = codebooks[name]
            dequantized = dequantize_from_dict(state,  module.weight.data.device).to(module.weight.data.dtype)
            mean_diff += (module.weight.data.to(dequantized.dtype) - dequantized).abs().mean().item() / module.weight.data.to(dequantized.dtype).abs().mean().item()
            module.weight.data = dequantized
            del codebooks[name]  # free memory
            torch.cuda.empty_cache()
            bit_per_index = int(math.log2(state["codebook"].numel()))
            # after dequantize_from_dict state["indexes"] contains the unpacked indexes, so we can directly use its numel() to calculate the size in bytes.
            sz_in_bytes += state["codebook"].numel() * state["codebook"].element_size() + (state["indexes"].numel() * bit_per_index) / 8 + state["scale"].numel() * state["scale"].element_size()
        else:
            if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                module.weight.data = fake_quantize_int8_per_channel(module.weight.data)
                sz_in_bytes += module.weight.data.numel() + module.weight.shape[0] * 2  # int8 weights + per-channel scales
            elif hasattr(module, 'weight'):    
                sz_in_bytes += module.weight.data.numel() * module.weight.data.element_size()
            else:
                continue
                #print(f"Module {name} does not have weight attribute.")
    if model.config.tie_word_embeddings:
        # If input and output embeddings are tied, we have already counted the input embeddings, so we should not count the output embeddings again.
        sz_in_bytes -= model.get_input_embeddings().weight.data.numel() + model.get_input_embeddings().weight.shape[0] * 2  # int8 weights + per-channel scales

    print(f"Total size of the model with applied codebooks: {sz_in_bytes / (1024 * 1024):.2f} MB or {sz_in_bytes / (1024 * 1024 * 1024):.2f} GB")
    print(f"Average relative difference between original and dequantized weights: {mean_diff / layer_counter if layer_counter > 0 else 0.0}")
    # Save the model with the dequantized codebooks
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
