import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import sys
from pathlib import Path
from all_values_tuning import dequantize_from_dict


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

# main.py --pretrained Qwen/Qwen3-4B --codebooks_path /home/aanuf/proj/learnable_codebooks/3bit/qwen3_4B/STE_LORA_512_1024_samples_20_plus_epoch_90bs_adam_no_exp_scale_diff_lr_last/codebook_layers.pth --output_dir /home/aanuf/proj/learnable_codebooks/3bit/qwen3_4B/STE_LORA_512_1024_samples_20_plus_epoch_90bs_adam_no_exp_scale_diff_lr_last/tmp/
def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    codebooks = torch.load(args.codebooks_path, map_location="cpu") if args.codebooks_path and Path(args.codebooks_path).is_file() else {}
    
    keys = list(codebooks.keys())
    
    min_cb = -1000.0
    for k in keys:
        if '_orig_mod.' in k:
            codebook_key = k.replace('_orig_mod.', '')
            if not codebook_key in codebooks:
                codebooks[codebook_key] = codebooks[k]
                min_cb = max(min_cb, codebooks[k]["codebook"].abs().max().item())
                del codebooks[k]

    layer_counter = 0
    mean_diff = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in codebooks:
            layer_counter += 1
            #print(name, codebooks[name]["codebook"])
            dequantized = dequantize_from_dict(codebooks[name],  module.weight.data.device).to(module.weight.data.dtype)
            diff = (module.weight.data.to(dequantized.dtype) - dequantized).abs().max().item()
            mean_diff += (module.weight.data.to(dequantized.dtype) - dequantized).abs().mean().item() / module.weight.data.to(dequantized.dtype).abs().mean().item()
            #print(f"Max absolute difference between original and dequantized weights for layer {name}: {diff}")
            module.weight.data = dequantized
            del codebooks[name]  # free memory
            torch.cuda.empty_cache()  # free memory
            # if args.n_layers is not None and layer_counter >= args.n_layers:
            #     break
    print(f"Average relative difference between original and dequantized weights: {mean_diff / layer_counter if layer_counter > 0 else 0.0}")
    # Save the model with the applied codebooks
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
