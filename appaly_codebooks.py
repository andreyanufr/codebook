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
    
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch.float16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    codebooks = torch.load(args.codebooks_path, map_location="cpu") if args.codebooks_path and Path(args.codebooks_path).is_file() else {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in codebooks:
            module.weight.data = dequantize_from_dict(codebooks[name], "cpu")
    
    # Save the model with the applied codebooks
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
