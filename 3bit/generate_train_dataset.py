# This script generates the training dataset for AutoModelForCausalLM of train part of gsm8k.
# The input prompts with model responses are generated and saved in a jsonl file. The dataset is used for training the learnable codebooks.
# parameters model_id, number of samples, number of tokens per one sample, and output path can be set in the arguments.

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training dataset from gsm8k using a causal LM.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="The model id or path of a pretrained HF model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2048,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Number of new tokens to generate per sample.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="gsm8k_train_dataset.jsonl",
        help="Output path for the jsonl file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    dataset = load_dataset("openai/gsm8k", "main", split="train")

    num_samples = min(args.num_samples, len(dataset))
    dataset = dataset.select(range(num_samples))

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)

    with open(args.output_path, "w") as f:
        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            batch = dataset[start:end]
            prompts = batch["question"]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            for i, output in enumerate(outputs):
                prompt = prompts[i]
                full_text = tokenizer.decode(output, skip_special_tokens=True)
                record = {
                    "prompt": prompt,
                    "response": full_text,
                    "answer": batch["answer"][i],
                }
                f.write(json.dumps(record) + "\n")

            print(f"Processed {end}/{num_samples} samples")

    print(f"Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
