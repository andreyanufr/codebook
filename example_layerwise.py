#!/usr/bin/env python3
"""
Example script demonstrating layer-wise fine-tuning for compressed LLM models.

This script shows how to use the finetune_layerwise_l2() function
to train a compressed model layer by layer using L2 loss.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import functions from main.py
from main import (
    wrap_model,
    unwrap_model,
    finetune_layerwise_l2,
    get_compression_calibration,
)


def main():
    """Run layer-wise fine-tuning example."""
    
    # Configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16
    output_dir = "./layerwise_example_output"
    
    print("="*80)
    print("Layer-wise Fine-tuning Example")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Step 1: Load original (teacher) model
    print("Step 1: Loading original bf16 model...")
    orig_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    orig_model.eval()
    print(f"  ✓ Loaded {model_name}\n")
    
    # Step 2: Load model to be compressed
    print("Step 2: Loading model to compress...")
    compressed_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    print(f"  ✓ Loaded {model_name}\n")
    
    # Step 3: Load tokenizer
    print("Step 3: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  ✓ Tokenizer loaded\n")
    
    # Step 4: Wrap model with learnable codebook layers
    print("Step 4: Wrapping model with codebook layers...")
    compressed_model = wrap_model(compressed_model, n_bits=2)
    print("  ✓ Model wrapped with 2-bit codebook layers\n")
    
    # Step 5: Prepare training data
    print("Step 5: Preparing training data...")
    train_loader = get_compression_calibration(
        num_samples=128,  # Use smaller number for quick example
        seqlen=512,
        tokenizer=tokenizer,
        device=device
    )
    print(f"  ✓ Prepared {len(train_loader)} training samples\n")
    
    # Step 6: Run layer-wise fine-tuning
    print("Step 6: Running layer-wise fine-tuning...")
    compressed_model = finetune_layerwise_l2(
        orig_model=orig_model,
        compressed_model=compressed_model,
        train_loader=train_loader,
        lr=1e-3,
        epochs_per_layer=5,  # Fewer epochs for quick example
        batch_size=32,
        microbatch_size=8,
        device=device,
        tb=None,  # Set to SummaryWriter instance for logging
        layer_type="both"  # Train both MLP and attention
    )
    print("  ✓ Layer-wise fine-tuning complete\n")
    
    # Step 7: Unwrap and save the model
    print("Step 7: Unwrapping and saving model...")
    compressed_model = unwrap_model(compressed_model)
    compressed_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  ✓ Model saved to {output_dir}\n")
    
    # Step 8: Test the model
    print("Step 8: Testing the compressed model...")
    test_input = tokenizer("Hello, how are you?", return_tensors="pt").to(device)
    with torch.no_grad():
        output = compressed_model.generate(**test_input, max_new_tokens=20)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  Input: 'Hello, how are you?'")
    print(f"  Output: '{generated_text}'\n")
    
    print("="*80)
    print("Example complete!")
    print("="*80)
    print(f"\nYou can now use the compressed model from: {output_dir}")
    print("\nTo train with different settings:")
    print("  - Change layer_type to 'mlp' or 'attention'")
    print("  - Adjust epochs_per_layer for more training")
    print("  - Increase num_samples for better quality")


if __name__ == "__main__":
    main()
