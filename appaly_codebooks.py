import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Llama-3.2-1B-Instruct"
codebooks_path = "qwen3_8B/STE_LORA_64_300_samples_40_epochs_torch_compile_seqlen_1024_last/codebook_layers.pth"


codebooks = torch.load(codebooks_path, map_location="cpu")

print(codebooks.keys())
