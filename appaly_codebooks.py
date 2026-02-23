import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "meta-llama/Llama-3.2-1B-Instruct"
codebooks_path = "/home/aanuf/libs/nncf_aa/llama_3.2_1b-codebook_lr/codebook_layers.pth"


codebooks = torch.load(codebooks_path)

print(codebooks.keys())
