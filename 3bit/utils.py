import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import track
from typing import Optional



def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def get_reciprocal(tensor):
    """Memory-efficient reciprocal: zero for near-zero elements."""
    eps = 1e-5 if tensor.dtype == torch.float16 else 1e-30
    mask = tensor.abs() < eps
    safe = tensor.masked_fill(mask, 1.0)
    return (1.0 / safe).masked_fill_(mask, 0.0)


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
    setattr(module, name_list[-1], new_module)


def collate_fn(data, indexes, device):
    hidden_states = torch.cat([data["hidden_states"][i] for i in indexes], dim=0).to(device)

    kwargs = {}
    for key in data:
        if key != "hidden_states":
            val = data[key]
            # Move tensor tuples (e.g. position_embeddings) and tensors to the
            # same device as hidden_states to avoid cross-device errors.
            if isinstance(val, (tuple, list)) and all(isinstance(v, Tensor) for v in val):
                val = type(val)(v.to(device) for v in val)
            elif isinstance(val, Tensor):
                val = val.to(device)
            kwargs[key] = val
    return hidden_states, kwargs




class BlockInputCacher(nn.Module):
    def __init__(self, block: nn.Module, name: str):
        super().__init__()
        self.block = block
        self.name = name
        self.cached_inputs = {}
        
        self.cached_inputs["hidden_states"] = []

    @property
    def attention_type(self):
        if hasattr(self.block, "attention_type"):
            return self.block.attention_type
        return None

    def forward(self, hidden_states, **kwargs):
        self.cached_inputs["hidden_states"].append(hidden_states.detach())
        for key in kwargs:
            if key not in self.cached_inputs:
                self.cached_inputs[key] = kwargs[key]
        return self.block(hidden_states, **kwargs)
    
    def dump_cached_inputs(self, dir: str):
        names = []
        for key, tensors in self.cached_inputs.items():
            stacked = torch.cat(tensors, dim=0)
            names.append(f"{dir}/{self.name}_{key}.pt")
            torch.save(stacked, f"{dir}/{self.name}_{key}.pt")
        return names



@torch.no_grad()
def get_first_block_inputs(model: nn.Module, dataset: list[Tensor]):
    model.eval()
    
    model.model.layers[0] = BlockInputCacher(model.model.layers[0], name="layer_0")

    with torch.no_grad():
        for batch in track(dataset, description="Caching block inputs..."):
            model(batch)
    
    # After running through the dataset, dump the cached inputs for each block
    res = model.model.layers[0].cached_inputs

    model.model.layers[0] = model.model.layers[0].block  # Unwrap the original block to restore model functionality

    return res


def log_gradients_in_model(model, tb, step, layer_idx):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tb.add_scalar(f"layer_{layer_idx}/{tag}/grad", value.grad.abs().cpu().mean(), step)


def get_abs_top_percent_mask(x: torch.Tensor, percent: float = 1.0):
    """
    Return a mask for the top `percent` absolute values in x and its inverse.

    Args:
        x (torch.Tensor): Input tensor.
        percent (float): Percentage of elements to select (0~100).

    Returns:
        mask (torch.BoolTensor): True for top `percent` abs elements.
        inv_mask (torch.BoolTensor): Inverse of mask.
    """
    flat = x.view(-1)
    k = max(1, int(flat.numel() * percent / 1000))  # 至少选1个
    _, idx = torch.topk(torch.abs(flat), k)

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    mask = mask.view_as(x)
    return mask, ~mask
