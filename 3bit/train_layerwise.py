"""
Layer-wise fine-tuning with Straight-Through Estimator (STE) and LoRA adapters.

This module provides a training pipeline where:
1. Codebook quantization uses STE so gradients flow through index assignments
   (soft backward, hard forward) giving the codebook richer gradient signal.
2. LoRA (Low-Rank Adaptation) adapters add learnable low-rank corrections on
   top of the quantized weight, dramatically increasing the effective degrees
   of freedom without touching the original weights.
3. Training proceeds layer-by-layer, minimising L2 loss between
   quantized+LoRA layer outputs and FP reference outputs.

Usage:
    from layerwise_ste_tuning import finetune_layerwise_ste

    finetune_layerwise_ste(
        model, tokenizer, train_loader,
        lr=1e-3, lora_rank=16, lora_alpha=32,
        epochs_per_layer=10, batch_size=64, microbatch_size=8,
        device="cuda:0", tb=tb_writer,
    )
"""

from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from pathlib import Path
from rich.progress import track
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from utils import get_reciprocal, set_module
from utils import (
    cleanup,
    collate_fn,
    get_first_block_inputs,
    log_gradients_in_model,
    get_abs_top_percent_mask
)


from one_hot_uint8 import one_hot as one_hot_uint8_impl
from pack_unpack import pack_4bit, pack_2bit

import numpy as np
from scipy.stats import norm

def create_normal_distributed_values(n_levels=8) -> np.ndarray:
    probs = (np.arange(n_levels) + 0.5) / n_levels

    # Inverse CDF (quantiles) of standard normal distribution
    values = norm.ppf(probs)

    # Normalize to [-1, 1]
    values = values / np.max(np.abs(values))
    
    if n_levels > 4:
        values[n_levels // 2] = 0.0

    return torch.tensor(values.astype(np.float32))


class CodebookLoRASTELinear(nn.Module):
    """Linear layer combining codebook quantisation with STE and LoRA adapters.

    Effective weight during training::

        W_eff = W_q_ste + B @ A * (alpha / rank)

    where ``W_q_ste`` is computed via STE (hard forward, soft backward) from
    codebook / scale / indexes, and ``B @ A`` is the low-rank LoRA correction.

    Parameters
    ----------
    orig_layer : nn.Linear
        The original fp linear layer to be wrapped.
    group_size : int
        Number of weight elements per scale group.
    n_bits : int
        Codebook size = 2 ** n_bits.
    lora_rank : int
        Rank *r* of the LoRA adapters A (r×in) and B (out×r).
    lora_alpha : float
        LoRA scaling factor (effective scaling = alpha / rank).
    use_exp_for_scale : bool
        If True, parameterise scale as exp(s) for positivity.
    ste_temperature : float
        Initial softmax temperature for STE soft assignment.
    """

    def __init__(
        self,
        orig_layer: nn.Linear,
        group_size: int = 32,
        n_bits: int = 2,
        lora_rank: int = 32,
        lora_alpha: float = 32.0,
        use_exp_for_scale: bool = False,
        use_exp_for_lora: bool = False,
        ste_temperature: float = 1.0,
    ):
        super().__init__()

        assert isinstance(orig_layer, nn.Linear), "Only nn.Linear layers are supported"
        assert orig_layer.bias is None, "Bias is not supported"

        self.orig_layer = orig_layer
        self.group_size = group_size
        self.n_bits = n_bits
        self.use_exp_for_scale = use_exp_for_scale
        self.use_exp_for_lora = use_exp_for_lora
        # Mutable – the training loop adjusts this each epoch
        self.ste_temperature: float = ste_temperature
        # Controls whether forward uses STE (True) or hard-only (False)
        self.training_mode_ste: bool = True
        self.use_one_hot = False

        out_features, in_features = orig_layer.weight.shape

        # ---- Codebook (float32 for optimizer stability) ----
        if n_bits == 2:
            initial_codebook = torch.tensor(
                [-1.0, -0.28, 0.28, 1.0],
                #[-2.0, -1.0, 0.0, 1.0],
                dtype=torch.float32,
                device=orig_layer.weight.device,
            )
        else:
            initial_codebook = torch.tensor(
                [i for i in range(-(2 ** (n_bits - 1)), 2 ** (n_bits - 1))],
                dtype=torch.float32,
                device=orig_layer.weight.device,
            ) / (2 ** (n_bits - 1))
            
            initial_codebook = create_normal_distributed_values(n_levels=2**n_bits).to(torch.float32).to(orig_layer.weight.device)

        self.codebook = nn.Parameter(initial_codebook, requires_grad=True)
        
        assert self.codebook.numel() == 2 ** n_bits, "Codebook size must match n_bits"

        # ---- Scale & indexes ----
        self._init_indexes_and_scale()

        # ---- LoRA adapters (placeholder – will be overwritten by SVD init) ----
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        if lora_rank == -1:
            self.lora = nn.Parameter(
                torch.zeros(out_features, in_features, dtype=torch.float32, device=orig_layer.weight.device), requires_grad=True
            )
        else:
            self.lora_A = nn.Parameter(
                torch.empty(lora_rank, in_features, dtype=torch.float32, device=orig_layer.weight.device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, lora_rank, dtype=torch.float32, device=orig_layer.weight.device), requires_grad=True
            )
            nn.init.kaiming_uniform_(self.lora_A)
            
            self.lora_c = nn.Parameter(
                torch.zeros(out_features, 1, dtype=torch.float32, device=orig_layer.weight.device), requires_grad=True
            )
            
            self.lora_r = nn.Parameter(
                torch.zeros(1, in_features, dtype=torch.float32, device=orig_layer.weight.device), requires_grad=True
            )

        # Freeze original weight
        self.orig_layer.weight.requires_grad = False

        # Weight-space MSE init for codebook + scale
        self._mse_init()
        
        # Save VRAM – move original weight to CPU (pulled back during STE fwd)
        self.orig_layer.to("cpu")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def set_device(self, device):
        pass

    @torch.no_grad()
    def _init_indexes_and_scale(self):
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size)

        if self.n_bits <= -2:
            scale = weight.abs().mean(dim=2, keepdim=True).clamp(min=1e-5) / self.codebook.abs().max().detach()
        else:
            scale = weight.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5) / self.codebook.abs().max().detach()
        if self.use_exp_for_scale:
            scale = torch.log(scale)

        self.scale = nn.Parameter(scale.float(), requires_grad=True)


    def get_lora(self):
        if self.lora_rank == -1:
            return self.lora
        else:
            return (self.lora_B @ self.lora_A) * (self.lora_alpha / self.lora_rank) + self.lora_c + self.lora_r


    def _get_effective_weight(self):
        """Return ``orig_weight + lora_delta`` (2-D, on codebook device).

        If LoRA parameters have not been created yet (during __init__),
        returns just the original weight.
        """

        if not (hasattr(self, "lora_B") or hasattr(self, "lora")):
            return self.orig_layer.weight.data.detach()

        lora = self.get_lora()
        if self.use_exp_for_lora:
            # Clamp to prevent overflow (exp(10) ~ 22026 in bf16 range)
            exponent = lora.clamp(-10.0, 10.0)
            return self.orig_layer.weight.data.to(self.codebook.device) / torch.exp(exponent)
        else:
            return self.orig_layer.weight.data.to(self.codebook.device) + lora



    def dequantize_by_distance(self, codebook, normalized, return_indexes = False):
        thresholds = (codebook[:-1] + codebook[1:]) * 0.5
        
        sigma = self.ste_temperature * (torch.abs(codebook[:-1] - codebook[1:]).mean() * 0.05 + 1e-8)
        
        # stochasticity for better exploration of codebook assignments during training
        idx = torch.bucketize(normalized + sigma * torch.randn_like(normalized), thresholds)
        
        if return_indexes:
            return idx

        if self.use_one_hot or self.n_bits == 2 or True:#codebook.requires_grad:
            # train only codebook
            one_hot = F.one_hot(
                idx, num_classes=2 ** self.n_bits
            ).to(codebook.device, codebook.dtype)

            quantized = (one_hot * codebook).sum(dim=-1)
        else:
            quantized = codebook[idx]

        return (normalized - normalized.detach()) + quantized

    def _get_scale(self):
        if self.use_exp_for_scale:
            return self.scale.clamp(-20.0, 20.0).exp()
        else:
            return self.scale.clamp(min=1e-5)

    def _get_normalized_weights(self, differentiable: bool = False):
        """Return ``(orig_weight + lora_delta) / scale`` (grouped).

        If *differentiable* is True, gradients flow through ``scale``
        and the LoRA parameters.
        """
        if differentiable:
            weight = self._get_effective_weight()
        else:
            with torch.no_grad():
                weight = self._get_effective_weight()

        out_features, in_features = weight.shape
        weight = weight.view(
            out_features, in_features // self.group_size, self.group_size
        )

        scale = self._get_scale()
        iscale = get_reciprocal(scale)
        normalized = weight * iscale
        return normalized

    # ------------------------------------------------------------------
    # Weight-space MSE initialization (same strategy as CodebookWrapperLinear)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def check_nans(self):
        if torch.isnan(self.codebook).any():
            raise ValueError("NaNs detected in codebook")
        if torch.isnan(self.scale).any():
            raise ValueError("NaNs detected in scale")
        
        if not torch.isfinite(self.codebook).all():
            raise ValueError("Non-finite values detected in codebook")
        if not torch.isfinite(self.scale).all():
            raise ValueError("Non-finite values detected in scale")


    def _mse_init(self, n_iters: int = 200, lr: float = 0.01):
        device = self.codebook.device
        self.orig_layer.to(device)
        orig_weight = self.orig_layer.weight.data.to(device)


        if self.lora_rank != -1:
            lora = self.lora_B
        else:
            lora = self.lora

        optimizer = torch.optim.Adam([self.scale, lora, self.codebook], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

        best_loss = float("inf")
        best_scale = self.scale.data.clone()
        best_lora = lora.data.clone()
        best_codebook = self.codebook.data.clone()
        
        print("initial codebook:", self.codebook.data)
        
        self.use_one_hot = True
        not_improved_iters = 0

        for i in range(n_iters):
            optimizer.zero_grad()
            deq_weight = self._dequantize_ste()

            loss = F.mse_loss(deq_weight, orig_weight.to(deq_weight.dtype))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.scale, lora, self.codebook], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if loss.item() < best_loss:
                    not_improved_iters = max(not_improved_iters - 1, 0)
                    best_loss = loss.item()
                    best_scale = self.scale.data.clone()
                    best_lora = lora.data.clone()
                    best_codebook = self.codebook.data.clone()
                else:
                    not_improved_iters += 1
            if not_improved_iters >= 20:
                break

        self.use_one_hot = False
        with torch.no_grad():
            self.scale.data.copy_(best_scale)
            if self.lora_rank != -1:
                self.lora_B.data.copy_(best_lora)
            else:
                self.lora.data.copy_(best_lora)
            self.codebook.data.copy_(best_codebook)

        optimizer.zero_grad()
        del orig_weight, optimizer, scheduler
        del best_scale, best_lora, best_codebook
        del best_loss
        cleanup()

        print("final codebook:", self.codebook.data)

        self.orig_layer.weight.data = self.orig_layer.weight.data.to("cpu")
        self.orig_layer.to("cpu")

    # ------------------------------------------------------------------
    # Dequantisation variants
    # ------------------------------------------------------------------

    def get_codebook(self):
        return self.codebook / self.codebook.abs().max().clamp(min=1e-8)


    def _dequantize_hard(self):
        """Standard hard dequantisation (one-hot from stored indexes)."""
        
        normalized = self._get_normalized_weights(differentiable=False)
        weight = self.dequantize_by_distance(self.get_codebook(), normalized)
        scale = self.scale.clamp(-20.0, 20.0).exp() if self.use_exp_for_scale else self.scale
        weight = weight * scale
        return weight.view(self.orig_layer.weight.shape).to(self.orig_layer.weight.dtype)

    def _dequantize_ste_impl(self):
        """Core STE dequantisation logic (may be wrapped by checkpoint)."""
        normalized = self._get_normalized_weights(differentiable=True)
        weight = self.dequantize_by_distance(self.get_codebook(), normalized)
        scale = self.scale.clamp(-20.0, 20.0).exp() if self.use_exp_for_scale else self.scale
        weight = weight * scale
        return weight.view(self.orig_layer.weight.shape).to(self.orig_layer.weight.dtype)

    def _dequantize_ste(self):
        """STE dequantisation: forward = hard assignment, backward = soft.

        Uses ``orig_weight + lora_delta`` as the reference weight so that the
        codebook, scale, and soft-assignment gradients all account for the LoRA
        correction.  This means the codebook naturally learns to represent the
        corrected weight, making LoRA merging virtually free.

        When ``use_one_hot`` is True, gradient checkpointing is applied to
        reduce memory by recomputing intermediates (e.g. the one-hot matrix)
        during backward instead of storing them.

        .. note::
           Assumes ``self.orig_layer`` is already on the same device as
           ``self.codebook`` (the training function handles this).
        """
        if not self.use_one_hot:
            return _checkpoint(self._dequantize_ste_impl, use_reentrant=False)
        return self._dequantize_ste_impl()

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _lora_weight(self):
        """Compute LoRA correction ``B @ A * (alpha / rank)``."""
        return self.get_lora()

    @torch.no_grad()
    def merge_lora(self):
        """Absorb the LoRA correction into the original weight.

        Because the STE path already quantises ``orig_weight + lora_delta``,
        the codebook, scale, and indexes are already adapted to the merged
        weight.  Merging therefore only needs to:

        1. ``orig_weight ← orig_weight + B @ A * α/r``
        2. Final ``update_indexes()`` to snap indexes to the merged weight.
        3. Zero out LoRA matrices.

        No expensive re-quantisation loop is required.
        """
        device = self.codebook.device

        # 1. Compute merged fp weight and write it back
        self.orig_layer.to(device)

        self.orig_layer.weight.data = self._get_effective_weight().to(self.orig_layer.weight.dtype)

        # 2. Zero out LoRA so _get_effective_weight() == orig_weight
        if self.lora_rank != -1:
            self.lora_B.data.zero_()
            self.lora_A.data.zero_()
            self.lora_c.data.zero_()
            self.lora_r.data.zero_()
        else:
            self.lora.data.zero_()

        # Move orig weight back to CPU to save VRAM
        self.orig_layer.to("cpu")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        if self.training and self.training_mode_ste:
            # LoRA is already folded into the STE quantisation path
            w = self._dequantize_ste()
        else:
            # After merge_lora(), LoRA is absorbed into orig_weight and
            # the codebook/scale/indexes already represent it.
            w = self._dequantize_hard()
        return F.linear(x, w)
    

    @torch.no_grad()
    def check_hard_and_ste_consistency(self, atol: float = 1e-4):
        """Check that hard and STE dequantisation are close (for debugging)."""
        w_hard = self._dequantize_hard()
        w_ste = self._dequantize_ste()
        if not torch.allclose(w_hard, w_ste, atol=atol):
            max_diff = (w_hard - w_ste).abs().max().item()
            print(f"WARNING: Hard and STE dequantisation differ by max {max_diff:.6f}")

    # ------------------------------------------------------------------
    # Inference / unwrap helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dequantize(self):
        """Return the dequantized weight (codebook only, after LoRA has been merged)."""
        return self._dequantize_hard()
    
    @torch.no_grad()
    def get_compressed_indexes(self):
        normalized = self._get_normalized_weights(differentiable=False)
        indexes = self.dequantize_by_distance(self.get_codebook(), normalized, return_indexes=True)
        indexes = indexes.to(torch.uint8)
        
        if self.n_bits == 2:
            packed = pack_2bit(indexes)
        elif self.n_bits in [3, 4]:
            packed = pack_4bit(indexes)
        else:
            raise ValueError("Unsupported n_bits for packing indexes")
        return packed


    @torch.no_grad()
    def get_state_dict(self):
        """Return a state dict containing just the codebook, scale, and indexes."""
        return {
            "codebook": self.get_codebook().data.cpu(),
            "scale": (self.scale.clamp(-20.0, 20.0).exp() if self.use_exp_for_scale else self.scale).data.cpu(),
            "shape": self.orig_layer.weight.shape,
            "indexes": self.get_compressed_indexes().cpu(),
        }


def save_codebook_layers(model: nn.Module, output_dir: Path, epoch: Optional[int] = None):
    """
    Saves the codebook layers of the model to the specified output directory.

    :param model: The model containing the codebook layers to be saved.
    :param output_dir: The directory where the codebook layers will be saved.
    :param epoch: The current epoch number (optional).
    """
    codebook_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, CodebookLoRASTELinear):
            codebook_state_dict[name] = module.get_state_dict()

    if epoch is not None:
        torch.save(codebook_state_dict, output_dir / f"codebook_layers_epoch_{epoch}.pth")
        print(f"Codebook layers saved to {output_dir / f'codebook_layers_epoch_{epoch}.pth'}")
    else:
        torch.save(codebook_state_dict, output_dir / "codebook_layers.pth")
        print(f"Codebook layers saved to {output_dir / 'codebook_layers.pth'}")


# ---------------------------------------------------------------------------
# Wrap / unwrap helpers
# ---------------------------------------------------------------------------

def wrap_model_block_ste(
    block: nn.Module,
    n_bits: int = 2,
    lora_rank: int = 32,
    group_size: int = 32,
    lora_alpha: float = 32.0,
    layer_index: int = -1,
    n_layers: int = -1,
    use_llama_cpp_scheme: bool = True,
) -> nn.Module:
    """Replace ``nn.Linear`` layers in *block* with ``CodebookLoRASTELinear``."""
    changed_modules: dict[str, nn.Module] = {}
    changed_modules_4_bit: dict[str, nn.Module] = {}

    if use_llama_cpp_scheme and layer_index >= 0 and n_layers > 0:
        for name, module in block.named_modules():
            if "lm_head" in name:
                continue
            if isinstance(module, nn.Linear) and ("v_proj" in name or "down_proj" in name) and n_bits == 2:
                changed_modules_4_bit[name] = module
            elif isinstance(module, nn.Linear):
                changed_modules[name] = module
    else:
        for name, module in block.named_modules():
            if "lm_head" in name or "v_proj" in name or "down_proj" in name:
                continue
            if isinstance(module, nn.Linear):
                changed_modules[name] = module

    for name, module in changed_modules.items():
        #print(f"  Wrapping {name} with CodebookLoRASTELinear 2bit (rank={lora_rank}, group_size={group_size})")
        set_module(
            block,
            name,
            CodebookLoRASTELinear(
                module,
                n_bits=n_bits,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                group_size=group_size,
            ),
        )
    cleanup()
    
    for name, module in changed_modules_4_bit.items():
        #print(f"  Wrapping {name} with CodebookLoRASTELinear 4bit (rank={lora_rank}, group_size={2 * group_size})")
        set_module(
            block,
            name,
            CodebookLoRASTELinear(
                module,
                n_bits=2 * n_bits,
                lora_rank=lora_rank // 2 if lora_rank != -1 else 512,
                lora_alpha=lora_alpha,
                group_size=2 * group_size,
            ),
        )
    
    for name, module in block.named_modules():
        if isinstance(module, CodebookLoRASTELinear):
            print(f"  Wrapped {name} with CodebookLoRASTELinear (n_bits={module.n_bits}, lora_rank={module.lora_rank}, group_size={module.group_size}), codebook shape: {module.codebook.shape} ")
    cleanup()

    return block


def unwrap_model_block_ste(block: nn.Module) -> nn.Module:
    """Unwrap ``CodebookLoRASTELinear`` → ``nn.Linear`` with dequantized weight.

    Assumes ``merge_lora()`` has already been called so the LoRA delta is
    folded into the codebook representation.  The unwrapped linear layer
    gets the dequantized (codebook-only) weight.
    """
    changed: dict[str, nn.Module] = {}
    for name, module in block.named_modules():
        if isinstance(module, CodebookLoRASTELinear):
            module.orig_layer.weight.data.copy_(module.dequantize().cpu())
            changed[name] = module.orig_layer
    for name, orig_layer in changed.items():
        set_module(block, name, orig_layer)
    return block


def wrap_model_ste(model: nn.Module,
                n_bits: int = 2,
                lora_rank: int = 32,
                group_size: int = 32,
                lora_alpha: float = 32.0,) -> nn.Module:
    """Wrap all ``nn.Linear`` layers in the full model with
    ``CodebookLoRASTELinear``."""
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = wrap_model_block_ste(layer, layer_index=i, n_layers=len(model.model.layers),
                                                     n_bits=n_bits, lora_rank=lora_rank,
                                                     group_size=group_size, lora_alpha=lora_alpha)
    return model


def unwrap_model_ste(model: nn.Module) -> nn.Module:
    """Unwrap all ``CodebookLoRASTELinear`` layers in the full model."""
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = unwrap_model_block_ste(layer)
    return model


# ---------------------------------------------------------------------------
# Single-layer training
# ---------------------------------------------------------------------------

def finetune_layer_ste(
    layer: nn.Module,
    fp_inputs: dict,
    fp_outputs: list[Tensor],
    layer_idx: int = -1,
    lr: float = 1e-4,
    lora_lr: Optional[float] = None,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    tb: Optional[SummaryWriter] = None,
    return_next_layer_inputs: bool = False,
    ste_temp_start: float = 1.0,
    ste_temp_end: float = 0.01,
    keep_data_on_cpu: bool = True,
    patience: int = 3,
) -> nn.Module:
    """Fine-tune a single transformer block using STE + LoRA with L2 loss.

    Parameters
    ----------
    layer : nn.Module
        A single transformer block whose ``CodebookLoRASTELinear`` sub-layers
        will be trained.
    fp_inputs : dict
        ``{"hidden_states": [...], ...}`` – reference inputs for this layer.
    fp_outputs : list[Tensor]
        FP reference outputs to match.
    lora_lr : float or None
        Separate learning rate for LoRA params (defaults to *lr*).
    ste_temp_start / ste_temp_end : float
        STE softmax temperature annealed linearly from start → end over epochs.
    patience : int
        Number of epochs to wait for improvement before early stopping.
    """
    if lora_lr is None:
        lora_lr = lr

    # ------------------------------------------------------------------
    # Categorise trainable parameters
    # ------------------------------------------------------------------
    codebooks: list[nn.Parameter] = []
    scales: list[nn.Parameter] = []
    lora_params: list[nn.Parameter] = []
    
    
    layer = torch.compile(layer)

    for name, param in layer.named_parameters():
        if "codebook" in name:
            print("Train: ", name)
            param.requires_grad = True
            codebooks.append(param)
        elif "scale" in name:
            print("Train: ", name)
            param.requires_grad = True
            scales.append(param)
        elif "lora" in name:
            print("Train: ", name)
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    param_groups = [
        {"params": codebooks, "lr":  lr, "label": "codebook"},
        {"params": scales, "lr":  lr, "label": "scale"},
        #{"params": lora_params, "lr": lora_lr, "label": "lora"},
        {"params": lora_params, "lr": lora_lr, "label": "lora"},
    ]
    # Drop empty groups
    param_groups = [g for g in param_groups if g["params"]]

    if not param_groups:
        print(f"WARNING: No trainable parameters in layer {layer_idx}, skipping.")
        if return_next_layer_inputs:
            num_samples = len(fp_inputs["hidden_states"])
            next_inputs = []
            with torch.no_grad():
                for i in range(num_samples):
                    hidden, kwargs = collate_fn(fp_inputs, [i], device=device)
                    out = layer(hidden, **kwargs)
                    next_inputs.append(out.detach().cpu() if keep_data_on_cpu else out)
                    del out
            return layer, next_inputs
        return layer

    # ------------------------------------------------------------------
    # Move original weights to device for STE (avoids repeated CPU→GPU copies)
    # ------------------------------------------------------------------
    ste_modules: list[CodebookLoRASTELinear] = [
        m for m in layer.modules() if isinstance(m, CodebookLoRASTELinear)
    ]
    for m in ste_modules:
        m.orig_layer.to(device)
        m.use_one_hot = True  # STE with checkpointing to save memory

    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(fp_inputs["hidden_states"])
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size
    total_opt_steps = epochs_per_layer * epoch_samples // batch_size

    # ------------------------------------------------------------------
    # Training bookkeeping
    # ------------------------------------------------------------------

    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
    #opt = torch.optim.NAdam(param_groups)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=lr * 1e-4, T_max=total_opt_steps)
    
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #             opt, start_factor=1.0, end_factor=0.0, total_iters=total_opt_steps
    #         )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    
    moving_average_gradient_norm = [0.0001 for _ in codebooks]
    alpha = 0.95
    best_loss = float("inf")

    for epoch in range(epochs_per_layer):
        opt.zero_grad()
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_loss = 0.0
        num_batches = 0
        loss_numerator = grad_steps = 0
        loss_denominator = 0.0
        max_norm = 1.0 #0.98**epoch
        

        # Anneal STE temperature
        progress = epoch / max(epochs_per_layer - 1, 1)
        current_temp = ste_temp_start + (ste_temp_end - ste_temp_start) * progress
        for m in ste_modules:
            m.ste_temperature = current_temp

        for indices in track(
            batch_indices_epoch,
            description=f"  Layer {layer_idx}, Epoch {epoch}/{epochs_per_layer} (T={current_temp:.3f})",
        ):
            indices = indices.tolist()

            hidden, kwargs = collate_fn(fp_inputs, indices, device=device)
            hidden = hidden.to(device)

            layer_outputs = layer(hidden, **kwargs)
            orig_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(device)

            #loss = F.mse_loss(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))

            _, mask = get_abs_top_percent_mask(torch.abs(layer_outputs - orig_output))  # This will update the mask used in the forward pass of CodebookWrapperLinear for the next iteration
            loss = torch.mean(((layer_outputs - orig_output.to(dtype=layer_outputs.dtype)) * mask)**2)

            if not torch.isfinite(loss).item():
                print(
                    f"WARNING: Non-finite loss at layer {layer_idx}, step {global_step}. Skipping batch."
                )
                opt.zero_grad()
                loss_numerator = grad_steps = 0
                loss_denominator = 0.0
                del hidden, layer_outputs, orig_output, loss
                continue

            loss_numerator += loss.item()
            loss_denominator += torch.mean(orig_output ** 2).detach().item()
            grad_steps += 1

            (loss / grad_accumulation_steps).backward()

            del hidden, layer_outputs, orig_output, loss

            if grad_steps == grad_accumulation_steps:
                for group in param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], max_norm)
                
                for i, param in enumerate(codebooks):
                    grad_norm = param.grad.data.norm().item()
                    moving_average_gradient_norm[i] = alpha * moving_average_gradient_norm[i] + (1 - alpha) * grad_norm
                    adaptive_clip_value = max(0.00000001, moving_average_gradient_norm[i])
                    torch.nn.utils.clip_grad_value_([param], adaptive_clip_value)

                opt.step()
                scheduler.step()

                # Refresh indexes during early (exploratory) epochs
                for m in ste_modules:
                    if hasattr(m, "update_indexes"):
                        m.update_indexes()

                rel_loss = loss_numerator / max(loss_denominator, 1e-8)
                agg_loss = loss_numerator / grad_steps
                epoch_loss += agg_loss
                num_batches += 1
                loss_numerator = loss_denominator = grad_steps = 0

                if tb is not None:
                    tb.add_scalar(f"ste_rel_loss/layer_{layer_idx}", rel_loss, global_step)
                    tb.add_scalar(f"ste_loss/layer_{layer_idx}", agg_loss, global_step)
                    tb.add_scalar(f"ste_lr_codebook/layer_{layer_idx}", opt.param_groups[0]["lr"], global_step)
                    tb.add_scalar(f"ste_temperature/layer_{layer_idx}", current_temp, global_step)
                    log_gradients_in_model(layer, tb, global_step, layer_idx)

                opt.zero_grad()
                global_step += 1

        cleanup()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"    Epoch {epoch}: avg loss = {avg_epoch_loss:.6f}")

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ------------------------------------------------------------------
    # Post-training: merge LoRA → requantize, switch to hard mode
    # ------------------------------------------------------------------
    print(f"  Merging LoRA into codebook representation ...")
    for m in ste_modules:
        m.training_mode_ste = False
        #m.check_hard_and_ste_consistency()
        m.merge_lora()

    del opt, scheduler
    cleanup()
    
    if hasattr(layer, "_orig_mod"):
        layer = layer._orig_mod

    print(f"\n{'='*80}")
    print(f"STE+LoRA fine-tuning complete for layer {layer_idx}!")
    print(f"{'='*80}\n")

    # Move orig_layers back to CPU to free GPU memory before computing next_inputs
    for m in ste_modules:
        m.orig_layer.to("cpu")
    cleanup()

    if return_next_layer_inputs:
        # Temporarily move orig_layers to device for the forward pass
        for m in ste_modules:
            m.orig_layer.to(device)

        next_inputs = []
        with torch.no_grad():
            for i in range(num_samples):
                hidden, kwargs = collate_fn(fp_inputs, [i], device=device)
                out = layer(hidden.to(device), **kwargs)
                next_inputs.append(out.detach().cpu() if keep_data_on_cpu else out)
                del out

        for m in ste_modules:
            m.orig_layer.to("cpu")
            if hasattr(m, "indexes"):
                m.indexes = m.indexes.cpu()

        return layer, next_inputs

    return layer


# ---------------------------------------------------------------------------
# Full model layer-wise orchestration
# ---------------------------------------------------------------------------

def finetune_layerwise_ste(
    model: nn.Module,
    train_loader: list[Tensor],
    lr: float = 1e-4,
    lora_lr: Optional[float] = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = torch.device("cuda:0"),
    tb: Optional[SummaryWriter] = None,
    ignored_layers: Optional[list[int]] = None,
    ste_temp_start: float = 1.0,
    ste_temp_end: float = 0.01,
    index_update_epochs: int = 4,
    group_size: int = 32,
    keep_data_on_cpu: bool = True,
    codebook_dst_dir: Optional[Path] = None,
    num_bits: int = 3
) -> nn.Module:
    """Layer-wise fine-tuning using STE codebook quantisation + LoRA.

    Processes each transformer block sequentially:
    1. Compute FP reference outputs for the block.
    2. Wrap linear sub-layers with ``CodebookLoRASTELinear``.
    3. Optimise codebook, scale, and LoRA parameters to minimise
       L2 distance to FP reference outputs.
    4. Propagate quantised outputs to the next block.

    Parameters
    ----------
    model : nn.Module
        HuggingFace causal-LM model.
    tokenizer
        Associated tokenizer (used only for ``get_first_block_inputs``).
    train_loader : list[Tensor]
        List of input_ids tensors.
    lr : float
        Base learning rate (codebook). Scale uses 10×lr.
    lora_lr : float or None
        LoRA learning rate (defaults to *lr*).
    lora_rank / lora_alpha : int, float
        LoRA hyper-parameters.
    ignored_layers : list[int] or None
        Layer indices to skip (supports negative indexing).
        Defaults to ``[0, 1, 2, 3, -1, -2, -3, -4]``.
    ste_temp_start / ste_temp_end : float
        STE softmax temperature annealed linearly from start → end over epochs.
    index_update_epochs : int
        Number of initial epochs during which indexes are refreshed after every
        optimiser step.
    group_size : int
        Number of weight elements per scale group in the quantisation scheme.
    keep_data_on_cpu : bool
        If True, keep intermediate activations on CPU to reduce GPU memory usage.
    codebook_dst_dir : Path or None
        If not None, directory to save the codebook and scale parameters for each layer after training (one with dict "codebook.pt").
    num_bits : int
        Number of bits for quantisation.
    """
    if ignored_layers is None:
        ignored_layers = [] #[0, 1, 2, 3, -1, -2, -3, -4]

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    model_name = model.config.name_or_path.replace("/", "_")
    
    cache_name = f"first_block_inputs_{model_name}.pt"
    if os.path.exists(cache_name):
        hidden_states = torch.load(cache_name, map_location="cpu")
        inputs = get_first_block_inputs(model, dataset=train_loader, only_one_batch=True)
        inputs["hidden_states"] = hidden_states    
    else:
        # Cache first-block inputs via a single FP forward pass
        inputs = get_first_block_inputs(model, dataset=train_loader)
        torch.save(inputs["hidden_states"], cache_name)
    
    print("Hidden shapes: ", len(inputs["hidden_states"]), inputs["hidden_states"][0].shape, inputs["hidden_states"][1].shape)

    if keep_data_on_cpu:
        inputs["hidden_states"] = [x.to("cpu") for x in inputs["hidden_states"]]

    model.to("cpu")
    torch.cuda.empty_cache()

    # Strip accelerate dispatch hooks (they conflict with explicit device mgmt)
    saved_accelerate_forwards: dict[nn.Module, object] = {}
    for module in model.modules():
        if hasattr(module, "_old_forward"):
            saved_accelerate_forwards[module] = module.forward
            module.forward = module._old_forward
            del module._old_forward

    # Move non-hidden-state inputs to device once
    if "position_embeddings" in inputs:
        inputs["position_embeddings"] = (
            inputs["position_embeddings"][0].to(device),
            inputs["position_embeddings"][1].to(device),
        )

    q_inputs = None
    n_layers = len(model.model.layers)
    ignored_set = {i if i >= 0 else n_layers + i for i in ignored_layers if (i if i >= 0 else n_layers + i) < n_layers}

    for layer_idx in range(n_layers):
        print(f"\n{'='*80}")
        print(f"STE+LoRA — Processing Layer {layer_idx}")
        print(f"{'='*80}\n")

        fp_inputs = inputs
        fp_outputs: list[Tensor] = []

        layer = model.model.layers[layer_idx].to(device)

        # Compute FP reference outputs (store on CPU to avoid GPU accumulation)
        with torch.no_grad():
            for i in range(len(fp_inputs["hidden_states"])):
                batch_input = {"hidden_states": fp_inputs["hidden_states"][i].to(device)}
                for key in fp_inputs:
                    if key != "hidden_states":
                        batch_input[key] = fp_inputs[key]
                output = layer(**batch_input)
                fp_outputs.append(output.detach().cpu())

        # --- Ignored (FP-passthrough) layers ---
        if layer_idx in ignored_set:
            model.model.layers[layer_idx].to("cpu")
            # Propagate quantised signal through ignored layers
            if q_inputs is not None:
                new_q = []
                layer_dev = model.model.layers[layer_idx].to(device)
                with torch.no_grad():
                    for i in range(len(q_inputs)):
                        qi = {"hidden_states": q_inputs[i].to(device)}
                        for key in fp_inputs:
                            if key != "hidden_states":
                                qi[key] = fp_inputs[key]
                        new_q.append(layer_dev(**qi).detach().cpu())
                model.model.layers[layer_idx].to("cpu")
                del q_inputs
                q_inputs = new_q
            del fp_inputs["hidden_states"]
            fp_inputs["hidden_states"] = fp_outputs
            del fp_outputs
            cleanup()
            continue

        # --- Quantised training ---
        print(f"  Starting STE+LoRA fine-tuning for layer {layer_idx} ...")

        if q_inputs is not None:
            del fp_inputs["hidden_states"]
            fp_inputs["hidden_states"] = q_inputs
            cleanup()

        layer = wrap_model_block_ste(
            layer.to(device),
            n_bits=num_bits,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            layer_index=layer_idx,
            n_layers=n_layers,
            use_llama_cpp_scheme=True,
            group_size=group_size,
        )

        layer, q_inputs = finetune_layer_ste(
            layer=layer,
            fp_inputs=fp_inputs,
            fp_outputs=fp_outputs,
            layer_idx=layer_idx,
            lr=lr,
            lora_lr=lora_lr,
            epochs_per_layer=epochs_per_layer + min(layer_idx, 15),
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            tb=tb,
            return_next_layer_inputs=True,
            ste_temp_start=ste_temp_start,
            ste_temp_end=ste_temp_end,
            keep_data_on_cpu=keep_data_on_cpu,
        )

        # Move q_inputs to CPU to prevent GPU memory accumulation
        q_inputs = [t.detach().cpu() if t.is_cuda else t for t in q_inputs]

        model.model.layers[layer_idx] = layer.to("cpu")
        del fp_inputs["hidden_states"]
        fp_inputs["hidden_states"] = fp_outputs
        del fp_outputs
        cleanup()

    # Restore accelerate dispatch hooks
    for module, fwd in saved_accelerate_forwards.items():
        module._old_forward = module.forward
        module.forward = fwd

    if codebook_dst_dir is not None:
        save_codebook_layers(model, codebook_dst_dir)

    torch.compiler.reset()
    model = unwrap_model_ste(model)

    return model
