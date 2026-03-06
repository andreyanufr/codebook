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

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from rich.progress import track
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from codebook_wrapper import get_reciprocal, set_module
from layerwise_tuning import (
    cleanup,
    collate_fn,
    get_first_block_inputs,
    log_gradients_in_model,
    get_abs_top_percent_mask
)


from one_hot_uint8 import one_hot as one_hot_uint8_impl
from pack_unpack import pack_4bit, pack_2bit
from pack_unpack import unpack_4bit, unpack_2bit


from torch.utils.checkpoint import checkpoint as _checkpoint


# ---------------------------------------------------------------------------
# _ste_recompute_fn – pure function wrapped by torch.utils.checkpoint
# ---------------------------------------------------------------------------

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
        lora_rank: int = -1,
        lora_alpha: float = 32.0,
        use_exp_for_scale: bool = True,
        ste_temperature: float = 1.0,
        codebook_stage: bool = False,  # If True, only initialize codebook/scale/indexes without LoRA (for ablation)
    ):
        super().__init__()

        assert isinstance(orig_layer, nn.Linear), "Only nn.Linear layers are supported"
        assert orig_layer.bias is None, "Bias is not supported"

        self.orig_layer = orig_layer
        self.group_size = group_size
        self.n_bits = n_bits
        self.use_exp_for_scale = use_exp_for_scale

        # Mutable – the training loop adjusts this each epoch
        self.ste_temperature: float = ste_temperature
        # Controls whether forward uses STE (True) or hard-only (False)
        self.training_mode_ste: bool = True

        out_features, in_features = orig_layer.weight.shape

        # ---- Codebook ----
        if n_bits == 2:
            initial_codebook = torch.tensor(
                [-1.0, -0.25, 0.0, 1.0],
                #[-1.0, -0.25, 0.25, 1.0],
                dtype=orig_layer.weight.dtype,
                device=orig_layer.weight.device,
            )
        else:
            vals = list(range(-(2 ** (n_bits - 1)) + 1, 2 ** (n_bits - 1) + 1))
            initial_codebook = torch.tensor(
                vals,
                dtype=orig_layer.weight.dtype,
                device=orig_layer.weight.device,
            ) / (2 ** (n_bits - 1))

        self.codebook = nn.Parameter(initial_codebook, requires_grad=True)

        # ---- Scale ----
        self._init_indexes_and_scale()

        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_alpha = lora_alpha
            self.lora_A = nn.Parameter(
                torch.empty(lora_rank, in_features, dtype=orig_layer.weight.dtype, device=orig_layer.weight.device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, lora_rank, dtype=orig_layer.weight.dtype, device=orig_layer.weight.device)
            )
            nn.init.kaiming_uniform_(self.lora_A)
        else:
            self.lora = nn.Parameter(
                torch.zeros(out_features, in_features, dtype=orig_layer.weight.dtype, device=orig_layer.weight.device)
            )

        # Freeze original weight
        self.orig_layer.weight.requires_grad = False

        # Weight-space MSE init for codebook + scale
        self._mse_init()

        # ---- Initialise LoRA from SVD of quantization error ----
        #self._svd_lora_init()

        # Save VRAM – move original weight to CPU (pulled back during STE fwd)
        self.orig_layer.to("cpu")

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_indexes_and_scale(self):
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size)

        scale = weight.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5)
        if self.use_exp_for_scale:
            scale = torch.log(scale)

        self.scale = nn.Parameter(scale, requires_grad=True)

    #     self.update_indexes()

    # @torch.no_grad()
    # def update_indexes(self):
    #     """Re-assign each weight position to its nearest codebook entry."""
    #     normalized = self._get_normalized_weights(differentiable=False)
    #     codebook_norm = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
    #     self.indexes = torch.argmin(
    #         (normalized.unsqueeze(-1) - codebook_norm).abs(), dim=-1
    #     ).to(torch.uint8)
    
    
    def dequantize_by_distance(self, codebook, normalized, return_indexes = False):
        thresholds = (codebook[:-1] + codebook[1:]) * 0.5
        
        idx = torch.bucketize(normalized, thresholds)
        quantized = codebook[idx]
        
        if return_indexes:
            return idx
        
        if codebook.requires_grad:
            # train only codebook
            one_hot = F.one_hot(
                idx, num_classes=2 ** self.n_bits
            ).to(codebook.device, codebook.dtype)

            quantized = (one_hot * codebook).sum(dim=-1)
            return quantized

        return (quantized - normalized).detach() + normalized


    def _get_effective_weight(self):
        """Return ``orig_weight + lora_delta`` (2-D, on codebook device).

        If LoRA parameters have not been created yet (during __init__),
        returns just the original weight.
        """

        if not (hasattr(self, "lora") or hasattr(self, "lora_A")):
            return self.orig_layer.weight.data.detach()

        if self.lora_rank > 0:
            return self.orig_layer.weight.data.to(self.codebook.device) + (self.lora_B @ self.lora_A) * (self.lora_alpha / self.lora_rank)

        return self.orig_layer.weight.data.to(self.codebook.device) + self.lora


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

        if differentiable:
            if self.use_exp_for_scale:
                iscale = get_reciprocal(self.scale.exp())
            else:
                iscale = get_reciprocal(self.scale)
            return weight * iscale
        else:
            with torch.no_grad():
                if self.use_exp_for_scale:
                    iscale = get_reciprocal(self.scale.exp())
                else:
                    iscale = get_reciprocal(self.scale)
                return weight * iscale

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


    def _mse_init(self, n_iters: int = 200, lr: float = 0.01, index_update_interval: int = 25):
        device = self.codebook.device
        self.orig_layer.to(device)
        orig_weight = self.orig_layer.weight.data.to(device)

        out_features, in_features = orig_weight.shape
        orig_weight_grouped = orig_weight.view(
            out_features, in_features // self.group_size, self.group_size
        )

        optimizer = torch.optim.Adam([self.codebook, self.scale], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

        best_loss = float("inf")
        best_codebook = self.codebook.data.clone()
        best_scale = self.scale.data.clone()

        soft_iters = int(n_iters * 0.75)
        temp_start, temp_end = 0.5, 0.01

        for i in range(n_iters):
            optimizer.zero_grad()

            if i < soft_iters:
                temperature = temp_start + (temp_end - temp_start) * (i / max(soft_iters - 1, 1))
                codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
                if self.use_exp_for_scale:
                    scale = self.scale.exp()
                else:
                    scale = self.scale
                iscale = get_reciprocal(scale)
                normalized = orig_weight_grouped * iscale
                neg_dist_sq = -(normalized.unsqueeze(-1) - codebook).pow(2) / temperature
                soft_assignment = F.softmax(neg_dist_sq, dim=-1)
                w = (codebook * soft_assignment).sum(dim=-1)
                deq_weight = (w * scale).view(out_features, in_features)
            else:
                deq_weight = self._dequantize_hard()

            loss = F.mse_loss(deq_weight, orig_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.codebook, self.scale], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_codebook = self.codebook.data.clone()
                    best_scale = self.scale.data.clone()

        with torch.no_grad():
            self.codebook.data.copy_(best_codebook)
            self.scale.data.copy_(best_scale)

        optimizer.zero_grad()
        cleanup()

        self.orig_layer.weight.data = self.orig_layer.weight.data.to("cpu")
        self.orig_layer.to("cpu")
        return best_loss


    # ------------------------------------------------------------------
    # Dequantisation variants
    # ------------------------------------------------------------------

    def _dequantize_hard(self):
        """Standard hard dequantisation (one-hot from stored indexes)."""
        codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
        
        normalized = self._get_normalized_weights(differentiable=False)
        
        weight = self.dequantize_by_distance(codebook, normalized)
        
        weight = weight * (self.scale.exp() if self.use_exp_for_scale else self.scale)

        out_features, in_features = self.orig_layer.weight.shape
        return weight.view(out_features, in_features)

    def _dequantize_ste(self):
        """STE dequantisation: forward = hard assignment, backward = soft.

        Uses ``orig_weight + lora_delta`` as the reference weight so that the
        codebook, scale, and soft-assignment gradients all account for the LoRA
        correction.  This means the codebook naturally learns to represent the
        corrected weight, making LoRA merging virtually free.

        .. note::
           Assumes ``self.orig_layer`` is already on the same device as
           ``self.codebook`` (the training function handles this).
        """
        out_features, in_features = self.orig_layer.weight.shape

        codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)

        normalized = self._get_normalized_weights(differentiable=True)
        
        weight = self.dequantize_by_distance(codebook, normalized)

        weight = weight * (self.scale.exp() if self.use_exp_for_scale else self.scale)

        return weight.view(out_features, in_features)


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
        
        if self.lora_rank > 0:
            lora_delta = (self.lora_B @ self.lora_A) * (self.lora_alpha / self.lora_rank)
        else:
            lora_delta = self.lora
        self.orig_layer.weight.data = (
            self.orig_layer.weight.data.to(device) + lora_delta
        ).to(self.orig_layer.weight.dtype)

        # 2. Zero out LoRA so _get_effective_weight() == orig_weight
        if self.lora_rank > 0:
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
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
    def get_compressed_indexes_and_normalized_codebook(self):
        codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
        normalized = self._get_normalized_weights(differentiable=False)
        indexes = self.dequantize_by_distance(codebook, normalized, return_indexes=True)

        indexes = indexes.to(torch.uint8)
        if self.n_bits == 2:
            packed = pack_2bit(indexes)
        elif self.n_bits == 4:
            packed = pack_4bit(indexes)
        else:
            raise ValueError("Unsupported n_bits for packing indexes")
        return packed, codebook


    @torch.no_grad()
    def get_state_dict(self):
        """Return a state dict containing just the codebook, scale, and indexes."""
        codebook, indexes = self.get_compressed_indexes_and_normalized_codebook()
        return {
            "codebook": codebook.cpu(),
            "scale": (self.scale.exp() if self.use_exp_for_scale else self.scale).data.cpu(),
            "shape": self.orig_layer.weight.shape,
            "indexes": indexes.cpu(),
        }


def save_codebook_layers(model: nn.Module, output_dir: Path):
    """
    Saves the codebook layers of the model to the specified output directory.

    :param model: The model containing the codebook layers to be saved.
    :param output_dir: The directory where the codebook layers will be saved.
    """
    codebook_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, CodebookLoRASTELinear):
            codebook_state_dict[name] = module.get_state_dict()

    torch.save(codebook_state_dict, output_dir / "codebook_layers.pth")
    print(f"Codebook layers saved to {output_dir / 'codebook_layers.pth'}")


def dequantize_from_dict(state_dict: dict, device: torch.device) -> Tensor:
    # dequantize weight from codebook, scale, and indexes in state_dict
    # {
    #     "codebook": self.codebook.data.cpu(),
    #     "scale": self.scale.data.cpu(),
    #     "shape": self.orig_layer.weight.shape,
    #     "indexes": self.get_compressed_indexes().cpu(),
    # }
    codebook = state_dict["codebook"].to(device)
    scale = state_dict["scale"].to(device)
    shape = state_dict["shape"]
    indexes = state_dict["indexes"].to(device)
    
    
    if indexes.dtype == torch.uint8 and codebook.shape[0] == 16:
        indexes = unpack_4bit(indexes)
    elif indexes.dtype == torch.uint8 and codebook.shape[0] == 4:
        indexes = unpack_2bit(indexes)

    weight = codebook[indexes.flatten().long()].reshape(indexes.shape)
    weight = weight * scale

    out_features, in_features = shape
    return weight.view(out_features, in_features)

    # scale = scale.exp()
    
    # out_features, in_features = shape
    
    # # normalize codebook
    # codebook = codebook / codebook.abs().max().clamp(min=1e-8)
    
    # # unpack indexes if needed
    # if indexes.dtype == torch.uint8 and codebook.shape[0] == 16:
    #     indexes = unpack_4bit(indexes)
    # elif indexes.dtype == torch.uint8 and codebook.shape[0] == 4:
    #     indexes = unpack_2bit(indexes)
    
    # w = (codebook[indexes.flatten().long()].reshape(indexes.shape)) * scale
    # return w.view(out_features, in_features)
    
    

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
            if isinstance(module, nn.Linear) and ("v_proj" in name or ("down_proj" in name and layer_index < 1000000 *  n_layers / 8)):
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
        print(f"  Wrapping {name} with CodebookLoRASTELinear 2bit (rank={-1}, group_size={group_size})")
        set_module(
            block,
            name,
            CodebookLoRASTELinear(
                module,
                n_bits=n_bits,
                lora_rank=-1,
                lora_alpha=lora_alpha,
                group_size=group_size,
            ),
        )
    cleanup()
    
    for name, module in changed_modules_4_bit.items():
        print(f"  Wrapping {name} with CodebookLoRASTELinear 4bit (rank={lora_rank}, group_size={2 * group_size})")
        set_module(
            block,
            name,
            CodebookLoRASTELinear(
                module,
                n_bits=2 * n_bits,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                group_size=2 * group_size,
            ),
        )
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
    index_update_epochs: int = 4,
    keep_data_on_cpu: bool = True,
    warm_up_codebook_epochs: int = 4,
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
    index_update_epochs : int
        Number of initial epochs during which indexes are refreshed after every
        optimiser step.
    """
    if lora_lr is None:
        lora_lr = lr

    # ------------------------------------------------------------------
    # Categorise trainable parameters
    # ------------------------------------------------------------------
    codebooks: list[nn.Parameter] = []
    scales: list[nn.Parameter] = []
    lora_params: list[nn.Parameter] = []
    
    # lead to NaN in one A100 GPU
    if torch.cuda.device_count() > 1:
        layer = torch.compile(layer)

    for name, param in layer.named_parameters():
        if "codebook" in name:
            param.requires_grad = True
            codebooks.append(param)
        elif "scale" in name:
            param.requires_grad = True
            scales.append(param)
        elif "lora" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    param_groups = [
        {"params": codebooks, "lr": lr, "label": "codebook"},
        {"params": scales, "lr": lr, "label": "scale"},
        {"params": lora_params, "lr": 0.1 * lora_lr, "label": "lora"},
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
    
    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(fp_inputs["hidden_states"])
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size
    total_opt_steps = epochs_per_layer * epoch_samples // batch_size

    if warm_up_codebook_epochs > 0:
        total_wup_opt_steps = warm_up_codebook_epochs * epoch_samples // batch_size
        
        print(f"  Warming up codebook parameters for {warm_up_codebook_epochs} epoch(s) with STE temperature {ste_temp_start}...")
        opt_warmup = torch.optim.AdamW(
            [p for g in param_groups for p in g["params"] if g["label"] == "codebook"],
            lr=lr,
            weight_decay=0.01,
        )
        for p in lora_params:
            p.requires_grad = False
        for p in scales:
            p.requires_grad = False

        
        for m in ste_modules:
            m.ste_temperature = ste_temp_start
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_warmup, eta_min=lr * 1e-4, T_max=total_wup_opt_steps)
        
        global_step = 0
        
        for _ in range(warm_up_codebook_epochs):
            batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
            grad_steps = 0
            loss_numerator = 0.0
            loss_denominator = 0.0
        

            for indices in track(
                batch_indices_epoch,
                description=f"    Warm-up epoch",
            ):
                indices = indices.tolist()
                hidden, kwargs = collate_fn(fp_inputs, indices, device=device)
                hidden = hidden.to(device)
                layer_outputs = layer(hidden, **kwargs)
                orig_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(device)
                loss = F.mse_loss(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))
                
                loss_numerator += loss.detach().item()
                loss_denominator += torch.mean(orig_output ** 2).detach().item()

                (loss / grad_accumulation_steps).backward()
                grad_steps += 1
                
                if grad_steps == grad_accumulation_steps:
                    for group in param_groups:
                        torch.nn.utils.clip_grad_norm_(group["params"], 1.0)
                    grad_steps = 0
                    opt_warmup.step()
                    scheduler.step()
                    
                    rel_loss = loss_numerator / max(loss_denominator, 1e-8)
                    agg_loss = loss_numerator / grad_accumulation_steps
                    loss_numerator = loss_denominator = grad_steps = 0

                    if tb is not None:
                        tb.add_scalar(f"warmup_rel_loss/layer_{layer_idx}", rel_loss, global_step)
                        tb.add_scalar(f"warmup_ste_loss/layer_{layer_idx}", agg_loss, global_step)
                        log_gradients_in_model(layer, tb, -global_step, layer_idx)
                    global_step += 1
                    opt_warmup.zero_grad()

        opt_warmup.zero_grad()
        del opt_warmup
        for p in lora_params:
            p.requires_grad = True
        for p in scales:
            p.requires_grad = True
        for p in codebooks:
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Training bookkeeping
    # ------------------------------------------------------------------
    # train all parameters except codebook (which is frozen after warm-up)
    param_groups = [g for g in param_groups if g["label"] != "codebook"]
    opt = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # LR schedule: constant during index-update phase, then linear decay
    warmup_steps = index_update_epochs * epoch_samples // batch_size

    def lr_lambda(step):
        if step < warmup_steps:
            return 1.0
        remaining = total_opt_steps - warmup_steps
        if remaining <= 0:
            return 1.0
        return max(1e-4, 1.0 - (step - warmup_steps) / remaining)

    #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=lr * 1e-4, T_max=total_opt_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0

    # compute graient clipping exponent based on total steps and desired final max norm
    final_max_norm = 0.1
    start_max_norm = 1.0
    max_norm_decay = (final_max_norm / start_max_norm) ** (1 / max(total_opt_steps - 1, 1))
    
    codebook_gradient_max_value = 0.01
    
    for epoch in range(epochs_per_layer):
        opt.zero_grad()
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_loss = 0.0
        num_batches = 0
        loss_numerator = grad_steps = 0
        loss_denominator = 0.0
        max_norm = start_max_norm * (max_norm_decay ** epoch)
        

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
                raise ValueError(
                    f"Non-finite loss ({loss.item()}) at layer {layer_idx}, step {global_step}"
                )

            loss_numerator += loss.item()
            loss_denominator += torch.mean(orig_output ** 2).detach().item()
            grad_steps += 1

            (loss / grad_accumulation_steps).backward()

            del hidden, layer_outputs, orig_output, loss

            if grad_steps == grad_accumulation_steps:
                for group in param_groups:
                    torch.nn.utils.clip_grad_norm_(group["params"], max_norm)
                
                for group in param_groups:
                    if group["label"] == "codebook":
                        torch.nn.utils.clip_grad_value_(group["params"], codebook_gradient_max_value)

                opt.step()
                scheduler.step()

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

    # ------------------------------------------------------------------
    # Post-training: merge LoRA → requantize, switch to hard mode
    # ------------------------------------------------------------------
    print(f"  Merging LoRA into codebook representation ...")
    for m in ste_modules:
        m.training_mode_ste = False
        m.check_hard_and_ste_consistency()
        m.merge_lora()

    del opt, scheduler
    cleanup()
    
    for p in lora_params:
        p.requires_grad = False
    for p in scales:
        p.requires_grad = False
    for p in codebooks:
        p.requires_grad = False

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

        return layer, next_inputs

    return layer


# ---------------------------------------------------------------------------
# Full model layer-wise orchestration
# ---------------------------------------------------------------------------

def finetune_layerwise_ste(
    model: nn.Module,
    tokenizer,
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
    """
    if ignored_layers is None:
        ignored_layers = [] #[0, 1, 2, 3, -1, -2, -3, -4]

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    # Cache first-block inputs via a single FP forward pass
    inputs = get_first_block_inputs(model, dataset=train_loader)

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
            n_bits=2,
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
            epochs_per_layer=epochs_per_layer + min(layer_idx, 15),  # optional: increase epochs for deeper layers
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            tb=tb,
            return_next_layer_inputs=True,
            ste_temp_start=ste_temp_start,
            ste_temp_end=ste_temp_end,
            index_update_epochs=index_update_epochs,
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
