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

from codebook_wrapper import CodebookWrapperLinear, wrap_model, unwrap_model, wrap_model_block



def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

def print_tensor_memory_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.dtype, obj.device, obj.shape)
        except:
            pass


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
def dump_block_inputs(model: nn.Module, dir: str, dataset: list[Tensor]):
    model.eval()
    
    if not os.path.exists(dir):
        os.makedirs(dir)

    names = {}
    
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = BlockInputCacher(layer, name=f"layer_{i}")

    with torch.no_grad():
        for batch in track(dataset, description="Caching block inputs..."):
            model(batch)
    
    # After running through the dataset, dump the cached inputs for each block
    for name, module in model.named_modules():
        if isinstance(module, BlockInputCacher):
            names[name] = module.dump_cached_inputs(dir)

    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = layer.block  # Unwrap the original block to restore model functionality

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


def load_cached_inputs(names: list[str]) -> dict[str, Tensor]:
    inputs = {}
    for name in names:
        key = name.split("/")[-1].rsplit(".", 1)[0]  # Extract "layer_{i}_{key}"
        inputs[key] = torch.load(name)
    return inputs


def finetune_layerwise(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    train_loader: list[Tensor],
    lr: float = 1e-4,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = "cuda:0",
    tb: Optional[SummaryWriter] = None,
    ignored_layers: Optional[list[int]] = [0, 1, 2, 3, -1, -2, -3, -4] #0, 1, 2, 3, 4, 5, 6,  -1, -2, -3, -4, -5, -6, -7]
) -> nn.Module:
    # Normalize device to a concrete torch.device (e.g. "cuda" -> "cuda:0")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    # dump block inputs for each layer
    inputs = get_first_block_inputs(model, dataset=train_loader)

    model.to('cpu')
    torch.cuda.empty_cache()
    
    # Strip accelerate dispatch hooks from all layers (and their submodules)
    # before processing.  accelerate replaces module.forward with a hooked
    # version and saves the original as module._old_forward.  The hooks move
    # tensors between devices which conflicts with our explicit device management.
    saved_accelerate_forwards: dict[nn.Module, object] = {}
    for module in model.modules():
        if hasattr(module, "_old_forward"):
            saved_accelerate_forwards[module] = module.forward
            module.forward = module._old_forward
            del module._old_forward
    
    # Move non-hidden-state inputs (e.g. position_embeddings) to device once
    if "position_embeddings" in inputs:
        inputs["position_embeddings"] = (
            inputs["position_embeddings"][0].to(device),
            inputs["position_embeddings"][1].to(device),
        )

    # For each layer, load the cached inputs and fine-tune that layer
    q_inputs = None
    
    n_layers = len(model.model.layers)
    ignored_layers = [i if i >= 0 else n_layers + i for i in ignored_layers if i < n_layers]

    for layer_idx in range(len(model.model.layers)):
        print(f"\n{'='*80}")
        print(f"Fine-tuning Layer {layer_idx}")
        print(f"{'='*80}\n")
        
        fp_inputs = inputs
        fp_outputs = []
        
        
        layer = model.model.layers[layer_idx].to(device)

        with torch.no_grad():
            for i in range(len(fp_inputs["hidden_states"])):
                batch_input = {}
                batch_input["hidden_states"] = fp_inputs["hidden_states"][i].to(device)
                for key in fp_inputs:
                    if key != "hidden_states":
                        batch_input[key] = fp_inputs[key]

                output = layer(**batch_input)
                fp_outputs.append(output.detach())
        
        if layer_idx in ignored_layers:
            model.model.layers[layer_idx].to('cpu')#= layer.to('cpu')
            del fp_inputs["hidden_states"]
            fp_inputs["hidden_states"] = fp_outputs
            cleanup()  # Clear cache after each layer
            continue

        print(f"  Starting fine-tuning for layer {layer_idx}...")

        if q_inputs is not None:
            del fp_inputs["hidden_states"]
            fp_inputs["hidden_states"] = q_inputs  # Use quantized inputs for subsequent layers after the first one
            cleanup()

        layer = wrap_model_block(layer.to(device), n_bits=2, layer_index=layer_idx, n_layers=len(model.model.layers), use_llama_cpp_scheme=True)  # Wrap the layer with codebook wrapper for fine-tuning
        layer, q_inputs = finetune_layer_l2(
            layer=layer,
            fp_inputs=fp_inputs,
            fp_outputs=fp_outputs,
            layer_idx=layer_idx,
            lr=lr,
            epochs_per_layer=epochs_per_layer,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            tb=tb,
            return_next_layer_inputs=True
        )
        model.model.layers[layer_idx] = layer.to('cpu')  # Move back to CPU after fine-tuning this layer
        del fp_inputs["hidden_states"]
        fp_inputs["hidden_states"] = fp_outputs
        cleanup()  # Clear cache after each layer

    # Restore accelerate dispatch hooks on all modules that had them
    for module, fwd in saved_accelerate_forwards.items():
        module._old_forward = module.forward
        module.forward = fwd


def collate_fn(data, indexes):
    hidden_states = torch.cat([data["hidden_states"][i] for i in indexes], dim=0)

    kwargs = {}
    for key in data:
        if key != "hidden_states":
            val = data[key]
            # Move tensor tuples (e.g. position_embeddings) and tensors to the
            # same device as hidden_states to avoid cross-device errors.
            if isinstance(val, (tuple, list)) and all(isinstance(v, Tensor) for v in val):
                val = type(val)(v.to(hidden_states.device) for v in val)
            elif isinstance(val, Tensor):
                val = val.to(hidden_states.device)
            kwargs[key] = val
    return hidden_states, kwargs


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


def finetune_layer_l2(
    layer: nn.Module,
    fp_inputs: list[Tensor],
    fp_outputs: list[Tensor],
    layer_idx: int = -1,
    lr: float = 1e-4,
    epochs_per_layer: int = 10,
    batch_size: int = 64,
    microbatch_size: int = 8,
    device: torch.device = "cuda",
    tb: Optional[SummaryWriter] = None,
    return_next_layer_inputs: bool = False,
    percent_of_warmup_soft_epochs: float = 0.0
) -> nn.Module:
    """
    Fine-tunes a compressed model layer-wise using L2 loss between outputs of original and compressed models.
    
    :param layer: The layer to be fine-tuned
    :param fp_inputs: List of input tensors for the layer
    :param fp_outputs: List of output tensors for the layer
    :param lr: Learning rate for optimization
    :param epochs_per_layer: Number of epochs to train each layer
    :param batch_size: Total batch size for training
    :param microbatch_size: Size of each microbatch for gradient accumulation
    :param device: Device to run training on
    :param tb: Optional TensorBoard SummaryWriter for logging
    :param return_next_layer_inputs: Whether to return the inputs for the next layer
    :return: The fine-tuned compressed model
    """
        
    # Set up parameters to train for this layer only
    param_to_train = []
    codebooks_to_train = []
    scales_to_train = []
    for name, param in layer.named_parameters():
        if "codebook" in name:
            param.requires_grad = True
            codebooks_to_train.append(param)
        elif "scale" in name:
            param.requires_grad = True
            scales_to_train.append(param)
        else:
            param.requires_grad = False
    
    param_to_train.extend(codebooks_to_train) #[{"params": codebooks_to_train, "lr": lr}, {"params": scales_to_train, "lr": 10 * lr}]
    param_to_train.extend(scales_to_train)
    
    if not param_to_train:
        print(f"WARNING: No trainable parameters found in layer {layer_idx}, skipping fine-tuning for this layer.")
        if return_next_layer_inputs:
            num_samples = len(fp_inputs["hidden_states"])
            next_inputs = []
            with torch.no_grad():
                for i in range(num_samples):
                    hidden, kwargs = collate_fn(fp_inputs, [i])
                    next_inputs.append(layer(hidden.to(device), **kwargs))
            return layer, next_inputs
        return layer

    # Create optimizer for this layer
    # lambda_lr = lambda epoch: 0.99 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)
    
    
    # Training loop for this layer
    grad_accumulation_steps = batch_size // microbatch_size
    num_samples = len(fp_inputs["hidden_states"])
    epoch_samples = num_samples - num_samples % microbatch_size
    microbatches_per_epoch = epoch_samples // microbatch_size
    loss_comput = nn.SmoothL1Loss(beta=10.0)

    global_step = 0
    
    soft_epochs = int(percent_of_warmup_soft_epochs * epochs_per_layer)
    if soft_epochs > 0:
        for name, sub_layer in layer.named_modules():
            if isinstance(sub_layer, CodebookWrapperLinear):
                sub_layer.use_soft_forward = True

        opt = torch.optim.AdamW(param_to_train, weight_decay=0.0, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=lr * 1e-4, T_max=soft_epochs * epoch_samples // batch_size)
        for epoch in range(soft_epochs):
            opt.zero_grad()  # Clear any stale gradients from prior epoch's incomplete accumulation
            batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
            epoch_loss = 0.0
            num_batches = 0
            loss_numerator = grad_steps = 0
            loss_denuminator = 0

            for indices in track(
                batch_indices_epoch, 
                description=f"  Layer {layer_idx}, Epoch {epoch}/{soft_epochs}"
            ):
                indices = indices.tolist()
                
                # Form batch
                hidden, kwargs = collate_fn(fp_inputs, indices)
                hidden = hidden.to(device)

                layer_outputs = layer(hidden, **kwargs)#[0]
                #orig_output = torch.stack([fp_outputs[i] for i in indices], dim=0).to(device)
                orig_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(device)
                
                # Compute L2 loss between outputs
                loss = loss_comput(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))
                
                if not torch.isfinite(loss).item():
                    err = f"Fine-tuning loss is {loss} at layer {layer_idx}"
                    raise ValueError(err)
                
                # Gradient accumulation
                loss_numerator += loss.item()
                loss_denuminator += torch.mean(orig_output**2).detach().item()
                grad_steps += 1
                
                (loss / grad_accumulation_steps).backward()

                if grad_steps == grad_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(param_to_train, 0.1)
                        
                    opt.step()
                    scheduler.step()
                    
                    rel_aggregated_loss = (loss_numerator /  max(loss_denuminator, 1e-8)) #/  grad_steps
                    aggregated_loss = loss_numerator / grad_steps
                    epoch_loss += aggregated_loss
                    num_batches += 1
                    loss_numerator = loss_denuminator = grad_steps = 0
                    
                    if tb is not None:
                        tb.add_scalar(f"layerwise_relative_soft_loss/layer_{layer_idx}", rel_aggregated_loss, global_step)
                        tb.add_scalar(f"layerwise_soft_loss/layer_{layer_idx}", aggregated_loss, global_step)
                        # tb.add_scalar(f"layerwise_lr/layer_{layer_idx}", opt.param_groups[0]["lr"], global_step)
                        # log_gradients_in_model(layer, tb, global_step, layer_idx)
                    
                    opt.zero_grad()
                    
                    global_step += 1
        
        
        for name, sub_layer in layer.named_modules():
            if isinstance(sub_layer, CodebookWrapperLinear):
                sub_layer.use_soft_forward = False
                sub_layer.update_indexes()  # Update indexes after soft fine-tuning to prepare for hard fine-tuning

    
    opt = torch.optim.AdamW(param_to_train, weight_decay=0.0, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=lr * 1e-4, T_max=epochs_per_layer * epoch_samples // batch_size)
    
    updtable_epochs = max(int(0.05 * epochs_per_layer), 1)

    for epoch in range(epochs_per_layer):
        opt.zero_grad()  # Clear any stale gradients from prior epoch's incomplete accumulation
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        epoch_loss = 0.0
        num_batches = 0
        loss_numerator = grad_steps = 0
        loss_denuminator = 0

        # if epoch < updtable_epochs:
        #     for name, sub_layer in layer.named_modules():
        #         if isinstance(sub_layer, CodebookWrapperLinear):
        #             sub_layer.update_indexes()

        for indices in track(
            batch_indices_epoch, 
            description=f"  Layer {layer_idx}, Epoch {epoch}/{epochs_per_layer}"
        ):
            indices = indices.tolist()
            
            # Form batch
            hidden, kwargs = collate_fn(fp_inputs, indices)
            hidden = hidden.to(device)

            layer_outputs = layer(hidden, **kwargs)#[0]
            #orig_output = torch.stack([fp_outputs[i] for i in indices], dim=0).to(device)
            orig_output = torch.cat([fp_outputs[i] for i in indices], dim=0).to(device)
            
            # Compute L2 loss between outputs
            loss = F.mse_loss(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))
            #loss = loss_comput(layer_outputs, orig_output.to(dtype=layer_outputs.dtype))
            
            # _, mask = get_abs_top_percent_mask(torch.abs(layer_outputs - orig_output))  # This will update the mask used in the forward pass of CodebookWrapperLinear for the next iteration
            # loss = torch.mean(((layer_outputs - orig_output.to(dtype=layer_outputs.dtype)) * mask)**2)
            
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss} at layer {layer_idx}"
                raise ValueError(err)
            
            # Gradient accumulation
            loss_numerator += loss.item()
            loss_denuminator += torch.mean(orig_output**2).detach().item()
            grad_steps += 1
            
            (loss / grad_accumulation_steps).backward()

            if grad_steps == grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(param_to_train, 0.1)
                    
                opt.step()
                scheduler.step()
                
                rel_aggregated_loss = (loss_numerator /  max(loss_denuminator, 1e-8)) #/  grad_steps
                aggregated_loss = loss_numerator / grad_steps
                epoch_loss += aggregated_loss
                num_batches += 1
                loss_numerator = loss_denuminator = grad_steps = 0
                
                if tb is not None:
                    tb.add_scalar(f"layerwise_relative_loss/layer_{layer_idx}", rel_aggregated_loss, global_step)
                    tb.add_scalar(f"layerwise_loss/layer_{layer_idx}", aggregated_loss, global_step)
                    tb.add_scalar(f"layerwise_lr/layer_{layer_idx}", opt.param_groups[0]["lr"], global_step)
                    log_gradients_in_model(layer, tb, global_step, layer_idx)
                
                opt.zero_grad()
                
                global_step += 1

        #scheduler.step()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"    Epoch {epoch}: avg loss = {avg_epoch_loss:.6f}")
    
    del opt, scheduler
    cleanup()
    # Clear cache after each layer
    torch.cuda.empty_cache()
    print(f"\n{'='*80}")
    print("Layer-wise fine-tuning complete!")
    print(f"{'='*80}\n")

    if return_next_layer_inputs:
        next_inputs = []
        with torch.no_grad():
            for i in range(num_samples):
                hidden, kwargs = collate_fn(fp_inputs, [i])
                next_input = layer(hidden, **kwargs)#.cpu()
                next_inputs.append(next_input)
        return layer, next_inputs

    return layer



def example_layerwise_training():
    """
    Example demonstrating how to use the layer-wise fine-tuning function independently.
    
    This example shows:
    1. Loading original and compressed models
    2. Preparing training data
    3. Running layer-wise fine-tuning with L2 loss
    4. Saving the fine-tuned model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16
    
    # Load models
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    orig_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    compressed_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Wrap compressed model with codebook layers
    compressed_model = wrap_model(compressed_model, n_bits=2)
    
    # Prepare training data
    train_loader = get_compression_calibration(
        num_samples=128, 
        seqlen=512, 
        tokenizer=tokenizer, 
        device=device
    )
    
    # Run layer-wise fine-tuning
    compressed_model = finetune_layerwise_l2(
        orig_model=orig_model,
        compressed_model=compressed_model,
        train_loader=train_loader,
        lr=1e-3,
        epochs_per_layer=5,
        batch_size=32,
        microbatch_size=4,
        device=device,
        tb=None,  # Can pass SummaryWriter for logging
        layer_type="both"  # Train both MLP and attention layers
    )
    
    # Unwrap and save the model
    compressed_model = unwrap_model(compressed_model)
    compressed_model.save_pretrained("./layerwise_finetuned_model")
    tokenizer.save_pretrained("./layerwise_finetuned_model")
    
    print("Layer-wise fine-tuned model saved successfully!")
