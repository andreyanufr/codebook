import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans


def get_reciprocal(tensor):
    """Memory-efficient reciprocal: zero for near-zero elements."""
    eps = 1e-5 if tensor.dtype == torch.float16 else 1e-30
    mask = tensor.abs() < eps
    safe = tensor.masked_fill(mask, 1.0)
    return (1.0 / safe).masked_fill_(mask, 0.0)


class CodebookWrapperLinear(torch.nn.Module):
    def __init__(
        self,
        orig_layer,
        group_size: int = 32,
        signed_scale: bool = False,
        n_bits: int = 2,
        use_exp_for_scale: bool = True,
        use_soft_forward: bool = False
    ):
        super().__init__()
        
        assert isinstance(orig_layer, torch.nn.Linear), "Only linear layers are supported"
        assert orig_layer.bias is None, "Bias is not supported in this example"

        self.orig_layer = orig_layer
        self.group_size = group_size
        self.signed_scale = signed_scale
        self.n_bits = n_bits
        self.use_exp_for_scale = use_exp_for_scale
        self.use_soft_forward = use_soft_forward

        if n_bits == 2:
            initial_codebook = torch.tensor([-1.0, -0.25,  0.25, 1.0], dtype=orig_layer.weight.dtype).to(orig_layer.weight.device)
        else:
            initial_codebook = torch.tensor([i for i in range(-2 ** (n_bits - 1) + 1, 2 ** (n_bits - 1) + 1)], dtype=orig_layer.weight.dtype).to(orig_layer.weight.device) / (2 ** (n_bits - 1))
        
        
        self.codebook = torch.nn.Parameter(initial_codebook, requires_grad=True)

        self.init_indexes_and_scale()
        
        self.orig_layer.weight.requires_grad = False  # Freeze original weight, only update codebook and scale
        self.orig_layer.to('cpu')
        self.orig_layer.weight.to('cpu')

        self.mse_init()
        #self.k_means_init()

    
    @torch.no_grad()
    def init_indexes_and_scale(self):
        # reshape weight to (out_features, in_features // group_size, group_size)
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size)
        
        # calculate scale and indexes
        scale = weight.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5)
        if self.use_exp_for_scale:
            scale = torch.log(scale)

        self.scale = torch.nn.Parameter(scale, requires_grad=True)
        
        self.update_indexes()
    
    
    @torch.no_grad()
    def get_normalized_weights(self):
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size).to(self.codebook.device)

        if self.use_exp_for_scale:
            iscale = get_reciprocal(self.scale.exp())
        else:
            iscale = get_reciprocal(self.scale)

        normalized = weight * iscale
        return normalized

    @torch.no_grad()
    def update_indexes(self):
        weight = self.orig_layer.weight.data
        out_features, in_features = weight.shape
        weight = weight.view(out_features, in_features // self.group_size, self.group_size).to(self.codebook.device)

        if self.use_exp_for_scale:
            iscale = get_reciprocal(self.scale.exp())
        else:
            iscale = get_reciprocal(self.scale)

        # Normalize weight by scale, find nearest normalized codebook entry
        # (equivalent to original but consistent with dequantize_weight normalization)
        normalized = weight * iscale
        codebook_norm = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
        self.indexes = torch.argmin(
            (normalized.unsqueeze(-1) - codebook_norm).abs(), dim=-1
        ).to(torch.uint8)
    
    
    def k_means_init(self, n_iters: int = 10):
        """Initialize codebook using k-means clustering on the original weights."""
        weight = self.get_normalized_weights().float().cpu().numpy()
        flat_weight = weight.flatten().reshape(-1, 1)  # KMeans expects 2D input

        kmeans = KMeans(n_clusters=2 ** self.n_bits, n_init=10, max_iter=n_iters, init=self.codebook.data.float().cpu().numpy().reshape(-1, 1))
        kmeans.fit(flat_weight)

        with torch.no_grad():
            self.codebook.copy_(torch.from_numpy(kmeans.cluster_centers_.flatten()).to(self.codebook.device).to(self.codebook.dtype))

        self.update_indexes()


    def mse_init(self, n_iters: int = 200, lr: float = 0.01, index_update_interval: int = 25):
        """
        Learn codebook and scale by minimizing MSE between dequantized and original weights.

        Uses differentiable soft assignment for the first 75% of iterations (temperature
        annealed from warm→cold), then switches to hard assignment for final fine-tuning.
        This allows the optimizer to smoothly adjust codebook-to-weight assignments.

        :param n_iters: Number of optimization iterations.
        :param lr: Learning rate for the optimizer.
        :param index_update_interval: How often (in iterations) to reassign indexes.
        """
        device = self.codebook.device
        orig_weight = self.orig_layer.weight.data.to(device)

        self.orig_layer.to(device)

        out_features, in_features = orig_weight.shape
        orig_weight_grouped = orig_weight.view(
            out_features, in_features // self.group_size, self.group_size
        )

        optimizer = torch.optim.Adam([self.codebook, self.scale], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

        best_loss = float("inf")
        best_codebook = self.codebook.data.clone()
        best_scale = self.scale.data.clone()
        best_indexes = self.indexes.clone()

        # Soft → hard transition: use differentiable soft assignment for first 75%
        soft_iters = int(n_iters * 0.75)
        temp_start, temp_end = 0.5, 0.01

        for i in range(n_iters):
            optimizer.zero_grad()

            if i < soft_iters:
                # Temperature annealing: start soft (exploratory), end nearly hard
                temperature = temp_start + (temp_end - temp_start) * (i / max(soft_iters - 1, 1))
                deq_weight = self._dequantize_soft(orig_weight_grouped, temperature)
                deq_weight = deq_weight.view(out_features, in_features)
            else:
                deq_weight = self.dequantize_weight()

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
                    best_indexes = self.indexes.clone()
            if (i + 1) % index_update_interval == 0:
                self.update_indexes()

        # Restore best parameters
        with torch.no_grad():
            self.codebook.data.copy_(best_codebook)
            self.scale.data.copy_(best_scale)
            self.indexes = best_indexes

        self.orig_layer.weight.data = self.orig_layer.weight.data.to("cpu")
        self.orig_layer.to("cpu")
        return best_loss

    def _dequantize_soft(self, orig_weight_grouped, temperature: float = 0.1):
        """
        Differentiable dequantization with soft nearest-neighbor assignment.
        Allows gradients to flow through the codebook-to-weight assignment
        (not just through codebook values), enabling smoother optimization.

        :param orig_weight_grouped: Original weight as (out, n_groups, group_size), on device.
        :param temperature: Softmax temperature — lower = harder assignment.
        """
        codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)

        if self.use_exp_for_scale:
            scale = self.scale.exp()
        else:
            scale = self.scale

        iscale = get_reciprocal(scale)
        normalized = orig_weight_grouped * iscale

        # Soft assignment via negative squared distance
        neg_dist_sq = -(normalized.unsqueeze(-1) - codebook).pow(2) / temperature
        soft_assignment = F.softmax(neg_dist_sq, dim=-1)

        weight = (codebook * soft_assignment).sum(dim=-1)
        return weight * scale

    def dequantize_weight(self, memory_saving=False):
        """Dequantize using index_select — memory-efficient and differentiable w.r.t. codebook."""
        codebook = self.codebook / self.codebook.abs().max().clamp(min=1e-8)
        
        if memory_saving:
            flat_indexes = self.indexes.reshape(-1).long()
            weight = codebook.index_select(0, flat_indexes).view_as(self.indexes)
        else:
            one_hot = F.one_hot(self.indexes.long(), num_classes=2 ** self.n_bits).to(self.codebook.device).to(self.codebook.dtype)
            weight = (codebook * one_hot).sum(dim=3)

        if self.use_exp_for_scale:
            weight = weight * self.scale.exp()
        else:
            weight = weight * self.scale

        out_features, in_features = self.orig_layer.weight.shape
        return weight.view(out_features, in_features)
        
            
    def forward(self, x):
        if self.use_soft_forward:
            device = self.codebook.device
            orig_weight = self.orig_layer.weight.data.to(device)

            out_features, in_features = orig_weight.shape
            orig_weight_grouped = orig_weight.view(
                out_features, in_features // self.group_size, self.group_size
            )
            weight = self._dequantize_soft(orig_weight_grouped, temperature=0.01).view(out_features, in_features)
        else:
            weight = self.dequantize_weight()
        
        return F.linear(x, weight)


    @torch.no_grad()
    def dequantize(self):
        return self.dequantize_weight()


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


@torch.no_grad()
def update_indexes(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, CodebookWrapperLinear):
            module.update_indexes()


def wrap_model(model: nn.Module, n_bits: int = 2) -> nn.Module:
    skip = 5
    for i, layer in enumerate(model.model.layers):
        if i < skip or i > len(model.model.layers) - skip - 1:  # Skip first 2 layers and layers after 5 for faster example, can be removed for full training
            continue
        print(f"Wrapping layer {i} with CodebookWrapperLinear")
        model.model.layers[i] = wrap_model_block(layer, n_bits=n_bits)
    return model

def unwrap_model(model: nn.Module) -> nn.Module:
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = unwrap_model_block(layer)
    return model

def wrap_model_block(model: nn.Module, n_bits: int = 2, layer_index: int = -1, n_layers: int = -1, use_llama_cpp_scheme: bool = True) -> nn.Module:
    """
    Wraps the linear layers of the model with CodebookWrapperLinear for adaptive codebook compression.

    :param model: The original model to be wrapped.
    :param n_bits: The number of bits for quantization.

    :param adaptive_codebook: A boolean flag indicating whether to use adaptive codebook compression.
    :return: The wrapped model with CodebookWrapperLinear layers.
    """
    
    changed_modules = {}
    
    if use_llama_cpp_scheme and layer_index >= 0 and n_layers > 0:
        # For LLaMA-like models, only wrap q_proj, k_proj, v_proj, and out_proj of attention layers, and skip the first and last few layers
        for name, module in model.named_modules():
            if 'lm_head' in name or 'v_proj' in name or ('down_proj' in name and layer_index < n_layers / 8):
                continue
            if isinstance(module, nn.Linear):
                changed_modules[name] = module
    else:
        for name, module in model.named_modules():
            if 'lm_head' in name or 'v_proj' in name or 'down_proj' in name:
                continue
            # if not 'k_proj' in name:
            #     continue
            if isinstance(module, nn.Linear):
                changed_modules[name] = module
    
    for name, module in changed_modules.items():
        print(f"Wrapping layer {name} with CodebookWrapperLinear")
        set_module(model, name, CodebookWrapperLinear(module, n_bits=n_bits))
    return model


def unwrap_model_block(model: nn.Module) -> nn.Module:
    """
    Unwraps the CodebookWrapperLinear layers in the model back to their original linear layers.

    :param model: The model with CodebookWrapperLinear layers to be unwrapped.
    :return: The unwrapped model with original linear layers.
    """
    changed_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, CodebookWrapperLinear):
            module.orig_layer.weight.data.copy_(module.dequantize())
            changed_modules[name] = module.orig_layer
    for name, orig_layer in changed_modules.items():
        set_module(model, name, orig_layer)
    return model
