# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Learnable Codebook Compression with Layer-wise Fine-tuning

This module provides functionality for compressing LLM models using learnable codebooks
with two fine-tuning modes:

1. FULL MODEL MODE (default):
   - Trains all layers simultaneously using KL divergence on final logits
   - Uses pre-computed hidden states from teacher model
   - Best for general model compression

2. LAYER-WISE MODE (--layerwise):
   - Trains each layer sequentially using L2 loss between layer outputs
   - Compares original bf16 model output vs compressed model output per layer
   - Focuses on MLP and/or attention sublayers in model.model.layers
   - More granular control over training process
   
Usage Examples:

  # Full model training (default):
  python main.py --pretrained meta-llama/Llama-3.2-1B-Instruct --epochs 50

  # Layer-wise training (all sublayers):
  python main.py --pretrained meta-llama/Llama-3.2-1B-Instruct --layerwise --layerwise_epochs 10

  # Layer-wise training (MLP only):
  python main.py --pretrained meta-llama/Llama-3.2-1B-Instruct --layerwise --layerwise_type mlp

  # Layer-wise training (attention only):
  python main.py --pretrained meta-llama/Llama-3.2-1B-Instruct --layerwise --layerwise_type attention

See finetune_layerwise_l2() function for programmatic API usage.
"""

import argparse
import shutil
import sys
import warnings
import shutil
import os
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import OptimumLM

from torch import Tensor
from torch import nn
from torch.jit import TracerWarning
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from nncf.common.logging.track_progress import track


from two_step_tuning import finetune_layerwise_ste, wrap_model_ste, unwrap_model_ste, cleanup

def save_codebook_layers(model: nn.Module, output_dir: Path):
    """
    Saves the codebook layers of the model to the specified output directory.

    :param model: The model containing the codebook layers to be saved.
    :param output_dir: The directory where the codebook layers will be saved.
    """
    codebook_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if "codebook" in k or "scale" in k}
    torch.save(codebook_state_dict, output_dir / "codebook_layers.pth")
    print(f"Codebook layers saved to {output_dir / 'codebook_layers.pth'}")


def set_trainable(model: nn.Module) -> list[nn.Parameter]:
    """
    Sets the parameters of the model to be trainable for fine-tuning with LoRA and Fake Quantization.

    :param model: The model whose parameters are to be set as trainable.
    :param lora_lr: The learning rate for LoRA adapters.
    :param fq_lr: The learning rate for Fake Quantization parameters.
    :return: A list of trainable parameters in the model.
    """
    lora_to_train = []
    scales_to_train = []
    codebooks_to_train = []
    
    for name, param in model.named_parameters():
        if "codebook" in name:
            param.requires_grad = True
            codebooks_to_train.append(param)
        elif "lora" in name:
            param.requires_grad = True
            lora_to_train.append(param)
        elif "scale" in name:
            param.requires_grad = True
            scales_to_train.append(param)
        else:
            param.requires_grad = False
    return lora_to_train, scales_to_train, codebooks_to_train


warnings.filterwarnings("ignore", category=TracerWarning)


def get_compression_calibration(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    def preprocess_fn(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

    ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
    #ds = ds.shuffle(seed=42).select(range(10 * num_samples))
    #ds = ds.map(preprocess_fn)
    
    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    return trainloader


def get_pile(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device):
    def preprocess_fn(example):
        return {"text": tokenizer.apply_chat_template(example["text"], add_generation_prompt=False, tokenize=False)}

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    #ds = ds.shuffle(seed=42).select(range(10 * num_samples))
    #ds = ds.map(preprocess_fn)
    
    trainloader = []
    for example in ds:
        trainenc = tokenizer(example["text"], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            continue
        if trainenc.input_ids.shape[1] > seqlen + 1:
            i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        else:
            i = 0
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
        if len(trainloader) >= num_samples:
            break

    return trainloader



def get_wikitext2(num_samples: int, seqlen: int, tokenizer: Any, device: torch.device) -> list[Tensor]:
    """
    Loads and processes the Wikitext-2 dataset for training.

    :param num_samples: Number of samples to generate.
    :param seqlen: Sequence length for each sample.
    :param tokenizer: Tokenizer to encode the text.
    :param device: Device to move the tensors to (e.g., 'cpu' or 'cuda').
    :return: A list of tensors containing the tokenized text samples.
    """
    traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    limit = num_samples * seqlen // 4  # ~1k for 128 samples with seqlen=32 to be aligned with optimum
    text = "".join([" \n" if s == "" else s for s in traindata["text"][:limit]])
    trainenc = tokenizer(text, return_tensors="pt")
    trainloader = []
    for _ in range(num_samples):
        # Crop a sequence of tokens of length seqlen starting at a random position
        i = torch.randint(0, trainenc.input_ids.shape[1] - seqlen - 1, (1,)).item()
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j].to(device)
        trainloader.append(inp)
    return trainloader

@torch.no_grad()
def measure_perplexity(
    optimum_model,
    max_length: Optional[int] = None,
    limit: Optional[Union[int, float]] = None,
) -> float:
    """
    Measure perplexity on the Wikitext dataset, via rolling loglikelihoods for a given model.

    :param optimum_model: A model to be evaluated.
    :param max_length: The maximum sequence length for evaluation.
    :param limit: Limit the number of examples per task (only use this for testing).
        If <1, limit is a percentage of the total number of examples.
    :return: The similarity score as a float.
    """
    task = "wikitext"
    print("#" * 50 + " Evaluate via lm-eval-harness " + "#" * 50)
    lm_obj = OptimumLM(pretrained=optimum_model, max_length=max_length)
    results = simple_evaluate(lm_obj, tasks=[task], limit=limit, log_samples=False)
    return results["results"][task]["word_perplexity,none"]


@torch.no_grad()
def calc_hiddens(model: nn.Module, dataloader: list[Tensor]) -> list[Tensor]:
    """
    Calculate the hidden states for each input in the dataloader using the given model.

    :param model: The model used to calculate the hidden states.
    :param dataloader: The dataloader providing the inputs to the model.
    :return: A list of hidden states for each input in the dataloader.
    """
    orig_hiddens = []
    for data in track(dataloader, description="Calculating original hiddens"):
        model_input = get_model_input(data)
        orig_hiddens.append(model.model(**model_input).last_hidden_state)

    torch.cuda.empty_cache()
    return orig_hiddens


def get_model_input(input_ids: Tensor) -> dict[str, Tensor]:
    """
    Prepares the model input dictionary with input IDs, attention mask, and position IDs.

    :param input_ids: Tensor containing the input IDs.
    :return: A dictionary with keys "input_ids", "attention_mask", and "position_ids",
        each mapping to their respective tensors.
    """
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}


def kl_div(student_hiddens: torch.Tensor, teacher_hiddens: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence loss between the student and teacher hidden states.
    The input tensors are expected to have the same shape, and the last dimension represents the number of classes.

    :param student_hiddens: The hidden states from the student model.
    :param teacher_hiddens: The hidden states from the teacher model.
    :returns: The computed KL divergence loss.
    """
    num_classes = student_hiddens.shape[-1]
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, num_classes), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, num_classes), dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def limit_type(astr: str):
    value = float(astr)
    if value < 0 or value > 1:
        msg = "value not in range [0,1]"
        raise argparse.ArgumentTypeError(msg)
    return value


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
        help="Path to the directory for storing logs, tuning checkpoint, compressed model, validation references.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to start from previously saved checkpoint. If not specified or checkpoint does not exist, "
        "start from scratch by post-training weight compression initialization.",
    )
    parser.add_argument("--lora_rank", type=int, default=256, help="Rank of lora adapters")
    parser.add_argument(
        "--basic_init",
        action="store_true",
        help="Whether to initialize quantization with basic min-max round-to-nearest schema. By default, advanced "
        "data-aware post-training methods are used: AWQ + Scale Estimation. These methods typically provide better "
        "accuracy, but require a calibration dataset and additional initialization time "
        "(~20 sec for 1B and ~80 sec for 8B models).",
    )

    # Data params
    parser.add_argument("--num_train_samples", type=int, default=1024, help="Number of training samples")
    parser.add_argument("--train_seqlen", type=int, default=1024, help="Train data context length.")
    parser.add_argument("--eval_seqlen", type=int, default=2048, help="Evaluation data context length.")
    parser.add_argument(
        "--limit",
        type=limit_type,
        default=None,
        help="A percentage of the total number of examples for evaluation. "
        "Should be on the range [0,1]. If None, all samples will be used.",
    )

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-1,
        help="Learning rate for fine-tuning. "
        "For larger models (over 3 billion parameters), a learning rate of 5e-5 is recommended.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.") #default=10
    parser.add_argument("--batch_size", type=int, default=64, help="Size of training batch.")
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=4,
        help="Size of each training microbatch. Gradients will be accumulated until the batch size is reached.",
    )
    
    # Layer-wise training params
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Enable layer-wise fine-tuning mode using L2 loss between original and compressed model outputs.",
    )
    parser.add_argument(
        "--layerwise_epochs",
        type=int,
        default=10,
        help="Number of epochs to train each layer in layer-wise mode.",
    )
    parser.add_argument(
        "--layerwise_type",
        type=str,
        default="both",
        choices=["mlp", "attention", "both"],
        help="Which sublayers to train in layer-wise mode: 'mlp', 'attention', or 'both'.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of experiment to create directoty for tensorboard and model checkpoint.",
    )
    parser.add_argument(
        "--keep_data_on_cpu",
        action="store_true",
        help="Whether to keep the data on CPU during training. This can be useful for large datasets that do not fit in GPU memory.",
    )

    return parser


def main(argv) -> float:
    """
    Fine-tunes the specified model and returns the difference between initial and best validation perplexity in Torch,
    and the test perplexity for best model exported to OpenVINO.
    """
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    assert torch.cuda.is_available()
    transformers.set_seed(42)
    device = "cuda"
    torch_dtype = torch.bfloat16

    # Configure output and log files.
    output_dir = Path(args.output_dir)

    if args.description is not None:
        tensorboard_dir = output_dir / "tb" / f"{args.description}"
        last_dir = output_dir / f"{args.description}_last"
    else:
        tensorboard_dir = output_dir / "tb" / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        last_dir = output_dir / "last"
    
    if not args.resume:
        shutil.rmtree(last_dir, ignore_errors=True)
    for path in [output_dir, tensorboard_dir, last_dir]:
        path.mkdir(exist_ok=True, parents=True)
    ckpt_file = last_dir / "nncf_checkpoint.pth"
    print(f"To visualize the loss and validation metrics, open Tensorboard using the logs from: {tensorboard_dir}")
    tb = SummaryWriter(tensorboard_dir, "QAT with absorbable LoRA")

    # Load original model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.pretrained, torch_dtype=torch_dtype, device_map="auto", use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # Prepare training and calibration data
    # train_loader = get_wikitext2(
    #     num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    # )
    train_loader = get_pile(
        num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    )
    
    # train_loader = get_compression_calibration(
    #     num_samples=args.num_train_samples, seqlen=args.train_seqlen, tokenizer=tokenizer, device=device
    # )

    #model = wrap_model(model)
    
    src_files = os.listdir('.')
    for file_name in src_files:
        if not file_name.endswith('.py'):
            continue
        full_file_name = os.path.join('.', file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, last_dir)

    
    if args.layerwise:
        #finetune_layerwise(model, tokenizer, train_loader, lr=args.lr, epochs_per_layer=args.layerwise_epochs, batch_size=args.batch_size, microbatch_size=args.microbatch_size, device=device, tb=tb)

        finetune_layerwise_ste(model, tokenizer, train_loader, lr=args.lr,
                               epochs_per_layer=args.layerwise_epochs,
                               batch_size=args.batch_size,
                               microbatch_size=args.microbatch_size,
                               device=device, tb=tb, 
                               lora_rank=512, group_size=64,
                               codebook_dst_dir=last_dir,
                               keep_data_on_cpu=args.keep_data_on_cpu)
        #save_codebook_layers(model, last_dir)

        model = unwrap_model(model)
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        print("Layer-wise fine-tuning complete!")
        print(f"{'='*80}\n")
        return

    # Pre-compute hiddens of teacher model for distillation loss.
    hiddens_dir = output_dir / f"hiddens_{args.num_train_samples}_{args.train_seqlen}.pt"
    if hiddens_dir.exists():
        print(f"Loading pre-computed hiddens from {hiddens_dir}")
        orig_hiddens = torch.load(hiddens_dir)
    else:
        orig_hiddens = calc_hiddens(model, train_loader)
        torch.save(orig_hiddens, hiddens_dir)


    model = wrap_model_ste(model, lora_rank=args.lora_rank, group_size=32)
    torch.cuda.empty_cache()
    
    if torch.cuda.device_count() > 1:
        model = torch.compile(model)

    # Original full-model training with KL divergence
    print("\n" + "="*80)
    print("Starting FULL MODEL fine-tuning mode")
    print("="*80 + "\n")
    

    lora_to_train, scales_to_train, codebooks_to_train = set_trainable(model)
    params_to_train = [
        {"params": codebooks_to_train, "lr": args.lr, "label": "codebook"},
        {"params": scales_to_train, "lr": args.lr, "label": "scale"},
        {"params": lora_to_train, "lr": 0.1 * args.lr, "label": "lora"},
    ]
    
    # first stage train only codebooks
    first_stage_epochs = max(args.epochs // 4, 1)
    
    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    update_indexes_counter = 0
    
    opt = torch.optim.AdamW(params_to_train[0]["params"], lr=args.lr)
    for p in scales_to_train:
        p.requires_grad = False
    for p in lora_to_train:
        p.requires_grad = False
    
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    # lambda_lr = lambda epoch: 0.99 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)

    total_wup_opt_steps = first_stage_epochs * epoch_samples // args.batch_size    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=args.lr * 1e-4, T_max=total_wup_opt_steps)

    
    for epoch in range(first_stage_epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        opt.zero_grad()

        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                tb.add_scalar("first_stage_loss", aggregated_loss, total_steps)
                tb.add_scalar("first_stage_lr", opt.param_groups[0]["lr"], total_steps)

                if aggregated_loss < 0.007:
                    print(f"Early stopping at epoch {epoch} with loss {aggregated_loss:.6f}")
                    break
                
                cleanup()
                scheduler.step()
    
    del opt
    del scheduler
    cleanup()
    
    
    for p in scales_to_train:
        p.requires_grad = True
    for p in lora_to_train:
        p.requires_grad = True
    for p in codebooks_to_train:
        p.requires_grad = False

    opt = torch.optim.AdamW([params_to_train[1], params_to_train[2]])
    
    # lambda_lr = lambda epoch: 0.99 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_lr)

    total_steps = args.epochs * epoch_samples // args.batch_size    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, eta_min=args.lr * 1e-4, T_max=total_steps)
    
    # Run tuning with distillation loss and validation after each epoch.
    
    aggregated_loss = float("nan")
    loss_numerator = grad_steps = total_steps = 0
    
    for epoch in range(args.epochs):
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)
        # if epoch > 300 and epoch % 2 == 0:
        #     ppl = measure_perplexity(model)
        #     tb.add_scalar("ppl", ppl, epoch)
        #     torch.cuda.empty_cache()

        for indices in track(batch_indices_epoch, description=f"Train epoch {epoch}"):
            indices = indices.tolist()

            def form_batch(inputs: list[Tensor], model_input: bool):
                batch = torch.cat([inputs[i] for i in indices], dim=0)
                return get_model_input(batch) if model_input else batch.to(device=device, dtype=torch_dtype)

            # Compute distillation loss between logits of the original model and the model with FQ + LoRA.
            inputs = form_batch(train_loader, model_input=True)
            with torch.no_grad():
                targets = model.lm_head(form_batch(orig_hiddens, model_input=False))
                if hasattr(model.config, "final_logit_softcapping"):  # Gemma has post-processing after lm_head
                    fls = model.config.final_logit_softcapping
                    if fls is not None:
                        targets = targets / fls
                        targets = torch.tanh(targets)
                        targets = targets * fls
            outputs = model(**inputs).logits
            loss = kl_div(outputs, targets.to(dtype=torch_dtype, device=device))

            # Perform an optimization step after accumulating gradients over multiple minibatches.
            loss_numerator += loss.item()
            grad_steps += 1
            if not torch.isfinite(loss).item():
                err = f"Fine-tuning loss is {loss}"
                raise ValueError(err)
            (loss / grad_accumulation_steps).backward()
            if grad_steps == grad_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                
                aggregated_loss = loss_numerator / grad_steps
                loss_numerator = grad_steps = 0
                total_steps += 1
                tb.add_scalar("second_stage_loss", aggregated_loss, total_steps)
                tb.add_scalar("second_stage_lr", opt.param_groups[0]["lr"], total_steps)

                if aggregated_loss < 0.007:
                    print(f"Early stopping at epoch {epoch} with loss {aggregated_loss:.6f}")
                    break

                scheduler.step()
                cleanup()

    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    for m in model.modules():
        if hasattr(m, "merge_lora"):
            m.merge_lora()

    model = unwrap_model_ste(model)
    model.save_pretrained(last_dir)
    tokenizer.save_pretrained(last_dir)
    


if __name__ == "__main__":
    main(sys.argv[1:])
