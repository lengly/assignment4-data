"""
Train a language model on one or multiple GPUs.

Default config is `experiment/your_data`, which will train on your GPT-2 tokenized dataset and validate on `tokenized_paloma_c4_100_domains_validation.bin`.

To ready the config for your run, you should:
1. open the config file at `cs336-basics/configs/experiment/your_data.yaml` and set the `paths.train_bin` attribute to point to the file containing your tokenized training data.
2. You should also set an appropriate `training.wandb_entity` and `training.wandb_project` attribute for logging.

To run single-GPU training:

```
uv run python scripts/train.py --config-name=experiment/your_data
```

To run multi-GPU training, use `torchrun`. e.g., for single-node, 2 GPU:

```
uv run torchrun --standalone --nproc_per_node=2 scripts/train.py --config-name=experiment/your_data
```
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rich.pretty import pprint as pprint
from rich.traceback import install
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm, trange

import wandb
from cs336_basics.data import UniformMixDataLoader
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import get_wsd_lr
from cs336_basics.train_config import Config, register_configs

import random
from collections import deque

register_configs()

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

install(show_locals=True)

@hydra.main(version_base=None, config_path=str(Path("cs336-basics/configs").absolute().resolve()), config_name="experiment/your_data")
def main(cfg: Config) -> None:

    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    pprint(cfg_dict)

    # Take defaults
    default_cfg = OmegaConf.structured(Config())
    cfg = OmegaConf.merge(default_cfg, cfg_dict)

    seed = cfg.training.seed + (int(os.environ.get("RANK", 0)) if "RANK" in os.environ else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Use UniformMixDataLoader for training data
    print("Creating train loader")
    train_loader = UniformMixDataLoader(cfg.paths.train_bin, cfg.training.train_batch_size, cfg.model.context_length)
    print("Train loader created")
    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        cfg=cfg,
    )
    pprint(model)

    # Wrap model in DDP, if we're using it.
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        # Rank 0 does logging, file creation, etc.
        is_master_process = ddp_rank == 0
        if is_master_process:
            logger.info("Using DDP")
    else:
        ddp_world_size = 1
        is_master_process = True

    if is_master_process:
        logger.info(
            "Total number of tokens per training step: "
            + str(
                cfg.training.gradient_accumulation_steps
                * ddp_world_size
                * cfg.training.train_batch_size
                * cfg.model.context_length
            )
        )
        if cfg.training.wandb_project and cfg.training.wandb_entity:
            wandb.init(
                # Set the project where this run will be logged
                entity=cfg.training.wandb_entity,
                project=cfg.training.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.paths.model_output.name,
            )

    # Save the model config
    if is_master_process:
        cfg.paths.model_output.mkdir(parents=True, exist_ok=True)
        model_config_output_path = cfg.paths.model_output / "model_config.json"
        logger.info(f"Saving model config to {model_config_output_path}")
        model_config = model.config
        with open(model_config_output_path, "w") as f:
            json.dump(model_config, f, indent=4)

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.training.dtype]
    if is_master_process:
        logger.info(f"Using dtype: {torch_dtype}")

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)

    # Move model to the device
    model = model.to(cfg.training.device)

    # compile the model, requires torch 2.0
    if cfg.training.compile:
        print("Compiling model")
        model = torch.compile(model)
        print("Model compiled")

    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # Set up the AdamW optimizer.
    # First, we need to group the parameters that should
    # be decayed and those that shouldn't.
    # In particular, we do not apply decay on 1D parameters (e.g., biases and RMSNorms)
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Parameter groups for different learning rates
    attn_params = [p for n, p in param_dict.items() if ("attn" in n and p.dim() >= 2)]
    ffn_params = [p for n, p in param_dict.items() if ("ffn" in n and p.dim() >= 2)]
    other_params = [p for n, p in param_dict.items() if ("attn" not in n and "ffn" not in n and p.dim() >= 2)]
    not_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    # Set different learning rates for attn, ffn, and other parameters
    attn_lr_scale = cfg.training.mup_base_hidden_size / cfg.model.d_model
    ffn_lr_scale = cfg.training.mup_base_filter_size / cfg.model.d_ff
    attn_lr = cfg.training.lr * attn_lr_scale
    ffn_lr = cfg.training.lr * ffn_lr_scale
    basic_lr = cfg.training.lr
    print(f"basic_lr: {basic_lr}, attn_lr_scale: {attn_lr_scale}, ffn_lr_scale: {ffn_lr_scale}, attn_lr: {attn_lr}, ffn_lr: {ffn_lr}")

    optim_groups = [
        {"params": attn_params, "weight_decay": cfg.training.weight_decay, "lr": attn_lr},
        {"params": ffn_params, "weight_decay": cfg.training.weight_decay, "lr": ffn_lr},
        {"params": other_params, "weight_decay": cfg.training.weight_decay, "lr": basic_lr},
        {"params": not_decay_params, "weight_decay": 0.0, "lr": basic_lr},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        eps=cfg.training.adam_eps,
        fused=True,
    )

    # Get the first batch
    batch_x, batch_y = next(train_loader)
    batch_x = torch.tensor(batch_x, dtype=torch.long, device=cfg.training.device)
    batch_y = torch.tensor(batch_y, dtype=torch.long, device=cfg.training.device)
    loss_history = deque(maxlen=10)
    for i in (pbar := trange(cfg.training.train_steps, desc="Training", disable=not is_master_process)):
        lr = get_wsd_lr(
            i,
            max_learning_rate=cfg.training.lr,
            min_learning_rate=cfg.training.lr * 0.1,
            warmup_iters=int(cfg.training.train_steps * cfg.training.warmup_ratio),
            stable_iters=int(cfg.training.train_steps * cfg.training.stable_ratio),
            decay_iters=int(cfg.training.train_steps * cfg.training.decay_ratio),
        )
        # Update learning rates for each param group
        optimizer.param_groups[0]["lr"] = lr * attn_lr_scale  # attn
        optimizer.param_groups[1]["lr"] = lr * ffn_lr_scale   # ffn
        optimizer.param_groups[2]["lr"] = lr                  # other
        optimizer.param_groups[3]["lr"] = lr                  # not_decay

        total_loss = 0.0
        for micro_step_idx in range(cfg.training.gradient_accumulation_steps):
            if is_ddp:
                # When using DDP, don't all-reduce gradients until the last step.
                model.require_backward_grad_sync = micro_step_idx == cfg.training.gradient_accumulation_steps - 1

            with amp_ctx:
                logits = model(batch_x)

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                next_batch_x, next_batch_y = next(train_loader)
                next_batch_x = torch.tensor(next_batch_x, dtype=torch.long, device=cfg.training.device)
                next_batch_y = torch.tensor(next_batch_y, dtype=torch.long, device=cfg.training.device)

                # Calculate the loss with the logits
                loss = (
                    F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                    / cfg.training.gradient_accumulation_steps
                )

            loss.backward()
            total_loss += loss.item()

            batch_x = next_batch_x
            batch_y = next_batch_y

        if cfg.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # record the sum of all micro-step loss
        loss_history.append(total_loss)
        loss_float = sum(loss_history) / len(loss_history)

        if is_master_process:
            pbar.set_description(f"Training step {i}, Loss: {total_loss:.4f}, Smoothed Loss: {loss_float:.4f}")
            if cfg.training.wandb_project and i % cfg.training.log_interval == 0:
                wandb.log({"train_loss": loss_float, "lr": lr}, step=i)

        if i != 0 and i % cfg.training.eval_interval == 0:
            dev_loss = estimate_dev_loss(
                model=model,
                val_dataset=cfg.paths.valid_bin,
                batch_size=cfg.training.eval_batch_size,
                eval_iters=cfg.training.eval_iterations,
                device=cfg.training.device,
                context_length=cfg.model.context_length,
            )
            if is_master_process:
                logger.info(f"Estimated validation loss: {dev_loss}")
                if cfg.training.wandb_project:
                    wandb.log({"eval_loss": dev_loss}, step=i)

                if cfg.training.save_checkpoints:
                    model_weights_output_path = cfg.paths.model_output / f"step_{i:010d}" / "model.pt"
                    model_weights_output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Need both config and weights to load the model
                    # Write config:
                    with open(model_weights_output_path.parent / "model_config.json", "w") as f:
                        json.dump(model_config, f, indent=4)

                    # Write weights:
                    torch.save(model.state_dict(), model_weights_output_path)

    # Calculate final estimated dev loss
    dev_loss = estimate_dev_loss(
        model=model,
        val_dataset=cfg.paths.valid_bin,
        batch_size=cfg.training.eval_batch_size,
        eval_iters=cfg.training.eval_iterations,
        device=cfg.training.device,
        context_length=cfg.model.context_length,
    )
    if is_master_process:
        logger.info(f"Final estimated validation loss: {dev_loss}")
        if cfg.training.wandb_project:
            wandb.log({"eval_loss": dev_loss}, step=cfg.training.train_steps)

        # Save the model weights
        model_weights_output_path = cfg.paths.model_output / "model.pt"
        logger.info(f"Saving model weights to {model_weights_output_path}")
        torch.save(model.state_dict(), model_weights_output_path)

    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def estimate_dev_loss(
    model: BasicsTransformerLM,
    val_dataset: str,
    batch_size: int,
    eval_iters: int | None,
    device: str,
    context_length: int,
):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    val_loader = UniformMixDataLoader(
        val_dataset, batch_size, context_length,
        mode="eval", rank=rank, world_size=world_size,
        eval_iterations=eval_iters
    )
    model.eval()
    losses = []
    if eval_iters is not None:
        for k in tqdm(range(eval_iters)):
            batch_x, batch_y = next(val_loader)
            batch_x = torch.tensor(batch_x, dtype=torch.long, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            losses.append(loss.item())
    else:
        while True:
            try:
                batch_x, batch_y = next(val_loader)
            except StopIteration:
                break
            batch_x = torch.tensor(batch_x, dtype=torch.long, device=device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            losses.append(loss.item())

    total_loss = torch.tensor([sum(losses)], device=device)
    total_count = torch.tensor([len(losses)], device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / total_count.item() if total_count.item() > 0 else float('nan')

    model.train()
    return avg_loss


if __name__ == "__main__":
    main()
