import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable
from rich import print

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lm.model import DecoderLM
from lm.utils import (
    count_params,
    determine_device,
    enable_tf32,
    estimate_model_disk_size,
)


def random_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        An infinite generator that samples batches of sequences from the
        tokens. Each batch has shape (B x S). Every sequence in the batch is
        a contiguous subsequence of x, sampled uniformly at random. The
        output tensor should be on the right device.
    """

    while True:
        start_indices = torch.randint(0, len(tokens) - seq_len + 1, (batch_size,))
        batch = torch.stack([tokens[i : i + seq_len] for i in start_indices])
        yield batch.to(device)


def sequential_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        A generator that yields a batch of tokens at a time. Each batch has
        shape (B x S). Every sequence in the batch is a contiguous subsequence
        of x in sequential order. The output tensor should be on the right
        device.

    Note: If the last batch is incomplete, which could happen when the number
        of tokens is not divisible by (batch_size * seq_len), you could drop
        the last batch.
    """

    tokens_per_batch = batch_size * seq_len
    num_batches = len(tokens) // tokens_per_batch
    truncated_tokens = tokens[: num_batches * tokens_per_batch]
    batches = truncated_tokens.view(num_batches, batch_size, seq_len)

    for batch in batches:
        yield batch.to(device)


def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t

        Hint: Question 1.2
        """

        assert max_lr >= min_lr >= 0.0
        assert num_training_steps >= num_warmup_steps >= 0

        if t <= num_warmup_steps:
            lr = (t / num_warmup_steps) * max_lr if num_warmup_steps > 0 else max_lr
        elif t >= num_training_steps:
            lr = min_lr
        else:  # t >= num_training_steps
            progress = (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        return lr

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
    input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits

    Args:
        input_ids: the input token ids
        logits: the next token logits produced by the language model

    Returns:
        loss: the mean cross entropy loss for next token prediction

    Hint: Think about what are the groundtruth labels for next token prediction.
    """

    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1, :].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def train(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    optimizer: torch.optim.Optimizer,
    lr_schedule: Callable[[int], float],
    autocast: torch.autocast | nullcontext = nullcontext(),
    num_training_steps: int = 0,
    grad_accumulation_steps: int = 1,
    start_step: int = 0,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 5000,
    config: dict | None = None,
) -> None:
    """Train the model, periodically saving resumable checkpoints."""
    losses = deque(maxlen=20 * grad_accumulation_steps)

    for step in trange(start_step, num_training_steps):
        t0 = time.time()
        set_lr(optimizer, lr_schedule(step))
        optimizer.zero_grad()

        for _ in range(grad_accumulation_steps):
            input_ids = next(batch_sampler)
            with autocast:
                logits = model(input_ids)
                loss = compute_language_modeling_loss(input_ids, logits)
            (loss / grad_accumulation_steps).backward()
            losses.append(loss.item())

        optimizer.step()
        loss_mean = np.mean(losses).item()

        FLOPs_per_step = (
            model.flops_per_token
            * input_ids.shape[0]
            * input_ids.shape[1]
            * grad_accumulation_steps
        )
        dt = time.time() - t0

        wandb.log(
            {"train-loss": loss_mean, "learning-rate": lr_schedule(step)},
            step=step,
        )

        if checkpoint_dir is not None and (step + 1) % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step + 1,
                "config": config,
            }
            latest_path = os.path.join(checkpoint_dir, "checkpoint-latest.pt")
            torch.save(checkpoint, latest_path)
            print(f"saved latest checkpoint to {latest_path}")


@torch.inference_mode()
def evaluate(
    model: DecoderLM,
    batch_sampler: Iterator[torch.LongTensor],
    autocast: torch.autocast | nullcontext = nullcontext(),
) -> dict[str, float]:
    losses = []

    for input_ids in tqdm(batch_sampler, desc="evaluating.."):
        with autocast:
            logits = model(input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)

    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


def main():
    enable_tf32()

    # create an output directory and dump the configuration file
    assert len(sys.argv) > 1, "provide a configuration file"
    config = OmegaConf.load(sys.argv[1])
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="")
    wandb.init(project="llms-hw2", config=OmegaConf.to_container(config))

    # initialize tokenizer and model
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
    if model_disk_size_MB > 95:
        raise RuntimeError(
            f"Model is {model_disk_size_MB:.1f}MB, which exceeds the 95MB checkpoint budget. "
            "Reduce model size or training config before uploading."
        )
    print(
        f"Your model is {model_disk_size_MB:.1f}MB. This is within the 95MB checkpoint budget."
    )

    # prepare data and data generator
    assert config.seq_len <= config.model_config.n_positions
    tokens = np.load("data/tokens.npz")

    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )
    print(f"train dataset tokens = {len(train_tokens) / 1e6:.0f}M")
    FLOPs = (
        model.flops_per_token
        * config.num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    print(f"train FLOPs = {FLOPs:.2e}")
    if FLOPs > 1e17:
        print(
            f"[red]WARNING: your train FLOPs is {FLOPs:.2e}. "
            "This is more than the max compute that we allow (1e+17). "
            "Please reduce your model size or train steps.[/red]"
        )

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # will set this dynamically in the training loop
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    autocast = (
        torch.autocast(
            device,
            dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        )
        if device == "cuda"
        else nullcontext()
    )

    # Training with resumable checkpoints. Pass an optional third argument:
    # python src/lm/train.py config.yaml outputs/.../checkpoint-latest.pt
    start_step = 0
    resume_path = sys.argv[2] if len(sys.argv) > 2 else None
    if resume_path:
        resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_step = resume_checkpoint["step"]
        print(f"resumed from {resume_path} at step {start_step}")

    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        autocast,
        config.num_training_steps,
        config.grad_accumulation_steps,
        start_step=start_step,
        checkpoint_dir=config.output_dir,
        checkpoint_interval=5000,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # evaluation first so the checkpoint can include validation metrics
    model.eval()
    eval_results = evaluate(model, val_sampler, autocast)
    print("evaluation results:", json.dumps(eval_results))
    with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    # save the trained model checkpoint under the 95MB submission budget
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
        "eval_results": eval_results,
        "model_size_mb": estimate_model_disk_size(model) * 1e-6,
    }
    model_path = os.path.join(config.output_dir, "model.pt")
    checkpoint_path = os.path.join(config.output_dir, "checkpoint.pt")

    torch.save(checkpoint, checkpoint_path)
    torch.save(model.state_dict(), model_path)
    print(f"checkpoint saved to {checkpoint_path}")
    print(f"model saved to {model_path}")
    print("done!")


if __name__ == "__main__":
    main()
