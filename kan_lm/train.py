"""
Training script for the KAN Language Model.

Usage:
    python -m kan_lm.train
    python -m kan_lm.train --epochs 10 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from kan_lm.config import KANLMConfig
from kan_lm.data import build_dataloaders
from kan_lm.model import KANLanguageModel

SEED = 42
SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models" / "kan_lm"
LOG_DIR = Path(__file__).resolve().parent.parent / "runs" / "kan_lm"

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def get_lr(step, warmup_steps, max_steps, max_lr):
    """Linear warmup then cosine decay to 10% of max_lr."""
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps, use_amp):
    """Run a few validation batches and return mean loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, loss = model(input_ids=x, targets=y)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def train(config: KANLMConfig):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    use_amp = device.type == "cuda"

    print(f"Device: {device}")
    print(f"  Mixed precision: {'enabled' if use_amp else 'disabled (CUDA only)'}")
    print("Loading data (this downloads TinyStories on first run)...")
    train_loader, val_loader, tokenizer = build_dataloaders(config=config)
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")

    model = KANLanguageModel(config=config).to(device)
    print(f"  Parameters:    {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scaler = torch.amp.GradScaler(enabled=use_amp)
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    print(f"  TensorBoard:   tensorboard --logdir {LOG_DIR}")

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    model.train()
    for epoch in range(1, config.max_epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs}")
        for x, y in pbar:
            if global_step >= config.max_steps:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = get_lr(
                step=global_step,
                warmup_steps=config.warmup_steps,
                max_steps=config.max_steps,
                max_lr=config.learning_rate,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(input_ids=x, targets=y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            if (global_step + 1) % config.eval_interval == 0:
                val_loss = estimate_loss(
                    model=model,
                    val_loader=val_loader,
                    eval_steps=config.eval_steps,
                    use_amp=use_amp,
                )
                writer.add_scalar("val/loss", val_loss, global_step + 1)

                history["step"].append(global_step + 1)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["lr"].append(lr)

                improved = val_loss < best_val_loss
                tag = " *best*" if improved else ""
                print(
                    f"\n  Step {global_step+1:>6d} | "
                    f"Train {loss.item():.4f} | "
                    f"Val {val_loss:.4f} | "
                    f"LR {lr:.2e}{tag}"
                )

                if improved:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
                else:
                    patience_counter += 1

                if patience_counter >= config.patience:
                    print(f"Early stopping (no improvement for {config.patience} evals)")
                    break

                model.train()

            global_step += 1

        if global_step >= config.max_steps or patience_counter >= config.patience:
            break

    writer.close()

    # Save final checkpoint and history
    torch.save(model.state_dict(), SAVE_DIR / "final_model.pt")
    with open(SAVE_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(SAVE_DIR / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nTraining complete  |  Best val loss: {best_val_loss:.4f}")
    print(f"Artifacts saved to {SAVE_DIR}")
    return history


def parse_args():
    p = argparse.ArgumentParser(description="Train the KAN Language Model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--context-length", type=int)
    return p.parse_args()


def main():
    args = parse_args()
    config = KANLMConfig()

    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.context_length is not None:
        config.context_length = args.context_length

    train(config=config)


if __name__ == "__main__":
    main()
