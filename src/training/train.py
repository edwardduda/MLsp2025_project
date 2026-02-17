from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from src.models.cnn import BaselineCNN
from src.models.kan_cnn import KANCNN
from src.models.mlp import PlainMLP
from src.models.kan import PlainKAN

# ── constants ────────────────────────────────────────────────────────
SEED = 42
N_EPOCHS = 15
BATCH_SIZE = 128
IMAGE_SIZE = (28, 28)
N_CLASSES = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 4
MEAN = [0.1307]
STD = [0.3081]
MODELS_DIR = Path(__file__).parent.parent.parent / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── device ───────────────────────────────────────────────────────────
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ── data loading ─────────────────────────────────────────────────────
def get_dataloaders(batch_size):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_full = datasets.MNIST(root="data", train=True, download=True, transform=tfms)

    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfms)

    mk_loader = partial(DataLoader, batch_size=batch_size, pin_memory=True)
    train_loader = mk_loader(train_ds, shuffle=True, drop_last=True)
    val_loader = mk_loader(val_ds, shuffle=False, drop_last=False)
    test_loader = mk_loader(test_ds, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


# ── accuracy helper ──────────────────────────────────────────────────
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean()


# ── single epoch ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    loss_sum = acc_sum = 0.0
    for x, y in tqdm(loader, desc=f"Train {epoch:02d}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_logits(logits=logits, targets=y)
        loss_sum += loss.item() * x.size(0)
        acc_sum += acc.item() * x.size(0)

    n = len(loader.dataset)
    return {"loss": loss_sum / n, "acc": acc_sum / n}


# ── evaluation ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = acc_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        acc = accuracy_from_logits(logits=logits, targets=y)
        loss_sum += loss.item() * x.size(0)
        acc_sum += acc.item() * x.size(0)
    n = len(loader.dataset)
    return {"loss": loss_sum / n, "acc": acc_sum / n}


# ── main training loop ──────────────────────────────────────────────
def run_training(model, name, train_loader, val_loader, n_epochs, lr):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        tic = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )
        val_metrics = evaluate(
            model=model, loader=val_loader, criterion=criterion
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch:2d}/{n_epochs} | "
            f"Train Acc {train_metrics['acc']*100:5.2f}% | "
            f"Val Acc {val_metrics['acc']*100:5.2f}% | "
            f"{time.time()-tic:4.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_model_state = {
                k: v.cpu().detach().clone()
                for k, v in model.state_dict().items()
            }
            print(
                f"  -> New best at epoch {epoch} "
                f"(val loss {best_val_loss:.4f})"
            )

        if epoch - best_epoch >= PATIENCE:
            print(
                f"Early stopping after {PATIENCE} epochs without improvement"
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    save_path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    history_path = MODELS_DIR / f"{name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"History saved to {history_path}")

    return history


# ── model builders ───────────────────────────────────────────────────
def build_mlp_model():
    return BaselineCNN(n_classes=N_CLASSES, dropout_p=0.1, in_channels=1)


def build_kan_model():
    return KANCNN(
        n_classes=N_CLASSES,
        input_shape=(1, 28, 28),
        ch1=32,
        ch2=64,
        ch3=128,
        ch4=256,
        kan_1=64,
        kan_2=32,
        spline_cp=7,
        spline_deg=3,
        range_min=-3.0,
        range_max=10.0,
        dropout_p=0.1,
    )


def build_plain_mlp_model():
    return PlainMLP(
        n_classes=N_CLASSES,
        dropout_p=0.1,
        input_dim=IMAGE_SIZE[0] * IMAGE_SIZE[1],
    )


def build_plain_kan_model():
    return PlainKAN(
        n_classes=N_CLASSES,
        input_dim=IMAGE_SIZE[0] * IMAGE_SIZE[1],
        kan_1=128,
        kan_2=64,
        spline_cp=7,
        spline_deg=3,
        range_min=-3.0,
        range_max=10.0,
        dropout_p=0.1,
    )


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP and/or KAN on MNIST")
    parser.add_argument(
        "--model",
        choices=["mlp", "kan", "plain_mlp", "plain_kan", "all"],
        required=True,
        help="Which model(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Using device: {device}")
    print(f"Loading data (batch_size={args.batch_size})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size
    )

    models_to_train = []
    if args.model in ("mlp", "all"):
        models_to_train.append(("baseline_cnn", build_mlp_model()))
    if args.model in ("kan", "all"):
        models_to_train.append(("kan_cnn", build_kan_model()))
    if args.model in ("plain_mlp", "all"):
        models_to_train.append(("plain_mlp", build_plain_mlp_model()))
    if args.model in ("plain_kan", "all"):
        models_to_train.append(("plain_kan", build_plain_kan_model()))

    criterion = nn.CrossEntropyLoss()

    for name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")

        run_training(
            model=model,
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=args.epochs,
            lr=args.lr,
        )

        test_metrics = evaluate(
            model=model.to(device), loader=test_loader, criterion=criterion
        )
        print(
            f"\nTest Results for {name}: "
            f"Loss {test_metrics['loss']:.4f} | "
            f"Acc {test_metrics['acc']*100:.2f}%"
        )


if __name__ == "__main__":
    main()
