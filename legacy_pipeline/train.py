"""
Training loop for the corrected legacy pipeline.

Corrects two bugs in `train_model.py`:

D14 -- `evaluate()` did `outputs = [compute_batch_loss_acc(...)]` *inside*
its loop over the validation DataLoader, reassigning rather than
accumulating. It reported metrics from whichever batch happened to run
last, not an average over the validation set. Fixed here by accumulating
every batch's loss/correct-count/sample-count and dividing once at the end
(weighted by batch size, so an uneven final batch doesn't skew the result).

D15 -- `fit_one_cycle()` called `evaluate()` (and updated `history`) *inside*
the per-batch training loop rather than once per epoch, so a full validation
pass ran after every single training batch, and the value eventually
reported for "this epoch's accuracy" was really whatever the very last
training batch's subsequent evaluate() call produced -- itself only a
single validation batch, per D14. Fixed here: one evaluate() call at the
end of each epoch, not one per training batch.

The accuracy-drop stopping decision the whole pruning schedule depends on
(`accuracy_drop_threshold` in `PruningConfig`) is only meaningful if these
numbers are real epoch-level validation averages -- that's the reason this
file exists rather than reusing the original almost-as-is.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    lr: float


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = nn.functional.cross_entropy(outputs, labels, reduction="sum")
        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.numel()
    if total_samples == 0:
        return {"val_loss": float("nan"), "val_acc": float("nan")}
    return {"val_loss": total_loss / total_samples, "val_acc": total_correct / total_samples}


def _l1_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of absolute values across *all* parameters, globally -- not the
    original's `sum(nn.L1Loss()(param, zeros) for param in model.parameters())`,
    which averages within each parameter tensor before summing across tensors.
    That implicitly weights a tiny bias vector the same as a huge weight
    matrix regardless of how many numbers are in each, which isn't what "L1
    regularization strength" is supposed to mean. This sums raw magnitudes
    globally, then the caller scales by `l1_lambda`."""
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total = total + param.abs().sum()
    return total


def fit_one_cycle(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    max_lr: float = 0.01,
    weight_decay: float = 0.0,
    l1_lambda: float = 0.0,
    grad_clip: float | None = None,
    opt_func=torch.optim.Adam,
    log_path: Path | None = None,
) -> list[EpochResult]:
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=max(1, len(train_loader))
    )

    history: list[EpochResult] = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = nn.functional.cross_entropy(model(images), labels)
            if l1_lambda:
                loss = loss + l1_lambda * _l1_penalty(model)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

        # One evaluation per epoch, over the *entire* validation set --
        # not one per training batch, and not just the last validation batch.
        val_metrics = evaluate(model, test_loader, device)
        result = EpochResult(
            epoch=epoch,
            train_loss=sum(train_losses) / max(1, len(train_losses)),
            val_loss=val_metrics["val_loss"],
            val_acc=val_metrics["val_acc"],
            lr=optimizer.param_groups[0]["lr"],
        )
        history.append(result)

        line = (
            f"epoch {result.epoch:>3}  lr={result.lr:.6f}  "
            f"train_loss={result.train_loss:.4f}  val_loss={result.val_loss:.4f}  val_acc={result.val_acc:.4f}"
        )
        print(line)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(line + "\n")

    return history


def write_history_csv(history: list[EpochResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])
        for r in history:
            writer.writerow([r.epoch, r.train_loss, r.val_loss, r.val_acc, r.lr])
