"""Evaluation utilities for trajectory classification."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def classification_metrics(logits: Tensor, labels: Tensor, num_classes: int) -> dict[str, object]:
    """Compute paper-primary accuracy plus engineering supplementary metrics."""
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).to(torch.float32).mean().item()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(labels.to(torch.long), predictions.to(torch.long), strict=True):
        confusion[target, prediction] += 1
    totals = confusion.sum(dim=1)
    per_class = [
        (confusion[index, index].item() / totals[index].item()) if totals[index].item() else None
        for index in range(num_classes)
    ]
    return {"accuracy": accuracy, "confusion_matrix": confusion.tolist(), "per_class_accuracy": per_class}


def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    criterion: nn.Module,
    device: str | torch.device,
    num_classes: int,
) -> dict[str, object]:
    """Evaluate without gradient updates using lengths-aware model calls."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)
            logits = model(sequences, lengths=lengths)
            total_loss += criterion(logits, labels).item() * labels.shape[0]
            total_samples += labels.shape[0]
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    if was_training:
        model.train()
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels), num_classes)
    metrics["loss"] = total_loss / total_samples
    return metrics
