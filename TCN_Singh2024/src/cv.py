"""Sample-level K-fold orchestration for verified trajectory datasets."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from TCN_Singh2024.src.dataset import collate_trajectory_batch
from TCN_Singh2024.src.evaluation import evaluate
from TCN_Singh2024.src.train import set_seed, train_one_epoch


def make_kfold_splits(num_samples: int, folds: int, seed: int, shuffle: bool = True) -> list[tuple[list[int], list[int]]]:
    """Return reproducible sample-level K-fold train/validation index pairs."""
    if folds < 2 or folds > num_samples:
        raise ValueError("folds must be between 2 and num_samples")
    indices = list(range(num_samples))
    if shuffle:
        random.Random(seed).shuffle(indices)
    fold_sizes = [num_samples // folds + (fold < num_samples % folds) for fold in range(folds)]
    splits: list[tuple[list[int], list[int]]] = []
    start = 0
    for size in fold_sizes:
        validation = indices[start : start + size]
        train = indices[:start] + indices[start + size :]
        splits.append((train, validation))
        start += size
    return splits


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_cross_validation(
    dataset: Dataset,
    model_factory: Callable[[], nn.Module],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    folds: int,
    seed: int,
    output_dir: str | Path,
    device: str | torch.device,
    num_classes: int,
    shuffle_train: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    shuffle_splits: bool = True,
) -> dict[str, object]:
    """Train independent models per sample-level fold and persist only metrics JSON."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    output_path = Path(output_dir)
    split_pairs = make_kfold_splits(len(dataset), folds, seed, shuffle=shuffle_splits)
    fold_accuracies: list[float] = []
    for fold, (train_indices, validation_indices) in enumerate(split_pairs):
        set_seed(seed + fold)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle_train, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_trajectory_batch)
        validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_trajectory_batch)
        train_metrics: dict[str, float] = {}
        for _ in range(epochs):
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device, num_classes)
        fold_accuracies.append(float(validation_metrics["accuracy"]))
        _write_json(output_path / f"fold_{fold}" / "metrics.json", {
            "schema_version": 1,
            "fold": fold,
            "split_type": "sample_level_kfold",
            "seed": seed + fold,
            "num_classes": num_classes,
            "epochs": epochs,
            "train_size": len(train_indices),
            "validation_size": len(validation_indices),
            "validation_indices": validation_indices,
            "train": train_metrics,
            "validation": validation_metrics,
        })
    accuracy_tensor = torch.tensor(fold_accuracies, dtype=torch.float32)
    summary: dict[str, object] = {
        "schema_version": 1,
        "split_type": "sample_level_kfold",
        "number_of_folds": folds,
        "seed": seed,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": accuracy_tensor.mean().item(),
        "std_accuracy": accuracy_tensor.std(unbiased=False).item(),
        "fold_metrics": [f"fold_{fold}/metrics.json" for fold in range(folds)],
        "config_snapshot": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": "adam",
            "num_classes": num_classes,
            "shuffle_train": shuffle_train,
            "shuffle_splits": shuffle_splits,
            "num_workers": num_workers,
            "drop_last": drop_last,
        },
    }
    _write_json(output_path / "summary.json", summary)
    return summary
