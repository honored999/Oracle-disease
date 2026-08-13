"""Configuration-driven training entry point for the Singh & Koundal TCN."""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn

from TCN_Singh2024.src.dataset import TrajectoryClassificationDataset
from TCN_Singh2024.src.evaluation import classification_metrics
from TCN_Singh2024.src.model import TCNClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    """Run zero_grad, forward, loss, backward, and optimizer.step for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[Tensor] = []
    all_labels: list[Tensor] = []
    for batch in loader:
        sequences = batch["sequences"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(sequences, lengths=lengths)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits")
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise FloatingPointError("non-finite gradients")
        optimizer.step()
        total_loss += loss.item() * labels.shape[0]
        total_samples += labels.shape[0]
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    metrics = classification_metrics(torch.cat(all_logits), torch.cat(all_labels), model.classifier.out_features)
    return {"loss": total_loss / total_samples, "accuracy": float(metrics["accuracy"])}


def load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def run_synthetic_smoke(config: dict[str, object], output_dir: str | Path) -> dict[str, object]:
    """Run an explicitly synthetic, engineering-only CV smoke test."""
    from TCN_Singh2024.src.cv import run_cross_validation

    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    input_channels = int(model_config["input_channels"])
    num_classes = 2
    generator = torch.Generator().manual_seed(int(training_config["seed"]))
    dataset = TrajectoryClassificationDataset(
        [torch.randn(2 + index % 3, input_channels, generator=generator) for index in range(10)],
        [index % num_classes for index in range(10)],
    )

    def model_factory() -> TCNClassifier:
        return TCNClassifier(
            input_channels=input_channels,
            hidden_channels=list(model_config["hidden_channels"]),
            kernel_size=int(model_config["kernel_size"]),
            dilations=list(model_config["dilations"]),
            dropout=float(model_config["dropout"]),
            num_classes=num_classes,
        )

    return run_cross_validation(
        dataset=dataset,
        model_factory=model_factory,
        batch_size=min(int(training_config["batch_size"]), len(dataset)),
        learning_rate=float(training_config["learning_rate"]),
        epochs=1,
        folds=int(evaluation_config["folds"]),
        seed=int(training_config["seed"]),
        output_dir=output_dir,
        device="cpu",
        num_classes=num_classes,
        shuffle_train=bool(training_config.get("shuffle_train", True)),
        num_workers=int(training_config.get("num_workers", 0)),
        drop_last=bool(training_config.get("drop_last", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Singh & Koundal TCN once a verified dataset adapter is supplied.")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run the engineering-only synthetic CV smoke test")
    parser.add_argument("--output-dir", help="Metrics directory required with --synthetic-smoke")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.synthetic_smoke:
        if not arguments.output_dir:
            parser.error("--output-dir is required with --synthetic-smoke")
        summary = run_synthetic_smoke(config, arguments.output_dir)
        print(f"Synthetic engineering smoke completed; mean accuracy={summary['mean_accuracy']:.4f}. Not a paper result.")
        return
    raise SystemExit("No verified RTD/RTC/6DMG file-format adapter is available; use the library training functions with a verified Dataset.")


if __name__ == "__main__":
    main()
