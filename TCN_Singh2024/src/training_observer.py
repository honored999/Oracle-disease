"""Observation-only progress and artifact writer for RTC training runs."""

from __future__ import annotations

import json
import math
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"training observer requires metric {key!r}")
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError(f"training observer received non-finite metric {key!r}")
    return result


class RtcTrainingObserver:
    """Persist progress without selecting checkpoints or changing training."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_metadata: Mapping[str, Any],
        total_folds: int,
        total_epochs: int,
        mode: str = "formal",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_folds = int(total_folds)
        self.total_epochs = int(total_epochs)
        self.run_started_monotonic = time.perf_counter()
        run_prefix = "preflight" if mode == "preflight" else "baseline"
        self.run_id = f"rtc-{run_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.fold_started_monotonic: dict[int, float] = {}
        self.fold_info: dict[int, dict[str, int]] = {}
        self.histories: dict[int, list[dict[str, float | int]]] = {}
        self.fold_accuracies: dict[int, float] = {}
        self.plot_errors: list[dict[str, str]] = []
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "running",
            "eligible_for_aggregation": False,
            "start_time_utc": _utc_now(),
            "completed_folds": [],
            **dict(run_metadata),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        _write_json(self.output_dir / "run_manifest.json", self.manifest)

    def on_fold_start(self, fold: int, seed: int, train_size: int, validation_size: int) -> None:
        fold = int(fold)
        self.fold_started_monotonic[fold] = time.perf_counter()
        self.fold_info[fold] = {
            "seed": int(seed),
            "train_size": int(train_size),
            "validation_size": int(validation_size),
        }
        self.histories.setdefault(fold, [])
        print(
            f"Fold {fold + 1}/{self.total_folds} started "
            f"(seed={seed}, train={train_size}, validation={validation_size})",
            flush=True,
        )

    def _fold_metrics_payload(self, fold: int) -> dict[str, Any]:
        info = self.fold_info[fold]
        history = self.histories.get(fold, [])
        payload: dict[str, Any] = {
            "schema_version": 2,
            "fold": fold,
            "seed": info["seed"],
            "train_size": info["train_size"],
            "validation_size": info["validation_size"],
            "epochs": self.total_epochs,
            "epochs_requested": self.total_epochs,
            "epochs_completed": len(history),
            "epoch_history": history,
        }
        if history:
            final = history[-1]
            payload["train"] = {
                "loss": final["train_loss"],
                "accuracy": final["train_accuracy"],
            }
            payload["validation"] = {
                "loss": final["validation_loss"],
                "accuracy": final["validation_accuracy"],
            }
            payload["final_epoch"] = int(final["epoch"])
            payload["final_validation_accuracy"] = final["validation_accuracy"]
        return payload

    def _write_fold_metrics(self, fold: int) -> None:
        _write_json(
            self.output_dir / f"fold_{fold}" / "metrics.json",
            self._fold_metrics_payload(fold),
        )

    def on_epoch_end(
        self,
        fold: int,
        epoch: int,
        train_metrics: Mapping[str, Any],
        validation_metrics: Mapping[str, Any],
        epoch_seconds: float,
    ) -> None:
        fold = int(fold)
        epoch = int(epoch)
        fold_elapsed = time.perf_counter() - self.fold_started_monotonic[fold]
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": _metric(train_metrics, "loss"),
            "train_accuracy": _metric(train_metrics, "accuracy"),
            "validation_loss": _metric(validation_metrics, "loss"),
            "validation_accuracy": _metric(validation_metrics, "accuracy"),
            "epoch_seconds": float(epoch_seconds),
            "fold_elapsed_seconds": float(fold_elapsed),
        }
        self.histories.setdefault(fold, []).append(record)
        self._write_fold_metrics(fold)
        print(
            f"Fold {fold + 1}/{self.total_folds} | "
            f"epoch {epoch}/{self.total_epochs} | "
            f"train_loss={record['train_loss']:.6f} | "
            f"val_loss={record['validation_loss']:.6f} | "
            f"val_accuracy={record['validation_accuracy']:.6f} | "
            f"epoch={record['epoch_seconds']:.2f}s | "
            f"fold_elapsed={record['fold_elapsed_seconds']:.2f}s",
            flush=True,
        )

    def on_fold_complete(self, fold: int, validation_metrics: Mapping[str, Any]) -> None:
        fold = int(fold)
        history = self.histories.get(fold, [])
        if len(history) != self.total_epochs:
            raise ValueError(
                f"training observer: fold {fold} completed with "
                f"{len(history)} of {self.total_epochs} epochs"
            )
        accuracy = _metric(validation_metrics, "accuracy")
        self.fold_accuracies[fold] = accuracy
        self._plot_training_curves(fold)
        completed = sorted(self.fold_accuracies)
        self.manifest["completed_folds"] = completed
        self._write_manifest()
        running_mean = sum(self.fold_accuracies.values()) / len(self.fold_accuracies)
        fold_elapsed = time.perf_counter() - self.fold_started_monotonic[fold]
        print(
            f"Fold {fold + 1}/{self.total_folds} complete | "
            f"final_epoch_accuracy={accuracy:.6f} | "
            f"fold_elapsed={fold_elapsed:.2f}s | "
            f"running_mean={running_mean:.6f} (informational)",
            flush=True,
        )

    def _record_plot_error(self, plot_name: str, error: BaseException) -> None:
        detail = {"plot": plot_name, "error_type": type(error).__name__, "error": str(error)}
        self.plot_errors.append(detail)
        print(f"WARNING: observation plot {plot_name!r} was not generated: {error}", flush=True)

    def _plot_training_curves(self, fold: int) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            history = self.histories[fold]
            epochs = [int(record["epoch"]) for record in history]
            figure, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
            axes[0].plot(epochs, [record["train_loss"] for record in history], label="train loss")
            axes[0].set_ylabel("loss")
            axes[0].legend()
            axes[1].plot(epochs, [record["validation_loss"] for record in history], label="validation loss")
            axes[1].set_ylabel("loss")
            axes[1].legend()
            axes[2].plot(epochs, [record["validation_accuracy"] for record in history], label="validation accuracy")
            axes[2].set_xlabel("epoch")
            axes[2].set_ylabel("accuracy")
            axes[2].legend()
            figure.suptitle(f"RTC fold {fold} training curves")
            figure.tight_layout()
            figure.savefig(self.output_dir / f"fold_{fold}" / "training_curves.png", dpi=150)
            plt.close(figure)
        except Exception as error:  # plotting is observation-only and fail-safe
            self._record_plot_error(f"fold_{fold}/training_curves.png", error)

    def complete_run(self, summary: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
        completed = sorted(self.fold_accuracies)
        expected = list(range(self.total_folds))
        if completed != expected:
            raise ValueError(f"training observer: cannot complete run; completed folds={completed}")
        if any(len(self.histories.get(fold, [])) != self.total_epochs for fold in expected):
            raise ValueError("training observer: cannot complete run; an epoch history is incomplete")

        accuracies = [self.fold_accuracies[fold] for fold in expected]
        mean_accuracy = sum(accuracies) / len(accuracies)
        std_accuracy = math.sqrt(sum((value - mean_accuracy) ** 2 for value in accuracies) / len(accuracies))
        elapsed = time.perf_counter() - self.run_started_monotonic
        final_summary = dict(summary)
        final_summary.update(
            {
                "status": "complete",
                "run_id": self.run_id,
                "fold_accuracies": accuracies,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
                "total_time_seconds": elapsed,
                "protocol": dict(protocol),
            }
        )
        _write_json(self.output_dir / "summary.json", final_summary)
        self._plot_fold_accuracy(accuracies, mean_accuracy)
        self.manifest.update(
            {
                "status": "complete",
                "eligible_for_aggregation": True,
                "completed_folds": expected,
                "completion_time_utc": _utc_now(),
                "total_time_seconds": elapsed,
                "plot_errors": self.plot_errors,
            }
        )
        self._write_manifest()
        print(f"RTC formal run complete in {elapsed:.2f}s; summary written after all folds.", flush=True)
        return final_summary

    def complete_preflight(self) -> None:
        """Finalize fold-0 observation artifacts without formal aggregation."""

        completed = sorted(self.fold_accuracies)
        if completed != [0]:
            raise ValueError(
                f"training observer: cannot complete preflight; completed folds={completed}"
            )
        if len(self.histories.get(0, [])) != self.total_epochs:
            raise ValueError("training observer: cannot complete preflight; epoch history is incomplete")
        elapsed = time.perf_counter() - self.run_started_monotonic
        self.manifest.update(
            {
                "status": "preflight_complete",
                "eligible_for_aggregation": False,
                "completed_folds": [0],
                "completion_time_utc": _utc_now(),
                "total_time_seconds": elapsed,
                "plot_errors": self.plot_errors,
            }
        )
        self._write_manifest()
        print(
            f"RTC observation-only preflight complete in {elapsed:.2f}s; no formal summary written.",
            flush=True,
        )

    def _plot_fold_accuracy(self, accuracies: list[float], mean_accuracy: float) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 5))
            folds = list(range(len(accuracies)))
            axis.bar(folds, accuracies)
            axis.axhline(mean_accuracy, color="tab:red", linestyle="--", label=f"mean={mean_accuracy:.4f}")
            axis.set_xlabel("fold")
            axis.set_ylabel("accuracy")
            axis.set_title("RTC 10-fold validation accuracy")
            axis.set_xticks(folds)
            axis.legend()
            figure.tight_layout()
            figure.savefig(self.output_dir / "fold_accuracy.png", dpi=150)
            plt.close(figure)
        except Exception as error:  # plotting is observation-only and fail-safe
            self._record_plot_error("fold_accuracy.png", error)

    def mark_aborted(self, error: BaseException) -> None:
        self.manifest.update(
            {
                "status": "aborted",
                "eligible_for_aggregation": False,
                "completed_folds": sorted(self.fold_accuracies),
                "completion_time_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                "plot_errors": self.plot_errors,
            }
        )
        self._write_manifest()
