from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from TCN_Singh2024.src import cv as cv_module
from TCN_Singh2024.src import rtc_runner
from TCN_Singh2024.src.training_observer import RtcTrainingObserver


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rtc_reproduction.yaml"


@pytest.fixture
def fake_generated_root(monkeypatch, tmp_path):
    fake_runner = tmp_path / "fake_repo" / "TCN_Singh2024" / "src" / "rtc_runner.py"
    fake_runner.parent.mkdir(parents=True)
    generated_root = fake_runner.parents[1] / "results" / "generated"
    generated_root.mkdir(parents=True)
    monkeypatch.setattr(rtc_runner, "__file__", str(fake_runner))
    return generated_root


class _TinyDataset(Dataset):
    def __init__(self, count: int = 10) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sequence": torch.ones(2 + index % 3, 3),
            "label": index % 2,
        }


class _RecordingObserver:
    def __init__(self) -> None:
        self.fold_starts: list[tuple[int, int, int, int]] = []
        self.epoch_records: list[tuple[int, int, dict[str, float], dict[str, float]]] = []
        self.fold_completions: list[tuple[int, dict[str, float]]] = []

    def on_fold_start(self, fold: int, seed: int, train_size: int, validation_size: int) -> None:
        self.fold_starts.append((fold, seed, train_size, validation_size))

    def on_epoch_end(
        self,
        fold: int,
        epoch: int,
        train_metrics: dict[str, float],
        validation_metrics: dict[str, float],
        epoch_seconds: float,
    ) -> None:
        assert epoch_seconds >= 0.0
        self.epoch_records.append((fold, epoch, train_metrics, validation_metrics))

    def on_fold_complete(self, fold: int, validation_metrics: dict[str, float]) -> None:
        self.fold_completions.append((fold, validation_metrics))


def _locked_config() -> dict[str, object]:
    return rtc_runner.load_rtc_config(CONFIG_PATH)


def _synthetic_protocol() -> rtc_runner.RtcProtocolData:
    source_indices = tuple(range(10))
    return rtc_runner.RtcProtocolData(
        dataset=_TinyDataset(10),
        source_indices=source_indices,
        labels=tuple(index % 2 for index in source_indices),
        folds=rtc_runner.plan_rtc_folds(source_indices, folds=10, seed=42),
        test_sample_count=5552,
        test_used=False,
    )


def test_observer_enabled_preserves_final_epoch_metric_without_best_selection(monkeypatch, tmp_path):
    def model_factory() -> nn.Module:
        model = nn.Linear(3, 2)
        model.synthetic_epoch = 0
        return model

    def fake_train(model, loader, optimizer, criterion, device):
        del loader, optimizer, criterion, device
        model.synthetic_epoch += 1
        return {"loss": 1.0 / model.synthetic_epoch, "accuracy": 0.1 * model.synthetic_epoch}

    def fake_evaluate(model, loader, criterion, device, num_classes):
        del loader, criterion, device, num_classes
        accuracy_by_epoch = {1: 0.1, 2: 0.9, 3: 0.2}
        accuracy = accuracy_by_epoch[model.synthetic_epoch]
        return {"loss": 1.0 - accuracy, "accuracy": accuracy}

    monkeypatch.setattr(cv_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(cv_module, "evaluate", fake_evaluate)

    disabled = cv_module.run_cross_validation(
        dataset=_TinyDataset(4),
        model_factory=model_factory,
        batch_size=2,
        learning_rate=1e-3,
        epochs=3,
        folds=2,
        seed=42,
        output_dir=tmp_path / "disabled",
        device="cpu",
        num_classes=2,
    )
    observer = _RecordingObserver()
    enabled = cv_module.run_cross_validation(
        dataset=_TinyDataset(4),
        model_factory=model_factory,
        batch_size=2,
        learning_rate=1e-3,
        epochs=3,
        folds=2,
        seed=42,
        output_dir=tmp_path / "enabled",
        device="cpu",
        num_classes=2,
        observer=observer,
    )

    assert enabled["fold_accuracies"] == disabled["fold_accuracies"] == [0.2, 0.2]
    assert [record[3]["accuracy"] for record in observer.epoch_records if record[0] == 0] == [0.1, 0.9, 0.2]
    assert not hasattr(observer, "best")
    assert not list((tmp_path / "enabled").rglob("*checkpoint*"))


def test_observer_records_each_epoch_metrics_and_duration(tmp_path):
    observer = RtcTrainingObserver(
        tmp_path,
        run_metadata={"synthetic": True},
        total_folds=1,
        total_epochs=2,
    )
    observer.on_fold_start(0, 42, 4, 2)
    observer.on_epoch_end(
        0,
        1,
        {"loss": 1.0, "accuracy": 0.25},
        {"loss": 0.8, "accuracy": 0.5},
        0.125,
    )
    observer.on_epoch_end(
        0,
        2,
        {"loss": 0.7, "accuracy": 0.75},
        {"loss": 0.4, "accuracy": 1.0},
        0.250,
    )

    payload = json.loads((tmp_path / "fold_0" / "metrics.json").read_text(encoding="utf-8"))
    history = payload["epoch_history"]
    assert [record["epoch"] for record in history] == [1, 2]
    assert [record["validation_accuracy"] for record in history] == [0.5, 1.0]
    assert [record["epoch_seconds"] for record in history] == [0.125, 0.25]
    assert all(record["fold_elapsed_seconds"] >= 0.0 for record in history)
    assert payload["epochs_completed"] == 2


def test_test_split_cannot_reach_cv_runner(monkeypatch, fake_generated_root):
    main_dataset = _TinyDataset(10)
    test_dataset = object()
    protocol = SimpleNamespace(
        dataset=main_dataset,
        source_indices=tuple(range(10)),
        folds=rtc_runner.plan_rtc_folds(tuple(range(10)), folds=10, seed=42),
        test_sample_count=5552,
        test_used=False,
    )
    calls: dict[str, object] = {}

    class FakeObserver:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def complete_run(self, summary, protocol):
            del protocol
            return summary

        def mark_aborted(self, error):
            raise AssertionError(f"unexpected abort: {error}")

    def fake_prepare(config, *, repository_root=None):
        del config, repository_root
        return protocol

    def fake_cv(**kwargs):
        calls["kwargs"] = kwargs
        return {"mean_accuracy": 0.0}

    monkeypatch.setattr(rtc_runner, "RtcTrainingObserver", FakeObserver)
    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", fake_prepare)
    monkeypatch.setattr(rtc_runner, "write_protocol_manifest", lambda *args: None)
    monkeypatch.setattr(rtc_runner, "_annotate_fold_source_indices", lambda *args: None)
    monkeypatch.setattr(rtc_runner, "run_cross_validation", fake_cv)

    result = rtc_runner.run_rtc_cross_validation(
        _locked_config(),
        output_dir=fake_generated_root / "test_split",
        device="cpu",
    )

    assert result == {"mean_accuracy": 0.0}
    kwargs = calls["kwargs"]
    assert kwargs["dataset"] is main_dataset
    assert kwargs["dataset"] is not test_dataset
    assert "test" not in kwargs
    assert protocol.test_used is False


def test_output_path_must_be_generated_and_reused_nonempty_dir_is_rejected(
    fake_generated_root, tmp_path
):
    allowed = fake_generated_root / "rtc_baseline"
    assert rtc_runner._ensure_fresh_output_dir(allowed) == allowed.resolve()

    reused = fake_generated_root / "reused"
    reused.mkdir(parents=True)
    (reused / "sentinel.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh empty output directory"):
        rtc_runner._ensure_fresh_output_dir(reused)

    rejected_paths = (
        fake_generated_root / ".." / "reported" / "foo",
        fake_generated_root.parent / "not_generated" / "rtc_baseline",
        fake_generated_root.parents[1] / "src" / "output",
        tmp_path / "external" / "output",
    )
    for rejected in rejected_paths:
        with pytest.raises(ValueError, match="generated"):
            rtc_runner._ensure_fresh_output_dir(rejected)


def test_exception_marks_manifest_aborted_without_formal_summary(
    monkeypatch, fake_generated_root
):
    def fail_prepare(config, *, repository_root=None):
        del config, repository_root
        raise RuntimeError("synthetic preparation failure")

    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", fail_prepare)

    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        rtc_runner.run_rtc_cross_validation(
            _locked_config(),
            output_dir=fake_generated_root / "aborted",
            device="cpu",
        )

    output_dir = fake_generated_root / "aborted"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert manifest["eligible_for_aggregation"] is False
    assert not (output_dir / "summary.json").exists()


def test_complete_mocked_ten_fold_run_writes_formal_observation_artifacts(
    monkeypatch, fake_generated_root
):
    protocol = _synthetic_protocol()
    config = _locked_config()

    def fake_prepare(config_value, *, repository_root=None):
        del config_value, repository_root
        return protocol

    def fake_cv(**kwargs):
        observer = kwargs["observer"]
        output_dir = Path(kwargs["output_dir"])
        for plan in protocol.folds:
            observer.on_fold_start(plan.fold, plan.seed, len(plan.train_indices), len(plan.validation_indices))
            accuracy = 0.5 + plan.fold / 100.0
            for epoch in range(1, kwargs["epochs"] + 1):
                observer.on_epoch_end(
                    plan.fold,
                    epoch,
                    {"loss": 1.0 / epoch, "accuracy": accuracy},
                    {"loss": 0.8 / epoch, "accuracy": accuracy},
                    0.001,
                )
            observer.on_fold_complete(plan.fold, {"loss": 0.1, "accuracy": accuracy})
            metrics_path = output_dir / f"fold_{plan.fold}" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["validation_indices"] = list(plan.validation_indices)
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        return {
            "number_of_folds": 10,
            "fold_accuracies": [0.5 + fold / 100.0 for fold in range(10)],
            "mean_accuracy": 0.545,
        }

    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", fake_prepare)
    monkeypatch.setattr(rtc_runner, "run_cross_validation", fake_cv)

    output_dir = fake_generated_root / "complete"
    result = rtc_runner.run_rtc_cross_validation(config, output_dir=output_dir, device="cpu")

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert manifest["status"] == "complete"
    assert manifest["eligible_for_aggregation"] is True
    assert manifest["completed_folds"] == list(range(10))
    assert summary["status"] == "complete"
    assert summary["fold_accuracies"] == [0.5 + fold / 100.0 for fold in range(10)]
    assert (output_dir / "fold_accuracy.png").is_file()
    assert all((output_dir / f"fold_{fold}" / "training_curves.png").is_file() for fold in range(10))
    assert all((output_dir / f"fold_{fold}" / "metrics.json").is_file() for fold in range(10))


def test_observer_callback_preserves_fold_seeds_model_factory_and_adam_semantics(monkeypatch, tmp_path):
    models: list[nn.Module] = []
    optimizer_calls: list[tuple[tuple[nn.Parameter, ...], float]] = []
    optimizer_objects: list[object] = []
    seed_calls: list[int] = []
    train_optimizer_calls: list[object] = []

    def model_factory() -> nn.Module:
        model = nn.Linear(3, 2)
        models.append(model)
        return model

    def fake_adam(parameters, *, lr):
        params = tuple(parameters)
        optimizer_calls.append((params, lr))
        optimizer = object()
        optimizer_objects.append(optimizer)
        return optimizer

    def fake_train(model, loader, optimizer, criterion, device):
        del model, loader, criterion, device
        train_optimizer_calls.append(optimizer)
        return {"loss": 0.5, "accuracy": 0.5}

    monkeypatch.setattr(cv_module, "set_seed", lambda value: seed_calls.append(value))
    monkeypatch.setattr(cv_module.torch.optim, "Adam", fake_adam)
    monkeypatch.setattr(cv_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(cv_module, "evaluate", lambda *args, **kwargs: {"loss": 0.4, "accuracy": 0.6})

    observer = _RecordingObserver()
    cv_module.run_cross_validation(
        dataset=_TinyDataset(10),
        model_factory=model_factory,
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        folds=5,
        seed=42,
        output_dir=tmp_path,
        device="cpu",
        num_classes=2,
        observer=observer,
    )

    assert seed_calls == [42, 43, 44, 45, 46]
    assert len(models) == len(optimizer_calls) == 5
    assert len({id(model) for model in models}) == 5
    assert all(lr == pytest.approx(1e-3) for _, lr in optimizer_calls)
    assert all(optimizer is train_optimizer_calls[index] for index, optimizer in enumerate(optimizer_objects))
    assert len(observer.fold_starts) == 5
    assert [record[1] for record in observer.fold_starts] == [42, 43, 44, 45, 46]


def test_device_telemetry_failure_is_nonfatal_for_observer(
    monkeypatch, fake_generated_root
):
    def telemetry_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic device telemetry failure")

    protocol = _synthetic_protocol()

    observer_metadata: dict[str, object] = {}

    class FakeObserver:
        def __init__(self, *args, **kwargs):
            del args
            observer_metadata.update(kwargs["run_metadata"])

        def complete_run(self, summary, protocol):
            del protocol
            return summary

        def mark_aborted(self, error):
            raise AssertionError(f"unexpected abort: {error}")

    monkeypatch.setattr(rtc_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rtc_runner.torch.cuda, "get_device_name", telemetry_failure)
    monkeypatch.setattr(rtc_runner, "RtcTrainingObserver", FakeObserver)
    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(rtc_runner, "write_protocol_manifest", lambda *args: None)
    monkeypatch.setattr(rtc_runner, "_annotate_fold_source_indices", lambda *args: None)
    monkeypatch.setattr(rtc_runner, "run_cross_validation", lambda **kwargs: {"mean_accuracy": 0.0})

    result = rtc_runner.run_rtc_cross_validation(
        _locked_config(),
        output_dir=fake_generated_root / "telemetry",
        device="cuda",
    )

    assert result == {"mean_accuracy": 0.0}
    assert observer_metadata["cuda_device_name"] is None


def test_true_training_error_propagates_after_observer_abort(monkeypatch, fake_generated_root):
    protocol = _synthetic_protocol()
    aborted_errors: list[BaseException] = []

    class FakeObserver:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def mark_aborted(self, error):
            aborted_errors.append(error)

    monkeypatch.setattr(rtc_runner, "RtcTrainingObserver", FakeObserver)
    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(rtc_runner, "write_protocol_manifest", lambda *args: None)

    def raise_training_error(**kwargs):
        del kwargs
        raise RuntimeError("sentinel true training error")

    monkeypatch.setattr(rtc_runner, "run_cross_validation", raise_training_error)

    with pytest.raises(RuntimeError, match="sentinel true training error") as caught:
        rtc_runner.run_rtc_cross_validation(
            _locked_config(),
            output_dir=fake_generated_root / "training-error",
            device="cpu",
        )

    assert len(aborted_errors) == 1
    assert aborted_errors[0] is caught.value
