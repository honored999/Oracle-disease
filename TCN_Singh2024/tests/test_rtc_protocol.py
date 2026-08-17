from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from TCN_Singh2024.src import cv as cv_module
from TCN_Singh2024.src import rtc_runner


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rtc_reproduction.yaml"


@pytest.fixture
def fake_generated_root(monkeypatch, tmp_path):
    fake_runner = tmp_path / "fake_repo" / "TCN_Singh2024" / "src" / "rtc_runner.py"
    fake_runner.parent.mkdir(parents=True)
    generated_root = fake_runner.parents[1] / "results" / "generated"
    generated_root.mkdir(parents=True)
    monkeypatch.setattr(rtc_runner, "__file__", str(fake_runner))
    return generated_root


def _locked_config() -> dict[str, object]:
    return rtc_runner.load_rtc_config(CONFIG_PATH)


def _synthetic_raw_arrays(
    sample_count: int = 20,
    excluded_source_indices: tuple[int, ...] = (1, 4),
) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((sample_count, 6), dtype=np.float32)
    labels = np.zeros((sample_count, 26), dtype=np.float32)
    excluded = set(excluded_source_indices)
    for source_index in range(sample_count):
        label = source_index % 26
        labels[source_index, label] = 1.0
        if source_index not in excluded:
            features[source_index, :3] = (source_index + 1, source_index + 2, source_index + 3)
    return features, labels


def _small_spec() -> rtc_runner.RtcProtocolSpec:
    return rtc_runner.RtcProtocolSpec(
        main_raw_samples=20,
        excluded_source_indices=(1, 4),
        expected_cv_samples=18,
        test_expected_samples=5,
        folds=10,
        seed=42,
        input_channels=3,
        num_classes=26,
    )


def test_locked_config_records_main_count_exclusions_cv_count_test_and_model_contract():
    config = _locked_config()
    spec = rtc_runner.RtcProtocolSpec.from_config(config)

    assert spec.main_raw_samples == 20098
    assert list(spec.excluded_source_indices) == [15227, 19086]
    assert spec.expected_cv_samples == 20096
    assert spec.main_raw_samples - len(spec.excluded_source_indices) == spec.expected_cv_samples
    assert spec.test_expected_samples == 5552
    assert spec.input_channels == 3
    assert spec.num_classes == 26
    assert spec.folds == 10


def test_raw_boundary_validation_rejects_an_unapproved_empty_sample():
    features, labels = _synthetic_raw_arrays()
    features[7] = 0.0

    with pytest.raises(ValueError, match="all-zero Main source indices mismatch"):
        rtc_runner.validate_main_raw_arrays(
            features,
            labels,
            expected_samples=20,
            excluded_source_indices=(1, 4),
            expected_cv_samples=18,
            expected_excluded_labels={1: 1, 4: 4},
        )


def test_exclusions_happen_before_parsing_preprocessing_and_fold_construction(monkeypatch):
    features, labels = _synthetic_raw_arrays()
    parsed_indices: list[int] = []
    preprocessed_indices: list[int] = []
    fold_source_indices: list[tuple[int, ...]] = []
    original_plan = rtc_runner.plan_rtc_folds

    def sentinel_trim(row: np.ndarray, source_index: int) -> np.ndarray:
        assert source_index not in {1, 4}
        parsed_indices.append(source_index)
        return row[:3]

    def sentinel_preprocess(sequence: torch.Tensor, preprocess: bool) -> torch.Tensor:
        assert preprocess is True
        preprocessed_indices.append(len(preprocessed_indices))
        return sequence

    def recording_plan(source_indices, *, folds, seed):
        fold_source_indices.append(tuple(source_indices))
        return original_plan(source_indices, folds=folds, seed=seed)

    monkeypatch.setattr(rtc_runner, "_rtc_trim_trailing_scalar_zeros", sentinel_trim)
    monkeypatch.setattr(rtc_runner, "_apply_preprocessing", sentinel_preprocess)
    monkeypatch.setattr(rtc_runner, "plan_rtc_folds", recording_plan)

    protocol = rtc_runner.prepare_rtc_protocol_from_arrays(
        features,
        labels,
        test_sample_count=5,
        config={},
        spec=_small_spec(),
        expected_excluded_labels={1: 1, 4: 4},
    )

    usable = tuple(index for index in range(20) if index not in {1, 4})
    assert parsed_indices == list(usable)
    assert len(preprocessed_indices) == len(usable)
    assert fold_source_indices == [usable]
    assert 1 not in parsed_indices and 4 not in parsed_indices
    assert 1 not in fold_source_indices[0] and 4 not in fold_source_indices[0]
    assert len(protocol.source_indices) == 18


def test_prepared_main_pool_preserves_original_source_indices_and_test_is_held_out():
    features, labels = _synthetic_raw_arrays()
    protocol = rtc_runner.prepare_rtc_protocol_from_arrays(
        features,
        labels,
        test_sample_count=5,
        config={},
        spec=_small_spec(),
        expected_excluded_labels={1: 1, 4: 4},
    )
    test_source_indices = set(range(100, 105))

    assert protocol.source_indices == tuple(index for index in range(20) if index not in {1, 4})
    assert protocol.test_sample_count == 5
    assert protocol.test_used is False
    assert all(
        test_source_indices.isdisjoint(plan.train_source_indices)
        and test_source_indices.isdisjoint(plan.validation_source_indices)
        for plan in protocol.folds
    )
    for local_index, source_index in enumerate(protocol.source_indices):
        assert protocol.dataset[local_index]["metadata"]["source_index"] == source_index


def test_test_count_integrity_check_is_explicit_and_does_not_require_parsing():
    test_features = np.ones((5, 6), dtype=np.float32)
    test_labels = np.zeros((5, 26), dtype=np.float32)
    test_labels[:, 0] = 1.0

    assert rtc_runner.validate_test_raw_count(test_features, test_labels, expected_samples=5) == 5
    with pytest.raises(ValueError, match="Test raw sample count mismatch"):
        rtc_runner.validate_test_raw_count(test_features[:4], test_labels[:4], expected_samples=5)


def test_ten_sample_level_folds_cover_each_usable_sample_once_without_overlap():
    source_indices = tuple(index for index in range(20) if index not in {1, 4})
    plans = rtc_runner.plan_rtc_folds(source_indices, folds=10, seed=42)

    assert len(plans) == 10
    validation_indices = [index for plan in plans for index in plan.validation_indices]
    assert sorted(validation_indices) == list(range(len(source_indices)))
    assert len(validation_indices) == len(set(validation_indices)) == len(source_indices)
    assert all(not set(plan.train_indices).intersection(plan.validation_indices) for plan in plans)
    assert all(
        1 not in plan.train_source_indices
        and 1 not in plan.validation_source_indices
        and 4 not in plan.train_source_indices
        and 4 not in plan.validation_source_indices
        for plan in plans
    )


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"sequence": torch.ones(1, 3), "label": index % 2}


class _FormalPreflightDataset(Dataset):
    """Small-item synthetic dataset retaining the locked Main source IDs."""

    def __init__(self, source_indices: tuple[int, ...]) -> None:
        self.source_indices = source_indices

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sequence": torch.ones(2, 3),
            "label": self.source_indices[index] % 26,
        }


class _PreflightTinyModel(nn.Module):
    """Parameter-bearing model whose .to() never requires a real accelerator."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def to(self, device):
        del device
        return self


def _formal_preflight_protocol() -> rtc_runner.RtcProtocolData:
    source_indices = tuple(
        index for index in range(20098) if index not in {15227, 19086}
    )
    return rtc_runner.RtcProtocolData(
        dataset=_FormalPreflightDataset(source_indices),
        source_indices=source_indices,
        labels=tuple(index % 26 for index in source_indices),
        folds=rtc_runner.plan_rtc_folds(source_indices, folds=10, seed=42),
        test_sample_count=5552,
        test_used=False,
    )


def _patch_preflight_dependencies(monkeypatch, protocol, seen):
    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", lambda *args, **kwargs: protocol)
    monkeypatch.setattr(rtc_runner, "TCNClassifier", _PreflightTinyModel)
    monkeypatch.setattr(rtc_runner, "set_seed", lambda seed: seen.setdefault("seeds", []).append(seed))

    def fake_train(model, loader, optimizer, criterion, device):
        del model, optimizer, criterion, device
        subset = loader.dataset
        seen["train_indices"] = tuple(subset.indices)
        seen["train_source_indices"] = tuple(
            subset.dataset.source_indices[index] for index in subset.indices
        )
        seen["train_batch_size"] = loader.batch_size
        seen["train_calls"] = seen.get("train_calls", 0) + 1
        return {"loss": 0.75, "accuracy": 0.25}

    def fake_evaluate(model, loader, criterion, device, num_classes):
        del model, criterion, device, num_classes
        subset = loader.dataset
        seen["validation_indices"] = tuple(subset.indices)
        seen["validation_source_indices"] = tuple(
            subset.dataset.source_indices[index] for index in subset.indices
        )
        seen["validation_calls"] = seen.get("validation_calls", 0) + 1
        return {"loss": 0.5, "accuracy": 0.5}

    monkeypatch.setattr(rtc_runner, "train_one_epoch", fake_train)
    monkeypatch.setattr(rtc_runner, "evaluate", fake_evaluate)


@pytest.mark.parametrize(
    "argv",
    [
        ["--preflight", "--run-cv"],
        ["--preflight", "--manifest-only"],
    ],
)
def test_preflight_cli_is_distinct_and_mutually_exclusive(argv):
    with pytest.raises(SystemExit) as caught:
        rtc_runner.main(argv)

    assert caught.value.code == 2


def test_preflight_locks_formal_fold_zero_one_epoch_and_observation_artifacts(
    monkeypatch, fake_generated_root
):
    protocol = _formal_preflight_protocol()
    plan = protocol.folds[0]
    seen: dict[str, object] = {}
    _patch_preflight_dependencies(monkeypatch, protocol, seen)

    def fake_plot(observer, fold):
        seen["plot_calls"] = seen.get("plot_calls", 0) + 1
        path = observer.output_dir / f"fold_{fold}" / "training_curves.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic observation curve")

    monkeypatch.setattr(
        rtc_runner.RtcTrainingObserver,
        "_plot_training_curves",
        fake_plot,
    )

    output_dir = fake_generated_root / "rtc_preflight"
    result = rtc_runner.run_rtc_preflight(
        _locked_config(),
        output_dir=output_dir,
        device="cpu",
    )

    assert result["status"] == "preflight_complete"
    assert result["mode"] == "preflight"
    assert result["formal_result"] is False
    assert result["fold_id"] == 0
    assert result["epochs"] == 1
    assert seen["seeds"] == [42]
    assert seen["train_calls"] == seen["validation_calls"] == 1
    assert seen["train_indices"] == plan.train_indices
    assert seen["validation_indices"] == plan.validation_indices
    assert seen["train_source_indices"] == plan.train_source_indices
    assert seen["validation_source_indices"] == plan.validation_source_indices
    assert seen["train_batch_size"] == 32

    test_source_indices = set(range(20098, 25650))
    assert test_source_indices.isdisjoint(seen["train_source_indices"])
    assert test_source_indices.isdisjoint(seen["validation_source_indices"])

    preflight_manifest = json.loads(
        (output_dir / "preflight_manifest.json").read_text(encoding="utf-8")
    )
    assert preflight_manifest["mode"] == "preflight"
    assert preflight_manifest["formal_result"] is False
    assert preflight_manifest["completion_status"] == "preflight_complete"
    assert preflight_manifest["fold_id"] == 0
    assert preflight_manifest["epochs"] == 1
    assert preflight_manifest["folds"] == 10
    assert preflight_manifest["main_raw_samples"] == 20098
    assert preflight_manifest["excluded_source_indices"] == [15227, 19086]
    assert preflight_manifest["usable_cv_samples"] == 20096
    assert preflight_manifest["test_samples"] == 5552
    assert preflight_manifest["test_used_in_cv"] is False
    assert preflight_manifest["seed"] == 42
    assert preflight_manifest["fold_seed"] == 42
    assert preflight_manifest["train_indices"] == list(plan.train_indices)
    assert preflight_manifest["validation_indices"] == list(plan.validation_indices)
    assert preflight_manifest["train_source_indices"] == list(plan.train_source_indices)
    assert preflight_manifest["validation_source_indices"] == list(plan.validation_source_indices)
    assert preflight_manifest["start_time_utc"]
    assert preflight_manifest["end_time_utc"]
    assert preflight_manifest["artifacts"] == [
        "run_manifest.json",
        "fold_0/metrics.json",
        "fold_0/training_curves.png",
        "preflight_manifest.json",
    ]

    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["status"] == "preflight_complete"
    assert run_manifest["eligible_for_aggregation"] is False
    assert run_manifest["completed_folds"] == [0]
    metrics = json.loads(
        (output_dir / "fold_0" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["epochs"] == metrics["epochs_requested"] == 1
    assert metrics["epochs_completed"] == 1
    assert metrics["final_epoch"] == 1
    assert len(metrics["epoch_history"]) == 1
    assert metrics["validation"]["accuracy"] == pytest.approx(0.5)
    assert metrics["train_source_indices"] == list(plan.train_source_indices)
    assert metrics["validation_source_indices"] == list(plan.validation_source_indices)
    assert (output_dir / "fold_0" / "training_curves.png").is_file()
    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "fold_accuracy.png").exists()
    assert seen["plot_calls"] == 1


def test_preflight_telemetry_failure_is_nonfatal(monkeypatch, fake_generated_root):
    protocol = _formal_preflight_protocol()
    seen: dict[str, object] = {}
    _patch_preflight_dependencies(monkeypatch, protocol, seen)
    monkeypatch.setattr(rtc_runner, "_resolve_training_device", lambda requested: torch.device("cuda"))
    monkeypatch.setattr(rtc_runner.torch.cuda, "is_available", lambda: True)

    def telemetry_failure(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic telemetry failure")

    monkeypatch.setattr(rtc_runner.torch.cuda, "get_device_name", telemetry_failure)
    result = rtc_runner.run_rtc_preflight(
        _locked_config(),
        output_dir=fake_generated_root / "telemetry-preflight",
        device="cuda",
    )

    assert result["status"] == "preflight_complete"
    manifest = json.loads(
        (fake_generated_root / "telemetry-preflight" / "preflight_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["cuda_available"] is True
    assert manifest["cuda_device_name"] is None


def test_preflight_training_error_propagates_and_marks_aborted(monkeypatch, fake_generated_root):
    protocol = _formal_preflight_protocol()
    seen: dict[str, object] = {}
    _patch_preflight_dependencies(monkeypatch, protocol, seen)

    def raise_training_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("sentinel preflight training error")

    monkeypatch.setattr(rtc_runner, "train_one_epoch", raise_training_error)
    output_dir = fake_generated_root / "aborted-preflight"
    with pytest.raises(RuntimeError, match="sentinel preflight training error"):
        rtc_runner.run_rtc_preflight(
            _locked_config(),
            output_dir=output_dir,
            device="cpu",
        )

    preflight_manifest = json.loads(
        (output_dir / "preflight_manifest.json").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert preflight_manifest["completion_status"] == "aborted"
    assert preflight_manifest["formal_result"] is False
    assert preflight_manifest["error_type"] == "RuntimeError"
    assert "sentinel preflight training error" in preflight_manifest["error"]
    assert run_manifest["status"] == "aborted"
    assert run_manifest["eligible_for_aggregation"] is False
    assert seen.get("train_calls", 0) == 0


def test_runner_delegates_to_existing_cv_entry_point_without_using_test_data(
    monkeypatch, fake_generated_root
):
    dataset = _TinyDataset()
    protocol = SimpleNamespace(dataset=dataset)
    calls: dict[str, object] = {}

    def fake_prepare(config, *, repository_root=None):
        calls["prepared_config"] = config
        return protocol

    def fake_manifest(protocol_value, config, output_dir):
        calls["manifest_protocol"] = protocol_value
        return fake_generated_root / "protocol_manifest.json"

    def fake_cv(**kwargs):
        calls["cv_kwargs"] = kwargs
        models = [kwargs["model_factory"]() for _ in range(10)]
        calls["models"] = models
        return {"mean_accuracy": 0.0}

    monkeypatch.setattr(rtc_runner, "prepare_rtc_protocol", fake_prepare)
    monkeypatch.setattr(rtc_runner, "write_protocol_manifest", fake_manifest)
    monkeypatch.setattr(rtc_runner, "run_cross_validation", fake_cv)
    monkeypatch.setattr(rtc_runner, "_annotate_fold_source_indices", lambda *_: None)

    result = rtc_runner.run_rtc_cross_validation(
        _locked_config(),
        output_dir=fake_generated_root / "rtc_baseline",
    )

    kwargs = calls["cv_kwargs"]
    assert result == {"mean_accuracy": 0.0}
    assert calls["manifest_protocol"] is protocol
    assert kwargs["dataset"] is dataset
    assert kwargs["folds"] == 10
    assert kwargs["num_classes"] == 26
    assert kwargs["learning_rate"] == pytest.approx(1e-3)
    assert len({id(model) for model in calls["models"]}) == 10
    assert "test" not in kwargs


def test_existing_cv_creates_fresh_models_and_adam_optimizers_without_training(monkeypatch, tmp_path):
    models: list[nn.Module] = []
    optimizers: list[object] = []
    training_calls: list[object] = []

    def model_factory() -> nn.Module:
        model = nn.Linear(3, 2)
        models.append(model)
        return model

    def fake_adam(parameters, *, lr):
        optimizers.append((tuple(parameters), lr))
        return object()

    def fake_train(*args, **kwargs):
        training_calls.append(args[2])
        return {"loss": 0.0}

    monkeypatch.setattr(cv_module.torch.optim, "Adam", fake_adam)
    monkeypatch.setattr(cv_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(cv_module, "evaluate", lambda *args, **kwargs: {"accuracy": 0.0})

    summary = cv_module.run_cross_validation(
        dataset=_TinyDataset(),
        model_factory=model_factory,
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        folds=10,
        seed=42,
        output_dir=tmp_path,
        device="cpu",
        num_classes=2,
    )

    assert summary["number_of_folds"] == 10
    assert len(models) == 10
    assert len({id(model) for model in models}) == 10
    assert len(optimizers) == 10
    assert len({id(parameters) for parameters, _ in optimizers}) == 10
    assert all(lr == pytest.approx(1e-3) for _, lr in optimizers)
    assert len(training_calls) == 10
