"""Protocol-specific RTC preparation and future CV entry point.

This module deliberately sits beside the verified RTC adapter.  It validates
the locked Main/Test boundary from raw arrays first, removes only the two
approved Main source indices, and then reuses the adapter's trusted parsing
and preprocessing helpers.  It does not discover, concatenate, or preprocess
the Test split.

The command-line interface is safe by default: it performs no action unless a
caller explicitly requests ``--manifest-only``, ``--preflight``, or
``--run-cv``.  This module is an orchestration layer; formal fold construction
remains owned by :mod:`TCN_Singh2024.src.cv`, while preflight consumes the
audited fold-0 plan without formal aggregation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

from TCN_Singh2024.src.adapters import (
    ProvisionalTrajectoryDataset,
    _apply_preprocessing,
    _load_rtc_numpy_array,
    _rtc_one_hot_to_label,
    _rtc_trim_trailing_scalar_zeros,
)
from TCN_Singh2024.src.cv import make_kfold_splits, run_cross_validation
from TCN_Singh2024.src.dataset import collate_trajectory_batch
from TCN_Singh2024.src.evaluation import evaluate
from TCN_Singh2024.src.model import TCNClassifier
from TCN_Singh2024.src.train import load_config, set_seed, train_one_epoch
from TCN_Singh2024.src.training_observer import RtcTrainingObserver


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rtc_reproduction.yaml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "generated" / "rtc_baseline"
DEFAULT_PREFLIGHT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "generated" / "rtc_preflight"
DEFAULT_EXCLUDED_SOURCE_INDICES = (15227, 19086)
DEFAULT_EXCLUDED_LABELS = {15227: 7, 19086: 21}
DEFAULT_MAIN_RAW_SAMPLES = 20098
DEFAULT_TEST_SAMPLES = 5552
DEFAULT_USABLE_CV_SAMPLES = 20096
DEFAULT_FOLDS = 10
DEFAULT_SEED = 42


@dataclass(frozen=True)
class RtcProtocolSpec:
    """Locked data-boundary and fold settings read from the RTC YAML."""

    main_raw_samples: int
    excluded_source_indices: tuple[int, ...]
    expected_cv_samples: int
    test_expected_samples: int
    folds: int
    seed: int
    input_channels: int
    num_classes: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "RtcProtocolSpec":
        if config.get("dataset") != "RTC":
            raise ValueError("RTC protocol requires config dataset='RTC'")

        main = _mapping_value(config, "main")
        test = _mapping_value(config, "test")
        model = _mapping_value(config, "model")
        training = _mapping_value(config, "training")
        evaluation = _mapping_value(config, "evaluation")

        excluded = tuple(int(index) for index in _sequence_value(main, "excluded_source_indices"))
        if excluded != DEFAULT_EXCLUDED_SOURCE_INDICES:
            raise ValueError(
                "RTC protocol requires excluded_source_indices exactly "
                f"{list(DEFAULT_EXCLUDED_SOURCE_INDICES)}"
            )
        if int(main["raw_samples"]) != DEFAULT_MAIN_RAW_SAMPLES:
            raise ValueError("RTC protocol requires Main raw_samples=20098")
        if int(main["expected_cv_samples"]) != DEFAULT_USABLE_CV_SAMPLES:
            raise ValueError("RTC protocol requires expected_cv_samples=20096")
        if int(test["expected_samples"]) != DEFAULT_TEST_SAMPLES:
            raise ValueError("RTC protocol requires Test expected_samples=5552")
        if test.get("held_out") is not True or test.get("include_in_cv") is not False:
            raise ValueError("RTC Test must be held_out=true and include_in_cv=false")
        if int(model["input_channels"]) != 3 or int(model["num_classes"]) != 26:
            raise ValueError("RTC protocol requires input_channels=3 and num_classes=26")
        if [int(value) for value in _sequence_value(model, "hidden_channels")] != [32, 32, 32]:
            raise ValueError("RTC protocol requires hidden_channels=[32, 32, 32]")
        if int(model["kernel_size"]) != 3 or [int(value) for value in _sequence_value(model, "dilations")] != [1, 2, 4]:
            raise ValueError("RTC protocol requires kernel_size=3 and dilations=[1, 2, 4]")
        if float(model["dropout"]) != 0.2:
            raise ValueError("RTC protocol requires dropout=0.2")
        if training.get("optimizer") != "Adam" or float(training["learning_rate"]) != 1e-3:
            raise ValueError("RTC protocol requires Adam with learning_rate=1e-3")
        if int(training["batch_size"]) != 32 or int(training["epochs"]) != 20:
            raise ValueError("RTC protocol requires batch_size=32 and epochs=20")
        if config.get("loss") != "CrossEntropyLoss":
            raise ValueError("RTC protocol requires CrossEntropyLoss")
        if config.get("scheduler") != "none" or float(config.get("weight_decay", 0)) != 0.0:
            raise ValueError("RTC protocol requires no scheduler and weight_decay=0")
        if int(evaluation["folds"]) != DEFAULT_FOLDS:
            raise ValueError("RTC protocol requires evaluation folds=10")
        if int(training["seed"]) != DEFAULT_SEED:
            raise ValueError("RTC protocol requires training seed=42")
        if training.get("fold_seed_policy") != "base_seed_plus_fold":
            raise ValueError("RTC protocol requires fold_seed_policy='base_seed_plus_fold'")

        return cls(
            main_raw_samples=int(main["raw_samples"]),
            excluded_source_indices=excluded,
            expected_cv_samples=int(main["expected_cv_samples"]),
            test_expected_samples=int(test["expected_samples"]),
            folds=int(evaluation["folds"]),
            seed=int(training["seed"]),
            input_channels=int(model["input_channels"]),
            num_classes=int(model["num_classes"]),
        )


@dataclass(frozen=True)
class RtcFoldPlan:
    """One sample-level fold with local and original Main source indices."""

    fold: int
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_source_indices: tuple[int, ...]
    validation_source_indices: tuple[int, ...]


@dataclass(frozen=True)
class RtcProtocolData:
    """Prepared Main-only dataset and its source-preserving fold plan."""

    dataset: Dataset
    source_indices: tuple[int, ...]
    labels: tuple[int, ...]
    folds: tuple[RtcFoldPlan, ...]
    test_sample_count: int
    test_used: bool = False


def _mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"RTC config field {key!r} must be a mapping")
    return value


def _sequence_value(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = mapping.get(key)
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"RTC config field {key!r} must be a sequence")
    return value


def load_rtc_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the locked RTC reproduction configuration."""

    config = load_config(path)
    if not isinstance(config, dict):
        raise ValueError("RTC config must contain a YAML mapping")
    RtcProtocolSpec.from_config(config)
    return config


def _validate_numeric_array(array: np.ndarray, field_name: str) -> None:
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"RTC protocol: {field_name} must be a 2-D NumPy array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"RTC protocol: {field_name} must have a real numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"RTC protocol: {field_name} contains NaN or Inf")


def validate_main_raw_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    expected_samples: int = DEFAULT_MAIN_RAW_SAMPLES,
    excluded_source_indices: Sequence[int] = DEFAULT_EXCLUDED_SOURCE_INDICES,
    expected_cv_samples: int = DEFAULT_USABLE_CV_SAMPLES,
    expected_excluded_labels: Mapping[int, int] = DEFAULT_EXCLUDED_LABELS,
) -> tuple[int, ...]:
    """Validate locked Main provenance and return usable original indices.

    Validation occurs on fixed-width raw rows.  In particular, the all-zero
    check and exclusions happen before any trajectory trimming or
    preprocessing.  The default arguments encode the approved RTC protocol;
    explicit arguments also make this helper straightforward to exercise with
    small synthetic arrays in protocol tests.
    """

    _validate_numeric_array(features, "Main features")
    _validate_numeric_array(labels, "Main labels")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("RTC protocol: Main features and labels have different sample counts")
    if features.shape[0] != expected_samples:
        raise ValueError(
            f"RTC protocol: Main raw sample count mismatch; expected {expected_samples}, "
            f"found {features.shape[0]}"
        )
    if labels.shape[1] != 26:
        raise ValueError("RTC protocol: Main labels must have 26 classes")
    if features.shape[1] < 3:
        raise ValueError("RTC protocol: Main features must contain at least one XYZ triplet")

    excluded = tuple(int(index) for index in excluded_source_indices)
    excluded_set = set(excluded)
    if len(excluded_set) != len(excluded):
        raise ValueError("RTC protocol: excluded_source_indices must be unique")
    if not excluded_set.issubset(range(features.shape[0])):
        raise ValueError("RTC protocol: an excluded source index is outside the Main raw range")

    empty_indices = {
        index for index, row in enumerate(features) if bool(np.all(row == 0))
    }
    if empty_indices != excluded_set:
        raise ValueError(
            "RTC protocol: all-zero Main source indices mismatch; "
            f"expected {sorted(excluded_set)}, found {sorted(empty_indices)}"
        )

    for source_index, expected_label in expected_excluded_labels.items():
        if source_index not in excluded_set:
            raise ValueError(
                f"RTC protocol: expected label assertion index {source_index} is not excluded"
            )
        actual_label = _rtc_one_hot_to_label(labels[source_index], source_index)
        if actual_label != int(expected_label):
            raise ValueError(
                f"RTC protocol: excluded Main source_index={source_index} must have class "
                f"{expected_label}, found {actual_label}"
            )

    usable_indices = tuple(index for index in range(features.shape[0]) if index not in excluded_set)
    if len(usable_indices) != expected_cv_samples:
        raise ValueError(
            f"RTC protocol: usable Main count mismatch; expected {expected_cv_samples}, "
            f"found {len(usable_indices)}"
        )
    return usable_indices


def validate_test_raw_count(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    expected_samples: int = DEFAULT_TEST_SAMPLES,
) -> int:
    """Integrity-check Test count only; return it without parsing or using it."""

    _validate_numeric_array(features, "Test features")
    _validate_numeric_array(labels, "Test labels")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("RTC protocol: Test features and labels have different sample counts")
    if features.shape[0] != expected_samples:
        raise ValueError(
            f"RTC protocol: Test raw sample count mismatch; expected {expected_samples}, "
            f"found {features.shape[0]}"
        )
    return int(features.shape[0])


def resolve_rtc_raw_paths(
    config: Mapping[str, Any],
    repository_root: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve the four explicit official RTC raw paths without discovery."""

    data = _mapping_value(config, "data")
    root = Path(str(data.get("root", "data/RTC/raw")))
    if not root.is_absolute():
        base = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
        root = base / root
    paths = {
        "main_features": root / str(data.get("main_features_file", "features")),
        "main_labels": root / str(data.get("main_labels_file", "labels")),
        "test_features": root / str(data.get("test_features_file", "featuresTest")),
        "test_labels": root / str(data.get("test_labels_file", "labelsTest")),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(f"RTC protocol: missing official raw file(s): {missing}")
    return paths


def load_rtc_raw_splits(paths: Mapping[str, str | Path]) -> dict[str, np.ndarray]:
    """Load explicit Main/Test arrays with the adapter's restricted unpickler."""

    required = ("main_features", "main_labels", "test_features", "test_labels")
    missing = [key for key in required if key not in paths]
    if missing:
        raise ValueError(f"RTC protocol: raw path mapping is missing {missing}")
    return {
        "main_features": _load_rtc_numpy_array(paths["main_features"], "features"),
        "main_labels": _load_rtc_numpy_array(paths["main_labels"], "labels"),
        "test_features": _load_rtc_numpy_array(paths["test_features"], "featuresTest"),
        "test_labels": _load_rtc_numpy_array(paths["test_labels"], "labelsTest"),
    }


def plan_rtc_folds(
    source_indices: Sequence[int],
    *,
    folds: int = DEFAULT_FOLDS,
    seed: int = DEFAULT_SEED,
) -> tuple[RtcFoldPlan, ...]:
    """Plan sample-level folds via the existing CV splitter.

    Local indices are those consumed by ``cv.run_cross_validation``; the
    source-index fields retain the original Main row identities for audit.
    """

    source_tuple = tuple(int(index) for index in source_indices)
    split_pairs = make_kfold_splits(len(source_tuple), folds, seed, shuffle=True)
    plans = tuple(
        RtcFoldPlan(
            fold=fold,
            seed=seed + fold,
            train_indices=tuple(train_indices),
            validation_indices=tuple(validation_indices),
            train_source_indices=tuple(source_tuple[index] for index in train_indices),
            validation_source_indices=tuple(source_tuple[index] for index in validation_indices),
        )
        for fold, (train_indices, validation_indices) in enumerate(split_pairs)
    )
    validation_union = [index for plan in plans for index in plan.validation_indices]
    if sorted(validation_union) != list(range(len(source_tuple))):
        raise ValueError("RTC protocol: fold plan does not cover each usable Main sample exactly once")
    if any(set(plan.train_indices).intersection(plan.validation_indices) for plan in plans):
        raise ValueError("RTC protocol: train/validation fold overlap detected")
    return plans


def prepare_rtc_protocol_from_arrays(
    main_features: np.ndarray,
    main_labels: np.ndarray,
    *,
    test_sample_count: int,
    config: Mapping[str, Any],
    spec: RtcProtocolSpec | None = None,
    expected_excluded_labels: Mapping[int, int] | None = None,
) -> RtcProtocolData:
    """Build the Main-only prepared dataset after raw-boundary validation.

    The normal path derives and strictly validates ``spec`` from the locked
    YAML.  The explicit ``spec`` and ``expected_excluded_labels`` parameters
    are dependency-injection seams for small synthetic protocol tests; the
    CLI and raw-file path never use them.
    """

    spec = spec or RtcProtocolSpec.from_config(config)
    usable_source_indices = validate_main_raw_arrays(
        main_features,
        main_labels,
        expected_samples=spec.main_raw_samples,
        excluded_source_indices=spec.excluded_source_indices,
        expected_cv_samples=spec.expected_cv_samples,
        expected_excluded_labels=(
            DEFAULT_EXCLUDED_LABELS
            if expected_excluded_labels is None
            else expected_excluded_labels
        ),
    )
    if int(test_sample_count) != spec.test_expected_samples:
        raise ValueError(
            f"RTC protocol: Test raw sample count mismatch; expected {spec.test_expected_samples}, "
            f"found {test_sample_count}"
        )

    samples: list[dict[str, object]] = []
    labels: list[int] = []
    for source_index in usable_source_indices:
        valid_values = _rtc_trim_trailing_scalar_zeros(main_features[source_index], source_index)
        sequence = torch.tensor(valid_values, dtype=torch.float32).reshape(-1, spec.input_channels)
        label = _rtc_one_hot_to_label(main_labels[source_index], source_index)
        samples.append({
            "sequence": _apply_preprocessing(sequence, preprocess=True),
            "label": label,
            "metadata": {"source_index": source_index, "split": "main"},
        })
        labels.append(label)

    if len(samples) != spec.expected_cv_samples:
        raise ValueError(
            f"RTC protocol: prepared Main dataset count mismatch; expected {spec.expected_cv_samples}, "
            f"found {len(samples)}"
        )
    dataset = ProvisionalTrajectoryDataset(samples)
    folds = plan_rtc_folds(usable_source_indices, folds=spec.folds, seed=spec.seed)
    return RtcProtocolData(
        dataset=dataset,
        source_indices=usable_source_indices,
        labels=tuple(labels),
        folds=folds,
        test_sample_count=int(test_sample_count),
        test_used=False,
    )


def prepare_rtc_protocol(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
) -> RtcProtocolData:
    """Load raw arrays, validate both splits, and prepare Main-only CV data."""

    spec = RtcProtocolSpec.from_config(config)
    raw = load_rtc_raw_splits(resolve_rtc_raw_paths(config, repository_root))
    validate_test_raw_count(
        raw["test_features"],
        raw["test_labels"],
        expected_samples=spec.test_expected_samples,
    )
    return prepare_rtc_protocol_from_arrays(
        raw["main_features"],
        raw["main_labels"],
        test_sample_count=int(raw["test_features"].shape[0]),
        config=config,
    )


def _manifest_payload(protocol: RtcProtocolData, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "RTC",
        "cv_pool": {
            "split": "main",
            "main_raw_samples": DEFAULT_MAIN_RAW_SAMPLES,
            "excluded_source_indices": list(DEFAULT_EXCLUDED_SOURCE_INDICES),
            "usable_samples": len(protocol.source_indices),
            "source_indices": list(protocol.source_indices),
            "split_type": "sample_level_kfold",
            "folds": [
                {
                    "fold": plan.fold,
                    "seed": plan.seed,
                    "train_indices": list(plan.train_indices),
                    "validation_indices": list(plan.validation_indices),
                    "train_source_indices": list(plan.train_source_indices),
                    "validation_source_indices": list(plan.validation_source_indices),
                }
                for plan in protocol.folds
            ],
        },
        "test": {
            "expected_samples": protocol.test_sample_count,
            "held_out": True,
            "include_in_cv": False,
            "test_used": protocol.test_used,
        },
        "config_snapshot": json.loads(json.dumps(config)),
    }


def write_protocol_manifest(
    protocol: RtcProtocolData,
    config: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write a future-run protocol manifest and YAML config snapshot."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "protocol_manifest.json").write_text(
        json.dumps(_manifest_payload(protocol, config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "config_snapshot.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False),
        encoding="utf-8",
    )
    return output_path / "protocol_manifest.json"


def _annotate_fold_source_indices(output_dir: Path, protocol: RtcProtocolData) -> None:
    """Attach source-index lineage to metrics emitted by the existing CV loop."""

    for plan in protocol.folds:
        metrics_path = output_dir / f"fold_{plan.fold}" / "metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if tuple(payload.get("validation_indices", ())) != plan.validation_indices:
            raise ValueError(f"RTC protocol: cv.py fold {plan.fold} split differs from the audited fold plan")
        payload["validation_source_indices"] = list(plan.validation_source_indices)
        payload["train_source_indices"] = list(plan.train_source_indices)
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_fresh_output_dir(output_dir: str | Path) -> Path:
    """Require an empty generated directory for every future formal run."""

    generated_root = (Path(__file__).resolve().parents[1] / "results" / "generated").resolve(
        strict=False
    )
    output_path = Path(output_dir).resolve(strict=False)
    try:
        output_path.relative_to(generated_root)
    except ValueError as error:
        raise ValueError(
            "RTC formal run output directory must be within "
            f"{generated_root}; got {output_path}"
        ) from error
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"RTC formal run requires a fresh empty output directory; refusing to reuse {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _resolve_training_device(requested: str | torch.device) -> torch.device:
    value = str(requested)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value not in {"cpu", "cuda"}:
        raise ValueError("RTC device must be one of: auto, cpu, cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RTC device=cuda was requested but CUDA is unavailable")
    return torch.device(value)


def _git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "unavailable"


def _best_effort_cuda_telemetry(device: torch.device) -> dict[str, Any]:
    """Collect observer-only CUDA metadata without affecting training."""

    try:
        cuda_available: bool | None = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = None

    cuda_device_name: str | None = None
    if device.type == "cuda":
        try:
            cuda_device_name = str(torch.cuda.get_device_name(device))
        except Exception:
            cuda_device_name = None

    return {
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
    }


def _protocol_metadata(protocol: RtcProtocolData | None, spec: RtcProtocolSpec) -> dict[str, Any]:
    return {
        "dataset": "RTC",
        "cv_split": "main",
        "split_type": "sample_level_kfold",
        "main_raw_samples": spec.main_raw_samples,
        "excluded_source_indices": list(spec.excluded_source_indices),
        "usable_cv_samples": (
            len(protocol.source_indices) if protocol is not None else spec.expected_cv_samples
        ),
        "test_samples": (
            protocol.test_sample_count if protocol is not None else spec.test_expected_samples
        ),
        "test_used_in_cv": protocol.test_used if protocol is not None else False,
        "folds": spec.folds,
        "input_channels": spec.input_channels,
        "num_classes": spec.num_classes,
        "seed": spec.seed,
        "fold_seed_policy": "base_seed_plus_fold",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_preflight_manifest(output_path: Path, payload: Mapping[str, Any]) -> Path:
    manifest_path = output_path / "preflight_manifest.json"
    manifest_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _annotate_preflight_fold_source_indices(output_dir: Path, plan: RtcFoldPlan) -> None:
    """Attach audited Main source-index lineage to the one preflight fold."""

    metrics_path = output_dir / "fold_0" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    existing_validation = payload.get("validation_indices")
    if existing_validation is not None and tuple(existing_validation) != plan.validation_indices:
        raise ValueError("RTC preflight: cv fold 0 split differs from the audited fold plan")
    existing_train = payload.get("train_indices")
    if existing_train is not None and tuple(existing_train) != plan.train_indices:
        raise ValueError("RTC preflight: cv fold 0 train split differs from the audited fold plan")
    payload["validation_indices"] = list(plan.validation_indices)
    payload["train_indices"] = list(plan.train_indices)
    payload["validation_source_indices"] = list(plan.validation_source_indices)
    payload["train_source_indices"] = list(plan.train_source_indices)
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_rtc_preflight(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_PREFLIGHT_OUTPUT_DIR,
    device: str | torch.device = "auto",
) -> dict[str, object]:
    """Run exactly fold 0 for one epoch as an observation-only RTC preflight.

    The split is taken from the locked ten-fold plan produced by
    :func:`prepare_rtc_protocol`; this function does not create a second split
    or invoke the formal CV aggregation path.  It intentionally writes no
    ``summary.json`` and never creates checkpoints.
    """

    output_path = _ensure_fresh_output_dir(output_dir)
    started_at = _utc_now()
    repository_path = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "preflight",
        "formal_result": False,
        "fold_id": 0,
        "epochs": 1,
        "main_raw_samples": DEFAULT_MAIN_RAW_SAMPLES,
        "excluded_source_indices": list(DEFAULT_EXCLUDED_SOURCE_INDICES),
        "usable_cv_samples": DEFAULT_USABLE_CV_SAMPLES,
        "test_samples": DEFAULT_TEST_SAMPLES,
        "test_used_in_cv": False,
        "split_type": "sample_level_kfold",
        "seed": DEFAULT_SEED,
        "fold_seed": DEFAULT_SEED,
        "git_commit": _git_commit(repository_path),
        "device": str(device),
        "start_time_utc": started_at,
        "start_time": started_at,
        "end_time_utc": None,
        "end_time": None,
        "completion_status": "running",
        "config_snapshot": json.loads(json.dumps(config)),
    }
    _write_preflight_manifest(output_path, manifest)

    observer: RtcTrainingObserver | None = None
    protocol: RtcProtocolData | None = None
    try:
        spec = RtcProtocolSpec.from_config(config)
        model_config = _mapping_value(config, "model")
        training_config = _mapping_value(config, "training")
        resolved_device = _resolve_training_device(device)
        cuda_telemetry = _best_effort_cuda_telemetry(resolved_device)
        manifest.update(
            {
                "device": str(resolved_device),
                "seed": spec.seed,
                "fold_seed": spec.seed,
                "input_channels": spec.input_channels,
                "num_classes": spec.num_classes,
                "folds": spec.folds,
                **cuda_telemetry,
            }
        )
        observer = RtcTrainingObserver(
            output_path,
            run_metadata={
                "run_mode": "preflight",
                "mode": "preflight",
                "formal_result": False,
                "git_commit": manifest["git_commit"],
                "config_snapshot": json.loads(json.dumps(config)),
                "device": str(resolved_device),
                "torch_version": torch.__version__,
                **cuda_telemetry,
                "seed_policy": {
                    "base_seed": spec.seed,
                    "fold_seed_policy": "base_seed_plus_fold",
                },
                "protocol": _protocol_metadata(None, spec),
            },
            total_folds=1,
            total_epochs=1,
            mode="preflight",
        )

        protocol = prepare_rtc_protocol(config, repository_root=repository_root)
        if not isinstance(protocol, RtcProtocolData):
            raise TypeError("RTC preflight requires RtcProtocolData from prepare_rtc_protocol")
        if len(protocol.folds) != spec.folds or protocol.test_used:
            raise ValueError("RTC preflight: prepared protocol does not preserve the locked 10-fold Main-only plan")
        plan = protocol.folds[0]
        if plan.fold != 0 or plan.seed != spec.seed:
            raise ValueError("RTC preflight: fold 0 plan does not use the locked seed=42")

        manifest.update(
            {
                "usable_cv_samples": len(protocol.source_indices),
                "test_samples": protocol.test_sample_count,
                "test_used_in_cv": protocol.test_used,
                "train_size": len(plan.train_indices),
                "validation_size": len(plan.validation_indices),
                "train_indices": list(plan.train_indices),
                "validation_indices": list(plan.validation_indices),
                "train_source_indices": list(plan.train_source_indices),
                "validation_source_indices": list(plan.validation_source_indices),
            }
        )
        _write_preflight_manifest(output_path, manifest)

        def model_factory() -> TCNClassifier:
            return TCNClassifier(
                input_channels=int(model_config["input_channels"]),
                hidden_channels=[int(value) for value in _sequence_value(model_config, "hidden_channels")],
                kernel_size=int(model_config["kernel_size"]),
                dilations=[int(value) for value in _sequence_value(model_config, "dilations")],
                dropout=float(model_config["dropout"]),
                num_classes=int(model_config["num_classes"]),
            )

        # Keep this sequence aligned with cv.run_cross_validation while using
        # the already-audited fold-0 indices and the fixed one-epoch limit.
        set_seed(plan.seed)
        model = model_factory().to(resolved_device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
        )
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = DataLoader(
            Subset(protocol.dataset, plan.train_indices),
            batch_size=int(training_config["batch_size"]),
            shuffle=bool(training_config.get("shuffle_train", True)),
            drop_last=bool(training_config.get("drop_last", False)),
            num_workers=int(training_config.get("num_workers", 0)),
            collate_fn=collate_trajectory_batch,
        )
        validation_loader = DataLoader(
            Subset(protocol.dataset, plan.validation_indices),
            batch_size=int(training_config["batch_size"]),
            shuffle=False,
            num_workers=int(training_config.get("num_workers", 0)),
            collate_fn=collate_trajectory_batch,
        )

        observer.on_fold_start(0, plan.seed, len(plan.train_indices), len(plan.validation_indices))
        epoch_started = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            resolved_device,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            resolved_device,
            spec.num_classes,
        )
        observer.on_epoch_end(
            0,
            1,
            train_metrics,
            validation_metrics,
            time.perf_counter() - epoch_started,
        )
        observer.on_fold_complete(0, validation_metrics)
        _annotate_preflight_fold_source_indices(output_path, plan)
        observer.complete_preflight()

        ended_at = _utc_now()
        manifest.update(
            {
                "completion_status": "preflight_complete",
                "end_time_utc": ended_at,
                "end_time": ended_at,
                "completed_folds": [0],
                "plot_errors": list(observer.plot_errors),
                "artifacts": [
                    "run_manifest.json",
                    "fold_0/metrics.json",
                    "fold_0/training_curves.png",
                    "preflight_manifest.json",
                ],
                "metrics": {
                    "train": dict(train_metrics),
                    "validation": dict(validation_metrics),
                },
            }
        )
        _write_preflight_manifest(output_path, manifest)
        return {
            "status": "preflight_complete",
            "mode": "preflight",
            "formal_result": False,
            "fold_id": 0,
            "epochs": 1,
            "validation": dict(validation_metrics),
            "output_dir": str(output_path),
        }
    except BaseException as error:
        if observer is not None:
            observer.mark_aborted(error)
        ended_at = _utc_now()
        manifest.update(
            {
                "completion_status": "aborted",
                "end_time_utc": ended_at,
                "end_time": ended_at,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_folds": (
                    sorted(observer.fold_accuracies) if observer is not None else []
                ),
                "plot_errors": list(observer.plot_errors) if observer is not None else [],
            }
        )
        _write_preflight_manifest(output_path, manifest)
        raise


def run_rtc_cross_validation(
    config: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str | torch.device = "auto",
) -> dict[str, object]:
    """Run the explicit RTC CV path with observation-only progress recording."""

    output_path = _ensure_fresh_output_dir(output_dir)
    model_config = _mapping_value(config, "model")
    training_config = _mapping_value(config, "training")
    evaluation_config = _mapping_value(config, "evaluation")
    spec = RtcProtocolSpec.from_config(config)
    repository_path = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved_device = _resolve_training_device(device)
    cuda_telemetry = _best_effort_cuda_telemetry(resolved_device)
    observer: RtcTrainingObserver | None = RtcTrainingObserver(
        output_path,
        run_metadata={
            "git_commit": _git_commit(repository_path),
            "config_snapshot": json.loads(json.dumps(config)),
            "device": str(resolved_device),
            "torch_version": torch.__version__,
            **cuda_telemetry,
            "seed_policy": {
                "base_seed": spec.seed,
                "fold_seed_policy": "base_seed_plus_fold",
            },
            "protocol": _protocol_metadata(None, spec),
        },
        total_folds=spec.folds,
        total_epochs=int(training_config["epochs"]),
    )

    try:
        protocol = prepare_rtc_protocol(config, repository_root=repository_root)
        if not isinstance(protocol, RtcProtocolData):
            observer = None
        write_protocol_manifest(protocol, config, output_path)

        def model_factory() -> TCNClassifier:
            return TCNClassifier(
                input_channels=int(model_config["input_channels"]),
                hidden_channels=[int(value) for value in _sequence_value(model_config, "hidden_channels")],
                kernel_size=int(model_config["kernel_size"]),
                dilations=[int(value) for value in _sequence_value(model_config, "dilations")],
                dropout=float(model_config["dropout"]),
                num_classes=int(model_config["num_classes"]),
            )

        summary = run_cross_validation(
            dataset=protocol.dataset,
            model_factory=model_factory,
            batch_size=int(training_config["batch_size"]),
            learning_rate=float(training_config["learning_rate"]),
            epochs=int(training_config["epochs"]),
            folds=int(evaluation_config["folds"]),
            seed=int(training_config["seed"]),
            output_dir=output_path,
            device=resolved_device,
            num_classes=int(model_config["num_classes"]),
            shuffle_train=bool(training_config.get("shuffle_train", True)),
            num_workers=int(training_config.get("num_workers", 0)),
            drop_last=bool(training_config.get("drop_last", False)),
            shuffle_splits=True,
            observer=observer,
        )
        _annotate_fold_source_indices(output_path, protocol)
        if observer is not None:
            return observer.complete_run(summary, _protocol_metadata(protocol, spec))
        return summary
    except BaseException as error:
        if observer is not None:
            observer.mark_aborted(error)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Locked RTC Main-only reproduction protocol runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument(
        "--output-dir",
        help=(
            "Generated output directory; defaults to rtc_baseline for --run-cv, "
            "rtc_preflight for --preflight"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--manifest-only", action="store_true", help="Validate raw splits and write a protocol manifest")
    actions.add_argument("--run-cv", action="store_true", help="Explicitly run the future RTC 10-fold baseline")
    actions.add_argument(
        "--preflight",
        action="store_true",
        help="Run fixed RTC fold 0 for one observation-only epoch",
    )
    arguments = parser.parse_args(argv)

    if not arguments.manifest_only and not arguments.run_cv and not arguments.preflight:
        parser.print_help()
        return

    config = load_rtc_config(arguments.config)
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = str(DEFAULT_PREFLIGHT_OUTPUT_DIR if arguments.preflight else DEFAULT_OUTPUT_DIR)
    if arguments.manifest_only:
        protocol = prepare_rtc_protocol(config, repository_root=arguments.repository_root)
        manifest_path = write_protocol_manifest(protocol, config, output_dir)
        print(f"RTC protocol manifest written to {manifest_path}; test_used=false")
        return

    if arguments.preflight:
        result = run_rtc_preflight(
            config,
            repository_root=arguments.repository_root,
            output_dir=output_dir,
            device=arguments.device,
        )
        print(f"RTC observation-only preflight completed at {result['output_dir']}")
        return

    summary = run_rtc_cross_validation(
        config,
        repository_root=arguments.repository_root,
        output_dir=output_dir,
        device=arguments.device,
    )
    print(f"RTC CV completed with mean accuracy={summary['mean_accuracy']:.4f}; not a claimed paper result")


if __name__ == "__main__":
    main()
