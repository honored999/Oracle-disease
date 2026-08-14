"""Provisional adapters based on public format descriptions, not real-file validation."""

from __future__ import annotations

import pickle
import sqlite3
import struct
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from TCN_Singh2024.src.preprocess import min_max_normalize, root_point_translate


PROVISIONAL_NOTICE = "provisional format contract; unverified against real dataset files"


class ProvisionalTrajectoryDataset(Dataset[dict[str, object]]):
    """In-memory dataset retaining metadata from a provisional source adapter."""

    def __init__(self, samples: Sequence[dict[str, object]]) -> None:
        self._samples = []
        for sample in samples:
            sequence = sample.get("sequence")
            label = sample.get("label")
            if not isinstance(sequence, Tensor) or sequence.ndim != 2 or min(sequence.shape) < 1:
                raise ValueError(f"{PROVISIONAL_NOTICE}: each sequence must be non-empty [T, C]")
            self._samples.append({
                "sequence": sequence.detach().clone(),
                "label": int(label),
                "metadata": deepcopy(sample.get("metadata", {})),
            })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self._samples[index]
        return {"sequence": sample["sequence"].clone(), "label": sample["label"], "metadata": deepcopy(sample["metadata"])}


def _apply_preprocessing(sequence: Tensor, preprocess: bool) -> Tensor:
    if not preprocess:
        return sequence
    return min_max_normalize(root_point_translate(sequence, mode="paper_literal"))


def _load_pickle(path: str | Path) -> object:
    try:
        with Path(path).open("rb") as stream:
            return pickle.load(stream)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as error:
        raise ValueError(f"{PROVISIONAL_NOTICE}: unable to load pickle at {path}") from error


def _as_sample_collection(value: object, field_name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"{PROVISIONAL_NOTICE}: {field_name} must be a sequence collection, not {type(value).__name__}")
    try:
        return list(value)  # Supports list/tuple and unverified array-like pickle containers.
    except TypeError as error:
        raise ValueError(f"{PROVISIONAL_NOTICE}: {field_name} must be an iterable sample collection") from error


def _flat_coordinate_values(value: object, index: int) -> list[float]:
    if isinstance(value, Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
    elif not isinstance(value, (str, bytes, bytearray, Mapping)):
        try:
            values = list(value)
        except TypeError as error:
            raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} is not a flat coordinate sequence") from error
    else:
        raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} is not a flat coordinate sequence")
    if any(isinstance(item, (Sequence, Mapping, Tensor)) and not isinstance(item, (str, bytes, bytearray)) for item in values):
        raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} must be one-dimensional numeric coordinates")
    try:
        return [float(item) for item in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} contains non-numeric coordinates") from error


def _one_hot_to_label(one_hot: object, index: int) -> int:
    if isinstance(one_hot, Tensor):
        values = one_hot.detach().cpu().reshape(-1).tolist()
    elif isinstance(one_hot, Sequence) and not isinstance(one_hot, (str, bytes, bytearray)):
        values = list(one_hot)
    else:
        raise ValueError(f"{PROVISIONAL_NOTICE}: label {index} must be a one-hot sequence")
    active = [position for position, value in enumerate(values) if value == 1]
    if len(values) == 0 or len(active) != 1 or any(value not in (0, 1) for value in values):
        raise ValueError(f"{PROVISIONAL_NOTICE}: label {index} is not one-hot")
    return active[0]


def load_rtd_rtc_pickle_dataset(
    features_path: str | Path,
    labels_path: str | Path,
    preprocess: bool = True,
) -> ProvisionalTrajectoryDataset:
    """Load author-documented RTD/RTC pickle files under a provisional contract."""
    features = _as_sample_collection(_load_pickle(features_path), "features")
    labels = _as_sample_collection(_load_pickle(labels_path), "labels")
    if len(features) != len(labels):
        raise ValueError(f"{PROVISIONAL_NOTICE}: features and labels have different sample counts")
    samples: list[dict[str, object]] = []
    for index, (flat_sequence, one_hot) in enumerate(zip(features, labels, strict=True)):
        values = _flat_coordinate_values(flat_sequence, index)
        if len(values) == 0 or len(values) % 3:
            raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} length must be non-zero and divisible by 3")
        sequence = torch.tensor(values, dtype=torch.float32).reshape(-1, 3)
        if not torch.isfinite(sequence).all():
            raise ValueError(f"{PROVISIONAL_NOTICE}: feature {index} contains NaN or Inf")
        samples.append({
            "sequence": _apply_preprocessing(sequence, preprocess),
            "label": _one_hot_to_label(one_hot, index),
            "metadata": {"source_index": index},
        })
    return ProvisionalTrajectoryDataset(samples)


SixDmgFeatureSelection = Literal["position_xyz"]
_SIXDMG_RECORD_FORMAT = "<14f"


def load_sixdmg_sqlite_dataset(
    database_path: str | Path,
    table_name: str,
    feature_selection: SixDmgFeatureSelection,
    label_mapping: Mapping[str, int],
    preprocess: bool = True,
) -> ProvisionalTrajectoryDataset:
    """Load a documented 6DMG-like SQLite record layout provisionally.

    This supports only position_xyz because Singh & Koundal describe raw 3D
    trajectories; it does not infer quaternion or inertial feature usage.
    """
    if feature_selection != "position_xyz":
        raise ValueError(f"{PROVISIONAL_NOTICE}: only feature_selection='position_xyz' is supported")
    if not table_name.replace("_", "").isalnum():
        raise ValueError("table_name must contain only letters, numbers, and underscores")
    connection = sqlite3.connect(Path(database_path))
    try:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")]
        required = {"name", "tester", "trial", "length", "data"}
        if not required.issubset(columns):
            raise ValueError(f"{PROVISIONAL_NOTICE}: expected SQLite columns name, tester, trial, length, data")
        rows = connection.execute(f"SELECT * FROM {table_name}").fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"{PROVISIONAL_NOTICE}: expected SQLite columns name, tester, trial, length, data") from error
    finally:
        connection.close()
    samples: list[dict[str, object]] = []
    record_size = struct.calcsize(_SIXDMG_RECORD_FORMAT)
    for index, row in enumerate(rows):
        metadata_row = dict(zip(columns, row, strict=True))
        name, tester, trial, length, blob = (metadata_row.pop(key) for key in ("name", "tester", "trial", "length", "data"))
        if name not in label_mapping:
            raise ValueError(f"{PROVISIONAL_NOTICE}: no integer label mapping for name {name!r}")
        if not isinstance(blob, bytes) or length < 1 or len(blob) != length * record_size:
            raise ValueError(f"{PROVISIONAL_NOTICE}: invalid 6DMG binary record payload at row {index}")
        records = [struct.unpack_from(_SIXDMG_RECORD_FORMAT, blob, offset) for offset in range(0, len(blob), record_size)]
        sequence = torch.tensor([[record[1], record[2], record[3]] for record in records], dtype=torch.float32)
        if not torch.isfinite(sequence).all():
            raise ValueError(f"{PROVISIONAL_NOTICE}: 6DMG position contains NaN or Inf at row {index}")
        samples.append({
            "sequence": _apply_preprocessing(sequence, preprocess),
            "label": int(label_mapping[name]),
            "metadata": {"name": name, "tester": tester, "trial": trial, "source_index": index, **metadata_row},
        })
    return ProvisionalTrajectoryDataset(samples)


RtcSplit = Literal["main", "test"]
_RTC_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class _RtcNumpyArrayUnpickler(pickle.Unpickler):
    """Restrict RTC pickle reconstruction to NumPy array primitives."""

    def find_class(self, module: str, name: str) -> object:
        try:
            return _RTC_PICKLE_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"RTC adapter rejected pickle global {module}.{name}"
            ) from error


def _load_rtc_numpy_array(path: str | Path, field_name: str) -> np.ndarray:
    """Load one explicitly supplied RTC pickle as a finite 2-D NumPy array."""
    try:
        with Path(path).open("rb") as stream:
            value = _RtcNumpyArrayUnpickler(stream).load()
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError) as error:
        raise ValueError(f"RTC adapter: unable to load {field_name} NumPy array from {path}") from error

    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"RTC adapter: {field_name} must be a 2-D NumPy array")
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.complexfloating):
        raise ValueError(f"RTC adapter: {field_name} must have a real numeric dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"RTC adapter: {field_name} contains NaN or Inf")
    return value


def _validate_rtc_source_paths(
    features_path: str | Path,
    labels_path: str | Path,
    split: RtcSplit | None,
) -> RtcSplit | None:
    """Validate explicit RTC paths without discovering or merging another split."""
    feature_path = Path(features_path)
    label_path = Path(labels_path)
    for path in (feature_path, label_path):
        parts = {part.casefold() for part in path.parts}
        if "rtd" in parts and "raw" in parts:
            raise ValueError("RTC adapter refuses RTD raw paths; pass data/RTC/raw files")

    known_pairs = {
        ("features", "labels"): "main",
        ("featurestest", "labelstest"): "test",
    }
    pair = (feature_path.name.casefold(), label_path.name.casefold())
    inferred_split = known_pairs.get(pair)
    if pair[0] in {"features", "featurestest"} or pair[1] in {"labels", "labelstest"}:
        if inferred_split is None:
            raise ValueError("RTC adapter requires matching features/labels or featuresTest/labelsTest paths")
    if split is not None and split not in {"main", "test"}:
        raise ValueError("RTC adapter split must be 'main' or 'test'")
    if split is not None and inferred_split is not None and split != inferred_split:
        raise ValueError(f"RTC adapter paths belong to the {inferred_split!r} split, not {split!r}")
    return split or inferred_split


def _rtc_trim_trailing_scalar_zeros(row: np.ndarray, index: int) -> np.ndarray:
    """Return a non-empty row prefix without removing any interior zero."""
    nonzero_positions = np.flatnonzero(row != 0)
    if nonzero_positions.size == 0:
        raise ValueError(f"RTC adapter: feature {index} is an empty all-zero sequence")
    valid_length = int(nonzero_positions[-1]) + 1
    if not np.all(row[valid_length:] == 0):
        raise ValueError(f"RTC adapter: feature {index} has non-contiguous trailing padding")
    if valid_length % 3 != 0:
        raise ValueError(f"RTC adapter: feature {index} valid length must be divisible by 3")
    return row[:valid_length]


def _rtc_one_hot_to_label(row: np.ndarray, index: int) -> int:
    if row.ndim != 1 or row.shape[0] != 26:
        raise ValueError(f"RTC adapter: label {index} must have shape (26,) one-hot encoding")
    active = np.flatnonzero(row == 1)
    if not np.isin(row, (0, 1)).all() or active.size != 1 or row.sum() != 1:
        raise ValueError(f"RTC adapter: label {index} is not strict one-hot")
    return int(active[0])


def load_rtc_pickle_dataset(
    features_path: str | Path,
    labels_path: str | Path,
    preprocess: bool = True,
    split: RtcSplit | None = None,
) -> ProvisionalTrajectoryDataset:
    """Load one explicitly supplied RTC main or test pickle pair.

    The verified RTC files use a fixed-width row with zero padding.  This
    adapter removes only the contiguous scalar zero suffix from each row and
    reshapes the remaining interleaved XYZ prefix to [T, 3].  It never
    discovers, reads, or combines the other split.  Callers should pass paths
    resolved from the repository root, for example data/RTC/raw/features.
    """
    resolved_split = _validate_rtc_source_paths(features_path, labels_path, split)
    features = _load_rtc_numpy_array(features_path, "features")
    labels = _load_rtc_numpy_array(labels_path, "labels")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("RTC adapter: features and labels have different sample counts")
    if features.shape[1] < 3:
        raise ValueError("RTC adapter: features must contain at least one XYZ triplet")
    if labels.shape[1] != 26:
        raise ValueError("RTC adapter: labels must have 26 one-hot classes")

    samples: list[dict[str, object]] = []
    for index, (row, label_row) in enumerate(zip(features, labels, strict=True)):
        valid_values = _rtc_trim_trailing_scalar_zeros(row, index)
        sequence = torch.tensor(valid_values, dtype=torch.float32).reshape(-1, 3)
        samples.append({
            "sequence": _apply_preprocessing(sequence, preprocess),
            "label": _rtc_one_hot_to_label(label_row, index),
            "metadata": {"source_index": index, **({"split": resolved_split} if resolved_split else {})},
        })
    return ProvisionalTrajectoryDataset(samples)
