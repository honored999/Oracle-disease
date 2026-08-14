import pickle
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from TCN_Singh2024.src.adapters import (
    ProvisionalTrajectoryDataset,
    _load_rtc_numpy_array,
    load_rtd_rtc_pickle_dataset,
    load_rtc_pickle_dataset,
    load_sixdmg_sqlite_dataset,
)
from TCN_Singh2024.src.dataset import collate_trajectory_batch
from TCN_Singh2024.src.model import TCNClassifier
from TCN_Singh2024.src.preprocess import min_max_normalize, root_point_translate


def _write_pickle_pair(tmp_path, features, labels):
    (tmp_path / "features").write_bytes(pickle.dumps(features))
    (tmp_path / "labels").write_bytes(pickle.dumps(labels))
    return tmp_path / "features", tmp_path / "labels"


def test_rtd_rtc_provisional_adapter_reshapes_flat_xyz_and_one_hot_labels(tmp_path):
    features_path, labels_path = _write_pickle_pair(
        tmp_path,
        [[1.0, 2.0, 3.0, 3.0, 5.0, 7.0], [2.0, 0.0, 1.0]],
        [[0, 1, 0], [1, 0, 0]],
    )

    dataset = load_rtd_rtc_pickle_dataset(features_path, labels_path, preprocess=True)

    first = dataset[0]
    assert first["sequence"].shape == (2, 3)
    assert first["label"] == 1
    assert first["metadata"]["source_index"] == 0
    assert torch.all((first["sequence"] >= 0) & (first["sequence"] <= 1))


def test_rtd_rtc_adapter_accepts_variable_lengths_and_collates_into_current_tcn(tmp_path):
    features_path, labels_path = _write_pickle_pair(
        tmp_path,
        [[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3, 4, 4, 4]],
        [[1, 0], [0, 1]],
    )
    dataset = load_rtd_rtc_pickle_dataset(features_path, labels_path, preprocess=False)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_trajectory_batch)
    batch = next(iter(loader))
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=2)

    assert model(batch["sequences"], lengths=batch["lengths"]).shape == (2, 2)


def test_rtd_rtc_adapter_rejects_flat_sequences_not_divisible_by_three(tmp_path):
    features_path, labels_path = _write_pickle_pair(tmp_path, [[0, 1, 2, 3]], [[1, 0]])

    with pytest.raises(ValueError, match="divisible by 3"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)


def test_rtd_rtc_adapter_rejects_malformed_or_non_one_hot_pickle_content(tmp_path):
    features_path, labels_path = _write_pickle_pair(tmp_path, {"unexpected": "mapping"}, [[1, 0]])

    with pytest.raises(ValueError, match="sequence collection"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)

    features_path, labels_path = _write_pickle_pair(tmp_path, [[0, 0, 0]], [[1, 1]])
    with pytest.raises(ValueError, match="one-hot"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)


def test_rtd_rtc_adapter_reports_clear_errors_for_corrupt_non_numeric_and_count_mismatch_files(tmp_path):
    features_path = tmp_path / "features"
    labels_path = tmp_path / "labels"
    features_path.write_bytes(b"not a pickle")
    labels_path.write_bytes(pickle.dumps([[1, 0]]))
    with pytest.raises(ValueError, match="unable to load pickle"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)

    features_path, labels_path = _write_pickle_pair(tmp_path, [["not", "numeric", "data"]], [[1, 0]])
    with pytest.raises(ValueError, match="non-numeric"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)

    features_path, labels_path = _write_pickle_pair(tmp_path, [[0, 0, 0], [1, 1, 1]], [[1, 0]])
    with pytest.raises(ValueError, match="different sample counts"):
        load_rtd_rtc_pickle_dataset(features_path, labels_path)


def _sixdmg_fixture_database(path):
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE samples (name TEXT, tester INTEGER, trial INTEGER, length INTEGER, data BLOB, session TEXT)")
    records = [
        (0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        (0.1, 10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
    ]
    blob = b"".join(struct.pack("<14f", *record) for record in records)
    connection.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?)", ("A", 7, 2, len(records), blob, "morning"))
    connection.commit()
    connection.close()


def test_sixdmg_provisional_adapter_selects_only_position_and_preserves_metadata(tmp_path):
    database_path = tmp_path / "air_handwriting.sqlite"
    _sixdmg_fixture_database(database_path)

    dataset = load_sixdmg_sqlite_dataset(
        database_path,
        table_name="samples",
        feature_selection="position_xyz",
        label_mapping={"A": 0},
        preprocess=False,
    )

    sample = dataset[0]
    torch.testing.assert_close(sample["sequence"], torch.tensor([[1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]))
    assert sample["sequence"].shape == (2, 3)
    assert sample["label"] == 0
    assert sample["metadata"] == {"name": "A", "tester": 7, "trial": 2, "source_index": 0, "session": "morning"}


def test_sixdmg_adapter_variable_length_collates_and_reaches_model(tmp_path):
    database_path = tmp_path / "air_handwriting.sqlite"
    _sixdmg_fixture_database(database_path)
    dataset = load_sixdmg_sqlite_dataset(database_path, "samples", "position_xyz", {"A": 0}, preprocess=True)
    batch = collate_trajectory_batch([dataset[0], dataset[0]])
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=1)

    assert model(batch["sequences"], lengths=batch["lengths"]).shape == (2, 1)


def test_sixdmg_adapter_rejects_invalid_blob_size(tmp_path):
    database_path = tmp_path / "broken.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE samples (name TEXT, tester INTEGER, trial INTEGER, length INTEGER, data BLOB)")
    connection.execute("INSERT INTO samples VALUES (?, ?, ?, ?, ?)", ("A", 1, 1, 2, b"too short"))
    connection.commit()
    connection.close()

    with pytest.raises(ValueError, match="invalid 6DMG binary"):
        load_sixdmg_sqlite_dataset(database_path, "samples", "position_xyz", {"A": 0})


def test_provisional_dataset_preserves_metadata_and_returns_cloned_sequences():
    dataset = ProvisionalTrajectoryDataset([
        {"sequence": torch.tensor([[1.0, 2.0, 3.0]]), "label": 0, "metadata": {"writer": 4}},
    ])

    sample = dataset[0]
    sample["sequence"][0, 0] = -1

    assert dataset[0]["sequence"][0, 0].item() == 1.0
    assert dataset[0]["metadata"] == {"writer": 4}
def _write_rtc_pickle_pair(tmp_path, features, labels, split="main"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_name, label_name = (
        ("features", "labels") if split == "main" else ("featuresTest", "labelsTest")
    )
    feature_path = tmp_path / feature_name
    label_path = tmp_path / label_name
    feature_path.write_bytes(pickle.dumps(np.asarray(features), protocol=3))
    label_path.write_bytes(pickle.dumps(np.asarray(labels), protocol=3))
    return feature_path, label_path


def _rtc_labels(class_ids):
    labels = np.zeros((len(class_ids), 26), dtype=np.float32)
    for index, class_id in enumerate(class_ids):
        labels[index, class_id] = 1
    return labels


def _rtc_real_paths(split="main"):
    raw_root = Path(__file__).resolve().parents[2] / "data" / "RTC" / "raw"
    if split == "main":
        return raw_root / "features", raw_root / "labels"
    return raw_root / "featuresTest", raw_root / "labelsTest"


def test_rtc_main_adapter_loads_padded_xyz_and_preserves_contract_metadata(tmp_path):
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        np.array([[1, 2, 3, 4, 5, 6, 0, 0, 0]], dtype=np.float64),
        _rtc_labels([7]),
    )

    dataset = load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)
    sample = dataset[0]

    torch.testing.assert_close(
        sample["sequence"],
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    assert sample["sequence"].shape == (2, 3)
    assert sample["label"] == 7
    assert sample["metadata"] == {"source_index": 0, "split": "main"}


def test_rtc_real_main_arrays_match_verified_shapes_and_full_load_rejects_empty_rows():
    features_path, labels_path = _rtc_real_paths("main")
    if not features_path.exists() or not labels_path.exists():
        pytest.skip("official RTC main raw files are not available locally")

    features = _load_rtc_numpy_array(features_path, "features")
    labels = _load_rtc_numpy_array(labels_path, "labels")
    assert features.shape == (20098, 800)
    assert labels.shape == (20098, 26)

    with pytest.raises(ValueError, match="all-zero sequence"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_test_split_is_explicit_and_is_not_merged_with_main(tmp_path):
    main_paths = _write_rtc_pickle_pair(
        tmp_path / "main",
        [[1, 2, 3, 0, 0, 0]],
        _rtc_labels([0]),
        split="main",
    )
    test_paths = _write_rtc_pickle_pair(
        tmp_path / "test",
        [[4, 5, 6, 0, 0, 0], [7, 8, 9, 10, 11, 12]],
        _rtc_labels([1, 2]),
        split="test",
    )

    main_dataset = load_rtc_pickle_dataset(*main_paths, preprocess=False)
    test_dataset = load_rtc_pickle_dataset(*test_paths, preprocess=False)

    assert len(main_dataset) == 1
    assert len(test_dataset) == 2
    assert main_dataset[0]["metadata"]["split"] == "main"
    assert test_dataset[0]["metadata"]["split"] == "test"
    assert test_dataset[1]["metadata"]["source_index"] == 1


def test_rtc_adapter_rejects_non_triplet_valid_prefix(tmp_path):
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        np.array([[1, 2, 3, 4, 5, 0, 0]], dtype=np.float64),
        _rtc_labels([0]),
    )

    with pytest.raises(ValueError, match="divisible by 3"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_adapter_rejects_all_zero_sequence(tmp_path):
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        np.zeros((1, 6), dtype=np.float64),
        _rtc_labels([0]),
    )

    with pytest.raises(ValueError, match="all-zero sequence"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_adapter_rejects_non_one_hot_label(tmp_path):
    labels = _rtc_labels([0])
    labels[0, 1] = 1
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        np.array([[1, 2, 3]], dtype=np.float64),
        labels,
    )

    with pytest.raises(ValueError, match="strict one-hot"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


@pytest.mark.parametrize("target", ["features", "labels"])
def test_rtc_adapter_rejects_nan_or_inf(target, tmp_path):
    features = np.array([[1, 2, 3]], dtype=np.float64)
    labels = _rtc_labels([0])
    if target == "features":
        features[0, 0] = np.nan
    else:
        labels[0, 0] = np.inf
    features_path, labels_path = _write_rtc_pickle_pair(tmp_path, features, labels)

    with pytest.raises(ValueError, match="NaN or Inf"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_adapter_rejects_shape_and_count_mismatch(tmp_path):
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path / "shape",
        np.array([1, 2, 3], dtype=np.float64),
        _rtc_labels([0]),
    )
    with pytest.raises(ValueError, match="features must be a 2-D"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)

    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path / "count",
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        _rtc_labels([0]),
    )
    with pytest.raises(ValueError, match="different sample counts"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_adapter_rejects_non_matching_split_paths(tmp_path):
    main_features, _ = _write_rtc_pickle_pair(
        tmp_path / "main",
        [[1, 2, 3]],
        _rtc_labels([0]),
        split="main",
    )
    _, test_labels = _write_rtc_pickle_pair(
        tmp_path / "test",
        [[4, 5, 6]],
        _rtc_labels([1]),
        split="test",
    )

    with pytest.raises(ValueError, match="matching features/labels"):
        load_rtc_pickle_dataset(main_features, test_labels, preprocess=False)


def test_rtc_adapter_rejects_corrupt_pickle_with_clear_error(tmp_path):
    features_path = tmp_path / "features"
    labels_path = tmp_path / "labels"
    features_path.write_bytes(b"not a pickle")
    labels_path.write_bytes(pickle.dumps(_rtc_labels([0])))

    with pytest.raises(ValueError, match="unable to load features NumPy array"):
        load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)


def test_rtc_adapter_collates_variable_lengths_and_reaches_tcn(tmp_path):
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        np.array(
            [
                [1, 2, 3, 4, 5, 6, 0, 0, 0],
                [7, 8, 9, 10, 11, 12, 13, 14, 15],
            ],
            dtype=np.float64,
        ),
        _rtc_labels([2, 4]),
    )
    dataset = load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)
    batch = collate_trajectory_batch([dataset[0], dataset[1]])
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=26)

    logits = model(batch["sequences"], lengths=batch["lengths"])

    assert batch["sequences"].shape == (2, 3, 3)
    assert batch["lengths"].tolist() == [2, 3]
    assert logits.shape == (2, 26)
    assert torch.isfinite(logits).all()


def test_rtc_adapter_preprocess_false_preserves_xyz_and_true_applies_pipeline_once(tmp_path):
    points = torch.tensor([[10.0, 20.0, 30.0], [12.0, 23.0, 34.0], [11.0, 22.0, 32.0]])
    features_path, labels_path = _write_rtc_pickle_pair(
        tmp_path,
        points.reshape(1, -1).numpy(),
        _rtc_labels([3]),
    )

    raw_sample = load_rtc_pickle_dataset(features_path, labels_path, preprocess=False)[0]
    processed_sample = load_rtc_pickle_dataset(features_path, labels_path, preprocess=True)[0]
    expected_once = min_max_normalize(root_point_translate(points, mode="paper_literal"))
    expected_twice = min_max_normalize(root_point_translate(expected_once, mode="paper_literal"))

    torch.testing.assert_close(raw_sample["sequence"], points)
    torch.testing.assert_close(processed_sample["sequence"], expected_once)
    assert not torch.allclose(processed_sample["sequence"], expected_twice)
