import pickle
import sqlite3
import struct

import pytest
import torch
from torch.utils.data import DataLoader

from TCN_Singh2024.src.adapters import (
    ProvisionalTrajectoryDataset,
    load_rtd_rtc_pickle_dataset,
    load_sixdmg_sqlite_dataset,
)
from TCN_Singh2024.src.dataset import collate_trajectory_batch
from TCN_Singh2024.src.model import TCNClassifier


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
