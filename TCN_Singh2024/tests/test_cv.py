import json

import torch

from TCN_Singh2024.src.cv import make_kfold_splits, run_cross_validation
from TCN_Singh2024.src.dataset import TrajectoryClassificationDataset
from TCN_Singh2024.src.model import TCNClassifier


def test_kfold_splits_are_reproducible_disjoint_and_cover_each_sample_once():
    first = make_kfold_splits(num_samples=20, folds=10, seed=17)
    second = make_kfold_splits(num_samples=20, folds=10, seed=17)

    assert first == second
    validation_indices = []
    for train_indices, validation_indices_for_fold in first:
        assert not set(train_indices).intersection(validation_indices_for_fold)
        validation_indices.extend(validation_indices_for_fold)
    assert sorted(validation_indices) == list(range(20))
    assert len(validation_indices) == len(set(validation_indices))


def test_cross_validation_reinitializes_model_and_optimizer_per_fold_and_writes_metrics(tmp_path):
    dataset = TrajectoryClassificationDataset([torch.randn(2 + index % 3, 3) for index in range(10)], [index % 2 for index in range(10)])
    models = []

    def model_factory():
        model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=2)
        models.append(model)
        return model

    summary = run_cross_validation(
        dataset=dataset,
        model_factory=model_factory,
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        folds=5,
        seed=3,
        output_dir=tmp_path,
        device="cpu",
        num_classes=2,
    )

    assert len(models) == 5
    assert len({id(model) for model in models}) == 5
    assert summary["number_of_folds"] == 5
    assert len(summary["fold_accuracies"]) == 5
    assert (tmp_path / "summary.json").is_file()
    assert all((tmp_path / f"fold_{fold}" / "metrics.json").is_file() for fold in range(5))
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["split_type"] == "sample_level_kfold"
    assert summary["config_snapshot"]["optimizer"] == "adam"
    fold_metrics = json.loads((tmp_path / "fold_0" / "metrics.json").read_text(encoding="utf-8"))
    assert fold_metrics["schema_version"] == 1
    assert len(fold_metrics["validation_indices"]) == fold_metrics["validation_size"]


def test_kfold_can_disable_shuffle_explicitly():
    splits = make_kfold_splits(num_samples=6, folds=3, seed=9, shuffle=False)

    assert [validation for _, validation in splits] == [[0, 1], [2, 3], [4, 5]]
