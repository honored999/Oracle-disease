import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader

from TCN_Singh2024.src.dataset import TrajectoryClassificationDataset, collate_trajectory_batch
from TCN_Singh2024.src.evaluation import classification_metrics, evaluate
from TCN_Singh2024.src.model import TCNClassifier
from TCN_Singh2024.src.train import train_one_epoch
from TCN_Singh2024.src.train import run_synthetic_smoke


def _variable_length_loader() -> DataLoader:
    dataset = TrajectoryClassificationDataset(
        [torch.randn(2, 3), torch.randn(4, 3), torch.randn(3, 3), torch.randn(5, 3)],
        [0, 1, 0, 1],
    )
    return DataLoader(dataset, batch_size=2, collate_fn=collate_trajectory_batch, shuffle=False)


def test_train_one_epoch_updates_parameters_with_variable_length_batches():
    torch.manual_seed(0)
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_one_epoch(model, _variable_length_loader(), optimizer, nn.CrossEntropyLoss(), device="cpu")

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_train_one_epoch_rejects_nonfinite_logits_before_an_optimizer_update():
    class NonFiniteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = nn.Parameter(torch.tensor(1.0))

        def forward(self, sequences, lengths=None):
            return self.parameter * torch.full((sequences.shape[0], 2), float("nan"))

    model = NonFiniteModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(FloatingPointError, match="logits"):
        train_one_epoch(model, _variable_length_loader(), optimizer, nn.CrossEntropyLoss(), device="cpu")


def test_classification_metrics_computes_accuracy_confusion_matrix_and_per_class_accuracy():
    logits = torch.tensor([[3.0, 1.0], [0.1, 2.0], [4.0, 0.0]])
    labels = torch.tensor([0, 0, 1])

    metrics = classification_metrics(logits, labels, num_classes=2)

    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 0]]
    assert metrics["per_class_accuracy"] == [0.5, 0.0]


def test_evaluate_returns_finite_loss_and_accuracy_without_parameter_updates():
    torch.manual_seed(1)
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=2)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = evaluate(model, _variable_length_loader(), nn.CrossEntropyLoss(), device="cpu", num_classes=2)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_synthetic_smoke_runs_all_ten_folds_without_claiming_a_paper_result(tmp_path):
    config = {
        "model": {"input_channels": 3, "hidden_channels": [4], "kernel_size": 3, "dilations": [1], "dropout": 0.0},
        "training": {"batch_size": 32, "learning_rate": 1e-3, "seed": 9},
        "evaluation": {"folds": 10},
    }

    summary = run_synthetic_smoke(config, tmp_path)

    assert summary["number_of_folds"] == 10
    assert (tmp_path / "summary.json").is_file()
