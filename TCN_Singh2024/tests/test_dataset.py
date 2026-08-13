import pytest
import torch

from TCN_Singh2024.src.dataset import TrajectoryClassificationDataset, collate_trajectory_batch
from TCN_Singh2024.src.model import TCNClassifier


def test_dataset_returns_sequence_time_channel_and_integer_label_without_mutation():
    sequence = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    dataset = TrajectoryClassificationDataset([sequence], [7])

    sample = dataset[0]

    assert sample["sequence"].shape == (2, 3)
    assert sample["label"] == 7
    sample["sequence"][0, 0] = -99.0
    assert sequence[0, 0].item() == 1.0


def test_dataset_accepts_six_channel_trajectory_sequences():
    dataset = TrajectoryClassificationDataset([torch.randn(3, 6)], [1])

    assert dataset[0]["sequence"].shape == (3, 6)


def test_collate_right_pads_time_dimension_and_returns_original_lengths():
    batch = [
        {"sequence": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "label": 2},
        {"sequence": torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]), "label": 4},
    ]

    collated = collate_trajectory_batch(batch)

    assert collated["sequences"].shape == (2, 2, 3)
    assert collated["lengths"].tolist() == [2, 3]
    assert collated["labels"].tolist() == [2, 4]
    torch.testing.assert_close(collated["sequences"][0, :, :2], batch[0]["sequence"].transpose(0, 1))
    torch.testing.assert_close(collated["sequences"][0, :, 2], torch.zeros(2))


def test_collate_rejects_mixed_channel_counts():
    batch = [
        {"sequence": torch.randn(2, 3), "label": 0},
        {"sequence": torch.randn(4, 6), "label": 1},
    ]

    with pytest.raises(ValueError, match="channel count"):
        collate_trajectory_batch(batch)


def test_classifier_uses_each_sample_real_last_step_when_lengths_are_supplied():
    torch.manual_seed(3)
    model = TCNClassifier(2, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=3).eval()
    short_sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    long_sequence = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    collated = collate_trajectory_batch([
        {"sequence": short_sequence, "label": 0},
        {"sequence": long_sequence, "label": 1},
    ])

    batch_logits = model(collated["sequences"], lengths=collated["lengths"])
    short_logits = model(short_sequence.transpose(0, 1).unsqueeze(0))
    long_logits = model(long_sequence.transpose(0, 1).unsqueeze(0))

    torch.testing.assert_close(batch_logits[0], short_logits[0])
    torch.testing.assert_close(batch_logits[1], long_logits[0])


def test_classifier_lengths_ignore_nonzero_values_after_each_real_sequence_end():
    torch.manual_seed(5)
    model = TCNClassifier(2, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=3).eval()
    sequences = torch.tensor(
        [
            [[1.0, 3.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]],
            [[5.0, 7.0, 9.0, 11.0], [6.0, 8.0, 10.0, 12.0]],
        ]
    )
    lengths = torch.tensor([2, 4])
    altered_padding = sequences.clone()
    altered_padding[0, :, 2:] = 999.0

    baseline_logits = model(sequences, lengths=lengths)
    altered_logits = model(altered_padding, lengths=lengths)

    torch.testing.assert_close(baseline_logits, altered_logits)
