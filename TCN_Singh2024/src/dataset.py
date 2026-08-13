"""In-memory trajectory dataset and minimal variable-length batch collation."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _validate_sequence(sequence: Tensor) -> None:
    if sequence.ndim != 2 or sequence.shape[0] < 1 or sequence.shape[1] < 1:
        raise ValueError("each sequence must have shape [T, C] with positive dimensions")


class TrajectoryClassificationDataset(Dataset[dict[str, Tensor | int]]):
    """Generic in-memory trajectory samples using sequence [T, C] and label int."""

    def __init__(self, sequences: Sequence[Tensor], labels: Sequence[int]) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have equal length")
        self._sequences = []
        for sequence in sequences:
            _validate_sequence(sequence)
            self._sequences.append(sequence.detach().clone())
        self._labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> dict[str, Tensor | int]:
        return {"sequence": self._sequences[index].clone(), "label": self._labels[index]}


def collate_trajectory_batch(batch: Sequence[dict[str, Tensor | int]]) -> dict[str, Tensor]:
    """Right-pad [T, C] samples and return model-ready [B, C, T_max] plus lengths."""
    if not batch:
        raise ValueError("batch must not be empty")
    sequences = [item["sequence"] for item in batch]
    if not all(isinstance(sequence, Tensor) for sequence in sequences):
        raise TypeError("each batch item must contain a Tensor sequence")
    tensor_sequences = [sequence for sequence in sequences if isinstance(sequence, Tensor)]
    for sequence in tensor_sequences:
        _validate_sequence(sequence)
    channels = tensor_sequences[0].shape[1]
    if any(sequence.shape[1] != channels for sequence in tensor_sequences):
        raise ValueError("all sequences in a batch must have the same channel count")
    lengths = torch.tensor([sequence.shape[0] for sequence in tensor_sequences], dtype=torch.long)
    padded = tensor_sequences[0].new_zeros((len(tensor_sequences), channels, int(lengths.max())))
    for index, sequence in enumerate(tensor_sequences):
        padded[index, :, : sequence.shape[0]] = sequence.transpose(0, 1)
    return {
        "sequences": padded,
        "lengths": lengths,
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
    }
