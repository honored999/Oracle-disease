"""Trajectory preprocessing specified or motivated by Singh & Koundal (2024)."""

from typing import Literal

import torch
from torch import Tensor


TranslationMode = Literal["paper_literal", "conventional_relative"]


def _validate_sequence(sequence: Tensor) -> None:
    if sequence.ndim != 2:
        raise ValueError("sequence must have shape [T, C]")
    if sequence.shape[0] < 1 or sequence.shape[1] < 1:
        raise ValueError("sequence must have positive T and C")


def root_point_translate(sequence: Tensor, mode: TranslationMode = "paper_literal") -> Tensor:
    """Align a [T, C] trajectory to its first point without modifying it."""
    _validate_sequence(sequence)
    if mode == "paper_literal":
        return sequence[0:1] - sequence
    if mode == "conventional_relative":
        return sequence - sequence[0:1]
    raise ValueError("mode must be 'paper_literal' or 'conventional_relative'")


def min_max_normalize(sequence: Tensor) -> Tensor:
    """Min-Max normalize each channel of one [T, C] sequence over time."""
    _validate_sequence(sequence)
    minimum = sequence.amin(dim=0, keepdim=True)
    maximum = sequence.amax(dim=0, keepdim=True)
    denominator = maximum - minimum
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    return (sequence - minimum) / safe_denominator
