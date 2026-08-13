import pytest
import torch

from TCN_Singh2024.src.preprocess import min_max_normalize, root_point_translate


def test_paper_literal_root_translation_places_first_point_at_zero():
    sequence = torch.tensor([[2.0, 5.0, -1.0], [4.0, 3.0, 2.0]])

    translated = root_point_translate(sequence)

    torch.testing.assert_close(translated, torch.tensor([[0.0, 0.0, 0.0], [-2.0, 2.0, -3.0]]))


def test_conventional_relative_translation_is_explicit_and_has_opposite_direction():
    sequence = torch.tensor([[2.0, 5.0, -1.0], [4.0, 3.0, 2.0]])

    translated = root_point_translate(sequence, mode="conventional_relative")

    torch.testing.assert_close(translated, torch.tensor([[0.0, 0.0, 0.0], [2.0, -2.0, 3.0]]))


def test_preprocessing_does_not_modify_the_input_tensor():
    sequence = torch.tensor([[2.0, 5.0], [4.0, 3.0]])
    original = sequence.clone()

    _ = root_point_translate(sequence)
    _ = min_max_normalize(sequence)

    assert torch.equal(sequence, original)


def test_min_max_normalization_is_per_sequence_per_channel_over_time():
    sequence = torch.tensor([[2.0, 10.0], [4.0, 20.0], [6.0, 30.0]])

    normalized = min_max_normalize(sequence)

    torch.testing.assert_close(normalized, torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))
    assert torch.all((normalized >= 0.0) & (normalized <= 1.0))


def test_min_max_normalization_maps_constant_channels_to_zero_without_nonfinite_values():
    sequence = torch.tensor([[4.0, 5.0, 1.0], [4.0, 7.0, 1.0], [4.0, 9.0, 1.0]])

    normalized = min_max_normalize(sequence)

    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized[:, 0], torch.zeros(3))
    torch.testing.assert_close(normalized[:, 2], torch.zeros(3))


def test_min_max_normalization_preserves_unit_range_when_channel_span_is_less_than_one():
    sequence = torch.tensor([[0.20], [0.30], [0.40]])

    normalized = min_max_normalize(sequence)

    torch.testing.assert_close(normalized, torch.tensor([[0.0], [0.5], [1.0]]))


@pytest.mark.parametrize("channels", [3, 6])
def test_preprocess_supports_arbitrary_positive_channel_counts(channels):
    sequence = torch.arange(4 * channels, dtype=torch.float32).reshape(4, channels)

    assert root_point_translate(sequence).shape == (4, channels)
    assert min_max_normalize(sequence).shape == (4, channels)
