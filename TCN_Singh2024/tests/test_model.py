import torch
from torch import nn
import pytest

from TCN_Singh2024.src.model import CausalConv1d, TCNClassifier, TCNResidualBlock


def test_causal_conv_preserves_temporal_length():
    layer = CausalConv1d(2, 4, kernel_size=3, dilation=4)
    output = layer(torch.randn(3, 2, 17))

    assert output.shape == (3, 4, 17)


def test_causal_conv_future_changes_do_not_affect_past_outputs():
    layer = CausalConv1d(1, 1, kernel_size=3, dilation=2, bias=False)
    nn.init.constant_(layer.conv.weight, 1.0)
    baseline = torch.zeros(1, 1, 12)
    changed_future = baseline.clone()
    changed_future[:, :, 8:] = 7.0

    baseline_output = layer(baseline)
    changed_output = layer(changed_future)

    assert torch.equal(baseline_output[:, :, :8], changed_output[:, :, :8])


def test_causal_conv_uses_configured_dilation():
    layer = CausalConv1d(1, 1, kernel_size=3, dilation=3, bias=False)
    nn.init.constant_(layer.conv.weight, 1.0)
    sequence = torch.zeros(1, 1, 10)
    sequence[:, :, 0] = 1.0

    output = layer(sequence)

    assert layer.conv.dilation == (3,)
    assert layer.left_padding == 6
    assert torch.nonzero(output[0, 0], as_tuple=False).flatten().tolist() == [0, 3, 6]


def test_residual_block_uses_identity_shortcut_for_matching_channels():
    block = TCNResidualBlock(4, 4, kernel_size=3, dilation=1, dropout=0.0)

    assert isinstance(block.shortcut, nn.Identity)
    assert block(torch.randn(2, 4, 9)).shape == (2, 4, 9)


def test_residual_block_projects_channel_mismatch_with_pointwise_conv():
    block = TCNResidualBlock(3, 5, kernel_size=3, dilation=2, dropout=0.0)

    assert isinstance(block.shortcut, nn.Conv1d)
    assert block.shortcut.kernel_size == (1,)
    assert block(torch.randn(2, 3, 9)).shape == (2, 5, 9)


def test_classifier_stacks_blocks_and_returns_logits_for_any_input_channel_count():
    model = TCNClassifier(
        input_channels=6,
        hidden_channels=[5, 7, 7],
        kernel_size=3,
        dilations=[1, 2, 4],
        dropout=0.1,
        num_classes=4,
    )

    output = model(torch.randn(3, 6, 13))

    assert len(model.blocks) == 3
    assert model.forward_features(torch.randn(3, 6, 13)).shape == (3, 7, 13)
    assert output.shape == (3, 4)
    assert [block.conv1.conv.dilation for block in model.blocks] == [(1,), (2,), (4,)]
    assert [block.conv2.conv.dilation for block in model.blocks] == [(1,), (2,), (4,)]


@pytest.mark.parametrize("sequence_length", [1, 2])
def test_classifier_preserves_short_sequence_length(sequence_length):
    model = TCNClassifier(3, [4, 4], kernel_size=3, dilations=[1, 2], dropout=0.0, num_classes=2)

    assert model.forward_features(torch.randn(2, 3, sequence_length)).shape == (2, 4, sequence_length)


def test_classifier_rejects_empty_temporal_sequences():
    model = TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=0.0, num_classes=2)

    with pytest.raises(ValueError, match="temporal length"):
        model(torch.randn(2, 3, 0))


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_classifier_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        TCNClassifier(3, [4], kernel_size=3, dilations=[1], dropout=dropout, num_classes=2)


def test_stacked_classifier_future_changes_do_not_affect_past_features():
    torch.manual_seed(0)
    model = TCNClassifier(
        input_channels=3,
        hidden_channels=[4, 4],
        kernel_size=3,
        dilations=[1, 2],
        dropout=0.4,
        num_classes=2,
    ).eval()
    inputs = torch.randn(2, 3, 12)
    future_changed = inputs.clone()
    future_changed[:, :, 8:] += 100.0

    baseline_features = model.forward_features(inputs)
    changed_features = model.forward_features(future_changed)

    torch.testing.assert_close(baseline_features[:, :, :8], changed_features[:, :, :8])


def test_classifier_supports_cross_entropy_backward_without_nonfinite_values():
    model = TCNClassifier(
        input_channels=3,
        hidden_channels=[4, 4],
        kernel_size=3,
        dilations=[1, 2],
        dropout=0.0,
        num_classes=3,
    )
    logits = model(torch.randn(4, 3, 11))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 1]))
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert not any(isinstance(module, nn.Softmax) for module in model.modules())
