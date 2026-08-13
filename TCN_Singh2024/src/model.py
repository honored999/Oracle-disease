"""Causal dilated residual TCN classifier for trajectory sequences."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    """Conv1d with explicit left-only padding and unchanged temporal length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be at least 1")
        if dilation < 1:
            raise ValueError("dilation must be at least 1")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(nn.functional.pad(inputs, (self.left_padding, 0)))


class TCNResidualBlock(nn.Module):
    """Two causal dilated convolutions plus a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.shortcut: nn.Module
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.dropout1(self.relu1(self.conv1(inputs)))
        outputs = self.dropout2(self.relu2(self.conv2(outputs)))
        return outputs + self.shortcut(inputs)


class TCNClassifier(nn.Module):
    """Stacked TCN residual blocks followed by last-time-step classification."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        if not hidden_channels:
            raise ValueError("hidden_channels must contain at least one block")
        if len(hidden_channels) != len(dilations):
            raise ValueError("hidden_channels and dilations must have equal length")
        if input_channels < 1 or num_classes < 1:
            raise ValueError("input_channels and num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        channels = [input_channels, *hidden_channels]
        self.input_channels = input_channels
        self.blocks = nn.ModuleList(
            TCNResidualBlock(
                in_channels=channels[index],
                out_channels=channels[index + 1],
                kernel_size=kernel_size,
                dilation=dilations[index],
                dropout=dropout,
            )
            for index in range(len(hidden_channels))
        )
        self.classifier = nn.Linear(hidden_channels[-1], num_classes)

    def forward_features(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [B, C, T]")
        if inputs.shape[1] != self.input_channels:
            raise ValueError("input channel count does not match the configured model")
        if inputs.shape[2] < 1:
            raise ValueError("temporal length must be at least 1")
        features = inputs
        for block in self.blocks:
            features = block(features)
        return features

    def forward(self, inputs: Tensor, lengths: Tensor | None = None) -> Tensor:
        temporal_features = self.forward_features(inputs)
        if lengths is None:
            last_features = temporal_features[:, :, -1]
        else:
            if lengths.ndim != 1 or lengths.shape[0] != inputs.shape[0]:
                raise ValueError("lengths must have shape [B]")
            if not torch.all((lengths >= 1) & (lengths <= inputs.shape[2])):
                raise ValueError("lengths must be within the temporal input range")
            indices = lengths.to(device=inputs.device, dtype=torch.long) - 1
            last_features = temporal_features[torch.arange(inputs.shape[0], device=inputs.device), :, indices]
        return self.classifier(last_features)
