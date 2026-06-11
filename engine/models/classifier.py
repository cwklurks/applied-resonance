"""Tiny 4-conv-block CNN that classifies a log-mel clip by machine type."""

import torch
from torch import nn

MACHINE_CLASSES = ("fan", "pump")


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MachineTypeCNN(nn.Module):
    """4 conv blocks over a log-mel spectrogram ``(B, 1, 64, T)``.

    Channels 1 -> 16 -> 32 -> 64 -> 128, then ``AdaptiveAvgPool2d(1)``,
    ``Flatten`` and a ``Linear(128, n_classes)`` head. The adaptive pool makes
    the model agnostic to the time dimension ``T``.
    """

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(B, n_classes)``."""
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)
