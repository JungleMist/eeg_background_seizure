"""EEGNet — compact EEG classification CNN.

Reference: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.

Architecture
------------
Block 1: Temporal conv  — learns frequency-band-like filters along time.
Block 2: Depthwise spatial conv — combines channels within each filter.
Block 3: Separable temporal conv — refines temporal representation compactly.
Output:  Sigmoid scalar — probability of class 1 (control).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for binary EEG classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (height of the input "image").  Default 19.
    n_times : int
        Number of time samples per epoch.  Default 1000.
    F1 : int
        Number of temporal filters in Block 1.  Default 8.
    D : int
        Depth multiplier; F2 = F1 * D filters after Block 2.  Default 2.
    dropout : float
        Dropout probability applied after Blocks 2 and 3.  Default 0.25.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 1000,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        F2 = F1 * D

        # Block 1 — Temporal conv: learns spectral-band-like filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — Depthwise spatial conv: mixes channels per filter
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3 — Separable temporal conv: compact temporal refinement
        self.block3 = nn.Sequential(
            # Depthwise part
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False),
            # Pointwise part
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flatten size by passing a dummy tensor through the conv blocks.
        # This avoids hardcoding the arithmetic and adapts to any n_times value.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            flatten_size = dummy.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 1, n_channels, n_times)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``, values in ``[0, 1]``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
