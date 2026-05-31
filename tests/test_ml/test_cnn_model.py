"""Unit tests for EEGNet."""
import pytest
import torch

from eeg_bg.ml.cnn_model import EEGNet


def test_eegnet_output_shape():
    """Forward pass produces (batch, 1) output."""
    model = EEGNet(n_channels=19, n_times=1000, F1=8, D=2, dropout=0.0)
    x = torch.randn(4, 1, 19, 1000)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_eegnet_output_is_probability():
    """Sigmoid output must be in [0, 1]."""
    model = EEGNet(n_channels=19, n_times=1000)
    model.eval()
    x = torch.randn(4, 1, 19, 1000)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_eegnet_batch_size_one():
    """Model handles batch size of 1 without BatchNorm errors."""
    model = EEGNet(n_channels=19, n_times=1000)
    model.eval()
    x = torch.randn(1, 1, 19, 1000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_eegnet_custom_hyperparams():
    """Model instantiates and runs with non-default F1/D values."""
    model = EEGNet(n_channels=19, n_times=1000, F1=4, D=4, dropout=0.5)
    x = torch.randn(2, 1, 19, 1000)
    out = model(x)
    assert out.shape == (2, 1)
