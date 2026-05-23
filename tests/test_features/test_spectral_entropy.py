"""Unit tests for eeg_bg.features.spectral_entropy."""
import numpy as np
import pytest
from unittest.mock import patch

from eeg_bg.features.spectral_entropy import spectral_entropy


def test_returns_float(pure_sine_signal, sfreq):
    val = spectral_entropy(pure_sine_signal, sfreq=sfreq)
    assert isinstance(val, float)


def test_always_nonnegative(pure_sine_signal, sfreq):
    assert spectral_entropy(pure_sine_signal, sfreq=sfreq) >= 0.0


def test_white_noise_high_entropy(synthetic_epoch, sfreq):
    """Broadband noise should produce a higher entropy than a pure tone."""
    noise_sig = synthetic_epoch[0]
    t = np.arange(noise_sig.size) / sfreq
    pure_sig  = (np.sin(2 * np.pi * 10.0 * t) * 50.0).astype(np.float64)
    entropy_noise = spectral_entropy(noise_sig, sfreq=sfreq)
    entropy_pure  = spectral_entropy(pure_sig,  sfreq=sfreq)
    assert entropy_noise > entropy_pure


def test_pure_tone_low_entropy(pure_sine_signal, sfreq):
    """A pure sine should have near-zero spectral entropy (energy concentrated)."""
    ent = spectral_entropy(pure_sine_signal, sfreq=sfreq)
    # Max possible entropy ≈ log(n_freq_bins) ~ log(80) ≈ 4.4
    assert ent < 1.0


def test_uses_boxcar_welch(pure_sine_signal, sfreq):
    with patch("eeg_bg.features.spectral_entropy.welch", wraps=__import__(
        "scipy.signal", fromlist=["welch"]
    ).welch) as mock_welch:
        spectral_entropy(pure_sine_signal, sfreq=sfreq, nperseg=250)
        _, kwargs = mock_welch.call_args
        assert kwargs.get("window") == "boxcar"
        assert kwargs.get("nperseg") == 250
