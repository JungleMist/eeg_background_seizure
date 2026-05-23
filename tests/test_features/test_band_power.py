"""Unit tests for eeg_bg.features.band_power."""
import numpy as np
import pytest
from unittest.mock import patch

from eeg_bg.features.band_power import relative_band_power, BANDS


def test_band_keys(pure_sine_signal, sfreq):
    result = relative_band_power(pure_sine_signal, sfreq=sfreq)
    assert set(result.keys()) == set(BANDS.keys())


def test_relative_powers_are_floats(pure_sine_signal, sfreq):
    result = relative_band_power(pure_sine_signal, sfreq=sfreq)
    for v in result.values():
        assert isinstance(v, float)


def test_relative_powers_sum_to_one(pure_sine_signal, sfreq):
    """Five band powers should sum to ~1 within the total analysis band."""
    result = relative_band_power(pure_sine_signal, sfreq=sfreq)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.05  # trapezoidal-integration boundary tolerance


def test_dominant_band_for_pure_alpha_sine(pure_sine_signal, sfreq):
    """A 10 Hz sinusoid should have highest alpha-band power."""
    result = relative_band_power(pure_sine_signal, sfreq=sfreq)
    assert result["alpha"] == max(result.values())


def test_dominant_band_for_pure_delta_sine(sfreq):
    """A 2 Hz sinusoid should have highest delta-band power."""
    t = np.arange(1000) / sfreq
    sig = (np.sin(2 * np.pi * 2.0 * t) * 50.0).astype(np.float64)
    result = relative_band_power(sig, sfreq=sfreq)
    assert result["delta"] == max(result.values())


def test_uses_boxcar_welch(pure_sine_signal, sfreq):
    """Welch must be called with window='boxcar' and nperseg=250."""
    with patch("eeg_bg.features.band_power.welch", wraps=__import__(
        "scipy.signal", fromlist=["welch"]
    ).welch) as mock_welch:
        relative_band_power(pure_sine_signal, sfreq=sfreq, nperseg=250)
        assert mock_welch.call_count >= 1
        _, kwargs = mock_welch.call_args
        assert kwargs.get("window") == "boxcar"
        assert kwargs.get("nperseg") == 250


def test_all_values_nonnegative(pure_sine_signal, sfreq):
    result = relative_band_power(pure_sine_signal, sfreq=sfreq)
    for v in result.values():
        assert v >= 0.0
