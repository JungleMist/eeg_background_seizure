"""Unit tests for eeg_bg.features.wavelet."""
import numpy as np
import pytest
from eeg_bg.features.wavelet import wavelet_features, WAVELET_NAMES, _DEFAULT_LEVELS

SFREQ = 125.0


@pytest.fixture
def sine_signal():
    t = np.arange(1000) / SFREQ
    return (np.sin(2 * np.pi * 10.0 * t) * 50.0).astype(np.float64)


@pytest.fixture
def random_signal():
    return np.random.default_rng(0).standard_normal(1000).astype(np.float64) * 20.0


def test_wavelet_features_shape(sine_signal):
    out = wavelet_features(sine_signal)
    assert out.shape == (66,)


def test_wavelet_names_length():
    from eeg_bg.features._constants import _STANDARD_19
    assert len(WAVELET_NAMES) == len(_STANDARD_19) * 66
    assert len(WAVELET_NAMES) == 1254


def test_wavelet_names_unique():
    assert len(WAVELET_NAMES) == len(set(WAVELET_NAMES))


def test_wavelet_energy_nonnegative(random_signal):
    out = wavelet_features(random_signal)
    energies = out[0:6]
    assert np.all(energies >= 0.0)


def test_wavelet_entropy_nonnegative(random_signal):
    out = wavelet_features(random_signal)
    entropies = out[6:12]
    assert np.all(entropies >= 0.0)


def test_wavelet_sine_dominant_level(sine_signal):
    """10 Hz sine should have highest energy at level 3 (~8–16 Hz at 125 Hz)."""
    out = wavelet_features(sine_signal)
    energies = out[0:6]  # index 0=l1, 1=l2, 2=l3, ..., 5=l6 energies
    # Level 3 energy (index 2) should dominate for 10 Hz
    assert energies[2] == energies.max()


def test_wavelet_constant_zero_entropy():
    """Constant signal should have near-zero energy at all levels (floating-point rounding)."""
    const = np.full(1000, 5.0, dtype=np.float64)
    out = wavelet_features(const)
    energies = out[0:6]
    assert np.allclose(energies, 0.0, atol=1e-12)


def test_wavelet_output_dtype(random_signal):
    out = wavelet_features(random_signal)
    assert out.dtype == np.float64


def test_wavelet_detail_coef_stats_finite(random_signal):
    out = wavelet_features(random_signal)
    assert np.all(np.isfinite(out[12:30]))


def test_wavelet_approx_stats_finite(random_signal):
    out = wavelet_features(random_signal)
    assert np.all(np.isfinite(out[30:33]))


def test_wavelet_modulus_maxima_count_nonneg(random_signal):
    out = wavelet_features(random_signal)
    counts = out[33:45:2]  # mmx_count at even positions within the mmx group
    assert np.all(counts >= 0.0)


def test_wavelet_scale_energy_ratio_sums_to_one(random_signal):
    out = wavelet_features(random_signal)
    ratios = out[45:51]
    assert abs(ratios.sum() - 1.0) < 1e-6


def test_wavelet_reconstructed_power_nonneg(random_signal):
    out = wavelet_features(random_signal)
    # power is every 3rd value starting at index 53 within the rec group
    powers = out[51:66][2::3]  # indices 53, 56, 59, 62, 65 relative to full vector
    assert np.all(powers >= 0.0)


def test_wavelet_reconstructed_selectivity(sine_signal):
    """10 Hz sine: alpha band (level 3, 8-16 Hz) should have highest reconstructed power."""
    out = wavelet_features(sine_signal)
    powers = out[51:66][2::3]  # delta, theta, alpha, beta, gamma powers
    assert powers[2] == powers.max()  # alpha (index 2) dominates


def test_wavelet_zero_signal_all_zeros():
    const = np.zeros(1000, dtype=np.float64)
    out = wavelet_features(const)
    assert np.allclose(out, 0.0, atol=1e-12)
