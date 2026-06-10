"""Unit tests for eeg_bg.features.connectivity."""
import numpy as np
import pytest
from eeg_bg.features.connectivity import (
    connectivity_features,
    CONNECTIVITY_NAMES,
    ALL_PAIRS,
)
from eeg_bg.features._constants import _STANDARD_19

SFREQ = 125.0
N_PAIRS = 171  # C(19,2)
N_BANDS = 5
N_METRICS = 2
EXPECTED_DIM = N_PAIRS * N_BANDS * N_METRICS  # 1710


@pytest.fixture
def random_epoch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((19, 1000)).astype(np.float64) * 20.0


@pytest.fixture
def identical_epoch():
    """Epoch where all channels are identical — coherence and PLV should be 1."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(1000).astype(np.float64) * 20.0
    return np.tile(sig, (19, 1))


def test_all_pairs_count():
    assert len(ALL_PAIRS) == N_PAIRS


def test_all_pairs_ordered():
    """ch1 index < ch2 index for all pairs."""
    for ch1, ch2 in ALL_PAIRS:
        assert _STANDARD_19.index(ch1) < _STANDARD_19.index(ch2)


def test_connectivity_names_length():
    assert len(CONNECTIVITY_NAMES) == EXPECTED_DIM


def test_connectivity_names_unique():
    assert len(CONNECTIVITY_NAMES) == len(set(CONNECTIVITY_NAMES))


def test_connectivity_features_shape(random_epoch):
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert out.shape == (EXPECTED_DIM,)


def test_connectivity_features_dtype(random_epoch):
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert out.dtype == np.float64


def test_connectivity_range(random_epoch):
    """All coherence and PLV values should be in [0, 1]."""
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0 + 1e-6)


def test_connectivity_identical_channels_high_coherence(identical_epoch):
    """Identical channels should yield coherence ≈ 1 for all bands."""
    out = connectivity_features(identical_epoch, _STANDARD_19, sfreq=SFREQ)
    # Coherence values are at even indices within each pair-band block
    coh_values = out[0::2]
    assert np.all(coh_values > 0.99)


def test_connectivity_identical_channels_high_plv(identical_epoch):
    """Identical channels should yield PLV ≈ 1 for all bands."""
    out = connectivity_features(identical_epoch, _STANDARD_19, sfreq=SFREQ)
    plv_values = out[1::2]
    assert np.all(plv_values > 0.99)


def test_connectivity_missing_channel():
    """Missing channel → all 10 values for its pairs are zero."""
    rng = np.random.default_rng(2)
    # Only 2 channels present (FP1, FP2), all other channels missing
    epoch = rng.standard_normal((2, 1000)).astype(np.float64) * 20.0
    out = connectivity_features(epoch, ["FP1", "FP2"], sfreq=SFREQ)
    assert out.shape == (EXPECTED_DIM,)
    # The (FP1, FP2) pair is the first pair in ALL_PAIRS — those 10 values should be nonzero
    assert not np.all(out[:10] == 0.0)
    # All other pairs involve at least one absent channel — should be zero
    assert np.all(out[10:] == 0.0)
