"""Unit tests for eeg_bg.features.temporal_stats."""
import numpy as np
import pytest
from eeg_bg.features.temporal_stats import (
    temporal_stats_features,
    epoch_temporal_stats,
    TEMPORAL_NAMES,
    _DEFAULT_SCALES,
)
from eeg_bg.features._constants import _STANDARD_19


def test_temporal_names_length():
    assert len(TEMPORAL_NAMES) == 228
    assert len(TEMPORAL_NAMES) == len(_STANDARD_19) * len(_DEFAULT_SCALES) * 4


def test_temporal_names_unique():
    assert len(TEMPORAL_NAMES) == len(set(TEMPORAL_NAMES))


def test_temporal_stats_features_shape():
    out = temporal_stats_features(np.random.default_rng(0).standard_normal(1000))
    assert out.shape == (len(_DEFAULT_SCALES) * 4,)


def test_epoch_temporal_stats_shape():
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert out.shape == (228,)


def test_epoch_temporal_stats_dtype():
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert out.dtype == np.float64


def test_temporal_constant_zero_variance():
    """Constant signal has zero variance at every scale."""
    const = np.full(1000, 7.0, dtype=np.float64)
    out = temporal_stats_features(const)
    # Variance is at index 1, 5, 9 (every 4th starting at 1)
    var_indices = [i * 4 + 1 for i in range(len(_DEFAULT_SCALES))]
    assert np.allclose(out[var_indices], 0.0, atol=1e-8)


def test_temporal_constant_zero_skew():
    """Constant signal has zero skewness at every scale."""
    const = np.full(1000, 7.0, dtype=np.float64)
    out = temporal_stats_features(const)
    skew_indices = [i * 4 + 2 for i in range(len(_DEFAULT_SCALES))]
    assert np.allclose(out[skew_indices], 0.0, atol=1e-8)


def test_temporal_missing_channel_zero_padded():
    rng = np.random.default_rng(2)
    epoch = rng.standard_normal((1, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, ["FP1"])
    assert out.shape == (228,)
    fp1_idx = _STANDARD_19.index("FP1")
    n_feats_per_ch = len(_DEFAULT_SCALES) * 4  # 12
    other = [i for i in range(228) if not (fp1_idx * n_feats_per_ch <= i < (fp1_idx + 1) * n_feats_per_ch)]
    assert np.all(out[other] == 0.0)


def test_temporal_output_finite():
    rng = np.random.default_rng(3)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert np.all(np.isfinite(out))
