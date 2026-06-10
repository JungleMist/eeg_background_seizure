"""Unit tests for eeg_bg.features.complexity."""
import numpy as np
import pytest
from eeg_bg.features.complexity import (
    sample_entropy,
    lempel_ziv_complexity,
    complexity_features,
    COMPLEXITY_NAMES,
)
from eeg_bg.features._constants import _STANDARD_19


def test_complexity_names_length():
    assert len(COMPLEXITY_NAMES) == 38
    assert len(COMPLEXITY_NAMES) == len(_STANDARD_19) * 2


def test_complexity_names_unique():
    assert len(COMPLEXITY_NAMES) == len(set(COMPLEXITY_NAMES))


def test_complexity_features_shape():
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = complexity_features(epoch, _STANDARD_19)
    assert out.shape == (38,)


def test_complexity_features_dtype():
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = complexity_features(epoch, _STANDARD_19)
    assert out.dtype == np.float64


def test_sample_entropy_random_positive():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(1000).astype(np.float64)
    assert sample_entropy(sig) > 0.0


def test_sample_entropy_constant_zero():
    """Constant signal is perfectly regular → SampEn = 0."""
    const = np.full(1000, 3.0, dtype=np.float64)
    assert sample_entropy(const) == pytest.approx(0.0, abs=1e-6)


def test_lempel_ziv_range():
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(1000).astype(np.float64)
    lzc = lempel_ziv_complexity(sig)
    assert 0.0 <= lzc


def test_lempel_ziv_random_greater_than_constant():
    """Random signal should have higher LZC than a constant."""
    rng = np.random.default_rng(4)
    rand_sig  = rng.standard_normal(1000).astype(np.float64)
    const_sig = np.full(1000, 1.0, dtype=np.float64)
    assert lempel_ziv_complexity(rand_sig) > lempel_ziv_complexity(const_sig)


def test_complexity_features_missing_channel():
    """Missing channels yield zeros."""
    rng = np.random.default_rng(5)
    epoch = rng.standard_normal((1, 1000)).astype(np.float64) * 10.0
    out = complexity_features(epoch, ["FP1"])
    assert out.shape == (38,)
    # All channels except FP1 should be zero
    fp1_idx = _STANDARD_19.index("FP1")
    assert out[fp1_idx * 2]     > 0.0  # FP1 sample_entropy
    other = [i for i in range(38) if i not in (fp1_idx * 2, fp1_idx * 2 + 1)]
    assert np.all(out[other] == 0.0)
