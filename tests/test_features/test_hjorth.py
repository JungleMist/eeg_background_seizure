"""Unit tests for eeg_bg.features.hjorth."""
import numpy as np
import pytest

from eeg_bg.features.hjorth import hjorth_parameters


def test_activity_equals_variance(synthetic_epoch):
    sig = synthetic_epoch[0]
    activity, _, _ = hjorth_parameters(sig)
    assert abs(activity - float(np.var(sig))) < 1e-10


def test_returns_three_floats(pure_sine_signal):
    result = hjorth_parameters(pure_sine_signal)
    assert len(result) == 3
    for v in result:
        assert isinstance(v, float)


def test_constant_signal_no_division_error(constant_signal):
    """A flat signal (zero variance) must not raise ZeroDivisionError."""
    activity, mobility, complexity = hjorth_parameters(constant_signal)
    assert activity == pytest.approx(0.0, abs=1e-10)
    assert np.isfinite(mobility)
    assert np.isfinite(complexity)


def test_pure_sine_positive_values(pure_sine_signal):
    activity, mobility, complexity = hjorth_parameters(pure_sine_signal)
    assert activity > 0
    assert mobility > 0
    assert complexity > 0


def test_random_epoch_all_positive(synthetic_epoch):
    for ch_idx in range(synthetic_epoch.shape[0]):
        act, mob, cplx = hjorth_parameters(synthetic_epoch[ch_idx])
        assert act >= 0 and mob >= 0 and cplx >= 0
