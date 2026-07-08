import numpy as np
import pytest

from eeg_bg.decomposition.phase_gate import (
    phase_gate_weights,
    validate_phase_gate_threshold,
)


def _two_channel_spectrum(cross_values: list[complex]) -> np.ndarray:
    S = np.zeros((2, 2, len(cross_values)), dtype=complex)
    S[0, 0] = 1.0
    S[1, 1] = 1.0
    S[0, 1] = np.asarray(cross_values, dtype=complex)
    S[1, 0] = np.conj(S[0, 1])
    return S


def test_phase_zero_passes_at_zero_threshold():
    S = _two_channel_spectrum([1.0 + 0.0j])
    weights = phase_gate_weights(S, target_idx=0, threshold_rad=0.0)
    np.testing.assert_array_equal(weights, np.ones((1, 1)))


def test_phase_pi_is_blocked_below_pi_threshold():
    S = _two_channel_spectrum([-1.0 + 0.0j])

    weights_zero = phase_gate_weights(S, target_idx=0, threshold_rad=0.0)
    weights_default = phase_gate_weights(S, target_idx=0, threshold_rad=0.392)

    np.testing.assert_array_equal(weights_zero, np.zeros((1, 1)))
    np.testing.assert_array_equal(weights_default, np.zeros((1, 1)))


def test_threshold_pi_returns_all_ones():
    S = _two_channel_spectrum([1.0 + 0.0j, 1.0j, -1.0 + 0.0j])
    weights = phase_gate_weights(S, target_idx=0, threshold_rad=np.pi)
    np.testing.assert_array_equal(weights, np.ones((1, 3)))


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match=r"\[0, pi\]"):
        validate_phase_gate_threshold(np.pi + 0.1)
