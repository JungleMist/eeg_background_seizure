"""Zero-referenced phase gate helpers for Wiener decomposition."""
from __future__ import annotations

import numpy as np


DEFAULT_PHASE_GATE_THRESHOLD_RAD = 0.392


def validate_phase_gate_threshold(threshold_rad: float) -> float:
    """Return a valid phase-gate threshold in radians.

    The zero-referenced gate treats 0 as the minimum phase difference and pi
    as the maximum.  A threshold of pi therefore admits all phases.
    """
    threshold = float(threshold_rad)
    if not np.isfinite(threshold):
        raise ValueError("phase_gate_threshold_rad must be finite")
    if threshold < 0.0 or threshold > np.pi:
        raise ValueError("phase_gate_threshold_rad must be in [0, pi]")
    return threshold


def phase_gate_threshold_from_config(cfg: dict) -> float:
    """Read ``wiener.phase_gate_threshold_rad`` with the project default."""
    return validate_phase_gate_threshold(
        cfg["wiener"].get(
            "phase_gate_threshold_rad",
            DEFAULT_PHASE_GATE_THRESHOLD_RAD,
        )
    )


def phase_gate_weights(
    S: np.ndarray,
    target_idx: int,
    threshold_rad: float,
) -> np.ndarray:
    """Compute hard zero-referenced phase-gate weights.

    Parameters
    ----------
    S : np.ndarray
        Cross-PSD matrix, shape ``(n_ch, n_ch, n_freqs)``.
    target_idx : int
        Local channel index to predict from all other channels.
    threshold_rad : float
        Maximum accepted phase distance from 0, in radians.  A value of pi
        returns all ones exactly, making the gated complex Wiener filter
        identical to the frequency-domain Wiener filter.

    Returns
    -------
    np.ndarray
        Real weights with shape ``(n_ref, n_freqs)``.
    """
    threshold = validate_phase_gate_threshold(threshold_rad)
    ref_indices = [i for i in range(S.shape[0]) if i != target_idx]
    n_freqs = S.shape[2]

    if threshold == np.pi:
        return np.ones((len(ref_indices), n_freqs), dtype=np.float64)

    cross = S[target_idx, ref_indices, :]
    phase_dist = np.abs(np.angle(cross))
    return (phase_dist <= threshold).astype(np.float64)
