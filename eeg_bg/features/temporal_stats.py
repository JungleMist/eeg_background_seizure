"""Multi-scale temporal statistics per EEG channel.

Splits each epoch into non-overlapping windows of fixed sample lengths
and computes mean, variance, skewness, and kurtosis across windows.
Three scales capture burst structure at 1 s, 3 s, and 6 s granularities
(125, 375, 750 samples at 125 Hz).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

from eeg_bg.features._constants import _STANDARD_19

_DEFAULT_SCALES: list[int] = [125, 375, 750]

_TEMPORAL_SUFFIXES: list[str] = [
    f"scale{s}_{stat}"
    for s in _DEFAULT_SCALES
    for stat in ("mean", "var", "skew", "kurtosis")
]  # 3 × 4 = 12 per channel

TEMPORAL_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _TEMPORAL_SUFFIXES
]  # 19 × 12 = 228


def temporal_stats_features(
    signal: np.ndarray,
    scales: list[int] = _DEFAULT_SCALES,
) -> np.ndarray:
    """Compute multi-scale temporal statistics for one signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    scales : list[int]
        Non-overlapping window lengths in samples.

    Returns
    -------
    np.ndarray
        Shape ``(4 * len(scales),)``, dtype float64.
        Layout per scale: ``[mean, var, skew, kurtosis]``.
    """
    x = signal.astype(np.float64)
    n = len(x)
    feats: list[float] = []
    for s in scales:
        n_windows = n // s
        if n_windows == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        windows = x[:n_windows * s].reshape(n_windows, s)  # (n_windows, s)
        feats.append(float(windows.mean(axis=1).mean()))
        feats.append(float(windows.var(axis=1).mean()))
        skew_vals = scipy_skew(windows, axis=1)
        # Replace NaN (from constant windows) with 0.0
        skew_vals = np.nan_to_num(skew_vals, nan=0.0)
        feats.append(float(skew_vals.mean()))
        kurt_vals = scipy_kurtosis(windows, axis=1)
        # Replace NaN (from constant windows) with 0.0
        kurt_vals = np.nan_to_num(kurt_vals, nan=0.0)
        feats.append(float(kurt_vals.mean()))
    return np.asarray(feats, dtype=np.float64)


def epoch_temporal_stats(
    epoch: np.ndarray,
    ch_names: list[str],
    scales: list[int] = _DEFAULT_SCALES,
) -> np.ndarray:
    """Compute temporal statistics for all 19 standard channels.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    scales : list[int]
        Non-overlapping window lengths in samples.

    Returns
    -------
    np.ndarray
        Shape ``(228,)``, in ``_STANDARD_19`` order.
        Missing channels → zero-padded (12 zeros per absent channel).
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    n_per_ch = len(scales) * 4
    feats: list[float] = []
    for ch in _STANDARD_19:
        idx = ch_map.get(ch)
        if idx is None:
            feats.extend([0.0] * n_per_ch)
            continue
        feats.extend(temporal_stats_features(epoch[idx], scales=scales).tolist())
    return np.asarray(feats, dtype=np.float64)
