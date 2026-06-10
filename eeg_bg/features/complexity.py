"""Nonlinear complexity features: Sample Entropy and Lempel-Ziv Complexity.

SampEn uses scipy.spatial.cKDTree for O(n log n) template matching.
LZC uses an incremental substring-counting implementation of LZ76.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from eeg_bg.features._constants import _STANDARD_19

COMPLEXITY_NAMES: list[str] = [
    f"{ch}_{stat}"
    for ch in _STANDARD_19
    for stat in ("sample_entropy", "lempel_ziv")
]  # 19 × 2 = 38


def _count_template_pairs(T: np.ndarray, r: float) -> int:
    """Count pairs (i<j) with Chebyshev distance < r via KD-tree."""
    if len(T) < 2:
        return 0
    tree = cKDTree(T)
    return len(tree.query_pairs(r, p=np.inf))


def sample_entropy(
    signal: np.ndarray,
    m: int   = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute Sample Entropy of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(n_times,)``.
    m : int
        Embedding dimension (default 2).
    r_factor : float
        Tolerance as a fraction of the signal's standard deviation (default 0.2).

    Returns
    -------
    float
        SampEn >= 0.  Returns 0.0 for constant signals (std ≈ 0).
    """
    x = signal.astype(np.float64)
    std = float(np.std(x))
    if std < 1e-12:
        return 0.0
    r = r_factor * std

    T_m  = np.lib.stride_tricks.sliding_window_view(x, m)[:-1]
    T_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    B = _count_template_pairs(T_m, r)
    A = _count_template_pairs(T_m1, r)

    if B == 0:
        return 0.0
    return float(-np.log((A + 1e-30) / (B + 1e-30)))


def lempel_ziv_complexity(signal: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity (LZ76) of the binarised signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(n_times,)``.

    Returns
    -------
    float
        LZC >= 0, normalised by ``n / log2(n)``.
    """
    x = signal.astype(np.float64)
    binary = x > np.median(x)
    n = len(binary)

    sub_strings: set[bytes] = set()
    i, k = 0, 1
    while i + k <= n:
        sub = binary[i:i + k].tobytes()
        if sub in sub_strings:
            k += 1
        else:
            sub_strings.add(sub)
            i += k
            k = 1
    c = len(sub_strings)

    norm = n / max(np.log2(n), 1.0) if n > 1 else 1.0
    return float(c / norm)


def complexity_features(
    epoch: np.ndarray,
    ch_names: list[str],
    m: int        = 2,
    r_factor: float = 0.2,
) -> np.ndarray:
    """Compute SampEn and LZC for all 19 standard channels.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    m : int
        SampEn embedding dimension.
    r_factor : float
        SampEn tolerance as fraction of std.

    Returns
    -------
    np.ndarray
        Shape ``(38,)``, layout ``[se_ch1, lzc_ch1, se_ch2, lzc_ch2, ...]``
        in ``_STANDARD_19`` order.  Missing channels → 0.0.
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    feats: list[float] = []
    for ch in _STANDARD_19:
        idx = ch_map.get(ch)
        if idx is None:
            feats.extend([0.0, 0.0])
            continue
        sig = epoch[idx].astype(np.float64)
        feats.append(sample_entropy(sig, m=m, r_factor=r_factor))
        feats.append(lempel_ziv_complexity(sig))
    return np.asarray(feats, dtype=np.float64)
