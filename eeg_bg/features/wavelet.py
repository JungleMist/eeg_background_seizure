"""DWT energy and entropy per channel per decomposition level.

Uses PyWavelets (pywt) db4 wavelet at 6 levels by default.
Approximate band-to-level mapping at 125 Hz, level 6:
    level 1 → 32–62 Hz (γ)    cD1
    level 2 → 16–32 Hz (β)    cD2
    level 3 →  8–16 Hz (α/β)  cD3
    level 4 →  4– 8 Hz (θ)    cD4
    level 5 →  2– 4 Hz (δ-hi) cD5
    level 6 →  0.5–2 Hz (δ)   cD6
"""
from __future__ import annotations

import numpy as np
import pywt

from eeg_bg.features._constants import _STANDARD_19

_DEFAULT_WAVELET = "db4"
_DEFAULT_LEVELS = 6

# Per-signal output: [energy_l1, entropy_l1, energy_l2, entropy_l2, ..., energy_l6, entropy_l6]
# 2 stats × 6 levels = 12 features per channel
_WAVELET_SUFFIXES: list[str] = [
    f"dwt_l{l}_{stat}"
    for l in range(1, _DEFAULT_LEVELS + 1)
    for stat in ("energy", "entropy")
]  # length 12

WAVELET_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _WAVELET_SUFFIXES
]  # 19 × 12 = 228


def wavelet_features(
    signal: np.ndarray,
    wavelet: str = _DEFAULT_WAVELET,
    level: int = _DEFAULT_LEVELS,
) -> np.ndarray:
    """Compute DWT energy and entropy for one signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    wavelet : str
        PyWavelets wavelet name (default ``'db4'``).
    level : int
        Decomposition depth (default 6).

    Returns
    -------
    np.ndarray
        Shape ``(2 * level,)``, dtype float64.
        Layout: ``[energy_l1, entropy_l1, energy_l2, entropy_l2, ...,
        energy_l{level}, entropy_l{level}]``.
    """
    coeffs = pywt.wavedec(signal.astype(np.float64), wavelet, level=level)
    # coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    # cD_l lives at coeffs[level + 1 - l]
    feats: list[float] = []
    for l in range(1, level + 1):
        c = coeffs[level + 1 - l].astype(np.float64)
        sq = c * c
        raw_energy = float(sq.sum())
        if raw_energy == 0.0:
            feats.extend([0.0, 0.0])
            continue
        tot = raw_energy + 1e-30
        energy = float(tot / (len(c) + 1e-30))
        p = sq / tot
        entropy = float(-np.sum(p * np.log(p + 1e-30)))
        feats.extend([energy, entropy])
    return np.asarray(feats, dtype=np.float64)
