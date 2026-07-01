"""DWT features per channel — 27 features per channel, 3 feature groups.

Uses PyWavelets (pywt) db4 wavelet at 6 levels by default.
Approximate band-to-level mapping at 125 Hz, level 6:
    level 1 → 32–62 Hz  (γ)      cD1
    level 2 → 16–32 Hz  (β)      cD2
    level 3 →  8–16 Hz  (α/β)    cD3
    level 4 →  4– 8 Hz  (θ)      cD4
    level 5 →  2– 4 Hz  (δ-hi)   cD5
    level 6 →  0.5–2 Hz (δ)      cD6

Feature groups (27 per channel). Of the original 7 groups (66/channel), 4
were cut as redundant with existing Welch-based per-channel features
(band power, Hjorth, spectral entropy in extraction.py) and absent from
every SHAP top-20 across raw/ica/wiener conditions: detail entropy
(redundant with spectral_entropy), detail coeff mean/std/maxabs (redundant
with Hjorth activity/mobility), approx coeff stats (redundant with delta
power), and scale energy ratio (a normalized derivative of detail energy,
largely duplicate information). Modulus maxima kept only its mean variant
(count never appeared in top-20).
    1. Detail energy         dwt_l{l}_energy    l=1..6   → 6
    2. Modulus maxima mean   dwt_l{l}_mmx_mean  l=1..6   → 6
    3. Reconstructed stats   dwt_band_{b}_mean/std/power b=delta..gamma → 15
    Total: 6+6+15 = 27
"""
from __future__ import annotations

import numpy as np
import pywt

from eeg_bg.features._constants import _STANDARD_19

_DEFAULT_WAVELET = "db4"
_DEFAULT_LEVELS = 6

# Band-to-level mapping for reconstruction (125 Hz, db4, level=6)
_BAND_LEVELS: dict[str, list[int]] = {
    "delta": [5, 6],
    "theta": [4],
    "alpha": [3],
    "beta":  [2],
    "gamma": [1],
}
_BANDS_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]

_WAVELET_SUFFIXES: list[str] = (
    # Group 1: detail energy (6)
    [f"dwt_l{l}_energy" for l in range(1, 7)]
    # Group 2: modulus maxima mean (6)
    + [f"dwt_l{l}_mmx_mean" for l in range(1, 7)]
    # Group 3: reconstructed band stats (15)
    + [f"dwt_band_{b}_{s}" for b in _BANDS_ORDER for s in ("mean", "std", "power")]
)  # length 27

WAVELET_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _WAVELET_SUFFIXES
]  # 19 × 27 = 513


def wavelet_features(
    signal: np.ndarray,
    wavelet: str = _DEFAULT_WAVELET,
    level: int = _DEFAULT_LEVELS,
) -> np.ndarray:
    """Compute 66 DWT features for one 1-D signal.

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
        Shape ``(27,)``, dtype float64.
    """
    sig = signal.astype(np.float64)
    n = len(sig)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs[0] = cA_level, coeffs[level+1-l] = cD_l

    # --- group 1: detail energy per level ---
    energies: list[float] = []
    detail_coeffs: list[np.ndarray] = []
    for l in range(1, level + 1):
        c = coeffs[level + 1 - l]
        detail_coeffs.append(c)
        raw_energy = float((c * c).sum())
        energies.append(float(raw_energy / n))

    # --- group 2: modulus maxima mean ---
    mmx_feats: list[float] = []
    for c in detail_coeffs:
        if len(c) < 3:
            mmx_feats.append(0.0)
            continue
        ac = np.abs(c)
        peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0] + 1
        if len(peaks) == 0:
            mmx_feats.append(0.0)
        else:
            mmx_feats.append(float(ac[peaks].mean()))

    # --- group 3: reconstructed band stats ---
    band_stats: list[float] = []
    for band in _BANDS_ORDER:
        lvls = _BAND_LEVELS[band]
        # build zeroed coefficient list, fill chosen detail levels
        rec_coeffs = [np.zeros_like(c) for c in coeffs]
        for l in lvls:
            idx = level + 1 - l
            rec_coeffs[idx] = coeffs[idx].copy()
        rec = pywt.waverec(rec_coeffs, wavelet)
        if len(rec) < n:
            rec = np.pad(rec, (0, n - len(rec)))
        else:
            rec = rec[:n]
        band_stats.append(float(rec.mean()))
        band_stats.append(float(rec.std()))
        band_stats.append(float(np.dot(rec, rec) / (len(rec) + 1e-30)))

    feats = (
        energies       # group 1 (6)
        + mmx_feats    # group 2 (6)
        + band_stats   # group 3 (15)
    )
    return np.asarray(feats, dtype=np.float64)
