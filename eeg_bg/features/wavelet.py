"""DWT features per channel — 66 features per channel, 7 feature groups.

Uses PyWavelets (pywt) db4 wavelet at 6 levels by default.
Approximate band-to-level mapping at 125 Hz, level 6:
    level 1 → 32–62 Hz  (γ)      cD1
    level 2 → 16–32 Hz  (β)      cD2
    level 3 →  8–16 Hz  (α/β)    cD3
    level 4 →  4– 8 Hz  (θ)      cD4
    level 5 →  2– 4 Hz  (δ-hi)   cD5
    level 6 →  0.5–2 Hz (δ)      cD6

Feature groups (66 per channel):
    1. Detail energy         dwt_l{l}_energy         l=1..6   → 6
    2. Detail entropy        dwt_l{l}_entropy         l=1..6   → 6
    3. Detail coeff stats    dwt_l{l}_coef_mean/std/maxabs      → 18
    4. Approx coeff stats    dwt_approx_mean/std/maxabs          → 3
    5. Modulus maxima        dwt_l{l}_mmx_count/mean  l=1..6  → 12
    6. Scale energy ratio    dwt_l{l}_engratio        l=1..6   → 6
    7. Reconstructed stats   dwt_band_{b}_mean/std/power b=delta..gamma → 15
    Total: 6+6+18+3+12+6+15 = 66
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
    # Group 2: detail entropy (6)
    + [f"dwt_l{l}_entropy" for l in range(1, 7)]
    # Group 3: detail coeff stats (18)
    + [f"dwt_l{l}_coef_{s}" for l in range(1, 7) for s in ("mean", "std", "maxabs")]
    # Group 4: approx coeff stats (3)
    + ["dwt_approx_mean", "dwt_approx_std", "dwt_approx_maxabs"]
    # Group 5: modulus maxima (12)
    + [f"dwt_l{l}_mmx_{s}" for l in range(1, 7) for s in ("count", "mean")]
    # Group 6: scale energy ratio (6)
    + [f"dwt_l{l}_engratio" for l in range(1, 7)]
    # Group 7: reconstructed band stats (15)
    + [f"dwt_band_{b}_{s}" for b in _BANDS_ORDER for s in ("mean", "std", "power")]
)  # length 66

WAVELET_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _WAVELET_SUFFIXES
]  # 19 × 66 = 1254


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
        Shape ``(66,)``, dtype float64.
    """
    sig = signal.astype(np.float64)
    n = len(sig)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs[0] = cA_level, coeffs[level+1-l] = cD_l

    # --- groups 1 & 2: energy and entropy per detail level ---
    energies: list[float] = []
    entropies: list[float] = []
    detail_coeffs: list[np.ndarray] = []
    for l in range(1, level + 1):
        c = coeffs[level + 1 - l]
        detail_coeffs.append(c)
        sq = c * c
        raw_energy = float(sq.sum())
        if raw_energy == 0.0:
            energies.append(0.0)
            entropies.append(0.0)
        else:
            sum_sq = raw_energy
            energies.append(float(raw_energy / n))
            p = sq / sum_sq
            entropies.append(float(-np.sum(p * np.log(p + 1e-30))))

    # --- group 3: detail coeff stats ---
    def _coef_stats(c: np.ndarray) -> tuple[float, float, float]:
        if len(c) == 0:
            return 0.0, 0.0, 0.0
        return float(c.mean()), float(c.std()), float(np.abs(c).max())

    detail_stats: list[float] = []
    for c in detail_coeffs:
        m, s, mx = _coef_stats(c)
        detail_stats.extend([m, s, mx])

    # --- group 4: approx coeff stats ---
    ca = coeffs[0]
    approx_stats = list(_coef_stats(ca))

    # --- group 5: modulus maxima ---
    mmx_feats: list[float] = []
    for c in detail_coeffs:
        if len(c) < 3:
            mmx_feats.extend([0.0, 0.0])
            continue
        ac = np.abs(c)
        peaks = np.where((ac[1:-1] > ac[:-2]) & (ac[1:-1] > ac[2:]))[0] + 1
        if len(peaks) == 0:
            mmx_feats.extend([0.0, 0.0])
        else:
            mmx_feats.extend([float(len(peaks)), float(ac[peaks].mean())])

    # --- group 6: scale energy ratio ---
    total_energy = sum(energies) + 1e-30
    engratio: list[float] = [e / total_energy for e in energies]

    # --- group 7: reconstructed band stats ---
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
        + entropies    # group 2 (6)
        + detail_stats # group 3 (18)
        + approx_stats # group 4 (3)
        + mmx_feats    # group 5 (12)
        + engratio     # group 6 (6)
        + band_stats   # group 7 (15)
    )
    return np.asarray(feats, dtype=np.float64)
