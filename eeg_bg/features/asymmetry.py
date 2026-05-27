"""Hemispheric asymmetry features for EEG classification.

Computes normalised left–right power asymmetry for 8 symmetric electrode
pairs across 5 standard EEG frequency bands, yielding a 40-dimensional
feature vector appended after the per-channel statistics.

Formula (per pair, per band)::

    asym = (P_left - P_right) / (P_left + P_right + ε)

Values lie in (-1, +1).  Positive → left dominates; negative → right
dominates.  Zero-padded when either electrode in a pair is absent.
"""
from __future__ import annotations

import numpy as np

from eeg_bg.features.band_power import relative_band_power, BANDS

# 8 anatomically symmetric electrode pairs (left, right).
# Order is fixed; reordering invalidates saved SHAP .npy arrays.
SYMMETRIC_PAIRS: list[tuple[str, str]] = [
    ("FP1", "FP2"),
    ("F3",  "F4"),
    ("F7",  "F8"),
    ("C3",  "C4"),
    ("T3",  "T4"),
    ("T5",  "T6"),
    ("P3",  "P4"),
    ("O1",  "O2"),
]

# 8 pairs × 5 bands = 40 feature names, prefixed with "asym_" so that
# aggregate_shap_by_channel does not conflate them with per-channel features.
ASYMMETRY_NAMES: list[str] = [
    f"asym_{left}_{right}_{band}"
    for left, right in SYMMETRIC_PAIRS
    for band in BANDS
]  # length == 40


def hemispheric_asymmetry(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    nperseg: int = 250,
    freq_band: tuple[float, float] = (0.5, 40.0),
    psd_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Compute a 40-dimensional hemispheric asymmetry vector.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names corresponding to rows of *epoch*.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch window length (default 250, pipeline-consistent).
    freq_band : tuple[float, float]
        Analysis band in Hz.
    psd_cache : dict[str, tuple[np.ndarray, np.ndarray]] or None
        Pre-computed ``{ch_name: (freqs, psd)}`` mapping.  When provided,
        PSDs for electrode pairs are looked up rather than recomputed, avoiding
        redundant ``welch()`` calls when this function is called as part of a
        larger feature extraction loop.  If ``None`` (default), PSDs are
        computed on demand.

    Returns
    -------
    np.ndarray
        Shape ``(40,)``, dtype float64.  Feature order matches
        :data:`ASYMMETRY_NAMES`.  Any pair where one or both electrodes are
        absent yields five zeros (one per band).
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    features: list[float] = []
    for left, right in SYMMETRIC_PAIRS:
        left_idx  = ch_map.get(left)
        right_idx = ch_map.get(right)
        if left_idx is None or right_idx is None:
            # One or both electrodes absent — zero-pad all bands
            features.extend([0.0] * len(BANDS))
            continue

        bp_left  = relative_band_power(
            epoch[left_idx].astype(np.float64),
            sfreq=sfreq, nperseg=nperseg, freq_band=freq_band,
            freqs_psd=psd_cache.get(left)  if psd_cache is not None else None,
        )
        bp_right = relative_band_power(
            epoch[right_idx].astype(np.float64),
            sfreq=sfreq, nperseg=nperseg, freq_band=freq_band,
            freqs_psd=psd_cache.get(right) if psd_cache is not None else None,
        )
        for band in BANDS:
            p_left  = bp_left[band]
            p_right = bp_right[band]
            denom   = p_left + p_right + 1e-12
            features.append((p_left - p_right) / denom)

    return np.asarray(features, dtype=np.float64)
