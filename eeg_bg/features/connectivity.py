"""Pairwise functional connectivity: coherence and Phase-Locking Value (PLV).

Computes magnitude-squared coherence and per-band PLV for all 171 unique
electrode pairs (C(19,2)) across the 5 standard EEG frequency bands.

Efficiency: per-band bandpass + Hilbert is computed once per channel
(95 ops) and reused for all 171 pairs in each band.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import coherence as sp_coherence, hilbert, butter, sosfiltfilt

from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.band_power import BANDS

ALL_PAIRS: list[tuple[str, str]] = list(combinations(_STANDARD_19, 2))  # 171 pairs

CONNECTIVITY_NAMES: list[str] = [
    f"{metric}_{ch1}_{ch2}_{band}"
    for ch1, ch2 in ALL_PAIRS
    for band in BANDS
    for metric in ("coh", "plv")
]  # 171 × 5 × 2 = 1710


def _bandpass_sos(f_lo: float, f_hi: float, sfreq: float) -> np.ndarray:
    """Return SOS coefficients for a 4th-order Butterworth bandpass filter."""
    return butter(4, [f_lo, f_hi], btype="bandpass", fs=sfreq, output="sos")


def connectivity_features(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float = 125.0,
    nperseg: int = 250,
) -> np.ndarray:
    """Compute pairwise coherence and PLV for all standard channel pairs.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch window for coherence estimation (pipeline-consistent default 250).

    Returns
    -------
    np.ndarray
        Shape ``(1710,)``, dtype float64.
        Layout: for each pair in ``ALL_PAIRS``, for each band in ``BANDS``,
        ``[coherence, PLV]``.  Absent-channel pairs → 10 zeros.
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    present = {ch for ch in _STANDARD_19 if ch in ch_map}

    # ── Precompute raw signals and per-band phases for present channels ──────
    raw: dict[str, np.ndarray] = {
        ch: epoch[ch_map[ch]].astype(np.float64) for ch in present
    }
    phases: dict[tuple[str, str], np.ndarray] = {}  # {(ch, band): instantaneous phase}
    for ch in present:
        for band_name, (f_lo, f_hi) in BANDS.items():
            sos = _bandpass_sos(f_lo, f_hi, sfreq)
            filtered = sosfiltfilt(sos, raw[ch])
            phases[(ch, band_name)] = np.angle(hilbert(filtered))

    # ── Compute features for all pairs ───────────────────────────────────────
    feats: list[float] = []
    for ch1, ch2 in ALL_PAIRS:
        if ch1 not in present or ch2 not in present:
            feats.extend([0.0] * (len(BANDS) * 2))
            continue

        # Coherence (one call per pair, covers all bands)
        freqs, coh_xy = sp_coherence(raw[ch1], raw[ch2], fs=sfreq, nperseg=nperseg)

        for band_name, (f_lo, f_hi) in BANDS.items():
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            coh_val = float(coh_xy[band_mask].mean()) if band_mask.any() else 0.0

            # PLV from pre-computed per-band phases
            delta_phi = phases[(ch1, band_name)] - phases[(ch2, band_name)]
            plv_val = float(np.abs(np.mean(np.exp(1j * delta_phi))))

            feats.extend([coh_val, plv_val])

    return np.asarray(feats, dtype=np.float64)
