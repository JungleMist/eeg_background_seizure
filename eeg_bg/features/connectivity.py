"""Pairwise functional connectivity: coherence and Phase-Locking Value (PLV).

Computes magnitude-squared coherence and per-band PLV for all 171 unique
electrode pairs (C(19,2)) across the 5 standard EEG frequency bands.

Efficiency:
  - Coherence: one vectorised Welch STFT over all 19 channels, then all 171
    cross-spectral densities derived from that single FFT result — replaces
    171 individual scipy.signal.coherence calls.
  - PLV: per-band bandpass + Hilbert computed once per channel (95 ops) and
    reused across all 171 pairs in that band.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.signal.windows import hann

from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.band_power import BANDS

ALL_PAIRS: list[tuple[str, str]] = list(combinations(_STANDARD_19, 2))  # 171 pairs

CONNECTIVITY_NAMES: list[str] = [
    f"{metric}_{ch1}_{ch2}_{band}"
    for ch1, ch2 in ALL_PAIRS
    for band in BANDS
    for metric in ("coh", "plv")
]  # 171 × 5 × 2 = 1710

# Precompute pair index arrays once at import time for vectorised indexing.
_PAIR_I1: list[int] = []
_PAIR_I2: list[int] = []
_CH_TO_STD_IDX: dict[str, int] = {ch: i for i, ch in enumerate(_STANDARD_19)}
for _ch1, _ch2 in ALL_PAIRS:
    _PAIR_I1.append(_CH_TO_STD_IDX[_ch1])
    _PAIR_I2.append(_CH_TO_STD_IDX[_ch2])
_PAIR_I1 = np.asarray(_PAIR_I1)  # (171,)
_PAIR_I2 = np.asarray(_PAIR_I2)  # (171,)


def _bandpass_sos(f_lo: float, f_hi: float, sfreq: float) -> np.ndarray:
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
        Welch window length in samples (pipeline-consistent default 250).

    Returns
    -------
    np.ndarray
        Shape ``(1710,)``, dtype float64.
        Layout: for each pair in ``ALL_PAIRS``, for each band in ``BANDS``,
        ``[coherence, PLV]``.  Absent-channel pairs → 10 zeros.
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    n_std = len(_STANDARD_19)
    n_times = epoch.shape[1]

    # ── Build (n_std, n_times) array; zero-fill absent channels ──────────────
    epoch_std = np.zeros((n_std, n_times), dtype=np.float64)
    present = np.zeros(n_std, dtype=bool)
    for si, ch in enumerate(_STANDARD_19):
        if ch in ch_map:
            epoch_std[si] = epoch[ch_map[ch]].astype(np.float64)
            present[si] = True

    # ── Vectorised Welch coherence ────────────────────────────────────────────
    # Build overlapping Hann-windowed segments: (n_std, n_seg, nperseg)
    step = nperseg // 2
    win = hann(nperseg, sym=False)
    starts = np.arange(0, n_times - nperseg + 1, step)
    seg_idx = starts[:, None] + np.arange(nperseg)[None, :]  # (n_seg, nperseg)
    segs = epoch_std[:, seg_idx] * win[None, None, :]        # (n_std, n_seg, nperseg)

    # FFT → (n_std, n_seg, n_freq)
    F = np.fft.rfft(segs, axis=2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)

    # Auto-power per channel averaged over segments: (n_std, n_freq)
    Pxx = (F * F.conj()).real.mean(axis=1)

    # All pairwise cross-power: (171, n_freq)
    Cxy = (F[_PAIR_I1] * F[_PAIR_I2].conj()).mean(axis=1)
    P1  = Pxx[_PAIR_I1]   # (171, n_freq)
    P2  = Pxx[_PAIR_I2]   # (171, n_freq)
    coh_all = (np.abs(Cxy) ** 2) / (P1 * P2 + 1e-30)  # (171, n_freq)

    # ── Per-band Hilbert phases for PLV ───────────────────────────────────────
    phases: dict[str, np.ndarray] = {}  # band_name → (n_std, n_times)
    for band_name, (f_lo, f_hi) in BANDS.items():
        sos = _bandpass_sos(f_lo, f_hi, sfreq)
        filtered = sosfiltfilt(sos, epoch_std, axis=1)  # (n_std, n_times)
        phases[band_name] = np.angle(hilbert(filtered, axis=1))  # (n_std, n_times)

    # ── Pair absent mask ──────────────────────────────────────────────────────
    pair_present = present[_PAIR_I1] & present[_PAIR_I2]  # (171,)

    # ── Compute per-band coh and plv arrays ──────────────────────────────────
    # coh_bands[band] → (171,), plv_bands[band] → (171,)
    band_names = list(BANDS.keys())
    coh_bands: dict[str, np.ndarray] = {}
    plv_bands: dict[str, np.ndarray] = {}

    for band_name, (f_lo, f_hi) in BANDS.items():
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        coh_b = coh_all[:, band_mask].mean(axis=1) if band_mask.any() \
                else np.zeros(len(ALL_PAIRS))
        ph1 = phases[band_name][_PAIR_I1]  # (171, n_times)
        ph2 = phases[band_name][_PAIR_I2]  # (171, n_times)
        plv_b = np.abs(np.mean(np.exp(1j * (ph1 - ph2)), axis=1))
        coh_bands[band_name] = np.where(pair_present, coh_b, 0.0)
        plv_bands[band_name] = np.where(pair_present, plv_b, 0.0)

    # ── Assemble in CONNECTIVITY_NAMES order: pair → band → [coh, plv] ───────
    feats = np.empty(len(ALL_PAIRS) * len(BANDS) * 2, dtype=np.float64)
    out_idx = 0
    for pi in range(len(ALL_PAIRS)):
        for band_name in band_names:
            feats[out_idx]     = coh_bands[band_name][pi]
            feats[out_idx + 1] = plv_bands[band_name][pi]
            out_idx += 2

    return feats
