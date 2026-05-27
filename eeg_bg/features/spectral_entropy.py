"""Spectral entropy (Shannon entropy of normalised PSD) for a single channel.

Uses the same Welch parameters as the rest of the pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def spectral_entropy(
    signal: np.ndarray,
    sfreq: float,
    nperseg: int = 250,
    freq_band: tuple[float, float] = (0.5, 40.0),
    freqs_psd: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Shannon entropy of the normalised PSD within *freq_band*.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length (default 250, consistent with pipeline).
    freq_band : tuple[float, float]
        Analysis band in Hz.
    freqs_psd : tuple[np.ndarray, np.ndarray] or None
        Pre-computed ``(freqs, psd)`` from a prior ``welch()`` call on the same
        signal.  When provided, the internal ``welch()`` call is skipped.
        If ``None`` (default), ``welch()`` is called as normal.

    Returns
    -------
    float
        Spectral entropy ≥ 0.  A flat (white-noise) spectrum maximises the
        entropy; a pure sinusoid approaches zero.
    """
    if freqs_psd is not None:
        freqs, psd = freqs_psd
    else:
        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, window="boxcar")

    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    psd_band = psd[mask]

    # Normalise to a probability distribution
    total = psd_band.sum() + 1e-30
    p = psd_band / total

    entropy = float(-np.sum(p * np.log(p + 1e-30)))
    return entropy
