"""Relative band-power feature extraction for a single EEG channel.

Uses the same Welch parameters (boxcar window, nperseg=250) as the Wiener
decomposition and PSD visualisation modules for numerical consistency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Standard EEG frequency bands (Hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def relative_band_power(
    signal: np.ndarray,
    sfreq: float,
    nperseg: int = 250,
    freq_band: tuple[float, float] = (0.5, 40.0),
    freqs_psd: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute relative power in each EEG frequency band.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length. Default 250 (= 2 s at 125 Hz), consistent with
        the rest of the pipeline.
    freq_band : tuple[float, float]
        Total analysis band ``(f_low, f_high)`` in Hz.  Only power within this
        range is counted in the denominator.
    freqs_psd : tuple[np.ndarray, np.ndarray] or None
        Pre-computed ``(freqs, psd)`` from a prior ``welch()`` call on the same
        signal.  When provided, the internal ``welch()`` call is skipped, which
        allows callers that compute multiple features from the same channel to
        avoid redundant FFT work.  If ``None`` (default), ``welch()`` is called
        as normal.

    Returns
    -------
    dict[str, float]
        Keys are band names ("delta", "theta", "alpha", "beta", "gamma");
        values are relative power in [0, 1].  The five values sum to
        approximately 1 (small discrepancies due to trapezoidal integration
        at band boundaries are possible).
    """
    if freqs_psd is not None:
        freqs, psd = freqs_psd
    else:
        freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, window="boxcar")

    total_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask]))

    result: dict[str, float] = {}
    for band_name, (f_lo, f_hi) in BANDS.items():
        # Clamp sub-band to the total analysis band
        f_lo = max(f_lo, freq_band[0])
        f_hi = min(f_hi, freq_band[1])
        if f_lo >= f_hi:
            result[band_name] = 0.0
            continue
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        band_power = float(np.trapezoid(psd[band_mask], freqs[band_mask]))
        result[band_name] = band_power / (total_power + 1e-30)

    return result
