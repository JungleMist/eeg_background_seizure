"""Lagged (zero-lag-free) connectivity metrics for verification.

Zero-lag coherence and PLV (as in ``eeg_bg.features.connectivity``) are
confounded by volume conduction — a strong but artefactual zero-phase
coupling can produce high coherence / PLV even when the underlying neural
sources are independent.  The metrics here explicitly remove or down-weight
zero-lag contributions so that changes in genuine time-lagged functional
coupling can be tracked across the Wiener decomposition.

Metrics
-------
* **Imaginary coherence** :math:`|\\Im(C_{xy})| / \\sqrt{P_x P_y + \\varepsilon}`.
  Zero when the cross-spectrum is purely real (zero-lag / volume conduction).
* **Weighted Phase-Lag Index (wPLI)** :math:`|\\langle\\Im(C_{xy})\\rangle|
  / \\langle|\\Im(C_{xy})|\\rangle`.  Down-weights near-zero-phase differences
  by their magnitude in the imaginary plane.

Both are computed via a single vectorised Welch STFT, following the same
pattern as ``eeg_bg.features.connectivity`` for efficiency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import hann

from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.band_power import BANDS


def _welch_stft(
    epoch_std: np.ndarray,
    sfreq: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Welch STFT for all 19 standard channels.

    Returns
    -------
    freqs : (n_freqs,)
    Pxx : (19, n_freqs) — auto-power
    Cxy : (171, n_freqs) — segment-averaged cross-power
    Cxy_segments : (171, n_segments, n_freqs) — per-segment cross-power
    P1, P2 : (171, n_freqs) — auto-power of pair members
    """
    n_std, n_times = epoch_std.shape
    step = nperseg // 2
    win = hann(nperseg, sym=False)
    starts = np.arange(0, n_times - nperseg + 1, step)
    seg_idx = starts[:, None] + np.arange(nperseg)[None, :]
    segs = epoch_std[:, seg_idx]
    segs = segs - segs.mean(axis=2, keepdims=True)
    segs = segs * win[None, None, :]

    F = np.fft.rfft(segs, axis=2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)

    Pxx = (F * F.conj()).real.mean(axis=1)

    # All C(19,2) pair indices pre-computed
    n_pairs = n_std * (n_std - 1) // 2  # 171
    i1 = np.zeros(n_pairs, dtype=int)
    i2 = np.zeros(n_pairs, dtype=int)
    p = 0
    for a in range(n_std):
        for b in range(a + 1, n_std):
            i1[p] = a
            i2[p] = b
            p += 1

    Cxy_segments = F[i1] * F[i2].conj()
    Cxy = Cxy_segments.mean(axis=1)
    P1 = Pxx[i1]
    P2 = Pxx[i2]

    return freqs, Pxx, Cxy, P1, P2, Cxy_segments, i1, i2


def _pair_idx(ch_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices into the 171-pair arrays for all present channel pairs."""
    n_std = len(_STANDARD_19)
    n_pairs = n_std * (n_std - 1) // 2

    # Build index array mapping pair (i,j) → flat 171-index
    flat_idx = np.full((n_std, n_std), -1, dtype=int)
    p = 0
    for a in range(n_std):
        for b in range(a + 1, n_std):
            flat_idx[a, b] = p
            flat_idx[b, a] = p
            p += 1

    present = np.array([ch in ch_names for ch in _STANDARD_19], dtype=bool)
    # Mark pairs where both channels are present
    pair_present = np.zeros(n_pairs, dtype=bool)
    for a in range(n_std):
        for b in range(a + 1, n_std):
            if present[a] and present[b]:
                pair_present[flat_idx[a, b]] = True

    return flat_idx, present, pair_present


def imaginary_coherence(
    Cxy: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """Imaginary coherence for all C(19,2) pairs across all frequencies.

    .. math:: |Im(C_{xy})| / sqrt(P_x * P_y + epsilon)

    Parameters
    ----------
    Cxy : (171, n_freqs), complex
    P1, P2 : (171, n_freqs), real

    Returns
    -------
    (171, n_freqs), real
    """
    eps = 1e-30
    return np.abs(Cxy.imag) / np.sqrt(P1 * P2 + eps)


def wpli(Cxy: np.ndarray) -> np.ndarray:
    """Weighted Phase-Lag Index (wPLI) for all pairs, per frequency bin.

    .. math:: |mean(Im(C_{xy}))| / mean(|Im(C_{xy})|)

    Parameters
    ----------
    Cxy : (171, n_freqs), complex — cross-spectrum averaged over segments.

    Returns
    -------
    (171,) per-pair wPLI (single value across all frequencies).
    """
    if Cxy.ndim == 3:
        # Correct wPLI: observations are Welch segments (and, when used by
        # the caller, frequency bins within the selected band), not the
        # already-averaged cross-spectrum.
        imag = Cxy.imag.reshape(Cxy.shape[0], -1)
    elif Cxy.ndim == 2:
        # Backward-compatible fallback for callers that provide one value per
        # frequency bin. This is not segment-based wPLI.
        imag = Cxy.imag
    else:
        raise ValueError("Cxy must have shape (pairs, freqs) or (pairs, segments, freqs)")
    mean_imag = imag.mean(axis=1)
    mean_abs_imag = np.abs(imag).mean(axis=1)
    eps = 1e-30
    return np.abs(mean_imag) / (mean_abs_imag + eps)


def compute_connectivity_metrics(
    raw_epoch: np.ndarray,
    specific_epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    nperseg: int,
) -> dict:
    """Compute pre/post connectivity metrics for all pairs across all bands.

    Parameters
    ----------
    raw_epoch, specific_epoch : (n_ch, n_times)
    ch_names : list[str]
    sfreq : float
    nperseg : int

    Returns
    -------
    dict with keys for each ``(pair_key, band, metric, stage)``, where
    ``stage`` is ``"pre"`` (raw) or ``"post"`` (specific), and ``metric`` is
    one of ``"coh"``, ``"plv"``, ``"icoh"``, ``"wpli"``.
    Also includes ``"freqs"``, ``"i1"``, ``"i2"`` for downstream use.
    """
    # Build standard-ordered (19, n_times) arrays
    n_std = len(_STANDARD_19)
    n_times = raw_epoch.shape[1]
    ch_map = {name: i for i, name in enumerate(ch_names)}

    def to_std(epoch):
        out = np.zeros((n_std, n_times), dtype=np.float64)
        for si, ch in enumerate(_STANDARD_19):
            if ch in ch_map:
                out[si] = epoch[ch_map[ch]]
        return out

    raw_std = to_std(raw_epoch)
    spc_std = to_std(specific_epoch)

    freqs, Pxx_r, Cxy_r, P1_r, P2_r, Cxy_seg_r, i1, i2 = _welch_stft(raw_std, sfreq, nperseg)

    # Build pair present mask
    _, present, pair_present = _pair_idx(ch_names)

    result: dict = {"freqs": freqs, "i1": i1, "i2": i2, "pair_present": pair_present}

    for label, epoch_arr in [("pre", raw_std), ("post", spc_std)]:
        _, _, Cxy, P1, P2, Cxy_segments, _, _ = _welch_stft(epoch_arr, sfreq, nperseg)

        for band_name, (f_lo, f_hi) in BANDS.items():
            bm = (freqs >= f_lo) & (freqs <= f_hi)
            if not bm.any():
                for metric in ("coh", "plv", "icoh", "wpli"):
                    key = f"{band_name}_{metric}_{label}"
                    result[key] = np.zeros(len(i1))
                continue

            # Magnitude-squared coherence (band-averaged)
            coh_val = (np.abs(Cxy[:, bm]) ** 2 / (P1[:, bm] * P2[:, bm] + 1e-30)).mean(axis=1)
            result[f"{band_name}_coh_{label}"] = np.where(pair_present, coh_val, np.nan)

            # Imaginary coherence (band-averaged)
            icoh_val = imaginary_coherence(Cxy[:, bm], P1[:, bm], P2[:, bm]).mean(axis=1)
            result[f"{band_name}_icoh_{label}"] = np.where(pair_present, icoh_val, np.nan)

            # wPLI (computed across all freqs; then band-averaged isn't meaningful
            # for wPLI — compute per-band by restricting imag to band)
            wpli_val = wpli(Cxy_segments[:, :, bm])
            result[f"{band_name}_wpli_{label}"] = np.where(pair_present, wpli_val, np.nan)

            # PLV requires Hilbert (expensive) — single bandpass + angle per channel
            from scipy.signal import butter, sosfiltfilt, hilbert
            sos = butter(4, [f_lo, f_hi], btype="bandpass", fs=sfreq, output="sos")
            filtered = sosfiltfilt(sos, epoch_arr, axis=1)
            phases = np.angle(hilbert(filtered, axis=1))
            ph1 = phases[i1]
            ph2 = phases[i2]
            plv_val = np.abs(np.mean(np.exp(1j * (ph1 - ph2)), axis=1))
            result[f"{band_name}_plv_{label}"] = np.where(pair_present, plv_val, np.nan)

    return result
