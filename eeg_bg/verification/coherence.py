"""V1 coherence verification: pairwise coherence before/after decomposition.

Two variants:
  * ``run_v1`` — legacy wide-band (single 0.5–40 Hz average).
  * ``run_v1_per_band`` — per-band (delta/theta/alpha/beta/gamma) with
    pair-role classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import coherence as scipy_coherence

from eeg_bg.features.band_power import BANDS
from eeg_bg.verification._pair_roles import classify_pair


def compute_pairwise_coherence(
    data: np.ndarray,          # (n_ch, n_times)
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
) -> np.ndarray:
    n_ch = data.shape[0]
    freqs, _ = scipy_coherence(data[0], data[0], fs=sfreq, nperseg=nperseg)
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    coh_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        coh_matrix[i, i] = 1.0
        for j in range(i + 1, n_ch):
            _, coh = scipy_coherence(data[i], data[j], fs=sfreq, nperseg=nperseg)
            val = float(np.mean(coh[mask]))
            coh_matrix[i, j] = val
            coh_matrix[j, i] = val
    return coh_matrix


def run_v1(results: list, cfg: dict) -> pd.DataFrame:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    # Use freq_resolution_hz to derive nperseg for coherence estimation.
    # The Wiener nperseg may equal n_times (exact single-window filter), which
    # causes scipy.signal.coherence to return 1.0 everywhere (no averaging).
    freq_res = cfg["wiener"].get("freq_resolution_hz", 0.5)
    nperseg = int(sfreq / freq_res)   # e.g. 125 / 0.5 = 250 → 4 segments per epoch
    freq_band = tuple(cfg["wiener"]["freq_band"])

    rows = []
    for result in results:
        ch_names = result.ch_names
        coh_pre = compute_pairwise_coherence(result.raw, sfreq, nperseg, freq_band)
        coh_post = compute_pairwise_coherence(result.specific, sfreq, nperseg, freq_band)
        n_ch = len(ch_names)
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                rows.append({
                    "subject_id": result.subject_id,
                    "epoch_idx": result.epoch_idx,
                    "ch_i": ch_names[i],
                    "ch_j": ch_names[j],
                    "coh_pre": coh_pre[i, j],
                    "coh_post": coh_post[i, j],
                    "reduction": coh_pre[i, j] - coh_post[i, j],
                })
    return pd.DataFrame(rows)


def run_v1_per_band(
    raw_data: np.ndarray,
    specific_data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    freq_resolution_hz: float = 0.5,
    subject_id: str = "",
    recording_id: str = "",
    epoch_idx: int = 0,
    split: str = "",
    label: int = -1,
    channel_groups: list[list[str]] | None = None,
) -> pd.DataFrame:
    """Per-band pairwise coherence comparison for one epoch.

    Unlike ``run_v1`` (which averages across 0.5–40 Hz), this computes
    separate coherence values for delta, theta, alpha, beta, and gamma
    bands.  Each pair is also classified with a ``pair_role`` (targeted_edge,
    processed_untargeted_homotopic, passthrough_control, or other).

    Returns one row per ``(pair, band)``.
    """
    nperseg = int(sfreq / freq_resolution_hz)
    n_ch = len(ch_names)
    rows = []

    for band_name, (f_lo, f_hi) in BANDS.items():
        coh_pre = compute_pairwise_coherence(
            raw_data, sfreq, nperseg, (f_lo, f_hi),
        )
        coh_post = compute_pairwise_coherence(
            specific_data, sfreq, nperseg, (f_lo, f_hi),
        )
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                rows.append({
                    "subject_id":    subject_id,
                    "recording_id":  recording_id,
                    "epoch_idx":     epoch_idx,
                    "split":         split,
                    "label":         label,
                    "ch_i":          ch_names[i],
                    "ch_j":          ch_names[j],
                    "pair_role":     classify_pair(ch_names[i], ch_names[j], channel_groups),
                    "band":          band_name,
                    "coh_pre":       coh_pre[i, j],
                    "coh_post":      coh_post[i, j],
                    "reduction":     coh_pre[i, j] - coh_post[i, j],
                })
    return pd.DataFrame(rows)
