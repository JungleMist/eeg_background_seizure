"""
Ablation: scalar Wiener mode. Replaces frequency-dependent h(f) with a
single complex scalar per reference channel (average over frequency band).
Equivalent to the EKG-style fixed compensation described in the proposal.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import coherence as scipy_coherence, welch

from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
)


def _scalar_from_filter(h: np.ndarray, freq_mask: np.ndarray) -> np.ndarray:
    """Average h(f) over the target frequency band → scalar per ref channel."""
    return h[:, freq_mask].mean(axis=1, keepdims=True)  # (n_ref, 1)


def decompose_epoch(
    epoch: np.ndarray,
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
) -> WienerResult:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    coh_threshold = cfg["wiener"]["coherence_threshold"]
    channel_groups = cfg["channels"]["channel_groups"]
    freq_band = cfg["wiener"]["freq_band"]
    n_times = epoch.shape[1]

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg, window='boxcar')
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

    for pair in channel_groups:
        try:
            indices = [ch_names.index(ch) for ch in pair]
        except ValueError:
            skipped.append("-".join(pair))
            continue

        group_data = epoch[indices]
        max_pairwise_coh = 0.0
        for i, j in combinations(range(len(pair)), 2):
            _, c = scipy_coherence(group_data[i], group_data[j],
                                   fs=sfreq, nperseg=nperseg)
            max_pairwise_coh = max(max_pairwise_coh, np.max(c[freq_mask]))
        if max_pairwise_coh < coh_threshold:
            skipped.append("-".join(pair))
            continue

        _, S = estimate_cross_psd(group_data, sfreq, nperseg)
        pair_key = "-".join(pair)
        filters[pair_key] = {}

        for local_idx, (ch, global_idx) in enumerate(zip(pair, indices)):
            h_freq = compute_wiener_filter(S, target_idx=local_idx)
            h_scalar = _scalar_from_filter(h_freq, freq_mask)

            ref_indices = [i for i in range(len(pair)) if i != local_idx]
            ref_data = group_data[ref_indices]  # (n_ref, n_times)
            coherent_signal = np.sum(h_scalar.real * ref_data, axis=0)
            specific[global_idx] = epoch[global_idx] - coherent_signal
            coherent[global_idx] = coherent_signal
            filters[pair_key][ch] = h_scalar

    return WienerResult(
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        raw=epoch,
        specific=specific,
        coherent=coherent,
        filters=filters,
        freqs=freqs,
        ch_names=ch_names,
        skipped_pairs=skipped,
    )
