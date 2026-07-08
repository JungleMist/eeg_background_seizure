"""
Zero-referenced phase-gated complex Wiener mode.

This mode first estimates the standard complex per-frequency Wiener filter,
then keeps only reference/frequency coefficients whose target-reference
cross-spectrum phase is close enough to 0.  Phase difference pi is treated as
the maximum phase difference, so anti-phase coherent components are preserved
unless ``wiener.phase_gate_threshold_rad`` is set to pi.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import coherence as scipy_coherence, welch

from eeg_bg.decomposition.phase_gate import (
    phase_gate_threshold_from_config,
    phase_gate_weights,
)
from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
    apply_wiener_filter,
)


def compute_phasegated_filter(
    S: np.ndarray,
    target_idx: int,
    phase_gate_threshold_rad: float,
    reg_factor: float = 1e-4,
) -> np.ndarray:
    """Return a complex Wiener filter after zero-referenced phase gating."""
    h = compute_wiener_filter(S, target_idx=target_idx, reg_factor=reg_factor)
    weights = phase_gate_weights(S, target_idx, phase_gate_threshold_rad)
    return h * weights


def decompose_epoch(
    epoch: np.ndarray,       # (n_channels, n_times)
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
) -> WienerResult:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    coh_threshold = cfg["wiener"]["coherence_threshold"]
    mag_threshold = float(cfg["wiener"].get("filter_magnitude_threshold", 50.0))
    phase_gate_threshold = phase_gate_threshold_from_config(cfg)
    channel_groups = cfg["channels"]["channel_groups"]
    freq_band = cfg["wiener"]["freq_band"]
    n_times = epoch.shape[1]

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg)
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

        group_filters: dict[str, np.ndarray] = {}
        group_unstable = False
        for local_idx, ch in enumerate(pair):
            h = compute_phasegated_filter(
                S,
                target_idx=local_idx,
                phase_gate_threshold_rad=phase_gate_threshold,
            )
            if np.max(np.abs(h)) > mag_threshold:
                group_unstable = True
                break
            group_filters[ch] = h

        if group_unstable:
            skipped.append(pair_key)
            continue

        filters[pair_key] = {}
        for local_idx, (ch, global_idx) in enumerate(zip(pair, indices)):
            h = group_filters[ch]
            sp, co = apply_wiener_filter(group_data, h, local_idx, n_times)
            specific[global_idx] = sp
            coherent[global_idx] = co
            filters[pair_key][ch] = h

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


def decompose_subject(
    epochs: np.ndarray,      # (n_epochs, n_channels, n_times)
    ch_names: list[str],
    subject_id: str,
    cfg: dict,
) -> list[WienerResult]:
    return [
        decompose_epoch(epoch, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
        for i, epoch in enumerate(epochs)
    ]
