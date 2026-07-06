"""
Ablation: zero-phase Wiener mode. Constrains the per-frequency filter h(f)
to be real-valued (zero phase) rather than the unconstrained complex solve
in wiener.py, while keeping full per-frequency resolution (unlike the
band-averaged scalar in wiener_scalar.py).

Physical motivation: EEG-frequency tissue conduction is quasi-static, so
genuine volume-conducted/myogenic interference between electrodes should
show ~zero phase lag; independent neural sources should not. Constraining
h(f) to be real at every frequency bin targets exactly that component.

Math: minimizing E[|X_i(f) - h^T X_ref(f)|^2] over real h at a fixed
frequency bin gives the normal equations Re(S_ref(f)) h(f) = Re(s_cross(f)).
For single-reference (2-channel) groups this is identical to
Re(h_complex(f)) from compute_wiener_filter, since the auto-spectrum on the
diagonal is already real. For 2-reference (3-channel chain) groups it is a
genuine 2x2 real solve, not an approximation of the complex one.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import coherence as scipy_coherence, welch

from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    apply_wiener_filter,
)


def compute_zerophase_filter(
    S: np.ndarray,   # (n_ch, n_ch, n_freqs)
    target_idx: int,
    reg_factor: float = 1e-4,
) -> np.ndarray:
    """Real-constrained per-frequency Wiener filter.

    Solves Re(S_ref(f)) @ h(f) = Re(s_cross(f)) at each frequency bin.  Uses
    the same Tikhonov diagonal-loading regularisation as compute_wiener_filter
    in wiener.py, applied to the real matrices.

    Returns
    -------
    h : np.ndarray, shape ``(n_ref, n_freqs)``, real (float64)
    """
    n_ch = S.shape[0]
    n_freqs = S.shape[2]
    ref_indices = [i for i in range(n_ch) if i != target_idx]
    n_ref = len(ref_indices)
    h = np.zeros((n_ref, n_freqs), dtype=np.float64)

    for f in range(n_freqs):
        S_ref = np.real(S[np.ix_(ref_indices, ref_indices)][:, :, f])
        s_cross = np.real(S[target_idx, ref_indices, f])
        eps = reg_factor * max(float(np.diag(S_ref).mean()), 1e-30)
        S_ref_reg = S_ref + eps * np.eye(n_ref, dtype=np.float64)
        try:
            h[:, f] = np.linalg.solve(S_ref_reg, s_cross)
        except np.linalg.LinAlgError:
            pass
    return h


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
            h = compute_zerophase_filter(S, target_idx=local_idx)
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
