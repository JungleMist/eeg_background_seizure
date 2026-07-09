"""
Ablation: scalar Wiener mode. Replaces frequency-dependent h(f) with a
single complex scalar per reference channel (average over frequency band).
Equivalent to the EKG-style fixed compensation described in the proposal.
"""
from __future__ import annotations

import numpy as np

from eeg_bg.decomposition.wiener import (
    WienerResult,
    compute_wiener_filter,
    decompose_epoch_with_fusion,
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
    def candidate_fn(group_data, S, target_idx, freq_mask, n_times):
        h_freq = compute_wiener_filter(S, target_idx=target_idx)
        h_scalar = _scalar_from_filter(h_freq, freq_mask)

        ref_indices = [
            i for i in range(group_data.shape[0]) if i != target_idx
        ]
        ref_data = group_data[ref_indices]  # (n_ref, n_times)
        coherent_signal = np.sum(h_scalar.real * ref_data, axis=0)
        return h_scalar, coherent_signal

    return decompose_epoch_with_fusion(
        epoch,
        ch_names,
        cfg,
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        candidate_fn=candidate_fn,
    )
