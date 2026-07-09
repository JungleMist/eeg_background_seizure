"""
Zero-referenced phase-gated complex Wiener mode.

This mode first estimates the standard complex per-frequency Wiener filter,
then keeps only reference/frequency coefficients whose target-reference
cross-spectrum phase is close enough to 0.  Phase difference pi is treated as
the maximum phase difference, so anti-phase coherent components are preserved
unless ``wiener.phase_gate_threshold_rad`` is set to pi.
"""
from __future__ import annotations

import numpy as np

from eeg_bg.decomposition.phase_gate import (
    phase_gate_threshold_from_config,
    phase_gate_weights,
)
from eeg_bg.decomposition.wiener import (
    WienerResult,
    compute_wiener_filter,
    apply_wiener_filter,
    decompose_epoch_with_fusion,
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
    phase_gate_threshold = phase_gate_threshold_from_config(cfg)

    def candidate_fn(group_data, S, target_idx, freq_mask, n_times):
        h = compute_phasegated_filter(
            S,
            target_idx=target_idx,
            phase_gate_threshold_rad=phase_gate_threshold,
        )
        _, coherent = apply_wiener_filter(
            group_data, h, target_idx, n_times
        )
        return h, coherent

    return decompose_epoch_with_fusion(
        epoch,
        ch_names,
        cfg,
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        candidate_fn=candidate_fn,
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
