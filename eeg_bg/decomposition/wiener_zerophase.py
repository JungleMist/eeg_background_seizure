"""
Ablation: zero-phase Wiener mode. Constrains the per-frequency filter h(f)
to be real-valued rather than the unconstrained complex solve in wiener.py,
while keeping full per-frequency resolution (unlike the band-averaged scalar
in wiener_scalar.py).

Physical motivation: EEG-frequency tissue conduction is quasi-static, so
genuine volume-conducted/myogenic interference between electrodes should
show ~zero phase lag; independent neural sources should not. This module now
uses a zero-referenced phase gate, so phase difference pi is treated as the
maximum phase difference rather than zero-lag polarity reversal.

Math: minimizing E[|X_i(f) - h^T X_ref(f)|^2] over real h at a fixed
frequency bin gives the normal equations Re(S_ref(f)) h(f) = Re(s_cross(f)).
For single-reference (2-channel) groups this is identical to
Re(h_complex(f)) from compute_wiener_filter, since the auto-spectrum on the
diagonal is already real. For 2-reference (3-channel chain) groups it is a
genuine 2x2 real solve, not an approximation of the complex one.
"""
from __future__ import annotations

import numpy as np

from eeg_bg.decomposition.phase_gate import (
    phase_gate_threshold_from_config,
    phase_gate_weights,
    phase_gate_pass_fraction,
)
from eeg_bg.decomposition.wiener import (
    WienerResult,
    apply_wiener_filter,
    decompose_epoch_with_fusion,
)


def compute_zerophase_filter(
    S: np.ndarray,   # (n_ch, n_ch, n_freqs)
    target_idx: int,
    phase_gate_threshold_rad: float = np.pi,
    reg_factor: float = 1e-4,
) -> np.ndarray:
    """Real-constrained per-frequency Wiener filter with phase gating.

    Solves Re(S_ref(f)) @ h(f) = Re(s_cross(f)) at each frequency bin.  Uses
    the same Tikhonov diagonal-loading regularisation as compute_wiener_filter
    in wiener.py, applied to the real matrices.  The final real filter is then
    multiplied by zero-referenced phase-gate weights derived from the original
    complex cross-spectrum.

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
        h = compute_zerophase_filter(
            S,
            target_idx=target_idx,
            phase_gate_threshold_rad=phase_gate_threshold,
        )
        _, coherent = apply_wiener_filter(
            group_data, h, target_idx, n_times
        )
        pass_frac = phase_gate_pass_fraction(
            S, target_idx, phase_gate_threshold, freq_mask,
        )
        return h, coherent, {"pass_fraction": pass_frac}

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
