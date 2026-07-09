from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import csd, welch, coherence as scipy_coherence


@dataclass
class WienerResult:
    subject_id: str
    epoch_idx: int
    raw: np.ndarray           # (n_channels, n_times)
    specific: np.ndarray      # (n_channels, n_times)
    coherent: np.ndarray      # (n_channels, n_times)
    filters: dict             # {pair_key: {ch_name: h array (n_ref, n_freqs)}}
    freqs: np.ndarray         # frequency axis from Welch
    ch_names: list[str]
    skipped_pairs: list[str] = field(default_factory=list)
    channel_sources: dict[str, list[str]] = field(default_factory=dict)
    channel_weights: dict[str, dict[str, float]] = field(default_factory=dict)


def estimate_cross_psd(
    data: np.ndarray,   # (n_ch, n_times)
    sfreq: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate cross-power spectral density matrix using Welch's method.

    Uses a rectangular (boxcar) window so that when nperseg == n_times the
    filter can be applied exactly via rfft without any windowing mismatch.
    When nperseg < n_times multiple segments are averaged for stability.
    """
    n_ch = data.shape[0]
    freqs, _ = welch(data[0], fs=sfreq, nperseg=nperseg, window='boxcar')
    n_freqs = len(freqs)
    S = np.zeros((n_ch, n_ch, n_freqs), dtype=complex)

    for i in range(n_ch):
        _, psd = welch(data[i], fs=sfreq, nperseg=nperseg, window='boxcar')
        S[i, i] = psd.astype(complex)
        for j in range(i + 1, n_ch):
            _, cross = csd(data[i], data[j], fs=sfreq, nperseg=nperseg,
                           window='boxcar')
            S[i, j] = cross
            S[j, i] = np.conj(cross)
    return freqs, S


def compute_wiener_filter(
    S: np.ndarray,   # (n_ch, n_ch, n_freqs)
    target_idx: int,
    reg_factor: float = 1e-4,
) -> np.ndarray:
    """Estimate per-frequency Wiener filter coefficients.

    Uses Tikhonov (diagonal loading) regularisation proportional to the mean
    diagonal of S_ref to stabilise the solve for near-singular cross-PSD
    matrices.  This is essential for 3-electrode chains (2×2 S_ref) where the
    two reference channels (e.g. T5 and O1) may be highly correlated.

    Parameters
    ----------
    S : np.ndarray, shape ``(n_ch, n_ch, n_freqs)``
    target_idx : int
    reg_factor : float
        Diagonal loading as a fraction of the mean real diagonal of S_ref
        (default 1e-4).

    Returns
    -------
    h : np.ndarray, shape ``(n_ref, n_freqs)``, complex
    """
    n_ch = S.shape[0]
    n_freqs = S.shape[2]
    ref_indices = [i for i in range(n_ch) if i != target_idx]
    n_ref = len(ref_indices)
    h = np.zeros((n_ref, n_freqs), dtype=complex)

    for f in range(n_freqs):
        S_ref = S[np.ix_(ref_indices, ref_indices)][:, :, f]
        s_cross = S[target_idx, ref_indices, f]
        # Diagonal loading: eps = reg_factor × mean(diag(S_ref)), floored at
        # 1e-30 to avoid zero-regularisation on silent channels.
        eps = reg_factor * max(float(np.real(np.diag(S_ref)).mean()), 1e-30)
        S_ref_reg = S_ref + eps * np.eye(n_ref, dtype=complex)
        try:
            h[:, f] = np.linalg.solve(S_ref_reg, s_cross)
        except np.linalg.LinAlgError:
            pass
    return h


def apply_wiener_filter(
    group_data: np.ndarray,  # (n_ch, n_times) for the group
    h: np.ndarray,           # (n_ref, n_freqs_welch)  where n_freqs_welch = nperseg//2+1
    target_idx: int,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Wiener filter in the frequency domain.

    When nperseg == n_times (and the rectangular window was used for estimation),
    this is exact. When nperseg < n_times the filter is linearly interpolated
    to the full rfft grid; specific + coherent == raw is guaranteed by
    construction regardless of interpolation accuracy.
    """
    ref_indices = [i for i in range(group_data.shape[0]) if i != target_idx]
    n_ref, n_freqs_welch = h.shape
    n_freqs_full = n_times // 2 + 1

    if n_freqs_welch == n_freqs_full:
        # Exact: no interpolation needed
        h_full = h
    else:
        # Interpolate filter coefficients to the full rfft grid
        welch_bins = np.linspace(0.0, 1.0, n_freqs_welch)
        full_bins = np.linspace(0.0, 1.0, n_freqs_full)
        h_full = np.zeros((n_ref, n_freqs_full), dtype=complex)
        for r in range(n_ref):
            h_full[r].real = np.interp(full_bins, welch_bins, h[r].real)
            h_full[r].imag = np.interp(full_bins, welch_bins, h[r].imag)

    ref_fft = np.fft.rfft(group_data[ref_indices], axis=-1)  # (n_ref, n_freqs_full)
    coherent_fft = np.sum(h_full * ref_fft, axis=0)          # (n_freqs_full,)
    coherent = np.fft.irfft(coherent_fft, n=n_times)
    specific = group_data[target_idx] - coherent
    return specific, coherent


def _max_target_ref_coherence(
    group_data: np.ndarray,
    target_idx: int,
    sfreq: float,
    nperseg: int,
    freq_mask: np.ndarray,
) -> float:
    """Max band coherence between one target channel and its group references."""
    if not np.any(freq_mask):
        return 0.0

    max_coh = 0.0
    for ref_idx in range(group_data.shape[0]):
        if ref_idx == target_idx:
            continue
        _, coh = scipy_coherence(
            group_data[target_idx],
            group_data[ref_idx],
            fs=sfreq,
            nperseg=nperseg,
        )
        max_coh = max(max_coh, float(np.nanmax(coh[freq_mask])))
    return max_coh


def _frequency_candidate(
    group_data: np.ndarray,
    S: np.ndarray,
    target_idx: int,
    freq_mask: np.ndarray,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = compute_wiener_filter(S, target_idx=target_idx)
    _, coherent = apply_wiener_filter(group_data, h, target_idx, n_times)
    return h, coherent


def decompose_epoch_with_fusion(
    epoch: np.ndarray,
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
    candidate_fn: Callable[
        [np.ndarray, np.ndarray, int, np.ndarray, int],
        tuple[np.ndarray, np.ndarray],
    ] = _frequency_candidate,
) -> WienerResult:
    """Decompose one epoch using target-level gating and overlap fusion.

    Each channel group can contribute one coherent-signal candidate per target
    channel.  Overlapping channel candidates are combined by target-reference
    coherence, which avoids the old group-order-dependent overwrite behavior.
    """
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    coh_threshold = cfg["wiener"]["coherence_threshold"]
    mag_threshold = float(cfg["wiener"].get("filter_magnitude_threshold", 50.0))
    overlap_policy = cfg["wiener"].get("overlap_policy", "coherence_weighted")
    channel_groups = cfg["channels"]["channel_groups"]
    freq_band = cfg["wiener"]["freq_band"]
    n_times = epoch.shape[1]

    if overlap_policy != "coherence_weighted":
        raise ValueError("Only overlap_policy='coherence_weighted' is supported")

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []
    candidates_by_channel: dict[int, list[tuple[str, float, np.ndarray]]] = {}

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg, window="boxcar")
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

    for pair in channel_groups:
        pair_key = "-".join(pair)
        try:
            indices = [ch_names.index(ch) for ch in pair]
        except ValueError:
            skipped.append(pair_key)
            continue

        if len(indices) < 2:
            skipped.append(pair_key)
            continue

        group_data = epoch[indices]
        _, S = estimate_cross_psd(group_data, sfreq, nperseg)
        pair_filters: dict[str, np.ndarray] = {}

        for local_idx, (ch, global_idx) in enumerate(zip(pair, indices)):
            score = _max_target_ref_coherence(
                group_data,
                target_idx=local_idx,
                sfreq=sfreq,
                nperseg=nperseg,
                freq_mask=freq_mask,
            )
            if score < coh_threshold:
                continue

            h, candidate_coherent = candidate_fn(
                group_data,
                S,
                local_idx,
                freq_mask,
                n_times,
            )
            if h.size and np.max(np.abs(h)) > mag_threshold:
                continue

            pair_filters[ch] = h
            candidates_by_channel.setdefault(global_idx, []).append(
                (pair_key, score, candidate_coherent)
            )

        if pair_filters:
            filters[pair_key] = pair_filters
        else:
            skipped.append(pair_key)

    channel_sources: dict[str, list[str]] = {}
    channel_weights: dict[str, dict[str, float]] = {}
    for global_idx, candidates in candidates_by_channel.items():
        ordered = sorted(candidates, key=lambda item: item[0])
        scores = np.array([item[1] for item in ordered], dtype=float)
        if float(scores.sum()) > 0.0:
            weights = scores / float(scores.sum())
        else:
            weights = np.full(len(ordered), 1.0 / len(ordered), dtype=float)

        fused = np.zeros(n_times, dtype=coherent.dtype)
        sources: list[str] = []
        weights_by_source: dict[str, float] = {}
        for weight, (pair_key, _, candidate_coherent) in zip(weights, ordered):
            fused = fused + weight * candidate_coherent
            sources.append(pair_key)
            weights_by_source[pair_key] = float(weight)

        coherent[global_idx] = fused
        specific[global_idx] = epoch[global_idx] - fused
        ch = ch_names[global_idx]
        channel_sources[ch] = sources
        channel_weights[ch] = weights_by_source

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
        channel_sources=channel_sources,
        channel_weights=channel_weights,
    )


def decompose_epoch(
    epoch: np.ndarray,       # (n_channels, n_times)
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
) -> WienerResult:
    return decompose_epoch_with_fusion(
        epoch,
        ch_names,
        cfg,
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        candidate_fn=_frequency_candidate,
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
