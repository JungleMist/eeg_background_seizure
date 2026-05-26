from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations

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

        # Coherence gate — max pairwise coherence over all channel pairs in the group
        # (handles k=2 pairs and k=3 chains uniformly)
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

        # Pre-compute all filters and validate magnitudes before applying any.
        # If any channel's filter blows up (near-singular S_ref after
        # regularisation), skip the entire group — same behaviour as the
        # coherence gate above.
        group_filters: dict[str, np.ndarray] = {}
        group_unstable = False
        for local_idx, ch in enumerate(pair):
            h = compute_wiener_filter(S, target_idx=local_idx)
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
