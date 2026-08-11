from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.signal import csd, welch, coherence as scipy_coherence


# Status codes for per-candidate diagnostics added to WienerResult.
CANDIDATE_PROCESSED       = 0
CANDIDATE_BELOW_COHERENCE = 1
CANDIDATE_UNSTABLE_FILTER = 2
CANDIDATE_SOLVE_FAILED    = 3
CANDIDATE_MISSING_CHANNEL = 4
CANDIDATE_COHERENT_GATE_CLOSED = 5
CANDIDATE_UNUSED          = 255


class WienerSolveError(RuntimeError):
    """Raised when a per-frequency Wiener system cannot be solved safely."""


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
    # ── target-level diagnostics (added 2026-07-10) ────────────────────────
    # Fixed-size arrays padded to max_candidates = sum(len(pair) for pair in
    # channel_groups).  Keys are "group::channel" e.g. "FP1-FP2::FP1".
    candidate_keys: list[str] | None = None
    candidate_status: np.ndarray | None = None         # uint8 (n_candidates,)
    candidate_coherence: np.ndarray | None = None       # float64 (n_candidates,)
    candidate_max_abs_h: np.ndarray | None = None       # float64 (n_candidates,)
    phase_gate_pass_fraction: np.ndarray | None = None  # float64 (n_candidates,)
    candidate_fusion_weight: np.ndarray | None = None   # float64 (n_candidates,)
    # Group-level coherent power-gate diagnostics in channel_groups order.
    group_gate_keys: list[str] | None = None
    group_coherent_gate_open: np.ndarray | None = None  # bool (n_groups,)
    group_max_bin_rms_uv: np.ndarray | None = None      # float64 (n_groups,)


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

    Raises
    ------
    WienerSolveError
        If a frequency-bin system is singular or produces non-finite values.
    """
    n_ch = S.shape[0]
    n_freqs = S.shape[2]
    ref_indices = [i for i in range(n_ch) if i != target_idx]
    n_ref = len(ref_indices)
    h = np.zeros((n_ref, n_freqs), dtype=complex)

    for f in range(n_freqs):
        S_ref = S[np.ix_(ref_indices, ref_indices)][:, :, f]
        # scipy.signal.csd(x, y) returns conj(X) * Y.  Since the filter is
        # applied as sum(h * X_ref), the normal-equation right-hand side is
        # E[conj(X_ref) * X_target], i.e. S[ref, target].
        s_cross = S[ref_indices, target_idx, f]
        # Diagonal loading: eps = reg_factor × mean(diag(S_ref)), floored at
        # 1e-30 to avoid zero-regularisation on silent channels.
        eps = reg_factor * max(float(np.real(np.diag(S_ref)).mean()), 1e-30)
        S_ref_reg = S_ref + eps * np.eye(n_ref, dtype=complex)
        try:
            solution = np.linalg.solve(S_ref_reg, s_cross)
        except np.linalg.LinAlgError as exc:
            raise WienerSolveError(
                f"Wiener solve failed at frequency bin {f}"
            ) from exc
        if not np.all(np.isfinite(solution)):
            raise WienerSolveError(
                f"Wiener solve produced non-finite values at frequency bin {f}"
            )
        h[:, f] = solution
    return h


def apply_wiener_filter(
    group_data: np.ndarray,  # (n_ch, n_times) for the group
    h: np.ndarray,           # (n_ref, n_freqs_welch)  where n_freqs_welch = nperseg//2+1
    target_idx: int,
    n_times: int,
    sfreq: float | None = None,
    protected_band_hz: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Wiener filter in the frequency domain.

    When nperseg == n_times (and the rectangular window was used for estimation),
    this is exact. When nperseg < n_times the filter is linearly interpolated
    to the full rfft grid; specific + coherent == raw is guaranteed by
    construction regardless of interpolation accuracy.  When
    ``protected_band_hz`` is provided, the corresponding full-resolution FFT
    bins are forced to zero after interpolation so no coherent component is
    reconstructed inside that closed frequency interval.
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

    if protected_band_hz is not None:
        if sfreq is None:
            raise ValueError(
                "sfreq is required when protected_band_hz is enabled"
            )
        low_hz, high_hz = protected_band_hz
        full_freqs = np.fft.rfftfreq(n_times, d=1.0 / float(sfreq))
        protected_mask = (full_freqs >= low_hz) & (full_freqs <= high_hz)
        h_full = h_full.copy()
        h_full[:, protected_mask] = 0.0

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


def protected_band_from_config(
    cfg: dict,
) -> tuple[float, float] | None:
    """Return the optional closed frequency band protected from ECMAD."""
    value = cfg["wiener"].get("protected_band_hz", [5.0, 20.0])
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("wiener.protected_band_hz must be null or [low, high]")
    low_hz, high_hz = (float(value[0]), float(value[1]))
    if not np.isfinite(low_hz) or not np.isfinite(high_hz):
        raise ValueError("wiener.protected_band_hz values must be finite")
    if low_hz >= high_hz:
        raise ValueError(
            "wiener.protected_band_hz must satisfy low < high"
        )
    return low_hz, high_hz


def coherent_gate_from_config(cfg: dict) -> tuple[bool, float]:
    """Return and validate the group-level coherent power-gate settings."""
    wiener = cfg["wiener"]
    enabled = wiener.get("coherent_gate_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("wiener.coherent_gate_enabled must be a boolean")
    threshold = wiener.get("coherent_gate_threshold_uv", 100.0)
    if isinstance(threshold, bool):
        raise ValueError(
            "wiener.coherent_gate_threshold_uv must be a finite positive number"
        )
    threshold_uv = float(threshold)
    if not np.isfinite(threshold_uv) or threshold_uv <= 0.0:
        raise ValueError(
            "wiener.coherent_gate_threshold_uv must be a finite positive number"
        )
    return enabled, threshold_uv


def group_max_bin_rms_uv(
    S: np.ndarray,
    freqs: np.ndarray,
    freq_mask: np.ndarray,
) -> float:
    """Maximum single-bin RMS amplitude across a channel group, in microvolts."""
    delta_f = float(freqs[1] - freqs[0])
    diagonal_psd = np.stack(
        [np.real(S[index, index]) for index in range(S.shape[0])],
        axis=0,
    )
    bin_power = np.maximum(diagonal_psd[:, freq_mask], 0.0) * delta_f
    return float(np.sqrt(bin_power).max())


def build_frequency_masks(
    freqs: np.ndarray,
    freq_band: tuple[float, float] | list[float],
    protected_band_hz: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the analysis and protected-band-excluding score masks."""
    low_hz, high_hz = (float(freq_band[0]), float(freq_band[1]))
    analysis_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if protected_band_hz is None:
        score_mask = analysis_mask.copy()
    else:
        protected_low, protected_high = protected_band_hz
        if protected_low < low_hz or protected_high > high_hz:
            raise ValueError(
                "wiener.protected_band_hz must be within wiener.freq_band"
            )
        protected_mask = (
            (freqs >= protected_low) & (freqs <= protected_high)
        )
        score_mask = analysis_mask & ~protected_mask
    if not np.any(score_mask):
        raise ValueError(
            "wiener.protected_band_hz leaves no frequency bins for "
            "coherence scoring"
        )
    return analysis_mask, score_mask


def zero_protected_filter_bins(
    h: np.ndarray,
    sfreq: float,
    protected_band_hz: tuple[float, float] | None,
) -> np.ndarray:
    """Zero protected Welch-grid filter bins without modifying the input."""
    if protected_band_hz is None or h.shape[1] <= 1:
        return h
    freqs = np.linspace(0.0, float(sfreq) / 2.0, h.shape[1])
    low_hz, high_hz = protected_band_hz
    protected_mask = (freqs >= low_hz) & (freqs <= high_hz)
    masked = h.copy()
    masked[:, protected_mask] = 0.0
    return masked


def _frequency_candidate(
    group_data: np.ndarray,
    S: np.ndarray,
    target_idx: int,
    freq_mask: np.ndarray,
    n_times: int,
    *,
    sfreq: float | None = None,
    protected_band_hz: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h = compute_wiener_filter(S, target_idx=target_idx)
    if sfreq is not None:
        h = zero_protected_filter_bins(h, sfreq, protected_band_hz)
    _, coherent = apply_wiener_filter(
        group_data,
        h,
        target_idx,
        n_times,
        sfreq=sfreq,
        protected_band_hz=protected_band_hz,
    )
    return h, coherent, {}


def decompose_epoch_with_fusion(
    epoch: np.ndarray,
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
    candidate_fn: Callable[
        [np.ndarray, np.ndarray, int, np.ndarray, int],
        tuple[np.ndarray, np.ndarray, dict],
    ] | None = None,
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
    protected_band_hz = protected_band_from_config(cfg)
    coherent_gate_enabled, coherent_gate_threshold_uv = (
        coherent_gate_from_config(cfg)
    )
    n_times = epoch.shape[1]

    if overlap_policy != "coherence_weighted":
        raise ValueError("Only overlap_policy='coherence_weighted' is supported")

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []
    candidates_by_channel: dict[int, list[tuple[str, float, np.ndarray]]] = {}

    # ── Per-candidate diagnostic tracking ─────────────────────────────────
    _candidate_keys: list[str] = []
    _candidate_status: list[int] = []
    _candidate_coherence: list[float] = []
    _candidate_max_abs_h: list[float] = []
    _candidate_phase_pass: list[float] = []
    _group_gate_keys: list[str] = []
    _group_coherent_gate_open: list[bool] = []
    _group_max_bin_rms_uv: list[float] = []

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg, window="boxcar")
    _, freq_mask = build_frequency_masks(
        freqs,
        freq_band,
        protected_band_hz,
    )

    if candidate_fn is None:
        def candidate_fn(group_data, S, target_idx, freq_mask, n_times):
            return _frequency_candidate(
                group_data,
                S,
                target_idx,
                freq_mask,
                n_times,
                sfreq=sfreq,
                protected_band_hz=protected_band_hz,
            )

    for pair in channel_groups:
        pair_key = "-".join(pair)
        _group_gate_keys.append(pair_key)
        try:
            indices = [ch_names.index(ch) for ch in pair]
        except ValueError:
            _group_coherent_gate_open.append(False)
            _group_max_bin_rms_uv.append(float("nan"))
            # Record all targets as missing_channel, then skip the group.
            for ch in pair:
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_MISSING_CHANNEL)
                _candidate_coherence.append(0.0)
                _candidate_max_abs_h.append(0.0)
                _candidate_phase_pass.append(0.0)
            skipped.append(pair_key)
            continue

        if len(indices) < 2:
            _group_coherent_gate_open.append(False)
            _group_max_bin_rms_uv.append(float("nan"))
            for ch in pair:
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_MISSING_CHANNEL)
                _candidate_coherence.append(0.0)
                _candidate_max_abs_h.append(0.0)
                _candidate_phase_pass.append(0.0)
            skipped.append(pair_key)
            continue

        group_data = epoch[indices]
        _, S = estimate_cross_psd(group_data, sfreq, nperseg)
        max_bin_rms_uv = group_max_bin_rms_uv(S, freqs, freq_mask)
        coherent_gate_open = (
            not coherent_gate_enabled
            or max_bin_rms_uv > coherent_gate_threshold_uv
        )
        _group_coherent_gate_open.append(coherent_gate_open)
        _group_max_bin_rms_uv.append(max_bin_rms_uv)
        if not coherent_gate_open:
            for ch in pair:
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_COHERENT_GATE_CLOSED)
                _candidate_coherence.append(0.0)
                _candidate_max_abs_h.append(0.0)
                _candidate_phase_pass.append(0.0)
            skipped.append(pair_key)
            continue

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
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_BELOW_COHERENCE)
                _candidate_coherence.append(score)
                _candidate_max_abs_h.append(0.0)
                _candidate_phase_pass.append(0.0)
                continue

            try:
                h, candidate_coherent, diagnostics = candidate_fn(
                    group_data,
                    S,
                    local_idx,
                    freq_mask,
                    n_times,
                )
            except WienerSolveError:
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_SOLVE_FAILED)
                _candidate_coherence.append(score)
                _candidate_max_abs_h.append(0.0)
                _candidate_phase_pass.append(0.0)
                continue
            max_abs_h = float(np.max(np.abs(h))) if h.size else 0.0
            pass_frac = float(diagnostics.get("pass_fraction", 1.0))

            if h.size and max_abs_h > mag_threshold:
                _candidate_keys.append(f"{pair_key}::{ch}")
                _candidate_status.append(CANDIDATE_UNSTABLE_FILTER)
                _candidate_coherence.append(score)
                _candidate_max_abs_h.append(max_abs_h)
                _candidate_phase_pass.append(pass_frac)
                continue

            # ── Candidate accepted ─────────────────────────────────────────
            _candidate_keys.append(f"{pair_key}::{ch}")
            _candidate_status.append(CANDIDATE_PROCESSED)
            _candidate_coherence.append(score)
            _candidate_max_abs_h.append(max_abs_h)
            _candidate_phase_pass.append(pass_frac)

            pair_filters[ch] = h
            candidates_by_channel.setdefault(global_idx, []).append(
                (pair_key, score, candidate_coherent)
            )

        if pair_filters:
            filters[pair_key] = pair_filters
        else:
            skipped.append(pair_key)

    # ── Build fixed-size diagnostic arrays ───────────────────────────────
    n_candidates = len(_candidate_keys)
    candidate_keys: list[str] = _candidate_keys
    candidate_status = np.array(_candidate_status, dtype=np.uint8) if n_candidates else None
    candidate_coherence = np.array(_candidate_coherence, dtype=np.float64) if n_candidates else None
    candidate_max_abs_h = np.array(_candidate_max_abs_h, dtype=np.float64) if n_candidates else None
    phase_gate_pass_fraction = np.array(_candidate_phase_pass, dtype=np.float64) if n_candidates else None
    candidate_fusion_weight = np.zeros(n_candidates, dtype=np.float64) if n_candidates else None

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
            if candidate_fusion_weight is not None:
                key = f"{pair_key}::{ch_names[global_idx]}"
                try:
                    candidate_fusion_weight[candidate_keys.index(key)] = float(weight)
                except ValueError:
                    # Defensive guard: candidate keys are expected to be
                    # aligned, but a missing diagnostic must not affect the
                    # decomposed signal.
                    pass

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
        candidate_keys=candidate_keys,
        candidate_status=candidate_status,
        candidate_coherence=candidate_coherence,
        candidate_max_abs_h=candidate_max_abs_h,
        phase_gate_pass_fraction=phase_gate_pass_fraction,
        candidate_fusion_weight=candidate_fusion_weight,
        group_gate_keys=_group_gate_keys,
        group_coherent_gate_open=np.asarray(
            _group_coherent_gate_open, dtype=bool
        ),
        group_max_bin_rms_uv=np.asarray(
            _group_max_bin_rms_uv, dtype=np.float64
        ),
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
