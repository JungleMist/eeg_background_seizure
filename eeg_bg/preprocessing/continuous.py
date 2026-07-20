from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from eeg_bg.decomposition.wiener import (
    CANDIDATE_BELOW_COHERENCE,
    CANDIDATE_PROCESSED,
    CANDIDATE_SOLVE_FAILED,
    decompose_epoch as decompose_epoch_frequency,
)
from eeg_bg.decomposition.wiener_phasegated import (
    decompose_epoch as decompose_epoch_phasegated,
)
from eeg_bg.decomposition.wiener_zerophase import (
    decompose_epoch as decompose_epoch_zerophase,
)
from eeg_bg.exceptions import ProcessingCancelled


def select_wiener_decomposer(mode: str):
    dispatch = {
        "frequency": decompose_epoch_frequency,
        "phasegated": decompose_epoch_phasegated,
        "zerophase": decompose_epoch_zerophase,
    }
    try:
        return dispatch[mode]
    except KeyError as exc:
        raise ValueError(
            "Wiener mode must be 'frequency', 'phasegated', or 'zerophase'"
        ) from exc


def _active_groups(result) -> set[str]:
    active: set[str] = set()
    if result.candidate_keys is not None and result.candidate_status is not None:
        for key, status in zip(result.candidate_keys, result.candidate_status):
            if int(status) == CANDIDATE_PROCESSED:
                active.add(str(key).split("::", 1)[0])
        return active
    for sources in result.channel_sources.values():
        active.update(sources)
    return active


def candidate_diagnostics(result) -> dict:
    """Return JSON-serializable per-candidate diagnostics for one window."""
    arrays = {
        "candidate_status": result.candidate_status,
        "candidate_coherence": result.candidate_coherence,
        "candidate_max_abs_h": result.candidate_max_abs_h,
        "phase_gate_pass_fraction": result.phase_gate_pass_fraction,
        "candidate_fusion_weight": result.candidate_fusion_weight,
    }
    payload = {
        "candidate_keys": list(result.candidate_keys or []),
        "skipped_groups": list(result.skipped_pairs),
        "channel_sources": {
            key: list(value) for key, value in result.channel_sources.items()
        },
    }
    for key, value in arrays.items():
        payload[key] = value.tolist() if value is not None else []
    return payload


def wiener_continuous_raw(
    raw,
    cfg: dict,
    subject_id: str = "recording",
    *,
    cancel_requested: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
):
    """Apply windowed Wiener decomposition with 50% weighted overlap-add."""
    data = raw.get_data()
    if data.shape[1] < 2:
        raise ValueError("Recording is too short for Wiener processing")
    sfreq = float(raw.info["sfreq"])
    local_cfg = deepcopy(cfg)
    local_cfg["preprocessing"]["target_sfreq"] = sfreq
    mode = str(local_cfg["wiener"].get("mode", "frequency"))
    decomposer = select_wiener_decomposer(mode)
    requested = int(round(local_cfg["preprocessing"]["epoch_length_sec"] * sfreq))
    n_times = min(data.shape[1], max(2, requested))
    local_cfg["wiener"]["nperseg"] = min(
        int(local_cfg["wiener"]["nperseg"]), n_times
    )
    hop = max(1, n_times // 2)
    pad = hop if data.shape[1] > 1 else 0
    padded = np.pad(data, ((0, 0), (pad, pad)), mode="reflect")
    starts = list(range(0, padded.shape[1] - n_times + 1, hop))
    last = padded.shape[1] - n_times
    if not starts or starts[-1] != last:
        starts.append(last)

    window = np.hanning(n_times + 2)[1:-1]
    numerator = np.zeros_like(padded)
    denominator = np.zeros(padded.shape[1], dtype=float)
    processed_channel_windows = 0
    group_keys = ["-".join(group) for group in local_cfg["channels"]["channel_groups"]]
    group_active = {key: 0 for key in group_keys}
    solve_failures = 0
    below_coherence = 0
    window_diagnostics: list[dict] = []

    for epoch_idx, start in enumerate(starts):
        if cancel_requested is not None and cancel_requested():
            raise ProcessingCancelled("用户已取消处理")
        chunk = padded[:, start : start + n_times]
        result = decomposer(
            chunk,
            list(raw.ch_names),
            local_cfg,
            subject_id=subject_id,
            epoch_idx=epoch_idx,
        )
        numerator[:, start : start + n_times] += result.specific * window
        denominator[start : start + n_times] += window
        processed_channel_windows += len(result.channel_sources)
        for key in _active_groups(result):
            group_active[key] = group_active.get(key, 0) + 1
        if result.candidate_status is not None:
            solve_failures += int(
                np.count_nonzero(result.candidate_status == CANDIDATE_SOLVE_FAILED)
            )
            below_coherence += int(
                np.count_nonzero(result.candidate_status == CANDIDATE_BELOW_COHERENCE)
            )
        window_diagnostics.append({
            "window_index": epoch_idx,
            "start_sample": start - pad,
            **candidate_diagnostics(result),
        })
        if progress is not None:
            progress(epoch_idx + 1, len(starts))

    output = numerator / np.maximum(denominator, np.finfo(float).eps)
    denoised = raw.copy()
    denoised._data = output[:, pad : pad + data.shape[1]]
    total = len(starts)
    group_rates = {
        key: {
            "active_windows": group_active.get(key, 0),
            "total_windows": total,
            "rate": float(group_active.get(key, 0)) / total if total else 0.0,
        }
        for key in group_keys
    }
    return denoised, {
        "mode": mode,
        "windows": total,
        "processed_channel_windows": processed_channel_windows,
        "group_processing_rates": group_rates,
        "solve_failures": solve_failures,
        "below_coherence_candidates": below_coherence,
        "window_diagnostics": window_diagnostics,
    }
