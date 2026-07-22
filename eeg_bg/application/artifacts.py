from __future__ import annotations

from typing import Any

import numpy as np

from .models import ArtifactSettings


def artifact_mask(data_uv: np.ndarray, threshold_uv: float) -> np.ndarray:
    """Return a per-channel, per-sample strict-threshold mask."""
    return np.abs(np.asarray(data_uv)) > float(threshold_uv)


def _count_true_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.pad(mask.astype(np.int8), (1, 0))
    return int(np.count_nonzero(np.diff(padded) == 1))


def summarize_raw_artifacts(raw, settings: ArtifactSettings) -> dict[str, Any]:
    settings.validate()
    data_uv = raw.get_data() * 1e6
    max_abs_uv = float(np.max(np.abs(data_uv))) if data_uv.size else 0.0
    mask = artifact_mask(data_uv, settings.threshold_uv)
    affected_channels: list[str] = []
    channel_region_counts: dict[str, int] = {}
    for name, channel_mask in zip(raw.ch_names, mask):
        count = _count_true_runs(channel_mask)
        if count:
            affected_channels.append(str(name))
            channel_region_counts[str(name)] = count
    return {
        "enabled": bool(settings.enabled),
        "threshold_uv": float(settings.threshold_uv),
        "affected_channels": affected_channels,
        "affected_channel_count": len(affected_channels),
        "exceedance_region_count": int(sum(channel_region_counts.values())),
        "channel_region_counts": channel_region_counts,
        "max_abs_uv": max_abs_uv,
    }
