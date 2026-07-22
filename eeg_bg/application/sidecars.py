from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .models import RecordingEvent, RecordingSidecars


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON 顶层必须为对象")
    return payload


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _parse_events(rows: list[dict[str, str]]) -> tuple[list[RecordingEvent], int]:
    events: list[RecordingEvent] = []
    invalid = 0
    for row in rows:
        try:
            onset = float(row.get("onset", ""))
            if not math.isfinite(onset):
                raise ValueError
            duration_text = row.get("duration", "")
            duration = float(duration_text) if duration_text not in {"", "n/a"} else 0.0
            if not math.isfinite(duration):
                duration = 0.0
            sample_text = row.get("sample", "")
            sample = (
                int(round(float(sample_text)))
                if sample_text not in {"", "n/a"}
                else None
            )
        except (TypeError, ValueError):
            invalid += 1
            continue
        events.append(
            RecordingEvent(
                onset_sec=onset,
                duration_sec=max(0.0, duration),
                sample=sample,
                trial_type=row.get("trial_type", ""),
                value=row.get("value", ""),
                fields=dict(row),
            )
        )
    events.sort(key=lambda event: event.onset_sec)
    return events, invalid


def read_recording_sidecars(
    recording_path: str | Path,
) -> tuple[RecordingSidecars, list[str]]:
    """Read BIDS-style sidecars sharing an EEGLAB recording prefix."""
    path = Path(recording_path)
    if path.suffix.lower() != ".set":
        return RecordingSidecars(), []

    prefix = path.stem[:-4] if path.stem.lower().endswith("_eeg") else path.stem
    candidates = {
        "eeg": path.with_name(f"{prefix}_eeg.json"),
        "channels": path.with_name(f"{prefix}_channels.tsv"),
        "events": path.with_name(f"{prefix}_events.tsv"),
        "electrodes": path.with_name(f"{prefix}_electrodes.tsv"),
        "coordsystem": path.with_name(f"{prefix}_coordsystem.json"),
    }
    sidecars = RecordingSidecars()
    warnings: list[str] = []
    for kind, candidate in candidates.items():
        if not candidate.is_file():
            continue
        sidecars.paths[kind] = candidate.resolve()
        try:
            if kind in {"eeg", "coordsystem"}:
                setattr(sidecars, kind, _read_json(candidate))
                continue
            rows = _read_tsv(candidate)
            if kind == "events":
                sidecars.events, invalid = _parse_events(rows)
                if invalid:
                    warnings.append(f"events.tsv 中有 {invalid} 行缺少有效 onset，已跳过")
            else:
                setattr(sidecars, kind, rows)
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            warnings.append(f"无法读取 {candidate.name}：{exc}")
    return sidecars, warnings
