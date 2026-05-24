import csv
from pathlib import Path


def extract_bckg_intervals(
    csv_bi_path: Path, cfg: dict, recording_duration: float | None = None
) -> list[tuple[float, float]]:
    """Return background intervals from a TUEP csv_bi annotation file.

    When *recording_duration* is provided (recommended), background is defined
    as the full recording minus any seizure-annotated segments ± buffer.  This
    is the correct semantics for TUEP v3.1.0, whose csv_bi files only annotate
    a short representative sample as "bckg" even when the rest of the recording
    is also background.

    When *recording_duration* is None (legacy), background is derived solely
    from the explicit "bckg" rows in the file.
    """
    buffer = cfg["preprocessing"]["seizure_buffer_sec"]
    bckg_intervals: list[tuple[float, float]] = []
    seiz_intervals: list[tuple[float, float]] = []

    with open(csv_bi_path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        label = row["label"].strip()
        if label == "bckg":
            bckg_intervals.append((start, stop))
        elif label == "seiz":
            seiz_intervals.append((start, stop))

    excluded = [(max(0.0, s - buffer), e + buffer) for s, e in seiz_intervals]

    # Base intervals: full recording (new) or explicit bckg rows (legacy).
    base = [(0.0, recording_duration)] if recording_duration is not None else bckg_intervals

    result: list[tuple[float, float]] = []
    for b_start, b_end in base:
        segments = [(b_start, b_end)]
        for ex_start, ex_end in excluded:
            new_segs: list[tuple[float, float]] = []
            for seg_s, seg_e in segments:
                if ex_end <= seg_s or ex_start >= seg_e:
                    new_segs.append((seg_s, seg_e))
                else:
                    if seg_s < ex_start:
                        new_segs.append((seg_s, ex_start))
                    if seg_e > ex_end:
                        new_segs.append((ex_end, seg_e))
            segments = new_segs
        result.extend(segments)

    return result
