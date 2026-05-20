import csv
from pathlib import Path


def extract_bckg_intervals(
    csv_bi_path: Path, cfg: dict
) -> list[tuple[float, float]]:
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

    result: list[tuple[float, float]] = []
    for b_start, b_end in bckg_intervals:
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
