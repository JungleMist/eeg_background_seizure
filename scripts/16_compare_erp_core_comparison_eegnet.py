#!/usr/bin/env python3
"""Train EEGNet on response-locked ERN windows from comparison recordings.

The ``raw`` directory is the only metadata source.  Every other directory
under the comparison root is treated as an EEG condition and is aligned to
the raw response events by time.  Existing branch processing is not repeated;
this script only extracts shared ERN windows and trains EEGNet per condition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config
from eeg_bg.ml.erp_eegnet import (
    classification_metrics,
    train_condition,
)


ERP_ERN_CHANNELS: tuple[str, ...] = (
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7",
    "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4",
    "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2",
)
RAW_CONDITION = "raw"
_CACHE_SCHEMA = 1
_INNER_SPLIT_SEED_OFFSET = 1_000_003


@dataclass(frozen=True)
class ComparisonDataset:
    """Concatenated response-locked windows for all eligible subjects."""

    sequences: dict[str, np.ndarray]
    y: np.ndarray
    subject_ids: np.ndarray
    samples: np.ndarray

    def matrix(self, condition: str, normalize: bool = True) -> np.ndarray:
        try:
            values = np.asarray(self.sequences[condition], dtype=np.float32)
        except KeyError as exc:
            raise ValueError(f"Unknown comparison condition: {condition}") from exc
        if not normalize:
            return values
        mean = values.mean(axis=-1, keepdims=True)
        std = values.std(axis=-1, keepdims=True)
        normalized = (values - mean) / np.maximum(std, 1e-6)
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, help="Comparison root; defaults to config data_dir."
    )
    parser.add_argument(
        "--config",
        default="configs/erp_core_comparison_eegnet.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override result directory.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Subject extraction processes."
    )
    parser.add_argument("--random-state", type=int, help="Override split seed.")
    parser.add_argument(
        "--force", action="store_true", help="Regenerate window caches and outputs."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="Optional conditions to train; raw is always included as metadata source.",
    )
    return parser.parse_args()


def _load_script10():
    path = Path(__file__).with_name("10_benchmark_erp_core_flankers.py")
    spec = importlib.util.spec_from_file_location("_comparison_erp_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ERP-CORE helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _read_comparison_recording(mne, path: Path):
    """Read a comparison branch without applying any preprocessing."""
    if path.suffix.lower() == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose=False)
    if path.suffix.lower() == ".fif":
        return mne.io.read_raw_fif(path, preload=True, verbose=False)
    raise ValueError(f"Unsupported comparison recording format: {path}")


def _subject_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("sub-"):
        return stem.split("_", 1)[0]
    raise ValueError(f"Cannot infer subject ID from comparison filename: {path.name}")


def discover_conditions(
    data_dir: Path, requested: Sequence[str] | None = None
) -> tuple[list[str], dict[str, dict[str, Path]]]:
    """Discover raw and processed files indexed by condition and subject."""
    root = Path(data_dir).expanduser().resolve()
    raw_dir = root / RAW_CONDITION
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw comparison directory: {raw_dir}")

    condition_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name != RAW_CONDITION
    )
    discovered: dict[str, dict[str, Path]] = {RAW_CONDITION: {}}
    for path in sorted(raw_dir.glob("sub-*_raw.fif")):
        discovered[RAW_CONDITION][_subject_id(path)] = path
    if not discovered[RAW_CONDITION]:
        raise FileNotFoundError(f"No raw FIF files found below {raw_dir}")

    for condition_dir in condition_dirs:
        files = sorted(
            path
            for path in condition_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".edf", ".fif"}
        )
        by_subject = {
            _subject_id(path): path for path in files if path.stem.startswith("sub-")
        }
        if by_subject:
            discovered[condition_dir.name] = by_subject

    available = sorted(discovered)
    if requested:
        selected = list(dict.fromkeys([RAW_CONDITION, *requested]))
        missing = sorted(set(selected) - set(available))
        if missing:
            raise ValueError(
                f"Unknown comparison conditions: {missing}; available: {available}"
            )
        conditions = selected
    else:
        conditions = [RAW_CONDITION, *sorted(condition for condition in available if condition != RAW_CONDITION)]
    if len(conditions) < 2:
        raise ValueError("Comparison requires raw and at least one processed condition")

    missing_subjects = {
        condition: sorted(set(discovered[RAW_CONDITION]) - set(discovered[condition]))
        for condition in conditions[1:]
        if set(discovered[RAW_CONDITION]) - set(discovered[condition])
    }
    if missing_subjects:
        raise ValueError(
            "Every selected condition must contain every raw subject; missing: "
            f"{missing_subjects}"
        )
    return conditions, {
        subject_id: {condition: discovered[condition][subject_id] for condition in conditions}
        for subject_id in sorted(discovered[RAW_CONDITION])
    }


def _source_fingerprint(paths: dict[str, Path], cfg: dict) -> str:
    payload = {
        "schema": _CACHE_SCHEMA,
        "sources": {
            condition: {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for condition, path in sorted(paths.items())
        },
        "channels": ERP_ERN_CHANNELS,
        "ern": cfg["erp_core"]["ern"],
        "artifact_threshold_uv": cfg["preprocessing"]["artifact_threshold_uv"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _cache_path(cache_root: Path, subject_id: str) -> Path:
    return Path(cache_root) / subject_id / "ern_windows.npz"


def _load_cache(
    path: Path, fingerprint: str, conditions: Sequence[str]
) -> tuple[ComparisonDataset, dict] | None:
    metadata_path = path.with_name("diagnostics.json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    diagnostics = json.loads(metadata_path.read_text(encoding="utf-8"))
    if diagnostics.get("fingerprint") != fingerprint:
        return None
    with np.load(path, allow_pickle=False) as data:
        if set(conditions) - {key.removeprefix("X_") for key in data.files if key.startswith("X_")}:
            return None
        dataset = ComparisonDataset(
            sequences={condition: data[f"X_{condition}"] for condition in conditions},
            y=data["y"],
            subject_ids=data["subject_ids"].astype(str),
            samples=data["samples"],
        )
    return dataset, diagnostics


def _save_cache(
    path: Path, dataset: ComparisonDataset, fingerprint: str, diagnostics: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{f"X_{condition}": values for condition, values in dataset.sequences.items()},
        y=dataset.y,
        subject_ids=dataset.subject_ids,
        samples=dataset.samples,
    )
    _save_json({**diagnostics, "fingerprint": fingerprint}, path.with_name("diagnostics.json"))


def _branch_events(
    samples: np.ndarray,
    raw_sfreq: float,
    raw_first_samp: int,
    branch_sfreq: float,
    branch_first_samp: int,
) -> np.ndarray:
    times = (np.asarray(samples, dtype=np.float64) - raw_first_samp) / float(raw_sfreq)
    mapped = np.rint(times * float(branch_sfreq)).astype(np.int64) + int(branch_first_samp)
    return np.column_stack(
        [mapped, np.zeros(len(mapped), dtype=np.int64), np.ones(len(mapped), dtype=np.int64)]
    )


def _window_bounds(
    events: np.ndarray, sfreq: float, tmin: float, tmax: float, n_times: int
) -> np.ndarray:
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    starts = events[:, 0] + start_offset
    stops = events[:, 0] + stop_offset
    return (starts >= 0) & (stops < n_times)


def extract_subject_windows(
    subject_id: str,
    paths: dict[str, Path],
    cfg: dict,
    cache_root: Path,
    force: bool,
) -> tuple[ComparisonDataset, dict, bool]:
    """Extract one subject's shared ERN windows across all conditions."""
    conditions = list(paths)
    fingerprint = _source_fingerprint(paths, cfg)
    cache = _cache_path(cache_root, subject_id)
    if not force:
        loaded = _load_cache(cache, fingerprint, conditions)
        if loaded is not None:
            return loaded[0], loaded[1], True

    import mne

    raws = {
        condition: _read_comparison_recording(mne, path)
        for condition, path in paths.items()
    }
    sfreqs = {condition: float(raw.info["sfreq"]) for condition, raw in raws.items()}
    if any(not np.isclose(value, sfreqs[RAW_CONDITION]) for value in sfreqs.values()):
        raise ValueError(f"Sampling-rate mismatch for {subject_id}: {sfreqs}")
    for raw in raws.values():
        missing = [channel for channel in ERP_ERN_CHANNELS if channel not in raw.ch_names]
        if missing:
            raise ValueError(f"{subject_id} missing ERP EEG channels: {missing}")

    raw = raws[RAW_CONDITION]
    raw_events, event_id = mne.events_from_annotations(raw, verbose=False)
    helpers = _load_script10()
    table = helpers.build_response_table(
        raw_events,
        event_id,
        sfreqs[RAW_CONDITION],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    spec = cfg["erp_core"]["ern"]
    tmin = float(spec["tmin"])
    tmax = float(spec["tmax"])
    branch_events = {
        condition: _branch_events(
            table["sample"].to_numpy(np.int64),
            sfreqs[RAW_CONDITION],
            raws[RAW_CONDITION].first_samp,
            sfreqs[condition],
            raws[condition].first_samp,
        )
        for condition in conditions
    }
    valid = np.ones(len(table), dtype=bool)
    for condition in conditions:
        valid &= _window_bounds(
            branch_events[condition],
            sfreqs[condition],
            tmin,
            tmax,
            raws[condition].n_times,
        )
    if not valid.any():
        raise ValueError(f"No ERN windows fit all conditions for {subject_id}")
    table = table.loc[valid].reset_index(drop=True)
    branch_events = {condition: events[valid] for condition, events in branch_events.items()}

    eeg_raws = {}
    for condition, raw_branch in raws.items():
        eeg_raws[condition] = raw_branch.copy().pick(list(ERP_ERN_CHANNELS))
        eeg_raws[condition].reorder_channels(list(ERP_ERN_CHANNELS))
    raw_probe = mne.Epochs(
        eeg_raws[RAW_CONDITION],
        branch_events[RAW_CONDITION],
        event_id={"response": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject={"eeg": float(cfg["preprocessing"]["artifact_threshold_uv"]) * 1e-6},
        preload=True,
        verbose=False,
    )
    selected = raw_probe.selection
    if len(selected) == 0:
        raise ValueError(f"Raw artifact rejection removed every ERN window for {subject_id}")
    table = table.iloc[selected].reset_index(drop=True)
    baseline = tuple(float(value) for value in spec["baseline"])
    epochs = {
        condition: mne.Epochs(
            eeg_raws[condition],
            branch_events[condition][selected],
            event_id={"response": 1},
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject_by_annotation=False,
            preload=True,
            verbose=False,
        )
        for condition in conditions
    }
    labels = (~table["correct"].to_numpy(bool)).astype(np.int8)
    samples = table["sample"].to_numpy(np.int64)
    sequences = {
        condition: np.asarray(
            epochs[condition].get_data(picks=list(ERP_ERN_CHANNELS), copy=False) * 1e6,
            dtype=np.float32,
        )
        for condition in conditions
    }
    dataset = ComparisonDataset(
        sequences=sequences,
        y=labels,
        subject_ids=np.repeat(subject_id, len(labels)),
        samples=samples,
    )
    diagnostics = {
        "subject_id": subject_id,
        "n_response_events": int(len(table)),
        "n_correct": int(np.count_nonzero(labels == 0)),
        "n_incorrect": int(np.count_nonzero(labels == 1)),
        "n_events_before_shared_rejection": int(valid.sum()),
        "n_events_after_raw_rejection": int(len(labels)),
        "dropped_for_condition_boundaries": int((~valid).sum()),
        "conditions": conditions,
        "event_window_sec": [tmin, tmax],
        "baseline_sec": list(baseline),
        "sfreq": sfreqs,
        "n_channels": len(ERP_ERN_CHANNELS),
        "n_times": int(next(iter(sequences.values())).shape[-1]),
    }
    _save_cache(cache, dataset, fingerprint, diagnostics)
    return dataset, diagnostics, False


def _extract_subject_worker(args: tuple[str, dict[str, str], dict, str, bool]) -> dict:
    subject_id, serial_paths, cfg, cache_root, force = args
    try:
        with threadpool_limits(limits=1):
            dataset, diagnostics, cached = extract_subject_windows(
                subject_id,
                {condition: Path(path) for condition, path in serial_paths.items()},
                cfg,
                Path(cache_root),
                force,
            )
    except Exception as exc:
        return {
            "subject_id": subject_id,
            "eligible": False,
            "cached": False,
            "n_trials": 0,
            "n_correct": 0,
            "n_incorrect": 0,
            "reason": str(exc),
        }
    return {
        "subject_id": subject_id,
        "eligible": True,
        "cached": cached,
        "n_trials": len(dataset.y),
        "n_correct": int(np.count_nonzero(dataset.y == 0)),
        "n_incorrect": int(np.count_nonzero(dataset.y == 1)),
        "reason": "",
        "diagnostics": diagnostics,
    }


def extract_all_subjects(
    recordings: dict[str, dict[str, Path]],
    cfg: dict,
    cache_root: Path,
    force: bool,
    workers: int,
) -> list[dict]:
    jobs = [
        (subject_id, {condition: str(path) for condition, path in paths.items()}, cfg, str(cache_root), force)
        for subject_id, paths in sorted(recordings.items())
    ]
    if workers <= 1:
        return [_extract_subject_worker(job) for job in jobs]
    rows: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")
    ) as executor:
        futures = {executor.submit(_extract_subject_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    return [rows[job[0]] for job in jobs]


def load_comparison_dataset(
    subject_ids: Sequence[str], recordings: dict[str, dict[str, Path]], cache_root: Path, cfg: dict
) -> ComparisonDataset:
    conditions = list(next(iter(recordings.values())))
    loaded: list[ComparisonDataset] = []
    for subject_id in subject_ids:
        paths = recordings[subject_id]
        cached = _load_cache(
            _cache_path(cache_root, subject_id),
            _source_fingerprint(paths, cfg),
            conditions,
        )
        if cached is None:
            raise FileNotFoundError(f"Missing or stale ERN window cache: {subject_id}")
        loaded.append(cached[0])
    return ComparisonDataset(
        sequences={
            condition: np.concatenate([item.sequences[condition] for item in loaded], axis=0)
            for condition in conditions
        },
        y=np.concatenate([item.y for item in loaded]),
        subject_ids=np.concatenate([item.subject_ids for item in loaded]),
        samples=np.concatenate([item.samples for item in loaded]),
    )


def _split_ids(subject_ids: Sequence[str], fraction: float, random_state: int) -> tuple[list[str], list[str]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("Split fraction must be in (0, 1)")
    ordered = np.asarray(sorted(set(subject_ids)), dtype=object)
    if len(ordered) < 2:
        raise ValueError("At least two subjects are required for a split")
    rng = np.random.default_rng(random_state)
    shuffled = ordered[rng.permutation(len(ordered))]
    n_holdout = min(len(ordered) - 1, max(1, int(math.ceil(len(ordered) * fraction))))
    return sorted(map(str, shuffled[n_holdout:])), sorted(map(str, shuffled[:n_holdout]))


def split_subjects_two_stage(
    subject_ids: Sequence[str], test_size: float, validation_size: float, random_state: int
) -> dict[str, list[str]]:
    train_pool, test = _split_ids(subject_ids, test_size, random_state)
    train, validation = _split_ids(
        train_pool, validation_size, random_state + _INNER_SPLIT_SEED_OFFSET
    )
    partitions = {"train": train, "validation": validation, "test": test}
    sets = {key: set(value) for key, value in partitions.items()}
    if any(
        sets[left] & sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("Subject leakage between train, validation, and test")
    if any(not value for value in partitions.values()):
        raise RuntimeError("Two-stage subject split produced an empty partition")
    return partitions


def _condition_indices(dataset: ComparisonDataset, partitions: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        split: np.flatnonzero(np.isin(dataset.subject_ids, np.asarray(subjects, dtype=str)))
        for split, subjects in partitions.items()
    }


def _subject_predictions(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grouped = (
        frame.groupby("subject_id", sort=True)
        .agg(
            n_trials=("true_label", "size"),
            true_label=("true_label", "first"),
            pred_proba=("pred_proba", "mean"),
        )
        .reset_index()
    )
    grouped["predicted_label"] = (grouped["pred_proba"] >= threshold).astype(np.int8)
    return grouped


def _project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent.parent / path).resolve() if not path.is_absolute() else path.resolve()


def run(
    config_path: str | Path = "configs/erp_core_comparison_eegnet.yaml",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int = 1,
    random_state_override: int | None = None,
    force: bool = False,
    requested_conditions: Sequence[str] | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    experiment_cfg = cfg["erp_core"]["comparison_eegnet"]
    random_state = int(
        random_state_override
        if random_state_override is not None
        else experiment_cfg.get("random_state", 42)
    )
    data_root = (
        data_dir.expanduser().resolve()
        if data_dir is not None
        else Path(cfg["erp_core"]["data_dir"]).expanduser().resolve()
    )
    out = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _project_path(experiment_cfg["output_dir"], config_path)
    )
    cache_root = _project_path(experiment_cfg["cache_subdir"], config_path)
    out.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    conditions, recordings = discover_conditions(data_root, requested_conditions)
    extraction_rows = extract_all_subjects(recordings, cfg, cache_root, force, workers)
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "diagnostics"} for row in extraction_rows]
    ).to_csv(out / "eligibility.csv", index=False)
    eligible_subjects = [
        str(row["subject_id"]) for row in extraction_rows if row["eligible"]
    ]
    if not eligible_subjects:
        raise RuntimeError("No eligible comparison subjects remain after ERN extraction")
    dataset = load_comparison_dataset(eligible_subjects, recordings, cache_root, cfg)
    partitions = split_subjects_two_stage(
        eligible_subjects,
        float(experiment_cfg.get("test_size", 0.2)),
        float(experiment_cfg.get("validation_size", 0.2)),
        random_state,
    )
    indices = _condition_indices(dataset, partitions)
    if any(len(index) == 0 for index in indices.values()):
        raise RuntimeError("Two-stage subject split produced an empty trial partition")
    if len(np.unique(dataset.y[indices["train"]])) < 2:
        raise ValueError("Training split must contain both ERN classes")
    _save_json(
        {
            "random_state": random_state,
            "test_size": float(experiment_cfg.get("test_size", 0.2)),
            "validation_size_within_train_pool": float(experiment_cfg.get("validation_size", 0.2)),
            "inner_split_seed": random_state + _INNER_SPLIT_SEED_OFFSET,
            "conditions": conditions,
            "train_subjects": partitions["train"],
            "validation_subjects": partitions["validation"],
            "test_subjects": partitions["test"],
        },
        out / "split_manifest.json",
    )
    rows = []
    prediction_frames = []
    subject_frames = []
    for index, condition in enumerate(conditions, start=1):
        print(f"[Comparison EEGNet {index}/{len(conditions)}] {condition}")
        summary = train_condition(
            condition,
            dataset,
            partitions,
            dict(experiment_cfg),
            out / "conditions" / condition,
            random_state + index,
        )
        condition_indices = indices
        for split, key in (("validation", "validation_predictions"), ("test", "test_predictions")):
            frame = summary[key].copy()
            frame["sample"] = dataset.samples[condition_indices[split]]
            prediction_frames.append(frame)
            subject = _subject_predictions(frame, summary["threshold"])
            subject["condition"] = condition
            subject["split"] = split
            subject_frames.append(subject)
        val_metrics = summary["validation_metrics"]
        test_metrics = summary["test_metrics"]
        rows.append(
            {
                "condition": condition,
                "n_channels": summary["n_channels"],
                "n_times": summary["n_times"],
                "n_train_subjects": len(partitions["train"]),
                "n_validation_subjects": len(partitions["validation"]),
                "n_test_subjects": len(partitions["test"]),
                "n_train_trials": len(indices["train"]),
                "n_validation_trials": len(indices["validation"]),
                "n_test_trials": len(indices["test"]),
                "best_epoch": summary["best_epoch"],
                "best_validation_auprc": summary["best_validation_auprc"],
                "validation_threshold": summary["threshold"],
                "validation_auprc": val_metrics["auprc"],
                **{f"test_{key}": value for key, value in test_metrics.items() if key != "confusion_matrix"},
                "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
            }
        )
    condition_metrics = pd.DataFrame(rows)
    condition_metrics.to_csv(out / "condition_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.concat(subject_frames, ignore_index=True).to_csv(out / "subject_metrics.csv", index=False)
    resolved_cfg = deepcopy(cfg)
    resolved_cfg["erp_core"]["comparison_eegnet"]["random_state"] = random_state
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    _save_json(
        {
            "conditions": conditions,
            "n_subjects": len(eligible_subjects),
            "subject_level_split": True,
            "normalization": "trial_channel_zscore",
            "event_source": "raw annotations",
            "label_definition": "incorrect=1, correct=0",
            "event_window_sec": [float(cfg["erp_core"]["ern"]["tmin"]), float(cfg["erp_core"]["ern"]["tmax"])],
            "baseline_sec": cfg["erp_core"]["ern"]["baseline"],
            "input_shapes": {
                condition: list(dataset.matrix(condition, normalize=False).shape[1:])
                for condition in conditions
            },
            "comparison_summary": condition_metrics.to_dict(orient="records"),
        },
        out / "run_summary.json",
    )
    print(f"Comparison ERP-CORE ERN EEGNet complete: {out}")
    return out


def main() -> None:
    args = _parse_args()
    run(
        args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        random_state_override=args.random_state,
        force=args.force,
        requested_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
