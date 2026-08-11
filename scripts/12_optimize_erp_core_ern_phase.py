#!/usr/bin/env python3
"""Select ECMAD phase for ERP-CORE ERN trial classification with XGBoost.

The outer split is a single deterministic subject holdout.  Phase selection,
XGBoost tuning, and decision-threshold selection use training subjects only;
test subjects are processed once, after the phase and model are frozen.
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
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config
from eeg_bg.features.extraction import (
    build_feature_names,
    extract_epoch_features_for_layout,
)


ERP_ERN_CHANNELS: tuple[str, ...] = (
    "FP1", "FP2", "F3", "F4", "F7", "F8", "FC3", "FC4", "C3", "C4",
    "P3", "P4", "P7", "P8", "O1", "O2", "Fz", "FCz", "Cz",
)
ERP_ERN_SYMMETRIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC3", "FC4"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("O1", "O2"),
)
ERP_ERN_FEATURE_NAMES: tuple[str, ...] = tuple(
    build_feature_names(ERP_ERN_CHANNELS, ERP_ERN_SYMMETRIC_PAIRS)
)
_CACHE_SCHEMA = 1


@dataclass(frozen=True)
class PhaseDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    samples: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="ERP-CORE root containing sub-*/eeg/*_task-ERN_eeg.set.",
    )
    parser.add_argument(
        "--config",
        default="configs/erp_core_flankers.yaml",
        help="YAML config path (default: configs/erp_core_flankers.yaml).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument(
        "--metric",
        choices=("auroc", "f1", "accuracy"),
        help="Primary phase/model selection metric (default: config value).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Subject processes for ECMAD/feature generation and XGBoost CPU "
            "threads per fit (default: 1)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="Override the deterministic split/search seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute feature caches and overwrite result files.",
    )
    return parser.parse_args()


def _load_script10():
    path = Path(__file__).with_name("10_benchmark_erp_core_flankers.py")
    spec = importlib.util.spec_from_file_location("_erp_core_benchmark_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ERP-CORE helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_phase_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return start + k*step below stop, followed by the exact stop value."""
    if not (0.0 <= start <= stop <= math.pi):
        raise ValueError("Phase search range must satisfy 0 <= start <= stop <= pi")
    if step <= 0:
        raise ValueError("Phase search step must be positive")
    values = np.arange(start, stop, step, dtype=np.float64)
    values = values[values < stop]
    if not len(values) or not np.isclose(values[0], start):
        values = np.insert(values, 0, start)
    if not np.isclose(values[-1], stop):
        values = np.append(values, float(stop))
    else:
        values[-1] = float(stop)
    return values


def split_subjects(
    subject_ids: list[str],
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """Make one deterministic, uniformly sampled subject holdout."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    ordered = np.asarray(sorted(set(subject_ids)), dtype=object)
    if len(ordered) < 3:
        raise ValueError("At least three eligible subjects are required")
    rng = np.random.default_rng(random_state)
    shuffled = ordered[rng.permutation(len(ordered))]
    n_test = max(1, int(math.ceil(len(ordered) * test_size)))
    test = sorted(str(value) for value in shuffled[:n_test])
    train = sorted(str(value) for value in shuffled[n_test:])
    return train, test


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def _phase_slug(phase: float) -> str:
    if np.isclose(phase, math.pi):
        return "pi"
    return f"{phase:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def _cache_fingerprint(recording: Path, phase: float, cfg: dict) -> str:
    source_files = [recording]
    sidecar = recording.with_suffix(".fdt")
    if sidecar.is_file():
        source_files.append(sidecar)
    effective_wiener = deepcopy(cfg["wiener"])
    effective_wiener["mode"] = "phasegated"
    effective_wiener["phase_gate_threshold_rad"] = float(phase)
    payload = {
        "schema": _CACHE_SCHEMA,
        "source_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in source_files
        ],
        "phase": float(phase),
        "preprocessing": cfg["preprocessing"],
        "channels": cfg["channels"],
        "wiener": effective_wiener,
        "ern": cfg["erp_core"]["ern"],
        "response_pairing_window_sec": cfg["erp_core"][
            "response_pairing_window_sec"
        ],
        "feature_channels": ERP_ERN_CHANNELS,
        "feature_pairs": ERP_ERN_SYMMETRIC_PAIRS,
        "feature_freq_band": (0.5, 30.0),
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_root: Path, subject_id: str, phase: float) -> Path:
    return cache_root / subject_id / f"phase_{_phase_slug(phase)}.npz"


def _load_cached_subject_phase(
    path: Path,
    expected_fingerprint: str,
) -> PhaseDataset | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        fingerprint = str(data["fingerprint"].item())
        if fingerprint != expected_fingerprint:
            return None
        return PhaseDataset(
            X=np.asarray(data["X"], dtype=np.float64),
            y=np.asarray(data["y"], dtype=np.int8),
            subject_ids=np.asarray(data["subject_ids"]).astype(str),
            samples=np.asarray(data["samples"], dtype=np.int64),
        )


def _save_subject_phase(
    path: Path,
    dataset: PhaseDataset,
    fingerprint: str,
    diagnostics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        X=dataset.X.astype(np.float32),
        y=dataset.y.astype(np.int8),
        subject_ids=dataset.subject_ids.astype("U"),
        samples=dataset.samples.astype(np.int64),
        fingerprint=np.asarray(fingerprint),
        diagnostics=np.asarray(json.dumps(_jsonable(diagnostics), sort_keys=True)),
    )
    temporary.replace(path)


def _raw_ern_epochs(mne, helpers, recording: Path, cfg: dict):
    original = helpers._read_recording(mne, recording)
    common = helpers._common_preprocess(original, cfg)
    events, event_id = mne.events_from_annotations(common, verbose=False)
    table = helpers.build_response_table(
        events,
        event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    epochs = helpers._make_shared_epochs(
        {"raw": common},
        table,
        cfg["erp_core"]["ern"],
        float(cfg["preprocessing"]["artifact_threshold_uv"]),
    )["raw"]
    issue = helpers._task_epoch_issue("ern", {"raw": epochs})
    if issue is not None:
        raise ValueError(issue)
    return common, table, epochs


def scan_eligible_subjects(
    recordings: list[dict[str, Path | str]],
    cfg: dict,
    helpers,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply phase-independent preprocessing/rejection before the split."""
    import mne

    eligible: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(recordings, start=1):
        subject_id = str(item["subject_id"])
        recording = Path(item["ern"])
        print(f"[eligibility {index}/{len(recordings)}] {subject_id}")
        try:
            _, _, epochs = _raw_ern_epochs(mne, helpers, recording, cfg)
            missing = [
                channel
                for channel in ERP_ERN_CHANNELS
                if channel not in epochs.ch_names
            ]
            if missing:
                raise ValueError(f"missing ERP feature channels: {missing}")
            labels = (~epochs.metadata["correct"].to_numpy(bool)).astype(np.int8)
            counts = np.bincount(labels, minlength=2)
            if np.any(counts == 0):
                raise ValueError("no usable correct/incorrect trial pair")
        except Exception as exc:  # retain a complete eligibility manifest
            rows.append(
                {
                    "subject_id": subject_id,
                    "recording": str(recording),
                    "eligible": False,
                    "n_correct": 0,
                    "n_incorrect": 0,
                    "reason": str(exc),
                }
            )
            continue
        eligible.append(subject_id)
        rows.append(
            {
                "subject_id": subject_id,
                "recording": str(recording),
                "eligible": True,
                "n_correct": int(counts[0]),
                "n_incorrect": int(counts[1]),
                "reason": "",
            }
        )
    return eligible, rows


def _epochs_to_features(epochs, subject_id: str) -> PhaseDataset:
    missing = [channel for channel in ERP_ERN_CHANNELS if channel not in epochs.ch_names]
    if missing:
        raise ValueError(f"{subject_id} is missing ERP feature channels: {missing}")
    data_uv = epochs.get_data(
        picks=list(ERP_ERN_CHANNELS), copy=False
    ) * 1e6
    labels = (~epochs.metadata["correct"].to_numpy(bool)).astype(np.int8)
    nperseg = min(250, data_uv.shape[-1])
    rows = [
        extract_epoch_features_for_layout(
            epoch,
            list(ERP_ERN_CHANNELS),
            float(epochs.info["sfreq"]),
            channel_order=ERP_ERN_CHANNELS,
            symmetric_pairs=ERP_ERN_SYMMETRIC_PAIRS,
            nperseg=nperseg,
            freq_band=(0.5, 30.0),
        )
        for epoch in data_uv
    ]
    X = np.nan_to_num(
        np.asarray(rows, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if X.shape[1] != len(ERP_ERN_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(ERP_ERN_FEATURE_NAMES)} features, got {X.shape[1]}"
        )
    return PhaseDataset(
        X=X,
        y=labels,
        subject_ids=np.repeat(subject_id, len(labels)),
        samples=np.asarray(epochs.events[:, 0], dtype=np.int64),
    )


def extract_subject_phases(
    subject_id: str,
    recording: Path,
    phases: np.ndarray,
    cfg: dict,
    cache_root: Path,
    helpers,
    force: bool,
) -> None:
    """Extract all missing phase caches while loading one subject only once."""
    fingerprints = {
        float(phase): _cache_fingerprint(recording, float(phase), cfg)
        for phase in phases
    }
    missing_phases = [
        float(phase)
        for phase in phases
        if force
        or _load_cached_subject_phase(
            _cache_path(cache_root, subject_id, float(phase)),
            fingerprints[float(phase)],
        )
        is None
    ]
    if not missing_phases:
        print(f"  {subject_id}: all requested phase features cached")
        return

    import mne

    common, table, raw_epochs = _raw_ern_epochs(
        mne, helpers, recording, cfg
    )
    raw_labels = (~raw_epochs.metadata["correct"].to_numpy(bool)).astype(np.int8)
    raw_samples = np.asarray(raw_epochs.events[:, 0], dtype=np.int64)
    for index, phase in enumerate(missing_phases, start=1):
        print(
            f"  {subject_id}: phase {phase:.6f} "
            f"({index}/{len(missing_phases)})"
        )
        phase_cfg = deepcopy(cfg)
        phase_cfg["wiener"]["mode"] = "phasegated"
        phase_cfg["wiener"]["phase_gate_threshold_rad"] = float(phase)
        processed, diagnostics = helpers._wiener_continuous(
            common, phase_cfg, subject_id
        )
        phase_epochs = helpers._make_shared_epochs(
            {"raw": common, "wiener": processed},
            table,
            phase_cfg["erp_core"]["ern"],
            float(phase_cfg["preprocessing"]["artifact_threshold_uv"]),
        )["wiener"]
        dataset = _epochs_to_features(phase_epochs, subject_id)
        if not np.array_equal(dataset.y, raw_labels) or not np.array_equal(
            dataset.samples, raw_samples
        ):
            raise RuntimeError(
                f"{subject_id} phase {phase} changed the shared trial selection"
            )
        _save_subject_phase(
            _cache_path(cache_root, subject_id, phase),
            dataset,
            fingerprints[phase],
            diagnostics,
        )


def _extract_subject_phases_worker(
    args: tuple[str, str, tuple[float, ...], dict, str, bool],
    helpers: Any | None = None,
) -> str:
    """Process all requested phases for one subject in a child process."""
    subject_id, recording, phases, cfg, cache_root, force = args
    if helpers is None:
        helpers = _load_script10()
    with threadpool_limits(limits=1):
        extract_subject_phases(
            subject_id,
            Path(recording),
            np.asarray(phases, dtype=np.float64),
            cfg,
            Path(cache_root),
            helpers,
            force,
        )
    return subject_id


def extract_subjects_phases(
    subject_ids: list[str],
    recordings_by_subject: dict[str, Path],
    phases: np.ndarray,
    cfg: dict,
    cache_root: Path,
    force: bool,
    workers: int,
    stage: str,
) -> None:
    """Generate subject/phase caches with subject-level process parallelism."""
    jobs = [
        (
            subject_id,
            str(recordings_by_subject[subject_id]),
            tuple(float(phase) for phase in phases),
            cfg,
            str(cache_root),
            force,
        )
        for subject_id in subject_ids
    ]
    if not jobs:
        return

    if workers == 1:
        helpers = _load_script10()
        for index, job in enumerate(jobs, start=1):
            subject_id = job[0]
            print(f"[{stage} features {index}/{len(jobs)}] {subject_id}")
            try:
                _extract_subject_phases_worker(job, helpers)
            except Exception as exc:
                raise RuntimeError(
                    f"{stage} feature generation failed for {subject_id}"
                ) from exc
        return

    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=get_context("spawn"),
    )
    futures = {}
    try:
        for job in jobs:
            future = executor.submit(_extract_subject_phases_worker, job)
            futures[future] = job[0]
        completed = 0
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"{stage} feature generation failed for {subject_id}"
                ) from exc
            completed += 1
            print(f"[{stage} features {completed}/{len(jobs)}] {subject_id}")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def load_phase_dataset(
    subject_ids: list[str],
    recordings_by_subject: dict[str, Path],
    phase: float,
    cfg: dict,
    cache_root: Path,
) -> PhaseDataset:
    datasets: list[PhaseDataset] = []
    for subject_id in subject_ids:
        path = _cache_path(cache_root, subject_id, phase)
        fingerprint = _cache_fingerprint(
            recordings_by_subject[subject_id], phase, cfg
        )
        dataset = _load_cached_subject_phase(path, fingerprint)
        if dataset is None:
            raise FileNotFoundError(
                f"Missing or stale phase cache for {subject_id}, phase={phase}: {path}"
            )
        datasets.append(dataset)
    return PhaseDataset(
        X=np.concatenate([dataset.X for dataset in datasets]),
        y=np.concatenate([dataset.y for dataset in datasets]),
        subject_ids=np.concatenate(
            [dataset.subject_ids for dataset in datasets]
        ),
        samples=np.concatenate([dataset.samples for dataset in datasets]),
    )


def make_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(set(groups)) < n_splits:
        raise ValueError(
            f"Internal CV needs at least {n_splits} training subjects; "
            f"found {len(set(groups))}"
        )
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    for train_index, val_index in folds:
        overlap = set(groups[train_index]) & set(groups[val_index])
        if overlap:
            raise RuntimeError(f"Subject leakage inside CV: {sorted(overlap)}")
        if len(np.unique(y[val_index])) < 2:
            raise ValueError("Every internal validation fold must contain both classes")
    return folds


def _xgb_params(
    params: dict[str, Any],
    y_train: np.ndarray,
    random_state: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    counts = np.bincount(y_train.astype(int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Each XGBoost training fold must contain both classes")
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "verbosity": 0,
        "n_jobs": workers,
        "device": device,
        "scale_pos_weight": float(counts[0]) / float(counts[1]),
        **params,
    }


def oof_predict(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    random_state: int,
    workers: int,
    device: str,
) -> np.ndarray:
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    for train_index, val_index in folds:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_index])
        X_val = scaler.transform(X[val_index])
        model = xgb.XGBClassifier(
            **_xgb_params(
                params,
                y[train_index],
                random_state,
                workers,
                device,
            )
        )
        model.fit(X_train, y[train_index], verbose=False)
        probabilities[val_index] = model.predict_proba(X_val)[:, 1]
    if np.isnan(probabilities).any():
        raise RuntimeError("OOF prediction did not cover every training trial")
    return probabilities


def best_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
    metric: str,
) -> float:
    if metric not in {"f1", "accuracy"}:
        metric = "f1"
    best_value = -math.inf
    selected = 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int8)
        value = (
            f1_score(y, predictions, zero_division=0)
            if metric == "f1"
            else accuracy_score(y, predictions)
        )
        if value > best_value:
            best_value = float(value)
            selected = float(threshold)
    return selected


def classification_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "threshold": float(threshold),
    }


def score_oof(
    y: np.ndarray,
    probabilities: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    metric: str,
) -> dict[str, Any]:
    threshold = best_threshold(y, probabilities, metric)
    pooled = classification_metrics(y, probabilities, threshold)
    fold_values: list[float] = []
    for _, val_index in folds:
        if metric == "auroc":
            value = roc_auc_score(y[val_index], probabilities[val_index])
        elif metric == "f1":
            predictions = (probabilities[val_index] >= threshold).astype(np.int8)
            value = f1_score(y[val_index], predictions, zero_division=0)
        else:
            predictions = (probabilities[val_index] >= threshold).astype(np.int8)
            value = accuracy_score(y[val_index], predictions)
        fold_values.append(float(value))
    return {
        **pooled,
        "primary_metric": metric,
        "primary_mean": float(np.mean(fold_values)),
        "primary_std": float(np.std(fold_values, ddof=1)),
        "fold_scores": fold_values,
    }


def select_best_row(frame: pd.DataFrame) -> pd.Series:
    """Select highest mean, then lowest standard deviation, then phase."""
    return frame.sort_values(
        ["primary_mean", "primary_std", "phase_rad"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


def _baseline_params(search_cfg: dict) -> dict[str, Any]:
    return {
        key: value
        for key, value in search_cfg["baseline_xgboost"].items()
    }


def run_phase_search(
    phases: np.ndarray,
    train_subjects: list[str],
    recordings_by_subject: dict[str, Path],
    cfg: dict,
    cache_root: Path,
    folds: list[tuple[np.ndarray, np.ndarray]],
    metric: str,
    random_state: int,
    workers: int,
    device: str,
) -> tuple[pd.DataFrame, PhaseDataset]:
    rows: list[dict[str, Any]] = []
    reference: PhaseDataset | None = None
    params = _baseline_params(cfg["erp_core"]["phase_search"])
    for index, phase in enumerate(phases, start=1):
        print(f"[phase CV {index}/{len(phases)}] phase={phase:.6f}")
        dataset = load_phase_dataset(
            train_subjects,
            recordings_by_subject,
            float(phase),
            cfg,
            cache_root,
        )
        if reference is None:
            reference = dataset
        elif not (
            np.array_equal(dataset.y, reference.y)
            and np.array_equal(dataset.subject_ids, reference.subject_ids)
            and np.array_equal(dataset.samples, reference.samples)
        ):
            raise RuntimeError("Phase feature caches do not share identical trials")
        probabilities = oof_predict(
            dataset.X,
            dataset.y,
            folds,
            params,
            random_state,
            workers,
            device,
        )
        scores = score_oof(dataset.y, probabilities, folds, metric)
        rows.append(
            {
                "phase_rad": float(phase),
                "primary_metric": metric,
                "primary_mean": scores["primary_mean"],
                "primary_std": scores["primary_std"],
                "oof_auroc": scores["auroc"],
                "oof_f1": scores["f1"],
                "oof_accuracy": scores["accuracy"],
                "oof_threshold": scores["threshold"],
                **{
                    f"fold_{fold + 1}": value
                    for fold, value in enumerate(scores["fold_scores"])
                },
            }
        )
    assert reference is not None
    return pd.DataFrame(rows), reference


def tune_xgboost(
    dataset: PhaseDataset,
    folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict,
    metric: str,
    random_state: int,
    workers: int,
    device: str,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, dict[str, Any]]:
    param_grid = dict(cfg["ml"]["xgboost"]["param_grid"])
    n_iter = int(cfg["erp_core"]["phase_search"]["random_search_iterations"])
    candidates = list(
        ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)
    )
    rows: list[dict[str, Any]] = []
    probability_cache: list[np.ndarray] = []
    score_cache: list[dict[str, Any]] = []
    for index, params in enumerate(candidates, start=1):
        print(f"[XGBoost tuning {index}/{len(candidates)}]")
        probabilities = oof_predict(
            dataset.X,
            dataset.y,
            folds,
            params,
            random_state,
            workers,
            device,
        )
        scores = score_oof(dataset.y, probabilities, folds, metric)
        probability_cache.append(probabilities)
        score_cache.append(scores)
        rows.append(
            {
                "candidate": index,
                **_jsonable(params),
                "primary_metric": metric,
                "primary_mean": scores["primary_mean"],
                "primary_std": scores["primary_std"],
                "oof_auroc": scores["auroc"],
                "oof_f1": scores["f1"],
                "oof_accuracy": scores["accuracy"],
                "oof_threshold": scores["threshold"],
            }
        )
    frame = pd.DataFrame(rows)
    best_index = int(
        frame.sort_values(
            ["primary_mean", "primary_std", "candidate"],
            ascending=[False, True, True],
            kind="mergesort",
        ).index[0]
    )
    return (
        frame,
        _jsonable(candidates[best_index]),
        probability_cache[best_index],
        score_cache[best_index],
    )


def fit_final_model(
    dataset: PhaseDataset,
    params: dict[str, Any],
    random_state: int,
    workers: int,
    device: str,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataset.X)
    model = xgb.XGBClassifier(
        **_xgb_params(params, dataset.y, random_state, workers, device)
    )
    model.fit(X_scaled, dataset.y, verbose=False)
    return scaler, model


def predictions_frame(
    dataset: PhaseDataset,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": dataset.subject_ids,
            "sample": dataset.samples,
            "true_label": dataset.y,
            "pred_proba": probabilities,
            "predicted_label": (probabilities >= threshold).astype(np.int8),
        }
    )


def subject_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, group in predictions.groupby("subject_id", sort=True):
        scores = classification_metrics(
            group["true_label"].to_numpy(),
            group["pred_proba"].to_numpy(),
            threshold,
        )
        rows.append(
            {
                "subject_id": subject_id,
                "n_trials": len(group),
                "n_incorrect": int(group["true_label"].sum()),
                **scores,
            }
        )
    return pd.DataFrame(rows)


def plot_phase_search(frame: pd.DataFrame, best_phase: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.errorbar(
        frame["phase_rad"],
        frame["primary_mean"],
        yerr=frame["primary_std"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        capsize=2,
    )
    ax.axvline(best_phase, color="#D95F02", linestyle="--", label=f"best={best_phase:.4f}")
    ax.set(
        xlabel="Phase gate threshold (rad)",
        ylabel=f"Internal CV {frame['primary_metric'].iloc[0].upper()}",
        title="ECMAD phase search on training subjects",
    )
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    config_path: str | Path,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    metric_override: str | None = None,
    workers: int = 1,
    random_state_override: int | None = None,
    force: bool = False,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    config_path = Path(config_path)
    cfg = load_config(config_path)
    search_cfg = cfg["erp_core"]["phase_search"]
    metric = metric_override or str(search_cfg["metric"])
    if metric not in {"auroc", "f1", "accuracy"}:
        raise ValueError("phase_search.metric must be auroc, f1, or accuracy")
    random_state = (
        int(random_state_override)
        if random_state_override is not None
        else int(search_cfg["random_state"])
    )
    device = str(search_cfg.get("device", "cpu"))
    resolved_cfg = deepcopy(cfg)
    resolved_cfg["erp_core"]["phase_search"]["metric"] = metric
    resolved_cfg["erp_core"]["phase_search"]["random_state"] = random_state
    out = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _project_path(search_cfg["output_dir"], config_path)
    )
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {out}; pass --force to overwrite files"
        )
    out.mkdir(parents=True, exist_ok=True)
    cache_root = (
        Path(cfg["paths"]["cache_dir"])
        / str(search_cfg.get("cache_subdir", "erp_core_ern_phase_search"))
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    helpers = _load_script10()
    source_root = (
        data_dir.expanduser().resolve()
        if data_dir is not None
        else Path(cfg["erp_core"]["data_dir"]).expanduser().resolve()
    )
    recordings = helpers._select_recordings(
        helpers._resolve_recordings(source_root, None),
        subjects=None,
        task="ern",
    )
    recordings_by_subject = {
        str(item["subject_id"]): Path(item["ern"]) for item in recordings
    }
    eligible, eligibility_rows = scan_eligible_subjects(recordings, cfg, helpers)
    eligibility = pd.DataFrame(eligibility_rows)
    eligibility.to_csv(out / "eligibility.csv", index=False)

    cv_folds = int(search_cfg["cv_folds"])
    train_subjects, test_subjects = split_subjects(
        eligible,
        float(search_cfg["test_size"]),
        random_state,
    )
    if len(train_subjects) < cv_folds:
        raise ValueError(
            f"Need at least {cv_folds} eligible training subjects; "
            f"found {len(train_subjects)}"
        )
    split_manifest = {
        "random_state": random_state,
        "test_size": float(search_cfg["test_size"]),
        "eligible_subjects": eligible,
        "excluded_subjects": eligibility.loc[
            ~eligibility["eligible"], "subject_id"
        ].tolist(),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
    }
    _save_json(split_manifest, out / "split_manifest.json")

    phases = build_phase_grid(
        float(search_cfg["phase_start"]),
        float(search_cfg["phase_stop"]),
        float(search_cfg["phase_step"]),
    )
    print(
        f"Eligible subjects: {len(eligible)}; train={len(train_subjects)}, "
        f"test={len(test_subjects)}; phases={len(phases)}"
    )
    extract_subjects_phases(
        train_subjects,
        recordings_by_subject,
        phases,
        cfg,
        cache_root,
        force,
        workers,
        "training",
    )

    reference = load_phase_dataset(
        train_subjects,
        recordings_by_subject,
        float(phases[0]),
        cfg,
        cache_root,
    )
    folds = make_group_folds(
        reference.y,
        reference.subject_ids,
        cv_folds,
        random_state,
    )
    phase_results, _ = run_phase_search(
        phases,
        train_subjects,
        recordings_by_subject,
        cfg,
        cache_root,
        folds,
        metric,
        random_state,
        workers,
        device,
    )
    phase_results.to_csv(out / "phase_cv_results.csv", index=False)
    best_phase_row = select_best_row(phase_results)
    best_phase = float(best_phase_row["phase_rad"])
    best_phase_dataset = load_phase_dataset(
        train_subjects,
        recordings_by_subject,
        best_phase,
        cfg,
        cache_root,
    )
    plot_phase_search(phase_results, best_phase, out / "phase_search_curve.png")

    tuning_results, best_params, train_oof, train_scores = tune_xgboost(
        best_phase_dataset,
        folds,
        cfg,
        metric,
        random_state,
        workers,
        device,
    )
    tuning_results.to_csv(out / "xgb_search_results.csv", index=False)
    threshold = float(train_scores["threshold"])
    train_predictions = predictions_frame(
        best_phase_dataset, train_oof, threshold
    )
    train_predictions.to_csv(out / "train_oof_predictions.csv", index=False)
    scaler, model = fit_final_model(
        best_phase_dataset,
        best_params,
        random_state,
        workers,
        device,
    )
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(model, out / "model.joblib")
    _save_json(best_params, out / "best_params.json")

    # The held-out subjects are first ECMAD-processed here, after all choices freeze.
    extract_subjects_phases(
        test_subjects,
        recordings_by_subject,
        np.asarray([best_phase]),
        cfg,
        cache_root,
        force,
        workers,
        "test",
    )
    test_dataset = load_phase_dataset(
        test_subjects,
        recordings_by_subject,
        best_phase,
        cfg,
        cache_root,
    )
    test_probabilities = model.predict_proba(
        scaler.transform(test_dataset.X)
    )[:, 1]
    test_predictions = predictions_frame(
        test_dataset, test_probabilities, threshold
    )
    test_predictions.to_csv(out / "test_predictions.csv", index=False)
    by_subject = subject_metrics(test_predictions, threshold)
    by_subject.to_csv(out / "subject_test_metrics.csv", index=False)
    test_scores = classification_metrics(
        test_dataset.y, test_probabilities, threshold
    )
    test_scores["confusion_matrix"] = confusion_matrix(
        test_dataset.y,
        (test_probabilities >= threshold).astype(np.int8),
        labels=[0, 1],
    ).tolist()
    test_scores["n_subjects"] = len(test_subjects)
    test_scores["n_trials"] = len(test_dataset.y)
    _save_json(test_scores, out / "test_metrics.json")

    best_phase_payload = {
        "phase_rad": best_phase,
        "primary_metric": metric,
        "phase_cv": best_phase_row.to_dict(),
        "final_training_oof": train_scores,
        "decision_threshold": threshold,
        "tie_break": "higher mean, then lower fold SD, then lower phase",
    }
    _save_json(best_phase_payload, out / "best_phase.json")
    summary = {
        "input_data_dir": str(source_root),
        "n_discovered_subjects": len(recordings),
        "n_eligible_subjects": len(eligible),
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "phase_values_rad": phases.tolist(),
        "best_phase_rad": best_phase,
        "primary_metric": metric,
        "protected_band_hz": resolved_cfg["wiener"].get(
            "protected_band_hz"
        ),
        "coherent_gate_enabled": bool(resolved_cfg["wiener"].get(
            "coherent_gate_enabled", True
        )),
        "coherent_gate_threshold_uv": float(resolved_cfg["wiener"].get(
            "coherent_gate_threshold_uv", 100.0
        )),
        "xgboost_device": device,
        "feature_worker_processes": workers,
        "xgboost_workers": workers,
        "feature_parallelism": (
            "subject process pool; phases and trials are sequential within each "
            "subject; native worker threads are limited to one"
        ),
        "test_metrics": test_scores,
        "feature_count": len(ERP_ERN_FEATURE_NAMES),
        "feature_channels": ERP_ERN_CHANNELS,
        "symmetric_pairs": ERP_ERN_SYMMETRIC_PAIRS,
        "label_encoding": {"correct": 0, "incorrect": 1},
        "test_policy": (
            "Test subjects were processed only after phase, model parameters, "
            "and decision threshold were selected from training subjects."
        ),
    }
    _save_json(summary, out / "run_summary.json")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        f"Best phase={best_phase:.6f} rad; test AUROC={test_scores['auroc']:.3f}, "
        f"F1={test_scores['f1']:.3f}, accuracy={test_scores['accuracy']:.3f}"
    )
    return out


def main() -> None:
    args = _parse_args()
    output = run(
        args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metric_override=args.metric,
        workers=args.workers,
        random_state_override=args.random_state,
        force=args.force,
    )
    print(f"ERP-CORE ERN phase optimization complete: {output}")


if __name__ == "__main__":
    main()
