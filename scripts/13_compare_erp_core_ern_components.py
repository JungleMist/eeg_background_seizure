#!/usr/bin/env python3
"""Compare serial ERP-CORE ERN ECMAD components with XGBoost and SVM.

Every condition uses the same response-locked trials and one deterministic
subject-disjoint train/test split. Three ECMAD steps run in series, with each
specific output feeding the next step and each coherent condition representing
only the component separated at that step. By default both XGBoost and a fixed
RBF SVM are trained. The ``raw`` condition means common preprocessing without
ECMAD denoising; it is not the unfiltered EEGLAB recording.
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
import shutil
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

STEP_NAMES: tuple[str, ...] = ("step1", "step2", "step3")
COMPONENT_NAMES: tuple[str, ...] = tuple(
    f"{step}_{component}"
    for step in STEP_NAMES
    for component in ("specific", "coherent")
)
CONDITIONS: tuple[str, ...] = ("raw", *COMPONENT_NAMES)
MODEL_NAMES: tuple[str, ...] = ("xgboost", "svm")
_CACHE_SCHEMA = 2
_METRIC_NAMES: tuple[str, ...] = (
    "auroc",
    "auprc",
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "accuracy",
)


@dataclass(frozen=True)
class FeatureLayout:
    channels: tuple[str, ...]
    symmetric_pairs: tuple[tuple[str, str], ...]
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ComponentDataset:
    features: dict[str, np.ndarray]
    y: np.ndarray
    subject_ids: np.ndarray
    samples: np.ndarray

    def matrix(self, condition: str) -> np.ndarray:
        try:
            return self.features[condition]
        except KeyError as exc:
            raise ValueError(f"Unknown condition: {condition}") from exc


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
        "--workers",
        type=int,
        default=1,
        help=(
            "Subject processes and model-training CPU workers. With GridSearchCV, "
            "workers parallelize candidates and each fit uses one thread "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="Override the deterministic split/GridSearchCV seed.",
    )
    parser.add_argument(
        "--model",
        choices=("both", *MODEL_NAMES),
        default="both",
        help="Train both model families or only one (default: both).",
    )
    parser.add_argument(
        "--grid-search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable training-only XGBoost GridSearchCV. The config "
            "value erp_core.distributed_component_models.xgboost."
            "grid_search_enabled is used when omitted (default: disabled)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute feature caches and overwrite result files.",
    )
    parser.add_argument(
        "--recompute-components",
        action="store_true",
        help="Ignore valid shared continuous-component caches and regenerate them.",
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


def _load_script15():
    path = Path(__file__).with_name(
        "15_compare_erp_core_ern_distributed_components_eegnet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_erp_distributed_component_helpers", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load distributed ECMAD helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def _feature_layout(channels: Sequence[str]) -> FeatureLayout:
    ordered = tuple(channel for channel in ERP_ERN_CHANNELS if channel in channels)
    channel_set = set(ordered)
    pairs = tuple(
        pair
        for pair in ERP_ERN_SYMMETRIC_PAIRS
        if pair[0] in channel_set and pair[1] in channel_set
    )
    return FeatureLayout(
        channels=ordered,
        symmetric_pairs=pairs,
        feature_names=tuple(build_feature_names(ordered, pairs)),
    )


def build_feature_layouts(cfg: dict) -> dict[str, FeatureLayout]:
    full_names = tuple(
        build_feature_names(ERP_ERN_CHANNELS, ERP_ERN_SYMMETRIC_PAIRS)
    )
    full = FeatureLayout(
        channels=ERP_ERN_CHANNELS,
        symmetric_pairs=ERP_ERN_SYMMETRIC_PAIRS,
        feature_names=full_names,
    )
    try:
        steps = cfg["erp_core"]["distributed_components"]["steps"]
    except KeyError as exc:
        raise KeyError("Missing erp_core.distributed_components.steps") from exc
    if tuple(steps) != STEP_NAMES:
        raise ValueError(f"Distributed ECMAD steps must be ordered as {STEP_NAMES}")
    layouts = {"raw": full}
    for step_name in STEP_NAMES:
        grouped = {
            str(channel)
            for group in steps[step_name]["channel_groups"]
            for channel in group
        }
        coherent = _feature_layout(grouped)
        if not coherent.channels:
            raise ValueError(f"{step_name} has no ERP ERN feature channels")
        layouts[f"{step_name}_specific"] = full
        layouts[f"{step_name}_coherent"] = coherent
    return layouts


def condition_feature_names(
    layouts: dict[str, FeatureLayout],
) -> dict[str, tuple[str, ...]]:
    return {condition: layouts[condition].feature_names for condition in CONDITIONS}


def _source_files(recording: Path) -> list[Path]:
    paths = [recording]
    sidecar = recording.with_suffix(".fdt")
    if sidecar.is_file():
        paths.append(sidecar)
    return paths


def _cache_fingerprint(
    recording: Path,
    cfg: dict,
    layouts: dict[str, FeatureLayout],
) -> str:
    payload = {
        "schema": _CACHE_SCHEMA,
        "source_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in _source_files(recording)
        ],
        "preprocessing": cfg["preprocessing"],
        "wiener": cfg["wiener"],
        "distributed_components": cfg["erp_core"]["distributed_components"],
        "line_freq": cfg["erp_core"].get("line_freq"),
        "ern": cfg["erp_core"]["ern"],
        "response_pairing_window_sec": cfg["erp_core"][
            "response_pairing_window_sec"
        ],
        "feature_freq_band": (0.5, 30.0),
        "feature_layouts": {
            condition: {
                "channels": layout.channels,
                "symmetric_pairs": layout.symmetric_pairs,
                "feature_names": layout.feature_names,
            }
            for condition, layout in layouts.items()
        },
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_root: Path, subject_id: str) -> Path:
    return cache_root / subject_id / "features.npz"


def _load_subject_cache(
    path: Path,
    expected_fingerprint: str,
) -> tuple[ComponentDataset, dict] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"].item()) != expected_fingerprint:
            return None
        features = {
            condition: np.asarray(data[f"X_{condition}"], dtype=np.float64)
            for condition in CONDITIONS
        }
        dataset = ComponentDataset(
            features=features,
            y=np.asarray(data["y"], dtype=np.int8),
            subject_ids=np.asarray(data["subject_ids"]).astype(str),
            samples=np.asarray(data["samples"], dtype=np.int64),
        )
        diagnostics = json.loads(str(data["diagnostics"].item()))
    return dataset, diagnostics


def _save_subject_cache(
    path: Path,
    dataset: ComponentDataset,
    fingerprint: str,
    diagnostics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    arrays: dict[str, Any] = {
        f"X_{condition}": dataset.features[condition].astype(np.float32)
        for condition in CONDITIONS
    }
    np.savez_compressed(
        temporary,
        **arrays,
        y=dataset.y.astype(np.int8),
        subject_ids=dataset.subject_ids.astype("U"),
        samples=dataset.samples.astype(np.int64),
        fingerprint=np.asarray(fingerprint),
        diagnostics=np.asarray(json.dumps(_jsonable(diagnostics), sort_keys=True)),
    )
    temporary.replace(path)


def _epochs_to_features(
    epochs,
    layout: FeatureLayout,
) -> np.ndarray:
    missing = [channel for channel in layout.channels if channel not in epochs.ch_names]
    if missing:
        raise ValueError(f"Missing ERP feature channels: {missing}")
    data_uv = epochs.get_data(picks=list(layout.channels), copy=False) * 1e6
    nperseg = min(250, data_uv.shape[-1])
    rows = [
        extract_epoch_features_for_layout(
            epoch,
            list(layout.channels),
            float(epochs.info["sfreq"]),
            channel_order=layout.channels,
            symmetric_pairs=layout.symmetric_pairs,
            nperseg=nperseg,
            freq_band=(0.5, 30.0),
        )
        for epoch in data_uv
    ]
    X = np.nan_to_num(
        np.asarray(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    expected = len(layout.feature_names)
    if X.ndim != 2 or X.shape[1] != expected:
        raise ValueError(f"Expected {expected} features, got shape {X.shape}")
    return X


def extract_subject_components(
    subject_id: str,
    recording: Path,
    cfg: dict,
    cache_root: Path,
    helpers,
    force: bool,
    distributed: Any | None = None,
    recompute_components: bool = False,
) -> tuple[ComponentDataset, dict, bool]:
    layouts = build_feature_layouts(cfg)
    fingerprint = _cache_fingerprint(recording, cfg, layouts)
    path = _cache_path(cache_root, subject_id)
    if not force and not recompute_components:
        cached = _load_subject_cache(path, fingerprint)
        if cached is not None:
            dataset, diagnostics = cached
            return dataset, diagnostics, True

    import mne

    if distributed is None:
        distributed = _load_script15()
    original = helpers._read_recording(mne, recording)
    common = helpers._common_preprocess(original, cfg)
    missing = [
        channel
        for channel in distributed.ERP_CORE_EEG_CHANNELS
        if channel not in common.ch_names
    ]
    if missing:
        raise ValueError(f"Missing ERP-CORE 30-channel input: {missing}")
    common.pick(list(distributed.ERP_CORE_EEG_CHANNELS))
    common.reorder_channels(list(distributed.ERP_CORE_EEG_CHANNELS))
    events, event_id = mne.events_from_annotations(common, verbose=False)
    table = helpers.build_response_table(
        events,
        event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )

    branches, component_diagnostics, components_cached = (
        distributed.load_or_create_distributed_components(
            common,
            recording,
            cfg,
            subject_id,
            helpers,
            cache_root,
            recompute_components,
        )
    )
    epochs = helpers._make_shared_epochs(
        branches,
        table,
        cfg["erp_core"]["ern"],
        float(cfg["preprocessing"]["artifact_threshold_uv"]),
    )
    issue = helpers._task_epoch_issue("ern", epochs)
    if issue is not None:
        raise ValueError(issue)

    labels = (~epochs["raw"].metadata["correct"].to_numpy(bool)).astype(np.int8)
    samples = np.asarray(epochs["raw"].events[:, 0], dtype=np.int64)
    features: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        features[condition] = _epochs_to_features(
            epochs[condition], layouts[condition]
        )
        branch_labels = (
            ~epochs[condition].metadata["correct"].to_numpy(bool)
        ).astype(np.int8)
        branch_samples = np.asarray(epochs[condition].events[:, 0], dtype=np.int64)
        if not np.array_equal(branch_labels, labels) or not np.array_equal(
            branch_samples, samples
        ):
            raise RuntimeError(f"{condition} changed shared ERN trial selection")

    dataset = ComponentDataset(
        features=features,
        y=labels,
        subject_ids=np.repeat(subject_id, len(labels)),
        samples=samples,
    )
    counts = np.bincount(labels, minlength=2)
    diagnostics = {
        **component_diagnostics,
        "components_cached": components_cached,
        "n_trials": len(labels),
        "n_correct": int(counts[0]),
        "n_incorrect": int(counts[1]),
    }
    _save_subject_cache(path, dataset, fingerprint, diagnostics)
    return dataset, diagnostics, False


def _extract_subject_worker(
    args: tuple[str, str, dict, str, bool, bool],
    helpers: Any | None = None,
    distributed: Any | None = None,
) -> dict:
    subject_id, recording, cfg, cache_root, force, recompute_components = args
    if helpers is None:
        helpers = _load_script10()
    if distributed is None:
        distributed = _load_script15()
    with threadpool_limits(limits=1):
        try:
            dataset, diagnostics, cached = extract_subject_components(
                subject_id,
                Path(recording),
                cfg,
                Path(cache_root),
                helpers,
                force,
                distributed,
                recompute_components,
            )
        except Exception as exc:
            return {
                "subject_id": subject_id,
                "recording": recording,
                "eligible": False,
                "cached": False,
                "n_trials": 0,
                "n_correct": 0,
                "n_incorrect": 0,
                "reason": str(exc),
            }
    counts = np.bincount(dataset.y, minlength=2)
    return {
        "subject_id": subject_id,
        "recording": recording,
        "eligible": True,
        "cached": cached,
        "n_trials": len(dataset.y),
        "n_correct": int(counts[0]),
        "n_incorrect": int(counts[1]),
        "reason": "",
        "diagnostics": diagnostics,
    }


def extract_all_subjects(
    recordings: list[dict[str, Path | str]],
    cfg: dict,
    cache_root: Path,
    force: bool,
    recompute_components: bool,
    workers: int,
) -> list[dict]:
    jobs = [
        (
            str(item["subject_id"]),
            str(item["ern"]),
            cfg,
            str(cache_root),
            force,
            recompute_components,
        )
        for item in recordings
    ]
    if not jobs:
        return []
    if workers == 1:
        helpers = _load_script10()
        distributed = _load_script15()
        rows = []
        for index, job in enumerate(jobs, start=1):
            print(f"[features {index}/{len(jobs)}] {job[0]}")
            rows.append(_extract_subject_worker(job, helpers, distributed))
        return rows

    rows_by_subject: dict[str, dict] = {}
    executor = ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=get_context("spawn"),
    )
    futures = {}
    try:
        for job in jobs:
            future = executor.submit(_extract_subject_worker, job)
            futures[future] = job[0]
        completed = 0
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                rows_by_subject[subject_id] = future.result()
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(
                    f"Feature generation failed for {subject_id}"
                ) from exc
            completed += 1
            print(f"[features {completed}/{len(jobs)}] {subject_id}")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [rows_by_subject[job[0]] for job in jobs]


def load_component_dataset(
    subject_ids: Sequence[str],
    recordings_by_subject: dict[str, Path],
    cfg: dict,
    cache_root: Path,
) -> ComponentDataset:
    layouts = build_feature_layouts(cfg)
    datasets: list[ComponentDataset] = []
    for subject_id in subject_ids:
        fingerprint = _cache_fingerprint(recordings_by_subject[subject_id], cfg, layouts)
        cached = _load_subject_cache(
            _cache_path(cache_root, subject_id), fingerprint
        )
        if cached is None:
            raise FileNotFoundError(f"Missing or stale component cache: {subject_id}")
        datasets.append(cached[0])
    if not datasets:
        raise ValueError("No eligible subject caches were loaded")
    return ComponentDataset(
        features={
            condition: np.concatenate(
                [dataset.features[condition] for dataset in datasets], axis=0
            )
            for condition in CONDITIONS
        },
        y=np.concatenate([dataset.y for dataset in datasets]),
        subject_ids=np.concatenate([dataset.subject_ids for dataset in datasets]),
        samples=np.concatenate([dataset.samples for dataset in datasets]),
    )


def make_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = set(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Grouped CV needs at least {n_splits} subjects; "
            f"found {len(unique_groups)}"
        )
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    test_counts = np.zeros(len(y), dtype=np.int8)
    for train_index, test_index in folds:
        overlap = set(groups[train_index]) & set(groups[test_index])
        if overlap:
            raise RuntimeError(f"Subject leakage inside CV: {sorted(overlap)}")
        if len(np.unique(y[train_index])) < 2 or len(np.unique(y[test_index])) < 2:
            raise ValueError("Every train/test fold must contain both classes")
        test_counts[test_index] += 1
    if not np.all(test_counts == 1):
        raise RuntimeError("Every trial must appear in exactly one test fold")
    return folds


def split_subjects(
    subject_ids: Sequence[str],
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str]]:
    """Make one deterministic subject-level train/test split."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("distributed_component_models.test_size must be in (0, 1)")
    ordered = np.asarray(sorted(set(subject_ids)), dtype=object)
    if len(ordered) < 3:
        raise ValueError("At least three eligible subjects are required")
    rng = np.random.default_rng(random_state)
    shuffled = ordered[rng.permutation(len(ordered))]
    n_test = max(1, int(math.ceil(len(ordered) * test_size)))
    test = sorted(str(value) for value in shuffled[:n_test])
    train = sorted(str(value) for value in shuffled[n_test:])
    return train, test


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


def build_grid_param_space(cfg: dict) -> dict[str, list[Any]]:
    """Return Script-06-compatible GridSearchCV parameters for a Pipeline."""
    configured = dict(cfg["ml"]["xgboost"]["param_grid"])
    phase1 = {
        f"model__{key}": list(values)
        for key, values in configured.items()
        if key != "n_estimators"
    }
    phase1["model__n_estimators"] = [500]
    if any(not values for values in phase1.values()):
        raise ValueError("Every ml.xgboost.param_grid entry must be non-empty")
    return phase1


def fit_grid_search_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    cfg: dict,
    random_state: int,
    workers: int,
    device: str,
) -> tuple[StandardScaler, Any, dict[str, Any], float, pd.DataFrame]:
    """Tune on subject-disjoint inner folds and refit on all training data."""
    base_model = xgb.XGBClassifier(
        **_xgb_params(
            {},
            y_train,
            random_state,
            workers=1,
            device=device,
        )
    )
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", base_model)]
    )
    grid_jobs = 1 if device == "cuda" else workers
    # Script 06 can early-stop on a separate validation split.  Here the held-out
    # test set must remain untouched, so GridSearchCV refits the 500-tree best
    # pipeline on all training subjects and no test-set early stopping occurs.
    search = GridSearchCV(
        pipeline,
        build_grid_param_space(cfg),
        cv=inner_folds,
        scoring="average_precision",
        n_jobs=grid_jobs,
        verbose=0,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    scaler = best_pipeline.named_steps["scaler"]
    model = best_pipeline.named_steps["model"]
    best_params = {
        key.removeprefix("model__"): _jsonable(value)
        for key, value in search.best_params_.items()
    }
    cv_results = pd.DataFrame(
        {
            "candidate": np.arange(1, len(search.cv_results_["params"]) + 1),
            "rank_test_auprc": search.cv_results_["rank_test_score"],
            "mean_test_auprc": search.cv_results_["mean_test_score"],
            "std_test_auprc": search.cv_results_["std_test_score"],
            "mean_fit_time_sec": search.cv_results_["mean_fit_time"],
            "mean_score_time_sec": search.cv_results_["mean_score_time"],
            "params": [
                json.dumps(
                    {
                        key.removeprefix("model__"): _jsonable(value)
                        for key, value in params.items()
                    },
                    sort_keys=True,
                )
                for params in search.cv_results_["params"]
            ],
        }
    ).sort_values("rank_test_auprc", kind="mergesort")
    return scaler, model, best_params, float(search.best_score_), cv_results


def fit_default_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    workers: int,
    device: str,
) -> tuple[StandardScaler, Any]:
    """Fit one model using XGBoost defaults plus required task parameters."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = xgb.XGBClassifier(
        **_xgb_params(
            {},
            y_train,
            random_state,
            workers=workers,
            device=device,
        )
    )
    model.fit(X_scaled, y_train)
    return scaler, model


def fit_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    svm_cfg: dict,
) -> tuple[StandardScaler, SVC, dict[str, Any]]:
    """Fit the fixed probability-enabled RBF SVM from experiment config."""
    params = {
        "kernel": str(svm_cfg.get("kernel", "rbf")),
        "C": float(svm_cfg.get("C", 1.0)),
        "gamma": svm_cfg.get("gamma", "scale"),
        "class_weight": svm_cfg.get("class_weight", "balanced"),
        "probability": bool(svm_cfg.get("probability", True)),
        "random_state": random_state,
    }
    if params["kernel"] != "rbf":
        raise ValueError("distributed_component_models.svm.kernel must be 'rbf'")
    if not params["probability"]:
        raise ValueError("distributed_component_models.svm.probability must be true")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVC(**params)
    model.fit(X_scaled, y_train)
    return scaler, model, _jsonable(params)


def classification_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "auprc": float(average_precision_score(y, probabilities)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(
            true_negative / (true_negative + false_positive)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "accuracy": float(accuracy_score(y, predictions)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def _prediction_frame(
    condition: str,
    dataset: ComponentDataset,
    test_index: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    return pd.DataFrame(
        {
            "condition": condition,
            "subject_id": dataset.subject_ids[test_index],
            "sample": dataset.samples[test_index],
            "true_label": dataset.y[test_index],
            "pred_proba": probabilities,
            "predicted_label": (probabilities >= threshold).astype(np.int8),
        }
    )


def _subject_metric_rows(
    predictions: pd.DataFrame,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (condition, subject_id), group in predictions.groupby(
        ["condition", "subject_id"], sort=True
    ):
        scores = classification_metrics(
            group["true_label"].to_numpy(),
            group["pred_proba"].to_numpy(),
            threshold,
        )
        rows.append(
            {
                "condition": condition,
                "subject_id": subject_id,
                "n_trials": len(group),
                "n_incorrect": int(group["true_label"].sum()),
                **scores,
            }
        )
    return rows


def _comparison_summary(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "training_strategy",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "accuracy",
        "grid_best_inner_auprc",
        "n_train_subjects",
        "n_test_subjects",
        "n_train_trials",
        "n_test_trials",
    ]
    return condition_metrics.loc[:, columns].copy()


def _condition_deltas(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = condition_metrics.set_index("condition")
    rows = []
    for step_name in STEP_NAMES:
        comparisons = (
            ("specific_minus_raw", f"{step_name}_specific", "raw"),
            ("coherent_minus_raw", f"{step_name}_coherent", "raw"),
            (
                "specific_minus_coherent",
                f"{step_name}_specific",
                f"{step_name}_coherent",
            ),
        )
        for label, target, reference in comparisons:
            for metric in _METRIC_NAMES:
                rows.append({
                    "step": step_name,
                    "comparison": label,
                    "target_condition": target,
                    "reference_condition": reference,
                    "metric": metric,
                    "delta": float(
                        indexed.loc[target, metric] - indexed.loc[reference, metric]
                    ),
                })
    return pd.DataFrame(rows)


def _diagnostics_frame(eligibility_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for row in eligibility_rows:
        if not row["eligible"]:
            continue
        diagnostics = row["diagnostics"]
        result = {
            "subject_id": row["subject_id"],
            "cached": row["cached"],
            "max_abs_cumulative_conservation_error_uv": diagnostics[
                "max_abs_cumulative_conservation_error_uv"
            ],
        }
        for step_name in STEP_NAMES:
            step = diagnostics["steps"][step_name]
            result.update({
                f"{step_name}_windows": step.get("windows", 0),
                f"{step_name}_processed_channel_windows": step.get(
                    "processed_channel_windows", 0
                ),
                f"{step_name}_solve_failures": step.get("solve_failures", 0),
                f"{step_name}_below_coherence": step.get(
                    "below_coherence_candidates", 0
                ),
                f"{step_name}_active_channels": json.dumps(
                    step.get("active_channels", [])
                ),
                f"{step_name}_group_rates": json.dumps(
                    step.get("group_processing_rates", {}), sort_keys=True
                ),
                f"{step_name}_max_abs_conservation_error_uv": step.get(
                    "max_abs_step_conservation_error_uv", 0.0
                ),
            })
        rows.append(result)
    return pd.DataFrame(rows)


def _train_model_family(
    model_name: str,
    dataset: ComponentDataset,
    train_index: np.ndarray,
    test_index: np.ndarray,
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    feature_names: dict[str, tuple[str, ...]],
    cfg: dict,
    experiment_cfg: dict,
    random_state: int,
    workers: int,
    threshold: float,
    device: str,
    grid_search_enabled: bool,
    inner_folds: list[tuple[np.ndarray, np.ndarray]] | None,
    out: Path,
) -> pd.DataFrame:
    model_root = out / model_name
    condition_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        X = dataset.matrix(condition)
        if X.shape[1] != len(feature_names[condition]):
            raise RuntimeError(
                f"{condition} feature shape/name mismatch: "
                f"{X.shape[1]} != {len(feature_names[condition])}"
            )
        if model_name == "xgboost" and grid_search_enabled:
            print(f"[XGBoost GridSearchCV {condition_index}/{len(CONDITIONS)}] {condition}")
            if inner_folds is None:
                raise RuntimeError("GridSearchCV folds were not initialized")
            scaler, fitted, best_params, grid_score, grid_results = (
                fit_grid_search_xgboost(
                    X[train_index],
                    dataset.y[train_index],
                    inner_folds,
                    cfg,
                    random_state,
                    workers,
                    device,
                )
            )
            training_strategy = "grid_search"
        elif model_name == "xgboost":
            print(f"[XGBoost {condition_index}/{len(CONDITIONS)}] {condition}")
            scaler, fitted = fit_default_xgboost(
                X[train_index],
                dataset.y[train_index],
                random_state,
                workers,
                device,
            )
            best_params = {}
            grid_score = None
            grid_results = None
            training_strategy = "default_parameters"
        else:
            print(f"[SVM {condition_index}/{len(CONDITIONS)}] {condition}")
            scaler, fitted, best_params = fit_svm(
                X[train_index],
                dataset.y[train_index],
                random_state,
                experiment_cfg["svm"],
            )
            grid_score = None
            grid_results = None
            training_strategy = "fixed_parameters"

        X_test = scaler.transform(X[test_index])
        probabilities = fitted.predict_proba(X_test)[:, 1]
        scores = classification_metrics(dataset.y[test_index], probabilities, threshold)
        condition_rows.append({
            "condition": condition,
            "training_strategy": training_strategy,
            "n_train_subjects": len(train_subjects),
            "n_test_subjects": len(test_subjects),
            "n_train_trials": len(train_index),
            "n_test_trials": len(test_index),
            **{metric: scores[metric] for metric in _METRIC_NAMES},
            "threshold": threshold,
            "grid_best_inner_auprc": grid_score,
            "best_params": json.dumps(best_params, sort_keys=True),
            "confusion_matrix": json.dumps(scores["confusion_matrix"]),
        })
        predictions = _prediction_frame(
            condition, dataset, test_index, probabilities, threshold
        )
        prediction_frames.append(predictions)
        condition_dir = model_root / "conditions" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, condition_dir / "scaler.joblib")
        joblib.dump(fitted, condition_dir / "model.joblib")
        _save_json(best_params, condition_dir / "best_params.json")
        if grid_results is not None:
            grid_results.to_csv(condition_dir / "grid_search_results.csv", index=False)
        predictions.to_csv(condition_dir / "predictions.csv", index=False)
        _save_json(scores, condition_dir / "metrics.json")

    condition_metrics = pd.DataFrame(condition_rows)
    condition_metrics.to_csv(model_root / "condition_metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(model_root / "predictions.csv", index=False)
    pd.DataFrame(_subject_metric_rows(predictions, threshold)).to_csv(
        model_root / "subject_metrics.csv", index=False
    )
    comparison = _comparison_summary(condition_metrics)
    comparison.to_csv(model_root / "comparison_summary.csv", index=False)
    _condition_deltas(condition_metrics).to_csv(
        model_root / "condition_deltas.csv", index=False
    )
    return None


def run(
    config_path: str | Path,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int = 1,
    random_state_override: int | None = None,
    model: str = "both",
    grid_search_override: bool | None = None,
    force: bool = False,
    recompute_components: bool = False,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if model not in {"both", *MODEL_NAMES}:
        raise ValueError(f"Unknown model selection: {model}")
    selected_models = MODEL_NAMES if model == "both" else (model,)
    if model == "svm" and grid_search_override is True:
        raise ValueError("--grid-search applies only to XGBoost")
    config_path = Path(config_path)
    cfg = load_config(config_path)
    experiment_cfg = cfg["erp_core"]["distributed_component_models"]
    test_size = float(experiment_cfg["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("distributed_component_models.test_size must be in (0, 1)")
    random_state = (
        int(random_state_override)
        if random_state_override is not None
        else int(experiment_cfg["random_state"])
    )
    threshold = float(experiment_cfg.get("decision_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "distributed_component_models.decision_threshold must be in (0, 1)"
        )
    xgboost_cfg = experiment_cfg["xgboost"]
    device = str(xgboost_cfg.get("device", "cpu"))
    grid_search_enabled = "xgboost" in selected_models and (
        bool(grid_search_override)
        if grid_search_override is not None
        else bool(xgboost_cfg.get("grid_search_enabled", False))
    )
    resolved_cfg = deepcopy(cfg)
    resolved_experiment = resolved_cfg["erp_core"]["distributed_component_models"]
    resolved_experiment["random_state"] = random_state
    resolved_experiment["selected_models"] = list(selected_models)
    resolved_experiment["xgboost"]["grid_search_enabled"] = grid_search_enabled

    out = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _project_path(experiment_cfg["output_dir"], config_path)
    )
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {out}; pass --force to overwrite files"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = _project_path(
        cfg["erp_core"]["distributed_components"]["cache_subdir"], config_path
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    helpers = _load_script10()
    source_root = (
        data_dir.expanduser().resolve()
        if data_dir is not None
        else Path(cfg["erp_core"]["data_dir"]).expanduser().resolve()
    )
    recordings = helpers._select_recordings(
        helpers._resolve_recordings(source_root, None), subjects=None, task="ern"
    )
    recordings_by_subject = {
        str(item["subject_id"]): Path(item["ern"]) for item in recordings
    }
    eligibility_rows = extract_all_subjects(
        recordings, cfg, cache_root, force, recompute_components, workers
    )
    eligibility_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "diagnostics"}
         for row in eligibility_rows]
    )
    eligibility_frame.to_csv(out / "eligibility.csv", index=False)
    diagnostics_frame = _diagnostics_frame(eligibility_rows)
    diagnostics_frame.to_csv(out / "processing_diagnostics.csv", index=False)
    eligible_subjects = sorted(
        str(row["subject_id"])
        for row in eligibility_rows
        if row["eligible"]
    )
    dataset = load_component_dataset(
        eligible_subjects, recordings_by_subject, cfg, cache_root
    )
    train_subjects, test_subjects = split_subjects(
        eligible_subjects, test_size, random_state
    )
    train_index = np.flatnonzero(
        np.isin(dataset.subject_ids, np.asarray(train_subjects, dtype=str))
    )
    test_index = np.flatnonzero(
        np.isin(dataset.subject_ids, np.asarray(test_subjects, dtype=str))
    )
    if not len(train_index) or not len(test_index):
        raise RuntimeError("Deterministic subject split produced an empty partition")
    if set(dataset.subject_ids[train_index]) & set(dataset.subject_ids[test_index]):
        raise RuntimeError("Subject leakage between train and test partitions")
    grid_cv_folds: int | None = None
    inner_folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    if grid_search_enabled:
        grid_cv_folds = int(cfg["ml"]["xgboost"]["cv_folds"])
        if len(train_subjects) < grid_cv_folds:
            raise ValueError(
                f"GridSearchCV needs at least {grid_cv_folds} training subjects; "
                f"found {len(train_subjects)}"
            )
        inner_folds = make_group_folds(
            dataset.y[train_index],
            dataset.subject_ids[train_index],
            grid_cv_folds,
            random_state,
        )
    layouts = build_feature_layouts(cfg)
    feature_names = condition_feature_names(layouts)
    feature_layout_payload = {
        condition: {
            "feature_count": len(names),
            "feature_names": names,
            "channels": layouts[condition].channels,
            "symmetric_pairs": layouts[condition].symmetric_pairs,
        }
        for condition, names in feature_names.items()
    }
    _save_json(feature_layout_payload, out / "feature_layout.json")

    split_manifest_rows = []
    for subject_id in eligible_subjects:
        mask = dataset.subject_ids == subject_id
        split_manifest_rows.append(
            {
                "subject_id": subject_id,
                "split": "train" if subject_id in train_subjects else "test",
                "n_trials": int(mask.sum()),
                "n_correct": int(np.count_nonzero(dataset.y[mask] == 0)),
                "n_incorrect": int(np.count_nonzero(dataset.y[mask] == 1)),
            }
        )
    pd.DataFrame(split_manifest_rows).to_csv(
        out / "split_manifest.csv", index=False
    )
    _save_json(
        {
            "random_state": random_state,
            "test_size": test_size,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        },
        out / "split_manifest.json",
    )

    for model_name in selected_models:
        _train_model_family(
            model_name,
            dataset,
            train_index,
            test_index,
            train_subjects,
            test_subjects,
            feature_names,
            cfg,
            experiment_cfg,
            random_state,
            workers,
            threshold,
            device,
            grid_search_enabled,
            inner_folds,
            out,
        )

    summary = {
        "input_data_dir": str(source_root),
        "n_discovered_subjects": len(recordings),
        "n_eligible_subjects": len(eligible_subjects),
        "n_excluded_subjects": len(recordings) - len(eligible_subjects),
        "test_size": test_size,
        "n_train_subjects": len(train_subjects),
        "n_test_subjects": len(test_subjects),
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "conditions": list(CONDITIONS),
        "feature_counts": {
            condition: len(names) for condition, names in feature_names.items()
        },
        "feature_channels": {
            condition: layout.channels for condition, layout in layouts.items()
        },
        "random_state": random_state,
        "decision_threshold": threshold,
        "selected_models": list(selected_models),
        "model_output_roots": {
            model_name: str(out / model_name) for model_name in selected_models
        },
        "xgboost_training_strategy": (
            "grid_search" if grid_search_enabled else "default_parameters"
        ) if "xgboost" in selected_models else None,
        "xgboost_grid_search_enabled": grid_search_enabled,
        "xgboost_grid_cv_folds": grid_cv_folds,
        "xgboost_grid_scoring": (
            "average_precision" if grid_search_enabled else None
        ),
        "xgboost_grid_param_space": (
            build_grid_param_space(cfg) if grid_search_enabled else None
        ),
        "xgboost_grid_n_estimators_policy": (
            "Ignore configured n_estimators candidates and fix 500 during "
            "GridSearchCV, matching scripts/06_train_xgboost.py."
            if grid_search_enabled
            else None
        ),
        "xgboost_default_parameter_policy": (
            "Use XGBoost defaults except required task, class-weight, seed, "
            "device, and worker parameters."
            if "xgboost" in selected_models and not grid_search_enabled
            else None
        ),
        "xgboost_device": device if "xgboost" in selected_models else None,
        "svm_parameters": (
            _jsonable(experiment_cfg["svm"]) if "svm" in selected_models else None
        ),
        "feature_worker_processes": workers,
        "grid_search_parallel_jobs": (
            (1 if device == "cuda" else workers)
            if grid_search_enabled
            else None
        ),
        "xgboost_threads_per_fit": (
            (1 if grid_search_enabled else workers)
            if "xgboost" in selected_models
            else None
        ),
        "raw_definition": (
            "Common EEG selection, montage, notch, bandpass, resampling, raw-based "
            "trial rejection, and baseline; no ICA or Wiener denoising."
        ),
        "component_identities": {
            "step1": "raw = step1_specific + step1_coherent",
            "step2": "step1_specific = step2_specific + step2_coherent",
            "step3": "step2_specific = step3_specific + step3_coherent",
            "cumulative": (
                "raw = step3_specific + step1_coherent + "
                "step2_coherent + step3_coherent"
            ),
        },
        "label_encoding": {"correct": 0, "incorrect": 1},
        "shap_output": False,
        "continuous_component_output": False,
        "shared_component_cache": str(cache_root),
        "recompute_components": recompute_components,
    }
    _save_json(summary, out / "run_summary.json")
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"ERP-CORE ERN component comparison complete: {out}")
    return out


def main() -> None:
    args = _parse_args()
    output = run(
        args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        random_state_override=args.random_state,
        model=args.model,
        grid_search_override=args.grid_search,
        force=args.force,
        recompute_components=args.recompute_components,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
