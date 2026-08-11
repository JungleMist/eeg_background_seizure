#!/usr/bin/env python3
"""Compare seven ERP-CORE ERN signal-component conditions with XGBoost.

Every condition uses the same response-locked trials and one deterministic
subject-disjoint train/test split.  GridSearchCV runs only within the training
subjects.  The ``raw`` condition means common preprocessing without ICA or
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
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config
from eeg_bg.features.extraction import (
    build_feature_names,
    extract_epoch_features_for_layout,
)
from eeg_bg.ml.shap_analysis import compute_shap_values, plot_shap_summary


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

BASE_CONDITIONS: tuple[str, ...] = (
    "wiener_specific",
    "wiener_coherent",
    "ica_wiener_specific",
    "ica_wiener_coherent",
    "raw",
    "ica",
)
COMBINED_CONDITION = "wiener_specific_coherent"
CONDITIONS: tuple[str, ...] = (
    "wiener_specific",
    "wiener_coherent",
    COMBINED_CONDITION,
    "ica_wiener_specific",
    "ica_wiener_coherent",
    "raw",
    "ica",
)
_CACHE_SCHEMA = 1


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
        if condition == COMBINED_CONDITION:
            return np.concatenate(
                [
                    self.features["wiener_specific"],
                    self.features["wiener_coherent"],
                ],
                axis=1,
            )
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
            "Subject processes for feature generation and GridSearchCV parallel "
            "jobs; each XGBoost fit uses one CPU thread (default: 1)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="Override the deterministic split/GridSearchCV seed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute subject caches and overwrite result files.",
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


def build_feature_layouts(cfg: dict) -> tuple[FeatureLayout, FeatureLayout]:
    full_names = tuple(
        build_feature_names(ERP_ERN_CHANNELS, ERP_ERN_SYMMETRIC_PAIRS)
    )
    grouped = {
        str(channel)
        for group in cfg["channels"]["channel_groups"]
        for channel in group
    }
    coherent_channels = tuple(
        channel for channel in ERP_ERN_CHANNELS if channel in grouped
    )
    if not coherent_channels:
        raise ValueError(
            "No ERP ERN feature channel appears in channels.channel_groups"
        )
    coherent_set = set(coherent_channels)
    coherent_pairs = tuple(
        pair
        for pair in ERP_ERN_SYMMETRIC_PAIRS
        if pair[0] in coherent_set and pair[1] in coherent_set
    )
    coherent_names = tuple(
        build_feature_names(coherent_channels, coherent_pairs)
    )
    return (
        FeatureLayout(
            channels=ERP_ERN_CHANNELS,
            symmetric_pairs=ERP_ERN_SYMMETRIC_PAIRS,
            feature_names=full_names,
        ),
        FeatureLayout(
            channels=coherent_channels,
            symmetric_pairs=coherent_pairs,
            feature_names=coherent_names,
        ),
    )


def condition_feature_names(
    full_layout: FeatureLayout,
    coherent_layout: FeatureLayout,
) -> dict[str, tuple[str, ...]]:
    names = {
        "wiener_specific": full_layout.feature_names,
        "wiener_coherent": coherent_layout.feature_names,
        "ica_wiener_specific": full_layout.feature_names,
        "ica_wiener_coherent": coherent_layout.feature_names,
        "raw": full_layout.feature_names,
        "ica": full_layout.feature_names,
    }
    names[COMBINED_CONDITION] = tuple(
        [f"specific__{name}" for name in full_layout.feature_names]
        + [f"coherent__{name}" for name in coherent_layout.feature_names]
    )
    return names


def _source_files(recording: Path) -> list[Path]:
    paths = [recording]
    sidecar = recording.with_suffix(".fdt")
    if sidecar.is_file():
        paths.append(sidecar)
    return paths


def _cache_fingerprint(
    recording: Path,
    cfg: dict,
    full_layout: FeatureLayout,
    coherent_layout: FeatureLayout,
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
        "channels": cfg["channels"],
        "wiener": cfg["wiener"],
        "ern": cfg["erp_core"]["ern"],
        "response_pairing_window_sec": cfg["erp_core"][
            "response_pairing_window_sec"
        ],
        "standard_ica": cfg["erp_core"]["standard_ica"],
        "feature_freq_band": (0.5, 30.0),
        "full_channels": full_layout.channels,
        "full_pairs": full_layout.symmetric_pairs,
        "coherent_channels": coherent_layout.channels,
        "coherent_pairs": coherent_layout.symmetric_pairs,
    }
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_root: Path, subject_id: str) -> Path:
    return cache_root / subject_id / "components.npz"


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
            for condition in BASE_CONDITIONS
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
        for condition in BASE_CONDITIONS
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


def derive_coherent_raw(source, specific):
    if source.ch_names != specific.ch_names:
        raise ValueError("Source and specific Raw objects have different channels")
    if source.n_times != specific.n_times or not np.isclose(
        source.info["sfreq"], specific.info["sfreq"]
    ):
        raise ValueError("Source and specific Raw objects are not time-aligned")
    coherent = source.copy()
    coherent._data = source.get_data() - specific.get_data()
    return coherent


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


def _summarize_wiener(diagnostics: dict) -> dict:
    return {
        key: value
        for key, value in diagnostics.items()
        if key != "window_diagnostics"
    }


def extract_subject_components(
    subject_id: str,
    recording: Path,
    cfg: dict,
    cache_root: Path,
    helpers,
    force: bool,
) -> tuple[ComponentDataset, dict, bool]:
    full_layout, coherent_layout = build_feature_layouts(cfg)
    fingerprint = _cache_fingerprint(
        recording, cfg, full_layout, coherent_layout
    )
    path = _cache_path(cache_root, subject_id)
    if not force:
        cached = _load_subject_cache(path, fingerprint)
        if cached is not None:
            dataset, diagnostics = cached
            return dataset, diagnostics, True

    import mne

    original = helpers._read_recording(mne, recording)
    common = helpers._common_preprocess(original, cfg)
    events, event_id = mne.events_from_annotations(common, verbose=False)
    table = helpers.build_response_table(
        events,
        event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )

    ica_raw, excluded = helpers._standard_ica(common, cfg)
    wiener_specific, raw_wiener_diagnostics = helpers._wiener_continuous(
        common, cfg, subject_id
    )
    ica_wiener_specific, ica_wiener_diagnostics = helpers._wiener_continuous(
        ica_raw, cfg, f"{subject_id}_ica"
    )
    wiener_coherent = derive_coherent_raw(common, wiener_specific)
    ica_wiener_coherent = derive_coherent_raw(ica_raw, ica_wiener_specific)

    np.testing.assert_allclose(
        wiener_specific.get_data() + wiener_coherent.get_data(),
        common.get_data(),
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ica_wiener_specific.get_data() + ica_wiener_coherent.get_data(),
        ica_raw.get_data(),
        rtol=1e-7,
        atol=1e-12,
    )

    branches = {
        "raw": common,
        "wiener_specific": wiener_specific,
        "wiener_coherent": wiener_coherent,
        "ica_wiener_specific": ica_wiener_specific,
        "ica_wiener_coherent": ica_wiener_coherent,
        "ica": ica_raw,
    }
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
    for condition in BASE_CONDITIONS:
        layout = (
            coherent_layout
            if condition in {"wiener_coherent", "ica_wiener_coherent"}
            else full_layout
        )
        features[condition] = _epochs_to_features(epochs[condition], layout)
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
        "subject_id": subject_id,
        "recording": str(recording),
        "n_trials": len(labels),
        "n_correct": int(counts[0]),
        "n_incorrect": int(counts[1]),
        "ica_excluded_components": [int(value) for value in excluded],
        "raw_wiener": _summarize_wiener(raw_wiener_diagnostics),
        "ica_wiener": _summarize_wiener(ica_wiener_diagnostics),
    }
    _save_subject_cache(path, dataset, fingerprint, diagnostics)
    return dataset, diagnostics, False


def _extract_subject_worker(
    args: tuple[str, str, dict, str, bool],
    helpers: Any | None = None,
) -> dict:
    subject_id, recording, cfg, cache_root, force = args
    if helpers is None:
        helpers = _load_script10()
    with threadpool_limits(limits=1):
        try:
            dataset, diagnostics, cached = extract_subject_components(
                subject_id,
                Path(recording),
                cfg,
                Path(cache_root),
                helpers,
                force,
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
    workers: int,
) -> list[dict]:
    jobs = [
        (
            str(item["subject_id"]),
            str(item["ern"]),
            cfg,
            str(cache_root),
            force,
        )
        for item in recordings
    ]
    if not jobs:
        return []
    if workers == 1:
        helpers = _load_script10()
        rows = []
        for index, job in enumerate(jobs, start=1):
            print(f"[features {index}/{len(jobs)}] {job[0]}")
            rows.append(_extract_subject_worker(job, helpers))
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
    full_layout, coherent_layout = build_feature_layouts(cfg)
    datasets: list[ComponentDataset] = []
    for subject_id in subject_ids:
        fingerprint = _cache_fingerprint(
            recordings_by_subject[subject_id], cfg, full_layout, coherent_layout
        )
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
            for condition in BASE_CONDITIONS
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
        raise ValueError("component_xgboost.test_size must be in (0, 1)")
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
        scoring="roc_auc",
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
            "rank_test_auroc": search.cv_results_["rank_test_score"],
            "mean_test_auroc": search.cv_results_["mean_test_score"],
            "std_test_auroc": search.cv_results_["std_test_score"],
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
    ).sort_values("rank_test_auroc", kind="mergesort")
    return scaler, model, best_params, float(search.best_score_), cv_results


def classification_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "auroc": float(roc_auc_score(y, probabilities)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
    }


def _prediction_frame(
    condition: str,
    dataset: ComponentDataset,
    test_index: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
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


def _feature_metadata(name: str, component: str, index: int) -> dict[str, Any]:
    if name.startswith("asym_"):
        _, left, right, band = name.split("_", 3)
        return {
            "feature_index": index,
            "feature": f"{component}__{name}",
            "component": component,
            "channel": "",
            "pair_left": left,
            "pair_right": right,
            "family": "asymmetry",
            "band": band,
        }
    channel, suffix = name.split("_", 1)
    if suffix.endswith("_power"):
        family = "band_power"
        band = suffix.removesuffix("_power")
    elif suffix.startswith("hjorth_"):
        family = "hjorth"
        band = ""
    elif suffix == "spectral_entropy":
        family = "spectral_entropy"
        band = ""
    else:
        family = "other"
        band = ""
    return {
        "feature_index": index,
        "feature": f"{component}__{name}",
        "component": component,
        "channel": channel,
        "pair_left": "",
        "pair_right": "",
        "family": family,
        "band": band,
    }


def build_combined_feature_metadata(
    full_layout: FeatureLayout,
    coherent_layout: FeatureLayout,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component, names in (
        ("specific", full_layout.feature_names),
        ("coherent", coherent_layout.feature_names),
    ):
        for name in names:
            rows.append(_feature_metadata(name, component, len(rows)))
    return pd.DataFrame(rows)


def aggregate_shap(
    shap_values: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    if shap_values.ndim != 2 or shap_values.shape[1] != len(metadata):
        raise ValueError("SHAP values do not match combined feature metadata")
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feature = metadata.copy()
    feature["mean_abs_shap"] = mean_abs
    feature["rank"] = (
        feature["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    )
    feature = feature.sort_values("rank", kind="mergesort").reset_index(drop=True)

    component_rows = []
    total = float(np.sum(mean_abs))
    for component, group in metadata.groupby("component", sort=False):
        values = mean_abs[group["feature_index"].to_numpy(int)]
        component_total = float(np.sum(values))
        component_rows.append(
            {
                "component": component,
                "n_features": len(values),
                "mean_abs_shap_per_feature": float(np.mean(values)),
                "total_mean_abs_shap": component_total,
                "total_abs_share": component_total / total if total else 0.0,
            }
        )
    component_frame = pd.DataFrame(component_rows)

    family_rows = []
    for keys, group in metadata.groupby(
        ["component", "family", "band"], sort=False, dropna=False
    ):
        values = mean_abs[group["feature_index"].to_numpy(int)]
        family_rows.append(
            {
                "component": keys[0],
                "family": keys[1],
                "band": keys[2],
                "n_features": len(values),
                "mean_abs_shap": float(np.mean(values)),
                "total_mean_abs_shap": float(np.sum(values)),
            }
        )
    family_frame = pd.DataFrame(family_rows)

    channel_contributions: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in metadata.itertuples(index=False):
        value = float(mean_abs[int(row.feature_index)])
        if row.family == "asymmetry":
            for channel in (row.pair_left, row.pair_right):
                channel_contributions.setdefault(
                    (row.component, channel), []
                ).append((0.5 * value, 0.5))
        else:
            channel_contributions.setdefault(
                (row.component, row.channel), []
            ).append((value, 1.0))
    channel_frame = pd.DataFrame(
        [
            {
                "component": component,
                "channel": channel,
                "mean_abs_shap": sum(value for value, _ in contributions)
                / sum(weight for _, weight in contributions),
            }
            for (component, channel), contributions in channel_contributions.items()
        ]
    )
    return {
        "feature": feature,
        "component": component_frame,
        "family": family_frame,
        "channel": channel_frame,
    }


def _comparison_summary(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "auroc",
        "f1",
        "accuracy",
        "grid_best_inner_auroc",
        "n_train_subjects",
        "n_test_subjects",
        "n_train_trials",
        "n_test_trials",
    ]
    return condition_metrics.loc[:, columns].copy()


def _condition_deltas(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("wiener_specific_minus_raw", "wiener_specific", "raw"),
        ("wiener_coherent_minus_raw", "wiener_coherent", "raw"),
        (
            "specific_coherent_minus_specific",
            COMBINED_CONDITION,
            "wiener_specific",
        ),
        ("ica_minus_raw", "ica", "raw"),
        (
            "ica_wiener_specific_minus_ica",
            "ica_wiener_specific",
            "ica",
        ),
        (
            "ica_wiener_coherent_minus_coherent",
            "ica_wiener_coherent",
            "wiener_coherent",
        ),
    )
    indexed = condition_metrics.set_index("condition")
    rows = []
    for label, target, reference in comparisons:
        for metric in ("auroc", "f1", "accuracy"):
            rows.append(
                {
                    "comparison": label,
                    "target_condition": target,
                    "reference_condition": reference,
                    "metric": metric,
                    "delta": float(
                        indexed.loc[target, metric]
                        - indexed.loc[reference, metric]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _diagnostics_frame(eligibility_rows: list[dict]) -> pd.DataFrame:
    rows = []
    for row in eligibility_rows:
        if not row["eligible"]:
            continue
        diagnostics = row["diagnostics"]
        raw_wiener = diagnostics["raw_wiener"]
        ica_wiener = diagnostics["ica_wiener"]
        rows.append(
            {
                "subject_id": row["subject_id"],
                "cached": row["cached"],
                "ica_excluded_components": json.dumps(
                    diagnostics["ica_excluded_components"]
                ),
                "ica_excluded_count": len(
                    diagnostics["ica_excluded_components"]
                ),
                "raw_wiener_windows": raw_wiener.get("windows", 0),
                "raw_wiener_processed_channel_windows": raw_wiener.get(
                    "processed_channel_windows", 0
                ),
                "raw_wiener_solve_failures": raw_wiener.get("solve_failures", 0),
                "raw_wiener_below_coherence": raw_wiener.get(
                    "below_coherence_candidates", 0
                ),
                "raw_wiener_group_rates": json.dumps(
                    raw_wiener.get("group_processing_rates", {}), sort_keys=True
                ),
                "ica_wiener_windows": ica_wiener.get("windows", 0),
                "ica_wiener_processed_channel_windows": ica_wiener.get(
                    "processed_channel_windows", 0
                ),
                "ica_wiener_solve_failures": ica_wiener.get("solve_failures", 0),
                "ica_wiener_below_coherence": ica_wiener.get(
                    "below_coherence_candidates", 0
                ),
                "ica_wiener_group_rates": json.dumps(
                    ica_wiener.get("group_processing_rates", {}), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows)


def run(
    config_path: str | Path,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int = 1,
    random_state_override: int | None = None,
    force: bool = False,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    config_path = Path(config_path)
    cfg = load_config(config_path)
    experiment_cfg = cfg["erp_core"]["component_xgboost"]
    test_size = float(experiment_cfg["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ValueError("component_xgboost.test_size must be in (0, 1)")
    random_state = (
        int(random_state_override)
        if random_state_override is not None
        else int(experiment_cfg["random_state"])
    )
    threshold = float(experiment_cfg.get("decision_threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ValueError("component_xgboost.decision_threshold must be in (0, 1)")
    device = str(experiment_cfg.get("device", "cpu"))
    resolved_cfg = deepcopy(cfg)
    resolved_cfg["erp_core"]["component_xgboost"]["random_state"] = random_state

    out = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _project_path(experiment_cfg["output_dir"], config_path)
    )
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is not empty: {out}; pass --force to overwrite files"
        )
    out.mkdir(parents=True, exist_ok=True)
    cache_root = (
        Path(cfg["paths"]["cache_dir"])
        / str(experiment_cfg.get("cache_subdir", "erp_core_ern_components"))
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
        recordings, cfg, cache_root, force, workers
    )
    eligibility_frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "diagnostics"}
         for row in eligibility_rows]
    )
    eligibility_frame.to_csv(out / "eligibility.csv", index=False)
    diagnostics_frame = _diagnostics_frame(eligibility_rows)
    diagnostics_frame.to_csv(out / "ica_diagnostics.csv", index=False)
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
    grid_cv_folds = int(cfg["ml"]["xgboost"]["cv_folds"])
    if len(train_subjects) < grid_cv_folds:
        raise ValueError(
            f"GridSearchCV needs at least {grid_cv_folds} training subjects; "
            f"found {len(train_subjects)}"
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
    inner_folds = make_group_folds(
        dataset.y[train_index],
        dataset.subject_ids[train_index],
        grid_cv_folds,
        random_state,
    )
    full_layout, coherent_layout = build_feature_layouts(cfg)
    feature_names = condition_feature_names(full_layout, coherent_layout)
    feature_layout_payload = {
        condition: {
            "feature_count": len(names),
            "feature_names": names,
        }
        for condition, names in feature_names.items()
    }
    feature_layout_payload["full_channels"] = list(full_layout.channels)
    feature_layout_payload["full_symmetric_pairs"] = list(
        full_layout.symmetric_pairs
    )
    feature_layout_payload["coherent_channels"] = list(coherent_layout.channels)
    feature_layout_payload["coherent_symmetric_pairs"] = list(
        coherent_layout.symmetric_pairs
    )
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

    condition_metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    test_shap: np.ndarray | None = None
    test_scaled: np.ndarray | None = None
    combined_metadata = build_combined_feature_metadata(
        full_layout, coherent_layout
    )

    for condition_index, condition in enumerate(CONDITIONS, start=1):
        X = dataset.matrix(condition)
        if X.shape[1] != len(feature_names[condition]):
            raise RuntimeError(
                f"{condition} feature shape/name mismatch: "
                f"{X.shape[1]} != {len(feature_names[condition])}"
            )
        print(
            f"[GridSearchCV {condition_index}/{len(CONDITIONS)}] {condition}"
        )
        scaler, model, best_params, grid_best_score, grid_results = (
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
        X_test = scaler.transform(X[test_index])
        probabilities = model.predict_proba(X_test)[:, 1]
        scores = classification_metrics(
            dataset.y[test_index], probabilities, threshold
        )
        condition_metric_rows.append(
            {
                "condition": condition,
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "n_train_trials": len(train_index),
                "n_test_trials": len(test_index),
                "auroc": scores["auroc"],
                "f1": scores["f1"],
                "accuracy": scores["accuracy"],
                "threshold": threshold,
                "grid_best_inner_auroc": grid_best_score,
                "best_params": json.dumps(best_params, sort_keys=True),
                "confusion_matrix": json.dumps(scores["confusion_matrix"]),
            }
        )
        predictions = _prediction_frame(
            condition,
            dataset,
            test_index,
            probabilities,
            threshold,
        )
        prediction_frames.append(predictions)
        condition_dir = out / "conditions" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, condition_dir / "scaler.joblib")
        joblib.dump(model, condition_dir / "model.joblib")
        _save_json(best_params, condition_dir / "best_params.json")
        grid_results.to_csv(
            condition_dir / "grid_search_results.csv", index=False
        )
        predictions.to_csv(condition_dir / "predictions.csv", index=False)
        _save_json(scores, condition_dir / "metrics.json")

        if condition == COMBINED_CONDITION:
            test_shap = np.asarray(
                compute_shap_values(
                    model, X_test, list(feature_names[condition])
                ),
                dtype=np.float64,
            )
            if test_shap.shape != X_test.shape:
                raise RuntimeError(
                    f"Unexpected SHAP shape {test_shap.shape}; "
                    f"expected {X_test.shape}"
                )
            test_scaled = X_test

    condition_metrics = pd.DataFrame(condition_metric_rows)
    condition_metrics.to_csv(out / "condition_metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    subject_metrics = pd.DataFrame(
        _subject_metric_rows(predictions, threshold)
    )
    subject_metrics.to_csv(out / "subject_metrics.csv", index=False)
    comparison = _comparison_summary(condition_metrics)
    comparison.to_csv(out / "comparison_summary.csv", index=False)
    deltas = _condition_deltas(condition_metrics)
    deltas.to_csv(out / "condition_deltas.csv", index=False)

    if test_shap is None or test_scaled is None:
        raise RuntimeError("Combined-condition held-out SHAP was not generated")
    if np.isnan(test_shap).any() or np.isnan(test_scaled).any():
        raise RuntimeError("Combined-condition held-out SHAP contains NaN")
    shap_dir = out / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    np.save(shap_dir / "shap_values_test.npy", test_shap.astype(np.float32))
    np.save(shap_dir / "scaled_features_test.npy", test_scaled.astype(np.float32))
    shap_tables = aggregate_shap(test_shap, combined_metadata)
    shap_tables["feature"].to_csv(
        shap_dir / "shap_feature_importance.csv", index=False
    )
    shap_tables["component"].to_csv(
        shap_dir / "shap_component_importance.csv", index=False
    )
    shap_tables["family"].to_csv(
        shap_dir / "shap_family_importance.csv", index=False
    )
    shap_tables["channel"].to_csv(
        shap_dir / "shap_channel_importance.csv", index=False
    )
    plot_shap_summary(
        test_shap,
        test_scaled,
        list(feature_names[COMBINED_CONDITION]),
        "Held-out SHAP: Wiener Specific + Coherent",
        shap_dir / "shap_summary.png",
        max_display=int(cfg["ml"]["shap"]["max_display"]),
        dpi=int(cfg["ml"]["shap"]["dpi"]),
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
        "full_channels": full_layout.channels,
        "coherent_channels": coherent_layout.channels,
        "random_state": random_state,
        "decision_threshold": threshold,
        "xgboost_grid_cv_folds": grid_cv_folds,
        "xgboost_grid_scoring": "roc_auc",
        "xgboost_grid_param_space": build_grid_param_space(cfg),
        "xgboost_grid_n_estimators_policy": (
            "Ignore configured n_estimators candidates and fix 500 during "
            "GridSearchCV, matching scripts/06_train_xgboost.py."
        ),
        "xgboost_device": device,
        "feature_worker_processes": workers,
        "grid_search_parallel_jobs": 1 if device == "cuda" else workers,
        "xgboost_threads_per_fit": 1,
        "raw_definition": (
            "Common EEG selection, montage, notch, bandpass, resampling, raw-based "
            "trial rejection, and baseline; no ICA or Wiener denoising."
        ),
        "component_identities": {
            "raw_wiener": "common_raw = wiener_specific + wiener_coherent",
            "ica_wiener": (
                "ica = ica_wiener_specific + ica_wiener_coherent"
            ),
        },
        "label_encoding": {"correct": 0, "incorrect": 1},
        "shap_policy": (
            "Only wiener_specific_coherent; the final model explains only "
            "the deterministic held-out test subjects."
        ),
        "comparison_summary": comparison.to_dict(orient="records"),
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
        force=args.force,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
