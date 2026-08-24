#!/usr/bin/env python3
"""Compare nine ERP-CORE ERN conditions with EEGNet on raw time series.

Two serial ECMAD steps operate on raw then step-1-specific data, while ICA is
applied independently to raw before its own Wiener decomposition.  Each
condition is trained with one deterministic subject-level train/validation/test
split.  Validation AUPRC selects the checkpoint and validation balanced
accuracy selects the frozen decision threshold; the test set is evaluated only
after both choices are frozen.
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

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml
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
from threadpoolctl import threadpool_limits
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from eeg_bg.config.settings import load_config
from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.erp_eegnet import (
    TrialSequenceDataset as _PackageTrialSequenceDataset,
    classification_metrics as _package_classification_metrics,
    make_model as _package_make_model,
    predict_trials as _package_predict_trials,
    select_balanced_accuracy_threshold as _package_select_threshold,
    train_condition as _package_train_condition,
)


ERP_ERN_CHANNELS: tuple[str, ...] = (
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7",
    "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4",
    "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2",
)
STEP_NAMES: tuple[str, ...] = ("step1", "step2")
BASE_CONDITIONS: tuple[str, ...] = (
    "raw",
    "step1_specific",
    "step1_coherent",
    "step2_specific",
    "step2_coherent",
    "ica",
    "ica_wiener_specific",
    "ica_wiener_coherent",
)
COMBINED_CONDITION = "step1_specific_coherent"
CONDITIONS: tuple[str, ...] = (
    "raw",
    "step1_specific",
    "step1_coherent",
    COMBINED_CONDITION,
    "step2_specific",
    "step2_coherent",
    "ica",
    "ica_wiener_specific",
    "ica_wiener_coherent",
)
_CACHE_SCHEMA = 2
_INNER_SPLIT_SEED_OFFSET = 1_000_003


@dataclass(frozen=True)
class SequenceDataset:
    sequences: dict[str, np.ndarray]
    y: np.ndarray
    subject_ids: np.ndarray
    samples: np.ndarray

    def matrix(self, condition: str, normalize: bool = True) -> np.ndarray:
        if condition == COMBINED_CONDITION:
            array = np.concatenate(
                [self.sequences["step1_specific"], self.sequences["step1_coherent"]],
                axis=1,
            )
        else:
            try:
                array = self.sequences[condition]
            except KeyError as exc:
                raise ValueError(f"Unknown condition: {condition}") from exc
        if not normalize:
            return np.asarray(array, dtype=np.float32)
        return trial_channel_zscore(array)


TrialSequenceDataset = _PackageTrialSequenceDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        help="ERP-CORE root containing sub-*/eeg/*_task-ERN_eeg.set.",
    )
    parser.add_argument(
        "--config", default="configs/erp_core_flankers_2.yaml",
        help="YAML config path (default: configs/erp_core_flankers_2.yaml).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Subject processes for sequence extraction (default: 1).",
    )
    parser.add_argument("--random-state", type=int, help="Override split seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached outputs.")
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def _project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def trial_channel_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each trial/channel independently without cross-sample statistics."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected 3-D sequence array, got {values.shape}")
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    normalized = (values - mean) / np.maximum(std, eps)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _source_files(recording: Path) -> list[Path]:
    paths = [recording]
    sidecar = recording.with_suffix(".fdt")
    if sidecar.is_file():
        paths.append(sidecar)
    return paths


def _distributed_steps(cfg: dict) -> dict:
    try:
        steps = cfg["erp_core"]["distributed_components"]["steps"]
    except KeyError as exc:
        raise KeyError("Missing erp_core.distributed_components.steps config") from exc
    if tuple(steps) != STEP_NAMES:
        raise ValueError(f"Script 14 ECMAD steps must be ordered as {STEP_NAMES}")
    return steps


def build_step_config(cfg: dict, step_name: str) -> dict:
    """Build one serial ECMAD step without mutating the shared config."""
    steps = _distributed_steps(cfg)
    try:
        step = steps[step_name]
    except KeyError as exc:
        raise ValueError(f"Unknown ECMAD step: {step_name}") from exc
    local = deepcopy(cfg)
    local["channels"]["channel_groups"] = [list(group) for group in step["channel_groups"]]
    for key in (
        "mode", "phase_gate_threshold_rad", "protected_band_hz",
        "coherent_gate_enabled", "coherent_gate_threshold_uv",
    ):
        local["wiener"][key] = deepcopy(step[key])
    missing = sorted(
        set(channel for group in local["channels"]["channel_groups"] for channel in group)
        - set(ERP_ERN_CHANNELS)
    )
    if missing:
        raise ValueError(f"{step_name} uses channels outside ERP-CORE 30: {missing}")
    return local


def build_ica_wiener_config(cfg: dict) -> dict:
    """Build the raw-ICA Wiener branch from its recorded experiment settings."""
    try:
        branch = cfg["erp_core"]["component_eegnet"]["ica_wiener"]
    except KeyError as exc:
        raise KeyError("Missing erp_core.component_eegnet.ica_wiener config") from exc
    local = deepcopy(cfg)
    for key in (
        "mode", "phase_gate_threshold_rad", "protected_band_hz",
        "coherent_gate_enabled", "coherent_gate_threshold_uv",
    ):
        local["wiener"][key] = deepcopy(branch[key])
    return local


def _wiener_parameter_record(local_cfg: dict) -> dict:
    wiener = local_cfg["wiener"]
    return {
        "mode": wiener["mode"],
        "phase_gate_threshold_rad": float(wiener["phase_gate_threshold_rad"]),
        "freq_band": [float(value) for value in wiener["freq_band"]],
        "protected_band_hz": (
            [float(value) for value in wiener["protected_band_hz"]]
            if wiener.get("protected_band_hz") is not None
            else None
        ),
        "coherent_gate_enabled": bool(wiener["coherent_gate_enabled"]),
        "coherent_gate_threshold_uv": float(wiener["coherent_gate_threshold_uv"]),
        "channel_groups": [
            list(group) for group in local_cfg["channels"]["channel_groups"]
        ],
    }


def _cache_fingerprint(recording: Path, cfg: dict) -> str:
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
        "distributed_steps": _distributed_steps(cfg),
        "ica_wiener": cfg["erp_core"]["component_eegnet"]["ica_wiener"],
        "ern": cfg["erp_core"]["ern"],
        "response_pairing_window_sec": cfg["erp_core"]["response_pairing_window_sec"],
        "standard_ica": cfg["erp_core"]["standard_ica"],
        "sequence_channels": ERP_ERN_CHANNELS,
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(cache_root: Path, subject_id: str) -> Path:
    return cache_root / subject_id / "sequences.npz"


def _load_subject_cache(path: Path, fingerprint: str) -> tuple[SequenceDataset, dict] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            return None
        sequences = {
            condition: np.asarray(data[f"X_{condition}"], dtype=np.float32)
            for condition in BASE_CONDITIONS
        }
        dataset = SequenceDataset(
            sequences=sequences,
            y=np.asarray(data["y"], dtype=np.int8),
            subject_ids=np.asarray(data["subject_ids"]).astype(str),
            samples=np.asarray(data["samples"], dtype=np.int64),
        )
        diagnostics = json.loads(str(data["diagnostics"].item()))
    return dataset, diagnostics


def _save_subject_cache(path: Path, dataset: SequenceDataset, fingerprint: str, diagnostics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    arrays: dict[str, Any] = {
        f"X_{condition}": dataset.sequences[condition].astype(np.float32)
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


def _active_channels(diagnostics: dict) -> set[str]:
    active: set[str] = set()
    for window in diagnostics.get("window_diagnostics", []):
        active.update(str(channel) for channel in window.get("channel_sources", {}))
    return active


def _derive_components(source, candidate_specific, diagnostics: dict):
    """Create an exactly conserving pair and zero never-active coherent channels."""
    if source.ch_names != candidate_specific.ch_names or source.n_times != candidate_specific.n_times:
        raise ValueError("Component source and specific output are not time-aligned")
    if not np.isclose(source.info["sfreq"], candidate_specific.info["sfreq"]):
        raise ValueError("Component source and specific output use different sampling rates")
    source_data = source.get_data()
    specific = candidate_specific.copy()
    specific_data = specific.get_data()
    coherent_data = source_data - specific_data
    active = _active_channels(diagnostics)
    inactive_indices = [
        index for index, channel in enumerate(source.ch_names) if channel not in active
    ]
    if inactive_indices:
        specific_data[inactive_indices] = source_data[inactive_indices]
        coherent_data[inactive_indices] = 0.0
    specific._data = specific_data
    coherent = source.copy()
    coherent._data = coherent_data
    error_uv = float(
        np.max(np.abs(source_data - specific.get_data() - coherent.get_data())) * 1e6
    )
    return specific, coherent, sorted(active), error_uv


def _compact_wiener_diagnostics(
    diagnostics: dict,
    local_cfg: dict,
    active_channels: list[str],
    all_channels: Sequence[str],
    conservation_error_uv: float,
) -> dict:
    compact = {
        key: value for key, value in diagnostics.items() if key != "window_diagnostics"
    }
    compact.update({
        "input_parameters": _wiener_parameter_record(local_cfg),
        "active_channels": active_channels,
        "inactive_channels": [
            channel for channel in all_channels if channel not in active_channels
        ],
        "max_abs_conservation_error_uv": conservation_error_uv,
    })
    return compact


def build_continuous_branches(common, cfg: dict, subject_id: str, helpers: Any):
    """Build all eight stored branches from raw continuous data."""
    branches = {"raw": common}
    step_diagnostics: dict[str, dict] = {}
    current = common
    for step_name in STEP_NAMES:
        local_cfg = build_step_config(cfg, step_name)
        candidate_specific, diagnostics = helpers._wiener_continuous(
            current, local_cfg, f"{subject_id}_{step_name}"
        )
        specific, coherent, active, error_uv = _derive_components(
            current, candidate_specific, diagnostics
        )
        np.testing.assert_allclose(
            specific.get_data() + coherent.get_data(), current.get_data(),
            rtol=1e-7, atol=1e-12,
        )
        branches[f"{step_name}_specific"] = specific
        branches[f"{step_name}_coherent"] = coherent
        step_diagnostics[step_name] = _compact_wiener_diagnostics(
            diagnostics, local_cfg, active, current.ch_names, error_uv
        )
        current = specific

    ica_raw, excluded = helpers._standard_ica(common, cfg)
    ica_cfg = build_ica_wiener_config(cfg)
    candidate_specific, diagnostics = helpers._wiener_continuous(
        ica_raw, ica_cfg, f"{subject_id}_ica"
    )
    ica_specific, ica_coherent, active, error_uv = _derive_components(
        ica_raw, candidate_specific, diagnostics
    )
    np.testing.assert_allclose(
        ica_specific.get_data() + ica_coherent.get_data(), ica_raw.get_data(),
        rtol=1e-7, atol=1e-12,
    )
    branches.update({
        "ica": ica_raw,
        "ica_wiener_specific": ica_specific,
        "ica_wiener_coherent": ica_coherent,
    })
    ica_diagnostics = _compact_wiener_diagnostics(
        diagnostics, ica_cfg, active, ica_raw.ch_names, error_uv
    )
    return branches, step_diagnostics, ica_diagnostics, excluded


def _epochs_to_sequences(epochs) -> np.ndarray:
    missing = [channel for channel in ERP_ERN_CHANNELS if channel not in epochs.ch_names]
    if missing:
        raise ValueError(f"Missing ERP ERN sequence channels: {missing}")
    data_uv = epochs.get_data(picks=list(ERP_ERN_CHANNELS), copy=False) * 1e6
    return np.asarray(data_uv, dtype=np.float32)


def _validate_shared_epoch_selection(
    condition: str,
    epochs,
    labels: np.ndarray,
    samples: np.ndarray,
) -> None:
    branch_labels = (~epochs.metadata["correct"].to_numpy(bool)).astype(np.int8)
    branch_samples = np.asarray(epochs.events[:, 0], dtype=np.int64)
    if not np.array_equal(branch_labels, labels) or not np.array_equal(
        branch_samples, samples
    ):
        raise RuntimeError(f"{condition} changed shared ERN trial selection")


def extract_subject_sequences(
    subject_id: str,
    recording: Path,
    cfg: dict,
    cache_root: Path,
    helpers: Any,
    force: bool,
) -> tuple[SequenceDataset, dict, bool]:
    fingerprint = _cache_fingerprint(recording, cfg)
    path = _cache_path(cache_root, subject_id)
    if not force:
        cached = _load_subject_cache(path, fingerprint)
        if cached is not None:
            return cached[0], cached[1], True

    import mne

    original = helpers._read_recording(mne, recording)
    common = helpers._common_preprocess(original, cfg)
    missing = [channel for channel in ERP_ERN_CHANNELS if channel not in common.ch_names]
    if missing:
        raise ValueError(f"Missing ERP-CORE 30-channel input: {missing}")
    common.pick(list(ERP_ERN_CHANNELS))
    common.reorder_channels(list(ERP_ERN_CHANNELS))
    events, event_id = mne.events_from_annotations(common, verbose=False)
    table = helpers.build_response_table(
        events,
        event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    branches, step_diagnostics, ica_diagnostics, excluded = build_continuous_branches(
        common, cfg, subject_id, helpers
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
    sequences: dict[str, np.ndarray] = {}
    for condition in BASE_CONDITIONS:
        sequences[condition] = _epochs_to_sequences(epochs[condition])
        _validate_shared_epoch_selection(
            condition, epochs[condition], labels, samples
        )
    dataset = SequenceDataset(
        sequences=sequences,
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
        "branch_inputs": {
            "step1": "raw",
            "step2": "step1_specific",
            "ica": "raw",
            "ica_wiener": "ica",
        },
        "steps": step_diagnostics,
        "ica_wiener": ica_diagnostics,
    }
    _save_subject_cache(path, dataset, fingerprint, diagnostics)
    return dataset, diagnostics, False


def _extract_subject_worker(args: tuple[str, str, dict, str, bool], helpers: Any | None = None) -> dict:
    subject_id, recording, cfg, cache_root, force = args
    if helpers is None:
        helpers = _load_script10()
    with threadpool_limits(limits=1):
        try:
            dataset, diagnostics, cached = extract_subject_sequences(
                subject_id, Path(recording), cfg, Path(cache_root), helpers, force
            )
        except Exception as exc:
            return {
                "subject_id": subject_id, "recording": recording, "eligible": False,
                "cached": False, "n_trials": 0, "n_correct": 0, "n_incorrect": 0,
                "reason": str(exc),
            }
    counts = np.bincount(dataset.y, minlength=2)
    return {
        "subject_id": subject_id, "recording": recording, "eligible": True,
        "cached": cached, "n_trials": len(dataset.y),
        "n_correct": int(counts[0]), "n_incorrect": int(counts[1]),
        "reason": "", "diagnostics": diagnostics,
    }


def extract_all_subjects(
    recordings: list[dict[str, Path | str]], cfg: dict, cache_root: Path,
    force: bool, workers: int,
) -> list[dict]:
    jobs = [
        (str(item["subject_id"]), str(item["ern"]), cfg, str(cache_root), force)
        for item in recordings
    ]
    if not jobs:
        return []
    if workers == 1:
        helpers = _load_script10()
        return [
            _extract_subject_worker(job, helpers)
            for job in jobs
        ]
    rows_by_subject: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")) as executor:
        futures = {executor.submit(_extract_subject_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                rows_by_subject[subject_id] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Sequence generation failed for {subject_id}") from exc
    return [rows_by_subject[job[0]] for job in jobs]


def load_sequence_dataset(
    subject_ids: Sequence[str], recordings_by_subject: dict[str, Path],
    cfg: dict, cache_root: Path,
) -> SequenceDataset:
    datasets: list[SequenceDataset] = []
    for subject_id in subject_ids:
        recording = recordings_by_subject[subject_id]
        cached = _load_subject_cache(
            _cache_path(cache_root, subject_id), _cache_fingerprint(recording, cfg)
        )
        if cached is None:
            raise FileNotFoundError(f"Missing or stale sequence cache: {subject_id}")
        datasets.append(cached[0])
    if not datasets:
        raise ValueError("No eligible subject caches were loaded")
    return SequenceDataset(
        sequences={
            condition: np.concatenate([item.sequences[condition] for item in datasets], axis=0)
            for condition in BASE_CONDITIONS
        },
        y=np.concatenate([item.y for item in datasets]),
        subject_ids=np.concatenate([item.subject_ids for item in datasets]),
        samples=np.concatenate([item.samples for item in datasets]),
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
    holdout = sorted(str(value) for value in shuffled[:n_holdout])
    keep = sorted(str(value) for value in shuffled[n_holdout:])
    return keep, holdout


def split_subjects_two_stage(
    subject_ids: Sequence[str], test_size: float = 0.2,
    validation_size: float = 0.2, random_state: int = 42,
) -> dict[str, list[str]]:
    """Return deterministic train/validation/test subject partitions."""
    train_pool, test = _split_ids(subject_ids, test_size, random_state)
    train, validation = _split_ids(
        train_pool, validation_size, random_state + _INNER_SPLIT_SEED_OFFSET
    )
    partitions = {"train": train, "validation": validation, "test": test}
    sets = {key: set(value) for key, value in partitions.items()}
    if any(sets[left] & sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Subject leakage between train, validation, and test")
    if any(not value for value in partitions.values()):
        raise RuntimeError("Two-stage subject split produced an empty partition")
    return partitions


def _condition_indices(dataset: SequenceDataset, partitions: dict[str, list[str]]) -> dict[str, np.ndarray]:
    return {
        split: np.flatnonzero(np.isin(dataset.subject_ids, np.asarray(subjects, dtype=str)))
        for split, subjects in partitions.items()
    }


def _safe_auc(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else float("nan")


def _safe_average_precision(y: np.ndarray, probabilities: np.ndarray) -> float:
    return float(average_precision_score(y, probabilities)) if len(y) else float("nan")


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    return _package_classification_metrics(y, probabilities, threshold)


def select_balanced_accuracy_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    """Select threshold by validation balanced accuracy with deterministic ties."""
    return _package_select_threshold(y, probabilities)


def _predict_trials(model: EEGNet, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    return _package_predict_trials(model, loader, device)


def _make_model(n_channels: int, n_times: int, cfg: dict) -> EEGNet:
    return _package_make_model(n_channels, n_times, cfg)


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


def _legacy_train_condition(
    condition: str, dataset: SequenceDataset, partitions: dict[str, list[str]],
    model_cfg: dict, out_dir: Path, random_state: int,
) -> dict[str, Any]:
    indices = _condition_indices(dataset, partitions)
    X = dataset.matrix(condition, normalize=True)
    n_channels, n_times = X.shape[1:]
    train_index, validation_index, test_index = (
        indices["train"], indices["validation"], indices["test"]
    )
    if len(np.unique(dataset.y[train_index])) < 2:
        raise ValueError(f"Condition {condition} training split must contain both ERN classes")
    if len(np.unique(dataset.y[validation_index])) < 2:
        raise ValueError(f"Condition {condition} validation split must contain both ERN classes")
    train_loader = DataLoader(
        TrialSequenceDataset(X[train_index], dataset.y[train_index], dataset.subject_ids[train_index]),
        batch_size=int(model_cfg.get("batch_size", 64)), shuffle=True,
        num_workers=int(model_cfg.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(random_state),
    )
    validation_loader = DataLoader(
        TrialSequenceDataset(X[validation_index], dataset.y[validation_index], dataset.subject_ids[validation_index]),
        batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False,
        num_workers=int(model_cfg.get("num_workers", 0)),
    )
    test_loader = DataLoader(
        TrialSequenceDataset(X[test_index], dataset.y[test_index], dataset.subject_ids[test_index]),
        batch_size=int(model_cfg.get("batch_size", 64)), shuffle=False,
        num_workers=int(model_cfg.get("num_workers", 0)),
    )
    device_name = str(model_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("EEGNet device 'cuda' requested but CUDA is unavailable")
    if device_name.startswith("mps"):
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("EEGNet device 'mps' requested but MPS is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    model = _make_model(n_channels, n_times, model_cfg).to(device)
    counts = np.bincount(dataset.y[train_index].astype(int), minlength=2)
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = Adam(
        model.parameters(),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=float(model_cfg.get("lr_factor", 0.5)),
        patience=int(model_cfg.get("lr_patience", 10)),
    )
    max_epochs = int(model_cfg.get("max_epochs", 200))
    patience = int(model_cfg.get("patience", 20))
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for tensors, labels, _ in train_loader:
            labels = labels.float().to(device)
            predictions = model(tensors.to(device)).squeeze(1)
            weights = torch.where(labels == 1, pos_weight.expand_as(labels), torch.ones_like(labels))
            loss = (criterion(predictions, labels) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_predictions = _predict_trials(model, validation_loader, device)
        validation_score = _safe_average_precision(
            validation_predictions["true_label"].to_numpy(),
            validation_predictions["pred_proba"].to_numpy(),
        )
        scheduler.step(validation_score if np.isfinite(validation_score) else -float("inf"))
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_auprc": validation_score,
        })
        if np.isfinite(validation_score) and validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break
    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = len(history)
        best_score = float(history[-1]["validation_auprc"]) if history else float("nan")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    validation_predictions = _predict_trials(model, validation_loader, device)
    threshold = select_balanced_accuracy_threshold(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(),
    )
    test_predictions = _predict_trials(model, test_loader, device)
    validation_metrics = classification_metrics(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(), threshold,
    )
    test_metrics = classification_metrics(
        test_predictions["true_label"].to_numpy(),
        test_predictions["pred_proba"].to_numpy(), threshold,
    )
    validation_predictions["condition"] = condition
    validation_predictions["split"] = "validation"
    validation_predictions["predicted_label"] = (
        validation_predictions["pred_proba"] >= threshold
    ).astype(np.int8)
    test_predictions["condition"] = condition
    test_predictions["split"] = "test"
    test_predictions["predicted_label"] = (test_predictions["pred_proba"] >= threshold).astype(np.int8)
    validation_predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    _save_json(validation_metrics, out_dir / "val_metrics.json")
    _save_json(test_metrics, out_dir / "test_metrics.json")
    _save_json(
        {
            "condition": condition,
            "n_channels": n_channels,
            "n_times": n_times,
            "F1": int(model_cfg.get("F1", 8)),
            "D": int(model_cfg.get("D", 2)),
            "dropout": float(model_cfg.get("dropout", 0.25)),
            "lr": float(model_cfg.get("lr", 1e-3)),
            "batch_size": int(model_cfg.get("batch_size", 64)),
            "best_epoch": best_epoch,
            "best_validation_auprc": best_score,
            "threshold_selection": "validation_balanced_accuracy",
        },
        out_dir / "best_params.json",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {
        "condition": condition,
        "n_channels": n_channels,
        "n_times": n_times,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_score,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
    }
    return summary


def train_condition(
    condition: str, dataset: SequenceDataset, partitions: dict[str, list[str]],
    model_cfg: dict[str, Any], out_dir: Path, random_state: int,
) -> dict[str, Any]:
    """Compatibility entry point backed by :mod:`eeg_bg.ml.erp_eegnet`."""
    return _package_train_condition(
        condition, dataset, partitions, model_cfg, out_dir, random_state
    )


def _manifest(dataset: SequenceDataset, partitions: dict[str, list[str]]) -> pd.DataFrame:
    split_by_subject = {
        subject_id: split for split, subjects in partitions.items() for subject_id in subjects
    }
    rows = []
    for subject_id in sorted(set(dataset.subject_ids)):
        mask = dataset.subject_ids == subject_id
        rows.append({
            "subject_id": subject_id,
            "split": split_by_subject[subject_id],
            "n_trials": int(mask.sum()),
            "n_correct": int(np.count_nonzero(dataset.y[mask] == 0)),
            "n_incorrect": int(np.count_nonzero(dataset.y[mask] == 1)),
        })
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
    experiment_cfg = cfg["erp_core"]["component_eegnet"]
    test_size = float(experiment_cfg.get("test_size", 0.2))
    validation_size = float(experiment_cfg.get("validation_size", 0.2))
    random_state = int(random_state_override) if random_state_override is not None else int(experiment_cfg.get("random_state", 42))
    out = output_dir.expanduser().resolve() if output_dir is not None else _project_path(experiment_cfg["output_dir"], config_path)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output directory is not empty: {out}; pass --force to overwrite files")
    out.mkdir(parents=True, exist_ok=True)
    cache_root = _project_path(
        experiment_cfg.get(
            "cache_subdir", "cache/erp_core_ern_eegnet_2step_30ch"
        ),
        config_path,
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    helpers = _load_script10()
    source_root = data_dir.expanduser().resolve() if data_dir is not None else Path(cfg["erp_core"]["data_dir"]).expanduser().resolve()
    recordings = helpers._select_recordings(
        helpers._resolve_recordings(source_root, None), subjects=None, task="ern"
    )
    recordings_by_subject = {str(item["subject_id"]): Path(item["ern"]) for item in recordings}
    eligibility_rows = extract_all_subjects(recordings, cfg, cache_root, force, workers)
    pd.DataFrame([{key: value for key, value in row.items() if key != "diagnostics"} for row in eligibility_rows]).to_csv(out / "eligibility.csv", index=False)
    eligible_subjects = sorted(str(row["subject_id"]) for row in eligibility_rows if row["eligible"])
    dataset = load_sequence_dataset(eligible_subjects, recordings_by_subject, cfg, cache_root)
    partitions = split_subjects_two_stage(eligible_subjects, test_size, validation_size, random_state)
    indices = _condition_indices(dataset, partitions)
    if any(len(index) == 0 for index in indices.values()):
        raise RuntimeError("Two-stage subject split produced an empty trial partition")
    if len(np.unique(dataset.y[indices["train"]])) < 2:
        raise ValueError("Training split must contain both ERN classes")
    manifest = _manifest(dataset, partitions)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    _save_json(
        {
            "random_state": random_state,
            "test_size": test_size,
            "validation_size_within_train_pool": validation_size,
            "inner_split_seed": random_state + _INNER_SPLIT_SEED_OFFSET,
            "train_pool_subjects": sorted(partitions["train"] + partitions["validation"]),
            "train_subjects": partitions["train"],
            "validation_subjects": partitions["validation"],
            "test_subjects": partitions["test"],
        },
        out / "split_manifest.json",
    )
    model_cfg = dict(experiment_cfg)
    condition_summaries = []
    prediction_frames = []
    subject_frames = []
    for index, condition in enumerate(CONDITIONS, start=1):
        print(f"[EEGNet {index}/{len(CONDITIONS)}] {condition}")
        condition_summary = train_condition(
            condition,
            dataset,
            partitions,
            model_cfg,
            out / "conditions" / condition,
            random_state + index,
        )
        test_metrics = condition_summary["test_metrics"]
        val_metrics = condition_summary["validation_metrics"]
        condition_summaries.append({
            "condition": condition,
            "n_channels": condition_summary["n_channels"],
            "n_times": condition_summary["n_times"],
            "n_train_subjects": len(partitions["train"]),
            "n_validation_subjects": len(partitions["validation"]),
            "n_test_subjects": len(partitions["test"]),
            "n_train_trials": len(indices["train"]),
            "n_validation_trials": len(indices["validation"]),
            "n_test_trials": len(indices["test"]),
            "best_epoch": condition_summary["best_epoch"],
            "best_validation_auprc": condition_summary["best_validation_auprc"],
            "validation_threshold": condition_summary["threshold"],
            "validation_auprc": val_metrics["auprc"],
            **{f"test_{key}": value for key, value in test_metrics.items() if key != "confusion_matrix"},
            "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        })
        validation_frame = condition_summary["validation_predictions"].copy()
        test_frame = condition_summary["test_predictions"].copy()
        prediction_frames.extend([validation_frame, test_frame])
        for split, frame in (("validation", validation_frame), ("test", test_frame)):
            subject = _subject_predictions(frame, condition_summary["threshold"])
            subject["condition"] = condition
            subject["split"] = split
            subject_frames.append(subject)
    condition_metrics = pd.DataFrame(condition_summaries)
    condition_metrics.to_csv(out / "condition_metrics.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.concat(subject_frames, ignore_index=True).to_csv(out / "subject_metrics.csv", index=False)
    resolved_cfg = deepcopy(cfg)
    resolved_cfg["erp_core"]["component_eegnet"]["random_state"] = random_state
    resolved_cfg["erp_core"]["component_eegnet"]["grid_search"] = False
    resolved_cfg["erp_core"]["component_eegnet"]["early_stopping_metric"] = "validation_auprc"
    resolved_cfg["erp_core"]["component_eegnet"]["threshold_selection"] = "validation_balanced_accuracy"
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _save_json(
        {
            "conditions": list(CONDITIONS),
            "subject_level_split": True,
            "test_size": test_size,
            "validation_size_within_train_pool": validation_size,
            "actual_split_subject_counts": {key: len(value) for key, value in partitions.items()},
            "no_grid_search": True,
            "early_stopping_metric": "validation_auprc",
            "threshold_selection": "validation_balanced_accuracy",
            "final_test_evaluation": "single_frozen_evaluation",
            "normalization": "trial_channel_zscore",
            "stored_conditions": list(BASE_CONDITIONS),
            "combined_condition": {
                "name": COMBINED_CONDITION,
                "operation": "channel_axis_concatenation",
                "inputs": ["step1_specific", "step1_coherent"],
            },
            "branch_inputs": {
                "step1": "raw",
                "step2": "step1_specific",
                "ica": "raw",
                "ica_wiener": "ica",
            },
            "component_parameters": {
                step_name: _wiener_parameter_record(build_step_config(cfg, step_name))
                for step_name in STEP_NAMES
            } | {
                "ica_wiener": _wiener_parameter_record(
                    build_ica_wiener_config(cfg)
                )
            },
            "input_shapes": {
                condition: [len(ERP_ERN_CHANNELS) * (2 if condition == COMBINED_CONDITION else 1), int(dataset.matrix(condition, normalize=False).shape[-1])]
                for condition in CONDITIONS
            },
            "comparison_summary": condition_metrics.to_dict(orient="records"),
        },
        out / "run_summary.json",
    )
    print(f"ERP-CORE ERN EEGNet comparison complete: {out}")
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
