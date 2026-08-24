#!/usr/bin/env python3
"""Estimate task-relevant information in three serial ECMAD decompositions.

The experiment uses the ERP-CORE ERN task and the EEGNet training discipline
from Script 14.  Three independently configured ECMAD steps are applied in
series.  Each step's specific output becomes the next step's input, while the
coherent condition contains only the component newly separated at that step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import matthews_corrcoef
from threadpoolctl import threadpool_limits

from eeg_bg.config.settings import load_config
from eeg_bg.ml.erp_eegnet import train_condition as _package_train_condition


ERP_CORE_EEG_CHANNELS: tuple[str, ...] = (
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7",
    "PO3", "O1", "Oz", "Pz", "CPz", "FP2", "Fz", "F4", "F8", "FC4",
    "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2",
)
STEP_NAMES: tuple[str, ...] = ("step1", "step2", "step3")
COMPONENT_NAMES: tuple[str, ...] = tuple(
    f"{step}_{component}"
    for step in STEP_NAMES
    for component in ("specific", "coherent")
)
CONDITIONS: tuple[str, ...] = ("raw", *COMPONENT_NAMES)
_CACHE_SCHEMA = 2
_METRIC_NAMES: tuple[str, ...] = (
    "auprc", "auroc", "f1", "precision", "recall", "specificity",
    "balanced_accuracy", "mcc", "accuracy",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        help="ERP-CORE root containing sub-*/eeg/*_task-ERN_eeg.set.",
    )
    parser.add_argument(
        "--config", default="configs/erp_core_flankers.yaml",
        help="YAML config path (default: configs/erp_core_flankers.yaml).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Subject processes for component generation (default: 1).",
    )
    parser.add_argument("--random-state", type=int, help="Override split/model seed.")
    parser.add_argument(
        "--force", action="store_true",
        help="Replace result/model outputs while retaining valid component caches.",
    )
    parser.add_argument(
        "--recompute-components", action="store_true",
        help="Ignore valid subject caches and regenerate continuous components.",
    )
    return parser.parse_args()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_script14():
    return _load_module(
        "_erp_component_eegnet_helpers",
        Path(__file__).with_name("14_compare_erp_core_ern_components_eegnet.py"),
    )


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _project_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent.parent / path
    return path.resolve()


def _source_files(recording: Path) -> list[Path]:
    paths = [recording]
    sidecar = recording.with_suffix(".fdt")
    if sidecar.is_file():
        paths.append(sidecar)
    return paths


def _experiment_cfg(cfg: dict) -> dict:
    try:
        experiment = cfg["erp_core"]["distributed_component_eegnet"]
    except KeyError as exc:
        raise KeyError("Missing erp_core.distributed_component_eegnet config") from exc
    return experiment


def _distributed_cfg(cfg: dict) -> dict:
    try:
        distributed = cfg["erp_core"]["distributed_components"]
    except KeyError as exc:
        raise KeyError("Missing erp_core.distributed_components config") from exc
    steps = distributed.get("steps", {})
    if tuple(steps) != STEP_NAMES:
        raise ValueError(f"Distributed ECMAD steps must be ordered as {STEP_NAMES}")
    return distributed


def build_step_config(cfg: dict, step_name: str) -> dict:
    """Return one complete ECMAD config without mutating the shared config."""
    distributed = _distributed_cfg(cfg)
    try:
        step = distributed["steps"][step_name]
    except KeyError as exc:
        raise ValueError(f"Unknown distributed ECMAD step: {step_name}") from exc
    local = deepcopy(cfg)
    local["channels"]["channel_groups"] = [list(group) for group in step["channel_groups"]]
    for key in (
        "mode", "phase_gate_threshold_rad", "protected_band_hz",
        "coherent_gate_enabled", "coherent_gate_threshold_uv",
    ):
        local["wiener"][key] = deepcopy(step[key])
    missing = sorted(
        set(channel for group in local["channels"]["channel_groups"] for channel in group)
        - set(ERP_CORE_EEG_CHANNELS)
    )
    if missing:
        raise ValueError(f"{step_name} uses channels outside ERP-CORE 30: {missing}")
    return local


def _component_fingerprint(recording: Path, cfg: dict) -> str:
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
        "line_freq": cfg["erp_core"].get("line_freq"),
        "base_wiener": cfg["wiener"],
        "steps": _distributed_cfg(cfg)["steps"],
        "channels": ERP_CORE_EEG_CHANNELS,
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_fingerprint(recording: Path, cfg: dict) -> str:
    payload = {
        "schema": _CACHE_SCHEMA,
        "component_fingerprint": _component_fingerprint(recording, cfg),
        "ern": cfg["erp_core"]["ern"],
        "response_pairing_window_sec": cfg["erp_core"]["response_pairing_window_sec"],
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _subject_cache_dir(cache_root: Path, subject_id: str) -> Path:
    return cache_root / subject_id


def _sequence_cache_path(cache_root: Path, subject_id: str) -> Path:
    return _subject_cache_dir(cache_root, subject_id) / "sequences.npz"


def _component_cache_paths(cache_root: Path, subject_id: str) -> dict[str, Path]:
    root = _subject_cache_dir(cache_root, subject_id) / "continuous"
    return {name: root / f"{name}.edf" for name in COMPONENT_NAMES}


def _component_metadata_path(cache_root: Path, subject_id: str) -> Path:
    return _subject_cache_dir(cache_root, subject_id) / "continuous" / "cache.json"


def _load_subject_cache(
    cache_root: Path, subject_id: str, fingerprint: str, base: Any,
) -> tuple[Any, dict] | None:
    path = _sequence_cache_path(cache_root, subject_id)
    component_paths = _component_cache_paths(cache_root, subject_id)
    if (
        not path.is_file()
        or not _component_metadata_path(cache_root, subject_id).is_file()
        or any(not item.is_file() for item in component_paths.values())
    ):
        return None
    with np.load(path, allow_pickle=False) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            return None
        sequences = {
            condition: np.asarray(data[f"X_{condition}"], dtype=np.float32)
            for condition in CONDITIONS
        }
        dataset = base.SequenceDataset(
            sequences=sequences,
            y=np.asarray(data["y"], dtype=np.int8),
            subject_ids=np.asarray(data["subject_ids"]).astype(str),
            samples=np.asarray(data["samples"], dtype=np.int64),
        )
        diagnostics = json.loads(str(data["diagnostics"].item()))
    return dataset, diagnostics


def _save_subject_cache(
    cache_root: Path, subject_id: str, dataset: Any,
    fingerprint: str, diagnostics: dict,
) -> None:
    path = _sequence_cache_path(cache_root, subject_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            f"X_{condition}": dataset.sequences[condition].astype(np.float32)
            for condition in CONDITIONS
        },
        y=dataset.y.astype(np.int8),
        subject_ids=dataset.subject_ids.astype("U"),
        samples=dataset.samples.astype(np.int64),
        fingerprint=np.asarray(fingerprint),
        diagnostics=np.asarray(json.dumps(_jsonable(diagnostics), sort_keys=True)),
    )
    temporary.replace(path)
    _save_json(diagnostics, _subject_cache_dir(cache_root, subject_id) / "diagnostics.json")


def _active_channels(diagnostics: dict) -> set[str]:
    active: set[str] = set()
    for window in diagnostics.get("window_diagnostics", []):
        active.update(str(channel) for channel in window.get("channel_sources", {}))
    return active


def _derive_step_components(source, candidate_specific, diagnostics: dict):
    """Enforce exact zero coherent data for channels never processed."""
    if source.ch_names != candidate_specific.ch_names or source.n_times != candidate_specific.n_times:
        raise ValueError("Step input and specific output are not time-aligned")
    if not np.isclose(source.info["sfreq"], candidate_specific.info["sfreq"]):
        raise ValueError("Step input and specific output use different sampling rates")
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


def _compact_step_diagnostics(
    diagnostics: dict,
    active_channels: list[str],
    all_channels: Sequence[str],
    conservation_error_uv: float,
) -> dict:
    compact = {
        key: value for key, value in diagnostics.items() if key != "window_diagnostics"
    }
    compact.update({
        "active_channels": active_channels,
        "inactive_channels": [
            channel for channel in all_channels if channel not in active_channels
        ],
        "max_abs_step_conservation_error_uv": conservation_error_uv,
    })
    return compact


def run_distributed_cascade(common, cfg: dict, subject_id: str, helpers: Any):
    """Return raw plus six non-cumulative serial ECMAD component branches."""
    branches = {"raw": common}
    step_diagnostics: dict[str, dict] = {}
    current = common
    for step_name in STEP_NAMES:
        local_cfg = build_step_config(cfg, step_name)
        candidate_specific, diagnostics = helpers._wiener_continuous(
            current, local_cfg, f"{subject_id}_{step_name}"
        )
        specific, coherent, active, error_uv = _derive_step_components(
            current, candidate_specific, diagnostics
        )
        np.testing.assert_allclose(
            specific.get_data() + coherent.get_data(), current.get_data(),
            rtol=1e-7, atol=1e-12,
        )
        branches[f"{step_name}_specific"] = specific
        branches[f"{step_name}_coherent"] = coherent
        step_diagnostics[step_name] = _compact_step_diagnostics(
            diagnostics, active, current.ch_names, error_uv
        )
        current = specific
    cumulative = (
        branches["step3_specific"].get_data()
        + branches["step1_coherent"].get_data()
        + branches["step2_coherent"].get_data()
        + branches["step3_coherent"].get_data()
    )
    cumulative_error_uv = float(np.max(np.abs(common.get_data() - cumulative)) * 1e6)
    np.testing.assert_allclose(cumulative, common.get_data(), rtol=1e-7, atol=2e-12)
    return branches, step_diagnostics, cumulative_error_uv


def _write_component_edfs(branches: dict, cache_root: Path, subject_id: str) -> None:
    from eeg_bg.application.models import OutputFormat
    from eeg_bg.application.recording import RecordingService

    service = RecordingService(standard_channels=list(ERP_CORE_EEG_CHANNELS))
    for component, path in _component_cache_paths(cache_root, subject_id).items():
        service.write(branches[component], path, OutputFormat.EDF)


def _load_component_cache(
    common,
    cache_root: Path,
    subject_id: str,
    fingerprint: str,
) -> tuple[dict[str, Any], dict] | None:
    metadata_path = _component_metadata_path(cache_root, subject_id)
    component_paths = _component_cache_paths(cache_root, subject_id)
    if not metadata_path.is_file() or any(
        not path.is_file() for path in component_paths.values()
    ):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("component_fingerprint") != fingerprint:
        return None

    import mne

    branches: dict[str, Any] = {"raw": common}
    for component, path in component_paths.items():
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if raw.ch_names != common.ch_names:
            raise ValueError(
                f"Shared component cache channel mismatch for {subject_id}/{component}"
            )
        if raw.n_times != common.n_times or not np.isclose(
            raw.info["sfreq"], common.info["sfreq"]
        ):
            raise ValueError(
                f"Shared component cache time mismatch for {subject_id}/{component}"
            )
        branches[component] = raw
    return branches, metadata


def load_or_create_distributed_components(
    common,
    recording: Path,
    cfg: dict,
    subject_id: str,
    helpers: Any,
    cache_root: Path,
    recompute_components: bool = False,
) -> tuple[dict[str, Any], dict, bool]:
    """Load or generate the shared continuous six-component cache."""
    fingerprint = _component_fingerprint(recording, cfg)
    if not recompute_components:
        cached = _load_component_cache(common, cache_root, subject_id, fingerprint)
        if cached is not None:
            return cached[0], cached[1], True

    branches, step_diagnostics, cumulative_error_uv = run_distributed_cascade(
        common, cfg, subject_id, helpers
    )
    _write_component_edfs(branches, cache_root, subject_id)
    metadata = {
        "subject_id": subject_id,
        "recording": str(recording),
        "component_fingerprint": fingerprint,
        "sfreq": float(common.info["sfreq"]),
        "n_times": int(common.n_times),
        "duration_sec": float(common.n_times / common.info["sfreq"]),
        "n_channels": len(common.ch_names),
        "channels": list(common.ch_names),
        "steps": step_diagnostics,
        "max_abs_cumulative_conservation_error_uv": cumulative_error_uv,
    }
    _save_json(metadata, _component_metadata_path(cache_root, subject_id))
    cached = _load_component_cache(common, cache_root, subject_id, fingerprint)
    if cached is None:
        raise RuntimeError(f"Failed to reload shared component cache: {subject_id}")
    return cached[0], cached[1], False


def _epochs_to_sequences(epochs) -> np.ndarray:
    missing = [channel for channel in ERP_CORE_EEG_CHANNELS if channel not in epochs.ch_names]
    if missing:
        raise ValueError(f"Missing ERP-CORE model channels: {missing}")
    data_uv = epochs.get_data(picks=list(ERP_CORE_EEG_CHANNELS), copy=False) * 1e6
    return np.asarray(data_uv, dtype=np.float32)


def extract_subject_sequences(
    subject_id: str,
    recording: Path,
    cfg: dict,
    cache_root: Path,
    helpers: Any,
    base: Any,
    recompute_components: bool,
) -> tuple[Any, dict, bool]:
    fingerprint = _cache_fingerprint(recording, cfg)
    if not recompute_components:
        cached = _load_subject_cache(cache_root, subject_id, fingerprint, base)
        if cached is not None:
            return cached[0], cached[1], True

    import mne

    original = helpers._read_recording(mne, recording)
    common = helpers._common_preprocess(original, cfg)
    missing = [channel for channel in ERP_CORE_EEG_CHANNELS if channel not in common.ch_names]
    if missing:
        raise ValueError(f"Missing ERP-CORE 30-channel input: {missing}")
    common.pick(list(ERP_CORE_EEG_CHANNELS))
    common.reorder_channels(list(ERP_CORE_EEG_CHANNELS))
    events, event_id = mne.events_from_annotations(common, verbose=False)
    table = helpers.build_response_table(
        events,
        event_id,
        common.info["sfreq"],
        float(cfg["erp_core"]["response_pairing_window_sec"]),
    )
    branches, component_diagnostics, components_cached = (
        load_or_create_distributed_components(
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
    sequences: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        sequences[condition] = _epochs_to_sequences(epochs[condition])
        branch_labels = (~epochs[condition].metadata["correct"].to_numpy(bool)).astype(np.int8)
        branch_samples = np.asarray(epochs[condition].events[:, 0], dtype=np.int64)
        if not np.array_equal(branch_labels, labels) or not np.array_equal(branch_samples, samples):
            raise RuntimeError(f"{condition} changed shared ERN trial selection")
    dataset = base.SequenceDataset(
        sequences=sequences,
        y=labels,
        subject_ids=np.repeat(subject_id, len(labels)),
        samples=samples,
    )
    counts = np.bincount(labels, minlength=2)
    diagnostics = {
        **component_diagnostics,
        "cache_fingerprint": fingerprint,
        "components_cached": components_cached,
        "n_trials": len(labels),
        "n_correct": int(counts[0]),
        "n_incorrect": int(counts[1]),
    }
    _save_subject_cache(cache_root, subject_id, dataset, fingerprint, diagnostics)
    return dataset, diagnostics, False


def _extract_subject_worker(args: tuple[str, str, dict, str, bool]) -> dict:
    subject_id, recording, cfg, cache_root, recompute_components = args
    base = _load_script14()
    helpers = base._load_script10()
    with threadpool_limits(limits=1):
        try:
            dataset, diagnostics, cached = extract_subject_sequences(
                subject_id,
                Path(recording),
                cfg,
                Path(cache_root),
                helpers,
                base,
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
    recompute_components: bool,
    workers: int,
) -> list[dict]:
    jobs = [
        (str(item["subject_id"]), str(item["ern"]), cfg, str(cache_root), recompute_components)
        for item in recordings
    ]
    if workers == 1:
        return [_extract_subject_worker(job) for job in jobs]
    rows_by_subject: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=get_context("spawn")
    ) as executor:
        futures = {executor.submit(_extract_subject_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            subject_id = futures[future]
            try:
                rows_by_subject[subject_id] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Component generation failed for {subject_id}") from exc
    return [rows_by_subject[job[0]] for job in jobs]


def load_sequence_dataset(
    subject_ids: Sequence[str],
    recordings_by_subject: dict[str, Path],
    cfg: dict,
    cache_root: Path,
    base: Any,
):
    datasets = []
    for subject_id in subject_ids:
        cached = _load_subject_cache(
            cache_root,
            subject_id,
            _cache_fingerprint(recordings_by_subject[subject_id], cfg),
            base,
        )
        if cached is None:
            raise FileNotFoundError(f"Missing or stale distributed component cache: {subject_id}")
        datasets.append(cached[0])
    if not datasets:
        raise ValueError("No eligible subject caches were loaded")
    return base.SequenceDataset(
        sequences={
            condition: np.concatenate(
                [dataset.sequences[condition] for dataset in datasets], axis=0
            )
            for condition in CONDITIONS
        },
        y=np.concatenate([dataset.y for dataset in datasets]),
        subject_ids=np.concatenate([dataset.subject_ids for dataset in datasets]),
        samples=np.concatenate([dataset.samples for dataset in datasets]),
    )


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize_subject_artifacts(
    subject_id: str, recording: Path, cache_root: Path, out: Path, diagnostics: dict,
) -> list[dict]:
    result_root = out / "subjects" / subject_id
    _link_or_copy(
        _subject_cache_dir(cache_root, subject_id) / "diagnostics.json",
        result_root / "processing_diagnostics.json",
    )
    rows = []
    input_by_step = {"step1": "raw", "step2": "step1_specific", "step3": "step2_specific"}
    for component, cache_path in _component_cache_paths(cache_root, subject_id).items():
        result_path = result_root / "continuous" / cache_path.name
        materialization = _link_or_copy(cache_path, result_path)
        step_name = component.split("_", 1)[0]
        rows.append({
            "subject_id": subject_id,
            "source_recording": str(recording),
            "component": component,
            "step_input": input_by_step[step_name],
            "path": str(result_path),
            "materialization": materialization,
            "sfreq": diagnostics["sfreq"],
            "n_times": diagnostics["n_times"],
            "duration_sec": diagnostics["duration_sec"],
            "n_channels": diagnostics["n_channels"],
            "cache_fingerprint": diagnostics["cache_fingerprint"],
            "max_abs_step_conservation_error_uv": diagnostics["steps"][step_name][
                "max_abs_step_conservation_error_uv"
            ],
        })
    return rows


def classification_metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float, base: Any,
) -> dict[str, Any]:
    metrics = base.classification_metrics(y, probabilities, threshold)
    predictions = (np.asarray(probabilities, dtype=float) >= threshold).astype(np.int8)
    metrics["mcc"] = float(matthews_corrcoef(np.asarray(y, dtype=np.int8), predictions))
    return metrics


def _validate_device(model_cfg: dict, base: Any) -> None:
    device = str(model_cfg.get("device", "cpu"))
    if device.startswith("cuda") and not base.torch.cuda.is_available():
        raise RuntimeError("EEGNet device 'cuda' requested but CUDA is unavailable")
    if device.startswith("mps"):
        backend = getattr(base.torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("EEGNet device 'mps' requested but MPS is unavailable")


def train_condition(
    condition: str,
    dataset: Any,
    partitions: dict[str, list[str]],
    model_cfg: dict,
    out_dir: Path,
    random_state: int,
    base: Any,
) -> dict[str, Any]:
    summary = _package_train_condition(
        condition, dataset, partitions, model_cfg, out_dir, random_state
    )
    indices = base._condition_indices(dataset, partitions)
    for split, key in (("validation", "validation_predictions"), ("test", "test_predictions")):
        frame = summary[key]
        frame["sample"] = dataset.samples[indices[split]]
        metrics = classification_metrics(
            frame["true_label"].to_numpy(),
            frame["pred_proba"].to_numpy(),
            summary["threshold"],
            base,
        )
        summary[f"{split}_metrics"] = metrics
        filename = "val" if split == "validation" else "test"
        frame.to_csv(out_dir / f"{filename}_predictions.csv", index=False)
        _save_json(metrics, out_dir / f"{filename}_metrics.json")
    return summary


def _subject_metric_rows(
    frame: pd.DataFrame, condition: str, split: str, threshold: float, base: Any,
) -> list[dict]:
    rows = []
    for subject_id, group in frame.groupby("subject_id", sort=True):
        y = group["true_label"].to_numpy(dtype=np.int8)
        metrics = classification_metrics(
            y, group["pred_proba"].to_numpy(dtype=float), threshold, base
        )
        rows.append({
            "subject_id": str(subject_id),
            "condition": condition,
            "split": split,
            "n_trials": len(group),
            "n_correct": int(np.count_nonzero(y == 0)),
            "n_incorrect": int(np.count_nonzero(y == 1)),
            **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        })
    return rows


def component_metric_deltas(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    by_condition = condition_metrics.set_index("condition")
    rows = []
    for step_name in STEP_NAMES:
        row: dict[str, Any] = {"step": step_name}
        for metric in _METRIC_NAMES:
            raw = float(by_condition.loc["raw", f"test_{metric}"])
            specific = float(by_condition.loc[f"{step_name}_specific", f"test_{metric}"])
            coherent = float(by_condition.loc[f"{step_name}_coherent", f"test_{metric}"])
            row.update({
                f"raw_{metric}": raw,
                f"specific_{metric}": specific,
                f"coherent_{metric}": coherent,
                f"specific_minus_raw_{metric}": specific - raw,
                f"coherent_minus_raw_{metric}": coherent - raw,
                f"specific_minus_coherent_{metric}": specific - coherent,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _split_prevalence(dataset: Any, indices: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        split: float(np.mean(dataset.y[index])) for split, index in indices.items()
    }


def run(
    config_path: str | Path,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    workers: int = 1,
    random_state_override: int | None = None,
    force: bool = False,
    recompute_components: bool = False,
) -> Path:
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    config_path = Path(config_path)
    cfg = load_config(config_path)
    experiment = _experiment_cfg(cfg)
    random_state = (
        int(random_state_override)
        if random_state_override is not None
        else int(experiment.get("random_state", 42))
    )
    test_size = float(experiment.get("test_size", 0.2))
    validation_size = float(experiment.get("validation_size", 0.2))
    out = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _project_path(experiment["output_dir"], config_path)
    )
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {out}; pass --force")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = _project_path(_distributed_cfg(cfg)["cache_subdir"], config_path)
    cache_root.mkdir(parents=True, exist_ok=True)

    base = _load_script14()
    _validate_device(experiment, base)
    helpers = base._load_script10()
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
        recordings, cfg, cache_root, recompute_components, workers
    )
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "diagnostics"}
        for row in eligibility_rows
    ]).to_csv(out / "eligibility.csv", index=False)
    eligible_subjects = sorted(
        str(row["subject_id"]) for row in eligibility_rows if row["eligible"]
    )
    if not eligible_subjects:
        raise RuntimeError("No ERP-CORE subjects produced valid distributed components")

    manifest_rows = []
    by_subject = {str(row["subject_id"]): row for row in eligibility_rows}
    for subject_id in eligible_subjects:
        manifest_rows.extend(materialize_subject_artifacts(
            subject_id,
            recordings_by_subject[subject_id],
            cache_root,
            out,
            by_subject[subject_id]["diagnostics"],
        ))
    pd.DataFrame(manifest_rows).to_csv(out / "continuous_manifest.csv", index=False)

    dataset = load_sequence_dataset(
        eligible_subjects, recordings_by_subject, cfg, cache_root, base
    )
    partitions = base.split_subjects_two_stage(
        eligible_subjects, test_size, validation_size, random_state
    )
    indices = base._condition_indices(dataset, partitions)
    if any(len(index) == 0 for index in indices.values()):
        raise RuntimeError("Two-stage subject split produced an empty trial partition")
    split_manifest = base._manifest(dataset, partitions)
    split_manifest.to_csv(out / "split_manifest.csv", index=False)
    _save_json(
        {
            "random_state": random_state,
            "condition_training_seed": random_state,
            "test_size": test_size,
            "validation_size_within_train_pool": validation_size,
            "inner_split_seed": random_state + base._INNER_SPLIT_SEED_OFFSET,
            "train_pool_subjects": sorted(partitions["train"] + partitions["validation"]),
            "train_subjects": partitions["train"],
            "validation_subjects": partitions["validation"],
            "test_subjects": partitions["test"],
        },
        out / "split_manifest.json",
    )

    condition_rows = []
    prediction_frames = []
    subject_rows = []
    for index, condition in enumerate(CONDITIONS, start=1):
        print(f"[Distributed EEGNet {index}/{len(CONDITIONS)}] {condition}")
        summary = train_condition(
            condition,
            dataset,
            partitions,
            experiment,
            out / "conditions" / condition,
            random_state,
            base,
        )
        validation_metrics = summary["validation_metrics"]
        test_metrics = summary["test_metrics"]
        condition_rows.append({
            "condition": condition,
            "n_channels": summary["n_channels"],
            "n_times": summary["n_times"],
            "n_train_subjects": len(partitions["train"]),
            "n_validation_subjects": len(partitions["validation"]),
            "n_test_subjects": len(partitions["test"]),
            "n_train_trials": len(indices["train"]),
            "n_validation_trials": len(indices["validation"]),
            "n_test_trials": len(indices["test"]),
            "training_seed": random_state,
            "best_epoch": summary["best_epoch"],
            "best_validation_auprc": summary["best_validation_auprc"],
            "validation_threshold": summary["threshold"],
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
                if key != "confusion_matrix"
            },
            "validation_confusion_matrix": json.dumps(
                validation_metrics["confusion_matrix"]
            ),
            **{
                f"test_{key}": value
                for key, value in test_metrics.items()
                if key != "confusion_matrix"
            },
            "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        })
        for split, frame in (
            ("validation", summary["validation_predictions"]),
            ("test", summary["test_predictions"]),
        ):
            prediction_frames.append(frame.copy())
            subject_rows.extend(_subject_metric_rows(
                frame, condition, split, summary["threshold"], base
            ))

    condition_metrics = pd.DataFrame(condition_rows)
    condition_metrics.to_csv(out / "condition_metrics.csv", index=False)
    component_metric_deltas(condition_metrics).to_csv(
        out / "component_metric_deltas.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out / "predictions.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(out / "subject_metrics.csv", index=False)

    resolved = deepcopy(cfg)
    resolved_experiment = resolved["erp_core"]["distributed_component_eegnet"]
    resolved_experiment["random_state"] = random_state
    resolved_experiment["condition_training_seed"] = random_state
    resolved_experiment["grid_search"] = False
    resolved_experiment["early_stopping_metric"] = "validation_auprc"
    resolved_experiment["threshold_selection"] = "validation_balanced_accuracy"
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _save_json(
        {
            "conditions": list(CONDITIONS),
            "model_channels": list(ERP_CORE_EEG_CHANNELS),
            "model_input_channel_count": len(ERP_CORE_EEG_CHANNELS),
            "raw_definition": "common_preprocessed_continuous_eeg",
            "cascade": {
                "step1_input": "raw",
                "step2_input": "step1_specific",
                "step3_input": "step2_specific",
                "identity": (
                    "raw = step3_specific + step1_coherent + "
                    "step2_coherent + step3_coherent"
                ),
                "coherent_components_are_non_cumulative": True,
            },
            "subject_level_split": True,
            "same_training_seed_for_all_conditions": True,
            "test_size": test_size,
            "validation_size_within_train_pool": validation_size,
            "actual_split_subject_counts": {
                key: len(value) for key, value in partitions.items()
            },
            "split_positive_prevalence": _split_prevalence(dataset, indices),
            "no_grid_search": True,
            "early_stopping_metric": "validation_auprc",
            "threshold_selection": "validation_balanced_accuracy",
            "final_test_evaluation": "single_frozen_evaluation",
            "normalization": "trial_channel_zscore",
            "information_interpretation": (
                "Classification metrics are indirect proxies for task-relevant "
                "predictive information, not mutual information. Specific and "
                "coherent metrics are not additive."
            ),
            "continuous_edf": {
                "components_per_subject": list(COMPONENT_NAMES),
                "shared_cache_root": str(cache_root),
                "raw_exported": False,
                "annotations_preserved": True,
                "zero_channels": "exact_in_memory_edf_quantization_tolerance_on_readback",
            },
            "input_shapes": {
                condition: [
                    len(ERP_CORE_EEG_CHANNELS),
                    int(dataset.matrix(condition, normalize=False).shape[-1]),
                ]
                for condition in CONDITIONS
            },
            "comparison_summary": condition_metrics.to_dict(orient="records"),
        },
        out / "run_summary.json",
    )
    print(f"Distributed ERP-CORE ERN EEGNet comparison complete: {out}")
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
        recompute_components=args.recompute_components,
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
