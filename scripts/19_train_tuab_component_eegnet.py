#!/usr/bin/env python3
"""Train independent EEGNet models on Script 18 TUAB component epochs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import active_dataset_name
from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.erp_eegnet import (
    classification_metrics,
    select_balanced_accuracy_threshold,
)


SCHEMA_VERSION = 2
SCRIPT18_SCHEMA_VERSION = 2
COMBINED_CONDITION = "specific_coherent"
CONDITIONS = ("raw", "specific", "coherent", COMBINED_CONDITION)
SUPPORTED_MODES = ("frequency", "phasegated", "zerophase")
SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}
CONDITION_SEED_OFFSETS = {
    "raw": 1,
    "specific": 2,
    "coherent": 3,
    COMBINED_CONDITION: 4,
}
CONDITION_ARTIFACTS = (
    "best_model.pt",
    "best_params.json",
    "history.csv",
    "val_metrics.json",
    "test_metrics.json",
    "val_predictions.csv",
    "test_predictions.csv",
    "val_epoch_predictions.csv",
    "test_epoch_predictions.csv",
)


class CacheMismatch(ValueError):
    """Raised when an existing training result cannot be safely reused."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tuab.yaml")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=SUPPORTED_MODES)
    parser.add_argument(
        "--condition", choices=(*CONDITIONS, "all"), default="all"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(_jsonable(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def _combined_channel_names(ch_names: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    )


def _condition_n_channels(condition: str, base_channels: int) -> int:
    return 2 * base_channels if condition == COMBINED_CONDITION else base_channels


def _npz_array_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"Missing array {key!r} in {path}")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _inspect_record(path: Path, cfg: dict, mode: str) -> dict[str, Any]:
    required = {
        "raw", "specific", "coherent", COMBINED_CONDITION,
        "specific_coherent_ch_names", "epoch_start_samples", "epoch_start_sec",
        "label", "class_name", "split", "patient_id", "recording_id",
        "evaluation_id", "subject_id", "ch_names", "sfreq", "epoch_samples",
        "n_epochs", "source_mode", "fingerprint", "schema_version",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Missing Script 18 keys in {path}: {missing}")
        metadata = {
            "path": path,
            "label": int(_scalar(data, "label")),
            "class_name": str(_scalar(data, "class_name")),
            "source_split": str(_scalar(data, "split")),
            "patient_id": str(_scalar(data, "patient_id")),
            "recording_id": str(_scalar(data, "recording_id")),
            "evaluation_id": str(_scalar(data, "evaluation_id")),
            "subject_id": str(_scalar(data, "subject_id")),
            "ch_names": tuple(str(value) for value in data["ch_names"]),
            "specific_coherent_ch_names": tuple(
                str(value) for value in data["specific_coherent_ch_names"]
            ),
            "sfreq": float(_scalar(data, "sfreq")),
            "epoch_samples": int(_scalar(data, "epoch_samples")),
            "n_epochs": int(_scalar(data, "n_epochs")),
            "source_mode": str(_scalar(data, "source_mode")),
            "source_fingerprint": str(_scalar(data, "fingerprint")),
            "source_schema": int(_scalar(data, "schema_version")),
        }

    if metadata["source_schema"] != SCRIPT18_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Script 18 schema in {path}")
    if metadata["source_split"] not in SPLIT_MAP:
        raise ValueError(f"Invalid split in {path}: {metadata['source_split']!r}")
    metadata["split"] = SPLIT_MAP[metadata["source_split"]]
    expected_class = {0: "abnormal", 1: "normal"}.get(metadata["label"])
    if expected_class is None or metadata["class_name"] != expected_class:
        raise ValueError(
            f"Invalid TUAB label mapping in {path}: "
            f"{metadata['label']}={metadata['class_name']!r}"
        )
    if metadata["evaluation_id"] != path.stem:
        raise ValueError(f"evaluation_id does not match filename in {path}")
    if metadata["subject_id"] != metadata["evaluation_id"]:
        raise ValueError(f"subject_id must equal evaluation_id in {path}")
    if metadata["source_mode"] != mode:
        raise ValueError(
            f"Script 18 mode mismatch in {path}: expected {mode!r}, "
            f"found {metadata['source_mode']!r}"
        )

    expected_channels = tuple(str(value) for value in cfg["channels"]["standard_19"])
    expected_sfreq = float(cfg["preprocessing"]["target_sfreq"])
    expected_samples = int(
        expected_sfreq * float(cfg["preprocessing"]["epoch_length_sec"])
    )
    if metadata["ch_names"] != expected_channels:
        raise ValueError(f"Channel order mismatch in {path}")
    if metadata["specific_coherent_ch_names"] != _combined_channel_names(
        expected_channels
    ):
        raise ValueError(f"Combined channel order mismatch in {path}")
    if not np.isclose(metadata["sfreq"], expected_sfreq):
        raise ValueError(f"Sampling-rate mismatch in {path}")
    if metadata["epoch_samples"] != expected_samples:
        raise ValueError(f"Epoch length mismatch in {path}")
    if metadata["n_epochs"] < 1:
        raise ValueError(f"Script 18 cache has no epochs: {path}")

    for condition in CONDITIONS:
        expected_shape = (
            metadata["n_epochs"],
            _condition_n_channels(condition, len(expected_channels)),
            expected_samples,
        )
        shape, dtype = _npz_array_header(path, condition)
        if shape != expected_shape:
            raise ValueError(
                f"{condition} shape mismatch in {path}: "
                f"expected {expected_shape}, found {shape}"
            )
        if dtype != np.dtype(np.float32):
            raise ValueError(f"{condition} must be float32 in {path}")
    for key, dtype in (("epoch_start_samples", np.int64), ("epoch_start_sec", np.float64)):
        shape, actual_dtype = _npz_array_header(path, key)
        if shape != (metadata["n_epochs"],):
            raise ValueError(f"{key} shape mismatch in {path}")
        if actual_dtype != np.dtype(dtype):
            raise ValueError(f"{key} dtype mismatch in {path}")
    return metadata


def discover_records(input_root: Path, cfg: dict, mode: str) -> list[dict[str, Any]]:
    input_root = Path(input_root)
    manifest_path = input_root / "manifest.csv"
    resolved_path = input_root / "config_resolved.yaml"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError(
            f"Script 18 manifest/config not found in {input_root}"
        )
    manifest = pd.read_csv(manifest_path)
    required_columns = {"evaluation_id", "status"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(f"Invalid Script 18 manifest: {manifest_path}")
    if manifest["evaluation_id"].astype(str).duplicated().any():
        raise ValueError("Script 18 manifest contains duplicate evaluation_id values")
    statuses = set(manifest["status"].astype(str))
    allowed_statuses = {"processed", "cached", "skipped"}
    unexpected = sorted(statuses - allowed_statuses)
    if unexpected:
        raise ValueError(
            f"Script 18 manifest contains failed/unknown statuses: {unexpected}"
        )
    expected_ids = set(
        manifest.loc[
            manifest["status"].isin({"processed", "cached"}), "evaluation_id"
        ].astype(str)
    )
    actual_paths = sorted(input_root.glob("*.npz"))
    actual_ids = {path.stem for path in actual_paths}
    missing = sorted(expected_ids - actual_ids)
    orphaned = sorted(actual_ids - expected_ids)
    if missing or orphaned:
        raise ValueError(
            f"Script 18 NPZ/manifest mismatch; missing={missing[:5]}, "
            f"orphaned={orphaned[:5]}"
        )
    if not actual_paths:
        raise FileNotFoundError(f"No Script 18 epoch caches found in {input_root}")

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if active_dataset_name(resolved) != "tuab":
        raise ValueError("Script 18 resolved config is not dataset.active: tuab")
    records = [_inspect_record(path, cfg, mode) for path in actual_paths]
    frame = pd.DataFrame(records)
    if frame["evaluation_id"].duplicated().any():
        raise ValueError("Duplicate evaluation_id values in Script 18 caches")
    leakage = frame.groupby("patient_id")["split"].nunique()
    leaking_patients = sorted(leakage[leakage > 1].index.astype(str))
    if leaking_patients:
        raise ValueError(
            f"TUAB patient leakage across splits: {leaking_patients[:5]}"
        )
    for split in ("train", "validation", "test"):
        split_labels = set(frame.loc[frame["split"] == split, "label"].astype(int))
        if split_labels != {0, 1}:
            raise ValueError(
                f"TUAB {split} split must contain abnormal and normal records; "
                f"found labels {sorted(split_labels)}"
            )
    return sorted(records, key=lambda record: record["evaluation_id"])


def trial_channel_zscore(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each epoch/channel independently; constant channels remain zero."""
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected (epochs, channels, times), got {values.shape}")
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    normalized = (values - mean) / np.maximum(std, np.float32(eps))
    return np.nan_to_num(
        normalized, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)


def _load_record(record: dict[str, Any], condition: str) -> dict[str, Any]:
    with np.load(record["path"], allow_pickle=False) as data:
        epochs = np.asarray(data[condition], dtype=np.float32)
        starts = np.asarray(data["epoch_start_samples"], dtype=np.int64)
        starts_sec = np.asarray(data["epoch_start_sec"], dtype=np.float64)
    expected_shape = (
        record["n_epochs"],
        _condition_n_channels(condition, len(record["ch_names"])),
        record["epoch_samples"],
    )
    if epochs.shape != expected_shape:
        raise ValueError(f"Runtime shape mismatch in {record['path']}")
    if not np.isfinite(epochs).all():
        raise ValueError(f"Non-finite {condition} epochs in {record['path']}")
    return {
        "epochs": trial_channel_zscore(epochs),
        "epoch_start_samples": starts,
        "epoch_start_sec": starts_sec,
        **{
            key: record[key]
            for key in (
                "evaluation_id", "patient_id", "recording_id", "label",
                "class_name", "split", "n_epochs",
            )
        },
    }


class RecordingDataset(Dataset):
    """Load and normalize one compressed Script 18 recording per item."""

    def __init__(self, records: list[dict[str, Any]], condition: str) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        self.records = list(records)
        self.condition = condition

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _load_record(self.records[index], self.condition)


def resolve_device(requested: str) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        backend = getattr(torch.backends, "mps", None)
        if backend is not None and backend.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("EEGNet device 'cuda' requested but CUDA is unavailable")
    if requested == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("EEGNet device 'mps' requested but MPS is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    return requested


def record_class_weights(records: Iterable[dict[str, Any]]) -> dict[int, float]:
    labels = np.asarray([int(record["label"]) for record in records], dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training records must contain both TUAB classes")
    return {0: 1.0, 1: float(counts[0] / counts[1])}


def _make_loader(
    records: list[dict[str, Any]], condition: str, workers: int,
    *, shuffle: bool, seed: int,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": RecordingDataset(records, condition),
        "batch_size": None,
        "shuffle": shuffle,
        "num_workers": workers,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def _record_groups(loader: DataLoader, size: int):
    group: list[dict[str, Any]] = []
    for item in loader:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def _as_int(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _make_model(n_channels: int, n_times: int, model_cfg: dict) -> EEGNet:
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        F1=int(model_cfg["F1"]),
        D=int(model_cfg["D"]),
        dropout=float(model_cfg["dropout"]),
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    class_weights: dict[int, float],
    epoch_batch_size: int,
    records_per_step: int,
) -> float:
    model.train()
    weighted_record_losses: list[float] = []
    for group in _record_groups(loader, records_per_step):
        optimizer.zero_grad(set_to_none=True)
        for item in group:
            epochs = item["epochs"]
            if not isinstance(epochs, torch.Tensor):
                epochs = torch.as_tensor(epochs, dtype=torch.float32)
            label = _as_int(item["label"])
            n_epochs = int(epochs.shape[0])
            class_weight = float(class_weights[label])
            record_loss = 0.0
            for start in range(0, n_epochs, epoch_batch_size):
                tensors = epochs[start:start + epoch_batch_size].to(device)
                labels = torch.full(
                    (len(tensors),), float(label), dtype=torch.float32,
                    device=device,
                )
                probabilities = model(tensors.unsqueeze(1)).squeeze(1)
                losses = criterion(probabilities, labels)
                partial = (
                    losses.sum() / n_epochs * class_weight / len(group)
                )
                partial.backward()
                record_loss += float(losses.detach().sum().cpu()) / n_epochs
            weighted_record_losses.append(record_loss * class_weight)
        optimizer.step()
    return float(np.mean(weighted_record_losses))


def _predict_records(
    model: nn.Module,
    records: list[dict[str, Any]],
    condition: str,
    device: torch.device,
    epoch_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = _make_loader(records, condition, workers, shuffle=False, seed=seed)
    model.eval()
    record_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in loader:
            epochs = item["epochs"]
            if not isinstance(epochs, torch.Tensor):
                epochs = torch.as_tensor(epochs, dtype=torch.float32)
            probabilities: list[np.ndarray] = []
            for start in range(0, len(epochs), epoch_batch_size):
                batch = epochs[start:start + epoch_batch_size].to(device)
                values = model(batch.unsqueeze(1)).squeeze(1).cpu().numpy()
                probabilities.append(np.asarray(values, dtype=float))
            epoch_probabilities = np.concatenate(probabilities)
            evaluation_id = str(item["evaluation_id"])
            patient_id = str(item["patient_id"])
            recording_id = str(item["recording_id"])
            label = _as_int(item["label"])
            split = str(item["split"])
            starts = np.asarray(item["epoch_start_samples"], dtype=np.int64)
            starts_sec = np.asarray(item["epoch_start_sec"], dtype=np.float64)
            for epoch_index, (start_sample, start_sec, probability) in enumerate(
                zip(starts, starts_sec, epoch_probabilities)
            ):
                epoch_rows.append({
                    "condition": condition,
                    "split": split,
                    "evaluation_id": evaluation_id,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "epoch_index": epoch_index,
                    "epoch_start_sample": int(start_sample),
                    "epoch_start_sec": float(start_sec),
                    "true_label": label,
                    "pred_proba": float(probability),
                })
            record_rows.append({
                "condition": condition,
                "split": split,
                "evaluation_id": evaluation_id,
                "subject_id": evaluation_id,
                "patient_id": patient_id,
                "recording_id": recording_id,
                "n_epochs": len(epoch_probabilities),
                "true_label": label,
                "pred_proba": float(epoch_probabilities.mean()),
            })
    return pd.DataFrame(record_rows), pd.DataFrame(epoch_rows)


def _condition_fingerprint(
    records: list[dict[str, Any]], condition: str, model_cfg: dict,
    random_state: int, resolved_device: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "condition": condition,
        "n_channels": _condition_n_channels(condition, len(records[0]["ch_names"])),
        "inputs": [
            {
                "evaluation_id": record["evaluation_id"],
                "source_fingerprint": record["source_fingerprint"],
                "n_epochs": record["n_epochs"],
                "split": record["split"],
                "label": record["label"],
            }
            for record in records
        ],
        "model": _jsonable(model_cfg),
        "random_state": random_state,
        "resolved_device": resolved_device,
        "normalization": "epoch_channel_zscore",
        "loss": "record_mean_weighted_bce",
        "checkpoint_selection": "validation_recording_auprc",
        "threshold_selection": "validation_recording_balanced_accuracy",
        "recording_aggregation": "mean_epoch_probability",
        "positive_label": "1=normal",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_cache_marker(condition_dir: Path, fingerprint: str) -> None:
    artifacts = {
        name: {
            "size": (condition_dir / name).stat().st_size,
            "sha256": _sha256_file(condition_dir / name),
        }
        for name in CONDITION_ARTIFACTS
    }
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "artifacts": artifacts,
        },
        condition_dir / "cache.json",
    )


def _condition_cache_matches(condition_dir: Path, fingerprint: str) -> bool:
    if not condition_dir.exists() or not any(condition_dir.iterdir()):
        return False
    marker_path = condition_dir / "cache.json"
    if not marker_path.is_file():
        raise CacheMismatch(
            f"Incomplete EEGNet result cache: {condition_dir}; re-run with --force"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheMismatch(f"Unreadable EEGNet cache marker: {marker_path}") from exc
    if int(marker.get("schema_version", -1)) != SCHEMA_VERSION:
        raise CacheMismatch(
            f"EEGNet cache schema mismatch: {condition_dir}; re-run with --force"
        )
    if str(marker.get("fingerprint", "")) != fingerprint:
        raise CacheMismatch(
            f"EEGNet cache configuration/input mismatch: {condition_dir}; "
            "re-run with --force"
        )
    artifacts = marker.get("artifacts", {})
    for name in CONDITION_ARTIFACTS:
        path = condition_dir / name
        entry = artifacts.get(name, {})
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("size", -1))
            or _sha256_file(path) != str(entry.get("sha256", ""))
        ):
            raise CacheMismatch(
                f"Incomplete/corrupt EEGNet result cache: {path}; "
                "re-run with --force"
            )
    return True


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_condition(
    condition: str,
    records: list[dict[str, Any]],
    model_cfg: dict,
    condition_dir: Path,
    random_state: int,
    resolved_device: str,
    workers: int,
    fingerprint: str,
) -> dict[str, Any]:
    """Train one condition with record-equal loss and record-level selection."""
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    class_weights = record_class_weights(split_records["train"])
    _set_random_seed(random_state)
    device = torch.device(resolved_device)
    n_channels = _condition_n_channels(condition, len(records[0]["ch_names"]))
    n_times = int(records[0]["epoch_samples"])
    model = _make_model(n_channels, n_times, model_cfg).to(device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = Adam(
        model.parameters(),
        lr=float(model_cfg["lr"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(model_cfg["lr_factor"]),
        patience=int(model_cfg["lr_patience"]),
    )
    epoch_batch_size = int(model_cfg["batch_size"])
    records_per_step = int(model_cfg.get("records_per_step", 4))
    max_epochs = int(model_cfg["max_epochs"])
    patience = int(model_cfg["patience"])
    if epoch_batch_size < 1 or records_per_step < 1:
        raise ValueError("batch_size and records_per_step must be >= 1")

    train_loader = _make_loader(
        split_records["train"], condition, workers,
        shuffle=True, seed=random_state,
    )
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, class_weights,
            epoch_batch_size, records_per_step,
        )
        validation_records, _ = _predict_records(
            model, split_records["validation"], condition, device,
            epoch_batch_size, workers, random_state,
        )
        validation_score = float(average_precision_score(
            validation_records["true_label"],
            validation_records["pred_proba"],
        ))
        scheduler.step(validation_score)
        history.append({
            "epoch": epoch,
            "train_record_weighted_loss": train_loss,
            "validation_recording_auprc": validation_score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if np.isfinite(validation_score) and validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("No finite validation recording AUPRC was produced")
    model.load_state_dict(best_state)

    validation_predictions, validation_epoch_predictions = _predict_records(
        model, split_records["validation"], condition, device,
        epoch_batch_size, workers, random_state,
    )
    threshold = select_balanced_accuracy_threshold(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(),
    )
    test_predictions, test_epoch_predictions = _predict_records(
        model, split_records["test"], condition, device,
        epoch_batch_size, workers, random_state,
    )
    validation_metrics = classification_metrics(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(), threshold,
    )
    test_metrics = classification_metrics(
        test_predictions["true_label"].to_numpy(),
        test_predictions["pred_proba"].to_numpy(), threshold,
    )
    for frame in (
        validation_predictions, test_predictions,
        validation_epoch_predictions, test_epoch_predictions,
    ):
        frame["predicted_label"] = (
            frame["pred_proba"] >= threshold
        ).astype(np.int8)

    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "condition": condition,
        "fingerprint": fingerprint,
        "n_channels": n_channels,
        "n_times": n_times,
        "positive_label": "1=normal",
    }
    _atomic_torch_save(checkpoint, condition_dir / "best_model.pt")
    _atomic_csv(pd.DataFrame(history), condition_dir / "history.csv")
    _atomic_json(validation_metrics, condition_dir / "val_metrics.json")
    _atomic_json(test_metrics, condition_dir / "test_metrics.json")
    _atomic_csv(validation_predictions, condition_dir / "val_predictions.csv")
    _atomic_csv(test_predictions, condition_dir / "test_predictions.csv")
    _atomic_csv(
        validation_epoch_predictions,
        condition_dir / "val_epoch_predictions.csv",
    )
    _atomic_csv(test_epoch_predictions, condition_dir / "test_epoch_predictions.csv")
    best_params = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "condition": condition,
        "random_state": random_state,
        "resolved_device": resolved_device,
        "n_channels": n_channels,
        "n_times": n_times,
        "F1": int(model_cfg["F1"]),
        "D": int(model_cfg["D"]),
        "dropout": float(model_cfg["dropout"]),
        "lr": float(model_cfg["lr"]),
        "weight_decay": float(model_cfg["weight_decay"]),
        "batch_size": epoch_batch_size,
        "records_per_step": records_per_step,
        "class_weights": class_weights,
        "best_epoch": best_epoch,
        "best_validation_recording_auprc": best_score,
        "threshold": threshold,
        "checkpoint_selection": "validation_recording_auprc",
        "threshold_selection": "validation_recording_balanced_accuracy",
        "normalization": "epoch_channel_zscore",
        "loss": "record_mean_weighted_bce",
        "recording_aggregation": "mean_epoch_probability",
        "positive_label": "1=normal",
    }
    _atomic_json(best_params, condition_dir / "best_params.json")
    _write_cache_marker(condition_dir, fingerprint)
    return _load_condition_result(condition_dir)


def _load_condition_result(condition_dir: Path) -> dict[str, Any]:
    best_params = json.loads(
        (condition_dir / "best_params.json").read_text(encoding="utf-8")
    )
    validation_metrics = json.loads(
        (condition_dir / "val_metrics.json").read_text(encoding="utf-8")
    )
    test_metrics = json.loads(
        (condition_dir / "test_metrics.json").read_text(encoding="utf-8")
    )
    return {
        "condition": best_params["condition"],
        "best_params": best_params,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": pd.read_csv(
            condition_dir / "val_predictions.csv"
        ),
        "test_predictions": pd.read_csv(condition_dir / "test_predictions.csv"),
        "validation_epoch_predictions": pd.read_csv(
            condition_dir / "val_epoch_predictions.csv"
        ),
        "test_epoch_predictions": pd.read_csv(
            condition_dir / "test_epoch_predictions.csv"
        ),
    }


def _split_manifest(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "evaluation_id": record["evaluation_id"],
            "patient_id": record["patient_id"],
            "recording_id": record["recording_id"],
            "split": record["split"],
            "source_split": record["source_split"],
            "label": record["label"],
            "class_name": record["class_name"],
            "n_epochs": record["n_epochs"],
            "source_fingerprint": record["source_fingerprint"],
            "path": str(record["path"]),
        }
        for record in records
    ]
    return pd.DataFrame(rows)


def _condition_summary(result: dict[str, Any], records: list[dict[str, Any]]) -> dict:
    params = result["best_params"]
    val_metrics = result["validation_metrics"]
    test_metrics = result["test_metrics"]
    counts = pd.DataFrame(records).groupby("split").size().to_dict()
    return {
        "condition": result["condition"],
        "n_train_records": int(counts.get("train", 0)),
        "n_validation_records": int(counts.get("validation", 0)),
        "n_test_records": int(counts.get("test", 0)),
        "best_epoch": int(params["best_epoch"]),
        "best_validation_recording_auprc": float(
            params["best_validation_recording_auprc"]
        ),
        "validation_threshold": float(params["threshold"]),
        **{
            f"validation_{key}": value
            for key, value in val_metrics.items()
            if key != "confusion_matrix"
        },
        "validation_confusion_matrix": json.dumps(
            val_metrics["confusion_matrix"]
        ),
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
            if key != "confusion_matrix"
        },
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
    }


def run(
    config_path: str | Path = "configs/tuab.yaml",
    *, input_dir: Path | None = None, output_dir: Path | None = None,
    mode: str | None = None, condition: str = "all", device: str | None = None,
    workers: int | None = None, random_state: int | None = None,
    force: bool = False,
) -> Path:
    config_path = Path(config_path)
    cfg = deepcopy(load_config(config_path))
    if active_dataset_name(cfg) != "tuab":
        raise ValueError("Script 19 requires dataset.active: tuab")
    try:
        model_cfg = deepcopy(cfg["ml"]["tuab_component_eegnet"])
    except KeyError as exc:
        raise KeyError("Missing ml.tuab_component_eegnet config") from exc
    effective_mode = str(mode or cfg["wiener"].get("mode", "frequency"))
    if effective_mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported Wiener mode: {effective_mode!r}")
    if condition not in (*CONDITIONS, "all"):
        raise ValueError(f"Unknown condition: {condition!r}")
    effective_workers = int(
        workers if workers is not None else model_cfg.get("num_workers", 0)
    )
    if effective_workers < 0:
        raise ValueError("--workers must be >= 0")
    base_random_state = int(
        random_state if random_state is not None else model_cfg.get("random_state", 42)
    )
    requested_device = str(device or model_cfg.get("device", "auto"))
    resolved_device = resolve_device(requested_device)
    model_cfg["device"] = requested_device
    model_cfg["num_workers"] = effective_workers
    model_cfg["random_state"] = base_random_state

    input_root = (
        input_dir.expanduser().resolve() if input_dir is not None
        else Path(cfg["paths"]["cache_dir"])
        / f"tuab_continuous_wiener_{effective_mode}" / "epochs"
    )
    output_root = (
        output_dir.expanduser().resolve() if output_dir is not None
        else Path(cfg["paths"]["results_dir"])
        / f"tuab_component_eegnet_{effective_mode}"
    )
    records = discover_records(input_root, cfg, effective_mode)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_csv(_split_manifest(records), output_root / "split_manifest.csv")

    fingerprints = {
        item: _condition_fingerprint(
            records, item, model_cfg,
            base_random_state + CONDITION_SEED_OFFSETS[item], resolved_device,
        )
        for item in CONDITIONS
    }
    selected = CONDITIONS if condition == "all" else (condition,)
    for index, item in enumerate(selected, start=1):
        condition_dir = output_root / "conditions" / item
        fingerprint = fingerprints[item]
        if not force and _condition_cache_matches(condition_dir, fingerprint):
            print(f"[EEGNet {index}/{len(selected)}] {item}: cached")
            continue
        if force:
            (condition_dir / "cache.json").unlink(missing_ok=True)
        print(f"[EEGNet {index}/{len(selected)}] {item}: training")
        train_condition(
            item, records, model_cfg, condition_dir,
            base_random_state + CONDITION_SEED_OFFSETS[item],
            resolved_device, effective_workers, fingerprint,
        )

    completed: list[dict[str, Any]] = []
    for item in CONDITIONS:
        condition_dir = output_root / "conditions" / item
        try:
            if _condition_cache_matches(condition_dir, fingerprints[item]):
                completed.append(_load_condition_result(condition_dir))
        except CacheMismatch:
            if item in selected:
                raise

    condition_metrics = pd.DataFrame(
        [_condition_summary(result, records) for result in completed]
    )
    _atomic_csv(condition_metrics, output_root / "condition_metrics.csv")
    prediction_frames: list[pd.DataFrame] = []
    epoch_prediction_frames: list[pd.DataFrame] = []
    for result in completed:
        prediction_frames.extend([
            result["validation_predictions"], result["test_predictions"]
        ])
        epoch_prediction_frames.extend([
            result["validation_epoch_predictions"],
            result["test_epoch_predictions"],
        ])
    _atomic_csv(
        pd.concat(prediction_frames, ignore_index=True),
        output_root / "predictions.csv",
    )
    _atomic_csv(
        pd.concat(epoch_prediction_frames, ignore_index=True),
        output_root / "epoch_predictions.csv",
    )

    resolved_cfg = deepcopy(cfg)
    resolved_cfg["ml"]["tuab_component_eegnet"] = deepcopy(model_cfg)
    resolved_cfg["ml"]["tuab_component_eegnet"].update({
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "mode": effective_mode,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "selected_conditions": list(selected),
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
        "checkpoint_selection": "validation_recording_auprc",
        "threshold_selection": "validation_recording_balanced_accuracy",
        "normalization": "epoch_channel_zscore",
        "loss": "record_mean_weighted_bce",
        "recording_aggregation": "mean_epoch_probability",
        "positive_label": "1=normal",
    })
    _atomic_yaml(resolved_cfg, output_root / "config_resolved.yaml")
    split_frame = pd.DataFrame(records)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "mode": effective_mode,
            "requested_conditions": list(selected),
            "completed_conditions": [result["condition"] for result in completed],
            "requested_device": requested_device,
            "resolved_device": resolved_device,
            "random_state": base_random_state,
            "record_counts": {
                split: int((split_frame["split"] == split).sum())
                for split in ("train", "validation", "test")
            },
            "patient_counts": {
                split: int(split_frame.loc[
                    split_frame["split"] == split, "patient_id"
                ].nunique())
                for split in ("train", "validation", "test")
            },
            "patient_disjoint": True,
            "positive_label": "1=normal",
            "checkpoint_selection": "validation_recording_auprc",
            "threshold_selection": "validation_recording_balanced_accuracy",
            "final_test_evaluation": "single_frozen_evaluation_per_training_run",
            "normalization": "epoch_channel_zscore",
            "specific_coherent_layout": "channel_axis_specific_then_coherent",
            "loss": "record_mean_weighted_bce",
            "recording_aggregation": "mean_epoch_probability",
            "condition_fingerprints": {
                item: fingerprints[item] for item in CONDITIONS
            },
            "comparison_summary": condition_metrics.to_dict(orient="records"),
        },
        output_root / "run_summary.json",
    )
    print(f"TUAB component EEGNet training complete: {output_root}")
    return output_root


def main() -> None:
    args = _parse_args()
    run(
        args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        condition=args.condition,
        device=args.device,
        workers=args.workers,
        random_state=args.random_state,
        force=args.force,
    )


if __name__ == "__main__":
    main()
