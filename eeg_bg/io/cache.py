import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Callable


def make_cache_key(edf_path: Path, start_sec: float, cfg: dict) -> str:
    preprocessing = cfg.get("preprocessing", {})
    dataset_cfg = cfg.get("dataset", {})
    active = dataset_cfg.get("active", "tuep")
    active_block = dataset_cfg.get(active, dataset_cfg)
    fingerprint = {
        "edf_path": str(edf_path),
        "start_sec": round(float(start_sec), 4),
        "dataset": active,
        "target_sfreq": preprocessing.get("target_sfreq"),
        "bandpass": preprocessing.get("bandpass"),
        "epoch_length_sec": preprocessing.get("epoch_length_sec"),
        "artifact_threshold_uv": preprocessing.get("artifact_threshold_uv"),
        "standard_19": cfg.get("channels", {}).get("standard_19"),
        "seizure_buffer_sec": preprocessing.get("seizure_buffer_sec"),
        "max_recording_sec": active_block.get("max_recording_sec"),
    }
    raw = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


CACHE_METADATA_KEYS = (
    "dataset_name", "patient_id", "recording_id", "evaluation_id",
    "subject_id", "class_name", "label", "split", "source_partition",
    "n_epochs",
)


def build_cache_metadata(row: dict, n_epochs: int) -> dict[str, np.ndarray]:
    """Build the common identity payload stored in every pipeline cache."""
    evaluation_id = str(row["evaluation_id"])
    values = {
        "dataset_name": row["dataset_name"],
        "patient_id": row["patient_id"],
        "recording_id": row["recording_id"],
        "evaluation_id": evaluation_id,
        "subject_id": evaluation_id,  # compatibility alias
        "class_name": row["class_name"],
        "label": int(row["label"]),
        "split": row["split"],
        "source_partition": row["source_partition"],
        "n_epochs": int(n_epochs),
    }
    return {key: np.asarray(value) for key, value in values.items()}


def copy_cache_metadata(data) -> dict[str, np.ndarray]:
    """Copy common metadata, with fallbacks for legacy TUEP caches."""
    subject_id = str(data.get("subject_id", ""))
    evaluation_id = str(data.get("evaluation_id", subject_id))
    values = {
        "dataset_name": data.get("dataset_name", "tuep"),
        "patient_id": data.get("patient_id", subject_id),
        "recording_id": data.get("recording_id", ""),
        "evaluation_id": evaluation_id,
        "subject_id": evaluation_id,
        "class_name": data.get("class_name", ""),
        "label": data["label"],
        "split": data["split"],
        "source_partition": data.get("source_partition", ""),
        "n_epochs": data.get("n_epochs", 0),
    }
    return {key: np.asarray(value) for key, value in values.items()}


def load_or_compute(
    cache_path: Path,
    compute_fn: Callable[[], dict],
    force_recompute: bool = False,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return dict(np.load(cache_path, allow_pickle=True))
    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result
