"""Dataset discovery and split assignment for TUEP and TUAB."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


INDEX_COLUMNS = [
    "dataset_name", "patient_id", "session_id", "token_id",
    "recording_id", "evaluation_id", "class_name", "label",
    "reference", "source_partition", "edf_path",
]


def active_dataset_name(cfg: dict) -> str:
    """Return the selected dataset name, accepting legacy TUEP configs."""
    dataset_cfg = cfg.get("dataset", {})
    name = str(dataset_cfg.get("active", "tuep")).lower()
    if name not in {"tuep", "tuab"}:
        raise ValueError("dataset.active must be either 'tuep' or 'tuab'")
    return name


def active_dataset_config(cfg: dict) -> dict:
    """Return the active adapter block and validate its basic structure."""
    dataset_cfg = cfg.get("dataset", {})
    name = active_dataset_name(cfg)
    block = dataset_cfg.get(name)
    if block is None and name == "tuep":
        # Backward compatibility for local configs created before dataset.active.
        block = dataset_cfg
    if not isinstance(block, dict):
        raise ValueError(f"dataset.{name} must be a mapping")

    required = {"reference_scheme", "montage_dir", "classes"}
    if name == "tuab":
        required |= {
            "edf_dir", "train_partition", "eval_partition",
            "validation_fraction", "max_recording_sec",
        }
    missing = sorted(required - block.keys())
    if missing:
        raise ValueError(f"dataset.{name} is missing required keys: {missing}")

    _normalise_classes(block["classes"])
    reference = str(block["reference_scheme"]).lower()
    montage = str(block["montage_dir"]).lower()
    if reference not in {"ar", "le"}:
        raise ValueError(f"dataset.{name}.reference_scheme must be 'ar' or 'le'")
    if (reference == "ar") != ("tcp_ar" in montage):
        raise ValueError(
            f"dataset.{name}.reference_scheme and montage_dir do not agree"
        )
    if (reference == "le") != ("tcp_le" in montage):
        raise ValueError(
            f"dataset.{name}.reference_scheme and montage_dir do not agree"
        )
    if name == "tuab":
        val_fraction = float(block["validation_fraction"])
        if not 0.0 < val_fraction <= 0.5:
            raise ValueError("dataset.tuab.validation_fraction must be in (0, 0.5]")
        epoch_len = float(cfg["preprocessing"]["epoch_length_sec"])
        if float(block["max_recording_sec"]) < epoch_len:
            raise ValueError(
                "dataset.tuab.max_recording_sec must be at least one epoch"
            )
    return block


def _normalise_classes(classes: dict) -> dict[str, dict]:
    """Normalise old ``name: folder`` and new explicit class mappings."""
    if not isinstance(classes, dict) or len(classes) != 2:
        raise ValueError("dataset classes must define exactly two classes")
    result: dict[str, dict] = {}
    for class_name, value in classes.items():
        if isinstance(value, str):
            label = 0 if class_name in {"epilepsy", "abnormal"} else 1
            result[class_name] = {"folder": value, "label": label}
        elif isinstance(value, dict) and {"folder", "label"} <= value.keys():
            result[class_name] = {
                "folder": str(value["folder"]), "label": int(value["label"]),
            }
        else:
            raise ValueError(
                f"class {class_name!r} must define folder and label"
            )
    if {entry["label"] for entry in result.values()} != {0, 1}:
        raise ValueError("dataset class labels must be exactly {0, 1}")
    return result


def _parse_recording_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    patient_id = parts[0]
    session_id = parts[1] if len(parts) > 1 else ""
    token_id = parts[-1] if len(parts) > 2 else ""
    return patient_id, session_id, token_id


def _build_tuep_index(data_root: Path, block: dict) -> list[dict]:
    records: list[dict] = []
    for class_name, class_cfg in _normalise_classes(block["classes"]).items():
        class_dir = data_root / class_cfg["folder"]
        if not class_dir.exists():
            continue
        for subject_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                montage_path = session_dir / block["montage_dir"]
                for edf_file in sorted(montage_path.glob("*.edf")):
                    recording_id = edf_file.stem
                    _, parsed_session, token_id = _parse_recording_stem(recording_id)
                    label = int(class_cfg["label"])
                    evaluation_id = f"{label:02d}_{subject_dir.name}"
                    records.append({
                        "dataset_name": "tuep",
                        "patient_id": subject_dir.name,
                        "session_id": session_dir.name or parsed_session,
                        "token_id": token_id,
                        "recording_id": recording_id,
                        "evaluation_id": evaluation_id,
                        "class_name": class_name,
                        "label": label,
                        "reference": block["reference_scheme"],
                        "source_partition": "",
                        "edf_path": str(edf_file),
                    })
    return records


def _build_tuab_index(data_root: Path, block: dict) -> list[dict]:
    records: list[dict] = []
    classes = _normalise_classes(block["classes"])
    for partition in (block["train_partition"], block["eval_partition"]):
        for class_name, class_cfg in classes.items():
            montage_path = (
                data_root / block["edf_dir"] / partition
                / class_cfg["folder"] / block["montage_dir"]
            )
            for edf_file in sorted(montage_path.glob("*.edf")):
                recording_id = edf_file.stem
                patient_id, session_id, token_id = _parse_recording_stem(
                    recording_id
                )
                records.append({
                    "dataset_name": "tuab",
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "token_id": token_id,
                    "recording_id": recording_id,
                    "evaluation_id": recording_id,
                    "class_name": class_name,
                    "label": int(class_cfg["label"]),
                    "reference": block["reference_scheme"],
                    "source_partition": partition,
                    "edf_path": str(edf_file),
                })
    return records


def build_recording_index(cfg: dict) -> pd.DataFrame:
    """Discover EDF recordings for the active dataset."""
    data_root = Path(cfg["paths"]["data_root"])
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")
    name = active_dataset_name(cfg)
    block = active_dataset_config(cfg)
    records = (
        _build_tuep_index(data_root, block)
        if name == "tuep"
        else _build_tuab_index(data_root, block)
    )
    return pd.DataFrame(records, columns=INDEX_COLUMNS)


def _assign_tuep_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["split"]["random_seed"])
    train_frac = float(cfg["split"]["train"])
    val_frac = float(cfg["split"]["val"])
    train_set: set[str] = set()
    val_set: set[str] = set()

    units = index.drop_duplicates("evaluation_id")
    for _, group in units.groupby("label"):
        evaluation_ids = group["evaluation_id"].to_numpy(copy=True)
        rng.shuffle(evaluation_ids)
        n_train = int(len(evaluation_ids) * train_frac)
        n_val = max(1, int(len(evaluation_ids) * val_frac))
        train_set.update(evaluation_ids[:n_train])
        val_set.update(evaluation_ids[n_train:n_train + n_val])

    out = index.copy()
    out["split"] = np.where(
        out["evaluation_id"].isin(train_set), "train",
        np.where(out["evaluation_id"].isin(val_set), "val", "test"),
    )
    return out


def _assign_tuab_splits(index: pd.DataFrame, cfg: dict, block: dict) -> pd.DataFrame:
    train_partition = str(block["train_partition"])
    eval_partition = str(block["eval_partition"])
    out = index.copy()
    out["split"] = ""
    out.loc[out["source_partition"] == eval_partition, "split"] = "test"

    official_train = out[out["source_partition"] == train_partition]
    if official_train.empty:
        return out
    groups = official_train["patient_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = max(2, round(1.0 / float(block["validation_fraction"])))
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"TUAB validation split needs at least {n_splits} training patients; "
            f"found {len(unique_groups)}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(cfg["split"]["random_seed"]),
    )
    y = official_train["label"].to_numpy()
    target_total = len(official_train) / n_splits
    target_by_class = {
        label: int(np.sum(y == label)) / n_splits for label in (0, 1)
    }
    candidates: list[tuple[float, np.ndarray]] = []
    for _, val_pos in splitter.split(official_train, y, groups):
        val_labels = y[val_pos]
        score = abs(len(val_pos) - target_total)
        score += sum(
            abs(int(np.sum(val_labels == label)) - target_by_class[label])
            for label in (0, 1)
        )
        candidates.append((score, val_pos))
    _, best_val_pos = min(candidates, key=lambda item: item[0])
    val_patients = set(official_train.iloc[best_val_pos]["patient_id"])
    is_official_train = out["source_partition"] == train_partition
    out.loc[is_official_train, "split"] = "train"
    out.loc[
        is_official_train & out["patient_id"].isin(val_patients), "split"
    ] = "val"
    return out


def assign_dataset_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign dataset-aware train/validation/test splits."""
    if index.empty:
        out = index.copy()
        out["split"] = pd.Series(dtype=str)
        return out
    name = active_dataset_name(cfg)
    out = (
        _assign_tuep_splits(index, cfg)
        if name == "tuep"
        else _assign_tuab_splits(index, cfg, active_dataset_config(cfg))
    )
    if (out["split"] == "").any():
        unknown = sorted(out.loc[out["split"] == "", "source_partition"].unique())
        raise ValueError(f"Unrecognised source partitions: {unknown}")
    if out.groupby("patient_id")["split"].nunique().max() > 1 and name == "tuab":
        raise ValueError("TUAB patient leakage detected across data splits")
    if out.groupby("evaluation_id")["label"].nunique().max() > 1:
        raise ValueError("An evaluation_id has multiple labels")
    return out


def get_recording_intervals(
    row: pd.Series | dict, cfg: dict, recording_duration: float,
) -> list[tuple[float, float]]:
    """Return dataset-specific intervals eligible for epoch extraction."""
    if active_dataset_name(cfg) == "tuab":
        limit = float(active_dataset_config(cfg)["max_recording_sec"])
        return [(0.0, min(float(recording_duration), limit))]

    from eeg_bg.io.annotation import extract_bckg_intervals

    edf_path = Path(row["edf_path"])
    csv_bi_path = edf_path.with_suffix(".csv_bi")
    if csv_bi_path.exists():
        return extract_bckg_intervals(
            csv_bi_path, cfg, recording_duration=recording_duration
        )
    return [(0.0, float(recording_duration))]


def build_subject_index(cfg: dict) -> pd.DataFrame:
    """Backward-compatible alias returning the canonical recording index."""
    index = build_recording_index(cfg)
    index["subject_id"] = index["patient_id"]
    return index


def assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Backward-compatible alias for :func:`assign_dataset_splits`."""
    if "evaluation_id" not in index:
        index = index.copy()
        index["evaluation_id"] = index["subject_id"]
        index["patient_id"] = index["subject_id"]
        index["source_partition"] = ""
    return assign_dataset_splits(index, cfg)
