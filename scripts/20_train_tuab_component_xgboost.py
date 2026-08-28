"""Train independent TUAB XGBoost models from Script 18 epoch caches.

Script 18 stores paired ``raw``, ``specific``, ``coherent`` and channel-axis
``specific_coherent`` arrays in one NPZ per recording.  This entry point keeps
those four conditions independent while reusing the feature profiles and
XGBoost training/evaluation semantics used by Script 06.

``--workers`` controls file-level feature-extraction worker processes.  The
four conditions remain sequential, and cached features are loaded without
starting workers, matching Script 06's execution model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.features.extraction import FeatureDataset
from eeg_bg.features.profiles import PROFILES
from eeg_bg.io.dataset import active_dataset_name
from eeg_bg.ml.xgb_pipeline import (
    train_xgboost,
    evaluation_level_predict,
    evaluate_subject_level,
    find_optimal_threshold,
)
from eeg_bg.ml.shap_analysis import (
    compute_shap_values,
    aggregate_shap_by_band,
    aggregate_shap_by_channel,
    plot_shap_summary,
)


SCRIPT18_SCHEMA_VERSION = 2
COMBINED_CONDITION = "specific_coherent"
BASE_CONDITIONS = ("raw", "specific", "coherent")
CONDITIONS = (*BASE_CONDITIONS, COMBINED_CONDITION)
ARRAY_KEYS = {condition: condition for condition in CONDITIONS}


def _release_condition_memory() -> None:
    """Release objects left by one completed condition before the next."""
    gc.collect()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return parsed


def _scalar(data: Any, key: str) -> Any:
    value = data[key]
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _combined_channel_names(ch_names: list[str]) -> list[str]:
    return [
        *(f"specific::{channel}" for channel in ch_names),
        *(f"coherent::{channel}" for channel in ch_names),
    ]


def _condition_feature_names(profile_name: str, condition: str) -> list[str]:
    names = list(PROFILES[profile_name].names)
    if condition != COMBINED_CONDITION:
        return names
    return [
        *(f"specific::{name}" for name in names),
        *(f"coherent::{name}" for name in names),
    ]


def _condition_feature_dim(profile_name: str, condition: str) -> int:
    multiplier = 2 if condition == COMBINED_CONDITION else 1
    return multiplier * PROFILES[profile_name].dim


def _aggregate_feature_names(names: list[str]) -> list[str]:
    return [name.split("::", 1)[-1] for name in names]


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _source_inventory(input_root: Path, cfg: dict) -> tuple[list[Path], str, dict]:
    files = sorted(input_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No Script 18 epoch caches found in {input_root}")
    expected_channels = list(cfg["channels"]["standard_19"])
    records = []
    for path in files:
        with np.load(path, allow_pickle=True) as data:
            required = {
                "raw", "specific", "coherent", COMBINED_CONDITION,
                "ch_names", "specific_coherent_ch_names", "sfreq", "split",
                "label", "class_name", "patient_id", "recording_id",
                "evaluation_id", "fingerprint", "source_mode", "schema_version",
            }
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"Script 18 cache {path} missing keys: {missing}")
            channels = [str(ch) for ch in data["ch_names"]]
            if channels != expected_channels:
                raise ValueError(f"Script 18 cache {path} has invalid channel order")
            source_schema = int(_scalar(data, "schema_version"))
            if source_schema != SCRIPT18_SCHEMA_VERSION:
                raise ValueError(f"Unsupported Script 18 schema in {path}")
            combined_channels = [
                str(ch) for ch in data["specific_coherent_ch_names"]
            ]
            if combined_channels != _combined_channel_names(expected_channels):
                raise ValueError(
                    f"Script 18 cache {path} has invalid combined channel order"
                )
            sfreq = float(_scalar(data, "sfreq"))
            if not np.isclose(sfreq, float(cfg["preprocessing"]["target_sfreq"])):
                raise ValueError(f"Script 18 cache {path} has unexpected sampling rate")
            label = int(_scalar(data, "label"))
            class_name = str(_scalar(data, "class_name"))
            if (label, class_name) not in ((0, "abnormal"), (1, "normal")):
                raise ValueError(f"Invalid TUAB label mapping in {path}")
            split = str(_scalar(data, "split"))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split {split!r} in {path}")
            shapes = {
                key: tuple(np.asarray(data[key]).shape) for key in CONDITIONS
            }
            base_shape = shapes["raw"]
            if not base_shape or any(
                shapes[key] != base_shape for key in BASE_CONDITIONS[1:]
            ):
                raise ValueError(f"Component shape mismatch in {path}")
            if len(base_shape) != 3 or base_shape[1] != len(expected_channels):
                raise ValueError(f"Invalid epoch shape in {path}: {base_shape}")
            expected_combined_shape = (
                base_shape[0], 2 * len(expected_channels), base_shape[2]
            )
            if shapes[COMBINED_CONDITION] != expected_combined_shape:
                raise ValueError(
                    f"Invalid combined epoch shape in {path}: "
                    f"{shapes[COMBINED_CONDITION]}"
                )
            for key in CONDITIONS:
                if data[key].dtype != np.dtype(np.float32):
                    raise ValueError(f"Script 18 cache {path} has non-float32 {key}")
            records.append({
                "path": str(path),
                "fingerprint": str(_scalar(data, "fingerprint")),
                "source_mode": str(_scalar(data, "source_mode")),
                "evaluation_id": str(_scalar(data, "evaluation_id")),
                "schema_version": source_schema,
                "n_epochs": base_shape[0],
            })
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return files, hashlib.sha256(encoded).hexdigest(), {
        "input_dir": str(input_root),
        "n_files": len(files),
        "source_fingerprint": hashlib.sha256(encoded).hexdigest(),
        "source_modes": sorted({row["source_mode"] for row in records}),
    }


def _extract_file(args: tuple) -> tuple:
    index, path_str, condition, split, sfreq, nperseg, freq_band, profile_name, conn_nperseg = args
    profile = PROFILES[profile_name]
    with np.load(path_str, allow_pickle=True) as data:
        if str(_scalar(data, "split")) != split:
            return index, [], [], [], [], [], []
        epochs = np.asarray(data[ARRAY_KEYS[condition]])
        ch_names = [str(ch) for ch in data["ch_names"]]
        label = int(_scalar(data, "label"))
        evaluation_id = str(_scalar(data, "evaluation_id"))
        patient_id = str(_scalar(data, "patient_id"))
        recording_id = str(_scalar(data, "recording_id"))
        if condition == COMBINED_CONDITION:
            n_channels = len(ch_names)
            rows = [
                np.concatenate([
                    profile.extract_fn(
                        epoch[:n_channels], ch_names, sfreq, nperseg,
                        freq_band, conn_nperseg,
                    ),
                    profile.extract_fn(
                        epoch[n_channels:], ch_names, sfreq, nperseg,
                        freq_band, conn_nperseg,
                    ),
                ])
                for epoch in epochs
            ]
        else:
            rows = [profile.extract_fn(
                epoch, ch_names, sfreq, nperseg, freq_band, conn_nperseg
            ) for epoch in epochs]
    expected_dim = _condition_feature_dim(profile_name, condition)
    if any(len(row) != expected_dim for row in rows):
        raise ValueError(f"Feature profile {profile_name} returned an unexpected dimension")
    n = len(rows)
    return (index, rows, [label] * n, [evaluation_id] * n,
            [patient_id] * n, [recording_id] * n, ["tuab"] * n)


def _build_features(files: list[Path], condition: str, split: str, cfg: dict,
                    profile_name: str, max_workers: int | None) -> FeatureDataset:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = int(cfg["wiener"]["nperseg"])
    freq_band = tuple(cfg["wiener"]["freq_band"])
    conn_nperseg = int(cfg.get("ml", {}).get("features", {})
                       .get("connectivity", {}).get("nperseg", nperseg))
    args = [
        (index, str(path), condition, split, sfreq, nperseg, freq_band,
         profile_name, conn_nperseg)
        for index, path in enumerate(files)
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_file, item): item[0] for item in args}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Features [{profile_name}/{condition}/{split}]",
                           leave=False):
            result = future.result()
            results[result[0]] = result[1:]
    rows, labels, evaluation_ids, patient_ids, recording_ids, dataset_names = ([] for _ in range(6))
    for index in sorted(results):
        values = results[index]
        for target, source in zip(
            (rows, labels, evaluation_ids, patient_ids, recording_ids, dataset_names), values
        ):
            target.extend(source)
    if not rows:
        return FeatureDataset(
            np.empty((0, _condition_feature_dim(profile_name, condition))),
            np.empty(0, dtype=np.int64),
            [], [], [], [],
        )
    return FeatureDataset(np.asarray(rows, dtype=np.float64),
                          np.asarray(labels, dtype=np.int64), evaluation_ids,
                          patient_ids, recording_ids, dataset_names)


def _cache_path(root: Path, profile: str, condition: str, split: str) -> Path:
    return root / "features" / profile / f"{condition}_{split}.npz"


def _load_or_extract(files, input_root, source_fingerprint, condition, split,
                     cfg, profile_name, force, max_workers):
    feat_path = _cache_path(input_root, profile_name, condition, split)
    feature_names = _condition_feature_names(profile_name, condition)
    schema_payload = {
        "profile": profile_name,
        "profile_hash": PROFILES[profile_name].hash,
        "sfreq": float(cfg["preprocessing"]["target_sfreq"]),
        "nperseg": int(cfg["wiener"]["nperseg"]),
        "freq_band": list(cfg["wiener"]["freq_band"]),
        "dataset_name": "tuab",
        "source_fingerprint": source_fingerprint,
        "condition": condition,
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "specific_coherent_layout": (
            "channel_axis_specific_then_coherent"
            if condition == COMBINED_CONDITION else None
        ),
    }
    schema_hash = hashlib.sha256(json.dumps(
        schema_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()[:16]
    if feat_path.exists() and not force:
        with np.load(feat_path, allow_pickle=True) as data:
            if str(_scalar(data, "schema_hash")) != schema_hash:
                raise ValueError(f"Feature cache schema mismatch at {feat_path}; re-run with --force")
            if data["X"].ndim != 2 or data["X"].shape[1] != len(feature_names):
                raise ValueError(
                    f"Feature cache dimension mismatch at {feat_path}; "
                    "re-run with --force"
                )
            return FeatureDataset(
                data["X"].astype(np.float64), data["y"].astype(np.int64),
                [str(x) for x in data["evaluation_ids"]],
                [str(x) for x in data["patient_ids"]],
                [str(x) for x in data["recording_ids"]],
                [str(x) for x in data["dataset_names"]],
            )
    dataset = _build_features(files, condition, split, cfg, profile_name, max_workers)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feat_path, X=dataset.X, y=dataset.y,
        evaluation_ids=np.asarray(dataset.evaluation_ids, dtype="U"),
        patient_ids=np.asarray(dataset.patient_ids, dtype="U"),
        recording_ids=np.asarray(dataset.recording_ids, dtype="U"),
        dataset_names=np.asarray(dataset.dataset_names, dtype="U"),
        schema_hash=np.asarray(schema_hash), schema_payload=np.asarray(json.dumps(schema_payload)),
    )
    return dataset


def _split_stats(dataset: FeatureDataset) -> dict:
    return {
        "n_epochs": int(len(dataset.X)),
        "n_evaluations": int(len(set(dataset.evaluation_ids))),
        "n_patients": int(len(set(dataset.patient_ids))),
        "n_epochs_abnormal": int(np.sum(dataset.y == 0)),
        "n_epochs_normal": int(np.sum(dataset.y == 1)),
        "n_evaluations_abnormal": int(len({i for i, y in zip(dataset.evaluation_ids, dataset.y) if y == 0})),
        "n_evaluations_normal": int(len({i for i, y in zip(dataset.evaluation_ids, dataset.y) if y == 1})),
    }


def _assert_no_patient_overlap(datasets: dict[str, FeatureDataset]) -> None:
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(datasets[left].patient_ids) & set(datasets[right].patient_ids)
        if overlap:
            raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)[:5]}")


def run_condition(condition, cfg, files, input_root, source_fingerprint,
                  out_root, profile_name, force, max_workers, source_meta):
    datasets = {
        split: _load_or_extract(files, input_root, source_fingerprint, condition,
                                split, cfg, profile_name, force, max_workers)
        for split in ("train", "val", "test")
    }
    if not datasets["train"].X.shape[0] or not datasets["val"].X.shape[0]:
        raise ValueError(f"Condition {condition!r} requires non-empty train and val sets")
    _assert_no_patient_overlap(datasets)
    if set(datasets["train"].y) != {0, 1}:
        raise ValueError("Training split must contain both TUAB labels")
    out_dir = out_root / profile_name / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = _condition_feature_names(profile_name, condition)
    metadata = {
        **source_meta,
        "condition": condition,
        "feature_set": profile_name,
        "feature_dim": len(feature_names),
        "specific_coherent_layout": (
            "channel_axis_specific_then_coherent"
            if condition == COMBINED_CONDITION else None
        ),
    }
    meta_path = out_dir / "run_metadata.json"
    if meta_path.exists() and not force:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous.get("source_fingerprint") != source_fingerprint:
            raise ValueError(f"Existing results at {out_dir} use a different input; use --force or change results_dir")
    _save_json({
        **metadata, "data_stats": {key: _split_stats(value) for key, value in datasets.items()},
    }, meta_path)
    _save_json({
        **{key: _split_stats(value) for key, value in datasets.items()},
        "feature_set": profile_name, "dataset_name": "tuab", "aggregation_unit": "recording",
    }, out_dir / "data_stats.json")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(datasets["train"].X)
    X_val = scaler.transform(datasets["val"].X)
    X_test = scaler.transform(datasets["test"].X) if len(datasets["test"].X) else datasets["test"].X
    joblib.dump(scaler, out_dir / "scaler.joblib")
    model = train_xgboost(X_train, datasets["train"].y, X_val, datasets["val"].y,
                          cfg, groups=datasets["train"].patient_ids)
    joblib.dump(model, out_dir / "model.joblib")
    _save_json(model.get_params(), out_dir / "best_params.json")

    val_df = evaluation_level_predict(model, X_val, datasets["val"].y,
                                      datasets["val"].evaluation_ids, datasets["val"].patient_ids,
                                      datasets["val"].recording_ids, datasets["val"].dataset_names)
    threshold = find_optimal_threshold(val_df)
    val_df["predicted_label"] = (val_df["pred_proba"] >= threshold).astype(int)
    val_df.to_csv(out_dir / "val_predictions.csv", index=False)
    val_metrics = evaluate_subject_level(val_df, threshold)
    val_metrics.update({"positive_class": "normal", "aggregation_unit": "recording"})
    _save_json(val_metrics, out_dir / "val_metrics.json")

    test_metrics = {}
    if len(datasets["test"].X):
        test_df = evaluation_level_predict(model, X_test, datasets["test"].y,
                                            datasets["test"].evaluation_ids, datasets["test"].patient_ids,
                                            datasets["test"].recording_ids, datasets["test"].dataset_names)
        test_df["predicted_label"] = (test_df["pred_proba"] >= threshold).astype(int)
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)
        test_metrics = evaluate_subject_level(test_df, threshold)
        test_metrics.update({"positive_class": "normal", "aggregation_unit": "recording"})
        _save_json(test_metrics, out_dir / "test_metrics.json")

        names = feature_names
        aggregate_names = _aggregate_feature_names(names)
        shap_values = compute_shap_values(model, X_test, names)
        np.save(out_dir / "shap_values_test.npy", shap_values)
        _save_json(
            aggregate_shap_by_band(shap_values, aggregate_names),
            out_dir / "shap_by_band.json",
        )
        _save_json(
            aggregate_shap_by_channel(shap_values, aggregate_names),
            out_dir / "shap_by_channel.json",
        )
        plot_shap_summary(shap_values, X_test, names,
                          title=f"SHAP Summary — {condition} [{profile_name}] (test set)",
                          output_path=out_dir / "shap_summary.png",
                          max_display=int(cfg["ml"]["shap"]["max_display"]),
                          dpi=int(cfg["ml"]["shap"]["dpi"]))
    return {"condition": condition, "val_metrics": val_metrics,
            "test_metrics": test_metrics, "data_stats": {k: _split_stats(v) for k, v in datasets.items()}}


def main(config_path="configs/tuab.yaml", condition="all", force=False,
         feature_set="base211", max_workers=None):
    cfg = load_config(config_path)
    if active_dataset_name(cfg) != "tuab":
        raise ValueError("Script 20 requires dataset.active: tuab")
    mode = str(cfg["wiener"].get("mode", "frequency"))
    input_root = Path(cfg["paths"]["cache_dir"]) / f"tuab_continuous_wiener_{mode}" / "epochs"
    files, source_fingerprint, source_meta = _source_inventory(input_root, cfg)
    out_root = Path(cfg["paths"]["results_dir"]) / "tuab_component_xgboost"
    out_root.mkdir(parents=True, exist_ok=True)
    source_meta = {**source_meta, "requested_mode": mode}
    resolved_cfg = dict(cfg)
    resolved_cfg["tuab_component_xgboost"] = {
        "input_dir": str(input_root),
        "output_dir": str(out_root),
        "requested_mode": mode,
        "source_fingerprint": source_fingerprint,
        "conditions": list(CONDITIONS),
        "specific_coherent_layout": "channel_axis_specific_then_coherent",
    }
    (out_root / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=False), encoding="utf-8"
    )
    conditions = CONDITIONS if condition == "all" else (condition,)
    results = []
    for selected_condition in conditions:
        try:
            results.append(run_condition(
                selected_condition, cfg, files, input_root, source_fingerprint,
                out_root, feature_set, force, max_workers, source_meta,
            ))
        finally:
            _release_condition_memory()
    if len(results) > 1:
        rows = []
        for result in results:
            row = {"condition": result["condition"]}
            for prefix in ("val", "test"):
                row.update({f"{prefix}_{key}": value for key, value in result[f"{prefix}_metrics"].items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_root / feature_set / "comparison_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TUAB component XGBoost models from Script 18 epochs")
    parser.add_argument("--config", default="configs/tuab.yaml")
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
    parser.add_argument("--feature-set", choices=list(PROFILES), default="base211")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=_positive_int, default=None,
        help="Feature extraction worker processes "
             "(default: ProcessPoolExecutor default)",
    )
    args = parser.parse_args()
    main(args.config, args.condition, args.force, args.feature_set, args.workers)
