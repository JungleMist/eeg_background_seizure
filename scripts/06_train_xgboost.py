"""Train XGBoost classifiers and generate SHAP analysis for preprocessing conditions.

For each condition (raw / ica / wiener / wiener_phasegated /
wiener_zerophase), the script:
  1. Loads or extracts handcrafted features (see FEATURE_NAMES) from the NPZ caches.
  2. Scales features with ``StandardScaler`` (fit on train, applied to val/test).
  3. Trains an ``XGBClassifier`` via 5-fold GridSearchCV + early-stopping refit.
  4. Evaluates by dataset unit: TUEP patient or TUAB recording (mean epoch probability).
  5. Computes SHAP values on the test set and saves per-condition analysis files.

When all five XGBoost conditions have been processed (``--condition all``), a
cross-condition comparison CSV and a 2 × 5 publication SHAP figure are written.

Usage
-----
# All XGBoost conditions (default)
python scripts/06_train_xgboost.py

# Single condition
python scripts/06_train_xgboost.py --condition wiener

# Ignore feature cache and re-extract
python scripts/06_train_xgboost.py --force

# Re-extract with four feature worker processes
python scripts/06_train_xgboost.py --force --workers 4

``--workers`` controls file-level feature-extraction processes only. Conditions
still run sequentially, and cached features are loaded without starting workers.

Output
------
results/xgboost/{condition}/
    model.joblib           — fitted XGBClassifier
    scaler.joblib          — fitted StandardScaler
    best_params.json       — GridSearchCV best hyperparameters
    val_metrics.json       — {auroc, f1, accuracy} on validation set
    test_metrics.json      — {auroc, f1, accuracy} on test set
    val_predictions.csv    — evaluation/patient/recording IDs, probabilities, labels
    test_predictions.csv   — evaluation/patient/recording IDs, probabilities, labels
    shap_values_test.npy   — (n_test_epochs, len(FEATURE_NAMES)) raw SHAP values
    shap_summary.png       — SHAP beeswarm plot (top 20 features)
    shap_by_band.json      — mean |SHAP| per feature-type group
    shap_by_channel.json   — mean |SHAP| per EEG channel
    data_stats.json        — {train,val,test} subject + epoch counts

results/xgboost/
    comparison_summary.csv — 5 conditions × {val_auroc, test_auroc, f1, acc}
    shap_comparison.png    — 2×5 publication figure (after --condition all)
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must precede pyplot

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.features.extraction import FeatureDataset
from eeg_bg.io.dataset import active_dataset_config, active_dataset_name
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
    plot_shap_comparison,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_extract_features(
    cache_root: Path,
    feature_cache_dir: Path,
    condition: str,
    split: str,
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
    force: bool,
    feature_set: str = "base211",
    connectivity_nperseg: int | None = None,
    max_workers: int | None = None,
    dataset_name: str = "tuep",
) -> FeatureDataset:
    """Load or build a profile-aware feature dataset with identities."""
    from eeg_bg.features.profiles import PROFILES
    profile = PROFILES[feature_set]

    # Profile-aware cache path: base211 uses the legacy flat path for backward
    # compatibility; other profiles nest under features/{profile_name}/.
    if feature_set == "base211":
        feat_file = feature_cache_dir / f"{condition}_{split}.npz"
    else:
        feat_file = feature_cache_dir / feature_set / f"{condition}_{split}.npz"

    # Schema hash covers feature names, sfreq, nperseg, freq_band.
    schema_hash = hashlib.sha256(
        f"{profile.names}|{sfreq}|{nperseg}|{freq_band}|"
        f"{connectivity_nperseg}|{dataset_name}".encode()
    ).hexdigest()[:16]

    if feat_file.exists() and not force:
        data = np.load(feat_file, allow_pickle=True)
        saved_hash = str(data.get("schema_hash", ""))
        if saved_hash and saved_hash != schema_hash:
            raise ValueError(
                f"Feature cache schema hash mismatch ({saved_hash} vs "
                f"{schema_hash}) for {feature_set}/{condition}_{split}. "
                f"Re-run script 06 with --force."
            )
        X    = data["X"].astype(np.float64)
        y    = data["y"].astype(np.int64)
        evaluation_ids = list(data.get("evaluation_ids", data["subject_ids"]))
        patient_ids = list(data.get("patient_ids", data["subject_ids"]))
        recording_ids = list(data.get(
            "recording_ids", np.asarray([""] * len(y), dtype=object)
        ))
        dataset_names = list(data.get(
            "dataset_names", np.asarray([dataset_name] * len(y), dtype=object)
        ))
        if X.shape[1] != profile.dim:
            raise ValueError(
                f"Feature cache has {X.shape[1]} dims but profile "
                f"{feature_set!r} expects {profile.dim}. "
                f"Re-run script 06 with --force."
            )
        return FeatureDataset(
            X, y, evaluation_ids, patient_ids, recording_ids, dataset_names
        )

    from eeg_bg.features.extraction import build_feature_dataset_with_profile
    dataset = build_feature_dataset_with_profile(
        cache_root, condition, split,
        sfreq=sfreq, nperseg=nperseg, freq_band=freq_band,
        profile_name=feature_set, connectivity_nperseg=connectivity_nperseg,
        max_workers=max_workers,
    )
    if len(dataset.X):
        feat_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            feat_file,
            X=dataset.X,
            y=dataset.y,
            evaluation_ids=np.asarray(dataset.evaluation_ids, dtype=object),
            subject_ids=np.asarray(dataset.evaluation_ids, dtype=object),
            patient_ids=np.asarray(dataset.patient_ids, dtype=object),
            recording_ids=np.asarray(dataset.recording_ids, dtype=object),
            dataset_names=np.asarray(dataset.dataset_names, dtype=object),
            schema_hash=schema_hash,
        )
    return dataset


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _split_stats(
    dataset: FeatureDataset,
    class_names: dict[int, str],
) -> dict:
    """Compute generic epoch, patient, and evaluation-unit counts."""
    stats = {
        "n_epochs": int(len(dataset.X)),
        "n_evaluations": int(len(set(dataset.evaluation_ids))),
        "n_patients": int(len(set(dataset.patient_ids))),
    }
    stats["n_subjects"] = stats["n_evaluations"]  # compatibility alias
    for label, class_name in class_names.items():
        safe_name = class_name.replace(" ", "_")
        stats[f"n_epochs_{safe_name}"] = int(np.sum(dataset.y == label))
        stats[f"n_evaluations_{safe_name}"] = int(len({
            evaluation_id
            for evaluation_id, row_label
            in zip(dataset.evaluation_ids, dataset.y)
            if row_label == label
        }))
    return stats


def _print_data_stats(
    data_stats: dict, aggregation_unit: str = "evaluation unit",
) -> None:
    """Print split statistics without treating metadata as a split."""
    for split in ("train", "val", "test"):
        s = data_stats.get(split)
        if not isinstance(s, dict):
            continue
        n_evaluations = s.get("n_evaluations", s.get("n_subjects", 0))
        n_patients = s.get("n_patients", s.get("n_subjects", 0))
        print(
            f"  {split.capitalize():5}: {n_evaluations:4d} "
            f"{aggregation_unit}s / {n_patients:4d} patients / "
            f"{s['n_epochs']:5d} epochs"
        )


# ── Per-condition pipeline ────────────────────────────────────────────────────

def run_condition(
    condition: str,
    cfg: dict,
    cache_root: Path,
    feature_cache_dir: Path,
    out_root: Path,
    force: bool,
    feature_set: str = "base211",
    max_workers: int | None = None,
) -> dict:
    """Full pipeline for one condition.  Returns metrics + SHAP aggregates."""
    sfreq      = float(cfg["preprocessing"]["target_sfreq"])
    nperseg    = int(cfg["wiener"]["nperseg"])
    connectivity_nperseg = int(cfg.get("ml", {}).get("features", {})
                               .get("connectivity", {}).get("nperseg", nperseg))
    freq_band  = tuple(cfg["wiener"]["freq_band"])
    shap_cfg   = cfg["ml"]["shap"]
    dataset_name = active_dataset_name(cfg)
    dataset_block = active_dataset_config(cfg)
    class_names = {}
    for name, value in dataset_block["classes"].items():
        default_label = 0 if name in {"epilepsy", "abnormal"} else 1
        label = int(value.get("label", default_label)) if isinstance(value, dict) \
            else default_label
        class_names[label] = name
    positive_class = class_names[1]
    aggregation_unit = "recording" if dataset_name == "tuab" else "patient"
    out_dir    = out_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Condition: {condition.upper()}  |  Feature set: {feature_set}")
    print(f"{'='*60}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Loading features...")
    train_data = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "train",
        sfreq, nperseg, freq_band, force, feature_set,
        connectivity_nperseg, max_workers, dataset_name,
    )
    val_data = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "val",
        sfreq, nperseg, freq_band, force, feature_set,
        connectivity_nperseg, max_workers, dataset_name,
    )
    test_data = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "test",
        sfreq, nperseg, freq_band, force, feature_set,
        connectivity_nperseg, max_workers, dataset_name,
    )

    if len(train_data.X) == 0 or len(val_data.X) == 0:
        raise ValueError(
            f"Condition {condition!r} needs non-empty train and validation "
            "feature sets. Run script 01 on a complete training partition first."
        )
    for left_name, left, right_name, right in (
        ("train", train_data, "val", val_data),
        ("train", train_data, "test", test_data),
        ("val", val_data, "test", test_data),
    ):
        overlap = set(left.patient_ids) & set(right.patient_ids)
        if overlap:
            raise ValueError(
                f"Patient leakage between {left_name} and {right_name}: "
                f"{sorted(overlap)[:5]}"
            )

    # ── Data statistics ───────────────────────────────────────────────────────
    data_stats = {
        "train": _split_stats(train_data, class_names),
        "val":   _split_stats(val_data, class_names),
        "test":  _split_stats(test_data, class_names),
        "feature_set": feature_set,
        "dataset_name": dataset_name,
        "aggregation_unit": aggregation_unit,
    }
    _save_json(data_stats, out_dir / "data_stats.json")

    _print_data_stats(data_stats, aggregation_unit)

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(train_data.X)
    X_val_sc  = scaler.transform(val_data.X)
    X_test_sc = scaler.transform(test_data.X) if len(test_data.X) else test_data.X
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # ── Training ──────────────────────────────────────────────────────────────
    print("Training XGBoost (GridSearchCV + early stopping)...")
    model = train_xgboost(
        X_tr_sc, train_data.y, X_val_sc, val_data.y, cfg,
        groups=train_data.patient_ids,
    )
    joblib.dump(model, out_dir / "model.joblib")

    # Save best hyperparameters
    best_params = model.get_params()
    _save_json(best_params, out_dir / "best_params.json")

    # ── Validation evaluation ─────────────────────────────────────────────────
    val_metrics: dict = {}
    opt_threshold = 0.5
    if len(val_data.X):
        val_df = evaluation_level_predict(
            model, X_val_sc, val_data.y, val_data.evaluation_ids,
            val_data.patient_ids, val_data.recording_ids, val_data.dataset_names,
        )
        # Find F1-optimal threshold on validation set; apply to both val+test.
        opt_threshold = find_optimal_threshold(val_df)
        val_df["predicted_label"] = (
            val_df["pred_proba"] >= opt_threshold
        ).astype(int)
        val_df.to_csv(out_dir / "val_predictions.csv", index=False)
        val_metrics = evaluate_subject_level(val_df, threshold=opt_threshold)
        val_metrics.update({
            "positive_class": positive_class,
            "aggregation_unit": aggregation_unit,
        })
        _save_json(val_metrics, out_dir / "val_metrics.json")
        print(f"  Val  → AUROC {val_metrics['auroc']:.3f}  "
              f"F1 {val_metrics['f1']:.3f}  "
              f"Acc {val_metrics['accuracy']:.3f}  "
              f"(threshold={opt_threshold:.3f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_metrics: dict = {}
    if len(test_data.X):
        test_df = evaluation_level_predict(
            model, X_test_sc, test_data.y, test_data.evaluation_ids,
            test_data.patient_ids, test_data.recording_ids,
            test_data.dataset_names,
        )
        test_df["predicted_label"] = (
            test_df["pred_proba"] >= opt_threshold
        ).astype(int)
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)
        test_metrics = evaluate_subject_level(test_df, threshold=opt_threshold)
        test_metrics.update({
            "positive_class": positive_class,
            "aggregation_unit": aggregation_unit,
        })
        _save_json(test_metrics, out_dir / "test_metrics.json")
        print(f"  Test → AUROC {test_metrics['auroc']:.3f}  "
              f"F1 {test_metrics['f1']:.3f}  "
              f"Acc {test_metrics['accuracy']:.3f}  "
              f"(threshold={opt_threshold:.3f})")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    if len(test_data.X):
        from eeg_bg.features.profiles import PROFILES
        feat_names = PROFILES[feature_set].names
        print("Computing SHAP values...")
        shap_vals = compute_shap_values(model, X_test_sc, feat_names)
        np.save(out_dir / "shap_values_test.npy", shap_vals)

        band_agg = aggregate_shap_by_band(shap_vals, feat_names)
        ch_agg   = aggregate_shap_by_channel(shap_vals, feat_names)
        _save_json(band_agg, out_dir / "shap_by_band.json")
        _save_json(ch_agg,   out_dir / "shap_by_channel.json")

        plot_shap_summary(
            shap_vals, X_test_sc, feat_names,
            title=f"SHAP Summary — {condition.capitalize()} [{feature_set}] (test set)",
            output_path=out_dir / "shap_summary.png",
            max_display=int(shap_cfg["max_display"]),
            dpi=int(shap_cfg["dpi"]),
        )
        print(f"  SHAP saved to {out_dir / 'shap_summary.png'}")
    else:
        band_agg = {}
        ch_agg   = {}

    return {
        "condition":       condition,
        "val_metrics":     val_metrics,
        "test_metrics":    test_metrics,
        "shap_by_band":    band_agg,
        "shap_by_channel": ch_agg,
        "data_stats":      data_stats,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

XGB_CONDITIONS = [
    "raw", "ica", "wiener", "wiener_phasegated", "wiener_zerophase",
]


def _positive_int(value: str) -> int:
    """Argparse type for strictly positive worker counts."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return parsed


def main(config_path: str, condition: str, force: bool,
         feature_set: str = "base211",
         max_workers: int | None = None) -> None:
    cfg         = load_config(config_path)
    cache_root  = Path(cfg["paths"]["cache_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    feat_cache  = cache_root / "features"
    # Keep feature profiles isolated.  A profile-aware tree prevents the
    # connectivity readout from overwriting the backward-compatible 211-dim
    # results when both are trained for the same preprocessing condition.
    out_root    = results_dir / "xgboost" / feature_set

    conditions = XGB_CONDITIONS if condition == "all" else [condition]
    all_results: dict[str, dict] = {}

    for cond in conditions:
        result = run_condition(
            cond, cfg, cache_root, feat_cache, out_root, force,
            feature_set=feature_set, max_workers=max_workers,
        )
        if result:
            all_results[cond] = result

    # ── Cross-condition summary ───────────────────────────────────────────────
    if len(all_results) > 1:
        rows = []
        for cond, res in all_results.items():
            row = {"condition": cond}
            for prefix, metrics in [("val", res.get("val_metrics", {})),
                                     ("test", res.get("test_metrics", {}))]:
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v
            stats = res.get("data_stats", {})
            for split in ("train", "val", "test"):
                s = stats.get(split, {})
                row[f"{split}_n_subjects"] = s.get("n_subjects", "")
                row[f"{split}_n_epochs"]   = s.get("n_epochs", "")
            rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_path = out_root / "comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nComparison summary saved to {summary_path}")

        # ── SHAP comparison figure ────────────────────────────────────────────
        if all(c in all_results for c in XGB_CONDITIONS):
            shap_results = {
                c: {
                    "shap_by_band":    all_results[c].get("shap_by_band", {}),
                    "shap_by_channel": all_results[c].get("shap_by_channel", {}),
                }
                for c in XGB_CONDITIONS
            }
            comp_fig_path = out_root / "shap_comparison.png"
            plot_shap_comparison(
                shap_results,
                output_path=comp_fig_path,
                dpi=int(cfg["ml"]["shap"]["dpi"]),
            )
            print(f"SHAP comparison figure saved to {comp_fig_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifiers + generate SHAP analysis"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--condition",
        choices=XGB_CONDITIONS + ["all"],
        default="all",
        help="Preprocessing condition to train (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract features even if feature cache exists",
    )
    parser.add_argument(
        "--feature-set",
        choices=["base211", "base211_conn80"],
        default="base211",
        help="Feature profile to use (default: base211, 211-dim); "
             "base211_conn80 adds 80-dim connectivity → 291-dim total.",
    )
    parser.add_argument(
        "--workers", type=_positive_int, default=None,
        help="Feature extraction worker processes "
             "(default: ProcessPoolExecutor default)",
    )
    args = parser.parse_args()
    main(
        args.config, args.condition, args.force,
        feature_set=args.feature_set, max_workers=args.workers,
    )
