"""Train XGBoost classifiers and generate SHAP analysis for four preprocessing conditions.

For each condition (raw / ica / wiener / wiener_zerophase), the script:
  1. Loads or extracts handcrafted features (see FEATURE_NAMES) from the NPZ caches.
  2. Scales features with ``StandardScaler`` (fit on train, applied to val/test).
  3. Trains an ``XGBClassifier`` via 5-fold GridSearchCV + early-stopping refit.
  4. Evaluates at subject level (mean of per-epoch ``predict_proba``).
  5. Computes SHAP values on the test set and saves per-condition analysis files.

When all four conditions have been processed (``--condition all``), a
cross-condition comparison CSV and a 2 × 4 publication SHAP figure are written.

Usage
-----
# All three conditions (default)
python scripts/06_train_xgboost.py

# Single condition
python scripts/06_train_xgboost.py --condition wiener

# Ignore feature cache and re-extract
python scripts/06_train_xgboost.py --force

Output
------
results/xgboost/{condition}/
    model.joblib           — fitted XGBClassifier
    scaler.joblib          — fitted StandardScaler
    best_params.json       — GridSearchCV best hyperparameters
    val_metrics.json       — {auroc, f1, accuracy} on validation set
    test_metrics.json      — {auroc, f1, accuracy} on test set
    val_predictions.csv    — subject_id, pred_proba, true_label
    test_predictions.csv   — subject_id, pred_proba, true_label
    shap_values_test.npy   — (n_test_epochs, len(FEATURE_NAMES)) raw SHAP values
    shap_summary.png       — SHAP beeswarm plot (top 20 features)
    shap_by_band.json      — mean |SHAP| per feature-type group
    shap_by_channel.json   — mean |SHAP| per EEG channel
    data_stats.json        — {train,val,test} subject + epoch counts

results/xgboost/
    comparison_summary.csv — 4 conditions × {val_auroc, test_auroc, f1, acc}
    shap_comparison.png    — 2×4 publication figure (after --condition all)
"""
import argparse
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
from eeg_bg.features.extraction import (
    build_dataset,
    FEATURE_NAMES,
)
from eeg_bg.ml.xgb_pipeline import (
    train_xgboost,
    subject_level_predict,
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
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, subject_ids), using a feature-level NPZ cache."""
    feat_file = feature_cache_dir / f"{condition}_{split}.npz"
    if feat_file.exists() and not force:
        data = np.load(feat_file, allow_pickle=True)
        X    = data["X"].astype(np.float64)
        y    = data["y"].astype(np.int64)
        sids = list(data["subject_ids"])
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature cache has {X.shape[1]} dims but FEATURE_NAMES has "
                f"{len(FEATURE_NAMES)}. Re-run script 06 with --force."
            )
        return X, y, sids

    X, y, sids = build_dataset(
        cache_root, condition, split,
        sfreq=sfreq, nperseg=nperseg, freq_band=freq_band,
    )
    if len(X):
        feat_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feat_file, X=X, y=y, subject_ids=np.array(sids, dtype=object))
    return X, y, sids


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _split_stats(
    X: np.ndarray,
    y: np.ndarray,
    sids: list[str],
) -> dict:
    """Compute subject / epoch counts for one data split.

    Returns
    -------
    dict with keys:
        n_epochs, n_subjects,
        n_epochs_epilepsy, n_epochs_control,
        n_subjects_epilepsy, n_subjects_control
    """
    if len(X) == 0:
        return {
            "n_epochs": 0, "n_subjects": 0,
            "n_epochs_epilepsy": 0, "n_epochs_control": 0,
            "n_subjects_epilepsy": 0, "n_subjects_control": 0,
        }
    return {
        "n_epochs":            int(len(X)),
        "n_subjects":          int(len(set(sids))),
        "n_epochs_epilepsy":   int(np.sum(y == 0)),
        "n_epochs_control":    int(np.sum(y == 1)),
        "n_subjects_epilepsy": int(len({s for s, l in zip(sids, y) if l == 0})),
        "n_subjects_control":  int(len({s for s, l in zip(sids, y) if l == 1})),
    }


# ── Per-condition pipeline ────────────────────────────────────────────────────

def run_condition(
    condition: str,
    cfg: dict,
    cache_root: Path,
    feature_cache_dir: Path,
    out_root: Path,
    force: bool,
) -> dict:
    """Full pipeline for one condition.  Returns metrics + SHAP aggregates."""
    sfreq      = float(cfg["preprocessing"]["target_sfreq"])
    nperseg    = int(cfg["wiener"]["nperseg"])
    freq_band  = tuple(cfg["wiener"]["freq_band"])
    shap_cfg   = cfg["ml"]["shap"]
    out_dir    = out_root / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Condition: {condition.upper()}")
    print(f"{'='*60}")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Loading features...")
    X_train, y_train, sids_train = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "train",
        sfreq, nperseg, freq_band, force,
    )
    X_val,   y_val,   sids_val   = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "val",
        sfreq, nperseg, freq_band, force,
    )
    X_test,  y_test,  sids_test  = _load_or_extract_features(
        cache_root, feature_cache_dir, condition, "test",
        sfreq, nperseg, freq_band, force,
    )

    if len(X_train) == 0:
        print(f"  [WARNING] No training data found for condition '{condition}'. "
              "Run scripts 01-03 first.")
        return {}

    # ── Data statistics ───────────────────────────────────────────────────────
    data_stats = {
        "train": _split_stats(X_train, y_train, sids_train),
        "val":   _split_stats(X_val,   y_val,   sids_val),
        "test":  _split_stats(X_test,  y_test,  sids_test),
    }
    _save_json(data_stats, out_dir / "data_stats.json")

    for split, s in data_stats.items():
        print(f"  {split.capitalize():5}: {s['n_subjects']:4d} subjects / "
              f"{s['n_epochs']:5d} epochs  "
              f"({s['n_subjects_epilepsy']}E + {s['n_subjects_control']}C subjects)")

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_val_sc  = scaler.transform(X_val)  if len(X_val)  else X_val
    X_test_sc = scaler.transform(X_test) if len(X_test) else X_test
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # ── Training ──────────────────────────────────────────────────────────────
    print("Training XGBoost (GridSearchCV + early stopping)...")
    model = train_xgboost(X_tr_sc, y_train, X_val_sc, y_val, cfg)
    joblib.dump(model, out_dir / "model.joblib")

    # Save best hyperparameters
    best_params = model.get_params()
    _save_json(best_params, out_dir / "best_params.json")

    # ── Validation evaluation ─────────────────────────────────────────────────
    val_metrics: dict[str, float] = {}
    opt_threshold = 0.5
    if len(X_val):
        val_df = subject_level_predict(model, X_val_sc, sids_val, y_val)
        val_df.to_csv(out_dir / "val_predictions.csv", index=False)
        # Find F1-optimal threshold on validation set; apply to both val+test.
        opt_threshold = find_optimal_threshold(val_df)
        val_metrics = evaluate_subject_level(val_df, threshold=opt_threshold)
        _save_json(val_metrics, out_dir / "val_metrics.json")
        print(f"  Val  → AUROC {val_metrics['auroc']:.3f}  "
              f"F1 {val_metrics['f1']:.3f}  "
              f"Acc {val_metrics['accuracy']:.3f}  "
              f"(threshold={opt_threshold:.3f})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_metrics: dict[str, float] = {}
    if len(X_test):
        test_df = subject_level_predict(model, X_test_sc, sids_test, y_test)
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)
        test_metrics = evaluate_subject_level(test_df, threshold=opt_threshold)
        _save_json(test_metrics, out_dir / "test_metrics.json")
        print(f"  Test → AUROC {test_metrics['auroc']:.3f}  "
              f"F1 {test_metrics['f1']:.3f}  "
              f"Acc {test_metrics['accuracy']:.3f}  "
              f"(threshold={opt_threshold:.3f})")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    if len(X_test):
        print("Computing SHAP values...")
        shap_vals = compute_shap_values(model, X_test_sc, FEATURE_NAMES)
        np.save(out_dir / "shap_values_test.npy", shap_vals)

        band_agg = aggregate_shap_by_band(shap_vals, FEATURE_NAMES)
        ch_agg   = aggregate_shap_by_channel(shap_vals, FEATURE_NAMES)
        _save_json(band_agg, out_dir / "shap_by_band.json")
        _save_json(ch_agg,   out_dir / "shap_by_channel.json")

        plot_shap_summary(
            shap_vals, X_test_sc, FEATURE_NAMES,
            title=f"SHAP Summary — {condition.capitalize()} (test set)",
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

def main(config_path: str, condition: str, force: bool) -> None:
    cfg         = load_config(config_path)
    cache_root  = Path(cfg["paths"]["cache_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    feat_cache  = cache_root / "features"
    out_root    = results_dir / "xgboost"

    conditions = ["raw", "ica", "wiener", "wiener_zerophase"] if condition == "all" else [condition]
    all_results: dict[str, dict] = {}

    for cond in conditions:
        result = run_condition(
            cond, cfg, cache_root, feat_cache, out_root, force,
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
        if all(c in all_results for c in ["raw", "ica", "wiener", "wiener_zerophase"]):
            shap_results = {
                c: {
                    "shap_by_band":    all_results[c].get("shap_by_band", {}),
                    "shap_by_channel": all_results[c].get("shap_by_channel", {}),
                }
                for c in ["raw", "ica", "wiener", "wiener_zerophase"]
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
        choices=["raw", "ica", "wiener", "wiener_zerophase", "all"],
        default="all",
        help="Preprocessing condition to train (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract features even if feature cache exists",
    )
    args = parser.parse_args()
    main(args.config, args.condition, args.force)
