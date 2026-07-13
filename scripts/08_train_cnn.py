"""Script 08 — Train EEGNet CNN on raw / wiener / ICA epoch caches.

Usage
-----
conda run -n eeg_pipeline python scripts/08_train_cnn.py [OPTIONS]

Options
-------
--condition  raw | ica | wiener | all   Which condition(s) to train (default: all)
--config     PATH                       Config YAML path (default: configs/default.yaml)
--force                                 Re-train even if output already exists
--workers    N                          DataLoader num_workers override (default: from config)

Output
------
results/cnn/{condition}/
    best_model.pt, best_params.json,
    val_metrics.json, test_metrics.json,
    val_predictions.csv, test_predictions.csv
results/cnn/comparison_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import pandas as pd

# Resolve project root so the script works regardless of cwd
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import active_dataset_name
from eeg_bg.ml.cnn_pipeline import train_cnn


_CONDITIONS = ["raw", "ica", "wiener"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEGNet CNN on EEG epoch caches."
    )
    parser.add_argument(
        "--condition",
        default="all",
        choices=_CONDITIONS + ["all"],
        help="Which condition to train (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader num_workers override (default: value from config)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = load_config(args.config)
    if active_dataset_name(cfg) != "tuep":
        raise SystemExit(
            "Script 08 (EEGNet CNN) currently supports TUEP only. "
            "Use scripts 01-07 for TUAB."
        )

    # Allow --workers CLI flag to override the config value
    if args.workers is not None:
        cfg["ml"]["cnn"]["num_workers"] = args.workers

    results_dir = Path(cfg["paths"]["results_dir"])
    conditions  = _CONDITIONS if args.condition == "all" else [args.condition]

    all_metrics: list[dict] = []

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Training CNN — condition: {condition}")
        print(f"{'='*60}")

        out_dir = results_dir / "cnn" / condition
        metrics = train_cnn(
            condition=condition,
            cfg=cfg,
            out_dir=out_dir,
            force=args.force,
        )

        print(f"  val  AUROC={metrics['val']['auroc']:.4f}  "
              f"F1={metrics['val']['f1']:.4f}  "
              f"Acc={metrics['val']['accuracy']:.4f}")
        print(f"  test AUROC={metrics['test']['auroc']:.4f}  "
              f"F1={metrics['test']['f1']:.4f}  "
              f"Acc={metrics['test']['accuracy']:.4f}")

        all_metrics.append({
            "condition":     condition,
            "val_auroc":     metrics["val"]["auroc"],
            "val_f1":        metrics["val"]["f1"],
            "val_accuracy":  metrics["val"]["accuracy"],
            "test_auroc":    metrics["test"]["auroc"],
            "test_f1":       metrics["test"]["f1"],
            "test_accuracy": metrics["test"]["accuracy"],
        })

    # Write comparison summary only when all three conditions were run
    if len(all_metrics) == 3:
        summary_path = results_dir / "cnn" / "comparison_summary.csv"
        pd.DataFrame(all_metrics).to_csv(summary_path, index=False)
        print(f"\nComparison summary written to {summary_path}")


if __name__ == "__main__":
    main()
