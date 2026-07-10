#!/usr/bin/env python3
"""Summarise the fixed eight-cell Wiener phase/coherence experiment grid.

The script consumes the profile-aware ``results/exp_wiener_phase/expN`` trees
and uses the archived subject-level prediction CSVs for paired comparisons.
It never retrains a model and keeps the analysis bootstrap seed fixed at 42.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_bg.config.settings import load_config


MATRIX = [
    (1, "frequency", np.pi, 0.15),
    (2, "frequency", np.pi, 0.45),
    (3, "frequency", np.pi, 0.75),
    (4, "phasegated", np.pi / 2, 0.15),
    (5, "phasegated", np.pi / 5, 0.15),
    (6, "phasegated", np.pi / 10, 0.15),
    (7, "phasegated", np.pi / 10, 0.45),
    (8, "phasegated", np.pi / 10, 0.75),
]
PROFILES = ("base211", "base211_conn80")


def _metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _subject_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def _bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    left_subjects = set(left["subject_id"])
    right_subjects = set(right["subject_id"])
    if left_subjects != right_subjects:
        missing_left = sorted(right_subjects - left_subjects)
        missing_right = sorted(left_subjects - right_subjects)
        raise ValueError(
            "paired predictions have inconsistent subject_id sets "
            f"(missing_left={missing_left[:5]}, missing_right={missing_right[:5]})"
        )
    merged = left.merge(right, on="subject_id", suffixes=("_left", "_right"))
    if merged.empty:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_subjects": 0}
    if not np.array_equal(
        merged["true_label_left"].to_numpy(),
        merged["true_label_right"].to_numpy(),
    ):
        raise ValueError("paired predictions have inconsistent true_label values")
    y = merged["true_label_left"].to_numpy(dtype=int)
    pl = merged["pred_proba_left"].to_numpy(dtype=float)
    pr = merged["pred_proba_right"].to_numpy(dtype=float)
    point = _subject_auc(y, pl) - _subject_auc(y, pr)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        vals.append(_subject_auc(y[idx], pl[idx]) - _subject_auc(y[idx], pr[idx]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"delta": float(point), "ci_low": np.nan, "ci_high": np.nan,
                "n_subjects": int(len(y))}
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n_subjects": int(len(y)),
    }


def _bh_qvalues(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    valid = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if valid.any():
        order = np.argsort(p[valid])
        ranked = p[valid][order]
        vals = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        vals = np.minimum.accumulate(vals[::-1])[::-1]
        out = np.empty(len(ranked))
        out[order] = np.minimum(vals, 1.0)
        q[valid] = out
    return pd.Series(q, index=pvalues.index)


def _report_table(df: pd.DataFrame) -> str:
    """Render a dependency-free compact table for the markdown report."""
    return "```text\n" + df.to_string(index=False) + "\n```"


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"subject_id", "pred_proba", "true_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["subject_id", "pred_proba", "true_label"]].copy()
    if out["subject_id"].duplicated().any():
        dupes = out.loc[out["subject_id"].duplicated(), "subject_id"].tolist()
        raise ValueError(f"{path} has duplicate subject_id values: {dupes[:5]}")
    return out


def analyze(
    results_parent: Path,
    output_dir: Path,
    repeats: int,
    seed: int,
    config_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    paired = []
    for idx, mode, phase, coh in MATRIX:
        exp_root = results_parent / f"exp{idx}"
        cfg_path = config_dir / f"exp_wiener_phase_{idx}.yaml"
        cfg = load_config(cfg_path)
        condition = "wiener" if mode == "frequency" else "wiener_phasegated"
        for profile in PROFILES:
            metric_path = exp_root / "xgboost" / profile / condition / "test_metrics.json"
            pred_path = exp_root / "xgboost" / profile / condition / "test_predictions.csv"
            if not metric_path.exists() or not pred_path.exists():
                continue
            metrics = _metric(metric_path)
            rows.append({
                "experiment": f"exp{idx}", "profile": profile,
                "mode": mode, "phase_gate_threshold_rad": phase,
                "coherence_threshold": coh,
                "test_auroc": metrics.get("auroc"),
                "test_f1": metrics.get("f1"),
                "test_accuracy": metrics.get("accuracy"),
                "threshold": metrics.get("threshold"),
                "overlap_policy": cfg.get("wiener", {}).get("overlap_policy", "inherited"),
            })
            pred = _load_predictions(pred_path)
            raw_path = exp_root / "xgboost" / profile / "raw" / "test_predictions.csv"
            if raw_path.exists():
                raw = _load_predictions(raw_path)
                b = _bootstrap_delta(pred, raw, repeats, seed)
                paired.append({"experiment": f"exp{idx}", "profile": profile,
                                "comparison": "condition_minus_raw", **b})
    performance = pd.DataFrame(rows)
    performance.to_csv(output_dir / "performance.csv", index=False)
    pd.DataFrame(paired).to_csv(output_dir / "performance_paired.csv", index=False)

    profile_rows = []
    for idx, mode, _phase, _coh in MATRIX:
        condition = "wiener" if mode == "frequency" else "wiener_phasegated"
        exp_root = results_parent / f"exp{idx}" / "xgboost"
        left_path = exp_root / "base211_conn80" / condition / "test_predictions.csv"
        right_path = exp_root / "base211" / condition / "test_predictions.csv"
        if left_path.exists() and right_path.exists():
            profile_rows.append({
                "experiment": f"exp{idx}",
                **_bootstrap_delta(
                    _load_predictions(left_path),
                    _load_predictions(right_path),
                    repeats,
                    seed,
                ),
            })
    if profile_rows:
        pd.DataFrame(profile_rows).to_csv(output_dir / "feature_profile_comparison.csv", index=False)

    # Copy compact physical verification summaries into a grid-level table.
    for filename in ("fusion_summary.csv", "gate_summary.csv", "v1_summary.csv", "connectivity_summary.csv"):
        parts = []
        for idx, *_ in MATRIX:
            src = results_parent / f"exp{idx}" / "verification" / filename
            if src.exists():
                part = pd.read_csv(src)
                part.insert(0, "experiment", f"exp{idx}")
                parts.append(part)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(output_dir / filename.replace(".csv", "_grid.csv"), index=False)

    report = [
        "# Wiener phase/coherence grid summary", "",
        f"Bootstrap repeats: {repeats}; analysis seed: {seed}", "",
        "The grid keeps weighted overlap fusion fixed; it does not estimate a causal effect versus legacy last-write behavior.", "",
    ]
    if not performance.empty:
        report += ["## Performance", "", _report_table(performance), ""]
    if paired:
        report += ["## Paired AUROC bootstrap vs raw", "", _report_table(pd.DataFrame(paired)), ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/exp_wiener_phase")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parents[1] / "configs"),
        help="Directory containing exp_wiener_phase_1.yaml through _8.yaml",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir) if args.output_dir else root / "grid_summary"
    analyze(
        root,
        out,
        args.bootstrap_repeats,
        args.bootstrap_seed,
        Path(args.config_dir).resolve(),
    )
    print(f"Grid summary written to {out}")


if __name__ == "__main__":
    main()
