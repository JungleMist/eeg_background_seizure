"""Archive a pipeline run: copy per-condition results, re-derive the
cross-condition summary, regenerate the SHAP comparison figure, and write
both a human-readable ``report.md`` and a machine-readable ``experiment.json``
into a timestamped ``experiments/<run>/`` folder.

The script discovers whichever XGBoost conditions (raw / ica / wiener /
wiener_phasegated / wiener_zerophase) currently have
results in ``results/xgboost/`` and captures only those.  This means it is
safe to run even after a partial re-run (e.g. ``--condition wiener`` in
script 06) — the cross-condition summary and SHAP comparison figure are always
derived fresh from the conditions that are actually present, so they are never
stale.

Usage
-----
# Archive with an optional human label
python scripts/07_organize_experiment.py --name wiener-test

# Archive using a non-default config
python scripts/07_organize_experiment.py --config configs/local.yaml --name ablation

Output
------
experiments/YYYY-MM-DD_HHMMSS[_<name>]/
    config.yaml                 — copy of the config used
    experiment.json             — config snapshot + all found metrics
    report.md                   — human-readable summary
    comparison_summary.csv      — re-derived from per-condition metrics JSONs
    shap_comparison.png         — re-generated via plot_shap_comparison()
    xgboost/{base211,base211_conn80}/
        {raw,ica,wiener,wiener_phasegated,wiener_zerophase}/
        — present only when that condition has results
        val_metrics.json
        test_metrics.json
        best_params.json
        shap_by_band.json
        shap_by_channel.json
        shap_summary.png
        data_stats.json         — subject + epoch counts per split
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must precede pyplot

import pandas as pd
import yaml

from eeg_bg.config.settings import load_config
from eeg_bg.ml.shap_analysis import plot_shap_comparison


# ── Constants ─────────────────────────────────────────────────────────────────

XGB_CONDITIONS = [
    "raw", "ica", "wiener", "wiener_phasegated", "wiener_zerophase",
]
XGB_FEATURE_PROFILES = ["base211", "base211_conn80"]
CNN_CONDITIONS = ["raw", "ica", "wiener"]

# Files to copy from results/xgboost/{condition}/ into the experiment archive.
_CONDITION_FILES = [
    "val_metrics.json",
    "test_metrics.json",
    "best_params.json",
    "shap_by_band.json",      # needed to regenerate comparison figure
    "shap_by_channel.json",   # needed to regenerate comparison figure
    "shap_summary.png",
    "data_stats.json",        # subject + epoch counts per split
    "val_predictions.csv",
    "test_predictions.csv",
]

# Files to copy from results/cnn/{condition}/ into the experiment archive.
# best_model.pt is intentionally omitted — too large for archiving.
_CNN_CONDITION_FILES = [
    "val_metrics.json",
    "test_metrics.json",
    "best_params.json",
    "val_predictions.csv",
    "test_predictions.csv",
]

# Files to copy from results/verification/ into the experiment archive.
# Subject-level CSVs are omitted (too large); only summaries + metadata.
_VERIFICATION_FILES = [
    "verification_metadata.json",
    "v1_summary.csv",
    "v1_role_subject.csv",
    "v1_subject.csv",
    "gate_summary.csv",
    "gate_subject.csv",
    "fusion_summary.csv",
    "fusion_subject.csv",
    "skipped_pairs_summary.csv",
    "skipped_pairs_subject.csv",
    "connectivity_summary.csv",
    "connectivity_role_subject.csv",
    "connectivity_subject.csv",
]

# Config keys extracted into the snapshot (flat display name → nested path).
_SNAPSHOT_KEYS: list[tuple[str, list[str]]] = [
    ("target_sfreq",            ["preprocessing", "target_sfreq"]),
    ("bandpass",                ["preprocessing", "bandpass"]),
    ("epoch_length_sec",        ["preprocessing", "epoch_length_sec"]),
    ("artifact_threshold_uv",   ["preprocessing", "artifact_threshold_uv"]),
    ("seizure_buffer_sec",      ["preprocessing", "seizure_buffer_sec"]),
    ("split.train",             ["split", "train"]),
    ("split.val",               ["split", "val"]),
    ("split.test",              ["split", "test"]),
    ("split.random_seed",       ["split", "random_seed"]),
    ("wiener.mode",             ["wiener", "mode"]),
    ("wiener.nperseg",          ["wiener", "nperseg"]),
    ("wiener.freq_resolution_hz", ["wiener", "freq_resolution_hz"]),
    ("wiener.coherence_threshold", ["wiener", "coherence_threshold"]),
    ("wiener.overlap_policy", ["wiener", "overlap_policy"]),
    ("wiener.filter_magnitude_threshold", ["wiener", "filter_magnitude_threshold"]),
    ("wiener.freq_band", ["wiener", "freq_band"]),
    ("wiener.phase_gate_threshold_rad", ["wiener", "phase_gate_threshold_rad"]),
    ("ica.n_components",        ["ica", "n_components"]),
    ("ica.artifact_corr_threshold", ["ica", "artifact_corr_threshold"]),
    ("ml.cv_folds",             ["ml", "xgboost", "cv_folds"]),
    ("ml.early_stopping_rounds", ["ml", "xgboost", "early_stopping_rounds"]),
    ("ml.param_grid",           ["ml", "xgboost", "param_grid"]),
    ("ml.features.connectivity.nperseg", ["ml", "features", "connectivity", "nperseg"]),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_nested(cfg: dict, keys: list[str]):
    """Return cfg[keys[0]][keys[1]]... or None if any key is missing."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _extract_config_snapshot(cfg: dict) -> dict:
    """Flatten selected config keys into a display-friendly dict."""
    return {
        display: _get_nested(cfg, path)
        for display, path in _SNAPSHOT_KEYS
    }


def _discover_conditions(xgb_root: Path) -> list[str]:
    """Return XGBoost conditions that have a test_metrics.json (i.e. are complete)."""
    return [
        c for c in XGB_CONDITIONS
        if (xgb_root / c / "test_metrics.json").exists()
    ]


def _discover_profiles(xgb_root: Path) -> dict[str, list[str]]:
    """Discover profile-aware results, with legacy flat-tree fallback."""
    profile_results = {}
    for profile in XGB_FEATURE_PROFILES:
        found = _discover_conditions(xgb_root / profile)
        if found:
            profile_results[profile] = found
    if profile_results:
        return profile_results
    found = _discover_conditions(xgb_root)
    return {"base211": found} if found else {}


def _discover_cnn_conditions(cnn_root: Path) -> list[str]:
    """Return CNN conditions that have a test_metrics.json (i.e. are complete)."""
    return [
        c for c in CNN_CONDITIONS
        if (cnn_root / c / "test_metrics.json").exists()
    ]


def _copy_condition_files(
    xgb_root: Path, exp_dir: Path, profile: str, condition: str
) -> None:
    """Copy per-condition XGBoost result files into the experiment archive."""
    src_dir = xgb_root / profile / condition
    if not src_dir.exists() and profile == "base211":
        src_dir = xgb_root / condition  # legacy flat layout
    dst_dir = exp_dir / "xgboost" / profile / condition
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in _CONDITION_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def _copy_cnn_condition_files(
    cnn_root: Path, exp_dir: Path, condition: str
) -> None:
    """Copy per-condition CNN result files into the experiment archive."""
    src_dir = cnn_root / condition
    dst_dir = exp_dir / "cnn" / condition
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in _CNN_CONDITION_FILES:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def _copy_verification_files(
    verif_root: Path, exp_dir: Path
) -> bool:
    """Copy verification summary files into the experiment archive.

    Returns True if the verification metadata file existed (i.e. script 04
    was run), False otherwise.
    """
    meta = verif_root / "verification_metadata.json"
    if not meta.exists():
        return False
    dst_dir = exp_dir / "verification"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in _VERIFICATION_FILES:
        src = verif_root / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
    return True


def _derive_comparison_csv(
    exp_dir: Path, profile: str, found: list[str]
) -> None:
    """Re-build comparison_summary.csv from the archived per-condition metrics."""
    rows = []
    for cond in found:
        row: dict = {"condition": cond}
        for prefix in ("val", "test"):
            metrics_file = exp_dir / "xgboost" / profile / cond / f"{prefix}_metrics.json"
            if metrics_file.exists():
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v
        stats_file = exp_dir / "xgboost" / profile / cond / "data_stats.json"
        if stats_file.exists():
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
            for split in ("train", "val", "test"):
                s = stats.get(split, {})
                row[f"{split}_n_subjects"] = s.get("n_subjects", "")
                row[f"{split}_n_epochs"]   = s.get("n_epochs", "")
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            exp_dir / "xgboost" / profile / "comparison_summary.csv", index=False
        )


def _regenerate_shap_comparison(
    exp_dir: Path, profile: str, found: list[str], dpi: int
) -> bool:
    """Re-generate shap_comparison.png from archived JSON data.

    Returns True if the figure was written, False if no SHAP data was found.
    """
    shap_results: dict[str, dict] = {}
    for cond in found:
        band_file = exp_dir / "xgboost" / profile / cond / "shap_by_band.json"
        ch_file   = exp_dir / "xgboost" / profile / cond / "shap_by_channel.json"
        if band_file.exists() and ch_file.exists():
            shap_results[cond] = {
                "shap_by_band":    json.loads(band_file.read_text(encoding="utf-8")),
                "shap_by_channel": json.loads(ch_file.read_text(encoding="utf-8")),
            }
    if not shap_results:
        return False
    plot_shap_comparison(
        shap_results,
        output_path=exp_dir / "xgboost" / profile / "shap_comparison.png",
        dpi=dpi,
    )
    return True


def _write_experiment_json(
    exp_dir: Path,
    folder_name: str,
    timestamp: str,
    config_path: str,
    found_by_profile: dict[str, list[str]],
    found_cnn: list[str],
    snapshot: dict,
    has_verification: bool = False,
) -> None:
    """Write experiment.json."""
    results_by_profile: dict[str, dict] = {}
    for profile, found in found_by_profile.items():
        results: dict[str, dict] = {}
        for cond in found:
            entry: dict = {}
            for key in ("val_metrics", "test_metrics", "best_params", "data_stats"):
                fpath = exp_dir / "xgboost" / profile / cond / f"{key}.json"
                if fpath.exists():
                    entry[key] = json.loads(fpath.read_text(encoding="utf-8"))
            results[cond] = entry
        results_by_profile[profile] = results

    results_cnn: dict[str, dict] = {}
    for cond in found_cnn:
        entry = {}
        for key in ("val_metrics", "test_metrics", "best_params"):
            fpath = exp_dir / "cnn" / cond / f"{key}.json"
            if fpath.exists():
                entry[key] = json.loads(fpath.read_text(encoding="utf-8"))
        results_cnn[cond] = entry

    payload = {
        "name":                 folder_name,
        "timestamp":            timestamp,
        "config_path":          config_path,
        "conditions_found":     sorted({c for xs in found_by_profile.values() for c in xs}),
        "conditions_found_by_profile": found_by_profile,
        "cnn_conditions_found": found_cnn,
        "config_snapshot":      snapshot,
        "has_verification":     has_verification,
        "results_xgboost":      results_by_profile,
        # Keep the legacy key for consumers that expect a single profile.
        "results":              results_by_profile.get("base211", {}),
        "results_cnn":          results_cnn,
    }
    (exp_dir / "experiment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_report_md(
    exp_dir: Path,
    folder_name: str,
    timestamp: str,
    config_path: str,
    found_by_profile: dict[str, list[str]],
    found_cnn: list[str],
    snapshot: dict,
    has_shap_comparison: dict[str, bool],
    has_verification: bool = False,
) -> None:
    """Write report.md."""
    lines: list[str] = []

    # Header
    lines += [
        f"# Experiment: {folder_name}",
        "",
        f"**Date:** {timestamp.replace('T', ' ')}  ",
        f"**Config:** {config_path}  ",
        f"**Profiles found:** {', '.join(found_by_profile) if found_by_profile else '*(none)*'}",
        "",
    ]

    # Configuration snapshot table
    lines += [
        "## Configuration Snapshot",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for key, val in snapshot.items():
        val_str = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        lines.append(f"| `{key}` | {val_str} |")
    lines.append("")

    def _read_metric(base: Path, cond: str, prefix: str, key: str) -> str:
        fpath = base / cond / f"{prefix}_metrics.json"
        if not fpath.exists():
            return "—"
        d = json.loads(fpath.read_text(encoding="utf-8"))
        v = d.get(key)
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    # XGBoost results summary table
    for profile, found in found_by_profile.items():
        if not found:
            continue
        profile_base = exp_dir / "xgboost" / profile
        lines += [
            f"## XGBoost Results Summary — `{profile}`",
            "",
            "| Condition | Val AUROC | Val F1 | Val Acc | Test AUROC | Test F1 | Test Acc |",
            "|---|---|---|---|---|---|---|",
        ]
        for cond in found:
            lines.append(
                f"| {cond} "
                f"| {_read_metric(profile_base, cond, 'val', 'auroc')} "
                f"| {_read_metric(profile_base, cond, 'val', 'f1')} "
                f"| {_read_metric(profile_base, cond, 'val', 'accuracy')} "
                f"| {_read_metric(profile_base, cond, 'test', 'auroc')} "
                f"| {_read_metric(profile_base, cond, 'test', 'f1')} "
                f"| {_read_metric(profile_base, cond, 'test', 'accuracy')} |"
            )
        lines.append("")

    # CNN results summary table
    if found_cnn:
        cnn_base = exp_dir / "cnn"
        lines += [
            "## CNN Results Summary",
            "",
            "| Condition | Val AUROC | Val F1 | Val Acc | Test AUROC | Test F1 | Test Acc |",
            "|---|---|---|---|---|---|---|",
        ]
        for cond in found_cnn:
            lines.append(
                f"| {cond} "
                f"| {_read_metric(cnn_base, cond, 'val', 'auroc')} "
                f"| {_read_metric(cnn_base, cond, 'val', 'f1')} "
                f"| {_read_metric(cnn_base, cond, 'val', 'accuracy')} "
                f"| {_read_metric(cnn_base, cond, 'test', 'auroc')} "
                f"| {_read_metric(cnn_base, cond, 'test', 'f1')} "
                f"| {_read_metric(cnn_base, cond, 'test', 'accuracy')} |"
            )
        lines.append("")

    # Dataset statistics (read from first available condition — splits are shared)
    first_profile = next(iter(found_by_profile), None)
    first_stats_cond = next(
        (c for c in found_by_profile.get(first_profile, [])
         if (exp_dir / "xgboost" / first_profile / c / "data_stats.json").exists()), None
    ) if first_profile else None
    if first_stats_cond and first_profile:
        stats = json.loads(
            (exp_dir / "xgboost" / first_profile / first_stats_cond / "data_stats.json")
            .read_text(encoding="utf-8")
        )
        lines += [
            "## Dataset Statistics",
            "",
            "| Split | Subjects | Epochs | Epilepsy Subj | Control Subj |"
            " Epilepsy Ep | Control Ep |",
            "|---|---|---|---|---|---|---|",
        ]
        for split in ("train", "val", "test"):
            s = stats.get(split, {})
            lines.append(
                f"| {split} "
                f"| {s.get('n_subjects', '—')} "
                f"| {s.get('n_epochs', '—')} "
                f"| {s.get('n_subjects_epilepsy', '—')} "
                f"| {s.get('n_subjects_control', '—')} "
                f"| {s.get('n_epochs_epilepsy', '—')} "
                f"| {s.get('n_epochs_control', '—')} |"
            )
        lines.append("")

    lines += ["## Feature Profiles", "", ", ".join(f"`{p}`" for p in found_by_profile), ""]

    # Verification summary
    if has_verification:
        verif_meta = exp_dir / "verification" / "verification_metadata.json"
        if verif_meta.exists():
            vm = json.loads(verif_meta.read_text(encoding="utf-8"))
            lines += [
                "## Verification",
                "",
                f"- **Mode:** {vm.get('mode', '—')}",
                f"- **Checks:** {', '.join(vm.get('checks', []))}",
                f"- **Subjects:** {vm.get('n_subjects', '—')}",
                f"- **Recordings:** {vm.get('n_recordings', '—')}",
                "",
                "See `verification/` for per-band V1 coherence, gate diagnostics,",
                "and lagged connectivity (imaginary coherence, wPLI) summaries.",
                "",
            ]

    # SHAP comparison figure
    for profile, has_shap in has_shap_comparison.items():
        if has_shap:
            lines += [
                f"## SHAP Comparison — `{profile}`", "",
                f"![SHAP Comparison](xgboost/{profile}/shap_comparison.png)", "",
            ]

    # Per-condition SHAP summaries
    for profile, found in found_by_profile.items():
        conds_with_shap = [
            c for c in found
            if (exp_dir / "xgboost" / profile / c / "shap_summary.png").exists()
        ]
        if conds_with_shap:
            lines += [f"## Per-Condition SHAP Summaries — `{profile}`", ""]
            for cond in conds_with_shap:
                label = cond.capitalize()
                lines += [
                    f"### {label}", "",
                    f"![{label} SHAP](xgboost/{profile}/{cond}/shap_summary.png)", "",
                ]

    (exp_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main(config_path: str, name: str | None, results_dir_override: str | None) -> None:
    cfg         = load_config(config_path)
    results_dir = Path(results_dir_override) if results_dir_override \
                  else Path(cfg["paths"]["results_dir"])
    xgb_root    = results_dir / "xgboost"

    # Derive project root from config path (same convention as load_config)
    project_root = Path(config_path).resolve().parent.parent
    exp_root     = project_root / "experiments"

    # ── Discover conditions ───────────────────────────────────────────────────
    found_by_profile = _discover_profiles(xgb_root)
    if not found_by_profile:
        print(
            "[WARNING] No completed condition results found in "
            f"{xgb_root}. Run script 06 first.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Profiles found: " + ", ".join(
        f"{p}({','.join(cs)})" for p, cs in found_by_profile.items()
    ))

    # ── Create experiment folder ──────────────────────────────────────────────
    now          = datetime.now()
    ts_fs        = now.strftime("%Y-%m-%d_%H%M%S")      # filesystem-safe
    ts_iso       = now.strftime("%Y-%m-%dT%H:%M:%S")    # ISO 8601 for JSON
    config_stem  = Path(config_path).stem               # e.g. "default", "local"
    folder_name  = f"{ts_fs}_{config_stem}_{name}" if name else f"{ts_fs}_{config_stem}"
    exp_dir      = exp_root / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating experiment archive: {exp_dir}")

    # ── Copy XGBoost per-condition files ─────────────────────────────────────
    for profile, found in found_by_profile.items():
        for cond in found:
            print(f"  Copying xgboost/{profile}/{cond}/...")
            _copy_condition_files(xgb_root, exp_dir, profile, cond)

    # ── Copy CNN per-condition files ──────────────────────────────────────────
    cnn_root  = results_dir / "cnn"
    found_cnn = _discover_cnn_conditions(cnn_root)
    if found_cnn:
        print(f"CNN conditions found: {', '.join(found_cnn)}")
        for cond in found_cnn:
            print(f"  Copying cnn/{cond}/...")
            _copy_cnn_condition_files(cnn_root, exp_dir, cond)
    else:
        print("[INFO] No CNN results found — skipping CNN archiving.")

    # ── Copy verification files ──────────────────────────────────────────────
    verif_root = results_dir / "verification"
    has_verification = _copy_verification_files(verif_root, exp_dir)
    if has_verification:
        print("  Verification files archived.")
    else:
        print("[INFO] No verification results found — skipping verification archiving.")

    # ── Feature profile discovery ────────────────────────────────────────────
    # ── Re-derive profile-specific cross-condition summaries ──────────────────
    print("Re-deriving profile-specific comparison_summary.csv files...")
    for profile, found in found_by_profile.items():
        _derive_comparison_csv(exp_dir, profile, found)

    # ── Re-generate SHAP comparison figure ────────────────────────────────────
    print("Re-generating profile-specific shap_comparison.png files...")
    dpi = int(cfg["ml"]["shap"]["dpi"])
    has_shap: dict[str, bool] = {}
    for profile, found in found_by_profile.items():
        has_shap[profile] = _regenerate_shap_comparison(exp_dir, profile, found, dpi)
        if not has_shap[profile]:
            print(f"  [INFO] No SHAP JSON found for {profile}")

    # ── Extract config snapshot ───────────────────────────────────────────────
    snapshot = _extract_config_snapshot(cfg)

    # ── Write experiment.json ─────────────────────────────────────────────────
    print("Writing experiment.json...")
    _write_experiment_json(
        exp_dir, folder_name, ts_iso, config_path, found_by_profile, found_cnn,
        snapshot, has_verification=has_verification,
    )

    # ── Write report.md ───────────────────────────────────────────────────────
    print("Writing report.md...")
    _write_report_md(
        exp_dir, folder_name, ts_iso, config_path, found_by_profile, found_cnn,
        snapshot, has_shap, has_verification=has_verification,
    )

    # ── Write resolved config ─────────────────────────────────────────────────
    # Experiment configs may inherit from default.yaml; archive the fully
    # resolved config so the experiment folder is self-contained.
    (exp_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    shutil.copy2(exp_dir / "config.yaml", exp_dir / "config_resolved.yaml")

    print(f"\nDone. Experiment archived to:\n  {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Archive pipeline results into a timestamped experiments/ folder"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional label appended to the experiment folder name "
             "(e.g. 'wiener-test' → 2026-05-27_143022_wiener-test)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        dest="results_dir",
        help="Override the results directory from config "
             "(default: paths.results_dir from --config)",
    )
    args = parser.parse_args()
    main(args.config, args.name, args.results_dir)
