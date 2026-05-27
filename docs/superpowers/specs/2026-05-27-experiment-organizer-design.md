# Experiment Organizer — Design Spec

**Date:** 2026-05-27  
**Status:** Implemented  
**Script:** `scripts/07_organize_experiment.py`

---

## Problem

The EEG pipeline writes results for up to three preprocessing conditions
(raw / ica / wiener) into `results/xgboost/{condition}/`.  When the user
re-runs only one condition (e.g. `--condition wiener` in script 06), the
top-level `comparison_summary.csv` and `shap_comparison.png` become stale or
condition-incomplete.  There is also no persistent record of which config
produced a given set of results.

---

## Solution

`scripts/07_organize_experiment.py` — a standalone archiving script that:

1. Discovers which conditions currently have results.
2. Creates a timestamped folder under `experiments/`.
3. Copies per-condition metrics JSON files and SHAP plots.
4. **Re-derives** `comparison_summary.csv` from the discovered metrics.
5. **Re-generates** `shap_comparison.png` from the copied `shap_by_band.json`
   / `shap_by_channel.json` files.
6. Writes `experiment.json` (machine-readable) and `report.md` (human-readable).
7. Copies the config file used.

---

## Folder Layout

```
experiments/
└── YYYY-MM-DD_HHMMSS[_<name>]/
    ├── config.yaml                  — copy of --config
    ├── experiment.json              — config snapshot + all found metrics
    ├── report.md                    — human-readable summary
    ├── comparison_summary.csv       — re-derived from per-condition metrics
    ├── shap_comparison.png          — re-generated via plot_shap_comparison()
    ├── raw/                         — only if results/xgboost/raw/test_metrics.json exists
    │   ├── val_metrics.json
    │   ├── test_metrics.json
    │   ├── best_params.json
    │   ├── shap_by_band.json
    │   ├── shap_by_channel.json
    │   └── shap_summary.png
    ├── ica/   (same structure)
    └── wiener/  (same structure)
```

The `experiments/` root lives at the **project root** (sibling to `configs/`,
`results/`, `scripts/`).

---

## CLI

```
python scripts/07_organize_experiment.py [--config PATH] [--name LABEL] [--results-dir PATH]

  --config PATH        YAML config file (default: configs/default.yaml)
  --name LABEL         Optional label appended after the timestamp
  --results-dir PATH   Override results dir (default: paths.results_dir from config)
```

---

## `experiment.json` Schema

```json
{
  "name": "2026-05-27_143022_wiener-test",
  "timestamp": "2026-05-27T14:30:22",
  "config_path": "configs/default.yaml",
  "conditions_found": ["raw", "ica", "wiener"],
  "config_snapshot": {
    "target_sfreq": 125,
    "bandpass": [0.5, 40.0],
    "epoch_length_sec": 8.0,
    "artifact_threshold_uv": 200.0,
    "seizure_buffer_sec": 30.0,
    "split.train": 0.70,
    "split.val": 0.10,
    "split.test": 0.20,
    "split.random_seed": 42,
    "wiener.mode": "frequency",
    "wiener.nperseg": 250,
    "wiener.freq_resolution_hz": 0.5,
    "wiener.coherence_threshold": 0.15,
    "ica.n_components": 19,
    "ica.artifact_corr_threshold": 0.8,
    "ml.cv_folds": 5,
    "ml.early_stopping_rounds": 30,
    "ml.param_grid": { "...": "..." }
  },
  "results": {
    "raw":    { "val_metrics": {...}, "test_metrics": {...}, "best_params": {...} },
    "ica":    { "..." },
    "wiener": { "..." }
  }
}
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Self-contained script (no new `eeg_bg/` module) | Archiving is pure file I/O + formatting — no domain logic |
| Re-derive CSV instead of copying existing | Existing `comparison_summary.csv` may be stale after a partial condition re-run |
| Re-generate SHAP comparison figure | Same reason; uses existing `plot_shap_comparison()` from `eeg_bg.ml.shap_analysis` |
| Copy `shap_by_band.json` / `shap_by_channel.json` | Small JSON files needed to regenerate the comparison figure |
| No `.joblib` or `.npy` files copied | Large files; not needed for a documentation snapshot |
| `experiments/` at project root | Sibling to `results/` — clear separation between live results and archived runs |

---

## Reused Functions

- `eeg_bg.config.settings.load_config` — same config loading as all other scripts
- `eeg_bg.ml.shap_analysis.plot_shap_comparison` — SHAP comparison figure generation
