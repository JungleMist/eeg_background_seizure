# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python via **conda env `eeg_pipeline`** (`C:\ProgramData\anaconda3\envs\eeg_pipeline`). Use `conda run -n eeg_pipeline` or activate it before running commands. The conda base env has NumPy 2.x which is incompatible with the pinned scipy/matplotlib.
- Install the package in development mode: `pip install -e .`
- Core dependencies: numpy 2.4.6, scipy 1.17.1, mne 1.11.0, scikit-learn 1.8.0, joblib 1.5.3, pandas 2.3.3, matplotlib 3.10.8, pyyaml 6.0.3, pytest 9.0.2, tqdm 4.67.3, xgboost 3.2.0, shap 0.51.0.
- **Note**: `np.trapezoid` is used (not `np.trapz`, removed in NumPy 2.0).

## Common Commands

```bash
# Install
conda run -n eeg_pipeline pip install -e .

# Run all unit tests (no real EDF data required)
conda run -n eeg_pipeline python -m pytest tests/ -v

# Skip integration tests (default — no real EDF needed)
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration"

# Run a single test module
conda run -n eeg_pipeline python -m pytest tests/test_decomposition/test_wiener.py -v

# Run a single test by name
conda run -n eeg_pipeline python -m pytest tests/test_decomposition/test_wiener.py::test_decompose_epoch_reduces_coherence -v
```

## Pipeline Scripts (run in order)

All scripts accept `--config configs/default.yaml` (default) and `--force` to re-run even if cached output exists.

```bash
# 01 — Extract background epochs from EDF files → cache/epochs/
conda run -n eeg_pipeline python scripts/01_extract_epochs.py [--force] [--workers N]

# 02 — Wiener decomposition on cached epochs → cache/wiener_frequency/
conda run -n eeg_pipeline python scripts/02_run_wiener.py [--mode frequency|scalar] [--force] [--workers N]

# 03 — ICA ablation → cache/ica/
conda run -n eeg_pipeline python scripts/03_run_ica.py [--force] [--workers N]

# 04 — Verification experiments V1/V2/V3 → results/verification/*.csv
conda run -n eeg_pipeline python scripts/04_run_verification.py

# Scripts 02, 03, and 04 all read only from cache/epochs/ — they can run in parallel after 01.

# 05 — Per-subject waveform and PSD figures → results/figures/{subject_id}/
conda run -n eeg_pipeline python scripts/05_run_visualization.py [--n-subjects N] [--epoch-idx I] [--channels "FP1,FP2,T3"]

# 06 — XGBoost + SHAP for all three conditions → results/xgboost/
conda run -n eeg_pipeline python scripts/06_train_xgboost.py [--condition raw|ica|wiener|all] [--force] [--workers N]
```

### Cache invalidation tiers

Changing a `configs/default.yaml` key requires re-running all scripts at or after its tier with `--force`:

| Tier | Scripts to re-run | Config sections that trigger it |
|------|-------------------|----------------------------------|
| 1 | `01+` (all scripts) | `paths`, `dataset`, `split`, `preprocessing`, `channels.standard_19` |
| 2 | `02+` | `channels.channel_groups`, `channels.passthrough`, `wiener` |
| 3 | `03+` | `ica` |
| 4 | `06` only | `ml` |

Scripts 04 and 05 produce no cache and always re-run from existing caches — changing `verification` or `visualization` keys requires no `--force`.

### Running with a local config

To experiment without modifying the tracked `default.yaml`:

```bash
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml freely
conda run -n eeg_pipeline python scripts/01_extract_epochs.py --config configs/local.yaml
```

Add `configs/local*.yaml` to `.gitignore` to keep it untracked. **The config file must live directly inside `configs/`** (one level below the project root) — `load_config()` derives the project root as `config_path.parent.parent`, so placing it elsewhere breaks relative `cache_dir`/`results_dir` resolution. Use absolute paths in the config as an alternative.

## Architecture

The package is `eeg_bg/`, pip-installed from `setup.py`. Configuration is loaded from `configs/default.yaml` via `eeg_bg.config.settings.load_config()`, which resolves relative `cache_dir` / `results_dir` paths against the project root.

**Data flow (unidirectional):**
```
EDF files (TUEP v3.1.0, D:/EEGdata/TUEP/v3.1.0)
  → io/edf_reader.py      — loads EDF, normalises channel names (strips "EEG " prefix and "-REF" suffix), resamples to 125 Hz, converts V→µV
  → io/annotation.py      — parses csv_bi files; excludes ±30 s seizure buffers from background intervals
  → preprocessing/epoch.py — slices 8 s epochs, rejects epochs exceeding 200 µV
  → cache/epochs/          — checkpoint 1 (.npz via io/cache.py)
  → decomposition/wiener.py — vector Wiener decomposition (core method)
  → decomposition/ica.py    — FastICA ablation baseline
  → cache/wiener_frequency|ica/ — checkpoint 2
  → verification/           — V1 coherence, V2 transitivity, V3 frequency variation
  → visualization/          — matplotlib figures returned as plt.Figure (never call plt.show())
  → features/extraction.py  — extract_epoch_features() → 171-dim vector per epoch
  → ml/xgb_pipeline.py      — GridSearchCV + early stopping; subject-level aggregation
  → ml/shap_analysis.py     — TreeExplainer SHAP; band/channel aggregation; comparison plot
  → results/xgboost/        — model.joblib, metrics JSON, SHAP plots
```

### Key modules

| Module | Responsibility |
|--------|---------------|
| `eeg_bg/config/settings.py` | `load_config(path)` — loads YAML and resolves relative `cache_dir`/`results_dir` to absolute paths against the project root. |
| `eeg_bg/io/edf_reader.py` | `load_edf(path, cfg)` — loads EDF via MNE, strips `"EEG "` prefix and `"-REF"` suffix from channel names, resamples to 125 Hz, converts V→µV. |
| `eeg_bg/io/annotation.py` | Parses `csv_bi` seizure annotation files; `extract_bckg_intervals` returns non-seizure segments with ±`seizure_buffer_sec` guard zones excluded. |
| `eeg_bg/preprocessing/epoch.py` | `slice_epochs` — cuts fixed-length (8 s) epochs from a continuous recording; rejects epochs where any channel exceeds `artifact_threshold_uv` (200 µV). |
| `eeg_bg/decomposition/wiener.py` | Core Wiener filter: `estimate_cross_psd` → `compute_wiener_filter` → `apply_wiener_filter` → `decompose_epoch` / `decompose_subject`. Returns `WienerResult` (raw, specific, coherent, filters dict, skipped_pairs). |
| `eeg_bg/decomposition/wiener_scalar.py` | Fixed-scalar ablation baseline for comparison. |
| `eeg_bg/decomposition/ica.py` | `fit_ica` uses FP1/FP2 as artifact-reference channels; `apply_ica` removes components correlated above `artifact_corr_threshold`. Stores cleaned signal as `specific` key (mirrors Wiener). |
| `eeg_bg/verification/coherence.py` | V1: pairwise coherence matrix before/after decomposition (`run_v1`). Uses `freq_resolution_hz` (not `nperseg`) to set estimation window so coherence averaging is valid. |
| `eeg_bg/verification/transitivity.py` | V2: single-point-source transitivity constraint; V3: frequency variation of \|h(f)\| across target band. |
| `eeg_bg/visualization/psd_plots.py` | `plot_psd_comparison`: PSD overlay (raw / Wiener-specific / ICA-cleaned) for target channels. Channels default to `cfg["visualization"]["psd_target_channels"]` (FP1, FP2). Uses boxcar Welch consistent with decomposition. |
| `eeg_bg/visualization/waveform_plots.py` | `plot_multichannel_comparison`: stacked all-channel waveform with up to 3 panels (raw / Wiener / ICA). |
| `eeg_bg/preprocessing/reference.py` | `detect_reference` infers AR/LE from montage dir name; `filter_by_reference` subsets the subject index to one scheme. |
| `eeg_bg/io/dataset.py` | Traverses TUEP directory tree → subject index DataFrame; `assign_splits` splits by subject (not recording). |
| `eeg_bg/io/cache.py` | `load_or_compute` wraps any `compute_fn` with `.npz` on-disk caching; cache key = SHA-256 of `edf_path|start_sec|sfreq|bandpass`. |
| `eeg_bg/features/extraction.py` | `extract_epoch_features(epoch, ch_names, sfreq)` → `(171,)` vector; `build_dataset(cache_root, condition, split, ...)` → `(X, y, subject_ids)`. Feature cache in `cache/features/{condition}_{split}.npz`. |
| `eeg_bg/ml/xgb_pipeline.py` | `train_xgboost`: Phase 1 GridSearchCV, Phase 2 early-stopping refit. `subject_level_predict`: epoch-level proba → subject-mean. `evaluate_subject_level`: AUROC/F1/Acc. |
| `eeg_bg/ml/shap_analysis.py` | `compute_shap_values` (TreeExplainer), `aggregate_shap_by_band/channel`, `plot_shap_summary` (beeswarm), `plot_shap_comparison` (2×3 cross-condition publication figure). |
| `eeg_bg/visualization/coherence_plots.py` | `plot_coherence_matrix` (pre/post heatmap side-by-side), `plot_coherence_reduction` (boxplot by pair or subject), `plot_signal_decomposition` (raw / coherent / specific waveform panels for one channel). |
| `eeg_bg/visualization/filter_plots.py` | `plot_wiener_filter_response` (amplitude + phase for one pair), `plot_all_pairs_response` (grid across all pairs in a `WienerResult`). |
| `eeg_bg/visualization/verification_plots.py` | `plot_v2_transitivity`, `plot_v3_frequency_variation`, `plot_ica_vs_wiener_coherence` (3-panel raw/ICA/Wiener coherence matrix). |
| `eeg_bg/features/band_power.py` | `relative_band_power(signal, sfreq, band)` → scalar; `BANDS` dict mapping name→(low, high) Hz for delta/theta/alpha/beta/gamma. |
| `eeg_bg/features/hjorth.py` | `hjorth_parameters(signal)` → `(activity, mobility, complexity)` triple. |
| `eeg_bg/features/spectral_entropy.py` | `spectral_entropy(signal, sfreq)` → scalar normalised Shannon entropy of PSD. |
| `eeg_bg/features/extraction.py` (constants) | `FEATURE_NAMES` — public list of 171 strings (`"{ch}_{suffix}"`) built at import time; must stay stable since downstream `.npy` SHAP arrays are indexed by position. `_CONDITION_TO_SUBDIR` maps `"wiener"→"wiener_frequency"` etc. |

### Channel groups (G1–G6)

The Wiener filter operates on **movement-artifact conduction pathways**, not bilateral pairs. Groups are defined in `configs/default.yaml` under `channels.channel_groups`:
- G1 `[FP1, FP2]` — symmetric facial (frontalis)
- G2 `[F7, T3]` — left SCM
- G3 `[T3, T5, O1]` — left posterior neck (3-channel chain)
- G4 `[O1, O2]` — bilateral occipitalis
- G5 `[F8, T4]` — right SCM
- G6 `[T4, T6, O2]` — right posterior neck (3-channel chain)

Passthrough channels (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz`) are never filtered.

### Wiener filter implementation details

- PSD estimated with **boxcar window** so that when `nperseg == n_times` the filter can be applied exactly via rfft without windowing mismatch.
- When `nperseg < n_times`, filter coefficients are linearly interpolated to the full rfft grid; `specific + coherent == raw` is guaranteed by construction regardless.
- A coherence gate (max pairwise coherence across all pairs in the group, over the target frequency band) skips groups below `coherence_threshold` (default 0.15).
- `nperseg` in `wiener:` is for filter estimation; V1 coherence uses `freq_resolution_hz` (125 / 0.5 = 250 samples = 4 segments per 1000-sample epoch) to avoid trivial coherence=1.

### Feature vector layout

`extract_epoch_features` produces a 171-dim vector: **19 channels × 9 features** each.  
Inner order per channel: `delta_power, theta_power, alpha_power, beta_power, gamma_power, hjorth_activity, hjorth_mobility, hjorth_complexity, spectral_entropy`.  
Channels iterate in the canonical 19-channel order from `configs/default.yaml`. Missing channels fill with zeros so the vector length is always 171.

### Label encoding

`label = 0` → epilepsy (`00_` prefix in cache dirs); `label = 1` → control (`01_` prefix). This is the TUEP convention preserved throughout the pipeline.

### Cache directory layout

```
cache/
├── epochs/{label_prefix}_{subject_id}/{sha256_key}.npz   — keys: epochs, ch_names, label, subject_id, split
├── wiener_frequency/ (same tree)                          — keys: specific, coherent, label, subject_id, split
├── ica/ (same tree)                                       — keys: specific, n_artifacts_removed, label, subject_id, split
└── features/{condition}_{split}.npz                       — keys: X, y, subject_ids
```
The `wiener` condition in `build_dataset` / `--condition` maps to `cache/wiener_frequency/` (not `cache/wiener/`).

### Output directory structure

```
results/
├── figures/{subject_id}/
│   ├── waveform_comparison.png       — all-channel stacked waveform (raw | Wiener | ICA)
│   └── psd_comparison.png            — PSD overlay for psd_target_channels
├── verification/
│   ├── v1_coherence.csv
│   ├── v2_transitivity.csv
│   └── v3_frequency_variation.csv
└── xgboost/
    ├── {raw,ica,wiener}/
    │   ├── model.joblib              — fitted XGBClassifier
    │   ├── scaler.joblib             — StandardScaler (fit on train)
    │   ├── best_params.json          — GridSearchCV best hyperparameters
    │   ├── val_metrics.json / test_metrics.json  — {auroc, f1, accuracy}
    │   ├── val_predictions.csv / test_predictions.csv  — subject_id, pred_proba, true_label
    │   ├── shap_values_test.npy      — (n_test_epochs, 171) raw SHAP values
    │   ├── shap_summary.png          — beeswarm plot (top 20 features)
    │   ├── shap_by_band.json         — mean |SHAP| per feature-type group
    │   └── shap_by_channel.json      — mean |SHAP| per EEG channel
    ├── comparison_summary.csv        — 3 conditions × {val_auroc, test_auroc, f1, acc}
    └── shap_comparison.png           — 2×3 publication comparison figure
```

### Matplotlib backend

Any script that saves figures must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot`. Scripts 05 and 06 already do this. Visualization functions in `eeg_bg/visualization/` return `plt.Figure` objects and never call `plt.show()`.

## Tests

All tests run without real EDF data. There are three conftest scopes:

- **`tests/conftest.py`** (root): `synthetic_epoch` (19-ch 1000-sample epoch with a single point source, SNR ≈ 50:1), `synthetic_epochs_batch` (batch of 5), `tmp_cache_dir`, `cfg` (deep-copied `BASE_CFG` dict).
- **`tests/test_features/conftest.py`**: `ch_names_19`, `sfreq`, `synthetic_epoch` (simple random — independent of root fixture), `pure_sine_signal` (10 Hz sine), `constant_signal`.
- **`tests/test_ml/conftest.py`**: `tiny_xgb_model` (10-feature, 5-estimator, session-scoped), `full_feature_xgb_model` (171-feature, session-scoped).

Test files under `tests/test_visualization/` must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot` (same rule as scripts).

Integration tests (requiring real TUEP EDF files) should be marked `@pytest.mark.integration`.

`check_fixtures.py` (project root) is a standalone debug script that reconstructs fixture arrays and prints their shapes — useful for verifying conftest parity outside pytest.

## Non-obvious constraints and gotchas

- **Paired config keys**: `dataset.reference_scheme` and `dataset.montage_dir` must always change together (`"ar"` ↔ `"01_tcp_ar"`, `"le"` ↔ `"02_tcp_le"`). Mismatching them produces zero EDF files for one class. Similarly, `wiener.freq_band` must be a subset of `preprocessing.bandpass`, and `wiener.nperseg` must be ≤ `target_sfreq × epoch_length_sec`.
- **`bandpass_filter()` in `epoch.py` is not used in the pipeline**: `load_edf` in `edf_reader.py` applies MNE's `raw.filter()` on the continuous signal before resampling. The standalone `bandpass_filter()` function in `epoch.py` (5th-order `sosfiltfilt`) is available for ad-hoc use but is never called by any script.
- **XGBoost `n_estimators` in `param_grid` is ignored during Phase 1**: `xgb_pipeline.py` overrides it to 500 for the grid search and uses early stopping in Phase 2 to find the final tree count. The entry in `configs/default.yaml` is documentation only.
- **`device="cuda"` automatically sets `n_jobs=1`**: `xgb_pipeline.py` detects the CUDA device setting and overrides GridSearchCV's `n_jobs` to avoid CUDA context conflicts across parallel workers. No manual change needed.
- **ICA fits on a 1 Hz high-pass copy but applies to 0.5 Hz data**: `fit_ica()` creates a temporary high-pass-filtered copy for FastICA convergence (MNE best practice), then applies the fitted mixing matrix to the original 0.5 Hz bandpass epochs. The `specific` output in the ICA cache is in the original 0.5 Hz bandpass domain. `max_iter` (default 1000) is read from `ica.max_iter` in the config and passed directly to the MNE ICA constructor — the scikit-learn default of 200 is too low for 19-channel EEG and causes frequent `ConvergenceWarning`.
- **`FEATURE_NAMES` must stay positionally stable**: SHAP `.npy` arrays `(n_test_epochs, 171)` are indexed by position against `FEATURE_NAMES`. Any reordering or insertion of channels in `configs/default.yaml` `standard_19` invalidates saved SHAP arrays.
- **`reference_scheme` filter is applied before any EDF is loaded**: Script 01 only processes recordings under the `montage_dir` subdirectory (default `01_tcp_ar`). Linked-ears (`tcp_le`) recordings are silently excluded.
- **Scripts 01–03 and 06 use `ProcessPoolExecutor`** (default `os.cpu_count()` workers). On Windows, multiprocessing uses `spawn`, so each worker re-imports the full module graph at startup — worker startup overhead is higher than on Linux. Use `--workers N` to cap concurrency on memory-constrained machines or when other processes need CPU. Script 05 is sequential (no `--workers` flag). Script 06 runs conditions sequentially but parallelises feature extraction within each condition.
- **Cache key composition for script 01**: the SHA-256 key is derived from `edf_path`, `target_sfreq`, and `bandpass` only. Changing `epoch_length_sec`, `artifact_threshold_uv`, or `seizure_buffer_sec` does NOT generate a new key — the existing `.npz` is silently reused unless you pass `--force`. Scripts 02 and 03 have no key-based invalidation at all; any config change to `wiener.*` or `ica.*` requires `--force` for those scripts.
- **`--force` does not cascade and does not clean up orphaned files**: running `01_extract_epochs.py --force` does not force-rerun scripts 02–06; each script must be passed `--force` independently. Old `.npz` files at old key paths are orphaned on disk (not deleted). For script 06, `--force` only bypasses the feature-extraction cache (`cache/features/`); model training, SHAP computation, and all output files always regenerate regardless.

## Reference Documentation

- **`docs/Developer_qa.md`** — Detailed Q&A: epoch validity criteria (4 criteria with decision flow), `.npz` cache file schemas (exact keys/shapes/dtypes per cache family), feature extraction internals (band definitions, Hjorth formulae, spectral entropy formula, per-condition signal mapping).
- **`eeg_bg/README.md`** — Full package API reference: every public function signature, return type, parameter table, and usage example.
