# AGENTS.md

This file provides guidance to ZCode / Claude Code agents working with this repository.
Last updated: 2026-07-07.

## Environment

- Python via **conda env `eeg_pipeline`**, Python 3.11. Use `conda run -n eeg_pipeline` to run commands. The conda base env has NumPy 2.x which is incompatible with the pinned scipy/matplotlib.
- **macOS**: conda env is at `/Users/jsu/miniconda3/envs/eeg_pipeline` (miniconda3). On Linux/AutoDL it may be under `/root/miniconda3/`.
- To recreate: `conda env create -f environment.yaml` (Windows export — `C:\` paths baked in, UTF-16). On macOS, maintain a local gitignored `environment_macos.yaml` instead.
- Install the package in development mode: `pip install -e .`
- Core dependencies: numpy 2.4.6, scipy 1.17.1, mne 1.11.0, scikit-learn 1.8.0, joblib 1.5.3, pandas 2.3.3, matplotlib 3.10.8, pyyaml 6.0.3, pytest 9.0.2, tqdm 4.67.3, xgboost 3.2.0, shap 0.51.0, pywavelets ≥1.4 (DWT features), torch (CNN pipeline, script 08 — CPU or CUDA build).
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
conda run -n eeg_pipeline python scripts/02_run_wiener.py [--mode frequency|phasegated|scalar|zerophase] [--force] [--workers N]

# 03 — ICA ablation → cache/ica/
conda run -n eeg_pipeline python scripts/03_run_ica.py [--force] [--workers N]

# 04 — Verification experiments V1/V2/V3 → results/verification/*.csv
conda run -n eeg_pipeline python scripts/04_run_verification.py [--workers N]

# Scripts 02, 03, and 04 all read only from cache/epochs/ — they can run in parallel after 01.

# 05 — Per-subject waveform and PSD figures → results/figures/{subject_id}/
conda run -n eeg_pipeline python scripts/05_run_visualization.py [--n-subjects N] [--epoch-idx I] [--channels "FP1,FP2,T3"] [--export-edf] [--export-edf-max-epochs N]

# 06 — XGBoost + SHAP for all five conditions → results/xgboost/
conda run -n eeg_pipeline python scripts/06_train_xgboost.py [--condition raw|ica|wiener|wiener_phasegated|wiener_zerophase|all] [--feature-set base211|base211_conn80] [--force] [--workers N]

# 07 — Archive experiment config + results (xgboost AND cnn, whichever are present) → experiments/<timestamp>/
conda run -n eeg_pipeline python scripts/07_organize_experiment.py [--name LABEL] [--config PATH] [--results-dir PATH]

# 08 — EEGNet CNN training directly on raw epoch tensors → results/cnn/
conda run -n eeg_pipeline python scripts/08_train_cnn.py [--condition raw|ica|wiener|all] [--config PATH] [--force] [--workers N]

# 09 — Aggregate the fixed 8-cell phase/coherence grid → results/exp_wiener_phase/grid_summary/
conda run -n eeg_pipeline python scripts/09_analyze_wiener_phase_grid.py [--results-root PATH]
```

Script 08 has no dependency on script 06 — it reads the same `cache/{epochs,wiener_frequency,ica}/` trees directly and trains an independent EEGNet model per condition. It can run any time after 01 (+02/03 for the wiener/ica conditions).

### Cache invalidation tiers

Changing a `configs/default.yaml` key requires re-running all scripts at or after its tier with `--force`:

| Tier | Scripts to re-run | Config sections that trigger it |
|------|-------------------|----------------------------------|
| 1 | `01+` (all scripts) | `paths`, `dataset`, `split`, `preprocessing`, `channels.standard_19` |
| 2 | `02+` | `channels.channel_groups`, `channels.passthrough`, `wiener` |
| 3 | `03+` | `ica` |
| 4 | `06` only | `ml.xgboost`, `ml.shap`, and feature-profile settings used by `base211_conn80` |
| 5 | `08` only | `ml.cnn` |

Scripts 04 and 05 produce no cache and always re-run from existing caches — changing `verification` or `visualization` keys requires no `--force`. This includes `visualization.export_edf`/`export_edf_max_epochs` — plain output-gating flags for script 05's optional `.edf` export step, not cache keys. `export_edf_max_epochs` must be a positive integer (`>=1`). Script 06 additionally has a **shape guard**: if a cached `cache/features/{condition}_{split}.npz` has a different column count than the current `FEATURE_NAMES` (e.g. after a feature-engineering code change bumped the vector length), `_load_or_extract_features` raises `ValueError` naming `--force` rather than silently training on a stale/misaligned feature matrix.

### Running with a local config

To experiment without modifying the tracked `default.yaml`:

```bash
cp configs/default.yaml configs/local.yaml
# edit configs/local.yaml freely
conda run -n eeg_pipeline python scripts/01_extract_epochs.py --config configs/local.yaml
```

Add `configs/local*.yaml` to `.gitignore` to keep it untracked. **The config file must live directly inside `configs/`** (one level below the project root) — `load_config()` derives the project root as `config_path.parent.parent`, so placing it elsewhere breaks relative `cache_dir`/`results_dir` resolution. Use absolute paths in the config as an alternative.

`configs/exp_chgroups_{1..5}.yaml` are pre-built variants used by `scripts/run_chgroups_experiment.sh` (see below) — each varies only `channels.channel_groups` (e.g. exp 1 = frontal-only `[FP1, FP2]`) to ablate which conduction pathways the Wiener filter targets, and points `cache_dir`/`results_dir` at experiment-specific paths. `configs/smoke_test.yaml` points at a small synthetic dataset for fast end-to-end iteration (see Smoke Testing below).

## Architecture

The package is `eeg_bg/`, pip-installed from `setup.py`. Configuration is loaded from `configs/default.yaml` via `eeg_bg.config.settings.load_config()`, which resolves relative `cache_dir` / `results_dir` paths against the project root.

**Data flow (unidirectional):**
```
EDF files (TUEP v3.1.0, /root/autodl-tmp/EEGdata/TUEP/v3.1.0)
  → io/edf_reader.py      — loads EDF, normalises channel names (strips "EEG " prefix and "-REF" suffix), resamples to 125 Hz, converts V→µV
  → io/annotation.py      — parses csv_bi files; excludes ±30 s seizure buffers from background intervals
  → preprocessing/epoch.py — slices 20 s epochs, rejects epochs exceeding 200 µV
  → cache/epochs/          — checkpoint 1 (.npz via io/cache.py)
  → decomposition/wiener.py — vector Wiener decomposition (core method)
  → decomposition/ica.py    — FastICA ablation baseline
  → cache/wiener_frequency|ica/ — checkpoint 2
  → verification/           — V1 coherence, V2 transitivity, V3 frequency variation
  → visualization/          — matplotlib figures returned as plt.Figure (never call plt.show())
  → features/extraction.py  — extract_epoch_features() → 211-dim vector per epoch (per-channel + asymmetry)
  → ml/xgb_pipeline.py      — GridSearchCV + early stopping; subject-level aggregation
  → ml/shap_analysis.py     — TreeExplainer SHAP; band/channel aggregation; comparison plot
  → results/xgboost/        — model.joblib, metrics JSON, SHAP plots
  → ml/cnn_pipeline.py      — parallel path: EEGNet trained directly on raw (19, 2500) epoch tensors (no hand-crafted features)
  → results/cnn/            — best_model.pt, metrics JSON, predictions CSV
```

### Key modules

| Module | Responsibility |
|--------|---------------|
| `eeg_bg/config/settings.py` | `load_config(path)` — loads YAML and resolves relative `cache_dir`/`results_dir` to absolute paths against the project root. |
| `eeg_bg/io/edf_reader.py` | `load_edf(path, cfg)` — loads EDF via MNE, strips `"EEG "` prefix and `"-REF"` suffix from channel names, resamples to 125 Hz, converts V→µV. |
| `eeg_bg/io/edf_writer.py` | `export_epoch_edf(epoch, ch_names, sfreq, out_path)` — inverse of `edf_reader.load_edf`; reconstructs a single (n_channels, n_times) µV array into a real `.edf` file via `mne.export.export_raw` (requires `edfio`). Used by script 05's optional `--export-edf` step; MNE is imported lazily only when export is enabled. |
| `eeg_bg/io/annotation.py` | Parses `csv_bi` seizure annotation files; `extract_bckg_intervals` returns non-seizure segments with ±`seizure_buffer_sec` guard zones excluded. |
| `eeg_bg/preprocessing/epoch.py` | `slice_epochs` — cuts fixed-length (20 s) epochs from a continuous recording; rejects epochs where any channel exceeds `artifact_threshold_uv` (200 µV). |
| `eeg_bg/decomposition/wiener.py` | Core Wiener filter: `estimate_cross_psd` → `compute_wiener_filter` → `apply_wiener_filter` → `decompose_epoch` / `decompose_subject`. Returns `WienerResult` (raw, specific, coherent, filters dict, skipped_pairs). |
| `eeg_bg/decomposition/wiener_scalar.py` | Fixed-scalar ablation baseline for comparison. |
| `eeg_bg/decomposition/phase_gate.py` | Shared zero-referenced hard phase gate for Wiener variants. `phase_gate_threshold_rad` is in `[0, π]`, where `0` admits only strictly in-phase coherence and `π` admits all phases; phase difference `π` is treated as the maximum phase difference. |
| `eeg_bg/decomposition/wiener_phasegated.py` | Complex Wiener ablation: computes the normal complex `h(f)` and multiplies coefficients by the zero-referenced phase gate from `phase_gate.py`. `threshold=π` is exactly the normal `frequency` mode; smaller thresholds preserve out-of-phase coherence. |
| `eeg_bg/decomposition/wiener_zerophase.py` | Real-constrained ("zerophase") per-frequency Wiener ablation: solves `Re(S_ref) h = Re(s_cross)` per frequency bin, then applies the same zero-referenced phase gate. Set `phase_gate_threshold_rad=π` to recover legacy un-gated zerophase behavior. |
| `eeg_bg/decomposition/ica.py` | `fit_ica` uses FP1/FP2 as artifact-reference channels; `apply_ica` removes components correlated above `artifact_corr_threshold`. Stores cleaned signal as `specific` key (mirrors Wiener). |
| `eeg_bg/verification/coherence.py` | V1: pairwise coherence matrix before/after decomposition (`run_v1`). Uses `freq_resolution_hz` (not `nperseg`) to set estimation window so coherence averaging is valid. |
| `eeg_bg/verification/transitivity.py` | V2: single-point-source transitivity constraint; V3: frequency variation of \|h(f)\| across target band. |
| `eeg_bg/visualization/psd_plots.py` | `plot_psd_comparison`: PSD overlay (raw / Wiener-specific / ICA-cleaned) for target channels. Channels default to `cfg["visualization"]["psd_target_channels"]` (FP1, FP2). Uses boxcar Welch consistent with decomposition. |
| `eeg_bg/visualization/waveform_plots.py` | `plot_multichannel_comparison`: stacked all-channel waveform with up to 3 panels (raw / Wiener / ICA). |
| `eeg_bg/preprocessing/reference.py` | `detect_reference` infers AR/LE from montage dir name; `filter_by_reference` subsets the subject index to one scheme. |
| `eeg_bg/io/dataset.py` | Traverses TUEP directory tree → subject index DataFrame; `assign_splits` splits by subject (not recording), stratified by label. |
| `eeg_bg/io/cache.py` | `load_or_compute` wraps any `compute_fn` with `.npz` on-disk caching; cache key = SHA-256 of `edf_path|start_sec|sfreq|bandpass`. |
| `eeg_bg/features/_constants.py` | `_STANDARD_19` — the canonical 19-channel order. Extracted to its own module (not `extraction.py`) purely to avoid a circular import: `connectivity.py` needs the channel list to build `ALL_PAIRS` at import time, but `extraction.py` imports `connectivity.py`. |
| `eeg_bg/features/asymmetry.py` | `hemispheric_asymmetry(epoch, ch_names, sfreq)` → `(40,)` vector; `ASYMMETRY_NAMES` — 40 strings (`"asym_{left}_{right}_{band}"`). Order is fixed; reordering invalidates saved SHAP `.npy` arrays. 8 symmetric pairs × 5 bands, formula: `(P_left − P_right) / (P_left + P_right + ε)`. |
| `eeg_bg/features/wavelet.py` | `wavelet_features(signal)` → `(27,)` per channel (513 total across 19 channels). PyWavelets `db4`, 6-level DWT (`level 1` ≈ 32–62 Hz γ … `level 6` ≈ 0.5–2 Hz δ). 27 features/channel: detail energy per level (6), modulus-maxima mean per level (6), reconstructed-band signal stats (15). Cut from an original 66/channel (7 subgroups) on 2026-07-01 — entropy, coefficient stats, approximation-coefficient stats, modulus-maxima counts, and scale-energy ratios were dropped as redundant with existing Welch-based features and absent from SHAP top-20s. `WAVELET_NAMES` built at import time. **Not called from `extraction.py`** — `master` was reverted to the original 211-dim vector (per-channel + asymmetry only); this module and its own unit tests remain in the codebase, untouched, for potential future use. |
| `eeg_bg/features/connectivity.py` | `connectivity_features(epoch, ch_names, sfreq, nperseg)` → `(80,)`. Magnitude-squared coherence + Phase-Locking Value (PLV, via Hilbert transform) for `ALL_PAIRS` = the 8 homotopic pairs in `asymmetry.SYMMETRIC_PAIRS` × 5 bands × 2 metrics. Restricted from all C(19,2) = 171 pairs on 2026-07-01 — inter-hemispheric coherence/PLV is a physiologically distinct signal from power-asymmetry (already captured in `asymmetry.py`) and a documented correlate of lateralized epileptogenic networks. Bandpass+Hilbert phase is computed once per channel per band and reused across pairs for efficiency. **Not called from `extraction.py`** — same reversion as `wavelet.py` above; module and tests untouched. |
| `eeg_bg/features/complexity.py` | `complexity_features(epoch, ch_names, m, r_factor)` → `(38,)`. Sample Entropy (embedding dim `m=2`, tolerance `r=0.2×std`, `scipy.spatial.cKDTree` for O(n log n) template matching) + Lempel-Ziv Complexity (binarised at median, normalised by `n/log2(n)`) per channel. **Not called from `extraction.py`** — same reversion as `wavelet.py` above; module and tests untouched. |
| `eeg_bg/features/temporal_stats.py` | `epoch_temporal_stats(epoch, ch_names, scales)` → `(228,)`. Mean/variance/skewness/kurtosis computed per non-overlapping window then averaged across windows, at 3 scales (125/375/750 samples = 1 s/3 s/6 s at 125 Hz) per channel. **Not called from `extraction.py`** — same reversion as `wavelet.py` above; module and tests untouched. |
| `eeg_bg/features/extraction.py` | `extract_epoch_features(epoch, ch_names, sfreq)` → `(211,)` vector; `build_dataset(cache_root, condition, split, ...)` → `(X, y, subject_ids)`. Feature cache in `cache/features/{condition}_{split}.npz`. |
| `eeg_bg/ml/xgb_pipeline.py` | `train_xgboost`: Phase 1 GridSearchCV, Phase 2 early-stopping refit. `subject_level_predict`: epoch-level proba → subject-mean. `evaluate_subject_level`: AUROC/F1/Acc. `find_optimal_threshold`: also reused by the CNN pipeline. |
| `eeg_bg/ml/shap_analysis.py` | `compute_shap_values` (TreeExplainer), `aggregate_shap_by_band/channel`, `plot_shap_summary` (beeswarm), `plot_shap_comparison` (2×5 cross-condition publication figure: raw/ica/wiener/wiener_phasegated/wiener_zerophase). |
| `eeg_bg/ml/cnn_model.py` | `EEGNet(n_channels=19, n_times=1000, F1, D, dropout)` — compact EEG CNN (Lawhern et al. 2018): temporal conv → depthwise spatial conv → separable conv → sigmoid. `n_times=1000` is just the constructor default; `cnn_pipeline.py` always derives the real value from the data (currently 2500, `epoch_length_sec × target_sfreq`). Input `(batch, 1, 19, n_times)`, output `(batch, 1)` probability. |
| `eeg_bg/ml/cnn_dataset.py` | `EEGEpochDataset(cache_root, condition, split)` — PyTorch `Dataset` reading the same `cache/{epochs,wiener_frequency,ica}/` trees as `build_dataset`; yields `(epoch_tensor, label, subject_id)` with each channel z-scored independently. |
| `eeg_bg/ml/cnn_pipeline.py` | `cnn_predict_epochs`: batched inference → subject-level DataFrame (epoch probas averaged per subject). `train_cnn`: training loop (weighted BCE for class imbalance, Adam, `ReduceLROnPlateau`, early stopping on val AUROC); writes `best_model.pt` + metrics/predictions to `out_dir`. Reuses `find_optimal_threshold`/`evaluate_subject_level` from `xgb_pipeline.py`. |
| `eeg_bg/visualization/coherence_plots.py` | `plot_coherence_matrix` (pre/post heatmap side-by-side), `plot_coherence_reduction` (boxplot by pair or subject), `plot_signal_decomposition` (raw / coherent / specific waveform panels for one channel). |
| `eeg_bg/visualization/filter_plots.py` | `plot_wiener_filter_response` (amplitude + phase for one pair), `plot_all_pairs_response` (grid across all pairs in a `WienerResult`). |
| `eeg_bg/visualization/verification_plots.py` | `plot_v2_transitivity`, `plot_v3_frequency_variation`, `plot_ica_vs_wiener_coherence` (3-panel raw/ICA/Wiener coherence matrix). |
| `eeg_bg/features/band_power.py` | `relative_band_power(signal, sfreq, band)` → scalar; `BANDS` dict mapping name→(low, high) Hz for delta/theta/alpha/beta/gamma. |
| `eeg_bg/features/hjorth.py` | `hjorth_parameters(signal)` → `(activity, mobility, complexity)` triple. |
| `eeg_bg/features/spectral_entropy.py` | `spectral_entropy(signal, sfreq)` → scalar normalised Shannon entropy of PSD. |
| `eeg_bg/features/extraction.py` (constants) | `FEATURE_NAMES` — public list of 211 strings built at import time; positionally stable (see below) since SHAP `.npy` arrays are indexed by position. `_CONDITION_TO_SUBDIR` maps `"wiener"→"wiener_frequency"` etc. |

### Channel groups (G1–G6)

The Wiener filter operates on **movement-artifact conduction pathways**, not bilateral pairs. Groups are defined in `configs/default.yaml` under `channels.channel_groups`:
- G1 `[FP1, FP2]` — symmetric facial (frontalis)
- G2 `[F7, T3]` — left SCM
- G3 `[T3, T5, O1]` — left posterior neck (3-channel chain)
- G4 `[O1, O2]` — bilateral occipitalis
- G5 `[F8, T4]` — right SCM
- G6 `[T4, T6, O2]` — right posterior neck (3-channel chain)

Passthrough channels (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz`) are never filtered.

`scripts/run_chgroups_experiment.sh` runs a 5-way ablation over alternative `channel_groups` definitions (`configs/exp_chgroups_{1..5}.yaml`, e.g. exp 1 = frontal-only `[FP1, FP2]`) to measure how the choice of conduction-pathway grouping affects downstream AUROC. It separates invariant work (epoch extraction, ICA, raw/ica XGBoost — run once) from per-experiment work (Wiener decomposition + wiener-only XGBoost + archive, run 5×), and writes a per-step runtime log to `results/exp_chgroups/runtime_<timestamp>.log`. Usage: `bash scripts/run_chgroups_experiment.sh [--workers N] [--from N]` (`--from` resumes at experiment N, skipping the one-time pre-steps).

### Wiener filter implementation details

- PSD estimated with **boxcar window** so that when `nperseg == n_times` the filter can be applied exactly via rfft without windowing mismatch.
- When `nperseg < n_times`, filter coefficients are linearly interpolated to the full rfft grid; `specific + coherent == raw` is guaranteed by construction regardless.
- A coherence gate (max pairwise coherence across all pairs in the group, over the target frequency band) skips groups below `coherence_threshold` (default 0.15).
- `nperseg` in `wiener:` is for filter estimation; V1 coherence uses `freq_resolution_hz` (125 / 0.5 = 250 samples = 10 segments per 2500-sample epoch) to avoid trivial coherence=1.

### Feature vector layout

`extract_epoch_features` produces the original **211-dim vector**, built by concatenating two blocks in this order:

| Range | Size | Block | Detail |
|-------|------|-------|--------|
| `[0:171]` | 171 | Per-channel statistics | 19 channels × 9 features (`delta/theta/alpha/beta/gamma_power, hjorth_activity/mobility/complexity, spectral_entropy`), in `_STANDARD_19` order |
| `[171:211]` | 40 | Hemispheric asymmetry | 8 pairs × 5 bands, see `ASYMMETRY_NAMES` |

**This vector is positionally stable** — reordering `standard_19` or `SYMMETRIC_PAIRS` invalidates saved SHAP `.npy` arrays (indexed by position, not name). A wavelet DWT + connectivity + complexity + multi-scale temporal-stats expansion (up to 1070 dims) was integrated on top of this vector and later reverted back to it; `eeg_bg/features/{wavelet,connectivity,complexity,temporal_stats}.py` and their unit tests still exist in the codebase (see the module table above) but are not called from `extraction.py`. `aggregate_shap_by_band` in `eeg_bg/ml/shap_analysis.py` still reports `"wavelet"`/`"connectivity"`/`"complexity"`/`"temporal"` keys in `shap_by_band.json` — these will correctly show `0.0` under this 211-dim vector since no matching feature names exist (pattern-matching degrades gracefully; not a bug).

The wavelet, complexity, and temporal-stat blocks remain disconnected. The
`base211_conn80` profile now calls `connectivity_features()` and uses
`ml.features.connectivity.nperseg`; `base211` remains the original 211-dim
vector. `ml.shap.pruning_threshold` remains unused.

### Label encoding

`label = 0` → epilepsy (`00_` prefix in cache dirs); `label = 1` → control (`01_` prefix). This is the TUEP convention preserved throughout the pipeline.

### Cache directory layout

```
cache/
├── epochs/{label_prefix}_{subject_id}/{sha256_key}.npz   — keys: epochs, ch_names, label, subject_id, split
├── wiener_frequency/ (same tree)                          — keys: specific, coherent, label, subject_id, split
├── wiener_scalar/ (same tree)                             — keys: specific, coherent, label, subject_id, split (--mode scalar output)
├── wiener_phasegated/ (same tree)                         — keys: specific, coherent, label, subject_id, split (--mode phasegated output)
├── wiener_zerophase/ (same tree)                          — keys: specific, coherent, label, subject_id, split (--mode zerophase output)
├── ica/ (same tree)                                       — keys: specific, n_artifacts_removed, label, subject_id, split
└── features/{condition}_{split}.npz                       — keys: X, y, subject_ids
```
The `wiener` condition in `build_dataset` / `--condition` maps to `cache/wiener_frequency/` (not `cache/wiener/`); `wiener_phasegated` maps to `cache/wiener_phasegated/`; `wiener_zerophase` maps to `cache/wiener_zerophase/`. Both phase-gated Wiener conditions are wired into `scripts/06_train_xgboost.py` (included in `all`) but **not** into the CNN pipeline — `eeg_bg/ml/cnn_dataset.py`'s `EEGEpochDataset` and `scripts/08_train_cnn.py` still only support `raw`/`ica`/`wiener`. `EEGEpochDataset` reads the same `cache/{epochs,wiener_frequency,ica}/` trees directly (no separate CNN cache) — it loads raw `(19, n_times)` tensors (currently `(19, 2500)`) rather than the 211-dim feature vectors.

### Output directory structure

```
results/
├── figures/{subject_id}/
│   ├── waveform_comparison.png       — all-channel stacked waveform (raw | Wiener | ICA)
│   ├── psd_comparison.png            — PSD overlay for psd_target_channels
│   └── edf/epoch_{i}/{condition}.edf — (only with --export-edf) up to export_edf_max_epochs (>=1)
│       epochs, each with raw/wiener/wiener_phasegated/wiener_zerophase/ica grouped in one folder for
│       side-by-side comparison
├── verification/
│   ├── v1_coherence.csv
│   ├── v2_transitivity.csv
│   └── v3_frequency_variation.csv
├── xgboost/
│   ├── {base211,base211_conn80}/
│   │   ├── {raw,ica,wiener,wiener_phasegated,wiener_zerophase}/
│   │   ├── model.joblib              — fitted XGBClassifier
│   │   ├── scaler.joblib             — StandardScaler (fit on train)
│   │   ├── best_params.json          — GridSearchCV best hyperparameters
│   │   ├── val_metrics.json / test_metrics.json  — {auroc, f1, accuracy}
│   │   ├── val_predictions.csv / test_predictions.csv  — subject_id, pred_proba, true_label
│   │   ├── shap_values_test.npy      — (n_test_epochs, 211) raw SHAP values
│   │   ├── shap_summary.png          — beeswarm plot (top 20 features)
│   │   ├── shap_by_band.json         — mean |SHAP| per feature-type group
│   │   └── shap_by_channel.json      — mean |SHAP| per EEG channel
│   │   ├── comparison_summary.csv    — profile-specific condition summary
│   │   └── shap_comparison.png       — profile-specific comparison figure
├── cnn/
│   ├── {raw,ica,wiener}/
│   │   ├── best_model.pt             — EEGNet state_dict
│   │   ├── best_params.json          — {F1, D, dropout, lr, batch_size, stopped_epoch}
│   │   ├── val_metrics.json / test_metrics.json  — {auroc, f1, accuracy, threshold}
│   │   └── val_predictions.csv / test_predictions.csv
│   └── comparison_summary.csv        — written when all 3 conditions have been trained
└── exp_chgroups/                     — output of scripts/run_chgroups_experiment.sh
    ├── runtime_<timestamp>.log
    └── {1..5}/xgboost/base211/{raw,ica,wiener}/  — per-channel-group-config results
```

### Experiment archive layout (script 07)

Script 07 snapshots the current config and `results/xgboost/` **and** `results/cnn/` into a timestamped directory, discovering whichever conditions actually have complete results in each:

```
experiments/<timestamp>_<name>/
├── config.yaml              — copy of the config used
├── experiment.json          — metadata + full metrics for all conditions (xgboost + cnn)
├── report.md                — human-readable summary table(s), including a CNN results table if any CNN results are found
├── xgboost/{base211,base211_conn80}/{raw,ica,wiener,wiener_phasegated,wiener_zerophase}/
│   └── per-profile metrics, predictions, SHAP data, and best_params
├── verification/             — V1, gate, skip, fusion, and connectivity summaries
└── cnn/{raw,ica,wiener}/    — per-condition CNN result files (only for conditions with a test_metrics.json)
```

### Matplotlib backend

Any script that saves figures must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot`. Scripts 05, 06, and 08 already do this. Visualization functions in `eeg_bg/visualization/` return `plt.Figure` objects and never call `plt.show()`.

## Smoke Testing

`scripts/create_smoke_data.py` generates a tiny synthetic TUEP-format dataset (8 subjects, 90 s each, random noise) directly under the hardcoded path `D:/EEGdata/TUEP/v3.1.0` — run it once to populate synthetic data before using `configs/smoke_test.yaml`. `scripts/run_smoke_test.py` then runs scripts 01–07 end-to-end against `configs/smoke_test.yaml` with `--workers 1`, stopping at the first failing step and printing per-step timing — useful for verifying the full pipeline wiring after a code change without touching real TUEP data or the default cache/results directories.

## Tests

All tests run without real EDF data. There are four conftest scopes:

- **`tests/conftest.py`** (root): `synthetic_epoch` (19-ch 1000-sample epoch with a single point source, SNR ≈ 50:1), `synthetic_epochs_batch` (batch of 5), `tmp_cache_dir`, `cfg` (deep-copied `BASE_CFG` dict).
- **`tests/test_features/conftest.py`**: `ch_names_19`, `sfreq`, `synthetic_epoch` (simple random — independent of root fixture), `pure_sine_signal` (10 Hz sine), `constant_signal`.
- **`tests/test_ml/conftest.py`**: `tiny_xgb_model` (10-feature, 5-estimator, session-scoped), `full_feature_xgb_model` (211-feature, session-scoped).
- **`tests/test_ml/test_cnn_*.py`** fixtures build a `tiny_cache` tmp_path fixture: minimal `epochs/{subject_id}/data.npz` files with `epochs`/`label`/`subject_id`/`split` keys, used to smoke-test `EEGEpochDataset`, `EEGNet`, and `train_cnn` without real data or GPU.

Test files under `tests/test_visualization/` must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot` (same rule as scripts).

Integration tests (requiring real TUEP EDF files) should be marked `@pytest.mark.integration`.

`check_fixtures.py` (project root) is a standalone debug script that reconstructs fixture arrays and prints their shapes — useful for verifying conftest parity outside pytest.

## AGENTS.md maintenance

This file was created/updated by the ZCode `/init` command. To update it in the future:
- Run `/init` again — it will read the current file and make targeted edits.
- Or manually edit `AGENTS.md` directly.

Changes to the pipeline (new scripts, new config keys, new output directories, or changes to the feature vector) should be reflected here so future agents don't miss context.

## Non-obvious constraints and gotchas

- **Paired config keys**: `dataset.reference_scheme` and `dataset.montage_dir` must always change together (`"ar"` ↔ `"01_tcp_ar"`, `"le"` ↔ `"02_tcp_le"`). Mismatching them produces zero EDF files for one class. Similarly, `wiener.freq_band` must be a subset of `preprocessing.bandpass`, and `wiener.nperseg` must be ≤ `target_sfreq × epoch_length_sec`.
- **`bandpass_filter()` in `epoch.py` is not used in the pipeline**: `load_edf` in `edf_reader.py` applies MNE's `raw.filter()` on the continuous signal before resampling. The standalone `bandpass_filter()` function in `epoch.py` (5th-order `sosfiltfilt`) is available for ad-hoc use but is never called by any script.
- **XGBoost `n_estimators` in `param_grid` is ignored during Phase 1**: `xgb_pipeline.py` overrides it to 500 for the grid search and uses early stopping in Phase 2 to find the final tree count. The entry in `configs/default.yaml` is documentation only.
- **`device="cuda"` automatically sets `n_jobs=1`**: `xgb_pipeline.py` detects the CUDA device setting and overrides GridSearchCV's `n_jobs` to avoid CUDA context conflicts across parallel workers. No manual change needed.
- **ICA fits on a 1 Hz high-pass copy but applies to 0.5 Hz data**: `fit_ica()` creates a temporary high-pass-filtered copy for FastICA convergence (MNE best practice), then applies the fitted mixing matrix to the original 0.5 Hz bandpass epochs. The `specific` output in the ICA cache is in the original 0.5 Hz bandpass domain. `max_iter` (default 1000) is read from `ica.max_iter` in the config and passed directly to the MNE ICA constructor — the scikit-learn default of 200 is too low for 19-channel EEG and causes frequent `ConvergenceWarning`.
- **`FEATURE_NAMES` (211 entries) must stay positionally stable**: SHAP `.npy` arrays `(n_test_epochs, 211)` are indexed by position against `FEATURE_NAMES`. Any reordering of channels in `configs/default.yaml` `standard_19`, of pairs in `asymmetry.SYMMETRIC_PAIRS`, or of the block order in `eeg_bg/features/extraction.py` invalidates saved SHAP arrays. Script 06 guards against a stale feature-dim mismatch and raises `ValueError` pointing at `--force`.
- **`reference_scheme` filter is applied before any EDF is loaded**: Script 01 only processes recordings under the `montage_dir` subdirectory (default `01_tcp_ar`). Linked-ears (`tcp_le`) recordings are silently excluded.
- **Scripts 01–04, 06, and 08 use `ProcessPoolExecutor`** (default `os.cpu_count()` workers; script 08 uses PyTorch `DataLoader(num_workers=...)` instead). On Windows, multiprocessing uses `spawn`, so each worker re-imports the full module graph at startup — worker startup overhead is higher than on Linux. Use `--workers N` to cap concurrency on memory-constrained machines or when other processes need CPU. Script 04 additionally runs V1/V2/V3 concurrently via `ThreadPoolExecutor` in Phase 2 after decomposition. Script 05 is sequential (no `--workers` flag). Script 06 runs conditions sequentially but parallelises feature extraction within each condition. Script 08 defaults `num_workers=0` (main-process loading) because it's Windows-safe by default; override via `ml.cnn.num_workers` or `--workers`.
- **Cache key composition for script 01**: the SHA-256 key is derived from `edf_path`, `target_sfreq`, and `bandpass` only. Changing `epoch_length_sec`, `artifact_threshold_uv`, or `seizure_buffer_sec` does NOT generate a new key — the existing `.npz` is silently reused unless you pass `--force`. Scripts 02 and 03 have no key-based invalidation at all; any config change to `wiener.*` or `ica.*` requires `--force` for those scripts.
- **`--force` does not cascade and does not clean up orphaned files**: running `01_extract_epochs.py --force` does not force-rerun scripts 02–08; each script must be passed `--force` independently. Old `.npz` files at old key paths are orphaned on disk (not deleted). For script 06, `--force` only bypasses the feature-extraction cache (`cache/features/`); model training, SHAP computation, and all output files always regenerate regardless. Script 08's `--force` behaves analogously: it re-trains and rewrites `results/cnn/{condition}/` even if metrics already exist there.
- **`create_smoke_data.py` writes to a hardcoded Windows-style path** (`D:/EEGdata/TUEP/v3.1.0`) — edit `DATA_ROOT` at the top of the script if generating smoke data on a non-Windows machine or a different data root, and make sure `configs/smoke_test.yaml`'s `paths.data_root` matches.

## Reference Documentation

- **`docs/Developer_qa.md`** — Detailed Q&A: epoch validity criteria (4 criteria with decision flow), `.npz` cache file schemas (exact keys/shapes/dtypes per cache family), feature extraction internals (band definitions, Hjorth formulae, spectral entropy formula, per-condition signal mapping). Its 211-dim feature-layout description is accurate again now that `master` has been reverted to this vector, but it predates the CNN pipeline and may still have stale label-dtype info — cross-check non-feature-layout numeric claims against the code before relying on them.
- **`docs/preprocessing.md`** — Every transformation applied to raw TUEP EDF recordings before epochs are written to `cache/epochs/`: dataset discovery, reference filtering, split assignment, EDF loading, channel normalisation, bandpass filtering, resampling, unit conversion, epoch slicing, and artifact rejection.
- **`docs/superpowers/plans/`** and **`docs/superpowers/specs/`** — implementation plans and design specs for major feature additions (Wiener framework, experiment organizer, feature-engineering expansion, CNN training). Useful for the *rationale* behind a module's design; treat exact dimension/line-number claims as historical snapshots, not current truth — the feature-engineering expansion plan targeted a 2415-dim vector, and `FEATURE_NAMES` briefly grew to 3441-dim, then 1070-dim, before the wavelet/connectivity/complexity/temporal blocks were disconnected entirely and `master` reverted to the original 211-dim vector (see "Feature vector layout" above).
- **`eeg_bg/README.md`** — Full package API reference: every public function signature, return type, parameter table, and usage example.
