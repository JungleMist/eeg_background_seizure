# AGENTS.md

This file provides guidance to ZCode / Claude Code agents working with this repository.
Last updated: 2026-07-30.

## Project identity and research scope

- The local repository and GitHub repository are both named **`eeg_wiener_decomposition`**. The `origin` remote is `git@github.com:JungleMist/eeg_wiener_decomposition.git`.
- The EEG denoising scheme is named **EEG Channel Matrix Adaptive Denoiser (ECMAD)**. Use **ECMAD** in research- and user-facing descriptions; identify the current core algorithm as channel-group matrix-adaptive vector Wiener decomposition. Keep established implementation identifiers such as `wiener`, `wiener_frequency`, and `WienerMode` unchanged for code, configuration, cache, and result compatibility.
- The primary research question is **EEG denoising with ECMAD**: remove shared physical/artifact components while preserving physiologically meaningful, locally generated EEG. Epilepsy/abnormality classification is an evaluation instrument, not the project identity or final scientific objective.
- The established evaluation track uses TUEP/TUAB downstream performance. Scripts 01–07 compare Raw, ICA, and Wiener variants with XGBoost/SHAP; script 08 adds a TUEP-only EEGNet comparison. AUROC/F1/accuracy quantify whether denoising preserves or improves task-relevant information, but they are indirect denoising metrics.
- The direct signal-quality track uses ERP-CORE Flankers. Script 10 compares Raw, standard ICA, and Wiener on response-locked ERN/LRP signals; script 11 performs paired participant-level statistics on script 10 outputs; script 12 performs leakage-safe ECMAD phase selection for ERN trial classification. Signal-quality measures remain primary in script 10, with trial classification retained as a secondary measure.
- Future evaluation work should keep these two tracks conceptually separate: **downstream utility on TUEP/TUAB** and **direct ERP denoising/signal preservation on ERP-CORE**. Do not describe the repository as merely a seizure or abnormal-EEG classifier.
- The PySide6 desktop frontend is branded **ECMAD Studio**. Its visible title, navigation identity, and Wiener-processing labels should use ECMAD, while the internal package and executable entry points remain `eeg_bg` / `eeg_bg_studio` unless a separate compatibility-breaking rename is explicitly requested.

## Environment

- Python via **conda env `eeg_pipeline`**, Python 3.11. Use `conda run -n eeg_pipeline` to run commands.
- Conda's install path is not fixed by the repository; use `conda info --base` / `conda env list` instead of assuming `/Users/jsu/miniconda3` or `/root/miniconda3`.
- To recreate: `conda env create -f environment.yaml` (Windows export — `C:\` paths baked in, UTF-16). On macOS, maintain a local gitignored `environment_macos.yaml` instead.
- Install the package in development mode: `pip install -e .`
- `requirements.txt` pins numpy 2.4.6, scipy 1.17.1, mne 1.11.0, scikit-learn 1.8.0, joblib 1.5.3, edfio 0.4.13, pandas 2.3.3, matplotlib 3.10.8, pyyaml 6.0.3, pytest 9.0.2, tqdm 4.67.3, xgboost 3.2.0, shap 0.51.0, PyWavelets 1.9.0, and torch 2.12.0+cu130. `requirements-gui.txt` separately pins the ECMAD Studio/PyInstaller stack.
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

## Pipeline Scripts

Scripts 01–08 form the cached TUEP/TUAB pipeline. Scripts 09–12 are analysis, ERP benchmarking/statistics, and ERP phase-search entry points with their own arguments; scripts 10 and 12 default to `configs/erp_core_flankers.yaml`, not `configs/default.yaml`.

```bash
# 01 — Extract background epochs from EDF files → cache/epochs/
conda run -n eeg_pipeline python scripts/01_extract_epochs.py [--force] [--workers N]

# 02 — Wiener decomposition on cached epochs → cache/wiener_frequency/
conda run -n eeg_pipeline python scripts/02_run_wiener.py [--mode frequency|phasegated|scalar|zerophase] [--force] [--workers N]

# 03 — ICA ablation → cache/ica/
conda run -n eeg_pipeline python scripts/03_run_ica.py [--force] [--workers N]

# 04 — Cache-driven V1/gate/connectivity verification by default
conda run -n eeg_pipeline python scripts/04_run_verification.py [--source cache|recompute] [--mode frequency|phasegated|scalar|zerophase] [--checks v1,gate,connectivity] [--workers N]

# After 01, scripts 02 and 03 are independent. Script 04's default cache mode
# requires the matching script 02 output; legacy --source recompute reads epochs.

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

# 10 — ERP-CORE Flankers Raw vs standard ICA vs Wiener benchmark
conda run -n eeg_pipeline python scripts/10_benchmark_erp_core_flankers.py [--data-dir PATH] [--fif PATH] [--config configs/erp_core_flankers.yaml] [--force]

# 11 — Paired participant-level statistics from script 10 subject_metrics.csv
conda run -n eeg_pipeline python scripts/11_analyze_erp_denoising_statistics.py [--results-dir PATH] [--output-dir PATH]

# 12 — Leakage-safe ERP-CORE ERN ECMAD phase/XGBoost search
conda run -n eeg_pipeline python scripts/12_optimize_erp_core_ern_phase.py [--data-dir PATH] [--metric auroc|f1|accuracy] [--workers N] [--force]

# 17 — Continuous TUAB specific/coherent Wiener component caches
conda run -n eeg_pipeline python scripts/17_cache_tuab_continuous_wiener.py [--data-dir PATH] [--cache-dir PATH] [--mode frequency|phasegated|zerophase] [--workers N] [--force]

# 18 — Paired TUAB raw/specific/coherent epochs from script 17 caches
conda run -n eeg_pipeline python scripts/18_extract_tuab_component_epochs.py [--input-dir PATH] [--output-dir PATH] [--mode frequency|phasegated|zerophase] [--workers N] [--force]

# 19 — Independent TUAB EEGNet models from script 18 paired epochs
conda run -n eeg_pipeline python scripts/19_train_tuab_component_eegnet.py [--input-dir PATH] [--output-dir PATH] [--mode frequency|phasegated|zerophase] [--condition raw|specific|coherent|all] [--device auto|cpu|cuda|mps] [--workers N] [--force]

# ERP-CORE counterpart of the fixed 8-cell Wiener phase/coherence experiment
bash scripts/run_erp_core_wiener_phase_grid.sh [--data-dir PATH] [--fif PATH]
```

Scripts 01–07 support TUEP and TUAB. Use `configs/default.yaml` for TUEP and `configs/tuab.yaml` for TUAB. Script 08 is explicitly **TUEP-only** and fails fast when `dataset.active: tuab`; it has no dependency on script 06 for supported TUEP runs.

Script 17 is a standalone TUAB continuous-cache path. It applies the shared
19-channel preprocessing, keeps the configured first 1200 seconds, runs 50%
overlap-add Wiener processing, and writes separate float32 NPZ sequences under
`tuab_continuous_wiener_{mode}/{specific,coherent}/`.

Script 18 reconstructs raw from those paired continuous components, applies a
shared raw-derived artifact-rejection mask to non-overlapping epochs, and stores
raw/specific/coherent together with TUAB train/val/test and abnormal/normal labels.

Script 19 trains separate raw, specific, and coherent EEGNet models from those
paired epochs. It streams one recording at a time, uses record-equal class-
weighted loss, selects checkpoints by validation recording AUPRC, freezes a
validation balanced-accuracy threshold, and reports the official test split at
the recording level.

The TUAB counterpart of the fixed 8-cell Wiener phase/coherence experiment is `bash scripts/run_tuab_wiener_threshold_phase_experiment.sh [--workers N] [--from N]`. Its `configs/exp_tuab_wiener_phase_{1..8}.yaml` files inherit through `exp_tuab_wiener_phase_base.yaml` → `tuab.yaml`, share `cache_tuab`, and write isolated outputs under `results_tuab/exp_wiener_phase/`. The wrapper also runs script 09 with the TUAB config prefix after training and archiving. To limit disk use, the shared experiment driver keeps `epochs/` as the required common input but enforces that `wiener_frequency/`, `wiener_phasegated/`, and `ica/` never coexist; the active derived cache is deleted immediately after its XGBoost feature extraction finishes.

### ERP-CORE direct denoising benchmark (script 10)

`scripts/10_benchmark_erp_core_flankers.py` is a standalone ERP evaluation path; it does not consume the TUEP/TUAB epoch caches. By default it discovers every `sub-*/eeg/*_task-ERN_eeg.set` recording below `erp_core.data_dir` (`~/Data/ERP_CORE`). `--data-dir` overrides that root, while `--fif` retains the legacy MNE one-subject input.

- Shared preprocessing: EEG selection, standard montage, optional 60 Hz notch, 0.1–30 Hz bandpass, resampling to 125 Hz, response-event pairing, epoch rejection, epoch windows, and baseline correction are identical for all three branches.
- Compared branches: `raw` (no denoising), `standard` (MNE FastICA with FP1/FP2 EOG proxies), and `wiener` (the repository Wiener implementation).
- Wiener is applied to continuous data as 20 s windows with 50% weighted overlap-add. The ERP config defines five overlapping groups spanning frontal, sensorimotor, posterior, and midline channels; channels outside those groups are effectively unchanged. Its `passthrough` list records that intended complement as `[P9, P10]`.
- Primary signal measures in `metrics.csv`: `ern_snr_db`, `ern_waveform_r`, `ern_rmse_vs_standard_uv`, ERN peak amplitude/latency, baseline noise SD, LRP peak amplitude/latency and half-peak onset, FP1/FP2 proxy variance, and target-channel RMS change. Root metrics are equal-weight participant means; `subject_metrics.csv` retains every participant/method row. Trial-level correctness classification accuracy/F1/AUC is secondary.
- `standard` ICA is a comparison reference, not clean ground truth. Therefore `ern_rmse_vs_standard_uv` and `ern_waveform_r` measure agreement with ICA, not absolute denoising error or proof of physiological correctness.
- Each participant is processed independently before aggregation. The local directory may contain only a subset of ERP-CORE, so outputs support paired participant-level analysis of the available sample but must not be described as results from the complete release.
- `scripts/run_erp_core_wiener_phase_grid.sh` runs the same fixed eight phase/coherence cells as the TUEP/TUAB experiment. Additional `configs/exp_erp_core_*` files cover high-coherence phase-gate sweeps and a zerophase case, but there is currently no ERP grid-summary aggregator analogous to script 09.

### ERP-CORE statistics and phase search (scripts 11–12)

- Script 11 reads only `subject_metrics.csv` (plus optional `run_summary.json`) from a completed script 10 result directory. It writes `statistics/paired_tests.csv`, `equivalence_tests.csv`, `summary.json`, and `report.md`; it does not reload EEG or compute trial-level SME/synthetic recovery.
- Script 12 discovers ERN `.set` recordings, creates one deterministic subject holdout, selects phase/model/decision threshold using training subjects only, and first generates held-out ECMAD features after those choices are frozen. Its 211 features use a 19-channel ERP layout plus 8 symmetric pairs, distinct from the TUEP/TUAB `_STANDARD_19` layout.
- Script 12's phase grid comes from `erp_core.phase_search.phase_start/phase_stop/phase_step` and includes the exact stop value (currently π). `--workers` controls subject processes for ECMAD/feature generation and XGBoost threads per fit; one subject process handles all requested phases sequentially with native threads capped at one.

### Cache invalidation dependencies

The pipeline is branched rather than a single linear tier chain:

| Changed inputs | Rebuild |
|---|---|
| dataset discovery/splits, preprocessing, or `channels.standard_19` | Script 01 with `--force`, then every derived condition actually used (02/03, verification/visualization, 06, and/or 08) |
| `channels.channel_groups` or Wiener numerical settings | Script 02 with `--force`, then cache-mode 04, Wiener visualizations/features/models, and archives as needed |
| `ica` | Script 03 with `--force`, then ICA visualizations/features/models and archives |
| `ml.features.connectivity.nperseg` or a feature-schema change | Script 06 with `--force` for the affected feature profile |
| `ml.xgboost` or `ml.shap` only | Rerun script 06; training/SHAP always regenerate, so feature `--force` is unnecessary |
| `ml.cnn` | Script 08 with `--force` |

Changing cache/results paths relocates inputs/outputs rather than mutating existing cache contents. Scripts 04 and 05 produce no processing cache. `visualization.export_edf` and `export_edf_max_epochs` only gate script 05 output; the latter must be a positive integer. Script 06 validates a profile-aware schema hash and expected dimension (211 or 291), raising a `--force` error instead of silently loading stale features.

Script 02 stores a Wiener configuration fingerprint in schema-v3 output caches. When `--force` is omitted, a legacy schema or any mismatch in the effective mode, sampling rate, channel groups, or Wiener decomposition settings raises `ValueError` naming `--force` instead of silently reusing stale output.

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
  → verification/           — cache mode: V1/gate/fusion/connectivity; legacy recompute: V1/V2/V3
  → visualization/          — matplotlib figures returned as plt.Figure (never call plt.show())
  → features/extraction.py  — base211, optionally +80 connectivity features
  → ml/xgb_pipeline.py      — patient-grouped CV; TUEP patient / TUAB recording aggregation
  → ml/shap_analysis.py     — TreeExplainer SHAP; band/channel aggregation; comparison plot
  → results/xgboost/        — model.joblib, metrics JSON, SHAP plots
  → ml/cnn_pipeline.py      — parallel path: EEGNet trained directly on raw (19, 2500) epoch tensors (no hand-crafted features)
  → results/cnn/            — best_model.pt, metrics JSON, predictions CSV
```

**ERP-CORE direct denoising branch:**
```
ERP-CORE Flankers EEGLAB SET recordings (or legacy one-subject FIF)
  → common filter/resample/event pairing
  → Raw | standard ICA | 20 s overlap-add Wiener
  → shared response-locked rejection, ERN and LRP epoching/baselines
  → signal-quality metrics + secondary trial classification
  → results/erp_core_flankers/
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
| `eeg_bg/io/dataset.py` | Dataset adapter for TUEP/TUAB recording discovery, patient-safe split assignment, and dataset-specific eligible intervals. TUAB official eval is test; official train produces patient-grouped train/val. |
| `eeg_bg/io/cache.py` | Cache fingerprinting plus shared dataset/patient/recording/evaluation metadata helpers. |
| `eeg_bg/features/_constants.py` | `_STANDARD_19` — the canonical positional feature order shared by extraction, connectivity, and optional feature modules. |
| `eeg_bg/features/asymmetry.py` | `hemispheric_asymmetry(epoch, ch_names, sfreq)` → `(40,)` vector; `ASYMMETRY_NAMES` — 40 strings (`"asym_{left}_{right}_{band}"`). Order is fixed; reordering invalidates saved SHAP `.npy` arrays. 8 symmetric pairs × 5 bands, formula: `(P_left − P_right) / (P_left + P_right + ε)`. |
| `eeg_bg/features/wavelet.py` | `wavelet_features(signal)` → `(27,)` per channel (513 total across 19 channels). PyWavelets `db4`, 6-level DWT; currently not selected by either feature profile. |
| `eeg_bg/features/connectivity.py` | `connectivity_features(epoch, ch_names, sfreq, nperseg)` → `(80,)`: coherence + PLV for 8 homotopic pairs × 5 bands × 2 metrics. It is appended by the `base211_conn80` profile. |
| `eeg_bg/features/complexity.py` | `complexity_features(epoch, ch_names, m, r_factor)` → `(38,)`; implemented and tested but not selected by either current profile. |
| `eeg_bg/features/temporal_stats.py` | `epoch_temporal_stats(epoch, ch_names, scales)` → `(228,)`; implemented and tested but not selected by either current profile. |
| `eeg_bg/features/extraction.py` | `extract_epoch_features(epoch, ch_names, sfreq)` → `(211,)` vector; `build_dataset(cache_root, condition, split, ...)` → `(X, y, subject_ids)`. Feature cache in `cache/features/{condition}_{split}.npz`. |
| `eeg_bg/features/profiles.py` | Registry for `base211` (211 dims) and `base211_conn80` (291 dims); profile names and ordered feature names define cache/SHAP schema identity. |
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
| `scripts/10_benchmark_erp_core_flankers.py` | Standalone ERP-CORE Flankers Raw/standard-ICA/Wiener benchmark. Builds shared response-locked ERN/LRP epochs, computes direct signal-quality and secondary classification metrics, and writes publication-oriented figures. |

### Channel groups (G1–G6)

The Wiener filter operates on **movement-artifact conduction pathways**, not bilateral pairs. Groups are defined in `configs/default.yaml` under `channels.channel_groups`:
- G1 `[FP1, FP2]` — symmetric facial (frontalis)
- G2 `[F7, T3]` — left SCM
- G3 `[T3, T5, O1]` — left posterior neck (3-channel chain)
- G4 `[O1, O2]` — bilateral occipitalis
- G5 `[F8, T4]` — right SCM
- G6 `[T4, T6, O2]` — right posterior neck (3-channel chain)

Channels absent from every group (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz` in the default config) are never filtered. `channels.passthrough` documents this intended complement, but `decompose_epoch` derives the effective passthrough set from group membership and does not read that list; changing the list alone has no numerical effect.

`scripts/run_chgroups_experiment.sh` runs a 5-way ablation over alternative `channel_groups` definitions (`configs/exp_chgroups_{1..5}.yaml`, e.g. exp 1 = frontal-only `[FP1, FP2]`) to measure how the choice of conduction-pathway grouping affects downstream AUROC. It separates invariant work (epoch extraction, ICA, raw/ica XGBoost — run once) from per-experiment work (Wiener decomposition + wiener-only XGBoost + archive, run 5×), and writes a per-step runtime log to `results/exp_chgroups/runtime_<timestamp>.log`. Usage: `bash scripts/run_chgroups_experiment.sh [--workers N] [--from N]` (`--from` resumes at experiment N, skipping the one-time pre-steps).

### Wiener filter implementation details

- PSD estimated with **boxcar window** so that when `nperseg == n_times` the filter can be applied exactly via rfft without windowing mismatch.
- When `nperseg < n_times`, filter coefficients are linearly interpolated to the full rfft grid; `specific + coherent == raw` is guaranteed by construction regardless.
- A target-level coherence gate uses the maximum target-to-reference coherence in the group over the target band; individual target candidates below `coherence_threshold` (default 0.15) are skipped.
- `nperseg` in `wiener:` is for filter estimation (currently 500 samples, 0.25 Hz bins). V1 derives its own 250-sample window from `freq_resolution_hz=0.5`; SciPy's default 50% overlap provides multiple estimates per 2500-sample epoch and avoids trivial single-window coherence.

### Feature vector layout

`extract_epoch_features` produces the original **211-dim vector**, built by concatenating two blocks in this order:

| Range | Size | Block | Detail |
|-------|------|-------|--------|
| `[0:171]` | 171 | Per-channel statistics | 19 channels × 9 features (`delta/theta/alpha/beta/gamma_power, hjorth_activity/mobility/complexity, spectral_entropy`), in `_STANDARD_19` order |
| `[171:211]` | 40 | Hemispheric asymmetry | 8 pairs × 5 bands, see `ASYMMETRY_NAMES` |

**This vector is positionally stable** — changing `_STANDARD_19`, `SYMMETRIC_PAIRS`, or block order invalidates saved SHAP arrays (indexed by position, not name). `base211_conn80` appends the 80 connectivity names/features and uses `ml.features.connectivity.nperseg`, producing 291 dimensions; `base211` remains the original 211-dimensional vector. Wavelet, complexity, and temporal-stat blocks remain outside both profiles. Consequently `shap_by_band.json` reports non-zero `"connectivity"` only for `base211_conn80`, while `"wavelet"`, `"complexity"`, and `"temporal"` remain `0.0` for both current profiles. `ml.shap.pruning_threshold` remains unused.

### Label encoding

Label 1 is always the probability returned by `predict_proba[:, 1]`. TUEP uses 0=epilepsy and 1=control, aggregated per patient. TUAB uses 0=abnormal and 1=normal, aggregated per recording. A TUAB patient may have recordings of both labels; those recordings stay separate, while patient-grouped splitting/CV prevents leakage.

### Cache directory layout

```
cache/
├── epochs/{evaluation_id}/{sha256_key}.npz
├── wiener_frequency/ (same tree)
├── wiener_scalar/ (same tree)
├── wiener_phasegated/ (same tree)
├── wiener_zerophase/ (same tree)
├── ica/ (same tree)
└── features/
    ├── {condition}_{split}.npz                       — base211 compatibility path
    └── base211_conn80/{condition}_{split}.npz        — 291-dim profile
```
Epoch and derived caches also carry shared dataset/patient/recording/evaluation metadata. Wiener caches add schema/fingerprint and target-candidate diagnostics. The `wiener` condition in feature extraction maps to `cache/wiener_frequency/` (not `cache/wiener/`); `wiener_phasegated` maps to `cache/wiener_phasegated/`; `wiener_zerophase` maps to `cache/wiener_zerophase/`. Both phase-gated Wiener conditions are wired into script 06 (included in `all`) but **not** into the CNN pipeline, which supports only `raw`/`ica`/`wiener` and reads `(19, n_times)` tensors (currently `(19, 2500)`) directly.

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
│   ├── verification_metadata.json
│   ├── v1_subject.csv / v1_role_subject.csv / v1_summary.csv
│   ├── gate_subject.csv / gate_summary.csv
│   ├── fusion_subject.csv / fusion_summary.csv
│   ├── skipped_pairs_subject.csv / skipped_pairs_summary.csv
│   └── connectivity_subject.csv / connectivity_role_subject.csv / connectivity_summary.csv
│       # Legacy --source recompute instead writes v1_coherence.csv,
│       # v2_transitivity.csv, and v3_frequency_variation.csv.
├── xgboost/
│   └── {base211,base211_conn80}/
│       ├── {raw,ica,wiener,wiener_phasegated,wiener_zerophase}/
│       │   ├── model.joblib / scaler.joblib
│       │   ├── best_params.json / data_stats.json
│       │   ├── val_metrics.json / test_metrics.json
│       │   ├── val_predictions.csv / test_predictions.csv
│       │   ├── shap_values_test.npy  — (n_test_epochs, 211 or 291)
│       │   ├── shap_summary.png
│       │   └── shap_by_band.json / shap_by_channel.json
│       ├── comparison_summary.csv
│       └── shap_comparison.png       — written after all five conditions
├── cnn/
│   ├── {raw,ica,wiener}/
│   │   ├── best_model.pt             — EEGNet state_dict
│   │   ├── best_params.json          — {F1, D, dropout, lr, batch_size, stopped_epoch}
│   │   ├── val_metrics.json / test_metrics.json  — {auroc, f1, accuracy, threshold}
│   │   └── val_predictions.csv / test_predictions.csv
│   └── comparison_summary.csv        — written when all 3 conditions have been trained
├── erp_core_flankers/
│   ├── metrics.csv                   — participant-equal Raw/standard-ICA/Wiener means
│   ├── subject_metrics.csv           — one metric row per participant and method
│   ├── response_trials.csv           — paired Flankers stimulus-response trials
│   ├── run_summary.json              — input/version/count/fairness/Wiener diagnostics
│   ├── config_resolved.yaml
│   ├── ern_fcz_difference.png / lrp_c3_c4.png  — participant-equal waveforms
│   ├── subjects/sub-*/               — per-participant metrics, trials, and six figures
│   ├── statistics/                    — script 11 paired/TOST outputs
│   └── phase_grid/exp{1..8}/         — per-cell script 10 outputs
├── erp_core_ern_phase_search/         — script 12 phase/model/test artifacts
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

Any script that saves figures must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot`. Scripts 05, 06, 08, 10, and 12 already do this. Visualization functions in `eeg_bg/visualization/` return `plt.Figure` objects and never call `plt.show()`.

## Smoke Testing

`scripts/create_smoke_data.py` generates a tiny synthetic TUEP-format dataset (8 subjects, 90 s each, random noise) directly under the hardcoded path `D:/EEGdata/TUEP/v3.1.0` — run it once to populate synthetic data before using `configs/smoke_test.yaml`. `scripts/run_smoke_test.py` then runs scripts 01–07 end-to-end against `configs/smoke_test.yaml` with `--workers 1`, stopping at the first failing step and printing per-step timing — useful for verifying the full pipeline wiring after a code change without touching real TUEP data or the default cache/results directories.

## Tests

The default unit suite runs without real EDF data. Important fixture groups include:

- **`tests/conftest.py`** (root): `synthetic_epoch` (19-ch 1000-sample epoch with a single point source, SNR ≈ 50:1), `synthetic_epochs_batch` (batch of 5), `tmp_cache_dir`, `cfg` (deep-copied `BASE_CFG` dict).
- **`tests/test_features/conftest.py`**: `ch_names_19`, `sfreq`, `synthetic_epoch` (simple random — independent of root fixture), `pure_sine_signal` (10 Hz sine), `constant_signal`.
- **`tests/test_ml/conftest.py`**: `tiny_xgb_model` (10-feature, 5-estimator, session-scoped), `full_feature_xgb_model` (211-feature, session-scoped).
- **`tests/test_ml/test_cnn_*.py`** fixtures build a `tiny_cache` tmp_path fixture: minimal `epochs/{subject_id}/data.npz` files with `epochs`/`label`/`subject_id`/`split` keys, used to smoke-test `EEGEpochDataset`, `EEGNet`, and `train_cnn` without real data or GPU.
- **`tests/test_scripts/test_10_erp_core_benchmark.py`** covers the ERP channel partition, eight-cell phase/coherence config inheritance, supported Wiener modes, high-coherence experimental configs, semantic/numeric response-event pairing, and LRP sign convention. It does not run the full FIF benchmark.
- **`tests/test_scripts/test_11_erp_denoising_statistics.py`** covers participant pairing, bootstrap/TOST outputs, and report summaries without rerunning script 10.
- **`tests/test_scripts/test_12_erp_phase_search.py`** covers phase-grid endpoints, subject-safe folds, feature-cache identity, held-out policy helpers, and serial/subject-process worker behavior without running the full ERP search.

Test files under `tests/test_visualization/` must call `matplotlib.use("Agg")` **before** any `import matplotlib.pyplot` (same rule as scripts).

Integration tests (requiring real TUEP EDF files) should be marked `@pytest.mark.integration`.

`check_fixtures.py` (project root) is a standalone debug script that reconstructs fixture arrays and prints their shapes — useful for verifying conftest parity outside pytest.

## AGENTS.md maintenance

This file was created/updated by the ZCode `/init` command. To update it in the future:
- Run `/init` again — it will read the current file and make targeted edits.
- Or manually edit `AGENTS.md` directly.

Changes to the pipeline (new scripts, new config keys, new output directories, or changes to the feature vector) should be reflected here so future agents don't miss context.

## Non-obvious constraints and gotchas

- **Dataset selection**: `dataset.active` selects `dataset.tuep` or `dataset.tuab`. Inside the active block, `reference_scheme` and `montage_dir` must agree (`"ar"` ↔ `"01_tcp_ar"`, `"le"` ↔ `"02_tcp_le"`). TUAB reads only the first `max_recording_sec` (1200 s by default), requires all 19 standard channels, and must use cache/results paths separate from TUEP.
- **ERP-CORE is not selected through `dataset.active`**: scripts 10–12 are standalone ERP paths that inherit shared numerical Wiener defaults from `default.yaml` through `configs/erp_core_flankers.yaml`. Do not send ERP-CORE through scripts 01–08 or the TUEP/TUAB cache layout.
- **ERP comparisons must remain branch-fair**: filtering, resampling, response events, rejection decisions, epoch windows, and baselines must be shared across Raw/standard-ICA/Wiener. Only the denoising operation may differ. Preserve `_make_shared_epochs()` selection semantics when modifying script 10.
- **ERP metrics do not currently have clean ground truth**: standard ICA is only a baseline/reference. Treat SNR, baseline noise, waveform correlation, ICA-relative RMSE, ERN/LRP morphology, and downstream correctness classification as complementary evidence; no single value establishes denoising quality by itself.
- **ERP statistics reflect the available local subset**: script 10 processes each discovered participant independently and writes participant-level metrics, but `~/Data/ERP_CORE` may not contain the complete ERP-CORE release. Do not claim complete-dataset results from a partial local download.
- **`bandpass_filter()` in `epoch.py` is not used in the pipeline**: `load_edf` in `edf_reader.py` applies MNE's `raw.filter()` on the continuous signal before resampling. The standalone `bandpass_filter()` function in `epoch.py` (5th-order `sosfiltfilt`) is available for ad-hoc use but is never called by any script.
- **XGBoost `n_estimators` in `param_grid` is ignored during Phase 1**: `xgb_pipeline.py` overrides it to 500 for the grid search and uses early stopping in Phase 2 to find the final tree count. The entry in `configs/default.yaml` is documentation only.
- **`device="cuda"` automatically sets `n_jobs=1`**: `xgb_pipeline.py` detects the CUDA device setting and overrides GridSearchCV's `n_jobs` to avoid CUDA context conflicts across parallel workers. No manual change needed.
- **ICA fits on a 1 Hz high-pass copy but applies to 0.5 Hz data**: `fit_ica()` creates a temporary high-pass-filtered copy for FastICA convergence (MNE best practice), then applies the fitted mixing matrix to the original 0.5 Hz bandpass epochs. The `specific` output in the ICA cache is in the original 0.5 Hz bandpass domain. `max_iter` (default 1000) is read from `ica.max_iter` in the config and passed directly to the MNE ICA constructor — the scikit-learn default of 200 is too low for 19-channel EEG and causes frequent `ConvergenceWarning`.
- **Feature profiles must stay positionally stable**: `base211` uses `FEATURE_NAMES` (211 entries); `base211_conn80` appends `CONNECTIVITY_NAMES` (291 total). Changing `_STANDARD_19`, `SYMMETRIC_PAIRS`, connectivity-name order, or block order invalidates saved SHAP arrays. Script 06 checks both a profile-aware schema hash and feature dimension and points stale caches at `--force`.
- **`reference_scheme` filter is applied before any EDF is loaded**: Script 01 only processes recordings under the `montage_dir` subdirectory (default `01_tcp_ar`). Linked-ears (`tcp_le`) recordings are silently excluded.
- **Multiprocessing differs by script**: scripts 01–04 and 06 use `ProcessPoolExecutor`; script 12 uses an explicit spawn-based subject process pool when `--workers > 1`. On Windows, process workers re-import the module graph, so cap `--workers` on memory-constrained machines. Script 04 only runs V1/V2/V3 via `ThreadPoolExecutor` in legacy `--source recompute`; default cache mode parallelises recordings. Script 05 is sequential. Script 06 runs conditions sequentially and parallelises per-file feature extraction. Script 08 uses PyTorch `DataLoader(num_workers=...)`, defaulting to 0; its `--workers` override controls that loader setting, not a `ProcessPoolExecutor`.
- **Cache key composition for script 01**: the SHA-256 fingerprint covers the EDF path, active dataset, sample rate, bandpass, canonical channels, epoch length, artifact threshold, TUEP seizure buffer, and TUAB recording-duration cap. Scripts 02 and 03 still require `--force` after their own method settings change.
- **`--force` does not cascade and does not clean up orphaned files**: running `01_extract_epochs.py --force` does not force-rerun scripts 02–08; each script must be passed `--force` independently. Old `.npz` files at old key paths are orphaned on disk (not deleted). For script 06, `--force` bypasses the feature cache; model training and SHAP outputs regenerate on every invocation. Script 08 skips a condition when metrics exist unless `--force` is passed. Script 12 refuses a non-empty output directory without `--force` and otherwise recomputes its subject/phase caches as requested.
- **`create_smoke_data.py` writes to a hardcoded Windows-style path** (`D:/EEGdata/TUEP/v3.1.0`) — edit `DATA_ROOT` at the top of the script if generating smoke data on a non-Windows machine or a different data root, and make sure `configs/smoke_test.yaml`'s `paths.data_root` matches.

## Reference Documentation

- **`docs/Developer_qa.md`** — Detailed Q&A: epoch validity criteria (4 criteria with decision flow), `.npz` cache file schemas (exact keys/shapes/dtypes per cache family), feature extraction internals (band definitions, Hjorth formulae, spectral entropy formula, per-condition signal mapping). Its 211-dim feature-layout description is accurate again now that `master` has been reverted to this vector, but it predates the CNN pipeline and may still have stale label-dtype info — cross-check non-feature-layout numeric claims against the code before relying on them.
- **`docs/preprocessing.md`** — Every transformation applied to raw TUEP EDF recordings before epochs are written to `cache/epochs/`: dataset discovery, reference filtering, split assignment, EDF loading, channel normalisation, bandpass filtering, resampling, unit conversion, epoch slicing, and artifact rejection.
- **`docs/superpowers/plans/`** and **`docs/superpowers/specs/`** — implementation plans and design specs for major feature additions. Treat exact dimensions and line numbers as historical snapshots: the current profiles are `base211` (211) and `base211_conn80` (291), while wavelet/complexity/temporal blocks are not selected.
- **`eeg_bg/README.md`** — Core package API reference and implementation notes; verify unlisted specialist modules directly in their source and tests.
