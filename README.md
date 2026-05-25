# EEG Background Seizure — Wiener Feature Engineering Framework

A research framework for studying **physical point-source Wiener decomposition** of background EEG as a feature engineering method for epilepsy classification. Built on the [TUH EEG Epilepsy Corpus (TUEP) v3.1.0](https://isip.piconepress.com/projects/tuh_eeg/).

---

## Table of Contents

1. [Research Hypothesis](#research-hypothesis)
2. [Environment Setup](#environment-setup)
3. [Project Structure](#project-structure)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Physical Verification Experiments](#physical-verification-experiments)
8. [Visualization](#visualization)
9. [Testing](#testing)
10. [Package API](#package-api)

---

## Research Hypothesis

Inter-channel coherent components in background EEG arise from **non-neural physical interference** — volume conduction, reference electrode contamination, and shared environmental noise. Removing these via a frequency-dependent vector Wiener filter reveals the **specific** (locally generated, cortical) signal component.

**Decomposition identity (exact):**

```
x_i(t) = specific_i(t) + coherent_i(t)

coherent_i(t) = IFFT( Σ_j h_ij(f) · X_j(f) )
specific_i(t) = x_i(t) - coherent_i(t)
```

where `h_ij(f)` is the optimal Wiener filter coefficient estimated from the cross-power spectral density matrix of each bilateral electrode pair.

---

## Environment Setup

**Prerequisite:** Anaconda installed at `C:\ProgramData\anaconda3\`

```bash
# Activate the project environment
conda activate eeg_pipeline

# Install the package in editable mode (run once)
pip install -e .
```

**Core dependencies** (pinned in `requirements.txt`):

| Package | Version | Role |
|---------|---------|------|
| numpy | ≥2.0 | Array operations |
| scipy | 1.17.1 | Welch PSD, CSD, coherence |
| mne | 1.11.0 | EDF I/O, ICA |
| scikit-learn | 1.8.0 | Dataset splits |
| pandas | ≥2.0 | Result DataFrames, CSV output |
| matplotlib | ≥3.7 | All visualization |
| pyyaml | ≥6.0 | Configuration loading |
| tqdm | ≥4.0 | Pipeline progress bars |
| xgboost | 3.2.0 | XGBoost classifier, GridSearchCV, early stopping |
| shap | 0.51.0 | TreeExplainer SHAP values, beeswarm summary plots |

---

## Project Structure

```
D:\eeg_background_seizure\
│
├── eeg_bg/                          ← Core Python package
│   ├── config/
│   │   └── settings.py              ← YAML loader, path resolution
│   ├── io/
│   │   ├── dataset.py               ← Subject index builder, train/val/test splits
│   │   ├── edf_reader.py            ← EDF loading, channel selection, resampling
│   │   ├── annotation.py            ← .csv_bi parser, background interval extraction
│   │   └── cache.py                 ← .npz read/write, SHA-256 cache keys
│   ├── preprocessing/
│   │   ├── epoch.py                 ← 8-second epoch slicing, artifact rejection, bandpass
│   │   └── reference.py             ← Montage reference detection (AR vs LE)
│   ├── decomposition/
│   │   ├── wiener.py                ← Vector Wiener filter (core method)
│   │   ├── wiener_scalar.py         ← Fixed scalar ablation (comparison baseline)
│   │   └── ica.py                   ← FastICA decomposition via MNE (comparison method)
│   ├── verification/
│   │   ├── coherence.py             ← V1: pairwise coherence reduction
│   │   └── transitivity.py          ← V2: transitivity constraint, V3: frequency variation
│   ├── visualization/
│   │   ├── filter_plots.py          ← |h(f)| and ∠h(f) curves
│   │   ├── coherence_plots.py       ← Coherence heatmaps, reduction boxplots, signal traces
│   │   ├── verification_plots.py    ← V1/V2/V3 summary figures, ICA vs Wiener comparison
│   │   └── waveform_plots.py        ← Stacked multi-channel waveform comparison
│   ├── features/
│   │   ├── band_power.py            ← Relative band power (5 EEG bands, boxcar Welch)
│   │   ├── hjorth.py                ← Hjorth activity / mobility / complexity
│   │   ├── spectral_entropy.py      ← Shannon entropy of normalized PSD
│   │   └── extraction.py           ← extract_epoch_features(), build_dataset(), FEATURE_NAMES
│   └── ml/
│       ├── xgb_pipeline.py          ← train_xgboost(), subject_level_predict(), evaluate_subject_level()
│       └── shap_analysis.py         ← compute_shap_values(), aggregate/plot functions
│
├── scripts/
│   ├── 01_extract_epochs.py         ← Step 1: EDF → cached epochs (.npz)
│   ├── 02_run_wiener.py             ← Step 2: cached epochs → Wiener decomposition
│   ├── 03_run_ica.py                ← Step 3: cached epochs → ICA decomposition
│   ├── 04_run_verification.py       ← Step 4: V1/V2/V3 verification → CSV reports
│   ├── 05_run_visualization.py      ← Step 5: cached results → waveform + PSD figures
│   └── 06_train_xgboost.py         ← Step 6: feature extraction + XGBoost × 3 conditions + SHAP
│
├── configs/
│   └── default.yaml                 ← All tunable parameters (including ml: section)
│
├── tests/
│   ├── conftest.py                  ← Synthetic EEG fixtures (no real data needed)
│   ├── test_config.py
│   ├── test_io/                     ← dataset, edf_reader, annotation, cache
│   ├── test_preprocessing/          ← epoch slicing, reference detection
│   ├── test_decomposition/          ← wiener, wiener_scalar, ica
│   ├── test_verification/           ← coherence V1, transitivity V2/V3
│   ├── test_features/               ← band_power, hjorth, spectral_entropy, extraction (~16 tests)
│   └── test_ml/                     ← xgb_pipeline, shap_analysis (~12 tests)
│
├── notebooks/                       ← Interactive exploration (cells to be written)
│   ├── 01_data_exploration.ipynb
│   ├── 02_wiener_demo.ipynb
│   ├── 03_ica_comparison.ipynb
│   └── 04_verification_results.ipynb
│
├── cache/                           ← Auto-generated, git-ignored
│   ├── epochs/                      ← Per-subject .npz epoch files + index.csv
│   ├── wiener_frequency/            ← Wiener-decomposed epochs
│   ├── wiener_scalar/               ← Scalar-ablation epochs
│   ├── ica/                         ← ICA-cleaned epochs
│   └── features/                    ← Extracted feature matrices {condition}_{split}.npz
│
├── docs/
│   └── Developer_qa.md              ← Developer Q&A (epoch validity, .npz schema, …)
│
├── results/                         ← Figure and CSV outputs, git-ignored
│   ├── figures/{subject_id}/        ← PNG outputs from script 05
│   │   ├── waveform_comparison.png
│   │   └── psd_comparison.png
│   ├── verification/                ← CSV reports from script 04
│   │   ├── v1_coherence.csv
│   │   ├── v2_transitivity.csv
│   │   └── v3_frequency_variation.csv
│   └── xgboost/                     ← Outputs from script 06
│       ├── {raw,ica,wiener}/        ← model.joblib, scaler.joblib, metrics JSONs,
│       │                               predictions CSVs, SHAP values + plots
│       ├── comparison_summary.csv
│       └── shap_comparison.png
├── setup.py
├── requirements.txt
└── pytest.ini
```

---

## Data Flow

```
D:\EEGdata\TUEP\v3.1.0\
  00_epilepsy/{subject}/{session}/01_tcp_ar/*.edf   (+ .csv_bi)
  01_no_epilepsy/{subject}/{session}/01_tcp_ar/*.edf (+ .csv_bi)
          │
          ▼  Script 01
    [io] build_subject_index  →  filter AR montages  →  assign train/val/test
    [io] load_edf             →  bandpass filter  →  resample to 125 Hz
    [io] extract_bckg_intervals  →  full recording minus seizure ±30 s buffers
    [preprocessing] slice_epochs  →  reject artifacts > 200 µV
          │
          ▼  cache/epochs/{subject_id}/{cache_key}.npz
          │     arrays: epochs (n, 19, 1000)  ch_names  label  subject_id  split
          │
     ┌────┴──────────────────────────────┬─────────────────────────────────────┐
     ▼                                   ▼                                     ▼
  Script 02                          Script 03                             Script 04
  Wiener                                ICA                        [verification] V1/V2/V3
  decompose                         fit + apply                    (re-runs decompose_epoch
     │                                   │                         inline; reads only epochs)
     ▼                                   ▼                                     │
cache/wiener_frequency/   cache/wiener_scalar/   cache/ica/                    ▼
  arrays: specific  coherent  label  subject_id  split               results/verification/
          │                                          │               v1_coherence.csv
          └────────────────────┬─────────────────────┘               v2_transitivity.csv
                               │                                     v3_frequency_variation.csv
               ┌───────────────┴───────────────┐
               ▼                               ▼
           Script 05                       Script 06
    [visualization] load epoch       [features] extract_epoch_features
      + wiener_frequency + ica          × 3 conditions (171 features/epoch)
    plot_multichannel_comparison       [ml] train_xgboost (GridSearchCV
    plot_psd_comparison                     + early stopping on val)
               │                       subject_level_predict
               ▼                       evaluate_subject_level (AUROC/F1/Acc)
    results/figures/                   compute_shap_values
      {subject_id}/                             │
        waveform_comparison.png                 ▼
        psd_comparison.png             results/xgboost/
                                         {condition}/model.joblib
                                         test_metrics.json
                                         shap_summary.png
                                         comparison_summary.csv
                                         shap_comparison.png
```

### Cache File Schema

Every `.npz` at `cache/epochs/{subject_id}/{cache_key}.npz` contains:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `epochs` | `(n_epochs, 19, 1000)` | float64 | Signal in µV |
| `ch_names` | `(19,)` | str | Channel names |
| `label` | scalar | int | 0 = epilepsy, 1 = control |
| `subject_id` | scalar | str | Anonymized patient ID |
| `split` | scalar | str | `train` / `val` / `test` |

Wiener/ICA output `.npz` files have the same structure but replace `epochs` with `specific` (and `coherent` for Wiener).

---

## Configuration

All parameters live in `configs/default.yaml`. Edit this file to change dataset paths, preprocessing settings, or method hyperparameters.

```yaml
paths:
  data_root: "D:/EEGdata/TUEP/v3.1.0"   # ← Set to your TUEP root
  cache_dir:   "cache"                    # relative to project root
  results_dir: "results"

dataset:
  reference_scheme: "ar"          # only process 01_tcp_ar montages
  montage_dir:      "01_tcp_ar"

split:
  train: 0.70
  val:   0.10
  test:  0.20
  random_seed: 42                  # subject-level split, reproducible

preprocessing:
  target_sfreq:         125        # Hz — resample target
  bandpass:             [0.5, 40.0]
  epoch_length_sec:     8.0        # → 1000 samples per epoch at 125 Hz
  artifact_threshold_uv: 200.0     # epochs exceeding this are discarded
  seizure_buffer_sec:   30.0       # exclusion zone around annotated seizures

channels:
  channel_groups:                  # G1–G6 movement-artifact conduction paths
    - [FP1, FP2]                   # G1 – symmetric facial (frontalis)
    - [F7,  T3]                    # G2 – left SCM
    - [T3,  T5, O1]                # G3 – left posterior neck (3-ch chain)
    - [O1,  O2]                    # G4 – bilateral occipitalis
    - [F8,  T4]                    # G5 – right SCM
    - [T4,  T6, O2]                # G6 – right posterior neck (3-ch chain)
  passthrough: [F3, F4, C3, C4, P3, P4, Fz, Cz, Pz]  # never filtered

wiener:
  nperseg:             250         # Welch segment length → 4 segments/epoch, 0.5 Hz resolution
  freq_resolution_hz:  0.5         # used by V1 coherence estimation
  coherence_threshold: 0.15        # skip group if max pairwise coherence < this
  freq_band:           [0.5, 40.0] # coherence gate + filter estimation band

ica:
  n_components:          19
  artifact_corr_threshold: 0.8     # correlation with FP1/FP2 proxy → artifact
  random_state:          42

verification:
  v2_transitivity_amp_threshold:   0.1    # ε_amp pass criterion
  v2_transitivity_phase_threshold: 0.392  # π/8 rad phase pass criterion
  v3_freq_variation_threshold:     0.20   # (max-min)/mean |h(f)| > 20% → frequency-dependent
```

### Key Design Decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| `nperseg = 250` | = epoch / 4 (0.5 Hz resolution) | Four-segment Welch → variance reduction in cross-PSD; filter interpolated to full rfft grid; `specific + coherent = raw` guaranteed algebraically |
| Coherence gate `0.15` | Skip groups with low coherence | Avoids fitting noise; groups without a true shared source are skipped |
| Rectangular window | `window='boxcar'` | Matches the FFT applied during filter application; no spectral leakage mismatch |
| Channel groups G1–G6 | 6 groups (2- and 3-channel), not all pairs | Physically motivated: each group models a specific movement-artifact conduction pathway |

---

## Running the Pipeline

> **Always activate the environment first:**
> ```bash
> conda activate eeg_pipeline
> ```

### Step 1 — Extract Background Epochs

Traverses the TUEP dataset, filters `.csv_bi` annotations for background segments, rejects artifacts, and caches clean 8-second epochs to disk.

```bash
python scripts/01_extract_epochs.py
# or with custom config:
python scripts/01_extract_epochs.py --config configs/default.yaml
# force recompute even if cache exists:
python scripts/01_extract_epochs.py --force
```

**Output:** `cache/epochs/index.csv` (subject index with splits) and one `.npz` per EDF file under `cache/epochs/{subject_id}/`.

---

### Step 2 — Run Wiener Decomposition

Loads cached epochs and applies the vector Wiener filter to each bilateral pair.

```bash
# Frequency-dependent Wiener (default, core method):
python scripts/02_run_wiener.py --mode frequency

# Fixed scalar ablation (comparison baseline):
python scripts/02_run_wiener.py --mode scalar

# Force recompute:
python scripts/02_run_wiener.py --mode frequency --force
```

**Output:** `cache/wiener_frequency/` or `cache/wiener_scalar/` mirroring the epoch cache layout.

**What `--mode frequency` does:** For each bilateral pair, estimates `S(f)` (cross-PSD matrix) via Welch, solves `h(f) = S_ref(f)^{-1} · s_cross(f)` at each frequency bin, applies in the frequency domain. Guarantees `specific + coherent = raw` exactly.

**What `--mode scalar` does:** Same filter estimation, but collapses `h(f)` to a single complex scalar per reference channel (mean over frequency band), then applies in the time domain. Ablation baseline to test whether frequency-dependence matters.

---

### Step 3 — Run ICA Decomposition

Fits FastICA on all epochs for each subject, automatically identifies artifact components by correlation with the FP1/FP2 frontal proxy, and removes them.

```bash
python scripts/03_run_ica.py
python scripts/03_run_ica.py --force   # recompute
```

**Output:** `cache/ica/` with cleaned epochs.

**Artifact detection logic:** Each ICA component's time course is correlated with `mean(FP1, FP2)`. Components with `|r| > artifact_corr_threshold` (default 0.8) are excluded.

---

### Step 4 — Run Physical Verification

Runs V1/V2/V3 experiments and saves CSV result tables.

```bash
python scripts/04_run_verification.py
```

**Output** in `results/`:

| File | Columns | Purpose |
|------|---------|---------|
| `v1_coherence.csv` | `subject_id, epoch_idx, ch_i, ch_j, coh_pre, coh_post, reduction` | Coherence purity before/after decomposition |
| `v2_transitivity.csv` | `subject_id, epoch_idx, triplet, eps_amp, eps_phase` | Single-source transitivity constraint errors |
| `v3_frequency_variation.csv` | `subject_id, epoch_idx, pair, channel, freq_variation, amp_mean, amp_std` | Frequency-dependence of `\|h(f)\|` |

---

### Step 5 — Generate Waveform Figures

Loads one representative epoch per subject from the epoch, Wiener, and ICA caches
and saves a stacked 19-channel waveform comparison (Raw | Wiener specific | ICA cleaned).
No decomposition is re-run — arrays are read directly from cached `.npz` files.

```bash
# All subjects, epoch 0 (default):
python scripts/05_run_visualization.py

# Limit to first 5 subjects, choose epoch 2:
python scripts/05_run_visualization.py --n-subjects 5 --epoch-idx 2
```

**Output:** `results/figures/{subject_id}/waveform_comparison.png`

Each figure contains up to three panels (panels are omitted if the corresponding cache
does not yet exist):

| Panel | Source | Content |
|-------|--------|---------|
| Raw | `cache/epochs/` | Unprocessed bandpass-filtered EEG |
| Wiener specific | `cache/wiener_frequency/` | Coherent component removed |
| ICA cleaned | `cache/ica/` | Artifact components projected out |

---

---

### Step 6 — Train XGBoost Classifiers + SHAP Analysis

Extracts 171 handcrafted features per epoch (relative band power × 5 bands,
Hjorth parameters × 3, spectral entropy — for each of 19 channels) under three
preprocessing conditions, trains an XGBoost classifier per condition via 5-fold
GridSearchCV + early-stopping refit on the validation set, evaluates at subject
level, and generates SHAP feature importance comparison figures.

```bash
# All three conditions (raw / ica / wiener) — default
python scripts/06_train_xgboost.py

# Single condition
python scripts/06_train_xgboost.py --condition wiener

# Force re-extraction of features (ignore feature cache)
python scripts/06_train_xgboost.py --force
```

**Output:** `results/xgboost/{condition}/` — model, metrics, per-subject
predictions, SHAP values and plots; `results/xgboost/shap_comparison.png` —
2 × 3 publication figure comparing SHAP band/channel importance across the three
conditions (Raw C | ICA B | Wiener A).

---

### Pipeline Dependency Graph

Script 01 is the only strict prerequisite. Everything downstream of it has no mutual dependency and can run in parallel.

```
EDF files
    │
    ▼
  01_extract_epochs              →  cache/epochs/
    │
    ├──────────────────────────────────┬──────────────────────────────────┐
    ▼                                  ▼                                  ▼
  02_run_wiener                     03_run_ica                  04_run_verification
  cache/wiener_frequency|scalar/    cache/ica/                  results/verification/
    │                                  │                        [independent of 02/03;
    └──────────────┬───────────────────┘                        re-runs decompose_epoch
                   │                                             inline from cache/epochs/]
          ┌────────┴────────┐
          ▼                 ▼
  05_run_visualization   06_train_xgboost
  results/figures/       results/xgboost/
```

**Script 06 per-condition dependencies:**

| `--condition` | Requires |
|---------------|----------|
| `raw` | 01 only |
| `wiener` | 01 + 02 |
| `ica` | 01 + 03 |
| `all` (default) | 01 + 02 + 03 |

**Script 05** reads `cache/wiener_frequency/` and `cache/ica/` optionally — panels are omitted if those caches are absent, so it can be run after 01 alone for raw-only figures.

---

### Complete Pipeline (sequential)

```bash
conda activate eeg_pipeline
python scripts/01_extract_epochs.py
python scripts/02_run_wiener.py --mode frequency
python scripts/02_run_wiener.py --mode scalar
python scripts/03_run_ica.py
python scripts/04_run_verification.py
python scripts/05_run_visualization.py
python scripts/06_train_xgboost.py
```

### Complete Pipeline (parallel)

```bash
conda activate eeg_pipeline
python scripts/01_extract_epochs.py

# Steps 2–4: all read only from cache/epochs/ — run concurrently
python scripts/02_run_wiener.py --mode frequency &
python scripts/02_run_wiener.py --mode scalar    &
python scripts/03_run_ica.py                     &
python scripts/04_run_verification.py            &
wait

# Steps 5–6: no dependency on each other — run concurrently
python scripts/05_run_visualization.py &
python scripts/06_train_xgboost.py     &
wait
```

---

## Physical Verification Experiments

### V1 — Decomposition Purity

**Hypothesis:** After Wiener decomposition, the specific components of formerly coherent bilateral pairs should have near-zero inter-channel coherence.

**Metric:** `reduction = coh_pre - coh_post` for all channel pairs. Positive reduction = coherence was removed. Passing criterion: `coh_post < 0.05` for bilateral pairs.

### V2 — Transitivity Constraint

**Hypothesis:** If channels i, j, k all receive signal from a single physical point source, the Wiener filter coefficients must satisfy:

```
|h_ij(f)| · |h_jk(f)| ≈ |h_ik(f)|          (amplitude transitivity)
∠h_ij(f) + ∠h_jk(f) ≈ ∠h_ik(f)  (mod 2π)  (phase transitivity)
```

**Metric:** `eps_amp` and `eps_phase` per channel triplet. Passing criteria: `eps_amp < 0.1`, `eps_phase < π/8 rad (0.392)`.

### V3 — Frequency Variation

**Hypothesis:** The physical medium (tissue, skull) imposes frequency-dependent attenuation. A truly frequency-dependent model should show significant variation in `|h(f)|` across the band.

**Metric:** `freq_variation = (max - min) / mean` of `|h(f)|` within the analysis band. Values > 20% support the frequency-dependent model over the scalar ablation baseline.

---

## Visualization

All visualization functions return `plt.Figure` objects and never call `plt.show()`. The caller is responsible for saving or displaying.

```python
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.visualization.filter_plots import plot_all_pairs_response
from eeg_bg.visualization.coherence_plots import (
    plot_coherence_matrix, plot_signal_decomposition
)
from eeg_bg.visualization.verification_plots import (
    plot_v2_transitivity, plot_v3_frequency_variation,
    plot_ica_vs_wiener_coherence
)
from eeg_bg.visualization.waveform_plots import plot_multichannel_comparison
import numpy as np

cfg = load_config()

# Load a cached epoch
data = np.load("cache/epochs/aaaaabhz/abc123.npz", allow_pickle=True)
epoch = data["epochs"][0]
ch_names = list(data["ch_names"])

# Decompose
result = decompose_epoch(epoch, ch_names, cfg, subject_id="aaaaabhz", epoch_idx=0)

# Filter frequency response for all bilateral pairs
fig = plot_all_pairs_response(result, save_path="results/filter_response.png")

# Raw / coherent / specific signal traces for FP1
fig = plot_signal_decomposition(result, ch_name="FP1", time_window=(0, 4))
fig.savefig("results/signal_decomp_FP1.png", dpi=150)

# Coherence matrix before/after
from eeg_bg.verification.coherence import compute_pairwise_coherence
coh_pre  = compute_pairwise_coherence(result.raw,      cfg["preprocessing"]["target_sfreq"],
                                       cfg["wiener"]["nperseg"], cfg["wiener"]["freq_band"])
coh_post = compute_pairwise_coherence(result.specific, cfg["preprocessing"]["target_sfreq"],
                                       cfg["wiener"]["nperseg"], cfg["wiener"]["freq_band"])
fig = plot_coherence_matrix(coh_pre, coh_post, ch_names, title="Subject aaaaabhz, Epoch 0")
fig.savefig("results/coherence_matrix.png", dpi=150)

# V2/V3 from verification CSVs
import pandas as pd
v2_df = pd.read_csv("results/v2_transitivity.csv")
v3_df = pd.read_csv("results/v3_frequency_variation.csv")
plot_v2_transitivity(v2_df).savefig("results/v2_transitivity.png", dpi=150)
plot_v3_frequency_variation(v3_df).savefig("results/v3_freq_variation.png", dpi=150)

# Stacked 19-channel waveform comparison (Raw | Wiener specific | ICA cleaned)
# Load arrays directly from caches — no re-decomposition needed
import numpy as np
edata = np.load("cache/epochs/aaaaabhz/abc123.npz", allow_pickle=True)
raw   = edata["epochs"][0]                                   # (19, 1000)
ch_names = list(edata["ch_names"])

wdata          = np.load("cache/wiener_frequency/aaaaabhz/abc123.npz", allow_pickle=True)
wiener_specific = wdata["specific"][0]                        # (19, 1000)

idata        = np.load("cache/ica/aaaaabhz/abc123.npz", allow_pickle=True)
ica_specific = idata["specific"][0]                           # (19, 1000)

fig = plot_multichannel_comparison(
    raw, wiener_specific, ica_specific,
    ch_names, sfreq=125.0, title="aaaaabhz  epoch 0"
)
fig.savefig("results/figures/aaaaabhz/waveform_comparison.png", dpi=150, bbox_inches="tight")
```

---

## Testing

The test suite uses **synthetic data only** — no real EDF files required on any machine.

```bash
# Run all unit tests
conda run -n eeg_pipeline pytest tests/ -v

# Exclude integration tests (default for CI)
conda run -n eeg_pipeline pytest tests/ -m "not integration" -v

# Run a specific module
conda run -n eeg_pipeline pytest tests/test_decomposition/ -v

# Quiet summary
conda run -n eeg_pipeline pytest tests/ -m "not integration" -q
```

**Current status:** ~96 tests, all passing.

| Test module | Count | What is verified |
|-------------|-------|-----------------|
| `test_config.py` | 5 | YAML loading, path resolution |
| `test_io/` | 23 | Dataset traversal, EDF channel normalisation, annotation parsing (incl. full-recording semantics), cache read/write |
| `test_preprocessing/` | 7 | Epoch slicing, artifact rejection, bandpass filter, reference detection |
| `test_decomposition/` | 8 | Wiener decomposition identity (`specific + coherent = raw`), scalar ablation, ICA shape |
| `test_verification/` | 4 | V1 coherence reduction, V2/V3 DataFrame structure |
| `test_visualization/` | 4 | PSD figure shape, 3-column grid layout, config channel defaults |
| `test_features/` | ~20 | Band power normalisation, Hjorth correctness, spectral entropy range, FEATURE_NAMES length/uniqueness, build_dataset split filtering |
| `test_ml/` | ~12 | subject_level_predict aggregation, evaluate_subject_level keys, SHAP shape/non-negativity, plot functions create files |

### Synthetic Fixture Design

`tests/conftest.py` generates a known point-source model:

```
x_i(t) = gain_i · source(t) + noise_i(t)
```

- `source(t)`: broadband Gaussian noise, σ = 50 µV
- `gains`: uniform random in [0.5, 1.0], seed-controlled
- `noise`: independent Gaussian per channel, σ = 1 µV (SNR ≈ 50 dB)
- `nperseg = 500` in `BASE_CFG` (2 segments per epoch) to ensure multi-segment Welch estimation for coherence tests

At this SNR, the Wiener filter should recover `gain_i / gain_j ≈ h_ij` and reduce cross-channel coherence of the specific component to near zero.

**Integration tests** (marked `@pytest.mark.integration`) require actual TUEP EDF files and are excluded from normal runs with `-m "not integration"`.

---

## Package API

See [`eeg_bg/README.md`](eeg_bg/README.md) for the full module-level API reference.
