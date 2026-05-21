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
| numpy | 1.26.4 | Array operations |
| scipy | 1.17.1 | Welch PSD, CSD, coherence |
| mne | 1.11.0 | EDF I/O, ICA |
| scikit-learn | 1.8.0 | Dataset splits |
| pandas | ≥2.0 | Result DataFrames, CSV output |
| matplotlib | ≥3.7 | All visualization |
| pyyaml | ≥6.0 | Configuration loading |
| tqdm | ≥4.0 | Pipeline progress bars |

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
│   └── visualization/
│       ├── filter_plots.py          ← |h(f)| and ∠h(f) curves
│       ├── coherence_plots.py       ← Coherence heatmaps, reduction boxplots, signal traces
│       ├── verification_plots.py    ← V1/V2/V3 summary figures, ICA vs Wiener comparison
│       └── waveform_plots.py        ← Stacked multi-channel waveform comparison
│
├── scripts/
│   ├── 01_extract_epochs.py         ← Step 1: EDF → cached epochs (.npz)
│   ├── 02_run_wiener.py             ← Step 2: cached epochs → Wiener decomposition
│   ├── 03_run_ica.py                ← Step 3: cached epochs → ICA decomposition
│   ├── 04_run_verification.py       ← Step 4: V1/V2/V3 verification → CSV reports
│   └── 05_run_visualization.py      ← Step 5: cached results → waveform figures
│
├── configs/
│   └── default.yaml                 ← All tunable parameters
│
├── tests/
│   ├── conftest.py                  ← Synthetic EEG fixtures (no real data needed)
│   ├── test_config.py
│   ├── test_io/                     ← dataset, edf_reader, annotation, cache
│   ├── test_preprocessing/          ← epoch slicing, reference detection
│   ├── test_decomposition/          ← wiener, wiener_scalar, ica
│   └── test_verification/           ← coherence V1, transitivity V2/V3
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
│   └── ica/                         ← ICA-cleaned epochs
│
├── docs/
│   └── Developer_qa.md              ← Developer Q&A (epoch validity, .npz schema, …)
│
├── results/                         ← Figure and CSV outputs, git-ignored
│   └── figures/                     ← PNG outputs from 05_run_visualization.py
│       └── {subject_id}/
│           └── waveform_comparison.png
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
    [io] extract_bckg_intervals  →  exclude seizure ±30 s buffers
    [preprocessing] slice_epochs  →  reject artifacts > 200 µV
          │
          ▼  cache/epochs/{subject_id}/{cache_key}.npz
          │     arrays: epochs (n, 19, 1000)  ch_names  label  subject_id  split
          │
     ┌────┴────┐
     ▼         ▼
  Script 02  Script 03
  Wiener      ICA
  decompose   fit + apply
     │         │
     ▼         ▼
cache/wiener_frequency/   cache/wiener_scalar/   cache/ica/
  arrays: specific  coherent  label  subject_id  split
          │
          ▼  Script 04
    [verification] V1 coherence reduction
    [verification] V2 transitivity constraint
    [verification] V3 frequency variation
          │
          ▼  results/
    v1_coherence.csv   v2_transitivity.csv   v3_frequency_variation.csv
          │
          ▼  Script 05
    [visualization] load epoch + wiener_frequency + ica caches
    [visualization] plot_multichannel_comparison  (19-ch stacked waveforms)
          │
          ▼  results/figures/{subject_id}/waveform_comparison.png
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
  bilateral_pairs:                 # 8 pairs processed by Wiener
    - [FP1, FP2]
    - [F3,  F4]
    - [F7,  F8]
    - [C3,  C4]
    - [T3,  T4]
    - [T5,  T6]
    - [P3,  P4]
    - [O1,  O2]
  midline: [Fz, Cz, Pz]           # pass-through (no bilateral partner)

wiener:
  nperseg:             1000        # Welch segment length (= epoch length → exact FFT match)
  coherence_threshold: 0.15        # skip pair if max coherence < this
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
| `nperseg = 1000` | = epoch length (8 s × 125 Hz) | Single-segment Welch → exact rectangular-window FFT; `h(f)` applied without interpolation |
| Coherence gate `0.15` | Skip pairs with low coherence | Avoids fitting noise; pairs without a true shared source are skipped |
| Rectangular window | `window='boxcar'` | Matches the FFT applied during filter application; no spectral leakage mismatch |
| Bilateral pairs only | 8 pairs, not all 171 channel pairs | Physically motivated: only contralateral homologous sites share a common source hypothesis |

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

### Complete Pipeline (one command chain)

```bash
conda activate eeg_pipeline
python scripts/01_extract_epochs.py
python scripts/02_run_wiener.py --mode frequency
python scripts/02_run_wiener.py --mode scalar
python scripts/03_run_ica.py
python scripts/04_run_verification.py
python scripts/05_run_visualization.py
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

**Current status:** 43 tests, all passing.

| Test module | Count | What is verified |
|-------------|-------|-----------------|
| `test_config.py` | 5 | YAML loading, path resolution |
| `test_io/` | 19 | Dataset traversal, EDF channel normalisation, annotation parsing, cache read/write |
| `test_preprocessing/` | 7 | Epoch slicing, artifact rejection, bandpass filter, reference detection |
| `test_decomposition/` | 8 | Wiener decomposition identity (`specific + coherent = raw`), scalar ablation, ICA shape |
| `test_verification/` | 4 | V1 coherence reduction, V2/V3 DataFrame structure |

### Synthetic Fixture Design

`tests/conftest.py` generates a known point-source model:

```
x_i(t) = gain_i · source(t) + noise_i(t)
```

- `source(t)`: 10 Hz sinusoid, amplitude 50 µV
- `gains`: uniform random in [0.5, 1.0], seed-controlled
- `noise`: Gaussian, σ = 1 µV (SNR ≈ 50 dB)
- `nperseg = 500` (half epoch length) to ensure multi-segment Welch estimation

At this SNR, the Wiener filter should recover `gain_i / gain_j ≈ h_ij` and reduce cross-channel coherence of the specific component to near zero.

**Integration tests** (marked `@pytest.mark.integration`) require actual TUEP EDF files and are excluded from normal runs with `-m "not integration"`.

---

## Package API

See [`eeg_bg/README.md`](eeg_bg/README.md) for the full module-level API reference.
