# Developer Q&A

Answers to recurring "how does X work?" questions about the codebase.

---

## Q: What is a "valid epoch" in `scripts/01_extract_epochs.py`?

An epoch produced by `slice_epochs()` (`eeg_bg/preprocessing/epoch.py`) is kept
if and only if it satisfies **all four** criteria below.  Any window that fails
even one criterion is silently discarded.

### Criterion 1 — Must lie within a background interval

The recording is first partitioned into candidate intervals:

| Situation | Source of intervals |
|-----------|---------------------|
| `.csv_bi` annotation file exists alongside the EDF | `extract_bckg_intervals()` in `eeg_bg/io/annotation.py` — reads rows labelled `bckg` |
| No annotation file | The entire recording `(0.0, duration_sec)` is treated as one background interval |

Rows labelled `seiz` in the annotation file are **never** used as candidate
intervals.

### Criterion 2 — Must be outside the seizure safety buffer

Even inside a `bckg`-labelled segment, any portion within
`preprocessing.seizure_buffer_sec` seconds of a seizure boundary is carved
out.

```
excluded zone = [seiz_start − buffer, seiz_end + buffer]
                 (clamped to 0 at the left edge)
```

With the default config (`seizure_buffer_sec: 30.0`) this means no epoch may
start or end within 30 seconds of a seizure onset or offset, even if the
annotation file labels that stretch as background.

**Code location:** `eeg_bg/io/annotation.py`, lines 24–40.

### Criterion 3 — Must be exactly `epoch_length_sec` seconds long (non-overlapping)

`slice_epochs()` tiles each background interval with a fixed-stride, fixed-length
sliding window:

```python
epoch_len = int(epoch_length_sec * sfreq)   # default: 8 s × 125 Hz = 1000 samples
pos = start_sample
while pos + epoch_len <= stop_sample:        # strict ≤: partial tail discarded
    epoch = data[:, pos : pos + epoch_len]
    ...
    pos += epoch_len                         # stride = epoch_len (no overlap)
```

- The stride equals the window length → **zero overlap** between epochs.
- Any trailing samples that would form a shorter window are **discarded**.

**Code location:** `eeg_bg/preprocessing/epoch.py`, lines 26–31.

### Criterion 4 — Peak absolute amplitude ≤ `artifact_threshold_uv`

```python
if np.max(np.abs(epoch)) <= artifact_threshold_uv:   # default: 200 µV
    epochs.append(epoch.copy())
```

This is a single scalar check: the largest absolute value across **all channels
and all time samples** in the window must not exceed the threshold.  Epochs
that contain electrode pops, saturation, or gross movement artefacts are
rejected here.

**Code location:** `eeg_bg/preprocessing/epoch.py`, line 29.

---

### Decision flow summary

```
Recording
   │
   ├─ annotation exists? ──yes──► bckg intervals from .csv_bi
   │                       no──► [(0.0, duration)]
   │
   ▼
Apply seizure buffer exclusion (±30 s around every seiz event)
   │
   ▼
Slide 8-second, non-overlapping windows across remaining intervals
   │
   ├─ partial tail window? ──► discard
   │
   ▼
Peak |amplitude| ≤ 200 µV? ──no──► discard
   │ yes
   ▼
Valid epoch  →  stored in cache .npz
```

### Relevant config keys (`configs/default.yaml`)

| Key | Default | Role |
|-----|---------|------|
| `preprocessing.epoch_length_sec` | `8.0` | Window duration (Criterion 3) |
| `preprocessing.target_sfreq` | `125` Hz | Determines samples per epoch (1000) |
| `preprocessing.artifact_threshold_uv` | `200.0` µV | Amplitude gate (Criterion 4) |
| `preprocessing.seizure_buffer_sec` | `30.0` s | Safety margin around seizures (Criterion 2) |

### Relevant source files

| File | Responsibility |
|------|---------------|
| `eeg_bg/io/annotation.py` | Parses `.csv_bi`, applies seizure buffer (Criteria 1 & 2) |
| `eeg_bg/preprocessing/epoch.py` → `slice_epochs()` | Windowing and amplitude gate (Criteria 3 & 4) |
| `scripts/01_extract_epochs.py` | Orchestration; skips EDF if zero valid epochs result |

---

## Q: What information do the `.npz` cache files contain?

The pipeline writes three families of `.npz` files, one per processing stage.
All are created with `np.savez_compressed` and read back with
`np.load(path, allow_pickle=True)`.

---

### 1. Epoch cache — `cache/epochs/<subject_id>/<hash>.npz`

**Written by:** `scripts/01_extract_epochs.py`

| Key | Shape | dtype | Units / notes |
|-----|-------|-------|---------------|
| `epochs` | `(n_epochs, 19, 1000)` | float64 | µV. Axis 0 = epoch index; axis 1 = channel (order matches `ch_names`); axis 2 = time sample at 125 Hz |
| `ch_names` | `(19,)` | `<U…` (str) | Canonical channel names, e.g. `['FP1', 'FP2', …, 'Fz', 'Cz', 'Pz']` |
| `label` | scalar | str | Class label: `'epilepsy'` or `'control'` (from `dataset.classes` config) |
| `subject_id` | scalar | str | Patient folder name used as the unique subject identifier |
| `split` | scalar | str | Dataset partition: `'train'`, `'val'`, or `'test'` |

`n_epochs` varies per subject; 1000 samples = 8 s × 125 Hz.  
This cache is the **primary input** to every downstream script.

---

### 2. Wiener decomposition cache — `cache/wiener_<mode>/<subject_id>/<hash>.npz`

**Written by:** `scripts/02_run_wiener.py`  
**Two variants:** `wiener_frequency/` (default) and `wiener_scalar/`

| Key | Shape | dtype | Units / notes |
|-----|-------|-------|---------------|
| `specific` | `(n_epochs, 19, 1000)` | float64 | µV. Source-specific component — coherent signal removed |
| `coherent` | `(n_epochs, 19, 1000)` | float64 | µV. Cross-electrode coherent (shared-source) component |
| `label` | scalar | str | Copied from epoch cache |
| `subject_id` | scalar | str | Copied from epoch cache |
| `split` | scalar | str | Copied from epoch cache |

**Decomposition identity:** `specific + coherent == raw` is guaranteed by
construction for `frequency` mode (exact in the FFT domain).  For `scalar`
mode the identity holds approximately (scalar filter is the mean of `h(f)`
over the analysis band).

Midline channels (`Fz`, `Cz`, `Pz`) are not part of any bilateral pair, so
their `coherent` row is zero and `specific` equals the raw signal unchanged.

---

### 3. ICA cache — `cache/ica/<subject_id>/<hash>.npz`

**Written by:** `scripts/03_run_ica.py`

| Key | Shape | dtype | Units / notes |
|-----|-------|-------|---------------|
| `specific` | `(n_epochs, 19, 1000)` | float64 | µV. ICA-cleaned epochs (artifact components projected out) |
| `n_artifacts_removed` | scalar | int | Number of ICA components identified as artefacts and removed |
| `label` | scalar | str | Copied from epoch cache |
| `subject_id` | scalar | str | Copied from epoch cache |
| `split` | scalar | str | Copied from epoch cache |

The key is named `specific` (not `cleaned`) for consistency with the Wiener
cache so downstream loaders can use the same key regardless of which
preprocessing branch they are reading.

---

### Summary — shared metadata keys

All three cache families carry the same three metadata scalars:

| Key | Description |
|-----|-------------|
| `label` | Class membership (`'epilepsy'` / `'control'`) |
| `subject_id` | Subject identifier (matches the subdirectory name) |
| `split` | Train/val/test assignment (determined once in `01_extract_epochs.py`) |

### Reading a cache file

```python
import numpy as np

data = np.load("cache/epochs/sub_001/abc123.npz", allow_pickle=True)

epochs   = data["epochs"]      # (n_epochs, 19, 1000) float64 µV
ch_names = list(data["ch_names"])   # ['FP1', 'FP2', ..., 'Pz']
label    = str(data["label"])       # 'epilepsy'
subject  = str(data["subject_id"])  # 'sub_001'
split    = str(data["split"])       # 'train'
```

### Relevant source files

| File | Role |
|------|------|
| `scripts/01_extract_epochs.py` | Writes epoch cache |
| `scripts/02_run_wiener.py` | Writes Wiener cache (frequency & scalar modes) |
| `scripts/03_run_ica.py` | Writes ICA cache |
| `eeg_bg/decomposition/wiener.py` | Defines `WienerResult`; `specific` and `coherent` come from here |
| `eeg_bg/decomposition/ica.py` → `apply_ica()` | Returns cleaned `(n_epochs, 19, 1000)` array stored as `specific` |

---

## Q: Which features are extracted for XGBoost training in `scripts/06_train_xgboost.py`?

### Short answer

Each epoch is converted to a **171-dimensional vector** by `extract_epoch_features()` in `eeg_bg/features/extraction.py`.  The vector is built as **19 channels × 9 features** in a fixed channel-major order; names are in the public constant `FEATURE_NAMES`.

### Feature types (9 per channel)

#### 1–5: Relative band powers (`eeg_bg/features/band_power.py`)

Power is estimated with Welch's method (boxcar window, `nperseg=250` = 2 s at 125 Hz — same parameters used by the Wiener decomposition).  Each value is the fraction of total power within the analysis band (0.5–40 Hz) that falls inside the sub-band, computed with `np.trapezoid`:

| Index | Name | Band (Hz) |
|-------|------|-----------|
| 0 | `delta_power` | 0.5 – 4.0 |
| 1 | `theta_power` | 4.0 – 8.0 |
| 2 | `alpha_power` | 8.0 – 13.0 |
| 3 | `beta_power`  | 13.0 – 30.0 |
| 4 | `gamma_power` | 30.0 – 40.0 |

The five values sum to approximately 1 (small discrepancies at band boundaries from trapezoidal integration are possible).

#### 6–8: Hjorth parameters (`eeg_bg/features/hjorth.py`)

Time-domain complexity descriptors (Hjorth 1970).  Let `x` = signal, `x'` = first difference, `x''` = second difference:

| Index | Name | Formula |
|-------|------|---------|
| 5 | `hjorth_activity`   | `var(x)` |
| 6 | `hjorth_mobility`   | `sqrt(var(x') / (var(x) + ε))` |
| 7 | `hjorth_complexity` | `mobility(x') / (mobility(x) + ε)` |

`ε = 1e-30` guards against division by zero for flat signals.  Activity is in µV²; mobility and complexity are dimensionless ratios.

#### 9: Spectral entropy (`eeg_bg/features/spectral_entropy.py`)

Shannon entropy of the normalised PSD within the analysis band:

```
p_i = PSD(f_i) / Σ PSD(f_j)   (for f_j in [0.5, 40] Hz)
H   = -Σ p_i · log(p_i + ε)
```

Same Welch parameters as band power (`nperseg=250`, boxcar).  A pure sinusoid → H ≈ 0; white noise → H is maximised.

### Channel order and missing channels

Features iterate over the canonical 19-channel order from `configs/default.yaml`:

```
FP1, FP2, F3, F4, F7, F8, C3, C4, T3, T4, T5, T6, P3, P4, O1, O2, Fz, Cz, Pz
```

If a channel is absent from the epoch's `ch_names`, its 9-element slot is filled with zeros.  The vector length is always 171 regardless.

### Which signal is used per condition

`build_dataset()` selects the NPZ array key based on the `--condition` argument:

| Condition | Cache directory | NPZ array key | Signal |
|-----------|----------------|---------------|--------|
| `raw`    | `cache/epochs/`           | `epochs`   | Raw EEG |
| `wiener` | `cache/wiener_frequency/` | `specific` | Wiener source-specific component |
| `ica`    | `cache/ica/`              | `specific` | ICA-cleaned signal |

### Scaling and caching

Before XGBoost training, a `StandardScaler` is fit on the training feature matrix and applied to val and test sets.  The scaler is saved to `results/xgboost/{condition}/scaler.joblib`.

Extracted features are cached as `cache/features/{condition}_{split}.npz` (`X`, `y`, `subject_ids`) to avoid re-extraction on subsequent runs.  Pass `--force` to bypass this cache.

### Feature name index

`FEATURE_NAMES` (public list, `eeg_bg/features/extraction.py`) contains all 171 names in vector order, e.g.:

```
FP1_delta_power, FP1_theta_power, ..., FP1_spectral_entropy,
FP2_delta_power, ..., Pz_spectral_entropy
```

SHAP value arrays (shape `(n_test_epochs, 171)`) are indexed positionally against this list, so `FEATURE_NAMES` must remain stable between runs.

### Relevant source files

| File | Role |
|------|------|
| `eeg_bg/features/extraction.py` | `extract_epoch_features()`, `build_dataset()`, `FEATURE_NAMES` |
| `eeg_bg/features/band_power.py` | `relative_band_power()`, `BANDS` dict |
| `eeg_bg/features/hjorth.py` | `hjorth_parameters()` |
| `eeg_bg/features/spectral_entropy.py` | `spectral_entropy()` |
| `scripts/06_train_xgboost.py` | Orchestration: feature loading, scaling, training, SHAP |

---

## Q: What preprocessing is applied to raw EDF data?

### Overview

Raw EDF data passes through two sequential stages before features are extracted:

1. **Stage 1 — EDF loading and epoch caching** (`scripts/01_extract_epochs.py`): signal-level steps applied to the continuous recording.
2. **Stage 2 — Condition-specific decomposition** (`scripts/02_run_wiener.py`, `scripts/03_run_ica.py`): optional artefact removal applied to cached epochs for the `wiener` and `ica` conditions.

The `raw` condition uses Stage 1 output directly; `wiener` and `ica` add Stage 2 on top.

---

### Stage 1: EDF loading and epoch caching

Steps are executed in this order inside `scripts/01_extract_epochs.py`:

#### 1. Reference-scheme filtering (`preprocessing/reference.py`)

Before any signal is loaded, the subject index is filtered to keep only recordings stored under the `01_tcp_ar` montage directory (average reference).  Linked-ears (`tcp_le`) recordings are excluded.

Config key: `dataset.reference_scheme` (default `"ar"`).

#### 2. Subject-level train/val/test split (`io/dataset.py → assign_splits`)

Splits are assigned **before** any signal is loaded and **by subject** (all EDF files for a subject land in the same partition).  The split is stratified per class so both `epilepsy` and `control` appear in every partition:

- Epilepsy and control subjects are shuffled independently with `numpy.random.default_rng(seed=42)`.
- Each class contributes proportionally: 70 % train / 10 % val / 20 % test.
- At least 1 val subject per class is guaranteed (`max(1, int(n * 0.10))`).
- The `split` label is stored in every `.npz` cache file and used unchanged by all downstream scripts.

#### 3. Channel selection and name normalisation (`io/edf_reader.py → load_edf`)

The EDF is read with `mne.io.read_raw_edf(..., preload=True)`.  Each raw channel name is normalised:

```
"EEG FP1-REF"  →  strip "EEG " prefix  →  split on "-"  →  uppercase  →  "FP1"
```

Only channels whose normalised name matches one of the 19 standard names in `configs/default.yaml` are retained (`raw.pick()`).  The channel is then renamed to the canonical config name (e.g. `"FZ"` → `"Fz"`).  EDFs with no matching channels raise `ValueError` and are skipped.

#### 4. Bandpass filtering (`io/edf_reader.py → load_edf`)

MNE's `raw.filter(low=0.5, high=40.0, method="iir")` is applied to the continuous signal **before** resampling using a 5th-order zero-phase Butterworth IIR filter.  This removes slow drift (< 0.5 Hz) and high-frequency noise (> 40 Hz).  IIR is used rather than MNE's default FIR because the FIR tap count for a 0.5 Hz lower cutoff (≈ 1691 taps at 256 Hz) exceeds the length of short EDF recordings, producing filter-distortion warnings with no benefit to output quality.

Config keys: `preprocessing.bandpass` (default `[0.5, 40.0]`).

> **Note:** `eeg_bg/preprocessing/epoch.py` also defines a `bandpass_filter()` function (5th-order Butterworth, `sosfiltfilt`) that is imported by `scripts/01_extract_epochs.py` but is **not called** in the main pipeline — filtering is handled entirely by MNE inside `load_edf`.  The function is available for ad-hoc use only.

#### 5. Resampling (`io/edf_reader.py → load_edf`)

If the recording's native sample rate differs from 125 Hz, `raw.resample(125)` is called.  TUEP recordings are typically already at 250 Hz, so resampling occurs for most files.

Config key: `preprocessing.target_sfreq` (default `125`).

#### 6. Unit conversion (`io/edf_reader.py → load_edf`)

`raw.get_data()` returns signals in volts.  The array is multiplied by `1e6` to convert to **µV**.  All subsequent processing (artifact thresholds, feature values, Wiener filter coefficients) operates in µV.

#### 7. Background interval extraction (`io/annotation.py → extract_bckg_intervals`)

The continuous signal is not stored wholesale; only segments labelled as background are eligible for epoch slicing:

| Situation | Base intervals |
|-----------|----------------|
| `.csv_bi` file exists alongside EDF | Full recording `(0, duration)` minus seizure exclusion zones |
| No `.csv_bi` file | `[(0.0, duration)]` — entire recording treated as background |

Seizure exclusion zones are computed as:
```
excluded = [max(0, seiz_start − buffer),  seiz_end + buffer]
```
Each zone is subtracted from the base intervals via exact interval arithmetic, potentially splitting one interval into two.

Config key: `preprocessing.seizure_buffer_sec` (default `30.0` s).

#### 8. Epoch slicing (`preprocessing/epoch.py → slice_epochs`)

Each background interval is tiled with **non-overlapping** fixed-length windows:

```
epoch_len = 8 s × 125 Hz = 1000 samples
stride    = epoch_len  (no overlap)
partial tail windows → discarded
```

#### 9. Amplitude artifact rejection (`preprocessing/epoch.py → slice_epochs`)

Each candidate epoch is tested with a single scalar check:

```python
if np.max(np.abs(epoch)) <= 200.0:   # µV, all channels and samples
    keep epoch
```

Epochs containing electrode pops, saturation, or gross movement artefacts are silently discarded.  If zero epochs survive for an EDF file, the file is skipped entirely (no `.npz` written).

Config key: `preprocessing.artifact_threshold_uv` (default `200.0`).

#### 10. Cache write

Surviving epochs are saved as `cache/epochs/{label_prefix}_{subject_id}/{cache_key}.npz` with keys `epochs (n_epochs, 19, 1000) float64`, `ch_names`, `label`, `subject_id`, `split`.  The cache key is a 16-character SHA-256 prefix of `"{edf_path}|0.0000|125|[0.5, 40.0]"`.

---

### Stage 2: Condition-specific decomposition (applied to cached epochs)

Scripts 02 and 03 operate **on cached epochs only** — they never re-read the EDF.

#### Wiener condition (`scripts/02_run_wiener.py`)

For each epoch, the vector Wiener filter (`eeg_bg/decomposition/wiener.py`) separates each channel group (G1–G6) into:
- `coherent` — the cross-electrode shared-source component.
- `specific` — the residual after subtracting the coherent component.

Only the `specific` array is used as the signal for feature extraction.  Groups below the coherence threshold (default 0.15) are skipped; their `specific` equals `raw` unchanged.

#### ICA condition (`scripts/03_run_ica.py`)

FastICA with 19 components is fit on each epoch.  Components whose absolute correlation with the FP1 or FP2 reference channels exceeds `artifact_corr_threshold` (default 0.8) are zeroed out in component space before reconstruction.  The cleaned signal is stored as `specific`.

---

### Stage 3: Feature scaling (per XGBoost condition, `scripts/06_train_xgboost.py`)

After epoch-level features are extracted, a `StandardScaler` is fit on the **training set only** and applied to val and test sets (zero-mean, unit-variance per feature).  Scaling is applied to the 171-dimensional feature matrix, not to the raw signal.

---

### End-to-end summary

```
EDF file
  │
  ├─ reference-scheme filter (keep tcp_ar only)
  │
  ▼
load_edf()
  ├─ channel selection + name normalisation
  ├─ MNE bandpass FIR filter  [0.5 – 40 Hz]
  ├─ resample to 125 Hz
  └─ V → µV
  │
  ▼
extract_bckg_intervals()
  └─ full recording minus (seiz ± 30 s) exclusion zones
  │
  ▼
slice_epochs()
  ├─ non-overlapping 8 s windows (1000 samples, no overlap)
  └─ reject peak |amplitude| > 200 µV
  │
  ▼
cache/epochs/  ←─── raw condition reads here
  │
  ├─ [wiener] Wiener decomposition → specific component
  │     cache/wiener_frequency/  ←─── wiener condition reads here
  │
  └─ [ica]    FastICA artefact removal → specific component
              cache/ica/           ←─── ica condition reads here
  │
  ▼
extract_epoch_features()  →  171-dim vector per epoch
  │
  ▼
StandardScaler (fit on train)  →  XGBoost input
```

### Relevant source files

| File | Role |
|------|------|
| `scripts/01_extract_epochs.py` | Orchestrates Stage 1 |
| `eeg_bg/io/edf_reader.py` | Steps 3–6: channel normalisation, filter, resample, unit convert |
| `eeg_bg/io/annotation.py` | Step 7: background interval extraction |
| `eeg_bg/preprocessing/epoch.py` | Steps 8–9: epoch slicing and amplitude rejection |
| `eeg_bg/preprocessing/reference.py` | Step 1: reference-scheme filtering |
| `eeg_bg/io/dataset.py` | Step 2: subject-level stratified split |
| `eeg_bg/io/cache.py` | Step 10: cache key and load-or-compute helper |
| `eeg_bg/decomposition/wiener.py` | Stage 2 (wiener): vector Wiener filter |
| `eeg_bg/decomposition/ica.py` | Stage 2 (ica): FastICA artefact removal |
| `scripts/06_train_xgboost.py` | Stage 3: StandardScaler fit and application |
