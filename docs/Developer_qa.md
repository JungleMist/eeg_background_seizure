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
