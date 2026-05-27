# Data Preprocessing

This document describes every transformation applied to raw TUEP EDF recordings
before epochs are written to the `cache/epochs/` directory by
`scripts/01_extract_epochs.py`.

---

## Workflow Overview

```
TUEP EDF files
    │
    ▼
1. Dataset discovery & subject index
    │
    ▼
2. Reference-scheme filtering
    │
    ▼
3. Train / val / test split assignment
    │
    ▼
4. EDF loading
    ├─ 4a. Channel selection & name normalisation
    ├─ 4b. Bandpass filtering (IIR)
    ├─ 4c. Resampling
    └─ 4d. Unit conversion  V → µV
    │
    ▼
5. Seizure-buffer exclusion  (annotation parsing)
    │
    ▼
6. Epoch slicing (non-overlapping, fixed-length)
    │
    ▼
7. Amplitude-based artifact rejection
    │
    ▼
cache/epochs/{label}_{subject_id}/{cache_key}.npz
```

---

## Step 1 — Dataset Discovery

**Module:** `eeg_bg/io/dataset.py` → `build_subject_index(cfg)`

The function traverses the TUEP directory tree rooted at `paths.data_root`:

```
data_root/
├── 00_epilepsy/
│   └── {subject_id}/{session_id}/{montage_dir}/*.edf
└── 01_no_epilepsy/
    └── {subject_id}/{session_id}/{montage_dir}/*.edf
```

Each discovered EDF file becomes one row in the subject index DataFrame with
columns: `subject_id`, `session_id`, `token_id`, `label`, `reference`,
`edf_path`.

**Label encoding** (hardcoded, must not change):

| Directory | `label` value |
|-----------|--------------|
| `00_epilepsy` | `0` |
| `01_no_epilepsy` | `1` |

---

## Step 2 — Reference-Scheme Filtering

**Module:** `eeg_bg/preprocessing/reference.py` → `filter_by_reference(index, scheme)`

Only recordings that live under the configured montage subdirectory are
processed. The active scheme is set by the paired config keys:

| `dataset.reference_scheme` | `dataset.montage_dir` |
|----------------------------|-----------------------|
| `"ar"` | `"01_tcp_ar"` |
| `"le"` | `"02_tcp_le"` |

Recordings under the other montage are silently excluded. The two keys must
always be changed together.

---

## Step 3 — Train / Val / Test Split Assignment

**Module:** `eeg_bg/io/dataset.py` → `assign_splits(index, cfg)`

Splits are assigned **at subject level** (all recordings of one subject land in
the same split) and are **stratified by class label** (each class is shuffled
and partitioned independently).

**Algorithm:**

For each class independently:

1. Shuffle the unique subject IDs with `numpy.random.default_rng(seed=42)`.
2. Assign the first $\lfloor n \times p_\text{train} \rfloor$ subjects to `train`.
3. Assign the next $\max(1,\, \lfloor n \times p_\text{val} \rfloor)$ subjects to `val`.
   The $\max(1, \cdot)$ floor guarantees at least one validation subject per class.
4. All remaining subjects go to `test`.

**Default fractions** (`configs/default.yaml`):

$$p_\text{train} = 0.70,\quad p_\text{val} = 0.10,\quad p_\text{test} \approx 0.20$$

Split assignments are stored in every `.npz` cache file at creation time.
Changing `random_seed`, `train`, or `val` requires a full cache rebuild
(`scripts/01_extract_epochs.py --force`).

---

## Step 4 — EDF Loading

**Module:** `eeg_bg/io/edf_reader.py` → `load_edf(edf_path, cfg)`

### 4a. Channel Selection & Name Normalisation

Raw TUEP channel names carry prefixes and suffixes (e.g. `"EEG FP1-REF"`).
These are stripped to obtain the bare electrode label before matching against
the 19-channel canonical list.

**Normalisation rules** (applied in order):

1. Strip leading whitespace.
2. Remove `"EEG "` prefix (case-insensitive).
3. Remove everything from the first `"-"` onward (`-REF`, `-AR`, `-LE`).
4. Convert to uppercase.

**Examples:**

| Raw EDF name | Normalised |
|---|---|
| `"EEG FP1-REF"` | `"FP1"` |
| `"EEG FZ-REF"` | `"FZ"` |
| `"EMG-REF"` | `"EMG"` |

Only channels whose normalised name appears in `channels.standard_19` are
retained; all others (ECG, EMG, etc.) are dropped.

**Standard 19 channels** (canonical order, positionally significant for the
feature vector):

```
FP1  FP2  F3   F4   F7   F8   C3   C4
T3   T4   T5   T6   P3   P4   O1   O2
Fz   Cz   Pz
```

### 4b. Bandpass Filtering

A **5th-order zero-phase Butterworth IIR** bandpass filter is applied to the
continuous multi-channel signal via MNE (`method="iir"`, `ftype="butter"`),
which calls `scipy.signal.sosfiltfilt` internally (forward + backward pass).

**Transfer function** of the $N$-th order Butterworth lowpass prototype,
normalised to cutoff $\omega_c = 1$ rad/s:

$$|H(j\omega)|^2 = \frac{1}{1 + \left(\dfrac{\omega}{\omega_c}\right)^{2N}}$$

With $N = 5$, this gives $-3\,\text{dB}$ at the cutoff frequencies and
$-100\,\text{dB/decade}$ roll-off in the stopband (per pole pair).

**Zero-phase implementation** via forward–backward filtering doubles the
effective filter order to $2N = 10$ and ensures zero group delay:

$$y = \mathrm{sosfiltfilt}(b, a,\; x)$$

**Default passband:** $[0.5,\; 40.0]\ \text{Hz}$

> **Why IIR, not FIR?**  A linear-phase FIR at 0.5 Hz lower cutoff would
> require ≈ 1691 taps at a 256 Hz input rate — longer than many short EDF
> segments. A 5th-order IIR achieves the same roll-off with negligible edge
> effects when applied zero-phase.

### 4c. Resampling

After filtering, the signal is resampled from the native EDF sampling frequency
to the target rate using MNE's `raw.resample()` (polyphase anti-aliasing):

$$f_s^\text{target} = 125\ \text{Hz}$$

$$n_\text{epoch} = f_s^\text{target} \times t_\text{epoch} = 125 \times 8 = 1000\ \text{samples}$$

### 4d. Unit Conversion

MNE returns raw EEG amplitudes in volts. They are converted to microvolts by:

$$x_{\mu\text{V}} = x_\text{V} \times 10^6$$

All downstream processing (artifact threshold, Wiener filter, feature
extraction) operates in µV.

---

## Step 5 — Seizure-Buffer Exclusion

**Module:** `eeg_bg/io/annotation.py` → `extract_bckg_intervals(csv_bi_path, cfg, recording_duration)`

### Annotation parsing

Each EDF recording has a companion `.csv_bi` annotation file listing
time-stamped segments labelled either `bckg` (background) or `seiz` (seizure).

**Base interval construction:**

- If `recording_duration` is provided (normal pipeline path): the full
  recording $[0,\; T_\text{rec}]$ is used as the single base interval,
  because TUEP v3.1.0 `csv_bi` files only annotate a short representative
  sample as `bckg` even when the rest of the recording is also background.
- If no `recording_duration` (legacy path): only rows explicitly labelled
  `bckg` are used as base intervals.

### Guard-zone formula

For each annotated seizure $(s_i,\, e_i)$, an exclusion zone is computed:

$$\text{exclude}_i = \bigl[\max(0,\; s_i - \Delta),\;\; e_i + \Delta\bigr]$$

where $\Delta = \texttt{seizure\_buffer\_sec} = 30\ \text{s}$ (default).

### Interval subtraction

Each exclusion zone is subtracted from the base intervals by interval
arithmetic:

```
for each base segment [b_s, b_e]:
    for each exclusion [ex_s, ex_e]:
        if [ex_s, ex_e] overlaps [b_s, b_e]:
            keep [b_s, ex_s]  if b_s < ex_s
            keep [ex_e, b_e]  if b_e > ex_e
        else:
            keep [b_s, b_e] unchanged
```

The result is a list of non-overlapping background intervals, free of any
seizure proximity.

---

## Step 6 — Epoch Slicing

**Module:** `eeg_bg/preprocessing/epoch.py` → `slice_epochs(...)`

Each background interval is tiled with a **non-overlapping, fixed-length**
sliding window:

$$\text{epoch length (samples)} = \lfloor t_\text{epoch} \times f_s \rfloor = \lfloor 8 \times 125 \rfloor = 1000$$

**Slicing algorithm:**

```
pos ← start_sample
while pos + epoch_len ≤ stop_sample:
    epoch ← data[:, pos : pos + epoch_len]
    [apply artifact check — see Step 7]
    pos ← pos + epoch_len          # stride == epoch_len → zero overlap
```

Any trailing samples that would form a shorter window are silently discarded.
Epochs do not overlap either within or across background intervals.

---

## Step 7 — Amplitude-Based Artifact Rejection

Applied inside the slicing loop immediately after each window is extracted.

**Rejection criterion:**

$$\max_{c \in \{1,\ldots,C\},\; t \in \{1,\ldots,T\}} |x_{c,t}| > \theta_\text{artifact}$$

where $\theta_\text{artifact} = 200\ \mu\text{V}$ (default) and $C = 19$
channels, $T = 1000$ samples.

An epoch is **kept** if and only if the peak absolute amplitude across all
channels and all time points does not exceed the threshold. Epochs that fail
are discarded without logging.

---

## Step 8 — Cache Key & Storage

**Module:** `eeg_bg/io/cache.py` → `make_cache_key(edf_path, start_sec, cfg)`

The cache key is a 16-character hex prefix of the SHA-256 digest of a
pipe-delimited string:

$$k = \text{SHA-256}\!\bigl(\underbrace{\text{edf\_path}}_{\text{abs. path}} \mid \underbrace{\text{start\_sec}}_{\text{4 d.p.}} \mid \underbrace{f_s^\text{target}}_{\text{125}} \mid \underbrace{\text{bandpass}}_{\text{[0.5, 40.0]}}\bigr)[\,{:}16\,]$$

Each accepted subject's epochs are saved to:

```
cache/epochs/{label_prefix}_{subject_id}/{cache_key}.npz
```

where `label_prefix` is `"00"` for epilepsy and `"01"` for control.

**NPZ contents:**

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `epochs` | `(N, 19, 1000)` | float64 | Accepted epochs in µV |
| `ch_names` | `(C,)` | str | Channel names in canonical order |
| `label` | scalar | int | 0 = epilepsy, 1 = control |
| `subject_id` | scalar | str | `"{label_prefix}_{subject_id}"` |
| `split` | scalar | str | `"train"` / `"val"` / `"test"` |

> **Cache-key scope:** The key encodes only `edf_path`, `target_sfreq`, and
> `bandpass`. Changing `epoch_length_sec`, `artifact_threshold_uv`, or
> `seizure_buffer_sec` does **not** generate a new key — the old `.npz` is
> silently reused unless `--force` is passed to `01_extract_epochs.py`.

---

## Parameter Reference

| Config key | Default | Effect if changed |
|---|---|---|
| `preprocessing.target_sfreq` | `125` Hz | Tier 1: re-run scripts 01+ |
| `preprocessing.bandpass` | `[0.5, 40.0]` Hz | Tier 1: re-run scripts 01+ |
| `preprocessing.epoch_length_sec` | `8.0` s | Tier 1: re-run 01+ (no new cache key — use `--force`) |
| `preprocessing.artifact_threshold_uv` | `200.0` µV | Tier 1: re-run 01+ (no new cache key — use `--force`) |
| `preprocessing.seizure_buffer_sec` | `30.0` s | Tier 1: re-run 01+ (no new cache key — use `--force`) |
| `split.train` / `split.val` | `0.70` / `0.10` | Tier 1: full cache rebuild |
| `split.random_seed` | `42` | Tier 1: full cache rebuild |
| `dataset.reference_scheme` | `"ar"` | Tier 1: must change with `montage_dir` |
| `dataset.montage_dir` | `"01_tcp_ar"` | Tier 1: must change with `reference_scheme` |
