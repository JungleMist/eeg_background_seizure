# `eeg_bg` — Package API Reference

The `eeg_bg` package is organized into five sub-packages. All public functions are importable from their respective module paths. This document describes every module's purpose, public API, and internal design notes.

---

## Table of Contents

1. [`config`](#config) — Configuration loading
2. [`io`](#io) — Data I/O and caching
3. [`preprocessing`](#preprocessing) — Epoch extraction and filtering
4. [`decomposition`](#decomposition) — Wiener and ICA signal decomposition
5. [`verification`](#verification) — Physical validation experiments
6. [`visualization`](#visualization) — Figure generation

---

## `config`

### `eeg_bg.config.settings`

```python
from eeg_bg.config.settings import load_config
```

#### `load_config(config_path="configs/default.yaml") -> dict`

Loads a YAML configuration file and resolves relative paths (`cache_dir`, `results_dir`) to absolute paths anchored at the project root (the directory containing `configs/`).

```python
cfg = load_config()                          # default path
cfg = load_config("configs/custom.yaml")    # custom path
cfg = load_config(Path("configs/custom.yaml"))

# Access values
sfreq  = cfg["preprocessing"]["target_sfreq"]   # 125.0
groups = cfg["channels"]["channel_groups"]        # list of channel groups (G1–G6)
cache  = cfg["paths"]["cache_dir"]               # absolute path string
```

**Returns:** Plain `dict` — pass directly to all downstream functions that accept `cfg`.

---

## `io`

### `eeg_bg.io.dataset`

```python
from eeg_bg.io.dataset import build_subject_index, assign_splits
```

#### `build_subject_index(cfg: dict) -> pd.DataFrame`

Recursively traverses `cfg["paths"]["data_root"]` to find all `.edf` files that live under the configured `montage_dir` (e.g., `01_tcp_ar`). Returns one row per EDF file.

**Returns:** DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | 8-character anonymized patient ID (e.g., `aaaaabhz`) |
| `session_id` | str | Session identifier (e.g., `s002`) |
| `token_id` | str | File segment within session (e.g., `t000`) |
| `label` | int | 0 = epilepsy (`00_epilepsy`), 1 = control (`01_no_epilepsy`) |
| `reference` | str | Montage type inferred from directory name |
| `edf_path` | str | Absolute path to the `.edf` file |

#### `assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame`

Performs a **subject-level** random train/val/test split. All EDF files belonging to the same subject are assigned to the same split — no data leakage across splits at the recording level.

Split ratios and random seed come from `cfg["split"]` (default: 70/10/20, seed 42).

**Returns:** Input DataFrame with an additional `split` column (`"train"` / `"val"` / `"test"`).

---

### `eeg_bg.io.edf_reader`

```python
from eeg_bg.io.edf_reader import load_edf
```

#### `load_edf(edf_path: Path, cfg: dict) -> tuple[np.ndarray, list[str], float]`

Loads a single EDF file using MNE, selects the 19 standard channels from `cfg["channels"]["standard_19"]`, applies a Butterworth bandpass filter, and resamples to `cfg["preprocessing"]["target_sfreq"]`.

**Returns:** `(data, ch_names, sfreq)`
- `data`: `(19, n_times)` array in **µV**
- `ch_names`: list of 19 channel name strings
- `sfreq`: actual sampling frequency after resampling

**Note:** MNE loads EEG in volts internally; `load_edf` multiplies by `1e6` before returning.

---

### `eeg_bg.io.annotation`

```python
from eeg_bg.io.annotation import extract_bckg_intervals
```

#### `extract_bckg_intervals(csv_bi_path: Path, cfg: dict, recording_duration: float | None = None) -> list[tuple[float, float]]`

Parses a `.csv_bi` annotation file (TUEP binary term format) and returns clean background time intervals, with seizure exclusion zones removed.

**When `recording_duration` is provided (recommended):** Background is defined as the full recording `(0, recording_duration)` minus any `seiz`-annotated segments ± `seizure_buffer_sec`. This is the correct semantics for TUEP v3.1.0, whose csv_bi files annotate only a short representative sample as `bckg` even when the rest of the recording is also background.

**When `recording_duration` is None (legacy):** Background is derived solely from the explicit `bckg` rows in the csv_bi file — any `bckg` segment overlapping a `seiz` label ± `seizure_buffer_sec` (default 30 s) is clipped or removed entirely.

**Returns:** List of `(start_sec, stop_sec)` tuples suitable for passing to `slice_epochs`.

**Format of `.csv_bi`:**
```
# version = csv_v1.0.0
# ...
channel,start_time,stop_time,label,confidence
TERM,0.0000,120.0000,bckg,1.0000
TERM,120.0000,135.0000,seiz,1.0000
TERM,135.0000,600.0000,bckg,1.0000
```

---

### `eeg_bg.io.cache`

```python
from eeg_bg.io.cache import make_cache_key, load_or_compute
```

#### `make_cache_key(edf_path: Path, start_sec: float, cfg: dict) -> str`

Computes a 16-character hexadecimal SHA-256 hash over the string `"{edf_path}|{start_sec:.4f}|{sfreq}|{bandpass}"`. Changing any preprocessing parameter produces a different cache key, forcing recomputation.

```python
key = make_cache_key(Path("path/to/file.edf"), 0.0, cfg)  # e.g., "a3f7c2b1d9e04521"
```

#### `load_or_compute(cache_path: Path, compute_fn: Callable, force_recompute: bool = False) -> dict`

Checks whether `cache_path` exists. If it does (and `force_recompute` is False), loads and returns its contents as a plain `dict`. Otherwise calls `compute_fn()`, saves the result as a compressed `.npz`, and returns the dict.

```python
result = load_or_compute(
    cache_path=Path("cache/epochs/subj/key.npz"),
    compute_fn=lambda: {"epochs": my_array, "label": np.array(0)},
)
```

---

## `preprocessing`

### `eeg_bg.preprocessing.epoch`

```python
from eeg_bg.preprocessing.epoch import bandpass_filter, slice_epochs
```

#### `bandpass_filter(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray`

Applies a zero-phase 5th-order Butterworth bandpass filter along the last axis using `scipy.signal.sosfiltfilt`. Safe for multichannel arrays.

```python
filtered = bandpass_filter(data, sfreq=125.0, low=0.5, high=40.0)
# data: (n_ch, n_times) → filtered: (n_ch, n_times)
```

#### `slice_epochs(data, sfreq, intervals, epoch_len_sec, artifact_threshold_uv) -> np.ndarray`

Cuts `data` into non-overlapping fixed-length epochs restricted to `intervals`. Discards any epoch where `max(|signal|) > artifact_threshold_uv` across all channels.

**Returns:** `(n_valid_epochs, n_ch, epoch_len)` array, or `(0, n_ch, epoch_len)` if no epochs survive.

```python
epochs = slice_epochs(
    data,                          # (19, n_times) µV
    sfreq=125.0,
    intervals=[(0.0, 300.0)],     # background windows
    epoch_len_sec=8.0,            # → 1000 samples
    artifact_threshold_uv=200.0,
)
# epochs.shape → (n_valid, 19, 1000)
```

---

### `eeg_bg.preprocessing.reference`

```python
from eeg_bg.preprocessing.reference import detect_reference, filter_by_reference
```

#### `detect_reference(montage_dir: str) -> str`

Infers the reference scheme from the montage directory name.

| Directory contains | Returns |
|-------------------|---------|
| `"_ar"` | `"ar"` |
| `"_le"` | `"le"` |
| neither | `"unknown"` |

#### `filter_by_reference(index: pd.DataFrame, scheme: str) -> pd.DataFrame`

Returns rows from `index` whose `reference` column matches `scheme`, with the index reset. Used to restrict processing to a single montage type.

---

## `decomposition`

### `eeg_bg.decomposition.wiener` — Core Method

```python
from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
    apply_wiener_filter,
    decompose_epoch,
    decompose_subject,
)
```

#### `WienerResult` dataclass

The output of every decomposition call.

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `subject_id` | str | — | Patient identifier |
| `epoch_idx` | int | — | Index within the subject's epoch list |
| `raw` | ndarray | `(n_ch, n_times)` | Original signal in µV |
| `specific` | ndarray | `(n_ch, n_times)` | Source-specific (local cortical) component |
| `coherent` | ndarray | `(n_ch, n_times)` | Coherent (shared physical interference) component |
| `filters` | dict | — | `{pair_key: {ch_name: h}}` where `h` is `(n_ref, n_freqs)` complex |
| `freqs` | ndarray | `(n_freqs,)` | Frequency axis from Welch estimation |
| `ch_names` | list[str] | — | Channel names for this epoch |
| `skipped_pairs` | list[str] | — | Pairs skipped due to low coherence or missing channels |

**Identity guarantee:** `result.specific + result.coherent == result.raw` (up to float64 precision) for all channels, regardless of `nperseg`.

---

#### `estimate_cross_psd(data, sfreq, nperseg) -> tuple[np.ndarray, np.ndarray]`

Estimates the full cross-power spectral density matrix for a group of channels using Welch's method with a rectangular (boxcar) window.

```python
freqs, S = estimate_cross_psd(group_data, sfreq=125.0, nperseg=1000)
# group_data: (n_ch, n_times)
# freqs: (n_freqs,)  — frequency axis
# S: (n_ch, n_ch, n_freqs)  — complex, Hermitian
#   S[i, i] = auto-PSD (real)
#   S[i, j] = cross-PSD, S[j, i] = conj(S[i, j])
```

**Window choice:** A boxcar (rectangular) window is used so that when `nperseg == n_times` there is no spectral leakage mismatch between the PSD estimation and the FFT applied during filter application. When `nperseg < n_times`, multiple segments are averaged for variance reduction.

---

#### `compute_wiener_filter(S, target_idx) -> np.ndarray`

Solves the optimal Wiener filter equation at each frequency bin:

```
h[:, f] = S_ref(f)^{-1} · s_cross(f)
```

where `S_ref` is the cross-PSD sub-matrix of the reference channels, and `s_cross` is the cross-PSD vector between the target channel and all reference channels.

```python
h = compute_wiener_filter(S, target_idx=0)
# h: (n_ref, n_freqs) complex
```

Uses `np.linalg.solve` (numerically stable) rather than explicit matrix inversion. Singular frequency bins (singular `S_ref`) are silently zeroed.

---

#### `apply_wiener_filter(group_data, h, target_idx, n_times) -> tuple[np.ndarray, np.ndarray]`

Applies the Wiener filter in the frequency domain. Handles the case where `nperseg ≠ n_times` by linearly interpolating `h(f)` from the Welch frequency grid to the full `rfft` frequency grid.

```python
specific, coherent = apply_wiener_filter(group_data, h, target_idx=0, n_times=1000)
# specific: (n_times,)   local cortical component
# coherent: (n_times,)   filtered coherent interference
# guaranteed: specific + coherent == group_data[target_idx]
```

**Interpolation detail:** When `n_freqs_welch ≠ n_freqs_full`, real and imaginary parts of `h` are independently linearly interpolated using `np.interp`. This preserves the energy balance identity.

---

#### `decompose_epoch(epoch, ch_names, cfg, subject_id="", epoch_idx=0) -> WienerResult`

Top-level function for decomposing a single epoch. Processes each channel group from `cfg["channels"]["channel_groups"]` (G1–G6) in sequence:

1. **Channel lookup** — skip group if any channel is missing from `ch_names`
2. **Coherence gate** — skip group if `max pairwise coherence in freq_band < coherence_threshold`
3. **Cross-PSD estimation** — call `estimate_cross_psd`
4. **Filter computation** — call `compute_wiener_filter` for each channel in the group
5. **Filter application** — call `apply_wiener_filter`, write back to `specific` and `coherent` arrays

Passthrough channels (`F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `Fz`, `Cz`, `Pz`) belong to no group and are not modified: their `specific = raw`, `coherent = 0`.

```python
result = decompose_epoch(epoch, ch_names, cfg, subject_id="aaaaabhz", epoch_idx=3)
```

---

#### `decompose_subject(epochs, ch_names, subject_id, cfg) -> list[WienerResult]`

Convenience wrapper that calls `decompose_epoch` for each epoch in a batch.

```python
results = decompose_subject(
    epochs,        # (n_epochs, n_ch, n_times)
    ch_names,
    subject_id="aaaaabhz",
    cfg=cfg,
)
# returns list of n_epochs WienerResult objects
```

---

### `eeg_bg.decomposition.wiener_scalar` — Scalar Ablation Baseline

```python
from eeg_bg.decomposition.wiener_scalar import decompose_epoch
```

Identical interface to `wiener.decompose_epoch`. Replaces the frequency-dependent filter `h(f)` with a single complex scalar per reference channel — the mean of `h(f)` over the analysis frequency band:

```
h_scalar = mean_f(h(f))   [shape: (n_ref, 1)]
coherent(t) = Re(h_scalar) · ref(t)   [time-domain multiplication]
```

This is mathematically equivalent to the EKG-style fixed compensation used in cardiac EEG artifact removal. Comparing V3 frequency variation against this baseline tests whether a frequency-dependent model is justified.

```python
from eeg_bg.decomposition import wiener_scalar
result = wiener_scalar.decompose_epoch(epoch, ch_names, cfg)
```

---

### `eeg_bg.decomposition.ica` — FastICA Comparison Method

```python
from eeg_bg.decomposition.ica import fit_ica, apply_ica
```

#### `fit_ica(epochs, ch_names, cfg) -> tuple[mne.preprocessing.ICA, list[int]]`

Concatenates all epochs for a subject, creates an MNE `RawArray`, and fits FastICA with `n_components` (default 19). Identifies artifact components by correlating each ICA source time course with the mean of `FP1` and `FP2` as an EOG proxy.

```python
ica_model, artifact_indices = fit_ica(epochs, ch_names, cfg)
# artifact_indices: list of component indices to remove
```

**Artifact threshold:** Components with `|Pearson r| > cfg["ica"]["artifact_corr_threshold"]` (default 0.8) are flagged.

#### `apply_ica(epochs, ica, artifact_indices, ch_names, cfg) -> np.ndarray`

Applies the fitted ICA model, excluding flagged components, and returns cleaned epochs.

```python
cleaned_epochs = apply_ica(epochs, ica_model, artifact_indices, ch_names, cfg)
# cleaned_epochs: (n_epochs, n_ch, n_times) in µV
```

**Unit handling:** Converts µV → V before MNE processing, then converts back to µV on output.

---

## `verification`

### `eeg_bg.verification.coherence`

```python
from eeg_bg.verification.coherence import compute_pairwise_coherence, run_v1
```

#### `compute_pairwise_coherence(data, sfreq, nperseg, freq_band) -> np.ndarray`

Computes the mean coherence in `freq_band` for all channel pairs.

```python
coh = compute_pairwise_coherence(
    data,                          # (n_ch, n_times)
    sfreq=125.0,
    nperseg=500,
    freq_band=(0.5, 40.0),
)
# coh: (n_ch, n_ch) symmetric, values in [0, 1]
# diagonal = 1.0 (self-coherence)
```

#### `run_v1(results: list[WienerResult], cfg: dict) -> pd.DataFrame`

Computes pairwise coherence before (`raw`) and after (`specific`) Wiener decomposition for every epoch in `results`.

**Returns DataFrame columns:** `subject_id`, `epoch_idx`, `ch_i`, `ch_j`, `coh_pre`, `coh_post`, `reduction`

---

### `eeg_bg.verification.transitivity`

```python
from eeg_bg.verification.transitivity import run_v2, run_v3
```

#### `run_v2(results, cfg) -> pd.DataFrame`

For each triplet of channels `(i, j, k)` that appear across different Wiener-processed pairs, tests the single-source transitivity constraint:

```
ε_amp   = |  |h_ij(f)| · |h_jk(f)| − |h_ik(f)|  |  (mean over freq_band)
ε_phase = |∠h_ij(f) + ∠h_jk(f) − ∠h_ik(f)|  (mod 2π, mean over freq_band)
```

**Returns DataFrame columns:** `subject_id`, `epoch_idx`, `triplet`, `eps_amp`, `eps_phase`

**Pass criteria:** `eps_amp < 0.1` and `eps_phase < 0.392 rad` (π/8).

#### `run_v3(results, cfg) -> pd.DataFrame`

Measures frequency variation of `|h(f)|` within the analysis band for each channel of each processed pair:

```
freq_variation = (max|h(f)| − min|h(f)|) / mean|h(f)|
```

Values > 0.20 (20%) indicate that a frequency-dependent model captures structure not representable by a scalar.

**Returns DataFrame columns:** `subject_id`, `epoch_idx`, `pair`, `channel`, `freq_variation`, `amp_mean`, `amp_std`

---

## `visualization`

All functions return `plt.Figure` and never call `plt.show()`. Save figures with `fig.savefig(path, dpi=150, bbox_inches="tight")`.

### `eeg_bg.visualization.filter_plots`

```python
from eeg_bg.visualization.filter_plots import plot_wiener_filter_response, plot_all_pairs_response
```

#### `plot_wiener_filter_response(result, pair_key, ax=None) -> plt.Figure`

Two-panel figure: `|h(f)|` (amplitude) and `∠h(f)` (phase) for the first reference of the specified bilateral pair.

```python
fig = plot_wiener_filter_response(result, pair_key="FP1-FP2")
```

#### `plot_all_pairs_response(result, save_path=None) -> plt.Figure`

Grid figure (2 rows × N pairs) showing filter response for all pairs with computed filters. Optionally saves to `save_path`.

---

### `eeg_bg.visualization.coherence_plots`

```python
from eeg_bg.visualization.coherence_plots import (
    plot_coherence_matrix,
    plot_coherence_reduction,
    plot_signal_decomposition,
)
```

#### `plot_coherence_matrix(coh_pre, coh_post, ch_names, title="") -> plt.Figure`

Side-by-side heatmaps of the `(n_ch, n_ch)` coherence matrices before and after decomposition. Uses `hot_r` colormap, values clipped to [0, 1].

#### `plot_coherence_reduction(v1_df, group_by="pair") -> plt.Figure`

Boxplot of `reduction = coh_pre - coh_post` grouped by `"pair"` (default) or `"subject_id"`. A horizontal dashed line at 0 marks the no-reduction baseline.

#### `plot_signal_decomposition(result, ch_name, sfreq=125.0, epoch_idx=0, time_window=(0, 4)) -> plt.Figure`

Three-row time series for a single channel: `raw x(t)`, `coherent component`, `specific component`. X-axis is time in seconds.

```python
fig = plot_signal_decomposition(result, ch_name="FP1", time_window=(0, 4))
```

---

### `eeg_bg.visualization.verification_plots`

```python
from eeg_bg.visualization.verification_plots import (
    plot_v2_transitivity,
    plot_v3_frequency_variation,
    plot_ica_vs_wiener_coherence,
)
```

#### `plot_v2_transitivity(v2_df) -> plt.Figure`

Two-panel histogram of `eps_amp` and `eps_phase` across all triplets and epochs, with red dashed threshold lines.

#### `plot_v3_frequency_variation(v3_df) -> plt.Figure`

Bar chart of mean ± std `freq_variation` per channel group, with a red dashed 20% threshold line.

#### `plot_ica_vs_wiener_coherence(raw_pre, ica_post, wiener_post, ch_names) -> plt.Figure`

Three-panel side-by-side coherence heatmaps for Raw / ICA / Wiener outputs. Useful for qualitative comparison in the notebook.

```python
fig = plot_ica_vs_wiener_coherence(
    raw_pre=result.raw,
    ica_post=cleaned_epochs[0],
    wiener_post=result.specific,
    ch_names=result.ch_names,
)
```

---

### `eeg_bg.visualization.psd_plots`

```python
from eeg_bg.visualization.psd_plots import plot_psd_comparison
```

#### `plot_psd_comparison(raw, ch_names, sfreq, channels, wiener_specific=None, ica_specific=None, nperseg=250, freq_band=(0.5, 40.0), title="") -> plt.Figure`

Grid figure with one row per target channel and three columns (Raw | Wiener Specific | ICA Cleaned). PSD is estimated with a boxcar Welch window consistent with the decomposition. Missing channels are shown as greyed-out placeholder panels; `None` inputs for optional arrays also render as "not available".

```python
fig = plot_psd_comparison(
    raw, ch_names, sfreq=125.0,
    channels=cfg["visualization"]["psd_target_channels"],  # e.g. ["FP1", "FP2"]
    wiener_specific=wiener_specific,
    ica_specific=ica_specific,
    title="Subject aaaaabhz — Epoch 0",
)
fig.savefig("results/figures/aaaaabhz/psd_comparison.png", dpi=150, bbox_inches="tight")
```

---

### `eeg_bg.visualization.waveform_plots`

```python
from eeg_bg.visualization.waveform_plots import plot_multichannel_comparison
```

#### `plot_multichannel_comparison(raw, wiener_specific, ica_specific, ch_names, sfreq=125.0, offset_uv=150.0, title="") -> plt.Figure`

Stacked 19-channel EEG waveform figure with 1–3 side-by-side panels (Raw always shown; Wiener-specific and ICA-cleaned panels added when not `None`). Channels are offset vertically by `offset_uv` µV for visual separation.

```python
fig = plot_multichannel_comparison(
    raw, wiener_specific, ica_specific,
    ch_names, sfreq=125.0, title="aaaaabhz  epoch 0",
)
fig.savefig("results/figures/aaaaabhz/waveform_comparison.png", dpi=150, bbox_inches="tight")
```

---

## `features`

### `eeg_bg.features.extraction`

```python
from eeg_bg.features.extraction import extract_epoch_features, build_dataset, FEATURE_NAMES
```

#### `FEATURE_NAMES: list[str]`

Stable ordered list of 171 feature names, built at import time. Each name has the form `"{channel}_{feature}"`, e.g. `"FP1_delta_power"`, `"T3_hjorth_activity"`. Channels iterate in the canonical 19-channel order from `configs/default.yaml`; features iterate in the order `delta_power, theta_power, alpha_power, beta_power, gamma_power, hjorth_activity, hjorth_mobility, hjorth_complexity, spectral_entropy`.

#### `extract_epoch_features(epoch, ch_names, sfreq, nperseg=250, freq_band=(0.5, 40.0)) -> np.ndarray`

Converts a single `(n_ch, n_times)` epoch into a fixed-length `(171,)` feature vector.

| Parameter | Type | Description |
|-----------|------|-------------|
| `epoch` | `np.ndarray (n_ch, n_times)` | Signal in µV |
| `ch_names` | `list[str]` | Channel names for axis 0 |
| `sfreq` | `float` | Sampling frequency in Hz |
| `nperseg` | `int` | Welch window length (default 250) |
| `freq_band` | `tuple[float, float]` | Analysis band in Hz |

Missing channels (not in `ch_names`) fill with zeros so the output length is always 171.

```python
feat = extract_epoch_features(epoch, ch_names, sfreq=125.0)
# feat.shape → (171,)
```

#### `build_dataset(cache_root, condition, split, sfreq, nperseg, freq_band) -> tuple[np.ndarray, np.ndarray, list[str]]`

Iterates all `.npz` files in the appropriate cache subdirectory, filters by split, and calls `extract_epoch_features` on every epoch.

| Condition | Cache subdirectory | Array key |
|-----------|-------------------|-----------|
| `"raw"` | `cache/epochs/` | `"epochs"` |
| `"wiener"` | `cache/wiener_frequency/` | `"specific"` |
| `"ica"` | `cache/ica/` | `"specific"` |

**Returns:** `(X, y, subject_ids)` where `X` is `(n_epochs, 171)`, `y` is `(n_epochs,)` int (0=epilepsy, 1=control), and `subject_ids` is a list of one subject ID per epoch row.

---

## `ml`

### `eeg_bg.ml.xgb_pipeline`

```python
from eeg_bg.ml.xgb_pipeline import train_xgboost, subject_level_predict, evaluate_subject_level
```

#### `train_xgboost(X_train, y_train, X_val, y_val, cfg) -> xgb.XGBClassifier`

Two-phase training pipeline:
1. **Phase 1 — GridSearchCV:** `StratifiedKFold` (5-fold, `scoring="roc_auc"`) over `cfg["ml"]["xgboost"]["param_grid"]`. `n_estimators` is fixed at 500 during the search.
2. **Phase 2 — Early-stopping refit:** Fresh estimator built from best params, fitted on full training set with `eval_set=[(X_val, y_val)]` and `early_stopping_rounds=30`. Val set is used **only** for early stopping, never for hyperparameter selection.

Returns the Phase 2 model (final number of trees determined by early stopping).

#### `subject_level_predict(model, X, subject_ids, labels) -> pd.DataFrame`

Averages epoch-level `predict_proba` outputs per subject.

**Returns DataFrame columns:** `subject_id`, `pred_proba` (mean across epochs), `true_label`.

#### `evaluate_subject_level(subject_df) -> dict[str, float]`

Computes metrics from the output of `subject_level_predict`.

**Returns:** `{"auroc": float, "f1": float, "accuracy": float}`

---

### `eeg_bg.ml.shap_analysis`

```python
from eeg_bg.ml.shap_analysis import (
    compute_shap_values,
    aggregate_shap_by_band,
    aggregate_shap_by_channel,
    plot_shap_summary,
    plot_shap_comparison,
)
```

#### `compute_shap_values(model, X, feature_names) -> np.ndarray`

Computes SHAP values using `shap.TreeExplainer`. Returns `(n_samples, n_features)` array for the positive class (control / label=1).

#### `aggregate_shap_by_band(shap_values, feature_names) -> dict[str, float]`

Mean `|SHAP|` grouped by feature type. Keys: `"delta"`, `"theta"`, `"alpha"`, `"beta"`, `"gamma"`, `"hjorth"`, `"spectral_entropy"`.

#### `aggregate_shap_by_channel(shap_values, feature_names) -> dict[str, float]`

Mean `|SHAP|` grouped by EEG channel. Keys are the 19 channel names.

#### `plot_shap_summary(shap_values, X, feature_names, save_path, max_display=20) -> None`

Beeswarm SHAP summary plot (top `max_display` features). Saves to `save_path` and does not return a figure.

#### `plot_shap_comparison(results_dir, save_path) -> None`

2 × 3 publication-quality figure comparing SHAP band and channel importance across the three conditions (Raw | ICA | Wiener). Saves to `save_path`.

---

## Internal Design Notes

### Why Channel Groups (G1–G6)?

The Wiener decomposition is applied **only** to six anatomically motivated channel groups (G1–G6), not all possible pairs. Each group models a specific **movement-artifact conduction pathway**:

| Group | Channels | Pathway |
|-------|----------|---------|
| G1 | `[FP1, FP2]` | Symmetric frontalis muscle |
| G2 | `[F7, T3]` | Left sternocleidomastoid (SCM) |
| G3 | `[T3, T5, O1]` | Left posterior neck (3-channel chain) |
| G4 | `[O1, O2]` | Bilateral occipitalis |
| G5 | `[F8, T4]` | Right SCM |
| G6 | `[T4, T6, O2]` | Right posterior neck (3-channel chain) |

Groups G3 and G6 are 3-channel chains, handled uniformly by `decompose_epoch` via pairwise coherence gating across all channel combinations in the group. Channels outside these groups (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz`) are designated as `passthrough` — they carry independent cortical signals and are left unmodified.

### Why `nperseg = 250` in Production?

At 125 Hz with 8-second epochs, `n_times = 1000`. Setting `nperseg = 250` gives four Welch segments per epoch, providing variance reduction in the cross-PSD estimate while still resolving individual frequency bins at 0.5 Hz resolution. Because `nperseg (250) < n_times (1000)`, the Wiener filter's frequency grid has 126 bins while the `rfft` grid has 501 bins; `apply_wiener_filter` linearly interpolates `h(f)` to the full rfft grid before applying it. The `specific + coherent = raw` identity is guaranteed algebraically regardless of this interpolation.

The test fixtures use `nperseg = 500` (two segments) to allow meaningful multi-segment coherence estimation in tests that check coherence reduction.

### `specific + coherent = raw` Guarantee

This identity is guaranteed by construction in `apply_wiener_filter`:

```python
coherent = irfft(sum(h_full * rfft(ref), axis=0), n=n_times)
specific = raw[target_idx] - coherent
```

The identity holds regardless of the accuracy of `h(f)` or whether interpolation was applied — it is purely algebraic.
