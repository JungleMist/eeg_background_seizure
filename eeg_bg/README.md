# `eeg_bg` — Package API Reference

This document covers the core numerical/data APIs in eight sections. The package also contains `application` and `gui` modules used by ECMAD Studio; those are documented by their source and tests rather than exhaustively listed here.

---

## Table of Contents

1. [`config`](#config) — Configuration loading
2. [`io`](#io) — Data I/O and caching
3. [`preprocessing`](#preprocessing) — Epoch extraction and filtering
4. [`decomposition`](#decomposition) — Wiener and ICA signal decomposition
5. [`verification`](#verification) — Physical validation experiments
6. [`visualization`](#visualization) — Figure generation
7. [`features`](#features) — Handcrafted feature extraction
8. [`ml`](#ml) — XGBoost + SHAP and EEGNet CNN pipelines

---

## `config`

### `eeg_bg.config.settings`

```python
from eeg_bg.config.settings import load_config
```

#### `load_config(config_path="configs/default.yaml") -> dict`

Loads a YAML configuration file, recursively resolves an optional `extends` parent, and deep-merges nested overrides. Relative `cache_dir` and `results_dir` values are anchored at `config_path.parent.parent`, so tracked/local configs should live directly under `configs/`.

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

Backward-compatible wrapper around `build_recording_index`. It discovers one EDF row per recording using the active `dataset.tuep` or `dataset.tuab` adapter and adds `subject_id = patient_id`.

**Returns:** DataFrame with columns:

| Column | Type | Description |
|--------|------|-------------|
| `dataset_name` | str | `"tuep"` or `"tuab"` |
| `patient_id` / `subject_id` | str | Patient identifier (`subject_id` is the compatibility alias) |
| `session_id` | str | Session identifier (e.g., `s002`) |
| `token_id` | str | File segment within session (e.g., `t000`) |
| `recording_id` | str | EDF stem |
| `evaluation_id` | str | TUEP patient/class unit or TUAB recording unit |
| `class_name` / `label` | str / int | Active class name and configured 0/1 label |
| `reference` | str | Configured active reference scheme (`"ar"` or `"le"`) |
| `source_partition` | str | TUAB official partition; empty for TUEP |
| `edf_path` | str | Absolute path to the `.edf` file |

#### `assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame`

Backward-compatible wrapper around dataset-aware splitting. TUEP uses class-stratified patient/evaluation-unit train/val/test assignment from `cfg["split"]`. TUAB keeps official `eval` as test and creates a patient-grouped validation fold from official train, preventing a patient from crossing splits even when that patient has recordings of both labels.

**Returns:** Input DataFrame with an additional `split` column (`"train"` / `"val"` / `"test"`).

---

### `eeg_bg.io.edf_reader`

```python
from eeg_bg.io.edf_reader import load_edf
```

#### `load_edf(edf_path: Path, cfg: dict) -> tuple[np.ndarray, list[str], float]`

Loads a single EDF file using MNE, normalises channel names, selects available configured standard channels, applies a 5th-order Butterworth IIR bandpass, and resamples to `cfg["preprocessing"]["target_sfreq"]`. TUEP may return a subset; TUAB raises if any of the 19 required channels are missing.

**Returns:** `(data, ch_names, sfreq)`
- `data`: `(n_available_channels, n_times)` array in **µV**
- `ch_names`: canonical channel names in configured order
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

Computes a 16-character SHA-256 fingerprint from EDF path/start, active dataset, sample rate, bandpass, epoch length, artifact threshold, canonical channels, seizure buffer, and the active dataset's recording-duration cap. Changing one of these values changes the cache key.

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
    epoch_len_sec=20.0,           # → 2500 samples (current default)
    artifact_threshold_uv=200.0,
)
# epochs.shape → (n_valid, 19, 2500)
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
| `filters` | dict | — | `{group_key: {ch_name: h}}` for accepted target-channel candidates |
| `freqs` | ndarray | `(n_freqs,)` | Frequency axis from Welch estimation |
| `ch_names` | list[str] | — | Channel names for this epoch |
| `skipped_pairs` | list[str] | — | Channel groups skipped due to missing channels or no accepted targets |
| `channel_sources` | dict[str, list[str]] | — | Accepted source groups contributing to each output channel |
| `channel_weights` | dict[str, dict[str, float]] | — | Normalised coherence-fusion weights by channel and source group |
| `candidate_keys` | list[str] or None | — | `"group::channel"` keys aligned with candidate diagnostic arrays |
| `candidate_status` | ndarray or None | `(n_candidates,)` | Status code per target candidate |
| `candidate_coherence` | ndarray or None | `(n_candidates,)` | Target-to-reference gate score |
| `candidate_max_abs_h` | ndarray or None | `(n_candidates,)` | Maximum fitted filter magnitude |
| `phase_gate_pass_fraction` | ndarray or None | `(n_candidates,)` | Fraction of frequency coefficients admitted by the phase gate |
| `candidate_fusion_weight` | ndarray or None | `(n_candidates,)` | Final fusion weights; rejected candidates are zero |
| `group_gate_keys` | list[str] or None | — | Channel-group keys aligned with coherent power-gate diagnostics |
| `group_coherent_gate_open` | ndarray or None | `(n_groups,)` | Whether each complete group passed the window-level coherent power gate |
| `group_max_bin_rms_uv` | ndarray or None | `(n_groups,)` | Maximum single-bin RMS amplitude in the effective score band, in µV |

**Identity guarantee:** `result.specific + result.coherent == result.raw` (up to float64 precision) for all channels, regardless of `nperseg`.

---

#### `estimate_cross_psd(data, sfreq, nperseg) -> tuple[np.ndarray, np.ndarray]`

Estimates the full cross-power spectral density matrix for a group of channels using Welch's method with a rectangular (boxcar) window.

```python
freqs, S = estimate_cross_psd(group_data, sfreq=125.0, nperseg=500)
# group_data: (n_ch, n_times)
# freqs: (n_freqs,)  — frequency axis
# S: (n_ch, n_ch, n_freqs)  — complex, Hermitian
#   S[i, i] = auto-PSD (real)
#   S[i, j] = cross-PSD, S[j, i] = conj(S[i, j])
```

**Window choice:** A boxcar (rectangular) window is used so that when `nperseg == n_times` there is no spectral leakage mismatch between the PSD estimation and the FFT applied during filter application. When `nperseg < n_times`, multiple segments are averaged for variance reduction.

---

#### `compute_wiener_filter(S, target_idx, reg_factor=1e-4) -> np.ndarray`

Solves the optimal Wiener filter equation at each frequency bin:

```
h[:, f] = S_ref(f)^{-1} · s_cross(f)
```

where `S_ref` is the cross-PSD sub-matrix of the reference channels, and `s_cross` is the cross-PSD vector between the target channel and all reference channels.

```python
h = compute_wiener_filter(S, target_idx=0)
# h: (n_ref, n_freqs) complex
```

Uses `np.linalg.solve` after Tikhonov diagonal loading proportional to the mean real diagonal of `S_ref`. A singular solve or non-finite result raises `WienerSolveError`; `decompose_epoch_with_fusion` records that target candidate as `CANDIDATE_SOLVE_FAILED` instead of silently treating it as a valid zero filter.

---

#### `apply_wiener_filter(group_data, h, target_idx, n_times, sfreq=None, protected_band_hz=None) -> tuple[np.ndarray, np.ndarray]`

Applies the Wiener filter in the frequency domain. Handles the case where `nperseg ≠ n_times` by linearly interpolating `h(f)` from the Welch frequency grid to the full `rfft` frequency grid.

```python
specific, coherent = apply_wiener_filter(group_data, h, target_idx=0, n_times=2500)
# specific: (n_times,)   local cortical component
# coherent: (n_times,)   filtered coherent interference
# guaranteed: specific + coherent == group_data[target_idx]
```

**Interpolation detail:** When `n_freqs_welch ≠ n_freqs_full`, real and imaginary parts of `h` are independently linearly interpolated using `np.interp`. This preserves the energy balance identity.

When `protected_band_hz=(low, high)` is enabled, the closed interval is
forced to zero on the final `rfft` grid after interpolation. The default
configuration protects 5–20 Hz; set `wiener.protected_band_hz: null` to
disable this behavior.

---

#### `decompose_epoch(epoch, ch_names, cfg, subject_id="", epoch_idx=0) -> WienerResult`

Top-level function for decomposing a single epoch. Processes each channel group from `cfg["channels"]["channel_groups"]` (G1–G6), but writes the final output once after target-level candidate fusion:

1. **Channel lookup** — record missing-channel candidates when a group is unavailable
2. **Cross-PSD estimation** — call `estimate_cross_psd` once for the available group
3. **Group coherent power gate** — over `freq_band` excluding `protected_band_hz`, convert diagonal PSD to single-bin RMS amplitude with `sqrt(PSD × Δf)`. If `coherent_gate_enabled` is true, at least one channel/bin must be strictly above `coherent_gate_threshold_uv` (default 100 µV), otherwise every target in that group is marked `CANDIDATE_COHERENT_GATE_CLOSED`
4. **Target coherence gate** — skip each target whose remaining max target-reference coherence is below `coherence_threshold`
5. **Filter computation** — call `compute_wiener_filter` for each accepted target channel in the group
6. **Overlap fusion** — combine multiple candidates for the same output channel using normalised target-reference coherence weights
7. **Final split** — set `specific[channel] = raw[channel] - coherent[channel]`, so the identity guarantee holds

Channels that belong to no configured group are not modified (`specific = raw`, `coherent = 0`). In the default config these are `F3`, `F4`, `C3`, `C4`, `P3`, `P4`, `Fz`, `Cz`, and `Pz`. The implementation derives this behavior from `channel_groups`; it does not read `channels.passthrough`.

```python
result = decompose_epoch(epoch, ch_names, cfg, subject_id="aaaaabhz", epoch_idx=3)
```

The epoch API expects microvolt input. Continuous MNE entry points convert
their volt-valued `Raw` windows to microvolts before evaluating this absolute
gate and convert the denoised output back to volts.

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

Uses the same `decompose_epoch` call signature as the frequency mode. It replaces the frequency-dependent filter `h(f)` with a single complex scalar per reference channel — the mean of `h(f)` over the analysis frequency band:

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

The configured/reference plot thresholds are `eps_amp < 0.1` and `eps_phase < 0.392 rad` (approximately π/8); `run_v2` itself returns measurements rather than a pass/fail column.

#### `run_v3(results, cfg) -> pd.DataFrame`

Measures frequency variation of `|h(f)|` within the analysis band for each channel of each processed pair:

```
freq_variation = (max|h(f)| − min|h(f)|) / mean|h(f)|
```

The configured/reference plot threshold is 0.20 (20%); `run_v3` returns the measurements rather than a pass/fail column.

**Returns DataFrame columns:** `subject_id`, `epoch_idx`, `pair`, `channel`, `freq_variation`, `amp_mean`, `amp_std`

---

## `visualization`

All functions return `plt.Figure` and never call `plt.show()`. Save figures with `fig.savefig(path, dpi=150, bbox_inches="tight")`.

### `eeg_bg.visualization.filter_plots`

```python
from eeg_bg.visualization.filter_plots import plot_wiener_filter_response, plot_all_pairs_response
```

#### `plot_wiener_filter_response(result, pair_key, ax=None) -> plt.Figure`

Two-panel figure: `|h(f)|` (amplitude) and `∠h(f)` (phase) for the first stored target/reference coefficient in the specified channel-group key.

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

Three-panel side-by-side coherence heatmaps for Raw / ICA / Wiener outputs. This helper currently hardcodes 125 Hz, `nperseg=500`, and 0.5–40 Hz.

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

Stable ordered list of 211 feature names, built at import time by concatenating two blocks: 171 per-channel names (form `"{channel}_{feature}"`) followed by 40 hemispheric-asymmetry names. Channels iterate in `eeg_bg.features._constants._STANDARD_19`; per-channel features iterate in the order `delta_power, theta_power, alpha_power, beta_power, gamma_power, hjorth_activity, hjorth_mobility, hjorth_complexity, spectral_entropy`.

The default `base211` profile uses this vector unchanged. The optional `base211_conn80` profile appends the 80 connectivity names from `CONNECTIVITY_NAMES`, producing 291 dimensions. Wavelet, complexity, and temporal-stat modules remain implemented and tested but are not selected by either current profile.

#### `extract_epoch_features(epoch, ch_names, sfreq, nperseg=250, freq_band=(0.5, 40.0)) -> np.ndarray`

Converts a single `(n_ch, n_times)` epoch into a fixed-length `(211,)` feature vector: 171 per-channel statistics followed by 40 hemispheric-asymmetry features (via `asymmetry.hemispheric_asymmetry`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `epoch` | `np.ndarray (n_ch, n_times)` | Signal in µV |
| `ch_names` | `list[str]` | Channel names for axis 0 |
| `sfreq` | `float` | Sampling frequency in Hz |
| `nperseg` | `int` | Welch window length (default 250) |
| `freq_band` | `tuple[float, float]` | Analysis band in Hz |

Missing channels (not in `ch_names`) fill with zeros so the output length is always 211.

```python
feat = extract_epoch_features(epoch, ch_names, sfreq=125.0)
# feat.shape → (211,)
```

#### `build_dataset(cache_root, condition, split, sfreq, nperseg, freq_band, max_workers=None) -> tuple[np.ndarray, np.ndarray, list[str]]`

Iterates all `.npz` files in the appropriate cache subdirectory, filters by split, and calls `extract_epoch_features` on every epoch.

| Condition | Cache subdirectory | Array key |
|-----------|-------------------|-----------|
| `"raw"` | `cache/epochs/` | `"epochs"` |
| `"wiener"` | `cache/wiener_frequency/` | `"specific"` |
| `"ica"` | `cache/ica/` | `"specific"` |
| `"wiener_phasegated"` | `cache/wiener_phasegated/` | `"specific"` |
| `"wiener_zerophase"` | `cache/wiener_zerophase/` | `"specific"` |

**Returns:** `(X, y, subject_ids)` where `X` is `(n_epochs, 211)`, `y` is `(n_epochs,)` int using the active dataset's configured labels, and `subject_ids` contains one evaluation ID per epoch (TUEP patient unit; TUAB recording unit).

---

### `eeg_bg.features.asymmetry`

```python
from eeg_bg.features.asymmetry import hemispheric_asymmetry, ASYMMETRY_NAMES, SYMMETRIC_PAIRS
```

#### `SYMMETRIC_PAIRS: list[tuple[str, str]]`

The 8 anatomically symmetric (left, right) electrode pairs: `(FP1,FP2), (F3,F4), (F7,F8), (C3,C4), (T3,T4), (T5,T6), (P3,P4), (O1,O2)`. Order is fixed — reordering invalidates saved SHAP `.npy` arrays, and `connectivity.ALL_PAIRS` derives from this same list.

#### `hemispheric_asymmetry(epoch, ch_names, sfreq, nperseg=250, freq_band=(0.5, 40.0), psd_cache=None, pairs=SYMMETRIC_PAIRS) -> np.ndarray`

Computes `(40,)` normalised left–right power asymmetry: `(P_left - P_right) / (P_left + P_right + ε)` for each of the 8 symmetric pairs × 5 bands. Values lie in `(-1, +1)`; positive → left dominates. `psd_cache` optionally reuses PSDs already computed by `extract_epoch_features`'s per-channel loop to avoid redundant `welch()` calls. Zero-padded when either electrode in a pair is absent.

#### `ASYMMETRY_NAMES: list[str]`

40 names of the form `"asym_{left}_{right}_{band}"`, e.g. `"asym_FP1_FP2_delta"`.

---

### Optional and disconnected feature modules

All four modules are implemented and unit-tested. Connectivity is selected by `base211_conn80` through `eeg_bg.features.profiles`; the other three are not selected by either current profile.

| Module | Function | Output | Description |
|--------|----------|--------|-------------|
| `wavelet.py` | `wavelet_features(signal, wavelet="db4", level=6) -> np.ndarray` | `(27,)` per channel | PyWavelets DWT: detail energy per level (6), modulus-maxima mean per level (6), reconstructed-band signal stats (15). `WAVELET_NAMES` (513 total across 19 channels). |
| `connectivity.py` | `connectivity_features(epoch, ch_names, sfreq, nperseg) -> np.ndarray` | `(80,)` | Magnitude-squared coherence + Phase-Locking Value for 8 homotopic pairs × 5 bands × 2 metrics. Appended by `base211_conn80`; `ml.features.connectivity.nperseg` controls its window. |
| `complexity.py` | `complexity_features(epoch, ch_names, m=2, r_factor=0.2) -> np.ndarray` | `(38,)` | Sample Entropy + Lempel-Ziv Complexity per channel. `COMPLEXITY_NAMES`. |
| `temporal_stats.py` | `epoch_temporal_stats(epoch, ch_names, scales=[125,375,750]) -> np.ndarray` | `(228,)` | Mean/variance/skewness/kurtosis per non-overlapping window, averaged across windows, at 3 time scales per channel. `TEMPORAL_NAMES`. |

`aggregate_shap_by_band` always reports `"wavelet"`, `"connectivity"`, `"complexity"`, and `"temporal"`. Under `base211`, all four are `0.0`; under `base211_conn80`, `"connectivity"` reflects the appended coherence/PLV features while the other three remain `0.0`.

---

## `ml`

### `eeg_bg.ml.xgb_pipeline`

```python
from eeg_bg.ml.xgb_pipeline import (
    train_xgboost,
    subject_level_predict,
    evaluation_level_predict,
    evaluate_subject_level,
    find_optimal_threshold,
)
```

#### `train_xgboost(X_train, y_train, X_val, y_val, cfg, groups=None) -> xgb.XGBClassifier`

Two-phase training pipeline:
1. **Phase 1 — GridSearchCV:** `StratifiedGroupKFold` when patient `groups` are supplied (as script 06 does), otherwise `StratifiedKFold`; fold count comes from config and scoring is AUROC. `n_estimators` is fixed at 500 during the search.
2. **Phase 2 — Early-stopping refit:** Fresh estimator built from best params, fitted on full training set with `eval_set=[(X_val, y_val)]` and the configured `early_stopping_rounds` (30 by default). Val data are used **only** for early stopping, never for hyperparameter selection.

Returns the Phase 2 model (final number of trees determined by early stopping).

#### `subject_level_predict(model, X, subject_ids, labels) -> pd.DataFrame`

Averages epoch-level `predict_proba` outputs per subject.

**Returns DataFrame columns:** `subject_id`, `pred_proba` (mean across epochs), `true_label`.

#### `evaluation_level_predict(model, X, labels, evaluation_ids, patient_ids, recording_ids, dataset_names) -> pd.DataFrame`

Dataset-aware aggregation used by script 06. It averages label-1 probabilities by `evaluation_id`, validates consistent labels/patients, and returns dataset/evaluation/patient/recording identity, epoch count, probability, and true label. TUEP evaluation units are patients; TUAB evaluation units are recordings while patient IDs remain available for leakage checks.

#### `find_optimal_threshold(subject_df) -> float`

Sweeps 181 evenly-spaced thresholds in `[0.05, 0.95]` and returns the first threshold achieving the highest macro-F1 encountered; it returns `0.5` only if every candidate has F1 equal to zero. Script 06 derives this threshold from validation predictions and applies it unchanged to test data. The CNN pipeline reuses the same helper.

#### `evaluate_subject_level(subject_df, threshold=0.5) -> dict[str, float]`

Computes metrics from the output of `subject_level_predict`, thresholding `pred_proba` at `threshold` to derive class predictions.

**Returns:** `{"auroc": float, "f1": float, "accuracy": float, "threshold": float}`

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

Computes SHAP values using `shap.TreeExplainer`. Returns `(n_samples, n_features)` array. `feature_names` is accepted for documentation/API symmetry with the aggregation functions below but isn't used inside this function — the positional correspondence between `X`'s columns and `feature_names` is the caller's responsibility (see `extraction.FEATURE_NAMES`).

#### `aggregate_shap_by_band(shap_values, feature_names) -> dict[str, float]`

Mean `|SHAP|` grouped by feature type, via substring/prefix matching against `feature_names`. Keys: `"delta"`, `"theta"`, `"alpha"`, `"beta"`, `"gamma"`, `"hjorth"`, `"spectral_entropy"`, `"asymmetry"`, `"wavelet"`, `"connectivity"`, `"complexity"`, `"temporal"`. Connectivity is non-zero for `base211_conn80`; wavelet/complexity/temporal remain zero for both current profiles.

#### `aggregate_shap_by_channel(shap_values, feature_names) -> dict[str, float]`

Mean `|SHAP|` grouped by EEG channel. Keys are the 19 channel names. Asymmetry features (`asym_` prefix) are excluded (reported under `aggregate_shap_by_band` instead); pairwise connectivity features split their `|SHAP|` 50/50 between the two channels involved.

#### `plot_shap_summary(shap_values, X, feature_names, title, output_path, max_display=20, dpi=150) -> None`

Beeswarm SHAP summary plot (top `max_display` features) via `shap.summary_plot`. Saves to `output_path` and does not return a figure.

#### `plot_shap_comparison(results, output_path, dpi=200) -> None`

2 × 5 publication-quality figure comparing Raw, ICA, Wiener, Wiener Phase-Gated, and Wiener Zero-Phase. Missing condition data render as zero-valued columns. `results` is keyed by condition and contains pre-aggregated `"shap_by_band"`/`"shap_by_channel"` dictionaries.

---

### `eeg_bg.ml.cnn_model`

```python
from eeg_bg.ml.cnn_model import EEGNet
```

#### `EEGNet(n_channels=19, n_times=1000, F1=8, D=2, dropout=0.25)` — `nn.Module`

Compact EEG classification CNN (Lawhern et al. 2018, *EEGNet*). Three blocks: temporal conv (learns frequency-band-like filters along time) → depthwise spatial conv (combines channels within each temporal filter, `F2 = F1 * D` filters) → separable temporal conv (compact temporal refinement) → `Linear` + `Sigmoid`. `forward(x)` takes `x` of shape `(batch, 1, n_channels, n_times)` and returns `(batch, 1)` probabilities in `[0, 1]`.

---

### `eeg_bg.ml.cnn_dataset`

```python
from eeg_bg.ml.cnn_dataset import EEGEpochDataset
```

#### `EEGEpochDataset(cache_root, condition, split)` — `torch.utils.data.Dataset`

Reads the same `cache/{epochs,wiener_frequency,ica}/` trees as feature extraction, but yields raw epoch tensors and loads matching epochs eagerly. `__getitem__` returns `(epoch_tensor, label, subject_id)` where `epoch_tensor` has shape `(1, 19, n_times)` and is z-scored independently per channel. Current production caches use `n_times=2500`; unit fixtures commonly use 1000. Only `raw`, `ica`, and `wiener` are supported.

---

### `eeg_bg.ml.cnn_pipeline`

```python
from eeg_bg.ml.cnn_pipeline import cnn_predict_epochs, train_cnn
```

#### `cnn_predict_epochs(model, dataloader, device="cpu") -> pd.DataFrame`

Runs `model` over `dataloader` in eval mode and averages epoch-level probabilities per subject, mirroring `xgb_pipeline.subject_level_predict`.

**Returns DataFrame columns:** `subject_id`, `pred_proba`, `true_label`.

#### `train_cnn(condition, cfg, out_dir, force=False) -> dict`

Full training loop for one condition: builds `EEGEpochDataset`/`DataLoader` for train/val/test, constructs an `EEGNet` sized from `cfg["ml"]["cnn"]` (`F1`, `D`, `dropout`) with `n_times` inferred from the actual data, trains with class-balanced weighted BCE loss (`pos_weight` from the train-split class ratio), `Adam` + `ReduceLROnPlateau` (mode `"max"` on val AUROC), and early stopping (`patience` epochs without val-AUROC improvement). Reuses `find_optimal_threshold`/`evaluate_subject_level` from `xgb_pipeline.py` for threshold selection and final metrics. If `force=False` and `out_dir/val_metrics.json` already exists, skips training and returns the cached metrics.

**Returns:** `{"val": {...}, "test": {...}}` metric dicts. **Writes** `best_model.pt` (state dict), `best_params.json`, `val_metrics.json`/`test_metrics.json`, `val_predictions.csv`/`test_predictions.csv` to `out_dir`.

### `eeg_bg.ml.erp_eegnet`

```python
from eeg_bg.ml.erp_eegnet import TrialSequenceDataset, train_condition
```

Reusable ERP trial-level EEGNet method used by the ERP-CORE component scripts.
`TrialSequenceDataset` yields `(1, n_channels, n_times)` tensors, while
`train_condition(condition, dataset, partitions, model_cfg, out_dir,
random_state)` performs subject-partitioned training, validation-AUPRC
checkpoint selection, validation balanced-accuracy threshold selection, and
test evaluation. The dataset object only needs `matrix()`, `y`, and
`subject_ids`, so decomposition and cache implementations remain independent.

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

Groups G3 and G6 are 3-channel chains, handled uniformly by `decompose_epoch` with a target-to-reference coherence gate. Channels outside every group (`F3, F4, C3, C4, P3, P4, Fz, Cz, Pz`) are left unmodified. `channels.passthrough` documents this complement but is not consumed by the decomposition function.

### Why `nperseg = 500` in Production?

At 125 Hz with 20-second epochs, `n_times = 2500`. The current `wiener.nperseg=500` gives 0.25 Hz filter bins and nine Welch segments with SciPy's default 50% overlap. The filter grid has 251 bins while the epoch `rfft` grid has 1251, so `apply_wiener_filter` linearly interpolates `h(f)` before applying it. The `specific + coherent = raw` identity is algebraic and does not depend on interpolation accuracy.

The root unit fixtures use 1000-sample synthetic epochs with the same `nperseg=500`, yielding three overlapping Welch segments. V1 verification independently derives a 250-sample coherence window from `freq_resolution_hz=0.5`.

### `specific + coherent = raw` Guarantee

This identity is guaranteed by construction in `apply_wiener_filter`:

```python
coherent = irfft(sum(h_full * rfft(ref), axis=0), n=n_times)
specific = raw[target_idx] - coherent
```

The identity holds regardless of the accuracy of `h(f)` or whether interpolation was applied — it is purely algebraic.
