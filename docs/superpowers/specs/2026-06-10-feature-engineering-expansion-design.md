# Feature Engineering Expansion Design

**Date:** 2026-06-10  
**Goal:** Extend the EEG feature vector from 211 to ~2415 dimensions by adding wavelet, functional connectivity, nonlinear complexity, and multi-scale temporal statistics. Both classification performance and neurophysiological interpretability (via SHAP) are equally weighted objectives.

---

## Context

The current 211-dim feature vector (`eeg_bg/features/extraction.py`) consists of:
- 171 per-channel features: 5 relative band powers (δ/θ/α/β/γ), Hjorth activity/mobility/complexity, spectral entropy — 19 channels × 9 features
- 40 hemispheric asymmetry features: 8 symmetric pairs × 5 bands

**What is missing:**
- Time-frequency detail beyond fixed Welch bands (wavelets)
- Cross-channel synchrony across the full electrode graph (connectivity)
- Nonlinear regularity measures sensitive to interictal background changes
- Multi-scale temporal burst statistics

Wavelet features are computed on the condition's signal directly (`raw` epoch, or `specific` component for `wiener`/`ica`) — not on the `coherent` component separately.

---

## Architecture

### New modules (all under `eeg_bg/features/`)

| File | Responsibility | Output dim |
|---|---|---|
| `wavelet.py` | DWT energy + entropy per channel per level | 228 |
| `connectivity.py` | Pairwise coherence + PLV, all 171 pairs × 5 bands | 1710 |
| `complexity.py` | Sample entropy + Lempel-Ziv per channel | 38 |
| `temporal_stats.py` | Mean/var/skew/kurtosis × 3 scales per channel | 228 |

### Updated `extraction.py`

- `FEATURE_NAMES` is rebuilt at import time by concatenating each module's name list after the existing 211
- Total: **2415 dims** (211 + 228 + 1710 + 38 + 228)
- Existing features stay at positions 0–210 — no invalidation of interpretation for pre-existing SHAP arrays
- The hardcoded `211` in the empty-array fallback is replaced with `len(FEATURE_NAMES)`
- `extract_epoch_features` signature is unchanged; new modules are called with the same `(epoch, ch_names, sfreq, nperseg, freq_band)` contract
- The PSD cache already computed for per-channel features is forwarded to `connectivity.py` for coherence reuse

---

## Feature Definitions

### 1. Wavelet features — `wavelet.py` (228 dims)

**Method:** `pywt.wavedec(signal, wavelet='db4', level=6)` → coefficient arrays at levels 1–6.

Per level per channel:
- **Energy**: `sum(c²) / len(c)` — normalised by coefficient count
- **Entropy**: Shannon entropy of normalised squared coefficients `p_i = c_i² / sum(c²)`; `H = -sum(p * log(p + ε))`

Approximate band mapping at 125 Hz, `db4`, level 6:

| Level | Approximate band |
|---|---|
| 6 | 0.5–2 Hz (δ-low) |
| 5 | 2–4 Hz (δ-high) |
| 4 | 4–8 Hz (θ) |
| 3 | 8–16 Hz (α/β-low) |
| 2 | 16–32 Hz (β) |
| 1 | 32–62.5 Hz (γ) |

**Feature names:** `{ch}_dwt_l{level}_energy`, `{ch}_dwt_l{level}_entropy`  
for `ch` in `_STANDARD_19`, `level` in 1–6.  
Missing channels → zero-padded (12 zeros per absent channel).

**Config key:** `ml.features.wavelet_family` (default `db4`), `ml.features.wavelet_levels` (default `6`).

---

### 2. Functional connectivity — `connectivity.py` (1710 dims)

**Pairs:** All 171 unique pairs from C(19,2) on `_STANDARD_19`, ordered by index (i < j).

Per pair per band (δ/θ/α/β/γ):
- **Magnitude-squared coherence**: `scipy.signal.coherence(x, y, fs=sfreq, nperseg=nperseg)` averaged over the band's frequency bins. Reuses existing `nperseg` for pipeline consistency.
- **Phase-Locking Value (PLV)**: instantaneous phase via `scipy.signal.hilbert`; `|mean(exp(i·(φ_x − φ_y)))|` over the epoch.

Both metrics ∈ [0, 1]. For absent channels, all 10 values (2 metrics × 5 bands) for that pair are zero-padded.

**Feature names:** `coh_{ch1}_{ch2}_{band}`, `plv_{ch1}_{ch2}_{band}`  
for all pairs (ch1, ch2) with ch1 < ch2 in `_STANDARD_19` index order.

**Config key:** `ml.features.connectivity_pairs` — `"all"` (171 pairs) or `"symmetric"` (8 symmetric pairs only, 80 dims). Default `"all"`.

---

### 3. Nonlinear complexity — `complexity.py` (38 dims)

Per channel:
- **Sample Entropy (SampEn)**: template matching with embedding dimension `m=2`, tolerance `r = 0.2 × std(signal)`. Measures regularity; lower values indicate more regularity (as seen in interictal EEG).
- **Lempel-Ziv Complexity (LZC)**: signal binarised around its median; LZ76 complexity normalised by `n / log2(n)`. Captures symbolic sequence complexity.

**Feature names:** `{ch}_sample_entropy`, `{ch}_lempel_ziv`.

**Config keys:** `ml.features.sample_entropy_m` (default `2`), `ml.features.sample_entropy_r` (default `0.2`).

**Implementation note:** SampEn is O(n²) for naïve implementations; use a KD-tree or the `antropy` library if extraction time is prohibitive. LZC is O(n log n).

---

### 4. Multi-scale temporal statistics — `temporal_stats.py` (228 dims)

Three non-overlapping window scales (configurable sample lengths): short=125 (1 s), medium=375 (3 s), full=750 (6 s, ¾-epoch at 125 Hz × 8 s = 1000 samples).

For each scale, the epoch is split into non-overlapping windows of that length. Four statistics are computed **per window** then **averaged across windows** → one scalar per statistic per scale per channel:
- Mean, variance, skewness (`scipy.stats.skew`), kurtosis (`scipy.stats.kurtosis`, excess)

**Feature names:** `{ch}_scale{s}_mean`, `{ch}_scale{s}_var`, `{ch}_scale{s}_skew`, `{ch}_scale{s}_kurtosis`  
for `ch` in `_STANDARD_19`, `s` in `[125, 375, 750]`.

**Config key:** `ml.features.temporal_scales` (default `[125, 375, 750]`).

---

## Config Changes

Add under `ml:` in `configs/default.yaml`:

```yaml
ml:
  features:
    wavelet_family: db4
    wavelet_levels: 6
    connectivity_pairs: all       # "all" (171 pairs) or "symmetric" (8 pairs)
    sample_entropy_m: 2
    sample_entropy_r: 0.2
    temporal_scales: [125, 375, 750]
    drop_low_shap: false          # if true, zero out features below pruning_threshold
  shap:
    pruning_threshold: 1.0e-4     # mean |SHAP| below this → flagged in report
```

---

## Cache Invalidation

- **Tier 4** (script 06 only): any change to `ml.features.*` keys invalidates `cache/features/{condition}_{split}.npz`
- `build_dataset` checks loaded `X.shape[1]` against `len(FEATURE_NAMES)` on load; raises `ValueError` with a `--force` hint if they differ — prevents silent use of stale 211-dim caches
- After this feature expansion, all existing feature caches must be regenerated: `python scripts/06_train_xgboost.py --force`

---

## SHAP Pruning (post-hoc, report-only by default)

After SHAP values are computed across all conditions, a pruning step computes `mean(|SHAP|)` per feature across all test epochs and all conditions. Features below `ml.shap.pruning_threshold` are written to `results/xgboost/shap_pruning_report.json`:

```json
{
  "threshold": 1e-4,
  "n_features_total": 2415,
  "n_features_below_threshold": 843,
  "features_below_threshold": ["FP1_scale750_mean", ...]
}
```

If `ml.features.drop_low_shap: true`, those feature positions are zeroed in the output of `extract_epoch_features` (positions preserved for SHAP array compatibility) and a re-run of script 06 without `--force` picks up the updated extractor.

---

## New Dependency

`pywt` (PyWavelets) is required by `wavelet.py`.

- Add `pywavelets>=1.4` to `environment.yaml` and `setup.py`
- Available on conda (`conda install pywavelets`) and pip

---

## Tests

New test files under `tests/test_features/`:

| File | Key assertions |
|---|---|
| `test_wavelet.py` | shape `(228,)`, energy ≥ 0, entropy ≥ 0 on sine input; zero-padded for absent channel |
| `test_connectivity.py` | shape `(1710,)`, coherence/PLV ∈ [0,1]; coherence=1 and PLV=1 for identical signals |
| `test_complexity.py` | shape `(38,)`, LZC ∈ [0,1], SampEn > 0 for random noise, SampEn=0 for constant signal |
| `test_temporal_stats.py` | shape `(228,)`, variance=0 for constant signal, skew=0 for symmetric signal |

`tests/test_features/test_extraction.py` updated:
- `assert len(FEATURE_NAMES) == 2415`
- `assert extract_epoch_features(epoch, ch_names, sfreq).shape == (2415,)`
- Assert first 211 names are unchanged (positional stability regression test)

---

## Summary

| Domain | Module | New dims |
|---|---|---|
| Wavelet (DWT energy + entropy) | `wavelet.py` | 228 |
| Functional connectivity (coherence + PLV) | `connectivity.py` | 1710 |
| Nonlinear complexity (SampEn + LZC) | `complexity.py` | 38 |
| Multi-scale temporal statistics | `temporal_stats.py` | 228 |
| **Total new** | | **2204** |
| **Grand total** | | **2415** |
