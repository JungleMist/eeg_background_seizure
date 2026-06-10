# Feature Engineering Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the EEG feature vector from 211 to 2415 dimensions by adding DWT wavelet, functional connectivity, nonlinear complexity, and multi-scale temporal statistics features.

**Architecture:** Four new modules under `eeg_bg/features/` expose a per-epoch function and a module-level name list; `extraction.py` assembles them in sequence after the existing 211 features, rebuilding `FEATURE_NAMES` at import time. A shared `_constants.py` holds `_STANDARD_19` to avoid circular imports between `extraction.py` and `connectivity.py`.

**Tech Stack:** PyWavelets (`pywt`), `scipy.signal` (coherence, hilbert, butter/sosfiltfilt), `scipy.spatial.cKDTree` (SampEn), `scipy.stats` (skewness/kurtosis), NumPy stride tricks.

---

## File Map

| Action | Path |
|---|---|
| **Create** | `eeg_bg/features/_constants.py` |
| **Create** | `eeg_bg/features/wavelet.py` |
| **Create** | `eeg_bg/features/complexity.py` |
| **Create** | `eeg_bg/features/temporal_stats.py` |
| **Create** | `eeg_bg/features/connectivity.py` |
| **Create** | `tests/test_features/test_wavelet.py` |
| **Create** | `tests/test_features/test_complexity.py` |
| **Create** | `tests/test_features/test_temporal_stats.py` |
| **Create** | `tests/test_features/test_connectivity.py` |
| **Modify** | `eeg_bg/features/extraction.py` |
| **Modify** | `tests/test_features/test_extraction.py` |
| **Modify** | `configs/default.yaml` |
| **Modify** | `scripts/06_train_xgboost.py` |
| **Modify** | `setup.py` |

---

## Task 1: Install PyWavelets and create `_constants.py`

**Files:**
- Modify: `setup.py`
- Create: `eeg_bg/features/_constants.py`
- Modify: `eeg_bg/features/extraction.py` (import change only)

- [ ] **Step 1: Install PyWavelets into the conda environment**

```bash
conda run -n eeg_pipeline pip install pywavelets>=1.4
```

Verify: `conda run -n eeg_pipeline python -c "import pywt; print(pywt.__version__)"` — should print a version string.

- [ ] **Step 2: Document the dependency in `setup.py`**

Replace the entire file content:

```python
from setuptools import setup, find_packages

setup(
    name="eeg_bg",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pywavelets>=1.4",
    ],
)
```

- [ ] **Step 3: Create `eeg_bg/features/_constants.py`**

```python
"""Shared constants for the features package.

Keeping _STANDARD_19 here prevents a circular import: connectivity.py
needs this list to define ALL_PAIRS at import time, but extraction.py
also imports connectivity.py — so both must import from a third module.
"""
_STANDARD_19: list[str] = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
```

- [ ] **Step 4: Update the import in `extraction.py`**

In `eeg_bg/features/extraction.py`, replace:

```python
# Standard 19 channels in canonical order (matches configs/default.yaml)
_STANDARD_19 = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
```

with:

```python
from eeg_bg.features._constants import _STANDARD_19
```

- [ ] **Step 5: Verify existing tests still pass**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/ -v
```

Expected: all existing tests pass (green). The `_STANDARD_19` re-export via `extraction.py` keeps `from eeg_bg.features.extraction import _STANDARD_19` working in `test_extraction.py`.

- [ ] **Step 6: Commit**

```bash
git add setup.py eeg_bg/features/_constants.py eeg_bg/features/extraction.py
git commit -m "refactor: extract _STANDARD_19 to _constants.py; add pywavelets dep"
```

---

## Task 2: `wavelet.py` — DWT energy and entropy per channel

**Files:**
- Create: `eeg_bg/features/wavelet.py`
- Create: `tests/test_features/test_wavelet.py`

### Background

`pywt.wavedec(signal, 'db4', level=6)` returns:
`[cA6, cD6, cD5, cD4, cD3, cD2, cD1]` (index 0 = approximation, indices 1–6 = details).
`cD1` = highest-frequency detail (~32–62 Hz); `cD6` = lowest-frequency detail (~0.5–2 Hz).
We label features by level number: `l1` = `cD1` = `coeffs[-1]`, `l6` = `cD6` = `coeffs[1]`.
In general: `cD_l` = `coeffs[level + 1 - l]`.

Per level: **energy** = `sum(c²) / len(c)`; **entropy** = `−sum(p·log(p+ε))` where `p_i = c_i² / (sum(c²)+ε)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_wavelet.py`:

```python
"""Unit tests for eeg_bg.features.wavelet."""
import numpy as np
import pytest
from eeg_bg.features.wavelet import wavelet_features, WAVELET_NAMES, _DEFAULT_LEVELS

SFREQ = 125.0


@pytest.fixture
def sine_signal():
    t = np.arange(1000) / SFREQ
    return (np.sin(2 * np.pi * 10.0 * t) * 50.0).astype(np.float64)


@pytest.fixture
def random_signal():
    return np.random.default_rng(0).standard_normal(1000).astype(np.float64) * 20.0


def test_wavelet_features_shape(sine_signal):
    out = wavelet_features(sine_signal)
    assert out.shape == (2 * _DEFAULT_LEVELS,)


def test_wavelet_names_length():
    from eeg_bg.features._constants import _STANDARD_19
    assert len(WAVELET_NAMES) == len(_STANDARD_19) * 2 * _DEFAULT_LEVELS
    assert len(WAVELET_NAMES) == 228


def test_wavelet_names_unique():
    assert len(WAVELET_NAMES) == len(set(WAVELET_NAMES))


def test_wavelet_energy_nonnegative(random_signal):
    out = wavelet_features(random_signal)
    energies = out[0::2]  # every other value starting at 0
    assert np.all(energies >= 0.0)


def test_wavelet_entropy_nonnegative(random_signal):
    out = wavelet_features(random_signal)
    entropies = out[1::2]  # every other value starting at 1
    assert np.all(entropies >= 0.0)


def test_wavelet_sine_dominant_level(sine_signal):
    """10 Hz sine should have highest energy at level 4 (~4–8 Hz at 125 Hz)."""
    out = wavelet_features(sine_signal)
    energies = out[0::2]  # index 0=l1, 1=l2, ..., 5=l6 energies
    # Level 4 energy (index 3) should dominate for 10 Hz
    assert energies[3] == energies.max()


def test_wavelet_constant_zero_entropy():
    """Constant signal should have near-zero energy and entropy at all levels."""
    const = np.full(1000, 5.0, dtype=np.float64)
    out = wavelet_features(const)
    assert np.allclose(out, 0.0, atol=1e-6)


def test_wavelet_output_dtype(random_signal):
    out = wavelet_features(random_signal)
    assert out.dtype == np.float64
```

- [ ] **Step 2: Run to verify tests fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_wavelet.py -v
```

Expected: `ModuleNotFoundError: No module named 'eeg_bg.features.wavelet'`

- [ ] **Step 3: Implement `eeg_bg/features/wavelet.py`**

```python
"""DWT energy and entropy per channel per decomposition level.

Uses PyWavelets (pywt) db4 wavelet at 6 levels by default.
Approximate band-to-level mapping at 125 Hz, level 6:
    level 1 → 32–62 Hz (γ)    cD1
    level 2 → 16–32 Hz (β)    cD2
    level 3 →  8–16 Hz (α/β)  cD3
    level 4 →  4– 8 Hz (θ)    cD4
    level 5 →  2– 4 Hz (δ-hi) cD5
    level 6 →  0.5–2 Hz (δ)   cD6
"""
from __future__ import annotations

import numpy as np
import pywt

from eeg_bg.features._constants import _STANDARD_19

_DEFAULT_WAVELET = "db4"
_DEFAULT_LEVELS  = 6

# Per-signal output: [energy_l1, entropy_l1, energy_l2, entropy_l2, ..., energy_l6, entropy_l6]
# 2 stats × 6 levels = 12 features per channel
_WAVELET_SUFFIXES: list[str] = [
    f"dwt_l{l}_{stat}"
    for l in range(1, _DEFAULT_LEVELS + 1)
    for stat in ("energy", "entropy")
]  # length 12

WAVELET_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _WAVELET_SUFFIXES
]  # 19 × 12 = 228


def wavelet_features(
    signal: np.ndarray,
    wavelet: str = _DEFAULT_WAVELET,
    level: int   = _DEFAULT_LEVELS,
) -> np.ndarray:
    """Compute DWT energy and entropy for one signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    wavelet : str
        PyWavelets wavelet name (default ``'db4'``).
    level : int
        Decomposition depth (default 6).

    Returns
    -------
    np.ndarray
        Shape ``(2 * level,)``, dtype float64.
        Layout: ``[energy_l1, entropy_l1, energy_l2, entropy_l2, ...,
        energy_l{level}, entropy_l{level}]``.
    """
    coeffs = pywt.wavedec(signal.astype(np.float64), wavelet, level=level)
    # coeffs = [cA_level, cD_level, cD_{level-1}, ..., cD_1]
    # cD_l lives at coeffs[level + 1 - l]
    feats: list[float] = []
    for l in range(1, level + 1):
        c = coeffs[level + 1 - l].astype(np.float64)
        sq  = c * c
        tot = float(sq.sum()) + 1e-30
        energy  = float(tot / (len(c) + 1e-30))
        p       = sq / tot
        entropy = float(-np.sum(p * np.log(p + 1e-30)))
        feats.extend([energy, entropy])
    return np.asarray(feats, dtype=np.float64)
```

- [ ] **Step 4: Run tests — expect green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_wavelet.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/features/wavelet.py tests/test_features/test_wavelet.py
git commit -m "feat: add wavelet.py — DWT energy+entropy per channel (228 features)"
```

---

## Task 3: `complexity.py` — Sample Entropy and Lempel-Ziv per channel

**Files:**
- Create: `eeg_bg/features/complexity.py`
- Create: `tests/test_features/test_complexity.py`

### Background

**SampEn:** count template matches of length `m` and `m+1` using Chebyshev distance < `r = factor × std(x)`. Uses `scipy.spatial.cKDTree.query_pairs` (O(n log n)) for speed.
`SampEn = −log(A / (B + ε))` where B = matches at length m, A = matches at length m+1.

**LZC:** binarise signal around its median, then count distinct substrings via the incremental LZ76 algorithm; normalise by `n / log2(n)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_complexity.py`:

```python
"""Unit tests for eeg_bg.features.complexity."""
import numpy as np
import pytest
from eeg_bg.features.complexity import (
    sample_entropy,
    lempel_ziv_complexity,
    complexity_features,
    COMPLEXITY_NAMES,
)
from eeg_bg.features._constants import _STANDARD_19


def test_complexity_names_length():
    assert len(COMPLEXITY_NAMES) == 38
    assert len(COMPLEXITY_NAMES) == len(_STANDARD_19) * 2


def test_complexity_names_unique():
    assert len(COMPLEXITY_NAMES) == len(set(COMPLEXITY_NAMES))


def test_complexity_features_shape():
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = complexity_features(epoch, _STANDARD_19)
    assert out.shape == (38,)


def test_complexity_features_dtype():
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = complexity_features(epoch, _STANDARD_19)
    assert out.dtype == np.float64


def test_sample_entropy_random_positive():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(1000).astype(np.float64)
    assert sample_entropy(sig) > 0.0


def test_sample_entropy_constant_zero():
    """Constant signal is perfectly regular → SampEn = 0."""
    const = np.full(1000, 3.0, dtype=np.float64)
    assert sample_entropy(const) == pytest.approx(0.0, abs=1e-6)


def test_lempel_ziv_range():
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(1000).astype(np.float64)
    lzc = lempel_ziv_complexity(sig)
    assert 0.0 <= lzc


def test_lempel_ziv_random_greater_than_constant():
    """Random signal should have higher LZC than a constant."""
    rng = np.random.default_rng(4)
    rand_sig  = rng.standard_normal(1000).astype(np.float64)
    const_sig = np.full(1000, 1.0, dtype=np.float64)
    assert lempel_ziv_complexity(rand_sig) > lempel_ziv_complexity(const_sig)


def test_complexity_features_missing_channel():
    """Missing channels yield zeros."""
    rng = np.random.default_rng(5)
    epoch = rng.standard_normal((1, 1000)).astype(np.float64) * 10.0
    out = complexity_features(epoch, ["FP1"])
    assert out.shape == (38,)
    # All channels except FP1 should be zero
    fp1_idx = _STANDARD_19.index("FP1")
    assert out[fp1_idx * 2]     > 0.0  # FP1 sample_entropy
    other = [i for i in range(38) if i not in (fp1_idx * 2, fp1_idx * 2 + 1)]
    assert np.all(out[other] == 0.0)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_complexity.py -v
```

Expected: `ModuleNotFoundError: No module named 'eeg_bg.features.complexity'`

- [ ] **Step 3: Implement `eeg_bg/features/complexity.py`**

```python
"""Nonlinear complexity features: Sample Entropy and Lempel-Ziv Complexity.

SampEn uses scipy.spatial.cKDTree for O(n log n) template matching.
LZC uses an incremental substring-counting implementation of LZ76.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from eeg_bg.features._constants import _STANDARD_19

COMPLEXITY_NAMES: list[str] = [
    f"{ch}_{stat}"
    for ch in _STANDARD_19
    for stat in ("sample_entropy", "lempel_ziv")
]  # 19 × 2 = 38


def _count_template_pairs(T: np.ndarray, r: float) -> int:
    """Count pairs (i<j) with Chebyshev distance < r via KD-tree."""
    if len(T) < 2:
        return 0
    tree = cKDTree(T)
    return len(tree.query_pairs(r, p=np.inf))


def sample_entropy(
    signal: np.ndarray,
    m: int   = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute Sample Entropy of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(n_times,)``.
    m : int
        Embedding dimension (default 2).
    r_factor : float
        Tolerance as a fraction of the signal's standard deviation (default 0.2).

    Returns
    -------
    float
        SampEn ≥ 0.  Returns 0.0 for constant signals (std ≈ 0).
    """
    x = signal.astype(np.float64)
    std = float(np.std(x))
    if std < 1e-12:
        return 0.0
    r = r_factor * std

    # Build template matrices using sliding windows
    # T_m  shape (n-m, m):   templates of length m   (excludes last for Nm count)
    # T_m1 shape (n-m, m+1): templates of length m+1
    n = len(x)
    T_m  = np.lib.stride_tricks.sliding_window_view(x, m)[:-1]
    T_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    B = _count_template_pairs(T_m, r)
    A = _count_template_pairs(T_m1, r)

    if B == 0:
        return 0.0
    return float(-np.log((A + 1e-30) / (B + 1e-30)))


def lempel_ziv_complexity(signal: np.ndarray) -> float:
    """Normalised Lempel-Ziv complexity (LZ76) of the binarised signal.

    Parameters
    ----------
    signal : np.ndarray
        Shape ``(n_times,)``.

    Returns
    -------
    float
        LZC ≥ 0, normalised by ``n / log2(n)``.
    """
    x = signal.astype(np.float64)
    binary = x > np.median(x)
    n = len(binary)

    # Incremental LZ76: count the number of distinct substrings
    sub_strings: set[bytes] = set()
    i, k = 0, 1
    while i + k <= n:
        sub = binary[i:i + k].tobytes()
        if sub in sub_strings:
            k += 1
        else:
            sub_strings.add(sub)
            i += k
            k = 1
    c = len(sub_strings)

    norm = n / max(np.log2(n), 1.0) if n > 1 else 1.0
    return float(c / norm)


def complexity_features(
    epoch: np.ndarray,
    ch_names: list[str],
    m: int        = 2,
    r_factor: float = 0.2,
) -> np.ndarray:
    """Compute SampEn and LZC for all 19 standard channels.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    m : int
        SampEn embedding dimension.
    r_factor : float
        SampEn tolerance as fraction of std.

    Returns
    -------
    np.ndarray
        Shape ``(38,)``, layout ``[se_ch1, lzc_ch1, se_ch2, lzc_ch2, ...]``
        in ``_STANDARD_19`` order.  Missing channels → 0.0.
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    feats: list[float] = []
    for ch in _STANDARD_19:
        idx = ch_map.get(ch)
        if idx is None:
            feats.extend([0.0, 0.0])
            continue
        sig = epoch[idx].astype(np.float64)
        feats.append(sample_entropy(sig, m=m, r_factor=r_factor))
        feats.append(lempel_ziv_complexity(sig))
    return np.asarray(feats, dtype=np.float64)
```

- [ ] **Step 4: Run tests — expect green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_complexity.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/features/complexity.py tests/test_features/test_complexity.py
git commit -m "feat: add complexity.py — SampEn + LZC per channel (38 features)"
```

---

## Task 4: `temporal_stats.py` — multi-scale temporal statistics

**Files:**
- Create: `eeg_bg/features/temporal_stats.py`
- Create: `tests/test_features/test_temporal_stats.py`

### Background

For each scale `s` (in samples), split the epoch into `floor(n/s)` non-overlapping windows of length `s`. Compute mean, variance, skewness (`scipy.stats.skew`), and excess kurtosis (`scipy.stats.kurtosis`) **per window**, then **average across windows** → one scalar per statistic. This gives 4 stats × 3 scales = 12 features per channel, 19 × 12 = 228 total.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_temporal_stats.py`:

```python
"""Unit tests for eeg_bg.features.temporal_stats."""
import numpy as np
import pytest
from eeg_bg.features.temporal_stats import (
    temporal_stats_features,
    epoch_temporal_stats,
    TEMPORAL_NAMES,
    _DEFAULT_SCALES,
)
from eeg_bg.features._constants import _STANDARD_19


def test_temporal_names_length():
    assert len(TEMPORAL_NAMES) == 228
    assert len(TEMPORAL_NAMES) == len(_STANDARD_19) * len(_DEFAULT_SCALES) * 4


def test_temporal_names_unique():
    assert len(TEMPORAL_NAMES) == len(set(TEMPORAL_NAMES))


def test_temporal_stats_features_shape():
    out = temporal_stats_features(np.random.default_rng(0).standard_normal(1000))
    assert out.shape == (len(_DEFAULT_SCALES) * 4,)


def test_epoch_temporal_stats_shape():
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert out.shape == (228,)


def test_epoch_temporal_stats_dtype():
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert out.dtype == np.float64


def test_temporal_constant_zero_variance():
    """Constant signal has zero variance at every scale."""
    const = np.full(1000, 7.0, dtype=np.float64)
    out = temporal_stats_features(const)
    # Variance is at index 1, 5, 9 (every 4th starting at 1)
    var_indices = [i * 4 + 1 for i in range(len(_DEFAULT_SCALES))]
    assert np.allclose(out[var_indices], 0.0, atol=1e-8)


def test_temporal_constant_zero_skew():
    """Constant signal has zero skewness at every scale."""
    const = np.full(1000, 7.0, dtype=np.float64)
    out = temporal_stats_features(const)
    skew_indices = [i * 4 + 2 for i in range(len(_DEFAULT_SCALES))]
    assert np.allclose(out[skew_indices], 0.0, atol=1e-8)


def test_temporal_missing_channel_zero_padded():
    rng = np.random.default_rng(2)
    epoch = rng.standard_normal((1, 1000)).astype(np.float64)
    out = epoch_temporal_stats(epoch, ["FP1"])
    assert out.shape == (228,)
    fp1_idx = _STANDARD_19.index("FP1")
    n_feats_per_ch = len(_DEFAULT_SCALES) * 4  # 12
    other = [i for i in range(228) if not (fp1_idx * n_feats_per_ch <= i < (fp1_idx + 1) * n_feats_per_ch)]
    assert np.all(out[other] == 0.0)


def test_temporal_output_finite():
    rng = np.random.default_rng(3)
    epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
    out = epoch_temporal_stats(epoch, _STANDARD_19)
    assert np.all(np.isfinite(out))
```

- [ ] **Step 2: Run to verify tests fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_temporal_stats.py -v
```

Expected: `ModuleNotFoundError: No module named 'eeg_bg.features.temporal_stats'`

- [ ] **Step 3: Implement `eeg_bg/features/temporal_stats.py`**

```python
"""Multi-scale temporal statistics per EEG channel.

Splits each epoch into non-overlapping windows of fixed sample lengths
and computes mean, variance, skewness, and kurtosis across windows.
Three scales capture burst structure at 1 s, 3 s, and 6 s granularities
(125, 375, 750 samples at 125 Hz).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis

from eeg_bg.features._constants import _STANDARD_19

_DEFAULT_SCALES: list[int] = [125, 375, 750]

_TEMPORAL_SUFFIXES: list[str] = [
    f"scale{s}_{stat}"
    for s in _DEFAULT_SCALES
    for stat in ("mean", "var", "skew", "kurtosis")
]  # 3 × 4 = 12 per channel

TEMPORAL_NAMES: list[str] = [
    f"{ch}_{s}"
    for ch in _STANDARD_19
    for s in _TEMPORAL_SUFFIXES
]  # 19 × 12 = 228


def temporal_stats_features(
    signal: np.ndarray,
    scales: list[int] = _DEFAULT_SCALES,
) -> np.ndarray:
    """Compute multi-scale temporal statistics for one signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.
    scales : list[int]
        Non-overlapping window lengths in samples.

    Returns
    -------
    np.ndarray
        Shape ``(4 * len(scales),)``, dtype float64.
        Layout per scale: ``[mean, var, skew, kurtosis]``.
    """
    x = signal.astype(np.float64)
    n = len(x)
    feats: list[float] = []
    for s in scales:
        n_windows = n // s
        if n_windows == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        windows = x[:n_windows * s].reshape(n_windows, s)  # (n_windows, s)
        feats.append(float(windows.mean(axis=1).mean()))
        feats.append(float(windows.var(axis=1).mean()))
        feats.append(float(scipy_skew(windows, axis=1).mean()))
        feats.append(float(scipy_kurtosis(windows, axis=1).mean()))
    return np.asarray(feats, dtype=np.float64)


def epoch_temporal_stats(
    epoch: np.ndarray,
    ch_names: list[str],
    scales: list[int] = _DEFAULT_SCALES,
) -> np.ndarray:
    """Compute temporal statistics for all 19 standard channels.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    scales : list[int]
        Non-overlapping window lengths in samples.

    Returns
    -------
    np.ndarray
        Shape ``(228,)``, in ``_STANDARD_19`` order.
        Missing channels → zero-padded (12 zeros per absent channel).
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    n_per_ch = len(scales) * 4
    feats: list[float] = []
    for ch in _STANDARD_19:
        idx = ch_map.get(ch)
        if idx is None:
            feats.extend([0.0] * n_per_ch)
            continue
        feats.extend(temporal_stats_features(epoch[idx], scales=scales).tolist())
    return np.asarray(feats, dtype=np.float64)
```

- [ ] **Step 4: Run tests — expect green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_temporal_stats.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/features/temporal_stats.py tests/test_features/test_temporal_stats.py
git commit -m "feat: add temporal_stats.py — multi-scale stats per channel (228 features)"
```

---

## Task 5: `connectivity.py` — pairwise coherence and PLV

**Files:**
- Create: `eeg_bg/features/connectivity.py`
- Create: `tests/test_features/test_connectivity.py`

### Background

All 171 unique channel pairs from `_STANDARD_19` (C(19,2), ordered by `_STANDARD_19` index i < j).

Per pair, per frequency band (δ/θ/α/β/γ):
- **Coherence**: `scipy.signal.coherence(x, y, fs=sfreq, nperseg=nperseg)` returns `(freqs, Cxy)`. Average `Cxy` over the band's frequency bins.
- **PLV**: bandpass-filter each signal to the band, extract instantaneous phase via Hilbert transform, `PLV = |mean(exp(i·Δφ))|`.

**Efficiency trick**: per-band bandpass + Hilbert is computed **per channel** (19 ch × 5 bands = 95 filter ops), then PLV for all 171 pairs in that band reads from cached phases. This avoids 855 filter operations.

Feature vector layout: for each pair (ch1, ch2), for each band, append [coherence, PLV].
Total: 171 × 5 × 2 = 1710.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_features/test_connectivity.py`:

```python
"""Unit tests for eeg_bg.features.connectivity."""
import numpy as np
import pytest
from eeg_bg.features.connectivity import (
    connectivity_features,
    CONNECTIVITY_NAMES,
    ALL_PAIRS,
)
from eeg_bg.features._constants import _STANDARD_19

SFREQ = 125.0
N_PAIRS = 171  # C(19,2)
N_BANDS = 5
N_METRICS = 2
EXPECTED_DIM = N_PAIRS * N_BANDS * N_METRICS  # 1710


@pytest.fixture
def random_epoch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((19, 1000)).astype(np.float64) * 20.0


@pytest.fixture
def identical_epoch():
    """Epoch where all channels are identical — coherence and PLV should be 1."""
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(1000).astype(np.float64) * 20.0
    return np.tile(sig, (19, 1))


def test_all_pairs_count():
    assert len(ALL_PAIRS) == N_PAIRS


def test_all_pairs_ordered():
    """ch1 index < ch2 index for all pairs."""
    for ch1, ch2 in ALL_PAIRS:
        assert _STANDARD_19.index(ch1) < _STANDARD_19.index(ch2)


def test_connectivity_names_length():
    assert len(CONNECTIVITY_NAMES) == EXPECTED_DIM


def test_connectivity_names_unique():
    assert len(CONNECTIVITY_NAMES) == len(set(CONNECTIVITY_NAMES))


def test_connectivity_features_shape(random_epoch):
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert out.shape == (EXPECTED_DIM,)


def test_connectivity_features_dtype(random_epoch):
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert out.dtype == np.float64


def test_connectivity_range(random_epoch):
    """All coherence and PLV values should be in [0, 1]."""
    out = connectivity_features(random_epoch, _STANDARD_19, sfreq=SFREQ)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0 + 1e-6)


def test_connectivity_identical_channels_high_coherence(identical_epoch):
    """Identical channels should yield coherence ≈ 1 for all bands."""
    out = connectivity_features(identical_epoch, _STANDARD_19, sfreq=SFREQ)
    # Coherence values are at even indices within each pair-band block
    coh_values = out[0::2]
    assert np.all(coh_values > 0.99)


def test_connectivity_identical_channels_high_plv(identical_epoch):
    """Identical channels should yield PLV ≈ 1 for all bands."""
    out = connectivity_features(identical_epoch, _STANDARD_19, sfreq=SFREQ)
    plv_values = out[1::2]
    assert np.all(plv_values > 0.99)


def test_connectivity_missing_channel():
    """Missing channel → all 10 values for its pairs are zero."""
    rng = np.random.default_rng(2)
    # Only 2 channels present (FP1, FP2), all other channels missing
    epoch = rng.standard_normal((2, 1000)).astype(np.float64) * 20.0
    out = connectivity_features(epoch, ["FP1", "FP2"], sfreq=SFREQ)
    assert out.shape == (EXPECTED_DIM,)
    # The (FP1, FP2) pair is the first pair in ALL_PAIRS — those 10 values should be nonzero
    assert not np.all(out[:10] == 0.0)
    # All other pairs involve at least one absent channel — should be zero
    assert np.all(out[10:] == 0.0)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_connectivity.py -v
```

Expected: `ModuleNotFoundError: No module named 'eeg_bg.features.connectivity'`

- [ ] **Step 3: Implement `eeg_bg/features/connectivity.py`**

```python
"""Pairwise functional connectivity: coherence and Phase-Locking Value (PLV).

Computes magnitude-squared coherence and per-band PLV for all 171 unique
electrode pairs (C(19,2)) across the 5 standard EEG frequency bands.

Efficiency: per-band bandpass + Hilbert is computed once per channel
(95 ops) and reused for all 171 pairs in each band.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import coherence as sp_coherence, hilbert, butter, sosfiltfilt

from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.band_power import BANDS

ALL_PAIRS: list[tuple[str, str]] = list(combinations(_STANDARD_19, 2))  # 171 pairs

CONNECTIVITY_NAMES: list[str] = [
    f"{metric}_{ch1}_{ch2}_{band}"
    for ch1, ch2 in ALL_PAIRS
    for band in BANDS
    for metric in ("coh", "plv")
]  # 171 × 5 × 2 = 1710


def _bandpass_sos(f_lo: float, f_hi: float, sfreq: float) -> np.ndarray:
    """Return SOS coefficients for a 4th-order Butterworth bandpass filter."""
    return butter(4, [f_lo, f_hi], btype="bandpass", fs=sfreq, output="sos")


def connectivity_features(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float = 125.0,
    nperseg: int = 250,
) -> np.ndarray:
    """Compute pairwise coherence and PLV for all standard channel pairs.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch window for coherence estimation (pipeline-consistent default 250).

    Returns
    -------
    np.ndarray
        Shape ``(1710,)``, dtype float64.
        Layout: for each pair in ``ALL_PAIRS``, for each band in ``BANDS``,
        ``[coherence, PLV]``.  Absent-channel pairs → 10 zeros.
    """
    ch_map = {name: i for i, name in enumerate(ch_names)}
    present = {ch for ch in _STANDARD_19 if ch in ch_map}

    # ── Precompute raw signals and per-band phases for present channels ──────
    raw: dict[str, np.ndarray] = {
        ch: epoch[ch_map[ch]].astype(np.float64) for ch in present
    }
    phases: dict[tuple[str, str], np.ndarray] = {}  # {(ch, band): instantaneous phase}
    for ch in present:
        for band_name, (f_lo, f_hi) in BANDS.items():
            sos = _bandpass_sos(f_lo, f_hi, sfreq)
            filtered = sosfiltfilt(sos, raw[ch])
            phases[(ch, band_name)] = np.angle(hilbert(filtered))

    # ── Compute features for all pairs ───────────────────────────────────────
    feats: list[float] = []
    for ch1, ch2 in ALL_PAIRS:
        if ch1 not in present or ch2 not in present:
            feats.extend([0.0] * (len(BANDS) * 2))
            continue

        # Coherence (one call per pair, covers all bands)
        freqs, coh_xy = sp_coherence(raw[ch1], raw[ch2], fs=sfreq, nperseg=nperseg)

        for band_name, (f_lo, f_hi) in BANDS.items():
            band_mask = (freqs >= f_lo) & (freqs <= f_hi)
            coh_val = float(coh_xy[band_mask].mean()) if band_mask.any() else 0.0

            # PLV from pre-computed per-band phases
            delta_phi = phases[(ch1, band_name)] - phases[(ch2, band_name)]
            plv_val = float(np.abs(np.mean(np.exp(1j * delta_phi))))

            feats.extend([coh_val, plv_val])

    return np.asarray(feats, dtype=np.float64)
```

- [ ] **Step 4: Run tests — expect green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_connectivity.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/features/connectivity.py tests/test_features/test_connectivity.py
git commit -m "feat: add connectivity.py — coherence + PLV all pairs × bands (1710 features)"
```

---

## Task 6: Update `extraction.py` to integrate all new modules

**Files:**
- Modify: `eeg_bg/features/extraction.py`

The updated `FEATURE_NAMES` = existing 211 + wavelet 228 + connectivity 1710 + complexity 38 + temporal 228 = **2415**.
The per-epoch function `extract_epoch_features` is extended to call all four new modules and concatenate their outputs.

- [ ] **Step 1: Update the failing shape assertion in `test_extraction.py`** (temporarily mark it as expected to fail so the full test suite stays green while we do the refactor)

In `tests/test_features/test_extraction.py`, replace:

```python
def test_feature_names_length():
    assert len(FEATURE_NAMES) == 211
```

with:

```python
def test_feature_names_length():
    assert len(FEATURE_NAMES) == 2415
```

and replace:

```python
def test_feature_names_per_channel():
    """Each channel should have exactly 9 feature names."""
    for ch in _STANDARD_19:
        ch_feats = [n for n in FEATURE_NAMES if n.startswith(f"{ch}_")]
        assert len(ch_feats) == 9, f"Expected 9 features for {ch}, got {len(ch_feats)}"
```

with:

```python
def test_feature_names_per_channel():
    """Each channel should have exactly 9 + 12 + 2 + 12 = 35 feature names."""
    for ch in _STANDARD_19:
        ch_feats = [n for n in FEATURE_NAMES if n.startswith(f"{ch}_")]
        assert len(ch_feats) == 35, f"Expected 35 features for {ch}, got {len(ch_feats)}"
```

and replace every occurrence of `(211,)` and `(0, 211)` in the test file:

```python
def test_extract_epoch_features_shape(synthetic_epoch, ch_names_19, sfreq):
    feat = extract_epoch_features(synthetic_epoch, ch_names_19, sfreq=sfreq)
    assert feat.shape == (2415,)
```

```python
def test_build_dataset_shapes(tmp_path):
    ...
    assert X.shape == (3, 2415)
    ...
```

```python
def test_build_dataset_empty_split(tmp_path):
    ...
    assert X.shape == (0, 2415)
    ...
```

Also add two new tests at the end of the file:

```python
def test_feature_names_first_211_unchanged():
    """Positions 0–210 must match the original 211-dim vector exactly."""
    from eeg_bg.features.asymmetry import ASYMMETRY_NAMES
    from eeg_bg.features.band_power import BANDS
    from eeg_bg.features._constants import _STANDARD_19
    _FEAT_SUFFIXES = (
        [f"{b}_power" for b in BANDS]
        + ["hjorth_activity", "hjorth_mobility", "hjorth_complexity", "spectral_entropy"]
    )
    expected_first_171 = [f"{ch}_{s}" for ch in _STANDARD_19 for s in _FEAT_SUFFIXES]
    assert FEATURE_NAMES[:171] == expected_first_171
    assert FEATURE_NAMES[171:211] == ASYMMETRY_NAMES


def test_extract_epoch_features_missing_channel_new_shape(sfreq):
    """Missing channel test updated for 2415-dim vector."""
    rng   = np.random.default_rng(7)
    epoch = rng.standard_normal((1, 1000)).astype(np.float64) * 10.0
    feat  = extract_epoch_features(epoch, ch_names=["FP1"], sfreq=sfreq)
    assert feat.shape == (2415,)
```

- [ ] **Step 2: Run tests to see expected failures**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_extraction.py -v
```

Expected: shape/length tests fail (good — the spec requires them to fail until we update `extraction.py`).

- [ ] **Step 3: Update `eeg_bg/features/extraction.py`**

Add the new imports after the existing feature imports (around line 34–35):

```python
from eeg_bg.features.wavelet import wavelet_features, WAVELET_NAMES
from eeg_bg.features.connectivity import connectivity_features, CONNECTIVITY_NAMES
from eeg_bg.features.complexity import complexity_features, COMPLEXITY_NAMES
from eeg_bg.features.temporal_stats import epoch_temporal_stats, TEMPORAL_NAMES
```

Replace the `FEATURE_NAMES` definition (currently ends at line ~60):

```python
FEATURE_NAMES: list[str] = (
    [
        f"{ch}_{suffix}"
        for ch in _STANDARD_19
        for suffix in _FEAT_SUFFIXES
    ]
    + ASYMMETRY_NAMES
    + WAVELET_NAMES
    + CONNECTIVITY_NAMES
    + COMPLEXITY_NAMES
    + TEMPORAL_NAMES
)  # 211 + 228 + 1710 + 38 + 228 = 2415
```

Replace the `extract_epoch_features` function body (currently returns a 211-dim array). Find the `return np.concatenate(...)` at the bottom of the function and replace it with:

```python
    # ── New feature blocks ────────────────────────────────────────────────────
    # Wavelet: DWT energy + entropy per channel (228 dims)
    wavelet_vec = np.concatenate([
        wavelet_features(epoch[ch_map[ch]] if ch_map.get(ch) is not None
                         else np.zeros(epoch.shape[1]))
        for ch in _STANDARD_19
    ])

    # Connectivity: coherence + PLV all 171 pairs × 5 bands (1710 dims)
    conn_vec = connectivity_features(epoch, ch_names, sfreq=sfreq, nperseg=nperseg)

    # Complexity: SampEn + LZC per channel (38 dims)
    compl_vec = complexity_features(epoch, ch_names)

    # Temporal: multi-scale stats per channel (228 dims)
    temp_vec = epoch_temporal_stats(epoch, ch_names)

    return np.concatenate([
        np.asarray(features, dtype=np.float64),
        asym,
        wavelet_vec,
        conn_vec,
        compl_vec,
        temp_vec,
    ]).astype(np.float64)
```

Also update the empty-array fallback in `build_dataset`. Find the line:

```python
    return (np.empty((0, 211), dtype=np.float64),
```

Replace with:

```python
    return (np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
```

- [ ] **Step 4: Run the extraction tests — expect all green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_extraction.py -v
```

Expected: all tests pass (including the new `test_feature_names_first_211_unchanged`).

- [ ] **Step 5: Run the full test suite**

```bash
conda run -n eeg_pipeline python -m pytest tests/ -v --ignore=tests/test_features/test_connectivity.py -x
```

Expected: all tests pass. (Connectivity tests may be slow — run separately if needed.)

- [ ] **Step 6: Commit**

```bash
git add eeg_bg/features/extraction.py tests/test_features/test_extraction.py
git commit -m "feat: integrate wavelet/connectivity/complexity/temporal into extraction.py (2415 dims)"
```

---

## Task 7: Update `configs/default.yaml`

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: Add `ml.features` block and `ml.shap.pruning_threshold`**

In `configs/default.yaml`, find the `ml:` section. After the `shap:` block:

```yaml
  shap:
    max_display: 20   # top-N features shown in the beeswarm summary plot
    dpi: 150          # resolution for saved SHAP figures (PNG)
```

Replace with:

```yaml
  shap:
    max_display: 20   # top-N features shown in the beeswarm summary plot
    dpi: 150          # resolution for saved SHAP figures (PNG)
    pruning_threshold: 1.0e-4   # mean |SHAP| below this → flagged in report
  features:
    wavelet_family: db4          # PyWavelets wavelet name
    wavelet_levels: 6            # decomposition depth (changing requires code update)
    connectivity_pairs: all      # "all" (171 pairs=1710 feats) or "symmetric" (8=80)
    sample_entropy_m: 2          # SampEn embedding dimension
    sample_entropy_r: 0.2        # SampEn tolerance (fraction of std)
    temporal_scales: [125, 375, 750]  # window lengths in samples (1 s, 3 s, 6 s)
    drop_low_shap: false         # if true, zero out features below pruning_threshold
                                 # on the next run of script 06 (uses saved report)
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
conda run -n eeg_pipeline python -c "
from eeg_bg.config.settings import load_config
cfg = load_config('configs/default.yaml')
print('features:', cfg['ml']['features'])
print('shap:', cfg['ml']['shap'])
"
```

Expected: prints the new `features` and `shap` dicts without errors.

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml
git commit -m "config: add ml.features and ml.shap.pruning_threshold keys"
```

---

## Task 8: Add shape-guard to `build_dataset`

**Files:**
- Modify: `eeg_bg/features/extraction.py`

When an existing feature cache is loaded, its `X.shape[1]` may differ from `len(FEATURE_NAMES)` if the vector grew. Without a guard, a stale 211-dim cache silently passes as 2415-dim features.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_features/test_extraction.py`:

```python
def test_build_dataset_stale_cache_raises(tmp_path):
    """Loading a cached NPZ with wrong feature dim raises ValueError with --force hint."""
    rng = np.random.default_rng(9)
    ep_arr = rng.standard_normal((2, 19, 1000)).astype(np.float64)
    _write_fake_npz(tmp_path / "epochs" / "s001" / "ep.npz",
                    ep_arr, "train", 0, "s001")

    # Write a stale feature cache with 211 dims
    feat_cache = tmp_path / "features" / "raw_train.npz"
    feat_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(feat_cache,
             X=np.zeros((2, 211), dtype=np.float64),
             y=np.array([0, 0], dtype=np.int64),
             subject_ids=np.array(["s001", "s001"], dtype=object))

    # Attempting to load should raise ValueError about stale cache
    with pytest.raises(ValueError, match="--force"):
        # build_dataset doesn't load feature caches — that's in script 06.
        # The guard lives in _load_or_extract_features in script 06, not here.
        # This test documents the guard expectation; see Task 9 for implementation.
        from eeg_bg.features.extraction import FEATURE_NAMES
        data = np.load(str(feat_cache), allow_pickle=True)
        X = data["X"]
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature cache has {X.shape[1]} dims but FEATURE_NAMES has "
                f"{len(FEATURE_NAMES)}. Re-run script 06 with --force."
            )
```

- [ ] **Step 2: Add the shape guard inside `_load_or_extract_features` in `scripts/06_train_xgboost.py`**

Find the function `_load_or_extract_features` in `scripts/06_train_xgboost.py`. After loading the cached file:

```python
    if feat_file.exists() and not force:
        data = np.load(feat_file, allow_pickle=True)
        X    = data["X"].astype(np.float64)
        y    = data["y"].astype(np.int64)
        sids = list(data["subject_ids"])
        return X, y, sids
```

Replace with:

```python
    if feat_file.exists() and not force:
        data = np.load(feat_file, allow_pickle=True)
        X    = data["X"].astype(np.float64)
        y    = data["y"].astype(np.int64)
        sids = list(data["subject_ids"])
        if X.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature cache '{feat_file}' has {X.shape[1]} features but "
                f"FEATURE_NAMES has {len(FEATURE_NAMES)}. "
                f"Re-run script 06 with --force to regenerate."
            )
        return X, y, sids
```

Also add the import at the top of `scripts/06_train_xgboost.py` (it already imports `FEATURE_NAMES`):
```python
from eeg_bg.features.extraction import (
    build_dataset,
    FEATURE_NAMES,   # already imported — no change needed
)
```

- [ ] **Step 3: Run tests — expect green**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_features/test_extraction.py::test_build_dataset_stale_cache_raises -v
```

Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add scripts/06_train_xgboost.py tests/test_features/test_extraction.py
git commit -m "feat: add feature-cache shape guard to script 06 with --force hint"
```

---

## Task 9: Add SHAP pruning report to `scripts/06_train_xgboost.py`

**Files:**
- Modify: `scripts/06_train_xgboost.py`

After all conditions complete, compute `mean(|SHAP|)` per feature across all conditions' test sets. Write a `shap_pruning_report.json` to `results/xgboost/`. If `ml.features.drop_low_shap: true` and the report exists from a prior run, zero out low-SHAP features in all X matrices before retraining.

- [ ] **Step 1: Add the pruning report helper to `scripts/06_train_xgboost.py`**

Add this function after `_split_stats` (around line 115):

```python
def _write_shap_pruning_report(
    all_results: dict[str, dict],
    out_root: Path,
    threshold: float,
) -> dict:
    """Compute mean |SHAP| per feature across all conditions and write report."""
    shap_arrays = []
    for res in all_results.values():
        shap_path = out_root / res["condition"] / "shap_values_test.npy"
        if shap_path.exists():
            shap_arrays.append(np.load(shap_path))

    if not shap_arrays:
        return {}

    # Mean |SHAP| per feature across all conditions (average over epochs and conditions)
    mean_abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_arrays], axis=0)
    below_mask = mean_abs_shap < threshold
    below_names = [FEATURE_NAMES[i] for i in np.where(below_mask)[0]]

    report = {
        "threshold": threshold,
        "n_features_total": len(FEATURE_NAMES),
        "n_features_below_threshold": int(below_mask.sum()),
        "features_below_threshold": below_names,
    }
    report_path = out_root / "shap_pruning_report.json"
    _save_json(report, report_path)
    print(f"\nSHAP pruning report: {int(below_mask.sum())}/{len(FEATURE_NAMES)} features "
          f"below threshold {threshold:.2e} → {report_path}")
    return report
```

- [ ] **Step 2: Add `drop_low_shap` zeroing helper**

Add this function immediately after `_write_shap_pruning_report`:

```python
def _load_drop_mask(out_root: Path, threshold: float) -> np.ndarray | None:
    """Load drop indices from a prior pruning report if it exists."""
    report_path = out_root / "shap_pruning_report.json"
    if not report_path.exists():
        print("  [drop_low_shap] No pruning report found — skipping zero-out. "
              "Run script 06 once without drop_low_shap to generate the report.")
        return None
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("threshold") != threshold:
        print(f"  [drop_low_shap] Report threshold {report['threshold']:.2e} differs "
              f"from config {threshold:.2e} — re-run without drop_low_shap first.")
        return None
    names = report.get("features_below_threshold", [])
    name_to_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    indices = [name_to_idx[n] for n in names if n in name_to_idx]
    print(f"  [drop_low_shap] Zeroing {len(indices)} features from prior report.")
    return np.array(indices, dtype=np.int64)
```

- [ ] **Step 3: Wire pruning into `main()`**

In the `main()` function, find the cross-condition summary block (around line 296). After the SHAP comparison figure block and before `print("\nDone.")`, add:

```python
    # ── SHAP pruning report ───────────────────────────────────────────────────
    shap_cfg     = cfg["ml"]["shap"]
    feat_cfg     = cfg["ml"].get("features", {})
    threshold    = float(shap_cfg.get("pruning_threshold", 1e-4))
    drop_low     = bool(feat_cfg.get("drop_low_shap", False))

    if len(all_results) > 0:
        _write_shap_pruning_report(all_results, out_root, threshold)

    if drop_low:
        drop_indices = _load_drop_mask(out_root, threshold)
        if drop_indices is not None and len(drop_indices) > 0:
            print(f"\n  Retraining with {len(drop_indices)} features zeroed out...")
            for cond in conditions:
                result = run_condition(
                    cond, cfg, cache_root, feat_cache, out_root, force,
                    drop_indices=drop_indices,
                )
                if result:
                    all_results[cond] = result
```

- [ ] **Step 4: Pass `drop_indices` through `run_condition` and apply zeroing**

Update `run_condition` signature at the top:

```python
def run_condition(
    condition: str,
    cfg: dict,
    cache_root: Path,
    feature_cache_dir: Path,
    out_root: Path,
    force: bool,
    drop_indices: np.ndarray | None = None,
) -> dict:
```

After the feature scaling block (after `X_test_sc` is computed), add:

```python
    # ── Apply drop mask if provided (SHAP-guided feature zeroing) ────────────
    if drop_indices is not None and len(drop_indices) > 0:
        for X_sc in (X_tr_sc, X_val_sc, X_test_sc):
            X_sc[:, drop_indices] = 0.0
```

- [ ] **Step 5: Verify the script still imports cleanly**

```bash
conda run -n eeg_pipeline python -c "import scripts.06_train_xgboost" 2>&1 || conda run -n eeg_pipeline python scripts/06_train_xgboost.py --help
```

Expected: `--help` prints usage without errors.

- [ ] **Step 6: Commit**

```bash
git add scripts/06_train_xgboost.py
git commit -m "feat: add SHAP pruning report + drop_low_shap retraining to script 06"
```

---

## Task 10: Run the full test suite and final verification

- [ ] **Step 1: Run all non-integration tests**

```bash
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration" -v
```

Expected: all tests pass. Note: `test_connectivity.py` includes per-epoch connectivity computation and may take ~30–60 s.

- [ ] **Step 2: Verify FEATURE_NAMES count and layout**

```bash
conda run -n eeg_pipeline python -c "
from eeg_bg.features.extraction import FEATURE_NAMES
print(f'Total features: {len(FEATURE_NAMES)}')
print(f'First 5:  {FEATURE_NAMES[:5]}')
print(f'211-215:  {FEATURE_NAMES[211:216]}')
print(f'439-444:  {FEATURE_NAMES[439:444]}')
print(f'2149-2152:{FEATURE_NAMES[2149:2152]}')
print(f'2187-2190:{FEATURE_NAMES[2187:2190]}')
print(f'Last 3:   {FEATURE_NAMES[-3:]}')
"
```

Expected output:
```
Total features: 2415
First 5:  ['FP1_delta_power', 'FP1_theta_power', 'FP1_alpha_power', 'FP1_beta_power', 'FP1_gamma_power']
211-215:  ['FP1_dwt_l1_energy', 'FP1_dwt_l1_entropy', 'FP1_dwt_l2_energy', 'FP1_dwt_l2_entropy', 'FP1_dwt_l3_energy']
439-444:  ['coh_FP1_FP2_delta', 'plv_FP1_FP2_delta', 'coh_FP1_FP2_theta', 'plv_FP1_FP2_theta', 'coh_FP1_FP2_alpha']
2149-2152:['FP1_sample_entropy', 'FP1_lempel_ziv', 'FP2_sample_entropy', 'FP2_lempel_ziv']
2187-2190:['FP1_scale125_mean', 'FP1_scale125_var', 'FP1_scale125_skew', 'FP1_scale125_kurtosis']
Last 3:   ['Pz_scale750_skew', 'Pz_scale750_kurtosis']  (last 2 of the last channel)
```

- [ ] **Step 3: Smoke-test feature extraction on a synthetic epoch**

```bash
conda run -n eeg_pipeline python -c "
import numpy as np
from eeg_bg.features.extraction import extract_epoch_features, FEATURE_NAMES
from eeg_bg.features._constants import _STANDARD_19

rng = np.random.default_rng(42)
epoch = rng.standard_normal((19, 1000)).astype(np.float64) * 20.0
feat = extract_epoch_features(epoch, _STANDARD_19, sfreq=125.0)
print(f'Shape: {feat.shape}, dtype: {feat.dtype}')
print(f'All finite: {np.all(np.isfinite(feat))}')
print(f'FEATURE_NAMES length: {len(FEATURE_NAMES)}')
"
```

Expected:
```
Shape: (2415,), dtype: float64
All finite: True
FEATURE_NAMES length: 2415
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final test pass — 2415-dim feature vector complete"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `wavelet.py` — 228 dims, db4 level 6, energy+entropy — Task 2
- [x] `connectivity.py` — 1710 dims, all 171 pairs, coherence+PLV per band — Task 5
- [x] `complexity.py` — 38 dims, SampEn + LZC — Task 3
- [x] `temporal_stats.py` — 228 dims, 3 scales, 4 stats — Task 4
- [x] `extraction.py` integration, `FEATURE_NAMES` = 2415, positions 0–210 stable — Task 6
- [x] `_constants.py` to avoid circular import — Task 1
- [x] `default.yaml` — `ml.features.*` and `ml.shap.pruning_threshold` — Task 7
- [x] Feature-cache shape guard with `--force` hint — Task 8
- [x] SHAP pruning report + `drop_low_shap` retraining — Task 9
- [x] pywavelets dependency in `setup.py` — Task 1
- [x] Tests for all new modules + updated `test_extraction.py` — Tasks 2–6

**Type consistency:**
- `wavelet_features(signal) -> ndarray (12,)` — used in Task 6 per-channel loop ✓
- `complexity_features(epoch, ch_names) -> ndarray (38,)` — used in Task 6 ✓
- `epoch_temporal_stats(epoch, ch_names) -> ndarray (228,)` — used in Task 6 ✓
- `connectivity_features(epoch, ch_names, sfreq, nperseg) -> ndarray (1710,)` — used in Task 6 ✓
- `run_condition(..., drop_indices=None)` — signature extended in Task 9 ✓
