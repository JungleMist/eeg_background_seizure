# EEG Wiener Feature Engineering Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-tested, modular Python package (`eeg_bg`) that extracts background EEG epochs from TUEP v3.1.0, applies Wiener decomposition and ICA feature engineering, runs physical verification experiments V1/V2/V3, and produces publication-ready visualizations.

**Architecture:** Layered pipeline (io → preprocessing → decomposition → verification → visualization) with disk caching at two checkpoints. All logic lives in the `eeg_bg` package; scripts and notebooks are thin wrappers. Each module has a single responsibility and is independently testable using synthetic data.

**Tech Stack:** Python 3.x, NumPy 1.26, SciPy 1.17, scikit-learn 1.8, MNE 1.11, pandas, matplotlib, pytest — all in conda env `eeg_pipeline`.

---

## Running Tests

All `pytest` commands below assume the `eeg_pipeline` conda environment. On Windows without conda on PATH:

```powershell
# Option A — full path
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/ -v

# Option B — activate first (run once in PowerShell session)
& "C:\ProgramData\anaconda3\Scripts\activate.bat" eeg_pipeline
pytest tests/ -v
```

In the steps below, `pytest <args>` always means running inside `eeg_pipeline`.

---

## File Map

| File | Responsibility |
|------|---------------|
| `setup.py` | Package installation |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Exclude cache/, results/, *.pyc |
| `configs/default.yaml` | All tunable parameters |
| `eeg_bg/__init__.py` | Package version |
| `eeg_bg/config/settings.py` | YAML loading, path resolution |
| `eeg_bg/io/cache.py` | npz read/write, cache-key hashing |
| `eeg_bg/io/annotation.py` | csv_bi parsing, bckg interval extraction |
| `eeg_bg/io/edf_reader.py` | EDF loading, resampling, channel selection |
| `eeg_bg/io/dataset.py` | Directory traversal, subject index, splits |
| `eeg_bg/preprocessing/epoch.py` | Bandpass filter, epoch slicing, artifact rejection |
| `eeg_bg/preprocessing/reference.py` | AR/LE detection, index filtering |
| `eeg_bg/decomposition/wiener.py` | WienerResult dataclass, cross-PSD, Wiener filter, decompose_epoch/subject |
| `eeg_bg/decomposition/wiener_scalar.py` | Scalar-mode ablation (same interface as wiener.py) |
| `eeg_bg/decomposition/ica.py` | MNE FastICA wrapper, artifact detection, reconstruction |
| `eeg_bg/verification/coherence.py` | Pairwise coherence matrix, V1 run |
| `eeg_bg/verification/transitivity.py` | V2 transitivity error, V3 frequency variation |
| `eeg_bg/visualization/filter_plots.py` | `|h(f)|` and `∠h(f)` curves |
| `eeg_bg/visualization/coherence_plots.py` | Coherence heatmaps, reduction boxplots, signal decomposition |
| `eeg_bg/visualization/verification_plots.py` | V2 histograms, V3 frequency curves, 3-panel ICA-vs-Wiener |
| `tests/conftest.py` | Synthetic EEG fixtures, tmp_cache_dir fixture |
| `tests/test_io/test_cache.py` | Cache key stability, save/load roundtrip |
| `tests/test_io/test_annotation.py` | bckg extraction, seiz buffer, no-seiz case |
| `tests/test_io/test_edf_reader.py` | Channel selection, resample shape (integration) |
| `tests/test_io/test_dataset.py` | Index building, split assignment, no-leakage |
| `tests/test_preprocessing/test_epoch.py` | Epoch count, artifact rejection, bandpass |
| `tests/test_preprocessing/test_reference.py` | Montage dir → reference string |
| `tests/test_decomposition/test_wiener.py` | Filter shape, coherence reduction, midline passthrough |
| `tests/test_decomposition/test_wiener_scalar.py` | Scalar vs frequency same interface |
| `tests/test_decomposition/test_ica.py` | Reconstruction shape, no artifact case |
| `tests/test_verification/test_coherence.py` | V1 output schema, coherence values in [0,1] |
| `tests/test_verification/test_transitivity.py` | V2 error ≥ 0, V3 variation ≥ 0 |
| `scripts/01_extract_epochs.py` | CLI: read EDF → cache epochs |
| `scripts/02_run_wiener.py` | CLI: load epochs → cache Wiener results |
| `scripts/03_run_ica.py` | CLI: load epochs → cache ICA results |
| `scripts/04_run_verification.py` | CLI: load results → run V1/V2/V3 → save CSV |

---

## Task 1: Project Scaffold

**Files:**
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `configs/default.yaml`
- Create: `eeg_bg/__init__.py`
- Create: `eeg_bg/config/__init__.py`, `eeg_bg/io/__init__.py`, `eeg_bg/preprocessing/__init__.py`, `eeg_bg/decomposition/__init__.py`, `eeg_bg/verification/__init__.py`, `eeg_bg/visualization/__init__.py`
- Create: `tests/__init__.py`, `tests/test_io/__init__.py`, `tests/test_preprocessing/__init__.py`, `tests/test_decomposition/__init__.py`, `tests/test_verification/__init__.py`

- [ ] **Step 1: Create `setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="eeg_bg",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

- [ ] **Step 2: Create `requirements.txt`**

```
numpy==1.26.4
scipy==1.17.1
scikit-learn==1.8.0
mne==1.11.0
pandas>=2.0
matplotlib>=3.7
pyyaml>=6.0
pytest>=7.0
tqdm>=4.0
```

- [ ] **Step 3: Create `.gitignore`**

```
cache/
results/
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Create `configs/default.yaml`**

```yaml
paths:
  data_root: "D:/EEGdata/TUEP/v3.1.0"
  cache_dir: "cache"
  results_dir: "results"

dataset:
  reference_scheme: "ar"
  montage_dir: "01_tcp_ar"
  classes:
    epilepsy: "00_epilepsy"
    control: "01_no_epilepsy"

split:
  train: 0.70
  val: 0.10
  test: 0.20
  random_seed: 42

preprocessing:
  target_sfreq: 125
  bandpass: [0.5, 40.0]
  epoch_length_sec: 8.0
  artifact_threshold_uv: 200.0
  seizure_buffer_sec: 30.0

channels:
  standard_19:
    - FP1
    - FP2
    - F3
    - F4
    - F7
    - F8
    - C3
    - C4
    - T3
    - T4
    - T5
    - T6
    - P3
    - P4
    - O1
    - O2
    - Fz
    - Cz
    - Pz
  bilateral_pairs:
    - [FP1, FP2]
    - [F3, F4]
    - [F7, F8]
    - [C3, C4]
    - [T3, T4]
    - [T5, T6]
    - [P3, P4]
    - [O1, O2]
  midline: [Fz, Cz, Pz]

wiener:
  mode: "frequency"
  nperseg: 1000
  freq_resolution_hz: 0.5
  coherence_threshold: 0.15
  freq_band: [0.5, 40.0]

ica:
  n_components: 19
  method: "fastica"
  artifact_corr_threshold: 0.8
  random_state: 42

verification:
  v2_transitivity_amp_threshold: 0.1
  v2_transitivity_phase_threshold: 0.392
  v3_freq_variation_threshold: 0.20
```

- [ ] **Step 5: Create all `__init__.py` files (all empty except `eeg_bg/__init__.py`)**

```python
# eeg_bg/__init__.py
__version__ = "0.1.0"
```

All other `__init__.py` files: empty.

- [ ] **Step 6: Install package in editable mode**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pip install -e .
```

Expected output includes: `Successfully installed eeg-bg-0.1.0`

- [ ] **Step 7: Commit**

```powershell
git add setup.py requirements.txt .gitignore configs/ eeg_bg/ tests/
git commit -m "chore: project scaffold with package structure and config"
```

---

## Task 2: Config Module

**Files:**
- Create: `eeg_bg/config/settings.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
import pytest
from eeg_bg.config.settings import load_config

def test_load_config_returns_dict(tmp_path):
    cfg_text = """
paths:
  data_root: "D:/EEGdata"
  cache_dir: "cache"
  results_dir: "results"
preprocessing:
  target_sfreq: 125
"""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    assert cfg["preprocessing"]["target_sfreq"] == 125

def test_load_config_resolves_relative_paths(tmp_path):
    cfg_text = """
paths:
  data_root: "D:/EEGdata"
  cache_dir: "cache"
  results_dir: "results"
"""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)
    # cache_dir should be resolved to absolute path
    assert Path(cfg["paths"]["cache_dir"]).is_absolute()
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_config.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement `eeg_bg/config/settings.py`**

```python
from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = config_path.parent.parent.resolve()
    for key in ("cache_dir", "results_dir"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            if not p.is_absolute():
                cfg["paths"][key] = str(project_root / p)
    return cfg
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_config.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/config/settings.py tests/test_config.py
git commit -m "feat: config loader with YAML parsing and path resolution"
```

---

## Task 3: Cache Module

**Files:**
- Create: `eeg_bg/io/cache.py`
- Create: `tests/test_io/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io/test_cache.py
import numpy as np
import pytest
from pathlib import Path
from eeg_bg.io.cache import make_cache_key, load_or_compute


def test_make_cache_key_is_stable():
    path = Path("D:/data/subject/file.edf")
    cfg = {"preprocessing": {"target_sfreq": 125, "bandpass": [0.5, 40.0]}}
    key1 = make_cache_key(path, 0.0, cfg)
    key2 = make_cache_key(path, 0.0, cfg)
    assert key1 == key2
    assert len(key1) == 16


def test_make_cache_key_changes_with_params():
    path = Path("D:/data/file.edf")
    cfg1 = {"preprocessing": {"target_sfreq": 125, "bandpass": [0.5, 40.0]}}
    cfg2 = {"preprocessing": {"target_sfreq": 256, "bandpass": [0.5, 40.0]}}
    assert make_cache_key(path, 0.0, cfg1) != make_cache_key(path, 0.0, cfg2)


def test_load_or_compute_saves_and_loads(tmp_path):
    cache_path = tmp_path / "test.npz"
    called = {"n": 0}

    def compute():
        called["n"] += 1
        return {"data": np.array([1.0, 2.0, 3.0]), "label": np.array([0])}

    result1 = load_or_compute(cache_path, compute)
    result2 = load_or_compute(cache_path, compute)

    assert called["n"] == 1  # compute called only once
    np.testing.assert_array_equal(result1["data"], result2["data"])


def test_load_or_compute_force_recompute(tmp_path):
    cache_path = tmp_path / "test.npz"
    called = {"n": 0}

    def compute():
        called["n"] += 1
        return {"data": np.zeros(5)}

    load_or_compute(cache_path, compute)
    load_or_compute(cache_path, compute, force_recompute=True)
    assert called["n"] == 2
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_cache.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/io/cache.py`**

```python
import hashlib
import numpy as np
from pathlib import Path
from typing import Callable


def make_cache_key(edf_path: Path, start_sec: float, cfg: dict) -> str:
    sfreq = cfg["preprocessing"]["target_sfreq"]
    bandpass = cfg["preprocessing"]["bandpass"]
    raw = f"{edf_path}|{start_sec:.4f}|{sfreq}|{bandpass}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_compute(
    cache_path: Path,
    compute_fn: Callable[[], dict],
    force_recompute: bool = False,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return dict(np.load(cache_path, allow_pickle=True))
    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_cache.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/io/cache.py tests/test_io/test_cache.py
git commit -m "feat: disk cache with sha256 key and force-recompute flag"
```

---

## Task 4: Annotation Parser

**Files:**
- Create: `eeg_bg/io/annotation.py`
- Create: `tests/test_io/test_annotation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io/test_annotation.py
import pytest
from pathlib import Path
from eeg_bg.io.annotation import extract_bckg_intervals

CSV_BI_NO_SEIZ = """\
# version = csv_v1.0.0
# bname = test
# duration = 100 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,100.0000,bckg,1.0000
"""

CSV_BI_WITH_SEIZ = """\
# version = csv_v1.0.0
# bname = test
# duration = 120 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,50.0000,bckg,1.0000
TERM,50.0000,60.0000,seiz,1.0000
TERM,60.0000,120.0000,bckg,1.0000
"""

CFG = {"preprocessing": {"seizure_buffer_sec": 30.0}}


def test_no_seizure_returns_full_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_NO_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG)
    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.0, 100.0))


def test_seiz_buffer_clips_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_WITH_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG)
    # seiz [50,60], buffer 30s → exclude [20,90]
    # bckg [0,50] → clip to [0,20]
    # bckg [60,120] → clip to [90,120]
    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((0.0, 20.0))
    assert intervals[1] == pytest.approx((90.0, 120.0))


def test_seiz_fully_covering_bckg_returns_empty(tmp_path):
    content = """\
# version = csv_v1.0.0
# bname = test
# duration = 30 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,10.0000,bckg,1.0000
TERM,10.0000,20.0000,seiz,1.0000
TERM,20.0000,30.0000,bckg,1.0000
"""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(content)
    intervals = extract_bckg_intervals(csv_path, CFG)
    # seiz [10,20], buffer 30 → exclude [-20, 50] → covers all bckg
    assert intervals == []
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_annotation.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/io/annotation.py`**

```python
import csv
from pathlib import Path


def extract_bckg_intervals(
    csv_bi_path: Path, cfg: dict
) -> list[tuple[float, float]]:
    buffer = cfg["preprocessing"]["seizure_buffer_sec"]
    bckg_intervals: list[tuple[float, float]] = []
    seiz_intervals: list[tuple[float, float]] = []

    with open(csv_bi_path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        start = float(row["start_time"])
        stop = float(row["stop_time"])
        label = row["label"].strip()
        if label == "bckg":
            bckg_intervals.append((start, stop))
        elif label == "seiz":
            seiz_intervals.append((start, stop))

    excluded = [(max(0.0, s - buffer), e + buffer) for s, e in seiz_intervals]

    result: list[tuple[float, float]] = []
    for b_start, b_end in bckg_intervals:
        segments = [(b_start, b_end)]
        for ex_start, ex_end in excluded:
            new_segs: list[tuple[float, float]] = []
            for seg_s, seg_e in segments:
                if ex_end <= seg_s or ex_start >= seg_e:
                    new_segs.append((seg_s, seg_e))
                else:
                    if seg_s < ex_start:
                        new_segs.append((seg_s, ex_start))
                    if seg_e > ex_end:
                        new_segs.append((ex_end, seg_e))
            segments = new_segs
        result.extend(segments)

    return result
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_annotation.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/io/annotation.py tests/test_io/test_annotation.py
git commit -m "feat: csv_bi annotation parser with seizure buffer exclusion"
```

---

## Task 5: Shared Test Fixtures (`conftest.py`)

This task must come before the Wiener/ICA/verification tests that depend on synthetic EEG data.

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import numpy as np
import pytest


CH_NAMES_19 = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]

BASE_CFG = {
    "preprocessing": {"target_sfreq": 125.0, "bandpass": [0.5, 40.0]},
    "wiener": {
        "mode": "frequency",
        "nperseg": 1000,
        "coherence_threshold": 0.05,   # Low so decomposition always runs
        "freq_band": [0.5, 40.0],
    },
    "ica": {
        "n_components": 19,
        "method": "fastica",
        "artifact_corr_threshold": 0.8,
        "random_state": 42,
    },
    "channels": {
        "standard_19": CH_NAMES_19,
        "bilateral_pairs": [
            ["FP1", "FP2"], ["F3", "F4"], ["F7", "F8"], ["C3", "C4"],
            ["T3", "T4"], ["T5", "T6"], ["P3", "P4"], ["O1", "O2"],
        ],
        "midline": ["Fz", "Cz", "Pz"],
    },
}


@pytest.fixture
def cfg():
    """Default test configuration."""
    import copy
    return copy.deepcopy(BASE_CFG)


@pytest.fixture
def synthetic_epoch(cfg):
    """
    19-channel, 1000-sample (8s @ 125Hz) synthetic EEG.
    A single broadband point source is mixed into all channels
    with known scalar gains. Independent noise is added.
    High SNR (source_std=50 uV, noise_std=1 uV) so Wiener
    decomposition can reliably remove the shared component.
    """
    rng = np.random.default_rng(42)
    n_ch = len(CH_NAMES_19)
    n_times = 1000

    source = rng.standard_normal(n_times) * 50.0
    gains = rng.uniform(0.5, 1.0, n_ch)
    noise = rng.standard_normal((n_ch, n_times)) * 1.0
    epoch = gains[:, None] * source[None, :] + noise

    return epoch.astype(np.float64), CH_NAMES_19, cfg, gains, source


@pytest.fixture
def synthetic_epochs_batch(cfg):
    """Batch of 5 synthetic epochs for subject-level tests."""
    rng = np.random.default_rng(99)
    n_epochs, n_ch, n_times = 5, 19, 1000
    source = rng.standard_normal((n_epochs, n_times)) * 50.0
    gains = rng.uniform(0.5, 1.0, n_ch)
    noise = rng.standard_normal((n_epochs, n_ch, n_times)) * 1.0
    epochs = gains[None, :, None] * source[:, None, :] + noise
    return epochs.astype(np.float64), CH_NAMES_19, cfg


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
```

- [ ] **Step 2: Verify conftest loads without error**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/ --collect-only -q
```

Expected: No import errors from conftest.py.

- [ ] **Step 3: Commit**

```powershell
git add tests/conftest.py
git commit -m "test: add synthetic EEG fixtures to conftest"
```

---

## Task 6: Preprocessing — Epoch Slicing & Bandpass Filter

**Files:**
- Create: `eeg_bg/preprocessing/epoch.py`
- Create: `tests/test_preprocessing/test_epoch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing/test_epoch.py
import numpy as np
import pytest
from eeg_bg.preprocessing.epoch import slice_epochs, bandpass_filter


def test_slice_epochs_correct_count():
    sfreq = 125.0
    # 24 seconds of data → 3 non-overlapping 8-second epochs
    data = np.zeros((19, int(24 * sfreq)))
    intervals = [(0.0, 24.0)]
    epochs = slice_epochs(data, sfreq, intervals, epoch_len_sec=8.0,
                          artifact_threshold_uv=200.0)
    assert epochs.shape == (3, 19, 1000)


def test_slice_epochs_rejects_artifact():
    sfreq = 125.0
    data = np.zeros((19, 3000))
    data[0, 1000:2000] = 300.0   # epoch index 1 has artifact
    intervals = [(0.0, 24.0)]
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape[0] == 2  # epoch 1 rejected


def test_slice_epochs_empty_when_all_artifact():
    sfreq = 125.0
    data = np.full((19, 2000), 300.0)   # all exceeds threshold
    intervals = [(0.0, 16.0)]
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape == (0, 19, 1000)


def test_slice_epochs_multiple_intervals():
    sfreq = 125.0
    data = np.zeros((19, int(40 * sfreq)))
    intervals = [(0.0, 8.0), (16.0, 24.0)]   # 1 epoch each, gap in between
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape[0] == 2


def test_bandpass_filter_preserves_shape():
    data = np.random.randn(19, 1000)
    filtered = bandpass_filter(data, sfreq=125.0, low=0.5, high=40.0)
    assert filtered.shape == data.shape


def test_bandpass_filter_attenuates_high_freq():
    sfreq = 125.0
    t = np.arange(1000) / sfreq
    # Pure 60 Hz signal (above 40 Hz cutoff)
    signal = np.sin(2 * np.pi * 60 * t)
    data = signal[None, :]   # (1, 1000)
    filtered = bandpass_filter(data, sfreq, low=0.5, high=40.0)
    assert np.std(filtered) < 0.1 * np.std(data)  # strongly attenuated
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_preprocessing/test_epoch.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/preprocessing/epoch.py`**

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    data: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    sos = butter(5, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def slice_epochs(
    data: np.ndarray,
    sfreq: float,
    intervals: list[tuple[float, float]],
    epoch_len_sec: float,
    artifact_threshold_uv: float,
) -> np.ndarray:
    epoch_len = int(epoch_len_sec * sfreq)
    n_ch = data.shape[0]
    epochs: list[np.ndarray] = []

    for start_sec, stop_sec in intervals:
        start_sample = int(start_sec * sfreq)
        stop_sample = int(stop_sec * sfreq)
        pos = start_sample
        while pos + epoch_len <= stop_sample:
            epoch = data[:, pos : pos + epoch_len]
            if np.max(np.abs(epoch)) <= artifact_threshold_uv:
                epochs.append(epoch.copy())
            pos += epoch_len

    if not epochs:
        return np.empty((0, n_ch, epoch_len))
    return np.stack(epochs)
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_preprocessing/test_epoch.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/preprocessing/epoch.py tests/test_preprocessing/test_epoch.py
git commit -m "feat: epoch slicing with artifact rejection and bandpass filter"
```

---

## Task 7: Preprocessing — Reference Detection

**Files:**
- Create: `eeg_bg/preprocessing/reference.py`
- Create: `tests/test_preprocessing/test_reference.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing/test_reference.py
import pandas as pd
import pytest
from eeg_bg.preprocessing.reference import detect_reference, filter_by_reference


def test_detect_reference_ar():
    assert detect_reference("01_tcp_ar") == "ar"
    assert detect_reference("03_tcp_ar_a") == "ar"


def test_detect_reference_le():
    assert detect_reference("02_tcp_le") == "le"
    assert detect_reference("04_tcp_le_a") == "le"


def test_filter_by_reference_keeps_matching():
    df = pd.DataFrame({
        "subject_id": ["s1", "s2", "s3"],
        "reference": ["ar", "le", "ar"],
    })
    result = filter_by_reference(df, "ar")
    assert list(result["subject_id"]) == ["s1", "s3"]
    assert result.index.tolist() == [0, 1]  # reset index
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_preprocessing/test_reference.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/preprocessing/reference.py`**

```python
import pandas as pd


def detect_reference(montage_dir: str) -> str:
    lower = montage_dir.lower()
    if "tcp_ar" in lower:
        return "ar"
    if "tcp_le" in lower:
        return "le"
    return "unknown"


def filter_by_reference(index: pd.DataFrame, scheme: str) -> pd.DataFrame:
    return index[index["reference"] == scheme].reset_index(drop=True)
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_preprocessing/test_reference.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/preprocessing/reference.py tests/test_preprocessing/test_reference.py
git commit -m "feat: reference scheme detection from montage directory name"
```

---

## Task 8: EDF Reader & Dataset Index

**Files:**
- Create: `eeg_bg/io/edf_reader.py`
- Create: `eeg_bg/io/dataset.py`
- Create: `tests/test_io/test_dataset.py`
- Create: `tests/test_io/test_edf_reader.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_io/test_dataset.py
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from eeg_bg.io.dataset import build_subject_index, assign_splits

MOCK_CFG = {
    "paths": {"data_root": "/fake/root"},
    "dataset": {
        "reference_scheme": "ar",
        "montage_dir": "01_tcp_ar",
        "classes": {"epilepsy": "00_epilepsy", "control": "01_no_epilepsy"},
    },
    "split": {"train": 0.7, "val": 0.1, "test": 0.2, "random_seed": 42},
}


def _make_fake_tree(tmp_path):
    """Create fake TUEP directory structure with 4 subjects."""
    for class_dir, label in [("00_epilepsy", 0), ("01_no_epilepsy", 1)]:
        for subj in [f"subj_{class_dir[:2]}_{i:02d}" for i in range(2)]:
            session = tmp_path / class_dir / subj / "s001_2020" / "01_tcp_ar"
            session.mkdir(parents=True)
            edf = session / f"{subj}_s001_t000.edf"
            edf.touch()
    return tmp_path


def test_build_subject_index_finds_edf_files(tmp_path):
    _make_fake_tree(tmp_path)
    cfg = {**MOCK_CFG, "paths": {"data_root": str(tmp_path)}}
    index = build_subject_index(cfg)
    assert len(index) == 4
    assert set(index.columns) >= {"subject_id", "label", "edf_path", "reference"}
    assert set(index["label"].unique()) == {0, 1}


def test_assign_splits_no_subject_leakage(tmp_path):
    _make_fake_tree(tmp_path)
    cfg = {**MOCK_CFG, "paths": {"data_root": str(tmp_path)}}
    index = assign_splits(build_subject_index(cfg), cfg)
    assert "split" in index.columns
    # Each subject is in exactly one split
    for subj, group in index.groupby("subject_id"):
        assert group["split"].nunique() == 1


def test_assign_splits_covers_all_rows(tmp_path):
    _make_fake_tree(tmp_path)
    cfg = {**MOCK_CFG, "paths": {"data_root": str(tmp_path)}}
    index = assign_splits(build_subject_index(cfg), cfg)
    assert index["split"].isna().sum() == 0
    assert set(index["split"].unique()).issubset({"train", "val", "test"})
```

```python
# tests/test_io/test_edf_reader.py
import pytest
# EDF reader requires a real EDF file — mark as integration test
pytestmark = pytest.mark.integration


def test_load_edf_shape_and_units():
    """Run with: pytest tests/test_io/test_edf_reader.py -m integration"""
    import numpy as np
    from pathlib import Path
    from eeg_bg.io.edf_reader import load_edf
    from eeg_bg.config.settings import load_config

    cfg = load_config("configs/default.yaml")
    edf_path = next(
        Path(cfg["paths"]["data_root"]).glob("00_epilepsy/**/01_tcp_ar/*.edf")
    )
    data, ch_names, sfreq = load_edf(edf_path, cfg)
    assert sfreq == cfg["preprocessing"]["target_sfreq"]
    assert data.ndim == 2
    assert data.shape[0] <= 19
    assert np.max(np.abs(data)) < 5000  # μV, not raw volts
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_dataset.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/io/edf_reader.py`**

```python
import mne
import numpy as np
from pathlib import Path


def load_edf(
    edf_path: Path, cfg: dict
) -> tuple[np.ndarray, list[str], float]:
    target_sfreq = cfg["preprocessing"]["target_sfreq"]
    standard_19 = cfg["channels"]["standard_19"]
    low, high = cfg["preprocessing"]["bandpass"]

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    available = [ch for ch in standard_19 if ch in raw.ch_names]
    if not available:
        raise ValueError(
            f"No standard channels found in {edf_path}. "
            f"Available: {raw.ch_names}"
        )
    raw.pick_channels(available)
    raw.filter(low, high, verbose=False)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    data = raw.get_data() * 1e6  # V → μV
    return data, list(raw.ch_names), float(raw.info["sfreq"])
```

- [ ] **Step 4: Implement `eeg_bg/io/dataset.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path


def build_subject_index(cfg: dict) -> pd.DataFrame:
    data_root = Path(cfg["paths"]["data_root"])
    montage_dir = cfg["dataset"]["montage_dir"]
    reference = cfg["dataset"]["reference_scheme"]
    classes = cfg["dataset"]["classes"]

    records = []
    for label_name, folder in classes.items():
        label = 0 if label_name == "epilepsy" else 1
        class_dir = data_root / folder
        if not class_dir.exists():
            continue
        for subject_dir in sorted(class_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                montage_path = session_dir / montage_dir
                if not montage_path.exists():
                    continue
                for edf_file in sorted(montage_path.glob("*.edf")):
                    token_id = edf_file.stem.split("_")[-1]
                    records.append({
                        "subject_id": subject_dir.name,
                        "session_id": session_dir.name,
                        "token_id": token_id,
                        "label": label,
                        "reference": reference,
                        "edf_path": str(edf_file),
                    })
    return pd.DataFrame(records)


def assign_splits(index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["split"]["random_seed"])
    subjects = index["subject_id"].unique().copy()
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * cfg["split"]["train"])
    n_val = int(n * cfg["split"]["val"])
    train_set = set(subjects[:n_train])
    val_set = set(subjects[n_train : n_train + n_val])

    def _split(sid: str) -> str:
        if sid in train_set:
            return "train"
        if sid in val_set:
            return "val"
        return "test"

    index = index.copy()
    index["split"] = index["subject_id"].apply(_split)
    return index
```

- [ ] **Step 5: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_io/test_dataset.py -v
```

Expected: `3 passed`

- [ ] **Step 6: Commit**

```powershell
git add eeg_bg/io/edf_reader.py eeg_bg/io/dataset.py tests/test_io/test_dataset.py tests/test_io/test_edf_reader.py
git commit -m "feat: EDF reader and dataset index builder with subject-level splits"
```

---

## Task 9: Wiener Decomposition (Core)

**Files:**
- Create: `eeg_bg/decomposition/wiener.py`
- Create: `tests/test_decomposition/test_wiener.py`

**Key constraint:** `nperseg` must equal `n_times` (both = 1000 for 8s @ 125Hz). This guarantees that the Welch PSD frequency bins match the `rfft` bins used in filtering, enabling exact frequency-domain multiplication without interpolation.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decomposition/test_wiener.py
import numpy as np
import pytest
from scipy.signal import coherence as scipy_coherence
from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
    apply_wiener_filter,
    decompose_epoch,
    decompose_subject,
)


def test_estimate_cross_psd_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    # Test with a 2-channel subset (a bilateral pair)
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    freqs, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    assert freqs.shape == (nperseg // 2 + 1,)
    assert S.shape == (2, 2, nperseg // 2 + 1)
    # Diagonal must be real positive (auto-PSD)
    assert np.all(S[0, 0].real > 0)
    assert np.allclose(S[0, 0].imag, 0, atol=1e-10)
    # Off-diagonal conjugate symmetry
    np.testing.assert_allclose(S[0, 1], np.conj(S[1, 0]))


def test_compute_wiener_filter_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h = compute_wiener_filter(S, target_idx=0)
    # 2-channel group → 1 reference → h shape (1, n_freqs)
    assert h.shape == (1, nperseg // 2 + 1)
    assert h.dtype == complex


def test_apply_wiener_filter_output_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h = compute_wiener_filter(S, target_idx=0)
    specific, coherent = apply_wiener_filter(pair_data, h, target_idx=0, n_times=pair_data.shape[1])
    assert specific.shape == (pair_data.shape[1],)
    assert coherent.shape == (pair_data.shape[1],)
    # specific + coherent = original
    np.testing.assert_allclose(specific + coherent, pair_data[0], atol=1e-8)


def test_decompose_epoch_reduces_coherence(synthetic_epoch):
    """High-SNR synthetic data: Wiener decomposition should reduce coherence."""
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    # Check FP1 vs FP2 (first bilateral pair)
    fp1_idx = ch_names.index("FP1")
    fp2_idx = ch_names.index("FP2")
    _, coh_pre = scipy_coherence(epoch[fp1_idx], epoch[fp2_idx],
                                  fs=sfreq, nperseg=nperseg)
    _, coh_post = scipy_coherence(result.specific[fp1_idx],
                                   result.specific[fp2_idx],
                                   fs=sfreq, nperseg=nperseg)
    assert np.mean(coh_pre) > np.mean(coh_post)


def test_decompose_epoch_midline_unchanged(synthetic_epoch):
    """Midline channels (Fz, Cz, Pz) must pass through unchanged."""
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    for ch in cfg["channels"]["midline"]:
        idx = ch_names.index(ch)
        np.testing.assert_array_equal(result.specific[idx], epoch[idx])


def test_decompose_epoch_result_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    assert result.raw.shape == epoch.shape
    assert result.specific.shape == epoch.shape
    assert result.coherent.shape == epoch.shape
    assert isinstance(result.filters, dict)
    assert isinstance(result.skipped_pairs, list)


def test_decompose_subject_returns_list(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    results = decompose_subject(epochs, ch_names, "test_subject", cfg)
    assert len(results) == epochs.shape[0]
    assert all(isinstance(r, WienerResult) for r in results)
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_wiener.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/decomposition/wiener.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import csd, welch, coherence as scipy_coherence


@dataclass
class WienerResult:
    subject_id: str
    epoch_idx: int
    raw: np.ndarray           # (n_channels, n_times)
    specific: np.ndarray      # (n_channels, n_times)
    coherent: np.ndarray      # (n_channels, n_times)
    filters: dict             # {pair_key: {ch_name: h array (n_ref, n_freqs)}}
    freqs: np.ndarray         # frequency axis from Welch
    ch_names: list[str]
    skipped_pairs: list[str] = field(default_factory=list)


def estimate_cross_psd(
    data: np.ndarray,   # (n_ch, n_times)
    sfreq: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_ch = data.shape[0]
    freqs, _ = welch(data[0], fs=sfreq, nperseg=nperseg)
    n_freqs = len(freqs)
    S = np.zeros((n_ch, n_ch, n_freqs), dtype=complex)

    for i in range(n_ch):
        _, psd = welch(data[i], fs=sfreq, nperseg=nperseg)
        S[i, i] = psd.astype(complex)
        for j in range(i + 1, n_ch):
            _, cross = csd(data[i], data[j], fs=sfreq, nperseg=nperseg)
            S[i, j] = cross
            S[j, i] = np.conj(cross)
    return freqs, S


def compute_wiener_filter(
    S: np.ndarray,   # (n_ch, n_ch, n_freqs)
    target_idx: int,
) -> np.ndarray:
    n_ch = S.shape[0]
    n_freqs = S.shape[2]
    ref_indices = [i for i in range(n_ch) if i != target_idx]
    n_ref = len(ref_indices)
    h = np.zeros((n_ref, n_freqs), dtype=complex)

    for f in range(n_freqs):
        S_ref = S[np.ix_(ref_indices, ref_indices)][:, :, f]
        s_cross = S[target_idx, ref_indices, f]
        try:
            h[:, f] = np.linalg.solve(S_ref, s_cross)
        except np.linalg.LinAlgError:
            pass  # leave zeros for this frequency bin
    return h


def apply_wiener_filter(
    group_data: np.ndarray,  # (n_ch, n_times) for the group
    h: np.ndarray,           # (n_ref, n_freqs)
    target_idx: int,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    ref_indices = [i for i in range(group_data.shape[0]) if i != target_idx]
    ref_fft = np.fft.rfft(group_data[ref_indices], axis=-1)  # (n_ref, n_freqs)
    coherent_fft = np.sum(h * ref_fft, axis=0)               # (n_freqs,)
    coherent = np.fft.irfft(coherent_fft, n=n_times)
    specific = group_data[target_idx] - coherent
    return specific, coherent


def decompose_epoch(
    epoch: np.ndarray,       # (n_channels, n_times)
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
) -> WienerResult:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    coh_threshold = cfg["wiener"]["coherence_threshold"]
    bilateral_pairs = cfg["channels"]["bilateral_pairs"]
    midline = cfg["channels"]["midline"]
    freq_band = cfg["wiener"]["freq_band"]
    n_times = epoch.shape[1]

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg)
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

    for pair in bilateral_pairs:
        try:
            indices = [ch_names.index(ch) for ch in pair]
        except ValueError:
            skipped.append("-".join(pair))
            continue

        group_data = epoch[indices]

        # Coherence gate
        _, coh = scipy_coherence(group_data[0], group_data[1],
                                  fs=sfreq, nperseg=nperseg)
        if np.max(coh[freq_mask]) < coh_threshold:
            skipped.append("-".join(pair))
            continue

        _, S = estimate_cross_psd(group_data, sfreq, nperseg)
        pair_key = "-".join(pair)
        filters[pair_key] = {}

        for local_idx, (ch, global_idx) in enumerate(zip(pair, indices)):
            h = compute_wiener_filter(S, target_idx=local_idx)
            sp, co = apply_wiener_filter(group_data, h, local_idx, n_times)
            specific[global_idx] = sp
            coherent[global_idx] = co
            filters[pair_key][ch] = h

    return WienerResult(
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        raw=epoch,
        specific=specific,
        coherent=coherent,
        filters=filters,
        freqs=freqs,
        ch_names=ch_names,
        skipped_pairs=skipped,
    )


def decompose_subject(
    epochs: np.ndarray,      # (n_epochs, n_channels, n_times)
    ch_names: list[str],
    subject_id: str,
    cfg: dict,
) -> list[WienerResult]:
    return [
        decompose_epoch(epoch, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
        for i, epoch in enumerate(epochs)
    ]
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_wiener.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/decomposition/wiener.py tests/test_decomposition/test_wiener.py
git commit -m "feat: vector Wiener decomposition with cross-PSD estimation"
```

---

## Task 10: Wiener Scalar Mode (Ablation)

**Files:**
- Create: `eeg_bg/decomposition/wiener_scalar.py`
- Create: `tests/test_decomposition/test_wiener_scalar.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decomposition/test_wiener_scalar.py
import numpy as np
import pytest
from eeg_bg.decomposition.wiener import WienerResult, decompose_epoch as freq_decompose
from eeg_bg.decomposition.wiener_scalar import decompose_epoch as scalar_decompose


def test_scalar_returns_wiener_result(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = scalar_decompose(epoch, ch_names, cfg)
    assert isinstance(result, WienerResult)
    assert result.specific.shape == epoch.shape


def test_scalar_specific_plus_coherent_equals_raw(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = scalar_decompose(epoch, ch_names, cfg)
    np.testing.assert_allclose(
        result.specific + result.coherent, result.raw, atol=1e-8
    )


def test_scalar_residual_coherence_higher_than_freq(synthetic_epoch):
    """Frequency-dependent mode should leave lower residual coherence than scalar."""
    from scipy.signal import coherence as scipy_coherence
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]

    freq_result = freq_decompose(epoch, ch_names, cfg)
    scalar_result = scalar_decompose(epoch, ch_names, cfg)

    fp1_idx = ch_names.index("FP1")
    fp2_idx = ch_names.index("FP2")

    _, coh_freq = scipy_coherence(
        freq_result.specific[fp1_idx], freq_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    _, coh_scalar = scipy_coherence(
        scalar_result.specific[fp1_idx], scalar_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    # Scalar mode leaves equal or higher residual coherence
    assert np.mean(coh_scalar) >= np.mean(coh_freq) - 0.05
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_wiener_scalar.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/decomposition/wiener_scalar.py`**

```python
"""
Ablation: scalar Wiener mode. Replaces frequency-dependent h(f) with a
single complex scalar per reference channel (average over frequency band).
Equivalent to the 'EKG-style' fixed compensation described in the proposal.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence as scipy_coherence, welch

from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
)


def _scalar_from_filter(h: np.ndarray, freq_mask: np.ndarray) -> np.ndarray:
    """Average h(f) over the target frequency band → scalar per ref channel."""
    return h[:, freq_mask].mean(axis=1, keepdims=True)  # (n_ref, 1)


def decompose_epoch(
    epoch: np.ndarray,
    ch_names: list[str],
    cfg: dict,
    subject_id: str = "",
    epoch_idx: int = 0,
) -> WienerResult:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    coh_threshold = cfg["wiener"]["coherence_threshold"]
    bilateral_pairs = cfg["channels"]["bilateral_pairs"]
    freq_band = cfg["wiener"]["freq_band"]
    n_times = epoch.shape[1]

    specific = epoch.copy()
    coherent = np.zeros_like(epoch)
    filters: dict = {}
    skipped: list[str] = []

    freqs, _ = welch(epoch[0], fs=sfreq, nperseg=nperseg)
    freq_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])

    for pair in bilateral_pairs:
        try:
            indices = [ch_names.index(ch) for ch in pair]
        except ValueError:
            skipped.append("-".join(pair))
            continue

        group_data = epoch[indices]
        _, coh = scipy_coherence(group_data[0], group_data[1],
                                  fs=sfreq, nperseg=nperseg)
        if np.max(coh[freq_mask]) < coh_threshold:
            skipped.append("-".join(pair))
            continue

        _, S = estimate_cross_psd(group_data, sfreq, nperseg)
        pair_key = "-".join(pair)
        filters[pair_key] = {}

        for local_idx, (ch, global_idx) in enumerate(zip(pair, indices)):
            h_freq = compute_wiener_filter(S, target_idx=local_idx)
            # Collapse to scalar: average over band
            h_scalar = _scalar_from_filter(h_freq, freq_mask)

            ref_indices = [i for i in range(len(pair)) if i != local_idx]
            ref_data = group_data[ref_indices]  # (n_ref, n_times)
            coherent_signal = np.sum(h_scalar.real * ref_data, axis=0)
            specific[global_idx] = epoch[global_idx] - coherent_signal
            coherent[global_idx] = coherent_signal
            filters[pair_key][ch] = h_scalar

    return WienerResult(
        subject_id=subject_id,
        epoch_idx=epoch_idx,
        raw=epoch,
        specific=specific,
        coherent=coherent,
        filters=filters,
        freqs=freqs,
        ch_names=ch_names,
        skipped_pairs=skipped,
    )
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_wiener_scalar.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/decomposition/wiener_scalar.py tests/test_decomposition/test_wiener_scalar.py
git commit -m "feat: scalar Wiener ablation mode (fixed-gain compensation)"
```

---

## Task 11: ICA Decomposition

**Files:**
- Create: `eeg_bg/decomposition/ica.py`
- Create: `tests/test_decomposition/test_ica.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decomposition/test_ica.py
import numpy as np
import pytest
from eeg_bg.decomposition.ica import fit_ica, apply_ica


def test_fit_ica_returns_model_and_indices(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, artifact_indices = fit_ica(epochs, ch_names, cfg)
    assert artifact_indices == [] or all(
        0 <= i < cfg["ica"]["n_components"] for i in artifact_indices
    )


def test_apply_ica_preserves_shape(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, artifact_indices = fit_ica(epochs, ch_names, cfg)
    cleaned = apply_ica(epochs, ica_model, artifact_indices, ch_names, cfg)
    assert cleaned.shape == epochs.shape


def test_apply_ica_no_artifacts_unchanged(synthetic_epochs_batch):
    """With no artifact components removed, reconstruction should be near-identical."""
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, _ = fit_ica(epochs, ch_names, cfg)
    cleaned = apply_ica(epochs, ica_model, [], ch_names, cfg)
    # Reconstruction error should be small (FastICA is near-exact)
    rel_error = np.linalg.norm(cleaned - epochs) / np.linalg.norm(epochs)
    assert rel_error < 0.01
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_ica.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/decomposition/ica.py`**

```python
from __future__ import annotations

import mne
import numpy as np


def fit_ica(
    epochs: np.ndarray,    # (n_epochs, n_ch, n_times) in μV
    ch_names: list[str],
    cfg: dict,
) -> tuple[mne.preprocessing.ICA, list[int]]:
    n_epochs, n_ch, n_times = epochs.shape
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    n_components = cfg["ica"]["n_components"]
    threshold = cfg["ica"]["artifact_corr_threshold"]
    random_state = cfg["ica"].get("random_state", 42)

    # Concatenate epochs: (n_ch, n_epochs * n_times)
    data_2d = epochs.transpose(1, 0, 2).reshape(n_ch, -1) / 1e6  # μV → V

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",
                           verbose=False)
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=n_components, method="fastica",
        random_state=random_state, verbose=False
    )
    ica.fit(raw, verbose=False)

    # Artifact detection: correlate each component with frontal proxy (EOG)
    sources = ica.get_sources(raw).get_data()  # (n_components, n_samples)
    frontal = [i for i, ch in enumerate(ch_names) if ch in ("FP1", "FP2")]
    artifact_indices: list[int] = []
    if frontal:
        proxy = data_2d[frontal].mean(axis=0)
        for comp_idx in range(sources.shape[0]):
            corr = np.corrcoef(sources[comp_idx], proxy)[0, 1]
            if abs(corr) > threshold:
                artifact_indices.append(comp_idx)

    return ica, artifact_indices


def apply_ica(
    epochs: np.ndarray,                # (n_epochs, n_ch, n_times) in μV
    ica: mne.preprocessing.ICA,
    artifact_indices: list[int],
    ch_names: list[str],
    cfg: dict,
) -> np.ndarray:
    n_epochs, n_ch, n_times = epochs.shape
    sfreq = float(cfg["preprocessing"]["target_sfreq"])

    data_2d = epochs.transpose(1, 0, 2).reshape(n_ch, -1) / 1e6  # μV → V
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg",
                           verbose=False)
    raw = mne.io.RawArray(data_2d, info, verbose=False)

    ica_copy = ica.copy()
    ica_copy.exclude = artifact_indices
    raw_clean = ica_copy.apply(raw.copy(), verbose=False)

    clean_2d = raw_clean.get_data() * 1e6  # V → μV
    return clean_2d.reshape(n_ch, n_epochs, n_times).transpose(1, 0, 2)
```

- [ ] **Step 4: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_decomposition/test_ica.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/decomposition/ica.py tests/test_decomposition/test_ica.py
git commit -m "feat: ICA decomposition using MNE with frontal artifact detection"
```

---

## Task 12: Verification — V1, V2, V3

**Files:**
- Create: `eeg_bg/verification/coherence.py`
- Create: `eeg_bg/verification/transitivity.py`
- Create: `tests/test_verification/test_coherence.py`
- Create: `tests/test_verification/test_transitivity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verification/test_coherence.py
import numpy as np
import pytest
import pandas as pd
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.verification.coherence import compute_pairwise_coherence, run_v1


def test_compute_pairwise_coherence_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    coh = compute_pairwise_coherence(epoch, sfreq, nperseg,
                                      tuple(cfg["wiener"]["freq_band"]))
    assert coh.shape == (len(ch_names), len(ch_names))
    assert np.all(coh >= 0) and np.all(coh <= 1)
    np.testing.assert_allclose(coh, coh.T, atol=1e-10)  # symmetric


def test_run_v1_output_schema(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v1([result], cfg)
    assert isinstance(df, pd.DataFrame)
    required_cols = {"subject_id", "epoch_idx", "ch_i", "ch_j",
                     "coh_pre", "coh_post", "reduction"}
    assert required_cols.issubset(df.columns)
    assert (df["coh_pre"] >= 0).all() and (df["coh_pre"] <= 1).all()


def test_run_v1_reduction_positive_for_bilateral_pairs(synthetic_epoch):
    """High-SNR data: decomposition must reduce coherence for bilateral pairs."""
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v1([result], cfg)
    bilateral_flat = [ch for pair in cfg["channels"]["bilateral_pairs"] for ch in pair]
    bilateral_df = df[df["ch_i"].isin(bilateral_flat) & df["ch_j"].isin(bilateral_flat)]
    if len(bilateral_df):
        assert bilateral_df["reduction"].mean() > 0
```

```python
# tests/test_verification/test_transitivity.py
import numpy as np
import pytest
import pandas as pd
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.verification.transitivity import run_v2, run_v3


def test_run_v2_error_nonnegative(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v2([result], cfg)
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert (df["eps_amp"] >= 0).all()
        assert (df["eps_phase"] >= 0).all()


def test_run_v3_variation_nonnegative(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v3([result], cfg)
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert (df["freq_variation"] >= 0).all()
```

- [ ] **Step 2: Run to verify failure**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_verification/ -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement `eeg_bg/verification/coherence.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import coherence as scipy_coherence


def compute_pairwise_coherence(
    data: np.ndarray,          # (n_ch, n_times)
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
) -> np.ndarray:
    n_ch = data.shape[0]
    freqs, _ = scipy_coherence(data[0], data[0], fs=sfreq, nperseg=nperseg)
    mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    coh_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        coh_matrix[i, i] = 1.0
        for j in range(i + 1, n_ch):
            _, coh = scipy_coherence(data[i], data[j], fs=sfreq, nperseg=nperseg)
            val = float(np.mean(coh[mask]))
            coh_matrix[i, j] = val
            coh_matrix[j, i] = val
    return coh_matrix


def run_v1(results: list, cfg: dict) -> pd.DataFrame:
    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    nperseg = cfg["wiener"]["nperseg"]
    freq_band = tuple(cfg["wiener"]["freq_band"])

    rows = []
    for result in results:
        ch_names = result.ch_names
        coh_pre = compute_pairwise_coherence(result.raw, sfreq, nperseg, freq_band)
        coh_post = compute_pairwise_coherence(result.specific, sfreq, nperseg, freq_band)
        n_ch = len(ch_names)
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                rows.append({
                    "subject_id": result.subject_id,
                    "epoch_idx": result.epoch_idx,
                    "ch_i": ch_names[i],
                    "ch_j": ch_names[j],
                    "coh_pre": coh_pre[i, j],
                    "coh_post": coh_post[i, j],
                    "reduction": coh_pre[i, j] - coh_post[i, j],
                })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Implement `eeg_bg/verification/transitivity.py`**

```python
"""
V2: Transitivity constraint — for a single point source, the filter chain
    h_ij * h_jk should equal h_ik (in amplitude) and sum in phase.
V3: Frequency variation of |h_ij(f)| across the target band.
    Large variation supports the frequency-dependent model over scalar.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd


def _get_filter_amplitude_phase(
    result, pair_key: str, ch_name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if pair_key not in result.filters:
        return None
    if ch_name not in result.filters[pair_key]:
        return None
    h = result.filters[pair_key][ch_name]   # (n_ref, n_freqs)
    amp = np.abs(h[0])
    phase = np.angle(h[0])
    return amp, phase


def run_v2(results: list, cfg: dict) -> pd.DataFrame:
    freq_band = cfg["wiener"]["freq_band"]
    rows = []
    for result in results:
        freqs = result.freqs
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        pairs = list(result.filters.keys())
        # Find triplets that share channels
        channels_per_pair = {k: list(result.filters[k].keys()) for k in pairs}
        all_channels = list({ch for chs in channels_per_pair.values() for ch in chs})
        for ch_i, ch_j, ch_k in combinations(all_channels, 3):
            # Find pairs connecting each channel combination
            pair_ij = f"{ch_i}-{ch_j}"
            pair_jk = f"{ch_j}-{ch_k}"
            pair_ik = f"{ch_i}-{ch_k}"
            try:
                amp_ij, phase_ij = _get_filter_amplitude_phase(result, pair_ij, ch_i) or (None, None)
                amp_jk, phase_jk = _get_filter_amplitude_phase(result, pair_jk, ch_j) or (None, None)
                amp_ik, phase_ik = _get_filter_amplitude_phase(result, pair_ik, ch_i) or (None, None)
            except TypeError:
                continue
            if amp_ij is None or amp_jk is None or amp_ik is None:
                continue
            eps_amp = float(np.mean(np.abs(amp_ij[mask] * amp_jk[mask] - amp_ik[mask])))
            phase_diff = np.angle(np.exp(1j * (phase_ij[mask] + phase_jk[mask] - phase_ik[mask])))
            eps_phase = float(np.mean(np.abs(phase_diff)))
            rows.append({
                "subject_id": result.subject_id,
                "epoch_idx": result.epoch_idx,
                "triplet": f"{ch_i}-{ch_j}-{ch_k}",
                "eps_amp": eps_amp,
                "eps_phase": eps_phase,
            })
    return pd.DataFrame(rows)


def run_v3(results: list, cfg: dict) -> pd.DataFrame:
    freq_band = cfg["wiener"]["freq_band"]
    rows = []
    for result in results:
        freqs = result.freqs
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        for pair_key, ch_dict in result.filters.items():
            for ch_name, h in ch_dict.items():
                amp = np.abs(h[0, mask])
                if amp.size == 0:
                    continue
                variation = float((amp.max() - amp.min()) / (amp.mean() + 1e-12))
                rows.append({
                    "subject_id": result.subject_id,
                    "epoch_idx": result.epoch_idx,
                    "pair": pair_key,
                    "channel": ch_name,
                    "freq_variation": variation,
                    "amp_mean": float(amp.mean()),
                    "amp_std": float(amp.std()),
                })
    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run to verify pass**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/test_verification/ -v
```

Expected: `5 passed`

- [ ] **Step 6: Commit**

```powershell
git add eeg_bg/verification/coherence.py eeg_bg/verification/transitivity.py \
        tests/test_verification/test_coherence.py tests/test_verification/test_transitivity.py
git commit -m "feat: V1/V2/V3 physical verification experiments"
```

---

## Task 13: Visualization Modules

**Files:**
- Create: `eeg_bg/visualization/filter_plots.py`
- Create: `eeg_bg/visualization/coherence_plots.py`
- Create: `eeg_bg/visualization/verification_plots.py`

No unit tests (matplotlib output); verified visually in notebooks.

- [ ] **Step 1: Implement `eeg_bg/visualization/filter_plots.py`**

```python
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from eeg_bg.decomposition.wiener import WienerResult


def plot_wiener_filter_response(
    result: WienerResult,
    pair_key: str,
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> plt.Figure:
    if pair_key not in result.filters:
        raise KeyError(f"Pair '{pair_key}' not in result.filters. "
                       f"Available: {list(result.filters.keys())}")
    ch_name = list(result.filters[pair_key].keys())[0]
    h = result.filters[pair_key][ch_name][0]  # (n_freqs,)
    freqs = result.freqs

    if ax is None:
        fig, (ax_amp, ax_phase) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    else:
        fig = ax[0].figure
        ax_amp, ax_phase = ax

    ax_amp.plot(freqs, np.abs(h))
    ax_amp.set_ylabel("|h(f)|")
    ax_amp.set_title(f"Wiener filter response: {pair_key}")
    ax_amp.grid(True, alpha=0.3)

    ax_phase.plot(freqs, np.angle(h))
    ax_phase.set_ylabel("∠h(f) [rad]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_xlim(0, 45)
    ax_phase.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_pairs_response(
    result: WienerResult,
    save_path=None,
) -> plt.Figure:
    pairs = list(result.filters.keys())
    n = len(pairs)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No filter results", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 5), sharex=True)
    for col, pair_key in enumerate(pairs):
        plot_wiener_filter_response(
            result, pair_key,
            ax=(axes[0, col], axes[1, col]) if n > 1 else (axes[0], axes[1])
        )
        axes[0, col].set_title(pair_key, fontsize=9)

    fig.suptitle("Wiener filter responses — all bilateral pairs", fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
```

- [ ] **Step 2: Implement `eeg_bg/visualization/coherence_plots.py`**

```python
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eeg_bg.decomposition.wiener import WienerResult


def plot_coherence_matrix(
    coh_pre: np.ndarray,
    coh_post: np.ndarray,
    ch_names: list[str],
    title: str = "",
) -> plt.Figure:
    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, label in [(ax_pre, coh_pre, "Before decomposition"),
                            (ax_post, coh_post, "After decomposition")]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="hot_r", aspect="auto")
        ax.set_xticks(range(len(ch_names)))
        ax.set_yticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
        ax.set_yticklabels(ch_names, fontsize=8)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_coherence_reduction(
    v1_df: pd.DataFrame,
    group_by: str = "pair",
) -> plt.Figure:
    if group_by == "pair":
        v1_df = v1_df.copy()
        v1_df["pair"] = v1_df["ch_i"] + "-" + v1_df["ch_j"]
        x_col = "pair"
    else:
        x_col = "subject_id"

    groups = sorted(v1_df[x_col].unique())
    data = [v1_df[v1_df[x_col] == g]["reduction"].values for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.6), 5))
    ax.boxplot(data, labels=groups, patch_artist=True)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Coherence reduction (pre − post)")
    ax.set_title("V1: Coherence reduction by " + x_col)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_signal_decomposition(
    result: WienerResult,
    ch_name: str,
    sfreq: float = 125.0,
    epoch_idx: int = 0,
    time_window: tuple[float, float] = (0.0, 4.0),
) -> plt.Figure:
    ch_idx = result.ch_names.index(ch_name)
    t = np.arange(result.raw.shape[1]) / sfreq
    mask = (t >= time_window[0]) & (t <= time_window[1])

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, sig, label in [
        (axes[0], result.raw[ch_idx], "Raw x(t)"),
        (axes[1], result.coherent[ch_idx], "Coherent component"),
        (axes[2], result.specific[ch_idx], "Specific component"),
    ]:
        ax.plot(t[mask], sig[mask])
        ax.set_ylabel(label + " [μV]")
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel("Time [s]")
    axes[0].set_title(f"Signal decomposition — {ch_name}, epoch {epoch_idx}")
    fig.tight_layout()
    return fig
```

- [ ] **Step 3: Implement `eeg_bg/visualization/verification_plots.py`**

```python
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_v2_transitivity(v2_df: pd.DataFrame) -> plt.Figure:
    fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(10, 4))
    ax_amp.hist(v2_df["eps_amp"], bins=30, edgecolor="k", alpha=0.7)
    ax_amp.axvline(0.1, color="r", linestyle="--", label="threshold 0.1")
    ax_amp.set_xlabel("ε_amplitude")
    ax_amp.set_title("V2: Transitivity amplitude error")
    ax_amp.legend()

    ax_phase.hist(v2_df["eps_phase"], bins=30, edgecolor="k", alpha=0.7)
    ax_phase.axvline(0.392, color="r", linestyle="--", label="threshold π/8")
    ax_phase.set_xlabel("ε_phase [rad]")
    ax_phase.set_title("V2: Transitivity phase error")
    ax_phase.legend()

    fig.tight_layout()
    return fig


def plot_v3_frequency_variation(v3_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair, grp in v3_df.groupby("pair"):
        ax.bar([pair], [grp["freq_variation"].mean()],
               yerr=[grp["freq_variation"].std()], capsize=4, alpha=0.7)
    ax.axhline(0.20, color="r", linestyle="--", label="threshold 20%")
    ax.set_ylabel("Relative frequency variation of |h(f)|")
    ax.set_title("V3: Frequency-dependence of Wiener filter")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_ica_vs_wiener_coherence(
    raw_pre: np.ndarray,
    ica_post: np.ndarray,
    wiener_post: np.ndarray,
    ch_names: list[str],
) -> plt.Figure:
    from eeg_bg.verification.coherence import compute_pairwise_coherence
    sfreq, nperseg, freq_band = 125.0, 1000, (0.5, 40.0)

    coh_raw = compute_pairwise_coherence(raw_pre, sfreq, nperseg, freq_band)
    coh_ica = compute_pairwise_coherence(ica_post, sfreq, nperseg, freq_band)
    coh_wiener = compute_pairwise_coherence(wiener_post, sfreq, nperseg, freq_band)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mat, title in [
        (axes[0], coh_raw, "Raw"),
        (axes[1], coh_ica, "ICA"),
        (axes[2], coh_wiener, "Wiener"),
    ]:
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="hot_r", aspect="auto")
        ax.set_xticks(range(len(ch_names)))
        ax.set_yticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
        ax.set_yticklabels(ch_names, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Pairwise coherence: Raw vs ICA vs Wiener", fontsize=12)
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Verify imports without error**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline python -c "
from eeg_bg.visualization.filter_plots import plot_wiener_filter_response, plot_all_pairs_response
from eeg_bg.visualization.coherence_plots import plot_coherence_matrix, plot_coherence_reduction, plot_signal_decomposition
from eeg_bg.visualization.verification_plots import plot_v2_transitivity, plot_v3_frequency_variation, plot_ica_vs_wiener_coherence
print('All visualization imports OK')
"
```

Expected: `All visualization imports OK`

- [ ] **Step 5: Commit**

```powershell
git add eeg_bg/visualization/
git commit -m "feat: visualization modules for filter response, coherence, and verification"
```

---

## Task 14: CLI Scripts

**Files:**
- Create: `scripts/01_extract_epochs.py`
- Create: `scripts/02_run_wiener.py`
- Create: `scripts/03_run_ica.py`
- Create: `scripts/04_run_verification.py`

- [ ] **Step 1: Implement `scripts/01_extract_epochs.py`**

```python
"""Extract background epochs from all EDF files and cache to disk."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.io.dataset import build_subject_index, assign_splits
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.io.annotation import extract_bckg_intervals
from eeg_bg.io.cache import make_cache_key
from eeg_bg.preprocessing.epoch import bandpass_filter, slice_epochs
from eeg_bg.preprocessing.reference import filter_by_reference


def main(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    cache_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    cache_root.mkdir(parents=True, exist_ok=True)

    index = build_subject_index(cfg)
    index = filter_by_reference(index, cfg["dataset"]["reference_scheme"])
    index = assign_splits(index, cfg)
    index.to_csv(cache_root / "index.csv", index=False)
    print(f"Found {len(index)} EDF files across {index['subject_id'].nunique()} subjects.")

    for _, row in tqdm(index.iterrows(), total=len(index), desc="Extracting epochs"):
        edf_path = Path(row["edf_path"])
        csv_bi_path = edf_path.with_suffix(".csv_bi")

        cache_key = make_cache_key(edf_path, 0.0, cfg)
        cache_path = cache_root / row["subject_id"] / f"{cache_key}.npz"

        if cache_path.exists() and not force:
            continue

        try:
            data, ch_names, sfreq = load_edf(edf_path, cfg)
        except Exception as e:
            print(f"  SKIP {edf_path.name}: {e}")
            continue

        if csv_bi_path.exists():
            intervals = extract_bckg_intervals(csv_bi_path, cfg)
        else:
            duration = data.shape[1] / sfreq
            intervals = [(0.0, duration)]

        epochs = slice_epochs(
            data, sfreq, intervals,
            cfg["preprocessing"]["epoch_length_sec"],
            cfg["preprocessing"]["artifact_threshold_uv"],
        )

        if epochs.shape[0] == 0:
            print(f"  SKIP {edf_path.name}: no valid epochs")
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            epochs=epochs,
            ch_names=np.array(ch_names),
            label=np.array(row["label"]),
            subject_id=np.array(row["subject_id"]),
            split=np.array(row["split"]),
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.force)
```

- [ ] **Step 2: Implement `scripts/02_run_wiener.py`**

```python
"""Run Wiener decomposition on cached epochs."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition import wiener as wiener_freq
from eeg_bg.decomposition import wiener_scalar


def main(config_path: str, mode: str = "frequency", force: bool = False) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / f"wiener_{mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    decompose = wiener_freq.decompose_epoch if mode == "frequency" else wiener_scalar.decompose_epoch

    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="Wiener"):
        out_path = out_root / npz_path.relative_to(epoch_root)
        if out_path.exists() and not force:
            continue

        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]             # (n_epochs, n_ch, n_times)
        ch_names = list(data["ch_names"])
        subject_id = str(data["subject_id"])

        results = []
        for i, epoch in enumerate(epochs):
            r = decompose(epoch, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
            results.append({
                "specific": r.specific,
                "coherent": r.coherent,
                "skipped_pairs": np.array(r.skipped_pairs),
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            specific=np.stack([r["specific"] for r in results]),
            coherent=np.stack([r["coherent"] for r in results]),
            label=data["label"],
            subject_id=data["subject_id"],
            split=data["split"],
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["frequency", "scalar"], default="frequency")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.mode, args.force)
```

- [ ] **Step 3: Implement `scripts/03_run_ica.py`**

```python
"""Run ICA decomposition on cached epochs."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.ica import fit_ica, apply_ica


def main(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / "ica"
    out_root.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="ICA"):
        out_path = out_root / npz_path.relative_to(epoch_root)
        if out_path.exists() and not force:
            continue

        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
        ch_names = list(data["ch_names"])

        try:
            ica_model, artifact_idx = fit_ica(epochs, ch_names, cfg)
            cleaned = apply_ica(epochs, ica_model, artifact_idx, ch_names, cfg)
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            specific=cleaned,
            n_artifacts_removed=np.array(len(artifact_idx)),
            label=data["label"],
            subject_id=data["subject_id"],
            split=data["split"],
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.force)
```

- [ ] **Step 4: Implement `scripts/04_run_verification.py`**

```python
"""Run physical verification experiments V1/V2/V3 and save CSV reports."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch, WienerResult
from eeg_bg.verification.coherence import run_v1
from eeg_bg.verification.transitivity import run_v2, run_v3


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[WienerResult] = []
    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="V1/V2/V3"):
        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
        ch_names = list(data["ch_names"])
        subject_id = str(data["subject_id"])
        # Re-run Wiener to get filter coefficients (needed for V2/V3)
        for i, epoch in enumerate(epochs):
            r = decompose_epoch(epoch, ch_names, cfg,
                                subject_id=subject_id, epoch_idx=i)
            all_results.append(r)

    print(f"Running V1 on {len(all_results)} epochs...")
    v1_df = run_v1(all_results, cfg)
    v1_df.to_csv(results_dir / "v1_coherence.csv", index=False)

    print("Running V2...")
    v2_df = run_v2(all_results, cfg)
    v2_df.to_csv(results_dir / "v2_transitivity.csv", index=False)

    print("Running V3...")
    v3_df = run_v3(all_results, cfg)
    v3_df.to_csv(results_dir / "v3_frequency_variation.csv", index=False)

    print(f"V1: {len(v1_df)} rows | V2: {len(v2_df)} rows | V3: {len(v3_df)} rows")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
```

- [ ] **Step 5: Verify scripts are importable**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline python -c "
import ast, pathlib
for f in pathlib.Path('scripts').glob('*.py'):
    ast.parse(f.read_text())
    print(f'  OK: {f.name}')
"
```

Expected: `OK: 01_extract_epochs.py` ... `OK: 04_run_verification.py`

- [ ] **Step 6: Commit**

```powershell
git add scripts/
git commit -m "feat: CLI pipeline scripts 01-04 for epoch extraction through verification"
```

---

## Task 15: Full Test Suite + Final Check

- [ ] **Step 1: Run the full unit test suite**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline pytest tests/ -m "not integration" -v
```

Expected: All tests pass. Target: ≥ 25 tests collected, 0 failures.

- [ ] **Step 2: Check import graph (no cycles)**

```powershell
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n eeg_pipeline python -c "
from eeg_bg.config.settings import load_config
from eeg_bg.io.cache import load_or_compute, make_cache_key
from eeg_bg.io.annotation import extract_bckg_intervals
from eeg_bg.io.edf_reader import load_edf
from eeg_bg.io.dataset import build_subject_index, assign_splits
from eeg_bg.preprocessing.epoch import bandpass_filter, slice_epochs
from eeg_bg.preprocessing.reference import detect_reference, filter_by_reference
from eeg_bg.decomposition.wiener import decompose_epoch, decompose_subject
from eeg_bg.decomposition.wiener_scalar import decompose_epoch as scalar_decompose
from eeg_bg.decomposition.ica import fit_ica, apply_ica
from eeg_bg.verification.coherence import compute_pairwise_coherence, run_v1
from eeg_bg.verification.transitivity import run_v2, run_v3
from eeg_bg.visualization.filter_plots import plot_wiener_filter_response
from eeg_bg.visualization.coherence_plots import plot_coherence_matrix
from eeg_bg.visualization.verification_plots import plot_v2_transitivity
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Final commit**

```powershell
git add -A
git commit -m "chore: final integration check — all imports and unit tests pass"
```

---

## What's Not In This Plan (Next Iteration)

- Jupyter notebook cell content (exploratory, written interactively)
- SVM / XGBoost hand-crafted feature extraction (band power, Hjorth, spectral entropy)
- Model training, cross-validation, AUROC evaluation
- Grad-CAM / SHAP interpretability
- CNN (requires PyTorch installation)
