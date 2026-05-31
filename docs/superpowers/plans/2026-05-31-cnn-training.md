# CNN Training (EEGNet) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an EEGNet-based CNN training pipeline (`scripts/08_train_cnn.py`) that classifies EEG epilepsy vs. control directly from raw `(19, 1000)` epoch time-series, with subject-level evaluation metrics comparable to the existing XGBoost pipeline.

**Architecture:** EEGNet (temporal conv → depthwise spatial conv → separable conv → sigmoid) operates on `(batch, 1, 19, 1000)` tensors loaded from the existing epoch caches. Training uses BCE loss with class-weight balancing, Adam + ReduceLROnPlateau, and early stopping on subject-level val AUROC. Evaluation reuses `find_optimal_threshold` and `evaluate_subject_level` from `eeg_bg/ml/xgb_pipeline.py`.

**Tech Stack:** PyTorch (CPU or CUDA), existing conda env `eeg_pipeline`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `eeg_bg/ml/cnn_model.py` | `EEGNet` PyTorch module |
| Create | `eeg_bg/ml/cnn_dataset.py` | `EEGEpochDataset` — loads epoch caches into a PyTorch Dataset |
| Create | `eeg_bg/ml/cnn_pipeline.py` | `cnn_predict_epochs`, `train_cnn` — training loop + output writing |
| Create | `scripts/08_train_cnn.py` | CLI entry point (mirrors `06_train_xgboost.py`) |
| Create | `tests/test_ml/test_cnn_model.py` | Unit tests for EEGNet |
| Create | `tests/test_ml/test_cnn_dataset.py` | Unit tests for EEGEpochDataset |
| Create | `tests/test_ml/test_cnn_pipeline.py` | Smoke test for `cnn_predict_epochs` and `train_cnn` |
| Modify | `configs/default.yaml` | Add `cnn:` section under `ml:` |

---

## Task 1: Install PyTorch and add `cnn:` config section

**Files:**
- Modify: `configs/default.yaml`

- [ ] **Step 1: Install PyTorch into the eeg_pipeline conda env**

```bash
conda run -n eeg_pipeline pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you have a CUDA GPU and want GPU training, use instead:
```bash
conda run -n eeg_pipeline pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Expected: `Successfully installed torch-...`

- [ ] **Step 2: Verify PyTorch is importable**

```bash
conda run -n eeg_pipeline python -c "import torch; print(torch.__version__)"
```

Expected: prints a version string such as `2.x.x+cpu`.

- [ ] **Step 3: Add `cnn:` section to `configs/default.yaml`**

Append the following block at the end of the `ml:` section in `configs/default.yaml` (after the existing `shap:` block):

```yaml
  cnn:
    # EEGNet architecture hyperparameters
    F1: 8                 # number of temporal filters in Block 1
    D: 2                  # depth multiplier; F2 = F1 * D = 16
    dropout: 0.25         # dropout probability applied after Blocks 2 and 3
    # Training hyperparameters
    lr: 0.001             # Adam learning rate
    weight_decay: 0.0001  # Adam L2 regularisation
    batch_size: 64        # DataLoader batch size
    max_epochs: 200       # hard ceiling on training epochs
    patience: 20          # early-stopping patience (val AUROC epochs)
    lr_patience: 10       # ReduceLROnPlateau patience
    lr_factor: 0.5        # ReduceLROnPlateau reduction factor
    device: "cpu"         # "cpu" or "cuda"
    num_workers: 0        # DataLoader num_workers (0 = main process, Windows-safe)
```

The full `ml:` section tail (for reference, showing where to place it — after `shap:` ending with `dpi: 150`):

```yaml
  shap:
    max_display: 20
    dpi: 150
  cnn:
    F1: 8
    D: 2
    dropout: 0.25
    lr: 0.001
    weight_decay: 0.0001
    batch_size: 64
    max_epochs: 200
    patience: 20
    lr_patience: 10
    lr_factor: 0.5
    device: "cpu"
    num_workers: 0
```

- [ ] **Step 4: Verify config loads cleanly**

```bash
conda run -n eeg_pipeline python -c "
from eeg_bg.config.settings import load_config
cfg = load_config('configs/default.yaml')
print(cfg['ml']['cnn'])
"
```

Expected: prints the cnn dict with keys F1, D, dropout, lr, etc.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml
git commit -m "feat: add cnn: config section and install pytorch"
```

---

## Task 2: EEGNet model (`cnn_model.py`)

**Files:**
- Create: `eeg_bg/ml/cnn_model.py`
- Create: `tests/test_ml/test_cnn_model.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml/test_cnn_model.py`:

```python
"""Unit tests for EEGNet."""
import pytest
import torch

from eeg_bg.ml.cnn_model import EEGNet


def test_eegnet_output_shape():
    """Forward pass produces (batch, 1) output."""
    model = EEGNet(n_channels=19, n_times=1000, F1=8, D=2, dropout=0.0)
    x = torch.randn(4, 1, 19, 1000)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_eegnet_output_is_probability():
    """Sigmoid output must be in [0, 1]."""
    model = EEGNet(n_channels=19, n_times=1000)
    model.eval()
    x = torch.randn(4, 1, 19, 1000)
    with torch.no_grad():
        out = model(x)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_eegnet_batch_size_one():
    """Model handles batch size of 1 without BatchNorm errors."""
    model = EEGNet(n_channels=19, n_times=1000)
    model.eval()
    x = torch.randn(1, 1, 19, 1000)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1)


def test_eegnet_custom_hyperparams():
    """Model instantiates and runs with non-default F1/D values."""
    model = EEGNet(n_channels=19, n_times=1000, F1=4, D=4, dropout=0.5)
    x = torch.randn(2, 1, 19, 1000)
    out = model(x)
    assert out.shape == (2, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_model.py -v
```

Expected: `ImportError` — `eeg_bg.ml.cnn_model` does not exist yet.

- [ ] **Step 3: Implement `eeg_bg/ml/cnn_model.py`**

```python
"""EEGNet — compact EEG classification CNN.

Reference: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.

Architecture
------------
Block 1: Temporal conv  — learns frequency-band-like filters along time.
Block 2: Depthwise spatial conv — combines channels within each filter.
Block 3: Separable temporal conv — refines temporal representation compactly.
Output:  Sigmoid scalar — probability of class 1 (control).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet for binary EEG classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (height of the input "image").  Default 19.
    n_times : int
        Number of time samples per epoch.  Default 1000.
    F1 : int
        Number of temporal filters in Block 1.  Default 8.
    D : int
        Depth multiplier; F2 = F1 * D filters after Block 2.  Default 2.
    dropout : float
        Dropout probability applied after Blocks 2 and 3.  Default 0.25.
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 1000,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        F2 = F1 * D

        # Block 1 — Temporal conv: learns spectral-band-like filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2 — Depthwise spatial conv: mixes channels per filter
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 3 — Separable temporal conv: compact temporal refinement
        self.block3 = nn.Sequential(
            # Depthwise part
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False),
            # Pointwise part
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flatten size by passing a dummy tensor through the conv blocks.
        # This avoids hardcoding the arithmetic and adapts to any n_times value.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            flatten_size = dummy.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, 1, n_channels, n_times)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)``, values in ``[0, 1]``.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_model.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/ml/cnn_model.py tests/test_ml/test_cnn_model.py
git commit -m "feat: add EEGNet model (cnn_model.py)"
```

---

## Task 3: EEGEpochDataset (`cnn_dataset.py`)

**Files:**
- Create: `eeg_bg/ml/cnn_dataset.py`
- Create: `tests/test_ml/test_cnn_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml/test_cnn_dataset.py`:

```python
"""Unit tests for EEGEpochDataset."""
import numpy as np
import pytest
import torch

from eeg_bg.ml.cnn_dataset import EEGEpochDataset


@pytest.fixture()
def fake_cache(tmp_path):
    """Create a minimal epoch cache with 2 subjects × 3 epochs."""
    rng = np.random.default_rng(0)

    # epilepsy subject in train split
    subdir0 = tmp_path / "epochs" / "00_subj0"
    subdir0.mkdir(parents=True)
    np.savez(
        subdir0 / "abc123.npz",
        epochs=rng.standard_normal((3, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj0",
        split="train",
    )

    # control subject in train split
    subdir1 = tmp_path / "epochs" / "01_subj1"
    subdir1.mkdir(parents=True)
    np.savez(
        subdir1 / "def456.npz",
        epochs=rng.standard_normal((3, 19, 1000)).astype(np.float32),
        label=np.int64(1),
        subject_id="01_subj1",
        split="train",
    )

    # subject in val split — must be excluded from train dataset
    subdir2 = tmp_path / "epochs" / "00_subj2"
    subdir2.mkdir(parents=True)
    np.savez(
        subdir2 / "ghi789.npz",
        epochs=rng.standard_normal((2, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj2",
        split="val",
    )

    return tmp_path


def test_dataset_length(fake_cache):
    """Dataset length equals total epochs across matching subjects."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    assert len(ds) == 6  # 3 + 3 epochs from two train subjects


def test_dataset_excludes_other_splits(fake_cache):
    """Val-split subject is not included in the train dataset."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    subject_ids = [ds[i][2] for i in range(len(ds))]
    assert "00_subj2" not in subject_ids


def test_epoch_tensor_shape(fake_cache):
    """Each item returns an epoch tensor of shape (1, 19, 1000)."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    epoch_tensor, label, subject_id = ds[0]
    assert epoch_tensor.shape == (1, 19, 1000)
    assert epoch_tensor.dtype == torch.float32


def test_epoch_is_z_scored(fake_cache):
    """Each channel of the returned tensor is approximately zero-mean, unit-variance."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    epoch_tensor, _, _ = ds[0]  # shape (1, 19, 1000)
    channel_means = epoch_tensor[0].mean(dim=-1)   # (19,)
    channel_stds  = epoch_tensor[0].std(dim=-1)    # (19,)
    assert torch.all(channel_means.abs() < 1e-5), "Channels not zero-mean after z-score"
    assert torch.all((channel_stds - 1.0).abs() < 0.01), "Channels not unit-var after z-score"


def test_label_values(fake_cache):
    """Labels are 0 or 1 integers."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_subject_id_is_string(fake_cache):
    """subject_id field is a string."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    _, _, subject_id = ds[0]
    assert isinstance(subject_id, str)


def test_wiener_condition_uses_specific_key(tmp_path):
    """Wiener condition reads 'specific' array key from wiener_frequency subdir."""
    rng = np.random.default_rng(1)
    subdir = tmp_path / "wiener_frequency" / "00_subj0"
    subdir.mkdir(parents=True)
    np.savez(
        subdir / "key.npz",
        specific=rng.standard_normal((2, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj0",
        split="train",
    )
    ds = EEGEpochDataset(cache_root=tmp_path, condition="wiener", split="train")
    assert len(ds) == 2
    epoch_tensor, label, sid = ds[0]
    assert epoch_tensor.shape == (1, 19, 1000)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_dataset.py -v
```

Expected: `ImportError` — `eeg_bg.ml.cnn_dataset` does not exist yet.

- [ ] **Step 3: Implement `eeg_bg/ml/cnn_dataset.py`**

```python
"""PyTorch Dataset that reads epoch caches for CNN training.

Mirrors the interface of ``build_dataset()`` in ``eeg_bg/features/extraction.py``
but returns raw epoch tensors instead of hand-crafted feature vectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Cache subdirectory names keyed by condition label — same mapping as
# eeg_bg/features/extraction.py::_CONDITION_TO_SUBDIR
_CONDITION_TO_SUBDIR: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "wiener_frequency",
    "ica":    "ica",
}

# NPZ array key holding signal data per condition
_CONDITION_TO_KEY: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "specific",
    "ica":    "specific",
}


class EEGEpochDataset(Dataset):
    """Dataset of EEG epochs loaded from the project's NPZ cache.

    Each item is a tuple ``(epoch_tensor, label, subject_id)`` where
    ``epoch_tensor`` has shape ``(1, 19, 1000)`` and is z-scored per channel.

    Parameters
    ----------
    cache_root : Path
        Project-level cache directory (resolved absolute path from ``load_config``).
    condition : str
        One of ``"raw"``, ``"wiener"``, ``"ica"``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.

    Raises
    ------
    ValueError
        If *condition* is not recognised.
    FileNotFoundError
        If the condition subdirectory does not exist.
    """

    def __init__(
        self,
        cache_root: Path,
        condition: str,
        split: str,
    ) -> None:
        if condition not in _CONDITION_TO_SUBDIR:
            raise ValueError(
                f"Unknown condition {condition!r}. "
                f"Expected one of {list(_CONDITION_TO_SUBDIR)}."
            )

        subdir = Path(cache_root) / _CONDITION_TO_SUBDIR[condition]
        if not subdir.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {subdir}. "
                f"Run the corresponding preprocessing script first."
            )

        array_key = _CONDITION_TO_KEY[condition]

        # Load all matching epochs eagerly into memory.
        # Epoch arrays are small (19 × 1000 × 4 bytes ≈ 76 KB each) and the
        # full dataset typically fits comfortably in RAM.
        self._epochs: list[np.ndarray] = []      # each (19, 1000) float32
        self._labels: list[int] = []
        self._subject_ids: list[str] = []

        for npz_path in sorted(subdir.rglob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            if str(data["split"]) != split:
                continue

            epochs_arr = data[array_key].astype(np.float32)  # (n_ep, 19, 1000)
            label      = int(data["label"])
            subject_id = str(data["subject_id"])

            for ep in epochs_arr:
                self._epochs.append(ep)
                self._labels.append(label)
                self._subject_ids.append(subject_id)

    def __len__(self) -> int:
        return len(self._epochs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        epoch = self._epochs[idx].copy()  # (19, 1000)

        # Z-score each channel independently across the 1000 time samples.
        mean = epoch.mean(axis=-1, keepdims=True)          # (19, 1)
        std  = epoch.std(axis=-1, keepdims=True) + 1e-8   # (19, 1), avoid /0
        epoch = (epoch - mean) / std

        # Unsqueeze to (1, 19, 1000) — single "colour channel" for Conv2d
        tensor = torch.from_numpy(epoch).unsqueeze(0)  # (1, 19, 1000)
        return tensor, self._labels[idx], self._subject_ids[idx]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_dataset.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add eeg_bg/ml/cnn_dataset.py tests/test_ml/test_cnn_dataset.py
git commit -m "feat: add EEGEpochDataset (cnn_dataset.py)"
```

---

## Task 4: CNN inference helper and training loop (`cnn_pipeline.py`)

**Files:**
- Create: `eeg_bg/ml/cnn_pipeline.py`
- Create: `tests/test_ml/test_cnn_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ml/test_cnn_pipeline.py`:

```python
"""Tests for cnn_predict_epochs and train_cnn."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from eeg_bg.ml.cnn_dataset import EEGEpochDataset
from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.cnn_pipeline import cnn_predict_epochs, train_cnn


# ---------------------------------------------------------------------------
# Shared fixture: minimal epoch cache with train + val subjects
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_cache(tmp_path):
    """Cache with 4 train subjects (2 epilepsy, 2 control) and 2 val subjects."""
    rng = np.random.default_rng(99)

    def _write(subdir, label_int, subject_id, split, n_epochs=3):
        subdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            subdir / "data.npz",
            epochs=rng.standard_normal((n_epochs, 19, 1000)).astype(np.float32),
            label=np.int64(label_int),
            subject_id=subject_id,
            split=split,
        )

    for i in range(2):
        _write(tmp_path / "epochs" / f"00_ep{i}", 0, f"00_ep{i}", "train")
        _write(tmp_path / "epochs" / f"01_ct{i}", 1, f"01_ct{i}", "train")

    _write(tmp_path / "epochs" / "00_epval", 0, "00_epval", "val", n_epochs=2)
    _write(tmp_path / "epochs" / "01_ctval", 1, "01_ctval", "val", n_epochs=2)

    return tmp_path


# ---------------------------------------------------------------------------
# cnn_predict_epochs
# ---------------------------------------------------------------------------

def test_cnn_predict_epochs_returns_dataframe(tiny_cache):
    """cnn_predict_epochs returns a DataFrame with correct columns."""
    model = EEGNet(n_channels=19, n_times=1000, F1=4, D=2, dropout=0.0)
    ds = EEGEpochDataset(cache_root=tiny_cache, condition="raw", split="train")
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    result = cnn_predict_epochs(model, loader, device="cpu")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"subject_id", "pred_proba", "true_label"}


def test_cnn_predict_epochs_subject_level(tiny_cache):
    """One row per subject in the returned DataFrame."""
    model = EEGNet(n_channels=19, n_times=1000, F1=4, D=2, dropout=0.0)
    ds = EEGEpochDataset(cache_root=tiny_cache, condition="raw", split="train")
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    result = cnn_predict_epochs(model, loader, device="cpu")

    # 4 train subjects
    assert len(result) == 4
    assert result["pred_proba"].between(0, 1).all()


# ---------------------------------------------------------------------------
# train_cnn (smoke test)
# ---------------------------------------------------------------------------

def test_train_cnn_creates_output_files(tiny_cache, tmp_path):
    """train_cnn runs for a few epochs and writes all expected output files."""
    out_dir = tmp_path / "results" / "cnn" / "raw"

    # Minimal config that overrides training length for speed
    cfg = {
        "paths": {"cache_dir": str(tiny_cache), "results_dir": str(tmp_path / "results")},
        "ml": {
            "cnn": {
                "F1": 4, "D": 2, "dropout": 0.0,
                "lr": 0.01, "weight_decay": 0.0,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 2,
                "lr_patience": 1,
                "lr_factor": 0.5,
                "device": "cpu",
                "num_workers": 0,
            }
        },
    }

    train_cnn(condition="raw", cfg=cfg, out_dir=out_dir)

    assert (out_dir / "best_model.pt").exists()
    assert (out_dir / "best_params.json").exists()
    assert (out_dir / "val_metrics.json").exists()
    assert (out_dir / "test_metrics.json").exists()
    assert (out_dir / "val_predictions.csv").exists()
    assert (out_dir / "test_predictions.csv").exists()


def test_train_cnn_metrics_have_expected_keys(tiny_cache, tmp_path):
    """val_metrics.json and test_metrics.json contain auroc, f1, accuracy, threshold."""
    out_dir = tmp_path / "results" / "cnn" / "raw"
    cfg = {
        "paths": {"cache_dir": str(tiny_cache), "results_dir": str(tmp_path / "results")},
        "ml": {
            "cnn": {
                "F1": 4, "D": 2, "dropout": 0.0,
                "lr": 0.01, "weight_decay": 0.0,
                "batch_size": 4,
                "max_epochs": 2,
                "patience": 2,
                "lr_patience": 1,
                "lr_factor": 0.5,
                "device": "cpu",
                "num_workers": 0,
            }
        },
    }

    train_cnn(condition="raw", cfg=cfg, out_dir=out_dir)

    with open(out_dir / "val_metrics.json") as f:
        val_m = json.load(f)
    with open(out_dir / "test_metrics.json") as f:
        test_m = json.load(f)

    for key in ("auroc", "f1", "accuracy", "threshold"):
        assert key in val_m, f"Missing key {key!r} in val_metrics.json"
        assert key in test_m, f"Missing key {key!r} in test_metrics.json"
```

Note: The `tiny_cache` fixture only has `train` and `val` splits. `train_cnn` will need a `test` split too. Add test subjects to the fixture:

Update `test_cnn_pipeline.py` fixture so `tiny_cache` also writes test subjects:

```python
@pytest.fixture()
def tiny_cache(tmp_path):
    """Cache with train/val/test subjects for raw condition."""
    rng = np.random.default_rng(99)

    def _write(subdir, label_int, subject_id, split, n_epochs=3):
        subdir.mkdir(parents=True, exist_ok=True)
        np.savez(
            subdir / "data.npz",
            epochs=rng.standard_normal((n_epochs, 19, 1000)).astype(np.float32),
            label=np.int64(label_int),
            subject_id=subject_id,
            split=split,
        )

    # train: 2 epilepsy + 2 control
    for i in range(2):
        _write(tmp_path / "epochs" / f"00_ep{i}", 0, f"00_ep{i}", "train")
        _write(tmp_path / "epochs" / f"01_ct{i}", 1, f"01_ct{i}", "train")
    # val: 1 epilepsy + 1 control
    _write(tmp_path / "epochs" / "00_epval", 0, "00_epval", "val", n_epochs=2)
    _write(tmp_path / "epochs" / "01_ctval", 1, "01_ctval", "val", n_epochs=2)
    # test: 1 epilepsy + 1 control
    _write(tmp_path / "epochs" / "00_eptest", 0, "00_eptest", "test", n_epochs=2)
    _write(tmp_path / "epochs" / "01_cttest", 1, "01_cttest", "test", n_epochs=2)

    return tmp_path
```

Use this final version of the fixture (with test subjects) as the single `tiny_cache` fixture in the file — replace the earlier draft above.

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_pipeline.py -v
```

Expected: `ImportError` — `eeg_bg.ml.cnn_pipeline` does not exist yet.

- [ ] **Step 3: Implement `eeg_bg/ml/cnn_pipeline.py`**

```python
"""CNN training pipeline — training loop, inference, and output writing.

Mirrors the structure of ``eeg_bg/ml/xgb_pipeline.py``:
  - ``cnn_predict_epochs``: run a fitted model on a DataLoader and return
    subject-level predictions as a DataFrame.
  - ``train_cnn``: full training loop with early stopping; writes output files.

Reuses ``find_optimal_threshold`` and ``evaluate_subject_level`` from
``xgb_pipeline.py`` for metrics — keeping evaluation logic in one place.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from eeg_bg.ml.cnn_dataset import EEGEpochDataset
from eeg_bg.ml.cnn_model import EEGNet
from eeg_bg.ml.xgb_pipeline import evaluate_subject_level, find_optimal_threshold


def cnn_predict_epochs(
    model: EEGNet,
    dataloader: DataLoader,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run *model* over *dataloader* and return subject-level predictions.

    Epoch-level probabilities are averaged per subject, mirroring
    ``subject_level_predict`` in ``xgb_pipeline.py``.

    Parameters
    ----------
    model : EEGNet
        A fitted (or partially fitted) EEGNet model.
    dataloader : DataLoader
        Yields ``(epoch_tensor, label, subject_id)`` batches.
    device : str
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id``, ``pred_proba``, ``true_label``.
        One row per unique subject.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_probas: list[float] = []
    all_labels: list[int]   = []
    all_sids:   list[str]   = []

    with torch.no_grad():
        for epoch_tensors, labels, subject_ids in dataloader:
            epoch_tensors = epoch_tensors.to(dev)
            proba = model(epoch_tensors).squeeze(1).cpu().numpy()  # (batch,)
            all_probas.extend(proba.tolist())
            all_labels.extend(
                labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
            )
            all_sids.extend(list(subject_ids))

    df = pd.DataFrame({
        "subject_id":  all_sids,
        "epoch_proba": all_probas,
        "true_label":  all_labels,
    })

    subject_df = (
        df.groupby("subject_id")
          .agg(pred_proba=("epoch_proba", "mean"),
               true_label=("true_label",  "first"))
          .reset_index()
    )
    return subject_df


def train_cnn(
    condition: str,
    cfg: dict,
    out_dir: Path,
    force: bool = False,
) -> dict:
    """Train EEGNet for *condition* and write results to *out_dir*.

    Parameters
    ----------
    condition : str
        One of ``"raw"``, ``"wiener"``, ``"ica"``.
    cfg : dict
        Loaded ``default.yaml`` (from ``load_config``).  Reads
        ``cfg["paths"]["cache_dir"]`` and ``cfg["ml"]["cnn"]``.
    out_dir : Path
        Directory where output files are written.  Created if absent.
    force : bool
        If ``False`` and ``out_dir/val_metrics.json`` already exists,
        skip training and return the cached metrics.

    Returns
    -------
    dict
        Val and test metrics for the trained model.
    """
    out_dir = Path(out_dir)

    if not force and (out_dir / "val_metrics.json").exists():
        with open(out_dir / "val_metrics.json") as f:
            val_metrics = json.load(f)
        with open(out_dir / "test_metrics.json") as f:
            test_metrics = json.load(f)
        return {"val": val_metrics, "test": test_metrics}

    out_dir.mkdir(parents=True, exist_ok=True)

    cnn_cfg    = cfg["ml"]["cnn"]
    cache_root = Path(cfg["paths"]["cache_dir"])
    device     = cnn_cfg.get("device", "cpu")
    dev        = torch.device(device)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    train_ds = EEGEpochDataset(cache_root, condition, "train")
    val_ds   = EEGEpochDataset(cache_root, condition, "val")
    test_ds  = EEGEpochDataset(cache_root, condition, "test")

    batch_size   = int(cnn_cfg.get("batch_size", 64))
    num_workers  = int(cnn_cfg.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    # ── Model ───────────────────────────────────────────────────────────────
    model = EEGNet(
        n_channels=19,
        n_times=1000,
        F1=int(cnn_cfg.get("F1", 8)),
        D=int(cnn_cfg.get("D", 2)),
        dropout=float(cnn_cfg.get("dropout", 0.25)),
    ).to(dev)

    # ── Class balancing ─────────────────────────────────────────────────────
    train_labels = np.array(train_ds._labels)
    counts = np.bincount(train_labels.astype(int))
    pos_weight = (
        torch.tensor([counts[0] / counts[1]], dtype=torch.float32).to(dev)
        if len(counts) >= 2 and counts[1] > 0
        else torch.tensor([1.0]).to(dev)
    )

    criterion = nn.BCELoss(reduction="none")

    optimizer = Adam(
        model.parameters(),
        lr=float(cnn_cfg.get("lr", 1e-3)),
        weight_decay=float(cnn_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cnn_cfg.get("lr_factor", 0.5)),
        patience=int(cnn_cfg.get("lr_patience", 10)),
    )

    max_epochs = int(cnn_cfg.get("max_epochs", 200))
    patience   = int(cnn_cfg.get("patience", 20))

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_auroc  = -1.0
    epochs_no_improve = 0
    best_state_dict   = None
    best_epoch        = 0

    for epoch in range(max_epochs):
        model.train()
        for batch_tensors, batch_labels, _ in train_loader:
            batch_tensors = batch_tensors.to(dev)
            batch_labels  = batch_labels.float().to(dev)

            preds = model(batch_tensors).squeeze(1)  # (batch,)

            # Weighted BCE: positive class (control, label=1) gets pos_weight
            sample_weights = torch.where(
                batch_labels == 1,
                pos_weight.expand_as(batch_labels),
                torch.ones_like(batch_labels),
            )
            loss = (criterion(preds, batch_labels) * sample_weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_df     = cnn_predict_epochs(model, val_loader, device=device)
        val_auroc  = evaluate_subject_level(val_df)["auroc"]

        scheduler.step(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc    = val_auroc
            best_state_dict   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch        = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # ── Load best weights ────────────────────────────────────────────────────
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), out_dir / "best_model.pt")

    # ── Val evaluation (optimise threshold) ──────────────────────────────────
    val_df     = cnn_predict_epochs(model, val_loader, device=device)
    threshold  = find_optimal_threshold(val_df)
    val_metrics = evaluate_subject_level(val_df, threshold=threshold)

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_df      = cnn_predict_epochs(model, test_loader, device=device)
    test_metrics = evaluate_subject_level(test_df, threshold=threshold)

    # ── Write outputs ─────────────────────────────────────────────────────────
    best_params = {
        "F1":           int(cnn_cfg.get("F1", 8)),
        "D":            int(cnn_cfg.get("D", 2)),
        "dropout":      float(cnn_cfg.get("dropout", 0.25)),
        "lr":           float(cnn_cfg.get("lr", 1e-3)),
        "batch_size":   int(cnn_cfg.get("batch_size", 64)),
        "stopped_epoch": best_epoch,
    }

    with open(out_dir / "best_params.json",  "w") as f:
        json.dump(best_params, f, indent=2)
    with open(out_dir / "val_metrics.json",  "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    val_df[["subject_id", "pred_proba", "true_label"]].to_csv(
        out_dir / "val_predictions.csv", index=False
    )
    test_df[["subject_id", "pred_proba", "true_label"]].to_csv(
        out_dir / "test_predictions.csv", index=False
    )

    return {"val": val_metrics, "test": test_metrics}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n eeg_pipeline python -m pytest tests/test_ml/test_cnn_pipeline.py -v
```

Expected: all 4 tests PASS (may take ~30 seconds for the smoke tests).

- [ ] **Step 5: Run the full test suite to verify no regressions**

```bash
conda run -n eeg_pipeline python -m pytest tests/ -m "not integration" -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add eeg_bg/ml/cnn_pipeline.py tests/test_ml/test_cnn_pipeline.py
git commit -m "feat: add CNN training pipeline (cnn_pipeline.py)"
```

---

## Task 5: CLI script `scripts/08_train_cnn.py`

**Files:**
- Create: `scripts/08_train_cnn.py`

- [ ] **Step 1: Implement the script**

```python
"""Script 08 — Train EEGNet CNN on raw / wiener / ICA epoch caches.

Usage
-----
conda run -n eeg_pipeline python scripts/08_train_cnn.py [OPTIONS]

Options
-------
--condition  raw | ica | wiener | all   Which condition(s) to train (default: all)
--config     PATH                       Config YAML path (default: configs/default.yaml)
--force                                 Re-train even if output already exists
--workers    N                          DataLoader num_workers override (default: from config)

Output
------
results/cnn/{condition}/
    best_model.pt, best_params.json,
    val_metrics.json, test_metrics.json,
    val_predictions.csv, test_predictions.csv
results/cnn/comparison_summary.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import

import pandas as pd

# Resolve project root so the script works regardless of cwd
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from eeg_bg.config.settings import load_config
from eeg_bg.ml.cnn_pipeline import train_cnn


_CONDITIONS = ["raw", "ica", "wiener"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEGNet CNN on EEG epoch caches."
    )
    parser.add_argument(
        "--condition",
        default="all",
        choices=_CONDITIONS + ["all"],
        help="Which condition to train (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output files already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="DataLoader num_workers override (default: value from config)",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    cfg    = load_config(args.config)

    # Allow --workers CLI flag to override the config value
    if args.workers is not None:
        cfg["ml"]["cnn"]["num_workers"] = args.workers

    results_dir = Path(cfg["paths"]["results_dir"])
    conditions  = _CONDITIONS if args.condition == "all" else [args.condition]

    all_metrics: list[dict] = []

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"  Training CNN — condition: {condition}")
        print(f"{'='*60}")

        out_dir = results_dir / "cnn" / condition
        metrics = train_cnn(
            condition=condition,
            cfg=cfg,
            out_dir=out_dir,
            force=args.force,
        )

        print(f"  val  AUROC={metrics['val']['auroc']:.4f}  "
              f"F1={metrics['val']['f1']:.4f}  "
              f"Acc={metrics['val']['accuracy']:.4f}")
        print(f"  test AUROC={metrics['test']['auroc']:.4f}  "
              f"F1={metrics['test']['f1']:.4f}  "
              f"Acc={metrics['test']['accuracy']:.4f}")

        all_metrics.append({
            "condition":    condition,
            "val_auroc":    metrics["val"]["auroc"],
            "val_f1":       metrics["val"]["f1"],
            "val_accuracy": metrics["val"]["accuracy"],
            "test_auroc":   metrics["test"]["auroc"],
            "test_f1":      metrics["test"]["f1"],
            "test_accuracy": metrics["test"]["accuracy"],
        })

    # Write comparison summary only when all three conditions were run
    if len(all_metrics) == 3:
        summary_path = results_dir / "cnn" / "comparison_summary.csv"
        pd.DataFrame(all_metrics).to_csv(summary_path, index=False)
        print(f"\nComparison summary written to {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses correctly**

```bash
conda run -n eeg_pipeline python scripts/08_train_cnn.py --help
```

Expected: prints usage with `--condition`, `--config`, `--force`, `--workers` options.

- [ ] **Step 3: Run a smoke test on one condition (requires real cached epochs)**

If you have the epoch cache populated (script 01 has been run), test with:

```bash
conda run -n eeg_pipeline python scripts/08_train_cnn.py --condition raw --workers 0
```

Expected:
- Prints training progress per epoch
- `results/cnn/raw/val_metrics.json` is written
- `results/cnn/raw/test_metrics.json` is written

Verify:
```bash
conda run -n eeg_pipeline python -c "
import json
with open('results/cnn/raw/val_metrics.json') as f:
    print(json.load(f))
"
```

Expected: dict with `auroc`, `f1`, `accuracy`, `threshold` keys.

- [ ] **Step 4: Commit**

```bash
git add scripts/08_train_cnn.py
git commit -m "feat: add scripts/08_train_cnn.py CLI entry point"
```

---

## Verification Checklist

- [ ] All unit tests pass: `conda run -n eeg_pipeline python -m pytest tests/ -m "not integration" -v`
- [ ] `results/cnn/raw/val_metrics.json` contains `auroc`, `f1`, `accuracy`, `threshold`
- [ ] `results/cnn/raw/best_model.pt` exists and loads cleanly:
  ```bash
  conda run -n eeg_pipeline python -c "
  import torch; from eeg_bg.ml.cnn_model import EEGNet
  m = EEGNet(); m.load_state_dict(torch.load('results/cnn/raw/best_model.pt', map_location='cpu')); print('OK')
  "
  ```
- [ ] Running `--condition all` produces `results/cnn/comparison_summary.csv` with 3 rows
