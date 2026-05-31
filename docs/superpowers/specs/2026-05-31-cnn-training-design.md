# CNN Training Design — EEG Background Seizure Detection

**Date:** 2026-05-31  
**Status:** Approved

---

## Context

The existing pipeline classifies EEG background recordings (epilepsy vs. control) using hand-crafted 211-dimensional features fed into XGBoost (`scripts/06_train_xgboost.py`, `eeg_bg/ml/xgb_pipeline.py`). This design adds a comparable CNN training pipeline (`scripts/08_train_cnn.py`) that operates directly on raw `(19, 1000)` epoch time-series — no manual feature engineering — enabling end-to-end learned representations to be compared against the XGBoost baseline under identical conditions, splits, and evaluation metrics.

---

## Architecture — EEGNet

EEGNet (Lawhern et al., 2018) is a purpose-built, parameter-efficient architecture for EEG classification. It explicitly separates temporal filter learning from spatial (cross-channel) mixing, which maps naturally onto the `(channels, time)` epoch format.

**Input:** `(batch, 1, 19, 1000)` — epochs unsqueezed to a single-channel 2D "image".

```
Block 1 — Temporal conv
  Conv2d(1,    F1,    kernel=(1, 64), padding=(0, 32), bias=False)
  BatchNorm2d(F1)

Block 2 — Depthwise spatial conv  (mixes channels per filter)
  Conv2d(F1, F1*D, kernel=(19, 1), groups=F1, bias=False)
  BatchNorm2d(F1*D)
  ELU → AvgPool2d(1, 4) → Dropout(p)

Block 3 — Separable temporal conv
  Conv2d(F1*D, F1*D, kernel=(1, 16), padding=(0, 8), groups=F1*D, bias=False)
  Conv2d(F1*D, F2,   kernel=(1, 1),  bias=False)
  BatchNorm2d(F2)
  ELU → AvgPool2d(1, 8) → Dropout(p)

Flatten → Linear(F2 * T_out, 1) → Sigmoid
```

**Default hyperparameters** (stored in `configs/default.yaml` under `cnn:`):

| Param | Default | Meaning |
|-------|---------|---------|
| `F1` | 8 | Temporal filter count |
| `D` | 2 | Depth multiplier → F2 = F1*D = 16 |
| `dropout` | 0.25 | Dropout probability |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-4 | Adam L2 regularisation |
| `batch_size` | 64 | Training batch size |

The flatten dimension after the two pooling layers depends on exact padding arithmetic; the implementation computes it dynamically at `__init__` time by passing a dummy `(1, 1, 19, 1000)` tensor through the convolutional blocks. Total trainable parameters are approximately **5 000–6 000**.

---

## Data Pipeline

A new `EEGEpochDataset(torch.utils.data.Dataset)` class in `eeg_bg/ml/cnn_dataset.py`:

- **Constructor signature** mirrors `build_dataset()`: accepts `cache_root`, `condition` (`"raw"` / `"wiener"` / `"ica"`), `split` (`"train"` / `"val"` / `"test"`).
- Walks the same cache directories as the XGBoost pipeline (`cache/epochs/`, `cache/wiener_frequency/`, `cache/ica/`) — no new cache tier required.
- `__getitem__` returns `(epoch_tensor, label, subject_id)`:
  - `epoch_tensor`: shape `(1, 19, 1000)`, float32, **z-scored per channel** (mean 0, std 1 across 1000 time samples) inline — replaces the separate `StandardScaler` fit used in XGBoost.
  - `label`: int scalar (0 = epilepsy, 1 = control).
  - `subject_id`: string, for subject-level aggregation downstream.

**DataLoader settings:** `batch_size=64`, `shuffle=True` (train) / `False` (val/test), `num_workers=0` (Windows `spawn`-safe default; configurable).

---

## Training Loop

Implemented in `eeg_bg/ml/cnn_pipeline.py`, function `train_cnn(condition, cfg, out_dir, force)`.

| Setting | Value |
|---------|-------|
| Loss | `BCELoss` (binary, sigmoid output) |
| Class imbalance | `scale_pos_weight` from training label counts; applied as manual per-sample weight |
| Optimizer | Adam (`lr`, `weight_decay` from config) |
| LR scheduler | `ReduceLROnPlateau(factor=0.5, patience=10)` on val AUROC |
| Early stopping | Patience 20 epochs on subject-level val AUROC; saves `best_model.pt` |
| Max epochs | 200 |

Subject-level val AUROC is computed after each epoch by running inference on the full val set and calling the existing `subject_level_predict` + `evaluate_subject_level` functions from `eeg_bg/ml/xgb_pipeline.py`.

---

## Evaluation & Outputs

Reuses `subject_level_predict` and `evaluate_subject_level` from `eeg_bg/ml/xgb_pipeline.py` verbatim. Threshold optimisation (search [0.05, 0.95] for max macro-F1 on val set) is also reused.

Output directory: `results/cnn/{condition}/`

```
results/cnn/
├── {raw,ica,wiener}/
│   ├── best_model.pt              — PyTorch state dict (best val AUROC epoch)
│   ├── best_params.json           — F1, D, dropout, lr, batch_size, stopped_epoch
│   ├── val_metrics.json           — {auroc, f1, accuracy}
│   ├── test_metrics.json          — {auroc, f1, accuracy}
│   ├── val_predictions.csv        — subject_id, pred_proba, true_label
│   └── test_predictions.csv       — subject_id, pred_proba, true_label
└── comparison_summary.csv         — 3 conditions × {val_auroc, test_auroc, f1, acc}
```

No cross-model (XGBoost vs CNN) summary file is produced; the two pipelines remain fully independent.

---

## Script: `scripts/08_train_cnn.py`

Mirrors `scripts/06_train_xgboost.py` in structure and CLI interface:

```
--condition  raw | ica | wiener | all   (default: all)
--config     path to YAML              (default: configs/default.yaml)
--force      re-run even if outputs exist
--workers    DataLoader num_workers    (default: 0)
```

Runs conditions sequentially (raw → ica → wiener when `--condition all`). Writes `results/cnn/comparison_summary.csv` after all conditions complete.

---

## New Files

| File | Purpose |
|------|---------|
| `eeg_bg/ml/cnn_dataset.py` | `EEGEpochDataset` — PyTorch Dataset over epoch cache |
| `eeg_bg/ml/cnn_model.py` | `EEGNet` — PyTorch Module |
| `eeg_bg/ml/cnn_pipeline.py` | `train_cnn()` — training loop, early stopping, output writing |
| `scripts/08_train_cnn.py` | CLI entry point |

## Modified Files

| File | Change |
|------|--------|
| `configs/default.yaml` | Add `cnn:` section with default hyperparameters |
| `environment.yaml` | Add `pytorch` (CPU or CUDA build as appropriate) |

---

## Verification

```bash
# Install pytorch into the env first, then:
conda run -n eeg_pipeline python scripts/08_train_cnn.py --condition raw --workers 0

# Expected: val_metrics.json and test_metrics.json appear under results/cnn/raw/
# Check subject-level AUROC is a reasonable number (compare to results/xgboost/raw/val_metrics.json)

# Run all conditions
conda run -n eeg_pipeline python scripts/08_train_cnn.py --condition all

# Verify comparison_summary.csv has 3 rows
conda run -n eeg_pipeline python -c "import pandas as pd; print(pd.read_csv('results/cnn/comparison_summary.csv'))"
```
