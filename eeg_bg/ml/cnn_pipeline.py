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
    # Derive n_times from the actual data shape so the model adapts to any
    # epoch_length_sec value without requiring hardcoded assumptions.
    # train_ds loads eagerly so [0] is already in memory.
    n_times = train_ds[0][0].shape[-1]  # epoch_tensor: (1, n_channels, n_times)
    model = EEGNet(
        n_channels=19,
        n_times=n_times,
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
    best_val_auroc    = -1.0
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
        val_df    = cnn_predict_epochs(model, val_loader, device=device)
        val_auroc = evaluate_subject_level(val_df)["auroc"]

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
        "F1":            int(cnn_cfg.get("F1", 8)),
        "D":             int(cnn_cfg.get("D", 2)),
        "dropout":       float(cnn_cfg.get("dropout", 0.25)),
        "lr":            float(cnn_cfg.get("lr", 1e-3)),
        "batch_size":    int(cnn_cfg.get("batch_size", 64)),
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
