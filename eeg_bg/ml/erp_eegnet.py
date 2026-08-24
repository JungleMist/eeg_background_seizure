"""EEGNet training utilities for ERP trial-level experiments.

This module contains the reusable EEGNet method used by the ERP-CORE
component experiments.  The input dataset only needs ``matrix(condition,
normalize=True)``, ``y`` and ``subject_ids`` attributes, so experiment-specific
cache and decomposition code stays outside the machine-learning package.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from eeg_bg.ml.cnn_model import EEGNet


class TrialSequenceDataset(Dataset):
    """Torch dataset for ``(trials, channels, times)`` ERP sequences."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray
    ) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected (trials, channels, times), got {X.shape}")
        if len(X) != len(y) or len(X) != len(subject_ids):
            raise ValueError("X, y, and subject_ids must have equal lengths")
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.subject_ids = np.asarray(subject_ids).astype(str)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index].unsqueeze(0), self.y[index], self.subject_ids[index]


def make_model(n_channels: int, n_times: int, model_cfg: dict[str, Any]) -> EEGNet:
    """Construct an EEGNet from an experiment model configuration."""
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        F1=int(model_cfg.get("F1", 8)),
        D=int(model_cfg.get("D", 2)),
        dropout=float(model_cfg.get("dropout", 0.25)),
    )


def predict_trials(
    model: EEGNet, loader: DataLoader, device: torch.device
) -> pd.DataFrame:
    """Return one probability row per ERP trial."""
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for tensors, labels, subject_ids in loader:
            probabilities = model(tensors.to(device)).squeeze(1).cpu().numpy()
            for subject_id, label, probability in zip(
                subject_ids, labels.numpy(), probabilities
            ):
                rows.append(
                    {
                        "subject_id": str(subject_id),
                        "true_label": int(label),
                        "pred_proba": float(probability),
                    }
                )
    return pd.DataFrame(rows, columns=["subject_id", "true_label", "pred_proba"])


def select_balanced_accuracy_threshold(
    y: np.ndarray, probabilities: np.ndarray
) -> float:
    """Select a deterministic threshold using validation balanced accuracy."""
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [
        (
            balanced_accuracy_score(y, probabilities >= threshold),
            abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored, key=lambda item: (item[0], -item[1], -item[2]))[2]


def classification_metrics(
    y: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute trial-level binary metrics at a frozen decision threshold."""
    y = np.asarray(y, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    true_negative, false_positive, _, _ = matrix.ravel()
    denominator = true_negative + false_positive
    specificity = true_negative / denominator if denominator else float("nan")
    return {
        "auroc": float(roc_auc_score(y, probabilities))
        if len(np.unique(y)) == 2
        else float("nan"),
        "auprc": float(average_precision_score(y, probabilities))
        if len(y)
        else float("nan"),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "accuracy": float(accuracy_score(y, predictions)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
    }


def _save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def train_condition(
    condition: str,
    dataset: Any,
    partitions: dict[str, list[str]],
    model_cfg: dict[str, Any],
    out_dir: Path,
    random_state: int,
) -> dict[str, Any]:
    """Train and evaluate EEGNet for one ERP condition.

    ``dataset`` is intentionally duck-typed to keep ERP cache/decomposition
    classes out of this reusable package module.
    """
    indices = {
        split: np.flatnonzero(
            np.isin(dataset.subject_ids, np.asarray(subjects, dtype=str))
        )
        for split, subjects in partitions.items()
    }
    X = dataset.matrix(condition, normalize=True)
    n_channels, n_times = X.shape[1:]
    train_index, validation_index, test_index = (
        indices["train"], indices["validation"], indices["test"]
    )
    if len(np.unique(dataset.y[train_index])) < 2:
        raise ValueError(f"Condition {condition} training split must contain both ERN classes")
    if len(np.unique(dataset.y[validation_index])) < 2:
        raise ValueError(f"Condition {condition} validation split must contain both ERN classes")

    batch_size = int(model_cfg.get("batch_size", 64))
    num_workers = int(model_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        TrialSequenceDataset(X[train_index], dataset.y[train_index], dataset.subject_ids[train_index]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(random_state),
    )
    validation_loader = DataLoader(
        TrialSequenceDataset(X[validation_index], dataset.y[validation_index], dataset.subject_ids[validation_index]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TrialSequenceDataset(X[test_index], dataset.y[test_index], dataset.subject_ids[test_index]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device_name = str(model_cfg.get("device", "cpu"))
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("EEGNet device 'cuda' requested but CUDA is unavailable")
    if device_name.startswith("mps"):
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise RuntimeError("EEGNet device 'mps' requested but MPS is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    model = make_model(n_channels, n_times, model_cfg).to(device)

    counts = np.bincount(dataset.y[train_index].astype(int), minlength=2)
    pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)
    criterion = nn.BCELoss(reduction="none")
    optimizer = Adam(
        model.parameters(),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(model_cfg.get("lr_factor", 0.5)),
        patience=int(model_cfg.get("lr_patience", 10)),
    )

    max_epochs = int(model_cfg.get("max_epochs", 200))
    patience = int(model_cfg.get("patience", 20))
    best_score = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for tensors, labels, _ in train_loader:
            labels = labels.float().to(device)
            predictions = model(tensors.to(device)).squeeze(1)
            weights = torch.where(
                labels == 1, pos_weight.expand_as(labels), torch.ones_like(labels)
            )
            loss = (criterion(predictions, labels) * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_predictions = predict_trials(model, validation_loader, device)
        validation_score = float(
            average_precision_score(
                validation_predictions["true_label"],
                validation_predictions["pred_proba"],
            )
        )
        scheduler.step(validation_score)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else float("nan"),
                "validation_auprc": validation_score,
            }
        )
        if np.isfinite(validation_score) and validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = len(history)
        best_score = float(history[-1]["validation_auprc"]) if history else float("nan")
    model.load_state_dict(best_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    validation_predictions = predict_trials(model, validation_loader, device)
    threshold = select_balanced_accuracy_threshold(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(),
    )
    test_predictions = predict_trials(model, test_loader, device)
    validation_metrics = classification_metrics(
        validation_predictions["true_label"].to_numpy(),
        validation_predictions["pred_proba"].to_numpy(),
        threshold,
    )
    test_metrics = classification_metrics(
        test_predictions["true_label"].to_numpy(),
        test_predictions["pred_proba"].to_numpy(),
        threshold,
    )
    for frame, split in (
        (validation_predictions, "validation"),
        (test_predictions, "test"),
    ):
        frame["condition"] = condition
        frame["split"] = split
        frame["predicted_label"] = (frame["pred_proba"] >= threshold).astype(np.int8)
    validation_predictions.to_csv(out_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    _save_json(validation_metrics, out_dir / "val_metrics.json")
    _save_json(test_metrics, out_dir / "test_metrics.json")
    _save_json(
        {
            "condition": condition,
            "n_channels": n_channels,
            "n_times": n_times,
            "F1": int(model_cfg.get("F1", 8)),
            "D": int(model_cfg.get("D", 2)),
            "dropout": float(model_cfg.get("dropout", 0.25)),
            "lr": float(model_cfg.get("lr", 1e-3)),
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "best_validation_auprc": best_score,
            "threshold_selection": "validation_balanced_accuracy",
        },
        out_dir / "best_params.json",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    return {
        "condition": condition,
        "n_channels": n_channels,
        "n_times": n_times,
        "best_epoch": best_epoch,
        "best_validation_auprc": best_score,
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
    }
