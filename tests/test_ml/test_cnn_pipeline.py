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


def test_train_cnn_creates_output_files(tiny_cache, tmp_path):
    """train_cnn runs for a few epochs and writes all expected output files."""
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
