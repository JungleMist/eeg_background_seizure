"""Tests for the reusable ERP EEGNet training module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eeg_bg.ml.erp_eegnet import (
    TrialSequenceDataset,
    make_model,
    train_condition,
)


class _Dataset:
    def __init__(self) -> None:
        rng = np.random.default_rng(12)
        self.y = np.asarray([0, 0, 1, 1] * 2, dtype=np.int8)
        self.subject_ids = np.asarray([f"sub-{index:03d}" for index in range(len(self.y))])
        self._X = rng.standard_normal((len(self.y), 30, 128)).astype(np.float32)

    def matrix(self, condition: str, normalize: bool = True) -> np.ndarray:
        assert condition == "raw"
        return self._X


def test_trial_sequence_dataset_adds_eegnet_input_axis() -> None:
    dataset = TrialSequenceDataset(
        np.zeros((2, 30, 128), dtype=np.float32),
        np.asarray([0, 1]),
        np.asarray(["sub-001", "sub-002"]),
    )

    tensor, label, subject_id = dataset[0]

    assert tensor.shape == (1, 30, 128)
    assert int(label) == 0
    assert subject_id == "sub-001"


def test_make_model_supports_erp_channel_count() -> None:
    model = make_model(60, 128, {"F1": 2, "D": 1, "dropout": 0.0})

    with torch.no_grad():
        output = model(torch.randn(2, 1, 60, 128))

    assert output.shape == (2, 1)


def test_train_condition_is_package_level_api(tmp_path: Path) -> None:
    summary = train_condition(
        "raw",
        _Dataset(),
        {
            "train": ["sub-000", "sub-001", "sub-002", "sub-003"],
            "validation": ["sub-004", "sub-006"],
            "test": ["sub-005", "sub-007"],
        },
        {
            "F1": 2,
            "D": 1,
            "dropout": 0.0,
            "batch_size": 2,
            "max_epochs": 1,
            "patience": 1,
            "device": "cpu",
            "num_workers": 0,
        },
        tmp_path / "erp-eegnet",
        random_state=3,
    )

    assert summary["n_channels"] == 30
    assert (tmp_path / "erp-eegnet" / "best_model.pt").exists()
    assert (tmp_path / "erp-eegnet" / "val_metrics.json").exists()
