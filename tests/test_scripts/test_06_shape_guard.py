"""Test that script 06 shape-guard raises on stale feature cache."""
import argparse

import numpy as np
import pytest
from pathlib import Path


def test_shape_guard_raises_on_wrong_dim(tmp_path, monkeypatch):
    """_load_or_extract_features raises ValueError when cached dims mismatch."""
    from eeg_bg.features.extraction import FEATURE_NAMES

    # Write a stale cache with wrong dims (use a clearly wrong size)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    wrong_dim = len(FEATURE_NAMES) + 1
    stale_path = feat_dir / "raw_train.npz"
    np.savez(stale_path,
             X=np.zeros((3, wrong_dim), dtype=np.float64),
             y=np.array([0, 0, 1], dtype=np.int64),
             subject_ids=np.array(["s1", "s1", "s2"], dtype=object))

    # Import the guard function from the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "script06", Path("scripts/06_train_xgboost.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with pytest.raises(ValueError, match="--force"):
        mod._load_or_extract_features(
            cache_root=tmp_path,
            feature_cache_dir=feat_dir,
            condition="raw",
            split="train",
            sfreq=125.0,
            nperseg=250,
            freq_band=(0.5, 40.0),
            force=False,
        )


def test_workers_must_be_positive():
    """Script 06 rejects invalid process counts before extraction."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "script06_workers", Path("scripts/06_train_xgboost.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._positive_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        mod._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        mod._positive_int("-1")
