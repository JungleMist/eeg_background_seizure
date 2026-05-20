import pandas as pd
import pytest
from pathlib import Path
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
    """Create fake TUEP directory structure with 4 subjects (2 per class)."""
    for class_dir in ["00_epilepsy", "01_no_epilepsy"]:
        for i in range(2):
            subj = f"subj_{class_dir[:2]}_{i:02d}"
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


def test_build_subject_index_label_encoding(tmp_path):
    _make_fake_tree(tmp_path)
    cfg = {**MOCK_CFG, "paths": {"data_root": str(tmp_path)}}
    index = build_subject_index(cfg)
    # epilepsy class → label 0, no_epilepsy → label 1
    epilepsy_rows = index[index["edf_path"].str.contains("00_epilepsy")]
    control_rows = index[index["edf_path"].str.contains("01_no_epilepsy")]
    assert (epilepsy_rows["label"] == 0).all()
    assert (control_rows["label"] == 1).all()


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
