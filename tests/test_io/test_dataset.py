import pandas as pd
import pytest
from pathlib import Path
from eeg_bg.io.dataset import (
    assign_dataset_splits,
    assign_splits,
    build_recording_index,
    build_subject_index,
    get_recording_intervals,
)

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


def _make_tuab_tree(tmp_path, n_train_patients=20):
    """Create train/eval TUAB paths, including mixed-label train patients."""
    for patient_idx in range(n_train_patients):
        patient = f"train{patient_idx:03d}"
        labels = ["abnormal", "normal"] if patient_idx < 4 else [
            "abnormal" if patient_idx % 2 == 0 else "normal"
        ]
        for recording_idx, class_name in enumerate(labels):
            path = (
                tmp_path / "edf" / "train" / class_name / "01_tcp_ar"
                / f"{patient}_s{recording_idx + 1:03d}_t000.edf"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    for class_name in ("abnormal", "normal"):
        path = (
            tmp_path / "edf" / "eval" / class_name / "01_tcp_ar"
            / f"eval{class_name}_s001_t000.edf"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _tuab_cfg(tmp_path):
    return {
        "paths": {"data_root": str(tmp_path)},
        "dataset": {
            "active": "tuab",
            "tuab": {
                "edf_dir": "edf",
                "reference_scheme": "ar",
                "montage_dir": "01_tcp_ar",
                "train_partition": "train",
                "eval_partition": "eval",
                "validation_fraction": 0.1,
                "max_recording_sec": 1200.0,
                "classes": {
                    "abnormal": {"folder": "abnormal", "label": 0},
                    "normal": {"folder": "normal", "label": 1},
                },
            },
        },
        "split": {"random_seed": 42},
        "preprocessing": {"epoch_length_sec": 20.0},
    }


def test_tuab_index_uses_recording_labels_and_official_eval(tmp_path):
    _make_tuab_tree(tmp_path)
    cfg = _tuab_cfg(tmp_path)
    index = assign_dataset_splits(build_recording_index(cfg), cfg)

    assert set(index["dataset_name"]) == {"tuab"}
    assert (index["evaluation_id"] == index["recording_id"]).all()
    assert set(index.loc[index["source_partition"] == "eval", "split"]) == {"test"}
    assert set(index.loc[index["class_name"] == "abnormal", "label"]) == {0}
    assert set(index.loc[index["class_name"] == "normal", "label"]) == {1}


def test_tuab_mixed_label_patient_stays_on_one_split(tmp_path):
    _make_tuab_tree(tmp_path)
    cfg = _tuab_cfg(tmp_path)
    index = assign_dataset_splits(build_recording_index(cfg), cfg)
    mixed = index[index["patient_id"] == "train000"]

    assert set(mixed["label"]) == {0, 1}
    assert mixed["evaluation_id"].nunique() == 2
    assert mixed["split"].nunique() == 1
    train_index = index[index["source_partition"] == "train"]
    assert train_index.groupby("patient_id")["split"].nunique().max() == 1


def test_tuab_recording_intervals_cap_at_first_twenty_minutes(tmp_path):
    cfg = _tuab_cfg(tmp_path)
    row = {"edf_path": str(tmp_path / "example.edf")}
    assert get_recording_intervals(row, cfg, 1800.0) == [(0.0, 1200.0)]
    assert get_recording_intervals(row, cfg, 900.0) == [(0.0, 900.0)]
