"""Tests for comparison-root ERN window extraction and split handling."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path("scripts/16_compare_erp_core_comparison_eegnet.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script16_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discover_conditions_indexes_all_matching_subjects(tmp_path: Path) -> None:
    module = _load_script()
    raw = tmp_path / "raw"
    specific = tmp_path / "g0.05_specific"
    ica = tmp_path / "ica"
    for directory in (raw, specific, ica):
        directory.mkdir()
    (raw / "sub-001_raw.fif").touch()
    (raw / "sub-002_raw.fif").touch()
    (specific / "sub-001_g0.05_specific.edf").touch()
    (specific / "sub-002_g0.05_specific.edf").touch()
    (ica / "sub-001_ica.fif").touch()
    (ica / "sub-002_ica.fif").touch()

    conditions, recordings = module.discover_conditions(tmp_path)

    assert conditions == ["raw", "g0.05_specific", "ica"]
    assert sorted(recordings) == ["sub-001", "sub-002"]
    assert recordings["sub-001"]["raw"].name == "sub-001_raw.fif"


def test_discover_conditions_rejects_missing_processed_subject(tmp_path: Path) -> None:
    module = _load_script()
    raw = tmp_path / "raw"
    processed = tmp_path / "specific"
    raw.mkdir()
    processed.mkdir()
    (raw / "sub-001_raw.fif").touch()
    (raw / "sub-002_raw.fif").touch()
    (processed / "sub-001_specific.edf").touch()

    try:
        module.discover_conditions(tmp_path)
    except ValueError as exc:
        assert "sub-002" in str(exc)
    else:
        raise AssertionError("Missing processed subject was not rejected")


def test_response_pairing_preserves_correct_and_incorrect_labels() -> None:
    module = _load_script()
    helpers = module._load_script10()
    events = np.asarray(
        [[100, 0, 11], [200, 0, 111], [300, 0, 11], [400, 0, 211]],
        dtype=np.int64,
    )

    table = helpers.build_response_table(
        events,
        {"11": 11, "111": 111, "211": 211},
        sfreq=100.0,
        max_lag_sec=2.0,
    )

    assert table["correct"].tolist() == [True, False]
    assert (~table["correct"].to_numpy(bool)).astype(np.int8).tolist() == [0, 1]


def test_window_bounds_and_branch_time_mapping_are_shared() -> None:
    module = _load_script()
    raw_events = module._branch_events(
        np.asarray([1000, 100], dtype=np.int64),
        raw_sfreq=250.0,
        raw_first_samp=0,
        branch_sfreq=250.0,
        branch_first_samp=0,
    )

    valid = module._window_bounds(raw_events, 250.0, -0.6, 0.4, n_times=2000)

    assert raw_events[:, 0].tolist() == [1000, 100]
    assert valid.tolist() == [True, False]


def test_two_stage_subject_split_has_no_overlap() -> None:
    module = _load_script()
    partitions = module.split_subjects_two_stage(
        [f"sub-{index:03d}" for index in range(10)],
        test_size=0.2,
        validation_size=0.2,
        random_state=42,
    )

    sets = {key: set(value) for key, value in partitions.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert set().union(*sets.values()) == {f"sub-{index:03d}" for index in range(10)}
