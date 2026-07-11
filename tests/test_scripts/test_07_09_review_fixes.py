from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _touch_condition(root: Path, relative: str) -> None:
    path = root / relative / "test_metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")


def test_archive_discovers_mixed_profile_and_legacy_layouts(tmp_path):
    script07 = _load_script("07_organize_experiment")
    _touch_condition(tmp_path, "base211/raw")
    _touch_condition(tmp_path, "base211_conn80/wiener")
    _touch_condition(tmp_path, "ica")

    found = script07._discover_profiles(tmp_path)

    assert found == {
        "base211": ["ica", "raw"],
        "base211_conn80": ["wiener"],
    }


def test_grid_prediction_loader_rejects_duplicate_subjects(tmp_path):
    script09 = _load_script("09_analyze_wiener_phase_grid")
    path = tmp_path / "pred.csv"
    pd.DataFrame(
        {
            "subject_id": ["s1", "s1"],
            "pred_proba": [0.2, 0.3],
            "true_label": [0, 0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate subject_id"):
        script09._load_predictions(path)


def test_grid_paired_delta_rejects_label_mismatch():
    script09 = _load_script("09_analyze_wiener_phase_grid")
    left = pd.DataFrame(
        {"subject_id": ["s1", "s2"], "pred_proba": [0.2, 0.8], "true_label": [0, 1]}
    )
    right = pd.DataFrame(
        {"subject_id": ["s1", "s2"], "pred_proba": [0.3, 0.7], "true_label": [1, 1]}
    )

    with pytest.raises(ValueError, match="inconsistent true_label"):
        script09._bootstrap_delta(left, right, repeats=10, seed=42)


def test_verification_epoch_sampling_is_deterministic_and_bounded():
    script04 = _load_script("04_run_verification")
    first = script04._sample_epoch_indices(20, 5, 42, "recording-a")
    second = script04._sample_epoch_indices(20, 5, 42, "recording-a")
    other = script04._sample_epoch_indices(20, 5, 42, "recording-b")

    assert len(first) == 5
    assert len(set(first)) == 5
    assert (first == second).all()
    assert not (first == other).all()
    assert len(script04._sample_epoch_indices(3, 5, 42, "recording-a")) == 3
    with pytest.raises(ValueError, match="must be >= 1"):
        script04._sample_epoch_indices(3, 0, 42, "recording-a")


def test_xgboost_stats_printer_ignores_feature_profile_metadata(capsys):
    script06 = _load_script("06_train_xgboost")
    stats = {
        "train": {
            "n_subjects": 2, "n_epochs": 8,
            "n_subjects_epilepsy": 1, "n_subjects_control": 1,
        },
        "val": {
            "n_subjects": 1, "n_epochs": 4,
            "n_subjects_epilepsy": 1, "n_subjects_control": 0,
        },
        "test": {
            "n_subjects": 1, "n_epochs": 4,
            "n_subjects_epilepsy": 0, "n_subjects_control": 1,
        },
        "feature_set": "base211",
    }

    script06._print_data_stats(stats)

    output = capsys.readouterr().out
    assert "Train" in output
    assert "Val" in output
    assert "Test" in output
    assert "Feature_set" not in output
