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
