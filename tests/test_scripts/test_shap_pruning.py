"""Unit tests for _write_shap_pruning_report."""
import json
import numpy as np
import pytest
from pathlib import Path
import importlib.util


def _load_script(path="scripts/06_train_xgboost.py"):
    spec = importlib.util.spec_from_file_location("script06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pruning_report_written(tmp_path):
    mod = _load_script()
    rng = np.random.default_rng(0)
    shap_vals = rng.standard_normal((10, 5))
    # Make feature 2 small so it gets pruned
    shap_vals[:, 2] *= 1e-6
    names = ["f0", "f1", "f2", "f3", "f4"]
    surviving = mod._write_shap_pruning_report(shap_vals, names, tmp_path, threshold=1e-4)
    report_path = tmp_path / "pruning_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["n_features_pruned"] >= 1
    assert "f2" in [p["feature"] for p in report["pruned_features"]]
    assert 2 not in surviving


def test_all_survive(tmp_path):
    mod = _load_script()
    shap_vals = np.ones((5, 3)) * 0.5
    names = ["a", "b", "c"]
    surviving = mod._write_shap_pruning_report(shap_vals, names, tmp_path, threshold=1e-4)
    assert surviving == [0, 1, 2]
    report = json.loads((tmp_path / "pruning_report.json").read_text())
    assert report["n_features_pruned"] == 0
