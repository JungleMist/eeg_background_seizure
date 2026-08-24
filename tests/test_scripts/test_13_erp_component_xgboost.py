"""Tests for Script 13's distributed ECMAD XGBoost/SVM experiment."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path("scripts/13_compare_erp_core_ern_components.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script13_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module):
    return module.load_config("configs/erp_core_flankers.yaml")


def test_shared_config_and_feature_layouts_match_distributed_design():
    module = _load_script()
    cfg = _config(module)
    experiment = cfg["erp_core"]["distributed_component_models"]
    layouts = module.build_feature_layouts(cfg)
    names = module.condition_feature_names(layouts)

    assert tuple(cfg["erp_core"]["distributed_components"]["steps"]) == module.STEP_NAMES
    assert experiment == {
        "output_dir": "results/erp_core_ern_distributed_component_models",
        "test_size": 0.2,
        "random_state": 42,
        "decision_threshold": 0.5,
        "xgboost": {"device": "cpu", "grid_search_enabled": False},
        "svm": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": True,
        },
    }
    assert cfg["erp_core"]["distributed_components"]["cache_subdir"] == (
        "cache/erp_core_ern_distributed_components"
    )
    assert tuple(layouts) == module.CONDITIONS
    assert len(layouts["raw"].feature_names) == 211
    assert len(layouts["step1_specific"].feature_names) == 211
    assert len(layouts["step1_coherent"].feature_names) == 165
    assert len(layouts["step2_coherent"].feature_names) == 211
    assert len(layouts["step3_coherent"].feature_names) == 211
    assert layouts["step1_coherent"].channels == (
        "FP1", "FP2", "F3", "F4", "F7", "F8", "FC3", "FC4",
        "C3", "C4", "P3", "P4", "Fz", "FCz", "Cz",
    )
    assert names["step1_coherent"] == layouts["step1_coherent"].feature_names


def test_coherent_layout_uses_step_channels_and_complete_pairs_only():
    module = _load_script()
    cfg = deepcopy(_config(module))
    cfg["erp_core"]["distributed_components"]["steps"]["step1"][
        "channel_groups"
    ] = [["FP1", "FP2", "F3"]]

    layout = module.build_feature_layouts(cfg)["step1_coherent"]

    assert layout.channels == ("FP1", "FP2", "F3")
    assert layout.symmetric_pairs == (("FP1", "FP2"),)
    assert len(layout.feature_names) == 3 * 9 + 5


def test_cache_fingerprint_tracks_wiener_steps_and_layouts(tmp_path):
    module = _load_script()
    cfg = _config(module)
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"set")
    layouts = module.build_feature_layouts(cfg)
    baseline = module._cache_fingerprint(recording, cfg, layouts)

    changed_wiener = deepcopy(cfg)
    changed_wiener["wiener"]["coherence_threshold"] = 0.987
    changed_step = deepcopy(cfg)
    changed_step["erp_core"]["distributed_components"]["steps"]["step2"][
        "phase_gate_threshold_rad"
    ] = 0.25
    changed_layout = deepcopy(cfg)
    changed_layout["erp_core"]["distributed_components"]["steps"]["step1"][
        "channel_groups"
    ] = [["FP1", "FP2"]]

    assert module._cache_fingerprint(
        recording, changed_wiener, module.build_feature_layouts(changed_wiener)
    ) != baseline
    assert module._cache_fingerprint(
        recording, changed_step, module.build_feature_layouts(changed_step)
    ) != baseline
    assert module._cache_fingerprint(
        recording, changed_layout, module.build_feature_layouts(changed_layout)
    ) != baseline


class _FakeRaw:
    def __init__(self, data: np.ndarray, channels: tuple[str, ...], sfreq: float = 125.0):
        self._data = np.asarray(data, dtype=np.float64).copy()
        self.ch_names = list(channels)
        self.info = {"sfreq": sfreq}
        self.n_times = self._data.shape[1]

    def copy(self):
        return _FakeRaw(self._data, tuple(self.ch_names), self.info["sfreq"])

    def get_data(self):
        return self._data.copy()

    def pick(self, channels):
        indices = [self.ch_names.index(channel) for channel in channels]
        self._data = self._data[indices]
        self.ch_names = list(channels)
        return self

    def reorder_channels(self, channels):
        return self.pick(channels)


class _FakeEpochs:
    def __init__(self, raw: _FakeRaw):
        self.ch_names = list(raw.ch_names)
        self.info = {"sfreq": raw.info["sfreq"]}
        self.metadata = pd.DataFrame({"correct": [True, True, False, False]})
        self.events = np.column_stack(
            [np.asarray([10, 30, 50, 70]), np.zeros(4, int), np.ones(4, int)]
        )
        base = raw.get_data()
        self._data = np.stack(
            [base[:, :64], base[:, 16:80], base[:, 32:96], base[:, 48:112]]
        )

    def get_data(self, picks=None, copy=False):
        data = self._data
        if picks is not None:
            data = data[:, [self.ch_names.index(channel) for channel in picks]]
        return data.copy() if copy else data


class _FakeHelpers:
    def __init__(self, raw):
        self.raw = raw
        self.read_calls = 0
        self.common_calls = 0

    def _read_recording(self, _mne, _recording):
        self.read_calls += 1
        return self.raw.copy()

    def _common_preprocess(self, raw, _cfg):
        self.common_calls += 1
        return raw.copy()

    def build_response_table(self, *_args):
        return pd.DataFrame({"sample": [10, 30, 50, 70]})

    def _make_shared_epochs(self, raws, *_args):
        return {condition: _FakeEpochs(raw) for condition, raw in raws.items()}

    def _task_epoch_issue(self, _task, _epochs):
        return None


class _FakeDistributed:
    def __init__(self, channels):
        self.ERP_CORE_EEG_CHANNELS = channels
        self.calls = 0

    def load_or_create_distributed_components(
        self, common, _recording, _cfg, subject_id, _helpers, cache_root, _force
    ):
        branches = {"raw": common}
        diagnostics = {}
        current = common
        for index, step in enumerate(("step1", "step2", "step3"), start=1):
            self.calls += 1
            specific = current.copy()
            specific._data *= 0.75
            coherent = current.copy()
            coherent._data = current.get_data() - specific.get_data()
            branches[f"{step}_specific"] = specific
            branches[f"{step}_coherent"] = coherent
            diagnostics[step] = {
                "windows": 2,
                "active_channels": list(current.ch_names),
                "max_abs_step_conservation_error_uv": 0.0,
            }
            current = specific
        continuous = cache_root / subject_id / "continuous"
        continuous.mkdir(parents=True, exist_ok=True)
        for component in (
            f"{step}_{kind}"
            for step in ("step1", "step2", "step3")
            for kind in ("specific", "coherent")
        ):
            (continuous / f"{component}.edf").write_bytes(b"shared")
        metadata = {
            "subject_id": subject_id,
            "steps": diagnostics,
            "max_abs_cumulative_conservation_error_uv": 0.0,
        }
        return branches, metadata, False


def test_subject_extraction_runs_three_steps_and_reuses_feature_cache(
    tmp_path, monkeypatch
):
    module = _load_script()
    cfg = _config(module)
    distributed_module = module._load_script15()
    channels = tuple(distributed_module.ERP_CORE_EEG_CHANNELS)
    rng = np.random.default_rng(42)
    raw = _FakeRaw(rng.normal(scale=1e-6, size=(len(channels), 128)), channels)
    helpers = _FakeHelpers(raw)
    distributed = _FakeDistributed(channels)
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"set")
    import mne

    monkeypatch.setattr(
        mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (np.asarray([[1, 0, 1]]), {"event": 1}),
    )
    dataset, diagnostics, cached = module.extract_subject_components(
        "sub-001", recording, cfg, tmp_path / "cache", helpers, False, distributed
    )

    assert not cached
    assert distributed.calls == 3
    assert set(dataset.features) == set(module.CONDITIONS)
    assert dataset.features["raw"].shape == (4, 211)
    assert dataset.features["step1_coherent"].shape == (4, 165)
    assert dataset.features["step2_coherent"].shape == (4, 211)
    np.testing.assert_array_equal(dataset.y, [0, 0, 1, 1])
    assert tuple(diagnostics["steps"]) == module.STEP_NAMES
    assert len(list((tmp_path / "cache").rglob("*.edf"))) == 6

    second_helpers = _FakeHelpers(raw)
    second_distributed = _FakeDistributed(channels)
    cached_dataset, _, cached = module.extract_subject_components(
        "sub-001",
        recording,
        cfg,
        tmp_path / "cache",
        second_helpers,
        False,
        second_distributed,
    )
    assert cached
    assert second_helpers.read_calls == 0
    assert second_distributed.calls == 0
    np.testing.assert_allclose(cached_dataset.matrix("step3_specific"), dataset.matrix("step3_specific"))


def test_group_folds_and_subject_split_are_disjoint():
    module = _load_script()
    subjects = [f"sub-{index:02d}" for index in range(10)]
    groups = np.repeat(subjects, 4)
    y = np.tile([0, 0, 1, 1], len(subjects))
    folds = module.make_group_folds(y, groups, n_splits=5, random_state=42)

    test_counts = np.zeros(len(y), dtype=int)
    for train_index, test_index in folds:
        assert not set(groups[train_index]) & set(groups[test_index])
        test_counts[test_index] += 1
    np.testing.assert_array_equal(test_counts, 1)
    first = module.split_subjects(subjects, test_size=0.2, random_state=42)
    second = module.split_subjects(subjects, test_size=0.2, random_state=42)
    assert first == second
    assert len(first[0]) == 8
    assert len(first[1]) == 2
    assert not set(first[0]) & set(first[1])


def test_grid_space_and_fixed_svm_policy():
    module = _load_script()
    cfg = _config(module)
    grid = module.build_grid_param_space(cfg)
    assert grid["model__n_estimators"] == [500]
    assert set(grid) == {
        f"model__{key}" for key in cfg["ml"]["xgboost"]["param_grid"]
    }

    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.tile([0, 1], 5)
    scaler, fitted, params = module.fit_svm(
        X, y, 42, cfg["erp_core"]["distributed_component_models"]["svm"]
    )
    assert fitted.kernel == "rbf"
    assert fitted.class_weight == "balanced"
    assert fitted.probability is True
    assert params["C"] == 1.0
    assert scaler.transform(X).shape == X.shape


def test_classification_metrics_report_minority_class_performance():
    module = _load_script()
    scores = module.classification_metrics(
        np.asarray([0, 0, 0, 1], dtype=np.int8),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        0.5,
    )
    assert scores["auroc"] == 1.0
    assert scores["auprc"] == 1.0
    assert np.isclose(scores["f1"], 2.0 / 3.0)
    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
    assert np.isclose(scores["specificity"], 2.0 / 3.0)
    assert np.isclose(scores["balanced_accuracy"], 5.0 / 6.0)
    assert scores["accuracy"] == 0.75


class _FakeClassifier:
    fit_calls = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, verbose=False):
        self.fit_calls.append(self)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        values = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - values, values])


class _FakeGridSearchCV:
    calls = []

    def __init__(self, estimator, param_grid, cv, scoring, n_jobs, verbose, refit,
                 error_score, return_train_score):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.calls.append(self)

    def fit(self, X, y):
        scaler = StandardScaler().fit(X)
        model = self.estimator.named_steps["model"]
        model.fit(scaler.transform(X), y)
        self.best_estimator_ = Pipeline([("scaler", scaler), ("model", model)])
        self.best_params_ = {key: values[0] for key, values in self.param_grid.items()}
        self.best_score_ = 0.75
        self.cv_results_ = {
            "params": [self.best_params_],
            "rank_test_score": np.asarray([1]),
            "mean_test_score": np.asarray([self.best_score_]),
            "std_test_score": np.asarray([0.01]),
            "mean_fit_time": np.asarray([0.1]),
            "mean_score_time": np.asarray([0.01]),
        }
        return self


def _run_fixture(module, tmp_path, monkeypatch):
    cfg = deepcopy(_config(module))
    cfg["erp_core"]["data_dir"] = str(tmp_path / "data")
    cfg["erp_core"]["distributed_components"]["cache_subdir"] = str(
        tmp_path / "cache"
    )
    subjects = [f"sub-{index:02d}" for index in range(10)]
    recordings = [
        {"subject_id": subject, "ern": tmp_path / f"{subject}_task-ERN_eeg.set"}
        for subject in subjects
    ]
    groups = np.repeat(subjects, 4)
    y = np.tile([0, 0, 1, 1], len(subjects)).astype(np.int8)
    rng = np.random.default_rng(7)
    layouts = module.build_feature_layouts(cfg)
    dataset = module.ComponentDataset(
        features={
            condition: rng.normal(size=(len(y), len(layouts[condition].feature_names)))
            for condition in module.CONDITIONS
        },
        y=y,
        subject_ids=groups,
        samples=np.tile([10, 20, 30, 40], len(subjects)),
    )

    class FakeHelpers:
        def _resolve_recordings(self, *_args):
            return recordings

        def _select_recordings(self, values, **_kwargs):
            return values

    diagnostics = {
        "steps": {
            step: {"active_channels": [], "max_abs_step_conservation_error_uv": 0.0}
            for step in module.STEP_NAMES
        },
        "max_abs_cumulative_conservation_error_uv": 0.0,
    }
    eligibility = [
        {
            "subject_id": subject,
            "recording": str(recordings[index]["ern"]),
            "eligible": True,
            "cached": False,
            "n_trials": 4,
            "n_correct": 2,
            "n_incorrect": 2,
            "reason": "",
            "diagnostics": diagnostics,
        }
        for index, subject in enumerate(subjects)
    ]
    monkeypatch.setattr(module, "load_config", lambda _path: deepcopy(cfg))
    monkeypatch.setattr(module, "_load_script10", lambda: FakeHelpers())
    monkeypatch.setattr(module, "extract_all_subjects", lambda *_args: eligibility)
    monkeypatch.setattr(module, "load_component_dataset", lambda *_args: dataset)
    monkeypatch.setattr(module.xgb, "XGBClassifier", _FakeClassifier)
    monkeypatch.setattr(module, "SVC", _FakeClassifier)
    monkeypatch.setattr(module, "GridSearchCV", _FakeGridSearchCV)
    monkeypatch.setattr(
        module.joblib, "dump", lambda _value, path: Path(path).write_bytes(b"model")
    )
    _FakeClassifier.fit_calls = []
    _FakeGridSearchCV.calls = []
    return cfg


@pytest.mark.parametrize(
    ("model", "expected_roots", "expected_fits"),
    [
        ("both", {"xgboost", "svm"}, 14),
        ("xgboost", {"xgboost"}, 7),
        ("svm", {"svm"}, 7),
    ],
)
def test_run_modes_write_separate_complete_outputs(
    tmp_path, monkeypatch, model, expected_roots, expected_fits
):
    module = _load_script()
    _run_fixture(module, tmp_path, monkeypatch)
    out = module.run(
        "configs/erp_core_flankers.yaml",
        output_dir=tmp_path / "results",
        workers=1,
        model=model,
    )

    assert {path.name for path in out.iterdir() if path.is_dir()} == expected_roots
    assert len(_FakeClassifier.fit_calls) == expected_fits
    for model_name in expected_roots:
        root = out / model_name
        metrics = pd.read_csv(root / "condition_metrics.csv")
        assert len(metrics) == 7
        assert set(module._METRIC_NAMES).issubset(metrics.columns)
        assert pd.read_csv(root / "predictions.csv").groupby("condition").size().eq(8).all()
        assert (root / "subject_metrics.csv").is_file()
        assert (root / "comparison_summary.csv").is_file()
        assert (root / "condition_deltas.csv").is_file()
        assert (root / "conditions" / "raw" / "model.joblib").is_file()
    assert not (out / "condition_metrics.csv").exists()
    assert not (out / "shap").exists()
    assert not (out / "subjects").exists()
    assert not list(out.rglob("*.edf"))
    summary = json.loads((out / "run_summary.json").read_text())
    assert set(summary["selected_models"]) == expected_roots
    assert summary["shap_output"] is False
    assert summary["continuous_component_output"] is False
    assert summary["shared_component_cache"] == str((tmp_path / "cache").resolve())
    assert "comparison_summaries" not in summary


def test_xgboost_grid_search_stays_training_only(tmp_path, monkeypatch):
    module = _load_script()
    _run_fixture(module, tmp_path, monkeypatch)
    out = module.run(
        "configs/erp_core_flankers.yaml",
        output_dir=tmp_path / "results",
        model="xgboost",
        grid_search_override=True,
    )
    assert len(_FakeGridSearchCV.calls) == 7
    assert all(len(search.cv) == 5 for search in _FakeGridSearchCV.calls)
    assert all(search.scoring == "average_precision" for search in _FakeGridSearchCV.calls)
    metrics = pd.read_csv(out / "xgboost" / "condition_metrics.csv")
    assert metrics["training_strategy"].eq("grid_search").all()
    assert metrics["grid_best_inner_auprc"].eq(0.75).all()
    assert (out / "xgboost" / "conditions" / "raw" / "grid_search_results.csv").is_file()


def test_svm_rejects_explicit_grid_search(tmp_path):
    module = _load_script()
    with pytest.raises(ValueError, match="applies only to XGBoost"):
        module.run(
            "configs/erp_core_flankers.yaml",
            output_dir=tmp_path / "results",
            model="svm",
            grid_search_override=True,
        )
