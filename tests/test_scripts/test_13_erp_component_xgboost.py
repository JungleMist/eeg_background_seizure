"""Tests for the seven-condition ERP-CORE ERN component experiment."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path("scripts/13_compare_erp_core_ern_components.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script13_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_and_current_feature_layouts_are_211_211_422():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")

    full, coherent = module.build_feature_layouts(cfg)
    names = module.condition_feature_names(full, coherent)

    assert cfg["erp_core"]["component_xgboost"] == {
        "output_dir": "results/erp_core_ern_component_xgboost",
        "cache_subdir": "erp_core_ern_components",
        "test_size": 0.2,
        "random_state": 42,
        "device": "cpu",
        "decision_threshold": 0.5,
    }
    assert full.channels == module.ERP_ERN_CHANNELS
    assert coherent.channels == module.ERP_ERN_CHANNELS
    assert len(full.feature_names) == 211
    assert len(coherent.feature_names) == 211
    assert len(names[module.COMBINED_CONDITION]) == 422
    assert names[module.COMBINED_CONDITION][0].startswith("specific__")
    assert names[module.COMBINED_CONDITION][-1].startswith("coherent__")


def test_coherent_layout_tracks_grouped_channels_and_complete_pairs_only():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    cfg = deepcopy(cfg)
    cfg["channels"]["channel_groups"] = [["FP1", "FP2", "F3"]]

    full, coherent = module.build_feature_layouts(cfg)
    names = module.condition_feature_names(full, coherent)

    assert coherent.channels == ("FP1", "FP2", "F3")
    assert coherent.symmetric_pairs == (("FP1", "FP2"),)
    assert len(coherent.feature_names) == 3 * 9 + 5
    assert len(names[module.COMBINED_CONDITION]) == 211 + 32


def test_component_cache_fingerprint_tracks_wiener_and_layout_changes(tmp_path):
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"set")
    full, coherent = module.build_feature_layouts(cfg)

    baseline = module._cache_fingerprint(recording, cfg, full, coherent)
    changed_wiener = deepcopy(cfg)
    changed_wiener["wiener"]["coherence_threshold"] = 0.987
    changed_layout = deepcopy(cfg)
    changed_layout["channels"]["channel_groups"] = [["FP1", "FP2"]]
    _, reduced_coherent = module.build_feature_layouts(changed_layout)

    assert module._cache_fingerprint(
        recording, changed_wiener, full, coherent
    ) != baseline
    assert module._cache_fingerprint(
        recording, changed_layout, full, reduced_coherent
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


class _FakeEpochs:
    def __init__(self, raw: _FakeRaw):
        self.ch_names = list(raw.ch_names)
        self.info = {"sfreq": raw.info["sfreq"]}
        self.metadata = pd.DataFrame(
            {"correct": [True, True, False, False]}
        )
        self.events = np.column_stack(
            [np.asarray([10, 30, 50, 70]), np.zeros(4, int), np.ones(4, int)]
        )
        base = raw.get_data()
        self._data = np.stack(
            [base[:, :64], base[:, 16:80], base[:, 32:96], base[:, 48:112]]
        )

    def __len__(self):
        return len(self._data)

    def get_data(self, picks=None, copy=False):
        if picks is None:
            data = self._data
        else:
            indices = [self.ch_names.index(channel) for channel in picks]
            data = self._data[:, indices]
        return data.copy() if copy else data


class _FakeHelpers:
    def __init__(self, raw):
        self.raw = raw
        self.read_calls = 0
        self.common_calls = 0
        self.ica_calls = 0
        self.wiener_calls = []

    def _read_recording(self, _mne, _recording):
        self.read_calls += 1
        return self.raw.copy()

    def _common_preprocess(self, raw, _cfg):
        self.common_calls += 1
        return raw.copy()

    def build_response_table(self, *_args):
        return pd.DataFrame({"sample": [10, 30, 50, 70]})

    def _standard_ica(self, raw, _cfg):
        self.ica_calls += 1
        result = raw.copy()
        result._data *= 0.8
        return result, [1, 3]

    def _wiener_continuous(self, raw, _cfg, subject_id):
        self.wiener_calls.append(subject_id)
        result = raw.copy()
        result._data *= 0.75
        return result, {
            "windows": 2,
            "processed_channel_windows": 3,
            "solve_failures": 0,
            "below_coherence_candidates": 1,
            "group_processing_rates": {},
            "window_diagnostics": [{"omitted": True}],
        }

    def _make_shared_epochs(self, raws, *_args):
        return {condition: _FakeEpochs(raw) for condition, raw in raws.items()}

    def _task_epoch_issue(self, _task, _epochs):
        return None


def test_subject_extraction_runs_one_ica_two_wieners_and_reuses_cache(
    tmp_path, monkeypatch
):
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    rng = np.random.default_rng(42)
    raw = _FakeRaw(
        rng.normal(scale=1e-6, size=(19, 128)),
        module.ERP_ERN_CHANNELS,
    )
    helpers = _FakeHelpers(raw)
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"set")
    import mne

    monkeypatch.setattr(
        mne,
        "events_from_annotations",
        lambda *_args, **_kwargs: (np.asarray([[1, 0, 1]]), {"event": 1}),
    )

    dataset, diagnostics, cached = module.extract_subject_components(
        "sub-001",
        recording,
        cfg,
        tmp_path / "cache",
        helpers,
        force=False,
    )

    assert not cached
    assert helpers.read_calls == 1
    assert helpers.common_calls == 1
    assert helpers.ica_calls == 1
    assert helpers.wiener_calls == ["sub-001", "sub-001_ica"]
    assert set(dataset.features) == set(module.BASE_CONDITIONS)
    assert dataset.features["raw"].shape == (4, 211)
    assert dataset.features["wiener_coherent"].shape == (4, 211)
    assert dataset.matrix(module.COMBINED_CONDITION).shape == (4, 422)
    np.testing.assert_array_equal(dataset.y, [0, 0, 1, 1])
    assert diagnostics["ica_excluded_components"] == [1, 3]
    assert "window_diagnostics" not in diagnostics["raw_wiener"]

    second_helpers = _FakeHelpers(raw)
    cached_dataset, _, cached = module.extract_subject_components(
        "sub-001",
        recording,
        cfg,
        tmp_path / "cache",
        second_helpers,
        force=False,
    )
    assert cached
    assert second_helpers.read_calls == 0
    np.testing.assert_allclose(
        cached_dataset.matrix(module.COMBINED_CONDITION),
        dataset.matrix(module.COMBINED_CONDITION),
        rtol=1e-6,
        atol=1e-6,
    )


def test_group_folds_are_subject_disjoint_and_cover_each_trial_once():
    module = _load_script()
    subjects = [f"sub-{index:02d}" for index in range(10)]
    groups = np.repeat(subjects, 4)
    y = np.tile([0, 0, 1, 1], len(subjects))

    folds = module.make_group_folds(y, groups, n_splits=5, random_state=42)

    assert len(folds) == 5
    test_counts = np.zeros(len(y), dtype=int)
    for train_index, test_index in folds:
        assert not set(groups[train_index]) & set(groups[test_index])
        assert set(y[train_index]) == {0, 1}
        assert set(y[test_index]) == {0, 1}
        test_counts[test_index] += 1
    np.testing.assert_array_equal(test_counts, 1)


def test_single_subject_split_is_deterministic_and_disjoint():
    module = _load_script()
    subjects = [f"sub-{index:02d}" for index in range(10)]

    first = module.split_subjects(subjects, test_size=0.2, random_state=42)
    second = module.split_subjects(subjects, test_size=0.2, random_state=42)

    assert first == second
    train_subjects, test_subjects = first
    assert len(train_subjects) == 8
    assert len(test_subjects) == 2
    assert not set(train_subjects) & set(test_subjects)


def test_grid_space_matches_script06_phase_one_policy():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")

    grid = module.build_grid_param_space(cfg)

    assert grid["model__n_estimators"] == [500]
    assert grid["model__max_depth"] == [3, 4, 5, 6]
    assert grid["model__learning_rate"] == [0.01, 0.05, 0.1, 0.3]
    assert set(grid) == {
        f"model__{key}" for key in cfg["ml"]["xgboost"]["param_grid"]
    }


def test_held_out_shap_aggregation_reports_component_shares():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    full, coherent = module.build_feature_layouts(cfg)
    metadata = module.build_combined_feature_metadata(full, coherent)
    values = np.ones((6, len(metadata)), dtype=float)
    coherent_indices = metadata.loc[
        metadata["component"] == "coherent", "feature_index"
    ].to_numpy(int)
    values[:, coherent_indices] = 3.0

    tables = module.aggregate_shap(values, metadata)

    assert len(tables["feature"]) == 422
    components = tables["component"].set_index("component")
    assert components.loc["specific", "mean_abs_shap_per_feature"] == 1.0
    assert components.loc["coherent", "mean_abs_shap_per_feature"] == 3.0
    assert np.isclose(components["total_abs_share"].sum(), 1.0)
    assert set(tables["channel"]["component"]) == {"specific", "coherent"}


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
    assert scores["confusion_matrix"] == [[2, 1], [0, 1]]


class _FakeClassifier:
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, verbose=False):
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        values = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - values, values])


class _FakeGridSearchCV:
    calls = []

    def __init__(
        self,
        estimator,
        param_grid,
        cv,
        scoring,
        n_jobs,
        verbose,
        refit,
        error_score,
        return_train_score,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.calls.append(self)

    def fit(self, X, y):
        scaler = StandardScaler().fit(X)
        model = self.estimator.named_steps["model"]
        model.fit(scaler.transform(X), y)
        self.best_estimator_ = Pipeline(
            [("scaler", scaler), ("model", model)]
        )
        self.best_params_ = {
            key: values[0] for key, values in self.param_grid.items()
        }
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


def test_full_run_writes_one_metric_row_per_condition_and_test_shap(
    tmp_path, monkeypatch
):
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    cfg = deepcopy(cfg)
    cfg["paths"]["cache_dir"] = str(tmp_path / "cache")
    cfg["erp_core"]["data_dir"] = str(tmp_path / "data")
    subjects = [f"sub-{index:02d}" for index in range(10)]
    recordings = [
        {
            "subject_id": subject,
            "ern": tmp_path / f"{subject}_task-ERN_eeg.set",
        }
        for subject in subjects
    ]
    rng = np.random.default_rng(7)
    groups = np.repeat(subjects, 4)
    y = np.tile([0, 0, 1, 1], len(subjects)).astype(np.int8)
    samples = np.tile([10, 20, 30, 40], len(subjects))
    features = {
        condition: rng.normal(size=(len(y), 211))
        for condition in module.BASE_CONDITIONS
    }
    dataset = module.ComponentDataset(
        features=features,
        y=y,
        subject_ids=groups,
        samples=samples,
    )

    class FakeHelpers:
        def _resolve_recordings(self, *_args):
            return recordings

        def _select_recordings(self, values, **_kwargs):
            return values

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
            "diagnostics": {
                "ica_excluded_components": [],
                "raw_wiener": {},
                "ica_wiener": {},
            },
        }
        for index, subject in enumerate(subjects)
    ]

    monkeypatch.setattr(module, "load_config", lambda _path: deepcopy(cfg))
    monkeypatch.setattr(module, "_load_script10", lambda: FakeHelpers())
    monkeypatch.setattr(module, "extract_all_subjects", lambda *_args: eligibility)
    monkeypatch.setattr(module, "load_component_dataset", lambda *_args: dataset)
    monkeypatch.setattr(module.xgb, "XGBClassifier", _FakeClassifier)
    _FakeGridSearchCV.calls = []
    monkeypatch.setattr(module, "GridSearchCV", _FakeGridSearchCV)
    monkeypatch.setattr(
        module,
        "compute_shap_values",
        lambda _model, X, _names: np.full_like(X, 0.25),
    )
    monkeypatch.setattr(
        module,
        "plot_shap_summary",
        lambda *_args, **_kwargs: Path(_args[4]).write_bytes(b"png"),
    )
    monkeypatch.setattr(
        module.joblib,
        "dump",
        lambda _value, path: Path(path).write_bytes(b"model"),
    )

    out = module.run(
        "configs/erp_core_flankers.yaml",
        output_dir=tmp_path / "results",
        workers=1,
    )

    condition_metrics = pd.read_csv(out / "condition_metrics.csv")
    predictions = pd.read_csv(out / "predictions.csv")
    manifest = pd.read_csv(out / "split_manifest.csv")
    assert len(condition_metrics) == 7
    assert condition_metrics["condition"].nunique() == 7
    expected_test_metrics = {
        "auprc",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
    }
    assert expected_test_metrics.issubset(condition_metrics.columns)
    assert condition_metrics["grid_best_inner_auprc"].eq(0.75).all()
    assert len(_FakeGridSearchCV.calls) == 7
    assert all(len(search.cv) == 5 for search in _FakeGridSearchCV.calls)
    assert all(
        search.scoring == "average_precision"
        for search in _FakeGridSearchCV.calls
    )
    assert predictions.groupby("condition").size().eq(8).all()
    assert len(manifest) == len(subjects)
    assert manifest["split"].value_counts().to_dict() == {"train": 8, "test": 2}
    shap_values = np.load(out / "shap" / "shap_values_test.npy")
    assert shap_values.shape == (8, 422)
    assert np.isfinite(shap_values).all()
    assert (out / "shap" / "shap_summary.png").is_file()
    assert (out / "conditions" / "raw" / "model.joblib").is_file()
    assert (out / "conditions" / "raw" / "best_params.json").is_file()
    assert (
        out / "conditions" / "raw" / "grid_search_results.csv"
    ).is_file()
    grid_results = pd.read_csv(
        out / "conditions" / "raw" / "grid_search_results.csv"
    )
    assert {
        "rank_test_auprc",
        "mean_test_auprc",
        "std_test_auprc",
    }.issubset(grid_results.columns)
