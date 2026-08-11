"""Unit tests for ERP-CORE ERN ECMAD phase optimization helpers."""
from concurrent.futures import Future
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_bg.features.extraction import FEATURE_NAMES, extract_epoch_features


SCRIPT_PATH = Path("scripts/12_optimize_erp_core_ern_phase.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script12_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase_grid_includes_exact_pi_after_point_one_steps():
    module = _load_script()

    phases = module.build_phase_grid(0.0, math.pi, 0.1)

    assert len(phases) == 33
    np.testing.assert_allclose(phases[:-1], np.arange(0.0, 3.2, 0.1))
    assert phases[-1] == math.pi
    assert np.all(phases >= 0.0)
    assert np.all(phases <= math.pi)


def test_erp_layout_is_211_dimensions_without_changing_legacy_names():
    module = _load_script()
    rng = np.random.default_rng(42)
    epoch = rng.normal(size=(len(module.ERP_ERN_CHANNELS), 126))

    features = module.extract_epoch_features_for_layout(
        epoch,
        list(module.ERP_ERN_CHANNELS),
        125.0,
        channel_order=module.ERP_ERN_CHANNELS,
        symmetric_pairs=module.ERP_ERN_SYMMETRIC_PAIRS,
        nperseg=126,
        freq_band=(0.5, 30.0),
    )

    assert len(module.ERP_ERN_CHANNELS) == 19
    assert len(module.ERP_ERN_SYMMETRIC_PAIRS) == 8
    assert len(module.ERP_ERN_FEATURE_NAMES) == 211
    assert features.shape == (211,)
    assert np.isfinite(features).all()
    assert module.ERP_ERN_FEATURE_NAMES[-5:] == (
        "asym_O1_O2_delta",
        "asym_O1_O2_theta",
        "asym_O1_O2_alpha",
        "asym_O1_O2_beta",
        "asym_O1_O2_gamma",
    )

    legacy_epoch = rng.normal(size=(19, 250))
    legacy_channels = [
        "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
        "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
    ]
    assert len(FEATURE_NAMES) == 211
    assert extract_epoch_features(
        legacy_epoch, legacy_channels, 125.0, nperseg=250
    ).shape == (211,)


def test_subject_holdout_and_internal_folds_are_deterministic_and_disjoint():
    module = _load_script()
    subjects = [f"sub-{index:03d}" for index in range(10)]

    first = module.split_subjects(subjects, test_size=0.2, random_state=42)
    second = module.split_subjects(subjects, test_size=0.2, random_state=42)

    assert first == second
    train_subjects, test_subjects = first
    assert len(train_subjects) == 8
    assert len(test_subjects) == 2
    assert not set(train_subjects) & set(test_subjects)

    groups = np.repeat(train_subjects, 4)
    y = np.tile([0, 0, 1, 1], len(train_subjects))
    folds = module.make_group_folds(y, groups, n_splits=5, random_state=42)
    assert len(folds) == 5
    for train_index, val_index in folds:
        assert not set(groups[train_index]) & set(groups[val_index])
        assert set(y[val_index]) == {0, 1}


def test_phase_selection_tie_breaks_on_sd_then_lower_phase():
    module = _load_script()
    frame = pd.DataFrame(
        [
            {"phase_rad": 0.3, "primary_mean": 0.8, "primary_std": 0.04},
            {"phase_rad": 0.2, "primary_mean": 0.8, "primary_std": 0.02},
            {"phase_rad": 0.1, "primary_mean": 0.8, "primary_std": 0.02},
            {"phase_rad": 0.0, "primary_mean": 0.7, "primary_std": 0.01},
        ]
    )

    selected = module.select_best_row(frame)

    assert selected["phase_rad"] == 0.1


def test_oof_scoring_uses_incorrect_as_positive_class():
    module = _load_script()
    y = np.array([0, 0, 1, 1, 0, 1])
    probabilities = np.array([0.05, 0.1, 0.9, 0.8, 0.2, 0.95])
    folds = [
        (np.array([1, 3, 4, 5]), np.array([0, 2])),
        (np.array([0, 2, 4, 5]), np.array([1, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]

    scores = module.score_oof(y, probabilities, folds, metric="f1")

    assert scores["f1"] == 1.0
    assert scores["accuracy"] == 1.0
    assert scores["auroc"] == 1.0
    assert scores["primary_mean"] == 1.0


def test_phase_search_config_defaults_are_reproducible():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")
    search = cfg["erp_core"]["phase_search"]

    assert search["metric"] == "auroc"
    assert search["test_size"] == 0.2
    assert search["cv_folds"] == 5
    assert search["random_state"] == 42
    assert search["phase_step"] == 0.1
    assert search["random_search_iterations"] == 50
    assert search["device"] == "cpu"


def test_xgboost_oof_predicts_every_trial_without_subject_leakage():
    module = _load_script()
    rng = np.random.default_rng(7)
    groups = np.repeat([f"sub-{index:02d}" for index in range(10)], 4)
    y = np.tile([0, 0, 1, 1], 10)
    X = rng.normal(size=(len(y), 8))
    X[:, 0] += y * 0.5
    folds = module.make_group_folds(y, groups, n_splits=5, random_state=42)

    probabilities = module.oof_predict(
        X,
        y,
        folds,
        params={
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
        random_state=42,
        workers=1,
        device="cpu",
    )

    assert probabilities.shape == (len(y),)
    assert np.isfinite(probabilities).all()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_subject_phase_jobs_use_serial_path_for_one_worker(monkeypatch, tmp_path):
    module = _load_script()
    calls = []
    helper = object()

    def fake_worker(args, helpers):
        calls.append((args, helpers))
        return args[0]

    def fail_executor(*args, **kwargs):
        raise AssertionError("serial path must not construct a process pool")

    monkeypatch.setattr(module, "_extract_subject_phases_worker", fake_worker)
    monkeypatch.setattr(module, "_load_script10", lambda: helper)
    monkeypatch.setattr(module, "ProcessPoolExecutor", fail_executor)
    phases = np.asarray([0.0, 0.1, math.pi])
    subjects = ["sub-001", "sub-002"]
    recordings = {
        subject: tmp_path / f"{subject}_task-ERN_eeg.set"
        for subject in subjects
    }

    module.extract_subjects_phases(
        subjects,
        recordings,
        phases,
        {"test": True},
        tmp_path / "cache",
        False,
        workers=1,
        stage="training",
    )

    assert [args[0][0] for args in calls] == subjects
    assert all(args[0][2] == (0.0, 0.1, math.pi) for args in calls)
    assert all(args[1] is helper for args in calls)


def test_subject_phase_jobs_submit_one_full_phase_task_per_subject(
    monkeypatch, tmp_path
):
    module = _load_script()
    submitted = []
    executors = []

    class FakeExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.shutdown_args = None
            executors.append(self)

        def submit(self, function, args):
            submitted.append(args)
            future = Future()
            try:
                future.set_result(function(args))
            except Exception as exc:
                future.set_exception(exc)
            return future

        def shutdown(self, wait, cancel_futures):
            self.shutdown_args = (wait, cancel_futures)

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        module, "_extract_subject_phases_worker", lambda args: args[0]
    )
    subjects = ["sub-001", "sub-002", "sub-003"]
    recordings = {
        subject: tmp_path / f"{subject}_task-ERN_eeg.set"
        for subject in subjects
    }
    phases = np.asarray([0.0, 0.1, math.pi])

    module.extract_subjects_phases(
        subjects,
        recordings,
        phases,
        {"test": True},
        tmp_path / "cache",
        False,
        workers=2,
        stage="training",
    )

    assert len(executors) == 1
    assert executors[0].max_workers == 2
    assert executors[0].mp_context.get_start_method() == "spawn"
    assert executors[0].shutdown_args == (True, True)
    assert [args[0] for args in submitted] == subjects
    assert all(args[2] == (0.0, 0.1, math.pi) for args in submitted)


def test_subject_phase_worker_loads_helpers_and_limits_native_threads(
    monkeypatch, tmp_path
):
    module = _load_script()
    helper = object()
    calls = []
    limits = []

    @contextmanager
    def fake_threadpool_limits(limits: int):
        calls.append(("limit_enter", limits))
        yield
        calls.append(("limit_exit", limits))

    def fake_extract(
        subject_id, recording, phases, cfg, cache_root, helpers, force
    ):
        limits.append(
            (
                subject_id,
                recording,
                phases.tolist(),
                cfg,
                cache_root,
                helpers,
                force,
            )
        )

    monkeypatch.setattr(module, "_load_script10", lambda: helper)
    monkeypatch.setattr(module, "threadpool_limits", fake_threadpool_limits)
    monkeypatch.setattr(module, "extract_subject_phases", fake_extract)

    result = module._extract_subject_phases_worker(
        (
            "sub-001",
            str(tmp_path / "sub-001.set"),
            (0.0, 0.1),
            {"test": True},
            str(tmp_path / "cache"),
            False,
        )
    )

    assert result == "sub-001"
    assert calls == [("limit_enter", 1), ("limit_exit", 1)]
    assert limits[0][0] == "sub-001"
    assert limits[0][2] == [0.0, 0.1]
    assert limits[0][5] is helper


def test_subject_phase_job_error_identifies_subject(monkeypatch, tmp_path):
    module = _load_script()

    def fail_worker(args, helpers):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(module, "_extract_subject_phases_worker", fail_worker)
    monkeypatch.setattr(module, "_load_script10", object)

    with np.testing.assert_raises_regex(
        RuntimeError, "training feature generation failed for sub-001"
    ):
        module.extract_subjects_phases(
            ["sub-001"],
            {"sub-001": tmp_path / "sub-001.set"},
            np.asarray([0.0]),
            {"test": True},
            tmp_path / "cache",
            False,
            workers=1,
            stage="training",
        )


def test_xgboost_worker_count_still_sets_n_jobs():
    module = _load_script()

    params = module._xgb_params(
        {"max_depth": 2},
        np.asarray([0, 0, 1]),
        random_state=42,
        workers=3,
        device="cpu",
    )

    assert params["n_jobs"] == 3
