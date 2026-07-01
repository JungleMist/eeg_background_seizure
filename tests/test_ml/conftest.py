"""Fixtures for test_ml — includes a tiny pre-fitted XGBClassifier."""
import numpy as np
import pytest
import xgboost as xgb

from eeg_bg.features.extraction import FEATURE_NAMES


@pytest.fixture(scope="session")
def tiny_xgb_model():
    """XGBClassifier fitted on tiny synthetic data (fast, deterministic)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 10)).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30, dtype=np.int32)
    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X, y)
    return model


@pytest.fixture(scope="session")
def tiny_feature_names():
    return [f"feat_{i}" for i in range(10)]


@pytest.fixture(scope="session")
def full_feature_xgb_model():
    """XGBClassifier fitted on FEATURE_NAMES-length synthetic data."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((80, len(FEATURE_NAMES))).astype(np.float32)
    y = np.array([0] * 40 + [1] * 40, dtype=np.int32)
    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42,
        eval_metric="auc",
        verbosity=0,
    )
    model.fit(X, y)
    return model
