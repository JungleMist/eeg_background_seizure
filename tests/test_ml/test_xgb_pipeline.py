"""Unit tests for eeg_bg.ml.xgb_pipeline."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from eeg_bg.ml.xgb_pipeline import (
    subject_level_predict,
    evaluate_subject_level,
)


# ── subject_level_predict ────────────────────────────────────────────────────

def test_subject_level_predict_columns(tiny_xgb_model):
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((6, 10)).astype(np.float32)
    ids = ["s001", "s001", "s002", "s002", "s003", "s003"]
    y   = np.array([0, 0, 1, 1, 0, 0])

    df = subject_level_predict(tiny_xgb_model, X, ids, y)
    assert set(df.columns) == {"subject_id", "pred_proba", "true_label"}


def test_subject_level_predict_aggregation(tiny_xgb_model):
    """Two epochs per subject → one row per subject."""
    rng = np.random.default_rng(1)
    X   = rng.standard_normal((4, 10)).astype(np.float32)
    ids = ["s001", "s001", "s002", "s002"]
    y   = np.array([0, 0, 1, 1])

    df = subject_level_predict(tiny_xgb_model, X, ids, y)
    assert len(df) == 2
    assert set(df["subject_id"]) == {"s001", "s002"}


def test_subject_level_predict_proba_in_range(tiny_xgb_model):
    rng = np.random.default_rng(2)
    X   = rng.standard_normal((6, 10)).astype(np.float32)
    ids = ["s1", "s1", "s2", "s2", "s3", "s3"]
    y   = np.array([0, 0, 1, 1, 0, 0])

    df = subject_level_predict(tiny_xgb_model, X, ids, y)
    assert df["pred_proba"].between(0.0, 1.0).all()


# ── evaluate_subject_level ───────────────────────────────────────────────────

def test_evaluate_subject_level_keys():
    df = pd.DataFrame({
        "subject_id":  ["s1", "s2", "s3", "s4"],
        "pred_proba":  [0.9, 0.8, 0.2, 0.1],
        "true_label":  [1,   1,   0,   0],
    })
    metrics = evaluate_subject_level(df)
    assert set(metrics.keys()) == {"auroc", "f1", "accuracy", "threshold"}


def test_evaluate_subject_level_perfect():
    df = pd.DataFrame({
        "subject_id":  ["s1", "s2", "s3", "s4"],
        "pred_proba":  [0.95, 0.9, 0.05, 0.1],
        "true_label":  [1,    1,   0,    0],
    })
    metrics = evaluate_subject_level(df)
    assert metrics["auroc"]    == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluate_subject_level_float_outputs():
    df = pd.DataFrame({
        "subject_id":  ["s1", "s2"],
        "pred_proba":  [0.6, 0.4],
        "true_label":  [1,   0],
    })
    metrics = evaluate_subject_level(df)
    for v in metrics.values():
        assert isinstance(v, float)
