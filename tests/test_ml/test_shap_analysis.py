"""Unit tests for eeg_bg.ml.shap_analysis."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from eeg_bg.features.band_power import BANDS
from eeg_bg.features.extraction import FEATURE_NAMES
from eeg_bg.ml.shap_analysis import (
    compute_shap_values,
    aggregate_shap_by_band,
    aggregate_shap_by_channel,
    plot_shap_summary,
    plot_shap_comparison,
)


# ── compute_shap_values ───────────────────────────────────────────────────────

def test_compute_shap_values_shape(full_feature_xgb_model):
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((20, 211)).astype(np.float32)
    sv  = compute_shap_values(full_feature_xgb_model, X, FEATURE_NAMES)
    assert sv.shape == (20, 211)


def test_compute_shap_values_finite(full_feature_xgb_model):
    rng = np.random.default_rng(1)
    X   = rng.standard_normal((10, 211)).astype(np.float32)
    sv  = compute_shap_values(full_feature_xgb_model, X, FEATURE_NAMES)
    assert np.all(np.isfinite(sv))


# ── aggregate_shap_by_band ────────────────────────────────────────────────────

def test_aggregate_shap_by_band_keys():
    rng = np.random.default_rng(2)
    sv  = rng.standard_normal((10, 211))
    result = aggregate_shap_by_band(sv, FEATURE_NAMES)
    expected_keys = set(BANDS.keys()) | {"hjorth", "spectral_entropy", "asymmetry"}
    assert set(result.keys()) == expected_keys


def test_aggregate_shap_by_band_nonnegative():
    rng = np.random.default_rng(3)
    sv  = rng.standard_normal((10, 211))
    result = aggregate_shap_by_band(sv, FEATURE_NAMES)
    for v in result.values():
        assert v >= 0.0


def test_aggregate_shap_by_band_float_values():
    rng = np.random.default_rng(4)
    sv  = rng.standard_normal((5, 211))
    result = aggregate_shap_by_band(sv, FEATURE_NAMES)
    for v in result.values():
        assert isinstance(v, float)


# ── aggregate_shap_by_channel ─────────────────────────────────────────────────

def test_aggregate_shap_by_channel_keys():
    rng = np.random.default_rng(5)
    sv  = rng.standard_normal((10, 211))
    result = aggregate_shap_by_channel(sv, FEATURE_NAMES)
    assert len(result) == 19


def test_aggregate_shap_by_channel_nonnegative():
    rng = np.random.default_rng(6)
    sv  = rng.standard_normal((10, 211))
    result = aggregate_shap_by_channel(sv, FEATURE_NAMES)
    for v in result.values():
        assert v >= 0.0


# ── plot_shap_summary ─────────────────────────────────────────────────────────

def test_plot_shap_summary_creates_file(full_feature_xgb_model, tmp_path):
    rng = np.random.default_rng(7)
    X   = rng.standard_normal((15, 211)).astype(np.float32)
    sv  = compute_shap_values(full_feature_xgb_model, X, FEATURE_NAMES)
    out = tmp_path / "shap_summary.png"
    plot_shap_summary(sv, X, FEATURE_NAMES, title="Test", output_path=out, dpi=72)
    assert out.exists()
    assert out.stat().st_size > 1000


# ── plot_shap_comparison ──────────────────────────────────────────────────────

def test_plot_shap_comparison_creates_file(tmp_path):
    rng = np.random.default_rng(8)
    sv  = rng.standard_normal((10, 211))
    band_agg = aggregate_shap_by_band(sv, FEATURE_NAMES)
    ch_agg   = aggregate_shap_by_channel(sv, FEATURE_NAMES)

    results = {
        "raw":    {"shap_by_band": band_agg, "shap_by_channel": ch_agg},
        "ica":    {"shap_by_band": band_agg, "shap_by_channel": ch_agg},
        "wiener": {"shap_by_band": band_agg, "shap_by_channel": ch_agg},
    }
    out = tmp_path / "shap_comparison.png"
    plot_shap_comparison(results, output_path=out, dpi=72)
    assert out.exists()
    assert out.stat().st_size > 1000
