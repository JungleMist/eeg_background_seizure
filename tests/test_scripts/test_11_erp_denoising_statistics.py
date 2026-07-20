from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "11_analyze_erp_denoising_statistics.py"
)
SPEC = importlib.util.spec_from_file_location("erp_statistics_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_tost_equivalence_passes_and_fails() -> None:
    inside = MODULE.tost_equivalence(
        np.array([-0.10, 0.00, 0.10, 0.05, -0.05]), margin=0.5, alpha=0.05
    )
    outside = MODULE.tost_equivalence(
        np.array([0.8, 0.9, 1.0, 1.1, 1.2]), margin=0.5, alpha=0.05
    )

    assert inside["equivalent"] is True
    assert inside["ci_90_low"] > -0.5
    assert inside["ci_90_high"] < 0.5
    assert outside["equivalent"] is False


def test_load_subject_metrics_rejects_duplicate_pairs(tmp_path: Path) -> None:
    path = tmp_path / "subject_metrics.csv"
    pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001"],
            "method": ["wiener", "wiener"],
            "ern_snr_db": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="重复"):
        MODULE.load_subject_metrics(path)


def test_analyze_writes_outputs_from_subject_metrics_only(tmp_path: Path) -> None:
    rows = []
    for index in range(12):
        subject_id = f"sub-{index + 1:03d}"
        standard = {
            "ern_snr_db": 3.0 + index * 0.01,
            "ern_waveform_r": 1.0,
            "ern_rmse_vs_standard_uv": 0.0,
            "ern_peak_uv": -5.0,
            "ern_peak_latency_ms": 80.0,
            "baseline_noise_sd_uv": 2.0,
            "lrp_peak_uv": 2.0,
            "lrp_peak_latency_ms": -50.0,
            "lrp_half_peak_onset_ms": -100.0,
            "classification_accuracy": 0.8,
            "classification_f1": 0.7,
            "classification_auc": 0.8,
            "target_change_rms_uv": 0.0,
        }
        raw = dict(standard)
        raw["ern_snr_db"] = 2.0 + index * 0.01
        wiener = dict(standard)
        wiener.update(
            {
                "ern_snr_db": standard["ern_snr_db"] + 1.0,
                "ern_waveform_r": 0.99,
                "ern_rmse_vs_standard_uv": 0.1,
                "ern_peak_uv": -4.9,
                "ern_peak_latency_ms": 81.0,
                "baseline_noise_sd_uv": 1.0,
                "lrp_peak_uv": 2.1,
                "lrp_peak_latency_ms": -49.0,
                "lrp_half_peak_onset_ms": -99.0,
                "target_change_rms_uv": 0.2,
            }
        )
        for method, metrics in (("standard", standard), ("raw", raw), ("wiener", wiener)):
            rows.append({"subject_id": subject_id, "method": method, **metrics})

    pd.DataFrame(rows).to_csv(tmp_path / "subject_metrics.csv", index=False)
    (tmp_path / "run_summary.json").write_text(
        json.dumps({"n_subjects": 12}), encoding="utf-8"
    )
    args = argparse.Namespace(
        results_dir=tmp_path,
        output_dir=None,
        candidate="wiener",
        references="standard,raw",
        equivalence_reference="standard",
        bootstrap_repeats=500,
        bootstrap_seed=7,
        alpha=0.05,
        ern_amplitude_margin_uv=1.0,
        ern_latency_margin_ms=10.0,
        lrp_amplitude_margin_uv=0.5,
        lrp_latency_margin_ms=10.0,
    )

    paths = MODULE.analyze(args)

    assert all(path.exists() for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["verdict"]["snr_increase_vs_equivalence_reference"] is True
    assert summary["verdict"]["ern_amplitude_equivalent"] is True
    assert summary["verdict"]["ern_latency_equivalent"] is True
    assert summary["verdict"]["supports_better_ern_exposure"] is True
    assert "analytic_or_bootstrap_SME" in summary["unavailable_without_trial_waveforms"]

    paired = pd.read_csv(paths["paired_tests"])
    snr = paired.loc[
        (paired["metric"] == "ern_snr_db") & (paired["reference"] == "standard")
    ].iloc[0]
    assert snr["n_pairs"] == 12
    assert snr["mean_difference"] == pytest.approx(1.0)
    assert snr["better_fraction"] == pytest.approx(1.0)
