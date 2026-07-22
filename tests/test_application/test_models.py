import math

import pytest

from eeg_bg.application.models import (
    ArtifactSettings,
    ExtractionMode,
    ExtractionSpec,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
    pipeline_fingerprint,
)


def test_artifact_settings_defaults_and_validation():
    settings = ArtifactSettings()
    assert settings.as_serializable_dict() == {
        "enabled": True,
        "threshold_uv": 200.0,
    }
    with pytest.raises(ValueError):
        ArtifactSettings(threshold_uv=0.0).validate()


def test_gate_display_endpoint_maps_to_exact_pi():
    spec = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        wiener_mode=WienerMode.PHASEGATED,
        phase_gate_threshold_rad=3.14,
    )
    spec.validate()
    assert spec.effective_phase_gate_rad == math.pi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandpass_low_hz": 40.0, "bandpass_high_hz": 30.0},
        {"target_sfreq": 80.0, "bandpass_high_hz": 40.0},
        {"analysis_window_sec": 3.9},
        {"coherence_threshold": 1.01},
        {"phase_gate_threshold_rad": 3.15},
    ],
)
def test_processing_spec_rejects_invalid_boundaries(kwargs):
    with pytest.raises(ValueError):
        ProcessingSpec(**kwargs).validate()


def test_fingerprint_changes_with_effective_processing_parameter():
    base = ProcessingSpec()
    changed = ProcessingSpec(bandpass_high_hz=35.0)
    assert base.fingerprint != changed.fingerprint
    assert len(base.fingerprint) == 64


def test_pipeline_fingerprint_includes_extraction_settings():
    processing = ProcessingSpec()
    fixed = ExtractionSpec(mode=ExtractionMode.FIXED_WINDOWS, window_sec=20.0)
    changed = ExtractionSpec(mode=ExtractionMode.FIXED_WINDOWS, window_sec=10.0)
    assert pipeline_fingerprint(processing, fixed) != pipeline_fingerprint(processing, changed)


def test_frequency_fingerprint_ignores_disabled_gate():
    first = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        wiener_mode=WienerMode.FREQUENCY,
        phase_gate_threshold_rad=0.10,
    )
    second = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        wiener_mode=WienerMode.FREQUENCY,
        phase_gate_threshold_rad=2.50,
    )
    assert first.fingerprint == second.fingerprint


def test_specs_normalize_string_backed_enum_values():
    processing = ProcessingSpec(method="basic", wiener_mode="frequency")
    extraction = ExtractionSpec(mode="continuous")

    assert processing.method is ProcessingMethod.BASIC
    assert processing.wiener_mode is WienerMode.FREQUENCY
    assert extraction.mode is ExtractionMode.CONTINUOUS
    assert processing.as_serializable_dict()["method"] == "basic"
    assert extraction.as_serializable_dict()["mode"] == "continuous"


def test_ecmad_channel_groups_are_normalized_validated_and_fingerprinted():
    first = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        channel_groups=[["FP1", "FP2"], ["F7", "T3", "T5"]],
    )
    second = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        channel_groups=[["FP1", "Fz"]],
    )

    first.validate()
    assert first.channel_groups == (("FP1", "FP2"), ("F7", "T3", "T5"))
    assert first.as_serializable_dict()["channel_groups"] == [
        ["FP1", "FP2"],
        ["F7", "T3", "T5"],
    ]
    assert first.fingerprint != second.fingerprint

    with pytest.raises(ValueError, match="至少需要一个导联组"):
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            channel_groups=[],
        ).validate()
    with pytest.raises(ValueError, match="G1"):
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            channel_groups=[["FP1"]],
        ).validate()
