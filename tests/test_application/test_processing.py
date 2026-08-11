import numpy as np
import pytest

from eeg_bg.application.models import (
    ExtractionMode,
    ExtractionSpec,
    ProcessingMethod,
    ProcessingSpec,
    WienerMode,
)
from eeg_bg.application.processing import ProcessingEngine
from eeg_bg.exceptions import ProcessingCancelled
from eeg_bg.preprocessing.continuous import wiener_continuous_raw
from eeg_bg.decomposition.wiener import decompose_epoch


def _continuous():
    return ExtractionSpec(
        mode=ExtractionMode.CONTINUOUS,
        window_sec=4.0,
    )


def test_basic_selection_uses_shared_preprocessing(synthetic_fif):
    engine = ProcessingEngine()
    result = engine.process(
        synthetic_fif,
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(
            mode=ExtractionMode.SELECTION,
            start_sec=1.0,
            stop_sec=5.0,
            window_sec=4.0,
        ),
    )
    assert result.preview_raw.n_times == 500
    assert result.processed_segments[0].start_sec == 1.0


def test_fixed_windows_are_export_ready_segments(synthetic_fif):
    result = ProcessingEngine().process(
        synthetic_fif,
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(
            mode=ExtractionMode.FIXED_WINDOWS,
            window_sec=4.0,
        ),
    )
    assert [item.window_index for item in result.processed_segments] == [0, 1]
    assert all(item.raw.n_times == 500 for item in result.processed_segments)
    assert result.diagnostics["processed_windows"] == 2
    assert result.diagnostics["incomplete_tail_samples"] == 0


@pytest.mark.parametrize(
    "method",
    [ProcessingMethod.BASIC, ProcessingMethod.ICA, ProcessingMethod.WIENER],
)
def test_extreme_amplitude_fixed_windows_are_still_processed(
    synthetic_fif, tmp_path, method
):
    import mne

    raw = mne.io.read_raw_fif(synthetic_fif, preload=True, verbose=False)
    raw._data[0, 100:120] = 1200e-6
    source = tmp_path / f"extreme-{method.value}-raw.fif"
    raw.save(source, overwrite=True, verbose=False)
    spec = ProcessingSpec(
        method=method,
        analysis_window_sec=4.0,
        ica_n_components=4,
        coherence_threshold=0.0,
    )
    result = ProcessingEngine().process(
        source,
        spec,
        ExtractionSpec(mode=ExtractionMode.FIXED_WINDOWS, window_sec=4.0),
    )
    assert [segment.window_index for segment in result.processed_segments] == [0, 1]
    assert result.diagnostics["processed_windows"] == 2


@pytest.mark.parametrize("mode", list(WienerMode))
def test_continuous_wiener_modes_preserve_length(synthetic_fif, mode):
    result = ProcessingEngine().process(
        synthetic_fif,
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            wiener_mode=mode,
            analysis_window_sec=4.0,
            coherence_threshold=0.0,
        ),
        _continuous(),
    )
    assert result.preview_raw.n_times == 1000
    assert result.diagnostics["mode"] == mode.value
    assert np.isfinite(result.preview_raw.get_data()).all()


def test_ica_reports_removed_components_and_preserves_length(synthetic_fif):
    result = ProcessingEngine().process(
        synthetic_fif,
        ProcessingSpec(
            method=ProcessingMethod.ICA,
            analysis_window_sec=4.0,
            ica_n_components=4,
        ),
        _continuous(),
    )
    assert result.preview_raw.n_times == 1000
    assert "removed_components" in result.diagnostics


def test_selection_ica_fits_once_on_full_source(synthetic_fif):
    result = ProcessingEngine().process(
        synthetic_fif,
        ProcessingSpec(
            method=ProcessingMethod.ICA,
            analysis_window_sec=4.0,
            ica_n_components=4,
        ),
        ExtractionSpec(
            mode=ExtractionMode.SELECTION,
            start_sec=0.0,
            stop_sec=4.0,
            window_sec=4.0,
        ),
    )
    assert result.preview_raw.n_times == 500
    assert result.diagnostics["fit_windows"] == 2


def test_short_basic_selection_is_allowed(synthetic_fif):
    result = ProcessingEngine().process(
        synthetic_fif,
        ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
        ExtractionSpec(
            mode=ExtractionMode.SELECTION,
            start_sec=1.0,
            stop_sec=2.0,
            window_sec=4.0,
        ),
    )
    assert result.preview_raw.n_times == 125


def test_custom_ecmad_channel_groups_override_default_config():
    engine = ProcessingEngine()
    spec = ProcessingSpec(
        method=ProcessingMethod.WIENER,
        channel_groups=(("FP1", "FP2", "Fz"), ("C3", "C4")),
    )

    cfg = engine.build_config(spec, n_channels=19)

    assert cfg["channels"]["channel_groups"] == [
        ["FP1", "FP2", "Fz"],
        ["C3", "C4"],
    ]


def test_processing_config_propagates_or_disables_protected_band():
    engine = ProcessingEngine()

    enabled = engine.build_config(
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            protected_band_hz=(6.0, 18.0),
        ),
        n_channels=19,
    )
    disabled = engine.build_config(
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            protected_band_hz=None,
        ),
        n_channels=19,
    )

    assert enabled["wiener"]["protected_band_hz"] == [6.0, 18.0]
    assert disabled["wiener"]["protected_band_hz"] is None


def test_processing_config_propagates_coherent_gate():
    cfg = ProcessingEngine().build_config(
        ProcessingSpec(
            method=ProcessingMethod.WIENER,
            coherent_gate_enabled=False,
            coherent_gate_threshold_uv=250.0,
        ),
        n_channels=19,
    )

    assert cfg["wiener"]["coherent_gate_enabled"] is False
    assert cfg["wiener"]["coherent_gate_threshold_uv"] == 250.0


def test_continuous_wiener_preserves_protected_tone():
    import mne

    sfreq = 125.0
    n_times = 1000
    times = np.arange(n_times) / sfreq
    protected = np.sin(2 * np.pi * 10.0 * times)
    removable = 0.5 * np.sin(2 * np.pi * 30.0 * times)
    signal = protected + removable
    raw = mne.io.RawArray(
        np.vstack([signal, signal]) * 1e-6,
        mne.create_info(["A", "B"], sfreq, ch_types="eeg"),
        verbose=False,
    )
    cfg = {
        "preprocessing": {
            "target_sfreq": sfreq,
            "epoch_length_sec": 4.0,
        },
        "wiener": {
            "mode": "frequency",
            "nperseg": 500,
            "coherence_threshold": 0.0,
            "coherent_gate_enabled": False,
            "coherent_gate_threshold_uv": 100.0,
            "filter_magnitude_threshold": 1e6,
            "overlap_policy": "coherence_weighted",
            "phase_gate_threshold_rad": np.pi,
            "freq_band": [0.5, 40.0],
            "protected_band_hz": [5.0, 20.0],
        },
        "channels": {"channel_groups": [["A", "B"]]},
    }

    denoised, _ = wiener_continuous_raw(raw, cfg)
    coherent = raw.get_data()[0] - denoised.get_data()[0]
    frequencies = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    coherent_fft = np.fft.rfft(coherent)
    protected_idx = int(np.argmin(np.abs(frequencies - 10.0)))
    removable_idx = int(np.argmin(np.abs(frequencies - 30.0)))

    assert abs(coherent_fft[protected_idx]) < 1e-12
    assert abs(coherent_fft[removable_idx]) > 2e-4


def test_continuous_wiener_gate_uses_microvolt_units_like_epoch_api():
    import mne

    sfreq = 125.0
    n_times = 1000
    times = np.arange(n_times) / sfreq
    shared_uv = 160.0 * np.sin(2 * np.pi * 30.0 * times)
    raw = mne.io.RawArray(
        np.vstack([shared_uv, shared_uv]) * 1e-6,
        mne.create_info(["A", "B"], sfreq, ch_types="eeg"),
        verbose=False,
    )
    cfg = {
        "preprocessing": {
            "target_sfreq": sfreq,
            "epoch_length_sec": n_times / sfreq,
        },
        "wiener": {
            "mode": "frequency",
            "nperseg": 500,
            "coherence_threshold": 0.0,
            "coherent_gate_enabled": True,
            "coherent_gate_threshold_uv": 100.0,
            "filter_magnitude_threshold": 1e6,
            "overlap_policy": "coherence_weighted",
            "phase_gate_threshold_rad": np.pi,
            "freq_band": [0.5, 40.0],
            "protected_band_hz": [5.0, 20.0],
        },
        "channels": {"channel_groups": [["A", "B"]]},
    }

    epoch_result = decompose_epoch(
        raw.get_data() * 1e6, ["A", "B"], cfg
    )
    _, diagnostics = wiener_continuous_raw(raw, cfg)
    central_window = diagnostics["window_diagnostics"][1]

    assert central_window["group_coherent_gate_open"] == [True]
    np.testing.assert_allclose(
        central_window["group_max_bin_rms_uv"],
        epoch_result.group_max_bin_rms_uv,
        rtol=1e-12,
        atol=1e-12,
    )
    expected_closed = sum(
        not window["group_coherent_gate_open"][0]
        for window in diagnostics["window_diagnostics"]
    )
    assert diagnostics["coherent_gate_closed_group_windows"] == expected_closed


def test_processing_honours_cancellation_before_loading(synthetic_fif):
    with pytest.raises(ProcessingCancelled):
        ProcessingEngine().process(
            synthetic_fif,
            ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
            _continuous(),
            cancel_requested=lambda: True,
        )
