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


def test_processing_honours_cancellation_before_loading(synthetic_fif):
    with pytest.raises(ProcessingCancelled):
        ProcessingEngine().process(
            synthetic_fif,
            ProcessingSpec(method=ProcessingMethod.BASIC, analysis_window_sec=4.0),
            _continuous(),
            cancel_requested=lambda: True,
        )
