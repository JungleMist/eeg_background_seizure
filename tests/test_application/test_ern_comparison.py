from pathlib import Path

import mne
import numpy as np

from eeg_bg.application.ern_comparison import (
    ErnComparisonService,
    build_response_trials,
    summarize_ern_epochs,
)
from eeg_bg.application.models import ProcessingSpec, WienerMode


def test_build_response_trials_supports_bare_eeglab_codes():
    event_id = {"11": 1, "111": 2, "22": 3, "221": 4}
    events = np.array([[100, 0, 1], [140, 0, 2], [200, 0, 3], [250, 0, 4]])

    trials = build_response_trials(events, event_id, sfreq=100.0)

    assert [trial.response_side for trial in trials] == ["left", "right"]
    assert [trial.correct for trial in trials] == [True, False]
    assert [trial.reaction_time_sec for trial in trials] == [0.4, 0.5]


class _FakeEpochs:
    def __init__(self):
        self.times = np.array([-0.1, 0.0])
        self._data = np.array([[[1e-6, 1e-6]], [[3e-6, 3e-6]], [[0.0, 0.0]], [[2e-6, 2e-6]]])

    def __len__(self):
        return len(self._data)

    def get_data(self, picks=None, copy=False):
        assert picks == ["FCz"]
        return self._data


def test_summarize_ern_epochs_matches_benchmark_mean_and_sd_formula():
    waveform = summarize_ern_epochs(
        _FakeEpochs(), np.array([False, False, True, True])
    )

    np.testing.assert_allclose(waveform.times_ms, [-100.0, 0.0])
    np.testing.assert_allclose(waveform.incorrect_mean_uv, [2.0, 2.0])
    np.testing.assert_allclose(waveform.difference_mean_uv, [1.0, 1.0])
    np.testing.assert_allclose(waveform.incorrect_sd_uv, [1.0, 1.0])
    np.testing.assert_allclose(waveform.difference_sd_uv, np.sqrt(2.0))


def test_ern_config_uses_current_parameters_and_erp_channel_groups():
    service = ErnComparisonService()
    spec = ProcessingSpec(
        bandpass_low_hz=0.1,
        bandpass_high_hz=30.0,
        target_sfreq=125.0,
        analysis_window_sec=12.0,
        ica_n_components=15,
        ica_artifact_corr_threshold=0.72,
        wiener_mode=WienerMode.PHASEGATED,
        coherence_threshold=0.45,
        coherent_gate_enabled=False,
        coherent_gate_threshold_uv=250.0,
        phase_gate_threshold_rad=0.1,
        protected_band_hz=(5.0, 20.0),
    )

    cfg = service._build_config(spec, n_channels=30)

    assert cfg["preprocessing"]["bandpass"] == [0.1, 30.0]
    assert cfg["preprocessing"]["target_sfreq"] == 125.0
    assert cfg["preprocessing"]["epoch_length_sec"] == 12.0
    assert cfg["erp_core"]["standard_ica"]["n_components"] == 15
    assert cfg["wiener"]["mode"] == "phasegated"
    assert cfg["wiener"]["coherence_threshold"] == 0.45
    assert cfg["wiener"]["coherent_gate_enabled"] is False
    assert cfg["wiener"]["coherent_gate_threshold_uv"] == 250.0
    assert cfg["wiener"]["phase_gate_threshold_rad"] == 0.1
    assert cfg["wiener"]["protected_band_hz"] == [5.0, 20.0]
    assert "FCz" in cfg["channels"]["channel_groups"][0]


def test_ern_config_honours_custom_ecmad_channel_groups():
    service = ErnComparisonService()
    spec = ProcessingSpec(
        channel_groups=(("FCz", "Cz", "Fz"), ("FC3", "FC4")),
    )

    cfg = service._build_config(spec, n_channels=30)

    assert cfg["channels"]["channel_groups"] == [
        ["FCz", "Cz", "Fz"],
        ["FC3", "FC4"],
    ]


def test_compare_uses_shared_epochs_for_all_three_branches(monkeypatch, tmp_path):
    sfreq = 100.0
    info = mne.create_info(["FCz", "FP1"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((2, 600)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 1.4, 3.0, 3.5],
            duration=[0.0] * 4,
            description=["11", "111", "22", "221"],
        )
    )

    class FakeRecordingService:
        def load_eeg(self, source, preload=True):
            return raw.copy(), []

    service = ErnComparisonService(recording_service=FakeRecordingService())
    monkeypatch.setattr(service, "_common_preprocess", lambda value, cfg: value.copy())
    monkeypatch.setattr(
        service,
        "_apply_standard_ica",
        lambda value, cfg: (value.copy(), (1,)),
    )
    monkeypatch.setattr(
        "eeg_bg.application.ern_comparison.wiener_continuous_raw",
        lambda value, cfg, subject_id, **kwargs: (
            value.copy(),
            {"windows": 1, "processed_channel_windows": 1},
        ),
    )
    source = tmp_path / "sub-001_task-ERN_eeg.set"
    result = service.compare(
        source,
        ProcessingSpec(
            bandpass_low_hz=1.0,
            bandpass_high_hz=30.0,
            target_sfreq=100.0,
            analysis_window_sec=4.0,
        ),
    )

    assert set(result.waveforms) == {"raw", "ica", "wiener"}
    assert result.n_paired_trials == 2
    assert result.n_epochs == 2
    assert result.n_correct == 1
    assert result.n_incorrect == 1
    assert result.ica_excluded_components == (1,)
