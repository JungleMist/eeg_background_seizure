"""Unit tests for ERP-CORE Flankers benchmark helpers."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path("scripts/10_benchmark_erp_core_flankers.py")
EXPECTED_CHANNEL_GROUPS = [
    ["FP1", "F3", "F7", "FCz", "Fz", "FP2", "F4", "F8"],
    ["F3", "C3", "C5", "P3", "FC3"],
    ["F4", "C4", "C6", "P4", "FC4"],
    ["P7", "P8", "PO3", "PO4", "PO7", "PO8", "O1", "O2"],
    ["Cz", "CPz", "Oz", "Pz"],
]
EXPECTED_PASSTHROUGH = ["P9", "P10"]


def _load_script():
    spec = importlib.util.spec_from_file_location("script10_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_erp_core_channel_group_matches_requested_ern_focused_partition():
    module = _load_script()
    cfg = module.load_config("configs/erp_core_flankers.yaml")

    assert cfg["channels"]["channel_groups"] == EXPECTED_CHANNEL_GROUPS
    assert cfg["channels"]["passthrough"] == EXPECTED_PASSTHROUGH
    flattened = [
        channel for group in EXPECTED_CHANNEL_GROUPS for channel in group
    ] + EXPECTED_PASSTHROUGH
    assert len(set(flattened)) == 30
    assert "C3" in set(flattened[:-len(EXPECTED_PASSTHROUGH)])
    assert "C4" in set(flattened[:-len(EXPECTED_PASSTHROUGH)])
    assert "C3" not in EXPECTED_PASSTHROUGH
    assert "C4" not in EXPECTED_PASSTHROUGH
    assert Path(cfg["erp_core"]["data_dir"]).expanduser() == Path.home() / "Data/ERP_CORE"
    assert cfg["erp_core"]["line_freq"] == 60.0


def test_resolve_recordings_groups_available_tasks_by_subject(tmp_path):
    module = _load_script()
    paths = {}
    for subject, tasks in {
        "sub-010": ("ERN",),
        "sub-005": ("LRP",),
        "sub-008": ("ERN", "LRP"),
    }.items():
        eeg_dir = tmp_path / subject / "eeg"
        eeg_dir.mkdir(parents=True)
        for task in tasks:
            recording = eeg_dir / f"{subject}_task-{task}_eeg.set"
            recording.touch()
            paths[subject, task.lower()] = recording
    (tmp_path / "sub-005" / "eeg" / "sub-005_task-LRP_eeg.fdt").touch()

    assert module._resolve_recordings(tmp_path, None) == [
        {
            "subject_id": "sub-005",
            "lrp": paths["sub-005", "lrp"],
        },
        {
            "subject_id": "sub-008",
            "ern": paths["sub-008", "ern"],
            "lrp": paths["sub-008", "lrp"],
        },
        {
            "subject_id": "sub-010",
            "ern": paths["sub-010", "ern"],
        },
    ]


def test_select_recordings_limits_subject_and_task():
    module = _load_script()
    recordings = [
        {
            "subject_id": "sub-001",
            "ern": Path("sub-001_task-ERN_eeg.set"),
            "lrp": Path("sub-001_task-LRP_eeg.set"),
        },
        {
            "subject_id": "sub-002",
            "ern": Path("sub-002_task-ERN_eeg.set"),
            "lrp": Path("sub-002_task-LRP_eeg.set"),
        },
    ]

    assert module._select_recordings(recordings, ["sub-002"], "ern") == [
        {
            "subject_id": "sub-002",
            "ern": Path("sub-002_task-ERN_eeg.set"),
        }
    ]


def test_run_subject_only_writes_visualizations_for_available_task(tmp_path, monkeypatch):
    module = _load_script()
    calls = []
    table = pd.DataFrame(
        {
            "sample": [1, 2],
            "correct": [True, False],
            "response_side": ["left", "right"],
        }
    )

    def fake_prepare(_mne, recording, _cfg, _subject_id):
        return {
            "recording": recording,
            "raws": {method: object() for method in module.METHODS},
            "table": table,
            "ica_excluded_components": [],
            "wiener": {
                "windows": 1,
                "processed_channel_windows": 1,
                "group_processing_rates": {},
            },
        }

    def fake_metrics(_raws, ern_epochs, lrp_epochs, _cfg):
        metrics = pd.DataFrame(
            {
                "method": module.METHODS,
                "target_change_rms_uv": [0.0, 1.0, 1.0],
            }
        )
        ern = {method: object() for method in module.METHODS} if ern_epochs else None
        lrp = {method: object() for method in module.METHODS} if lrp_epochs else None
        return metrics, ern, lrp

    monkeypatch.setattr(module, "_prepare_recording", fake_prepare)
    monkeypatch.setattr(
        module,
        "_make_shared_epochs",
        lambda *_args: {method: [object()] for method in module.METHODS},
    )
    monkeypatch.setattr(module, "_compute_metrics", fake_metrics)
    monkeypatch.setattr(module, "_task_epoch_issue", lambda _task, _epochs: None)
    monkeypatch.setattr(
        module,
        "_lrp_compatibility_waveforms",
        lambda _epochs: {
            compatibility: {method: object() for method in module.METHODS}
            for compatibility in ("compatible", "incompatible")
        },
    )
    for name in (
        "_plot_ern",
        "_plot_topomaps",
        "_plot_lrp",
        "_plot_lrp_by_compatibility",
        "_plot_segment",
        "_plot_time_frequency",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *_args, plot_name=name, **kwargs: calls.append(
                (plot_name, kwargs)
            ),
        )

    cfg = {
        "preprocessing": {"artifact_threshold_uv": 200.0},
        "erp_core": {"ern": {}, "lrp": {}},
        "wiener": {
            "mode": "frequency",
            "coherence_threshold": 0.15,
            "phase_gate_threshold_rad": np.pi,
        },
    }
    fake_mne = type("FakeMNE", (), {"__version__": "test"})()
    ern_path = tmp_path / "sub-005_task-ERN_eeg.set"
    module._run_subject(
        fake_mne,
        {"subject_id": "sub-005", "ern": ern_path},
        cfg,
        tmp_path / "ern-only",
    )

    called = dict(calls)
    assert "_plot_ern" in called
    assert called["_plot_ern"]["show_trial_variance"] is True
    assert "_plot_topomaps" in called
    assert "_plot_lrp" not in called
    assert "_plot_lrp_by_compatibility" not in called

    calls.clear()
    lrp_path = tmp_path / "sub-006_task-LRP_eeg.set"
    module._run_subject(
        fake_mne,
        {"subject_id": "sub-006", "lrp": lrp_path},
        cfg,
        tmp_path / "lrp-only",
    )

    called = dict(calls)
    assert "_plot_ern" not in called
    assert "_plot_topomaps" not in called
    assert called["_plot_lrp"]["show_trial_variance"] is True
    assert called["_plot_lrp_by_compatibility"]["show_trial_variance"] is True


def test_mean_lrp_compatibility_waveforms_is_participant_equal():
    module = _load_script()
    times = np.array([-0.1, 0.0])
    results = []
    for value in (1.0, 3.0):
        results.append(
            {
                "lrp_by_compatibility": {
                    compatibility: {
                        method: (times, np.full(2, value))
                        for method in module.METHODS
                    }
                    for compatibility in ("compatible", "incompatible")
                }
            }
        )

    averaged = module._mean_lrp_compatibility_waveforms(results)

    for compatibility in ("compatible", "incompatible"):
        for method in module.METHODS:
            np.testing.assert_array_equal(averaged[compatibility][method][0], times)
            np.testing.assert_allclose(averaged[compatibility][method][1], [2.0, 2.0])


def test_task_epoch_issue_detects_missing_required_trials():
    module = _load_script()

    def epochs(metadata):
        return {"raw": type("FakeEpochs", (), {"metadata": metadata})()}

    ern_metadata = pd.DataFrame(
        {
            "correct": [True, True],
            "response_side": ["left", "right"],
            "compatibility": ["compatible", "incompatible"],
        }
    )
    assert "correct/incorrect" in module._task_epoch_issue(
        "ern", epochs(ern_metadata)
    )

    lrp_metadata = pd.DataFrame(
        {
            "correct": [True, True, True],
            "response_side": ["left", "right", "left"],
            "compatibility": ["compatible", "compatible", "incompatible"],
        }
    )
    assert "incompatible" in module._task_epoch_issue(
        "lrp", epochs(lrp_metadata)
    )


def test_erp_core_phase_grid_matches_existing_eight_cell_design():
    module = _load_script()
    expected = [
        ("frequency", 0.15, np.pi),
        ("frequency", 0.45, np.pi),
        ("frequency", 0.75, np.pi),
        ("phasegated", 0.15, np.pi / 2),
        ("phasegated", 0.15, np.pi / 5),
        ("phasegated", 0.15, np.pi / 10),
        ("phasegated", 0.45, np.pi / 10),
        ("phasegated", 0.75, np.pi / 10),
    ]

    actual = []
    for index in range(1, 9):
        cfg = module.load_config(f"configs/exp_erp_core_wiener_phase_{index}.yaml")
        actual.append(
            (
                cfg["wiener"]["mode"],
                cfg["wiener"]["coherence_threshold"],
                cfg["wiener"]["phase_gate_threshold_rad"],
            )
        )
        assert cfg["paths"]["results_dir"].endswith(f"phase_grid/exp{index}")

    for observed, wanted in zip(actual, expected):
        assert observed[:2] == wanted[:2]
        assert np.isclose(observed[2], wanted[2])


def test_phasegated_coh099_phase0314_experiment_config():
    module = _load_script()
    cfg = module.load_config(
        "configs/exp_erp_core_phasegated_coh099_phase0314.yaml"
    )

    assert cfg["wiener"]["mode"] == "phasegated"
    assert cfg["wiener"]["coherence_threshold"] == 0.99
    assert cfg["wiener"]["phase_gate_threshold_rad"] == 0.314
    assert cfg["paths"]["results_dir"].endswith(
        "erp_core_flankers/phasegated_coh099_phase0314"
    )
    assert "C3" not in cfg["channels"]["passthrough"]
    assert "C4" not in cfg["channels"]["passthrough"]


def test_wiener_mode_selects_all_supported_implementations():
    module = _load_script()

    assert module._select_wiener_decomposer("frequency") is module.decompose_epoch_frequency
    assert module._select_wiener_decomposer("phasegated") is module.decompose_epoch_phasegated
    assert module._select_wiener_decomposer("zerophase") is module.decompose_epoch_zerophase


def test_zerophase_coh095_experiment_config():
    module = _load_script()
    cfg = module.load_config("configs/exp_erp_core_zerophase_coh095.yaml")

    assert cfg["wiener"]["mode"] == "zerophase"
    assert cfg["wiener"]["coherence_threshold"] == 0.95
    assert cfg["wiener"]["phase_gate_threshold_rad"] == 0.392
    assert cfg["paths"]["results_dir"].endswith("erp_core_flankers/zerophase_coh095")


def test_phasegated_coh095_phase0_experiment_config():
    module = _load_script()
    cfg = module.load_config("configs/exp_erp_core_phasegated_coh095_phase0.yaml")

    assert cfg["wiener"]["mode"] == "phasegated"
    assert cfg["wiener"]["coherence_threshold"] == 0.95
    assert cfg["wiener"]["phase_gate_threshold_rad"] == 0.0
    assert cfg["paths"]["results_dir"].endswith(
        "erp_core_flankers/phasegated_coh095_phase0"
    )


def test_phasegated_coh095_phase_sweep_configs():
    module = _load_script()
    cases = [
        ("005", 0.05),
        ("010", 0.1),
        ("020", 0.2),
        ("030", 0.3),
    ]

    for suffix, phase in cases:
        cfg = module.load_config(
            f"configs/exp_erp_core_phasegated_coh095_phase{suffix}.yaml"
        )
        assert cfg["wiener"]["mode"] == "phasegated"
        assert cfg["wiener"]["coherence_threshold"] == 0.95
        assert cfg["wiener"]["phase_gate_threshold_rad"] == phase
        assert cfg["paths"]["results_dir"].endswith(
            f"erp_core_flankers/phasegated_coh095_phase{suffix}"
        )


def test_build_response_table_for_mne_semantic_annotations():
    module = _load_script()
    event_id = {
        "stimulus/compatible/target_left": 1,
        "stimulus/incompatible/target_right": 2,
        "response/left": 3,
        "response/right": 4,
    }
    events = np.array([[100, 0, 1], [150, 0, 3], [300, 0, 2], [360, 0, 3]])

    table = module.build_response_table(events, event_id, sfreq=100.0)

    assert table["correct"].tolist() == [True, False]
    assert table["compatibility"].tolist() == ["compatible", "incompatible"]
    assert table["reaction_time_sec"].tolist() == [0.5, 0.6]


def test_build_response_table_for_numeric_erp_core_codes():
    module = _load_script()
    event_id = {"stimulus/11": 1, "response/111": 2, "stimulus/22": 3, "response/221": 4}
    events = np.array([[100, 0, 1], [140, 0, 2], [200, 0, 3], [250, 0, 4]])

    table = module.build_response_table(events, event_id, sfreq=100.0)

    assert table["response_side"].tolist() == ["left", "right"]
    assert table["correct"].tolist() == [True, False]


def test_build_response_table_for_bare_numeric_eeglab_annotations():
    module = _load_script()
    event_id = {"11": 1, "111": 2, "22": 3, "221": 4}
    events = np.array([[100, 0, 1], [140, 0, 2], [200, 0, 3], [250, 0, 4]])

    table = module.build_response_table(events, event_id, sfreq=100.0)

    assert table["response_side"].tolist() == ["left", "right"]
    assert table["correct"].tolist() == [True, False]


class _FakeEpochs:
    def __init__(self, data, sides, correct, times):
        self._data = np.asarray(data)
        self.metadata = pd.DataFrame({"response_side": sides, "correct": correct})
        self.times = np.asarray(times)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, selection):
        selection = np.asarray(selection)
        return _FakeEpochs(
            self._data[selection],
            self.metadata.loc[selection, "response_side"].tolist(),
            self.metadata.loc[selection, "correct"].tolist(),
            self.times,
        )

    def get_data(self, picks=None, copy=False):
        assert picks == ["C3", "C4"]
        return self._data


def test_compute_lrp_is_ipsilateral_minus_contralateral():
    module = _load_script()
    # Left response: C3 ipsilateral=3, C4 contralateral=1 -> +2.
    # Right response: C4 ipsilateral=4, C3 contralateral=1 -> +3.
    epochs = _FakeEpochs(
        data=[[[3.0, 3.0], [1.0, 1.0]], [[1.0, 1.0], [4.0, 4.0]]],
        sides=["left", "right"],
        correct=[True, True],
        times=[-0.1, 0.0],
    )

    _, lrp, trial_sd = module.compute_lrp(epochs)

    np.testing.assert_allclose(lrp, [2.5, 2.5])
    np.testing.assert_allclose(trial_sd, [0.5, 0.5])


class _FakeErnEpochs:
    def __init__(self, data, correct, times):
        self._data = np.asarray(data)
        self.metadata = pd.DataFrame({"correct": correct})
        self.times = np.asarray(times)

    def __getitem__(self, selection):
        selection = np.asarray(selection)
        return _FakeErnEpochs(
            self._data[selection],
            self.metadata.loc[selection, "correct"].tolist(),
            self.times,
        )

    def get_data(self, picks=None, copy=False):
        assert picks == ["FCz"]
        return self._data


def test_ern_waveforms_include_trial_standard_deviation_bands():
    module = _load_script()
    epochs = _FakeErnEpochs(
        data=[[[1.0, 1.0]], [[3.0, 3.0]], [[0.0, 0.0]], [[2.0, 2.0]]],
        correct=[False, False, True, True],
        times=[-0.1, 0.0],
    )

    _, incorrect, difference, incorrect_sd, difference_sd = module._ern_waveforms(
        epochs
    )

    np.testing.assert_allclose(incorrect, [2.0, 2.0])
    np.testing.assert_allclose(difference, [1.0, 1.0])
    np.testing.assert_allclose(incorrect_sd, [1.0, 1.0])
    np.testing.assert_allclose(difference_sd, np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Per-group Wiener processing-rate diagnostics
# ---------------------------------------------------------------------------


def _make_result_like(
    candidate_keys=None,
    candidate_status=None,
    channel_sources=None,
):
    """Build a minimal object exposing the fields _active_groups_from_result reads."""
    return type(
        "WienerResultLike",
        (),
        {
            "candidate_keys": candidate_keys,
            "candidate_status": candidate_status,
            "channel_sources": channel_sources or {},
        },
    )()


def test_group_key_matches_wiener_pair_key_construction():
    module = _load_script()

    assert module._group_key(["FCz", "Cz", "Fz", "FC3", "FC4"]) == "FCz-Cz-Fz-FC3-FC4"
    assert module._group_key(["FP1", "FP2"]) == "FP1-FP2"


def test_active_groups_from_result_uses_candidate_status():
    module = _load_script()
    group_keys = ["FCz-Cz-Fz-FC3-FC4", "FC3-C3-C5-P3", "FC4-C4-C6-P4"]
    # Group 1: one candidate processed, one skipped (below coherence).
    # Group 2: both candidates skipped.
    # Group 3: both candidates processed.
    result = _make_result_like(
        candidate_keys=[
            "FCz-Cz-Fz-FC3-FC4::FCz",
            "FCz-Cz-Fz-FC3-FC4::Cz",
            "FC3-C3-C5-P3::FC3",
            "FC3-C3-C5-P3::C3",
            "FC4-C4-C6-P4::FC4",
            "FC4-C4-C6-P4::C4",
        ],
        candidate_status=np.array(
            [
                module.CANDIDATE_PROCESSED,
                module.CANDIDATE_BELOW_COHERENCE,
                module.CANDIDATE_BELOW_COHERENCE,
                module.CANDIDATE_SOLVE_FAILED,
                module.CANDIDATE_PROCESSED,
                module.CANDIDATE_PROCESSED,
            ],
            dtype=np.uint8,
        ),
    )

    active = module._active_groups_from_result(result, group_keys)

    assert active == {"FCz-Cz-Fz-FC3-FC4", "FC4-C4-C6-P4"}


def test_active_groups_from_result_falls_back_to_channel_sources():
    module = _load_script()
    # Diagnostics absent (legacy path) → derive group set from channel_sources,
    # which records the pair_key of each accepted candidate as sources.
    result = _make_result_like(
        candidate_keys=None,
        candidate_status=None,
        channel_sources={
            "FCz": ["FCz-Cz-Fz-FC3-FC4"],
            "C3": [],
            "C4": ["FC4-C4-C6-P4", "FCz-Cz-Fz-FC3-FC4"],
        },
    )

    active = module._active_groups_from_result(result, ["FCz-Cz-Fz-FC3-FC4", "FC4-C4-C6-P4"])

    assert active == {"FCz-Cz-Fz-FC3-FC4", "FC4-C4-C6-P4"}


def test_merge_group_processing_rates_sums_active_and_total_windows():
    module = _load_script()
    rates = [
        {
            "FCz-Cz-Fz-FC3-FC4": {"active_windows": 178, "total_windows": 178},
            "FC3-C3-C5-P3": {"active_windows": 100, "total_windows": 178},
            "FC4-C4-C6-P4": {"active_windows": 0, "total_windows": 178},
        },
        {
            "FCz-Cz-Fz-FC3-FC4": {"active_windows": 160, "total_windows": 160},
            "FC3-C3-C5-P3": {"active_windows": 80, "total_windows": 160},
        },
    ]

    merged = module._merge_group_processing_rates(rates)

    assert merged["FCz-Cz-Fz-FC3-FC4"] == {
        "active_windows": 338,
        "total_windows": 338,
        "rate": 1.0,
    }
    assert merged["FC3-C3-C5-P3"]["active_windows"] == 180
    assert merged["FC3-C3-C5-P3"]["total_windows"] == 338
    assert np.isclose(merged["FC3-C3-C5-P3"]["rate"], 180 / 338)
    assert merged["FC4-C4-C6-P4"]["active_windows"] == 0
    assert merged["FC4-C4-C6-P4"]["rate"] == 0.0


def test_merge_group_processing_rates_handles_empty_input():
    module = _load_script()

    assert module._merge_group_processing_rates([]) == {}
    assert module._merge_group_processing_rates([{}, {}]) == {}
