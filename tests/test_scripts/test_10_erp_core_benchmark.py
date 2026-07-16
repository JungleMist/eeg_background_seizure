"""Unit tests for ERP-CORE Flankers benchmark helpers."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path("scripts/10_benchmark_erp_core_flankers.py")
EXPECTED_CHANNEL_GROUPS = [
    ["FCz", "Cz", "Fz", "FC3", "FC4"],
]
EXPECTED_PASSTHROUGH = [
    "FP1", "F3", "F7", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3",
    "O1", "Oz", "Pz", "CPz", "FP2", "F4", "F8", "C4", "C6", "P4",
    "P8", "P10", "PO8", "PO4", "O2",
]


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
    assert len(flattened) == 30
    assert len(set(flattened)) == 30


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

    _, lrp = module.compute_lrp(epochs)

    np.testing.assert_allclose(lrp, [2.5, 2.5])
