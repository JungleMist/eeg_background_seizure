"""Tests for Script 15's serial distributed ECMAD experiment."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd

from eeg_bg.config.settings import load_config


SCRIPT_PATH = Path("scripts/15_compare_erp_core_ern_distributed_components_eegnet.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script15_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return load_config("configs/erp_core_flankers.yaml")


def test_distributed_step_configuration_matches_design():
    module = _load_script()
    cfg = _config()
    steps = cfg["erp_core"]["distributed_components"]["steps"]

    assert tuple(steps) == ("step1", "step2", "step3")
    assert cfg["erp_core"]["distributed_components"]["cache_subdir"] == (
        "cache/erp_core_ern_distributed_components"
    )
    assert "cache_subdir" not in cfg["erp_core"]["distributed_component_eegnet"]
    assert "cache_subdir" not in cfg["erp_core"]["distributed_component_models"]
    assert steps["step1"]["phase_gate_threshold_rad"] == 1.0
    assert steps["step1"]["protected_band_hz"] is None
    assert steps["step1"]["coherent_gate_enabled"] is True
    assert steps["step1"]["coherent_gate_threshold_uv"] == 100.0
    assert steps["step1"]["channel_groups"] == [
        ["FP2", "F4", "F8", "FC4", "FCz", "C4", "C6", "P4", "Pz"],
        ["FP1", "F3", "F7", "FC3", "Fz", "C3", "C5", "P3", "Cz", "CPz"],
    ]
    assert steps["step2"]["phase_gate_threshold_rad"] == 0.02
    assert steps["step2"]["coherent_gate_enabled"] is False
    assert steps["step2"]["channel_groups"] == [
        ["FP1", "F3", "F7", "FCz", "Fz", "FP2", "F4", "F8"],
        ["F3", "C5", "FC3", "Cz"],
        ["F7", "C3", "P3", "P7", "PO7"],
        ["F4", "C6", "FC4", "CPz"],
        ["F8", "C4", "P4", "P8", "PO8"],
        ["PO3", "PO4", "O1", "O2", "Oz", "Pz"],
    ]
    assert steps["step3"]["phase_gate_threshold_rad"] == 1.0
    assert steps["step3"]["protected_band_hz"] == [2.0, 30.0]
    assert steps["step3"]["coherent_gate_enabled"] is False
    assert steps["step3"]["channel_groups"] == [
        ["FP1", "F3", "F7", "FCz", "Fz", "FP2", "F4", "F8"],
        ["F3", "C3", "C5", "P3", "FC3"],
        ["F4", "C4", "C6", "P4", "FC4"],
        ["P7", "P8", "PO3", "PO4", "PO7", "PO8", "O1", "O2"],
        ["Cz", "CPz", "Oz", "Pz"],
    ]

    original_groups = deepcopy(cfg["channels"]["channel_groups"])
    local = module.build_step_config(cfg, "step3")
    assert local["wiener"]["freq_band"] == cfg["wiener"]["freq_band"]
    assert local["wiener"]["coherence_threshold"] == cfg["wiener"]["coherence_threshold"]
    assert local["channels"]["channel_groups"] == steps["step3"]["channel_groups"]
    assert cfg["channels"]["channel_groups"] == original_groups


def test_serial_cascade_conserves_components_and_zeroes_inactive_channels():
    module = _load_script()
    cfg = _config()
    data = np.vstack([
        np.linspace(-4.0, 4.0, 500),
        np.linspace(3.0, -3.0, 500),
    ]) * 1e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(["FP1", "P9"], 125.0, ch_types="eeg"),
        verbose=False,
    )
    inputs = []

    class Helpers:
        @staticmethod
        def _wiener_continuous(source, local_cfg, subject_id):
            inputs.append(source.get_data().copy())
            specific = source.copy()
            specific._data[0] *= 0.5
            # Deliberately perturb the inactive channel; Script 15 must restore it.
            specific._data[1] += 1e-9
            diagnostics = {
                "mode": local_cfg["wiener"]["mode"],
                "window_diagnostics": [{"channel_sources": {"FP1": [subject_id]}}],
                "group_processing_rates": {},
            }
            return specific, diagnostics

    branches, diagnostics, cumulative_error = module.run_distributed_cascade(
        raw, cfg, "sub-test", Helpers
    )

    np.testing.assert_allclose(inputs[0], data)
    np.testing.assert_allclose(inputs[1], branches["step1_specific"].get_data())
    np.testing.assert_allclose(inputs[2], branches["step2_specific"].get_data())
    for step, step_input in (
        ("step1", raw),
        ("step2", branches["step1_specific"]),
        ("step3", branches["step2_specific"]),
    ):
        np.testing.assert_allclose(
            branches[f"{step}_specific"].get_data()
            + branches[f"{step}_coherent"].get_data(),
            step_input.get_data(),
        )
        np.testing.assert_array_equal(
            branches[f"{step}_coherent"].get_data(picks=["P9"]), 0.0
        )
        assert diagnostics[step]["inactive_channels"] == ["P9"]
    np.testing.assert_allclose(
        raw.get_data(),
        branches["step3_specific"].get_data()
        + branches["step1_coherent"].get_data()
        + branches["step2_coherent"].get_data()
        + branches["step3_coherent"].get_data(),
    )
    assert cumulative_error < 1e-9
    assert not np.array_equal(
        branches["step2_coherent"].get_data(),
        branches["step1_coherent"].get_data() + branches["step2_coherent"].get_data(),
    )


def test_all_conditions_use_30_channels_and_constant_channels_normalize_to_zero():
    module = _load_script()
    base = module._load_script14()
    n_trials, n_times = 8, 126
    rng = np.random.default_rng(15)
    sequences = {
        condition: rng.standard_normal(
            (n_trials, len(module.ERP_CORE_EEG_CHANNELS), n_times)
        ).astype(np.float32)
        for condition in module.CONDITIONS
    }
    sequences["step1_coherent"][:, -2] = 0.0
    dataset = base.SequenceDataset(
        sequences,
        np.tile([0, 1], n_trials // 2).astype(np.int8),
        np.repeat(["sub-001", "sub-002"], n_trials // 2),
        np.arange(n_trials),
    )

    for condition in module.CONDITIONS:
        assert dataset.matrix(condition, normalize=False).shape == (n_trials, 30, n_times)
    np.testing.assert_array_equal(dataset.matrix("step1_coherent")[:, -2], 0.0)


def test_fingerprint_changes_with_any_step_configuration(tmp_path):
    module = _load_script()
    recording = tmp_path / "subject.set"
    recording.write_bytes(b"set")
    cfg = _config()
    original = module._cache_fingerprint(recording, cfg)

    for step_name, key, value in (
        ("step1", "phase_gate_threshold_rad", 0.9),
        ("step2", "coherent_gate_enabled", True),
        ("step3", "protected_band_hz", [3.0, 30.0]),
    ):
        changed = deepcopy(cfg)
        changed["erp_core"]["distributed_components"]["steps"][step_name][key] = value
        assert module._cache_fingerprint(recording, changed) != original


def test_metrics_include_mcc_subject_rows_and_component_deltas():
    module = _load_script()
    base = module._load_script14()
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.9])
    metrics = module.classification_metrics(y, probabilities, 0.5, base)

    assert metrics["mcc"] == 1.0
    frame = pd.DataFrame({
        "subject_id": ["sub-001"] * 4,
        "true_label": y,
        "pred_proba": probabilities,
    })
    rows = module._subject_metric_rows(frame, "raw", "test", 0.5, base)
    assert rows[0]["n_correct"] == 2
    assert rows[0]["n_incorrect"] == 2
    assert rows[0]["mcc"] == 1.0

    condition_rows = []
    values = {"raw": 0.5}
    values.update({condition: 0.6 for condition in module.COMPONENT_NAMES})
    for condition, value in values.items():
        row = {"condition": condition}
        row.update({f"test_{metric}": value for metric in module._METRIC_NAMES})
        condition_rows.append(row)
    deltas = module.component_metric_deltas(pd.DataFrame(condition_rows))
    np.testing.assert_allclose(deltas["specific_minus_raw_auprc"], 0.1)
    np.testing.assert_allclose(deltas["specific_minus_coherent_auprc"], 0.0)


def test_six_continuous_edfs_preserve_shape_annotations_and_zero_channel(tmp_path):
    module = _load_script()
    base = module._load_script14()
    sfreq, n_times = 125.0, 500
    info = mne.create_info(list(module.ERP_CORE_EEG_CHANNELS), sfreq, ch_types="eeg")
    data = np.zeros((len(module.ERP_CORE_EEG_CHANNELS), n_times))
    data[0] = np.sin(np.arange(n_times) / 20.0) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(mne.Annotations([1.0], [0.0], ["response/111"]))
    branches = {component: raw.copy() for component in module.COMPONENT_NAMES}

    module._write_component_edfs(branches, tmp_path, "sub-001")

    paths = module._component_cache_paths(tmp_path, "sub-001")
    assert len(paths) == 6
    assert all(path.is_file() for path in paths.values())
    reread = mne.io.read_raw_edf(paths["step3_coherent"], preload=True, verbose=False)
    assert reread.ch_names == list(module.ERP_CORE_EEG_CHANNELS)
    assert reread.n_times == n_times
    assert reread.info["sfreq"] == sfreq
    assert reread.annotations.description.tolist() == ["response/111"]
    assert np.max(np.abs(reread.get_data(picks=["P9"]))) < 5e-12

    dataset = base.SequenceDataset(
        {
            condition: np.zeros((1, len(module.ERP_CORE_EEG_CHANNELS), 126), np.float32)
            for condition in module.CONDITIONS
        },
        np.asarray([1], dtype=np.int8),
        np.asarray(["sub-001"]),
        np.asarray([125], dtype=np.int64),
    )
    diagnostics = {"subject_id": "sub-001", "cache_fingerprint": "fingerprint"}
    module._save_json(
        {"component_fingerprint": "component"},
        module._component_metadata_path(tmp_path, "sub-001"),
    )
    module._save_subject_cache(
        tmp_path, "sub-001", dataset, "fingerprint", diagnostics
    )
    cached = module._load_subject_cache(
        tmp_path, "sub-001", "fingerprint", base
    )
    assert cached is not None
    assert set(cached[0].sequences) == set(module.CONDITIONS)
    assert cached[1] == diagnostics


def test_shared_component_cache_is_generated_once_and_reused(tmp_path):
    module = _load_script()
    cfg = _config()
    sfreq, n_times = 125.0, 500
    info = mne.create_info(
        list(module.ERP_CORE_EEG_CHANNELS), sfreq, ch_types="eeg"
    )
    data = np.zeros((len(module.ERP_CORE_EEG_CHANNELS), n_times))
    data[0] = np.sin(np.arange(n_times) / 20.0) * 1e-6
    common = mne.io.RawArray(data, info, verbose=False)
    common.set_annotations(mne.Annotations([1.0], [0.0], ["response/111"]))
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"set")

    class Helpers:
        def __init__(self):
            self.calls = 0

        def _wiener_continuous(self, source, _local_cfg, subject_id):
            self.calls += 1
            specific = source.copy()
            specific._data[0] *= 0.5
            return specific, {
                "window_diagnostics": [
                    {"channel_sources": {"FP1": [subject_id]}}
                ],
                "group_processing_rates": {},
            }

    first_helpers = Helpers()
    first, diagnostics, cached = module.load_or_create_distributed_components(
        common, recording, cfg, "sub-001", first_helpers, tmp_path
    )
    assert not cached
    assert first_helpers.calls == 3
    assert set(first) == set(module.CONDITIONS)
    assert diagnostics["component_fingerprint"] == module._component_fingerprint(
        recording, cfg
    )
    assert module._component_metadata_path(tmp_path, "sub-001").is_file()

    second_helpers = Helpers()
    second, second_diagnostics, cached = (
        module.load_or_create_distributed_components(
            common, recording, cfg, "sub-001", second_helpers, tmp_path
        )
    )
    assert cached
    assert second_helpers.calls == 0
    assert second_diagnostics == diagnostics
    for condition in module.COMPONENT_NAMES:
        np.testing.assert_allclose(
            second[condition].get_data(), first[condition].get_data()
        )
