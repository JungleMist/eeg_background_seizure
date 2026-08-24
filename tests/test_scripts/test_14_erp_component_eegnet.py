"""Tests for the two-stage ERP-CORE ERN EEGNet experiment."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from eeg_bg.config.settings import load_config


SCRIPT_PATH = Path("scripts/14_compare_erp_core_ern_components_eegnet.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script14_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dataset(module, n_subjects=10, n_trials=4, n_times=128):
    rng = np.random.default_rng(7)
    subject_ids = np.repeat([f"sub-{i:03d}" for i in range(n_subjects)], n_trials)
    y = np.tile(np.asarray([0, 0, 1, 1], dtype=np.int8), n_subjects)
    samples = np.arange(len(y), dtype=np.int64)
    sequences = {
        condition: rng.standard_normal(
            (len(y), len(module.ERP_ERN_CHANNELS), n_times)
        ).astype(np.float32)
        for condition in module.BASE_CONDITIONS
    }
    return module.SequenceDataset(sequences, y, subject_ids, samples)


def _config():
    return load_config("configs/erp_core_flankers_2.yaml")


def test_conditions_and_input_shapes():
    module = _load_script()
    dataset = _dataset(module, n_subjects=4)

    assert module.CONDITIONS == (
        "raw",
        "step1_specific",
        "step1_coherent",
        "step1_specific_coherent",
        "step2_specific",
        "step2_coherent",
        "ica",
        "ica_wiener_specific",
        "ica_wiener_coherent",
    )
    assert len(module.BASE_CONDITIONS) == 8
    assert dataset.matrix("raw", normalize=False).shape[1:] == (30, 128)
    assert dataset.matrix(module.COMBINED_CONDITION, normalize=False).shape[1:] == (60, 128)


def test_eegnet_accepts_regular_and_combined_channel_inputs():
    module = _load_script()
    import torch

    for n_channels in (30, 60):
        model = module._make_model(
            n_channels,
            128,
            {"F1": 2, "D": 1, "dropout": 0.0},
        )
        with torch.no_grad():
            output = model(torch.randn(2, 1, n_channels, 128))
        assert output.shape == (2, 1)


def test_two_step_and_ica_wiener_configuration_matches_design():
    module = _load_script()
    cfg = _config()

    assert cfg["wiener"]["freq_band"] == [0.5, 45.0]
    assert cfg["erp_core"]["component_eegnet"]["output_dir"].endswith(
        "erp_core_ern_component_eegnet_2step_30ch"
    )
    assert cfg["erp_core"]["component_eegnet"]["cache_subdir"].endswith(
        "erp_core_ern_eegnet_2step_30ch"
    )
    original = deepcopy(cfg)
    step1 = module.build_step_config(cfg, "step1")
    step2 = module.build_step_config(cfg, "step2")
    ica_wiener = module.build_ica_wiener_config(cfg)

    assert step1["wiener"]["phase_gate_threshold_rad"] == 0.05
    assert step1["wiener"]["protected_band_hz"] is None
    assert step1["wiener"]["coherent_gate_enabled"] is False
    assert step2["wiener"]["phase_gate_threshold_rad"] == 0.5
    assert step2["wiener"]["protected_band_hz"] == [1.0, 25.0]
    assert step2["wiener"]["coherent_gate_enabled"] is False
    assert ica_wiener["wiener"]["phase_gate_threshold_rad"] == 1.0
    assert ica_wiener["wiener"]["protected_band_hz"] == [5.0, 45.0]
    assert ica_wiener["wiener"]["coherent_gate_enabled"] is False
    assert ica_wiener["wiener"]["coherent_gate_threshold_uv"] == 100.0
    assert ica_wiener["channels"]["channel_groups"] == cfg["channels"]["channel_groups"]
    assert all(
        local["wiener"]["freq_band"] == [0.5, 45.0]
        for local in (step1, step2, ica_wiener)
    )
    assert cfg == original


def test_continuous_branches_are_serial_conserving_and_ica_reads_raw():
    module = _load_script()
    cfg = _config()
    n_times = 500
    data = np.vstack([
        np.linspace(index, index + 1.0, n_times)
        for index in range(len(module.ERP_ERN_CHANNELS))
    ]) * 1e-6
    raw = mne.io.RawArray(
        data,
        mne.create_info(list(module.ERP_ERN_CHANNELS), 250.0, ch_types="eeg"),
        verbose=False,
    )

    class Helpers:
        def __init__(self):
            self.ica_inputs = []
            self.wiener_inputs = []
            self.wiener_configs = []
            self.identifiers = []

        def _standard_ica(self, source, _cfg):
            self.ica_inputs.append(source.get_data().copy())
            cleaned = source.copy()
            cleaned._data *= 0.8
            return cleaned, [2]

        def _wiener_continuous(self, source, local_cfg, identifier):
            self.wiener_inputs.append(source.get_data().copy())
            self.wiener_configs.append(local_cfg)
            self.identifiers.append(identifier)
            specific = source.copy()
            specific._data[0] *= 0.5
            specific._data[8] += 1e-9
            return specific, {
                "window_diagnostics": [
                    {"channel_sources": {"FP1": [identifier]}}
                ],
                "group_processing_rates": {},
            }

    helpers = Helpers()
    branches, step_diagnostics, ica_diagnostics, excluded = (
        module.build_continuous_branches(raw, cfg, "sub-test", helpers)
    )

    assert tuple(branches) == module.BASE_CONDITIONS
    assert helpers.identifiers == [
        "sub-test_step1", "sub-test_step2", "sub-test_ica"
    ]
    np.testing.assert_allclose(helpers.wiener_inputs[0], raw.get_data())
    np.testing.assert_allclose(
        helpers.wiener_inputs[1], branches["step1_specific"].get_data()
    )
    np.testing.assert_allclose(helpers.ica_inputs[0], raw.get_data())
    np.testing.assert_allclose(helpers.wiener_inputs[2], branches["ica"].get_data())
    for source, specific_name, coherent_name in (
        (raw, "step1_specific", "step1_coherent"),
        (branches["step1_specific"], "step2_specific", "step2_coherent"),
        (branches["ica"], "ica_wiener_specific", "ica_wiener_coherent"),
    ):
        np.testing.assert_allclose(
            source.get_data(),
            branches[specific_name].get_data() + branches[coherent_name].get_data(),
        )
        np.testing.assert_array_equal(
            branches[coherent_name].get_data(picks=["P9"]), 0.0
        )
    assert excluded == [2]
    assert step_diagnostics["step1"]["inactive_channels"] == [
        channel for channel in module.ERP_ERN_CHANNELS if channel != "FP1"
    ]
    assert ica_diagnostics["input_parameters"]["protected_band_hz"] == [5.0, 45.0]


def test_cache_fingerprint_tracks_both_steps_and_ica_branch(tmp_path):
    module = _load_script()
    recording = tmp_path / "sub-test_task-ERN_eeg.set"
    recording.write_bytes(b"set")
    cfg = _config()
    original = module._cache_fingerprint(recording, cfg)

    changed_step = deepcopy(cfg)
    changed_step["erp_core"]["distributed_components"]["steps"]["step2"][
        "phase_gate_threshold_rad"
    ] = 0.6
    changed_ica = deepcopy(cfg)
    changed_ica["erp_core"]["component_eegnet"]["ica_wiener"][
        "coherent_gate_threshold_uv"
    ] = 90.0

    assert module._cache_fingerprint(recording, changed_step) != original
    assert module._cache_fingerprint(recording, changed_ica) != original


def test_cache_roundtrip_stores_only_eight_physical_conditions(tmp_path):
    module = _load_script()
    dataset = _dataset(module, n_subjects=2)
    path = tmp_path / "sub-001" / "sequences.npz"

    module._save_subject_cache(path, dataset, "fingerprint", {"subject_id": "sub-001"})

    with np.load(path, allow_pickle=False) as data:
        stored = {key.removeprefix("X_") for key in data.files if key.startswith("X_")}
    assert stored == set(module.BASE_CONDITIONS)
    assert module.COMBINED_CONDITION not in stored
    cached = module._load_subject_cache(path, "fingerprint")
    assert cached is not None
    np.testing.assert_allclose(
        cached[0].matrix(module.COMBINED_CONDITION, normalize=False),
        dataset.matrix(module.COMBINED_CONDITION, normalize=False),
    )
    assert cached[1] == {"subject_id": "sub-001"}


def test_shared_epoch_selection_requires_identical_labels_and_samples():
    module = _load_script()
    labels = np.asarray([0, 1], dtype=np.int8)
    samples = np.asarray([250, 500], dtype=np.int64)
    epochs = SimpleNamespace(
        metadata=pd.DataFrame({"correct": [True, False]}),
        events=np.asarray([[250, 0, 1], [500, 0, 1]], dtype=np.int64),
    )

    module._validate_shared_epoch_selection("step1_specific", epochs, labels, samples)

    mismatched = SimpleNamespace(
        metadata=epochs.metadata,
        events=np.asarray([[250, 0, 1], [501, 0, 1]], dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="changed shared ERN trial selection"):
        module._validate_shared_epoch_selection(
            "step2_specific", mismatched, labels, samples
        )


def test_two_stage_split_is_deterministic_and_subject_disjoint():
    module = _load_script()
    subjects = [f"sub-{i:03d}" for i in range(10)]

    first = module.split_subjects_two_stage(subjects, 0.2, 0.2, 42)
    second = module.split_subjects_two_stage(subjects, 0.2, 0.2, 42)

    assert first == second
    assert len(first["test"]) == 2
    assert len(first["validation"]) == 2
    assert len(first["train"]) == 6
    assert not set(first["train"]) & set(first["validation"])
    assert not set(first["train"]) & set(first["test"])
    assert not set(first["validation"]) & set(first["test"])
    assert set().union(*map(set, first.values())) == set(subjects)


def test_trial_channel_zscore_is_independent_and_finite():
    module = _load_script()
    values = np.asarray([[[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]]], dtype=np.float32)
    normalized = module.trial_channel_zscore(values)

    np.testing.assert_allclose(normalized[0, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized[0, 0].std(), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[0, 1], 0.0, atol=1e-6)
    assert np.isfinite(normalized).all()


def test_threshold_uses_validation_only_and_prefers_balanced_accuracy():
    module = _load_script()
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])

    threshold = module.select_balanced_accuracy_threshold(y, probabilities)

    assert 0.2 <= threshold <= 0.8
    metrics = module.classification_metrics(y, probabilities, threshold)
    assert metrics["balanced_accuracy"] == 1.0


def test_subject_predictions_aggregate_probability_before_threshold():
    module = _load_script()
    frame = pd.DataFrame(
        {
            "subject_id": ["sub-001", "sub-001", "sub-002", "sub-002"],
            "true_label": [0, 0, 1, 1],
            "pred_proba": [0.2, 0.4, 0.7, 0.9],
        }
    )

    result = module._subject_predictions(frame, 0.5)

    np.testing.assert_allclose(result["pred_proba"].to_numpy(), [0.3, 0.8])
    assert result["predicted_label"].tolist() == [0, 1]
    assert result["n_trials"].tolist() == [2, 2]
