"""Tests for Script 17's continuous TUAB Wiener component cache."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path("scripts/17_cache_tuab_continuous_wiener.py")
CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


def _load_script():
    spec = importlib.util.spec_from_file_location("script17_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
        },
        "dataset": {
            "active": "tuab",
            "tuab": {
                "edf_dir": "edf",
                "reference_scheme": "ar",
                "montage_dir": "01_tcp_ar",
                "train_partition": "train",
                "eval_partition": "eval",
                "validation_fraction": 0.1,
                "max_recording_sec": 1200.0,
                "classes": {
                    "abnormal": {"folder": "abnormal", "label": 0},
                    "normal": {"folder": "normal", "label": 1},
                },
            },
        },
        "split": {"random_seed": 42},
        "preprocessing": {
            "target_sfreq": 1.0,
            "bandpass": [0.1, 0.4],
            "epoch_length_sec": 20.0,
            "artifact_threshold_uv": 200.0,
            "seizure_buffer_sec": 30.0,
        },
        "channels": {
            "standard_19": CHANNELS,
            "channel_groups": [["FP1", "FP2"]],
            "passthrough": CHANNELS[2:],
        },
        "wiener": {
            "mode": "frequency",
            "nperseg": 20,
            "coherence_threshold": 0.15,
            "coherent_gate_enabled": True,
            "coherent_gate_threshold_uv": 100.0,
            "filter_magnitude_threshold": 50.0,
            "overlap_policy": "coherence_weighted",
            "phase_gate_threshold_rad": 0.05,
            "freq_band": [0.1, 0.4],
            "protected_band_hz": None,
        },
    }


def _row(source: Path) -> dict:
    return {
        "dataset_name": "tuab",
        "patient_id": "aaaaa001",
        "session_id": "s001",
        "token_id": "t000",
        "recording_id": "aaaaa001_s001_t000",
        "evaluation_id": "aaaaa001_s001_t000",
        "class_name": "abnormal",
        "label": 0,
        "reference": "ar",
        "source_partition": "train",
        "edf_path": str(source),
        "split": "train",
    }


def _install_processing_fakes(monkeypatch, module, calls: list[str]) -> None:
    data = np.vstack([
        np.full(1201, channel_index + 1.0)
        for channel_index in range(len(CHANNELS))
    ])

    def fake_load_edf(path, cfg):
        calls.append("load")
        return data.copy(), list(CHANNELS), 1.0

    def fake_wiener(raw, cfg, subject_id="recording"):
        calls.append("wiener")
        specific = raw.copy()
        specific._data[0] *= 0.5
        specific._data[1] += 1e-9
        return specific, {
            "mode": cfg["wiener"]["mode"],
            "windows": 121,
            "processed_channel_windows": 121,
            "group_processing_rates": {},
            "solve_failures": 0,
            "below_coherence_candidates": 0,
            "window_diagnostics": [
                {"channel_sources": {"FP1": [subject_id]}}
            ],
        }

    monkeypatch.setattr(module, "load_edf", fake_load_edf)
    monkeypatch.setattr(module, "wiener_continuous_raw", fake_wiener)


def test_processes_first_1200_seconds_and_writes_separate_components(
    tmp_path, monkeypatch,
):
    module = _load_script()
    cfg = _config(tmp_path)
    source = tmp_path / "recording.edf"
    source.write_bytes(b"edf")
    output_root = tmp_path / "continuous"
    calls: list[str] = []
    _install_processing_fakes(monkeypatch, module, calls)

    result = module._process_one(
        (_row(source), cfg, str(output_root), "frequency", False)
    )

    assert result["status"] == "processed"
    assert result["n_times"] == 1200
    assert result["duration_sec"] == 1200.0
    assert calls == ["load", "wiener"]
    paths = module._cache_paths(output_root, _row(source)["evaluation_id"])
    with np.load(paths["specific"], allow_pickle=False) as specific_cache:
        specific = specific_cache["sequence"]
        assert specific.shape == (19, 1200)
        assert specific.dtype == np.float32
        assert specific_cache["ch_names"].tolist() == CHANNELS
        assert float(specific_cache["sfreq"]) == 1.0
        assert str(specific_cache["component"]) == "specific"
        assert str(specific_cache["split"]) == "train"
    with np.load(paths["coherent"], allow_pickle=False) as coherent_cache:
        coherent = coherent_cache["sequence"]
        assert coherent.dtype == np.float32
        assert str(coherent_cache["component"]) == "coherent"

    raw = np.vstack([
        np.full(1200, channel_index + 1.0, dtype=np.float32)
        for channel_index in range(len(CHANNELS))
    ])
    np.testing.assert_allclose(specific + coherent, raw)
    np.testing.assert_array_equal(coherent[1:], 0.0)
    np.testing.assert_allclose(coherent[0], 0.5)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["active_channels"] == ["FP1"]
    assert metadata["inactive_channels"] == CHANNELS[1:]
    assert metadata["diagnostics"]["windows"] == 121
    assert "window_diagnostics" not in metadata["diagnostics"]


def test_cache_hit_partial_recovery_mismatch_and_force(
    tmp_path, monkeypatch,
):
    module = _load_script()
    cfg = _config(tmp_path)
    source = tmp_path / "recording.edf"
    source.write_bytes(b"edf")
    output_root = tmp_path / "continuous"
    row = _row(source)
    calls: list[str] = []
    _install_processing_fakes(monkeypatch, module, calls)

    first = module._process_one((row, cfg, str(output_root), "frequency", False))
    second = module._process_one((row, cfg, str(output_root), "frequency", False))
    assert first["status"] == "processed"
    assert second["status"] == "cached"
    assert calls.count("wiener") == 1

    paths = module._cache_paths(output_root, row["evaluation_id"])
    paths["coherent"].unlink()
    recovered = module._process_one(
        (row, cfg, str(output_root), "frequency", False)
    )
    assert recovered["status"] == "processed"
    assert paths["coherent"].is_file()
    assert calls.count("wiener") == 2

    changed = deepcopy(cfg)
    changed["wiener"]["coherence_threshold"] = 0.5
    mismatch = module._process_one(
        (row, changed, str(output_root), "frequency", False)
    )
    assert mismatch["status"] == "failed"
    assert "--force" in mismatch["error"]
    forced = module._process_one(
        (row, changed, str(output_root), "frequency", True)
    )
    assert forced["status"] == "processed"
    assert calls.count("wiener") == 3


def test_main_applies_overrides_and_writes_failure_manifest(
    tmp_path, monkeypatch,
):
    module = _load_script()
    cfg = _config(tmp_path)
    source = tmp_path / "recording.edf"
    source.write_bytes(b"edf")
    index = pd.DataFrame([_row(source)])
    observed: dict = {}

    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))
    monkeypatch.setattr(module, "build_recording_index", lambda local: index.copy())
    monkeypatch.setattr(
        module, "assign_dataset_splits", lambda frame, local: frame.copy()
    )

    def fake_run_jobs(jobs, workers):
        row, local_cfg, output_root, mode, force = jobs[0]
        observed.update({
            "cfg": local_cfg,
            "output_root": output_root,
            "mode": mode,
            "workers": workers,
            "force": force,
        })
        paths = module._cache_paths(Path(output_root), row["evaluation_id"])
        return [{
            **module._manifest_base(row, paths),
            "status": "failed",
            "cache_hit": False,
            "fingerprint": "",
            "n_channels": np.nan,
            "n_times": np.nan,
            "sfreq": np.nan,
            "duration_sec": np.nan,
            "error": "ValueError: bad recording",
        }]

    monkeypatch.setattr(module, "_run_jobs", fake_run_jobs)
    data_override = tmp_path / "tuab-v3.0.1"
    cache_override = tmp_path / "cache-override"
    with pytest.raises(RuntimeError, match="1 TUAB recordings failed"):
        module.main(
            "ignored.yaml",
            data_dir=str(data_override),
            cache_dir=str(cache_override),
            mode="phasegated",
            workers=2,
            force=True,
        )

    assert observed["cfg"]["paths"]["data_root"] == str(data_override.resolve())
    assert observed["cfg"]["paths"]["cache_dir"] == str(cache_override.resolve())
    assert observed["cfg"]["wiener"]["mode"] == "phasegated"
    assert observed["mode"] == "phasegated"
    assert observed["workers"] == 2
    assert observed["force"] is True
    output_root = cache_override / "tuab_continuous_wiener_phasegated"
    manifest = pd.read_csv(output_root / "manifest.csv")
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "error"] == "ValueError: bad recording"
    assert not (output_root / "config_resolved.yaml").exists()


def test_main_writes_resolved_config_after_success(tmp_path, monkeypatch):
    module = _load_script()
    cfg = _config(tmp_path)
    source = tmp_path / "recording.edf"
    source.write_bytes(b"edf")
    index = pd.DataFrame([_row(source)])

    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))
    monkeypatch.setattr(module, "build_recording_index", lambda local: index.copy())
    monkeypatch.setattr(
        module, "assign_dataset_splits", lambda frame, local: frame.copy()
    )

    def fake_run_jobs(jobs, workers):
        row, local_cfg, output_root, mode, force = jobs[0]
        paths = module._cache_paths(Path(output_root), row["evaluation_id"])
        return [{
            **module._manifest_base(row, paths),
            "status": "cached",
            "cache_hit": True,
            "fingerprint": "fingerprint",
            "n_channels": 19,
            "n_times": 150000,
            "sfreq": 125.0,
            "duration_sec": 1200.0,
            "error": "",
        }]

    monkeypatch.setattr(module, "_run_jobs", fake_run_jobs)
    manifest = module.main("ignored.yaml")

    output_root = Path(cfg["paths"]["cache_dir"]) / (
        "tuab_continuous_wiener_frequency"
    )
    assert manifest.loc[0, "status"] == "cached"
    resolved = output_root / "config_resolved.yaml"
    assert resolved.is_file()
    assert "mode: frequency" in resolved.read_text(encoding="utf-8")


def test_main_rejects_non_tuab_and_invalid_worker_count(tmp_path, monkeypatch):
    module = _load_script()
    cfg = _config(tmp_path)
    cfg["dataset"]["active"] = "tuep"
    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))

    with pytest.raises(ValueError, match="dataset.active: tuab"):
        module.main("ignored.yaml")
    with pytest.raises(ValueError, match="workers"):
        module.main("ignored.yaml", workers=0)
