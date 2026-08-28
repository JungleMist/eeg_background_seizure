"""Focused tests for Script 20's Script 18 cache adapter."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path("scripts/20_train_tuab_component_xgboost.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script20", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


def _cfg():
    return {
        "channels": {"standard_19": CHANNELS},
        "preprocessing": {"target_sfreq": 125.0},
    }


def _write_epoch_cache(path, *, split="train", label=0, channels=CHANNELS):
    rng = np.random.default_rng(3)
    arrays = {
        key: rng.standard_normal((2, len(channels), 128)).astype(np.float32)
        for key in ("raw", "specific", "coherent")
    }
    arrays["specific_coherent"] = np.concatenate(
        [arrays["specific"], arrays["coherent"]], axis=1
    )
    combined_names = [
        *(f"specific::{channel}" for channel in channels),
        *(f"coherent::{channel}" for channel in channels),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, **arrays, ch_names=np.asarray(channels),
        specific_coherent_ch_names=np.asarray(combined_names), sfreq=125.0,
        split=split, label=label,
        class_name="abnormal" if label == 0 else "normal",
        patient_id="p1", recording_id="r1", evaluation_id="r1",
        fingerprint="source-1", source_mode="frequency", schema_version=2,
    )


def test_source_inventory_validates_and_hashes_script18_cache(tmp_path):
    module = _load_script()
    path = tmp_path / "r1.npz"
    _write_epoch_cache(path)

    files, fingerprint, metadata = module._source_inventory(tmp_path, _cfg())

    assert files == [path]
    assert len(fingerprint) == 64
    assert metadata["source_modes"] == ["frequency"]
    assert metadata["n_files"] == 1


def test_extract_file_maps_each_component_to_profile_features(tmp_path):
    module = _load_script()
    path = tmp_path / "r1.npz"
    _write_epoch_cache(path)
    args_base = ("train", 125.0, 64, (0.5, 40.0), "base211", 64)

    outputs = {}
    for condition in module.CONDITIONS:
        result = module._extract_file((0, str(path), condition, *args_base))
        outputs[condition] = result

    assert all(len(outputs[c][1]) == 2 for c in module.CONDITIONS)
    assert all(
        outputs[c][1][0].shape == (211,)
        for c in module.BASE_CONDITIONS
    )
    assert outputs["specific_coherent"][1][0].shape == (422,)
    np.testing.assert_allclose(
        outputs["specific_coherent"][1][0],
        np.concatenate([
            outputs["specific"][1][0], outputs["coherent"][1][0]
        ]),
    )
    assert outputs["raw"][3] == ["r1", "r1"]
    assert outputs["raw"][4] == ["p1", "p1"]
    assert outputs["raw"][5] == ["r1", "r1"]


def test_combined_profile_dimensions_and_names(tmp_path):
    module = _load_script()

    assert module._condition_feature_dim("base211", "specific_coherent") == 422
    assert (
        module._condition_feature_dim("base211_conn80", "specific_coherent")
        == 582
    )
    names = module._condition_feature_names("base211", "specific_coherent")
    assert len(names) == 422
    assert all(name.startswith("specific::") for name in names[:211])
    assert all(name.startswith("coherent::") for name in names[211:])
    assert module._aggregate_feature_names(names[:1]) == [
        module.PROFILES["base211"].names[0]
    ]

    path = tmp_path / "r1.npz"
    _write_epoch_cache(path)
    result = module._extract_file((
        0, str(path), "specific_coherent", "train", 125.0, 64,
        (0.5, 40.0), "base211_conn80", 64,
    ))
    assert result[1][0].shape == (582,)


def test_source_inventory_rejects_invalid_tuab_mapping(tmp_path):
    module = _load_script()
    path = tmp_path / "bad.npz"
    _write_epoch_cache(path, label=0)
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["class_name"] = np.asarray("normal")
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Invalid TUAB label mapping"):
        module._source_inventory(tmp_path, _cfg())


def test_source_inventory_rejects_legacy_script18_schema(tmp_path):
    module = _load_script()
    path = tmp_path / "legacy.npz"
    _write_epoch_cache(path)
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["schema_version"] = np.asarray(1)
    np.savez(path, **payload)

    with pytest.raises(ValueError, match="Unsupported Script 18 schema"):
        module._source_inventory(tmp_path, _cfg())


def test_workers_must_be_positive():
    module = _load_script()

    assert module._positive_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        module._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        module._positive_int("-1")


def test_feature_cache_schema_mismatch_requires_force(tmp_path, monkeypatch):
    module = _load_script()
    path = tmp_path / "r1.npz"
    _write_epoch_cache(path)
    cfg = {
        **_cfg(),
        "wiener": {"nperseg": 64, "freq_band": [0.5, 40.0]},
        "ml": {"features": {"connectivity": {"nperseg": 64}}},
    }
    cache_path = module._cache_path(tmp_path, "base211", "raw", "train")
    cache_path.parent.mkdir(parents=True)
    np.savez(cache_path, X=np.zeros((1, 211)), y=np.zeros(1),
             evaluation_ids=np.asarray(["r1"]), patient_ids=np.asarray(["p1"]),
             recording_ids=np.asarray(["r1"]), dataset_names=np.asarray(["tuab"]),
             schema_hash=np.asarray("stale"))

    with pytest.raises(ValueError, match="schema mismatch"):
        module._load_or_extract([path], tmp_path, "source-1", "raw", "train",
                                cfg, "base211", False, 1)


def test_main_all_runs_and_summarizes_four_conditions(tmp_path, monkeypatch):
    module = _load_script()
    cfg = {
        "dataset": {"active": "tuab"},
        "paths": {
            "cache_dir": str(tmp_path / "cache"),
            "results_dir": str(tmp_path / "results"),
        },
        "wiener": {"mode": "frequency"},
    }
    seen = []
    monkeypatch.setattr(module, "load_config", lambda _: cfg)
    monkeypatch.setattr(
        module,
        "_source_inventory",
        lambda *_: ([tmp_path / "record.npz"], "source", {}),
    )

    def fake_run_condition(condition, *args):
        seen.append(condition)
        (
            tmp_path / "results" / "tuab_component_xgboost"
            / "base211" / condition
        ).mkdir(parents=True, exist_ok=True)
        return {"condition": condition, "val_metrics": {}, "test_metrics": {}}

    monkeypatch.setattr(module, "run_condition", fake_run_condition)

    module.main("ignored.yaml", condition="all", feature_set="base211")

    assert seen == list(module.CONDITIONS)
    summary = tmp_path / "results" / "tuab_component_xgboost" / "base211"
    assert list(np.loadtxt(
        summary / "comparison_summary.csv",
        delimiter=",", dtype=str, skiprows=1, usecols=0,
    )) == list(module.CONDITIONS)
