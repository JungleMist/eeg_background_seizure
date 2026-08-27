"""Focused tests for Script 20's Script 18 cache adapter."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, **arrays, ch_names=np.asarray(channels), sfreq=125.0,
        split=split, label=label,
        class_name="abnormal" if label == 0 else "normal",
        patient_id="p1", recording_id="r1", evaluation_id="r1",
        fingerprint="source-1", source_mode="frequency", schema_version=1,
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
    assert all(outputs[c][1][0].shape == (211,) for c in module.CONDITIONS)
    assert outputs["raw"][3] == ["r1", "r1"]
    assert outputs["raw"][4] == ["p1", "p1"]
    assert outputs["raw"][5] == ["r1", "r1"]


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
