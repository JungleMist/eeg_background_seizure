import numpy as np
import pytest
from pathlib import Path
from eeg_bg.io.cache import make_cache_key, load_or_compute


def test_make_cache_key_is_stable():
    path = Path("D:/data/subject/file.edf")
    cfg = {"preprocessing": {"target_sfreq": 125, "bandpass": [0.5, 40.0]}}
    key1 = make_cache_key(path, 0.0, cfg)
    key2 = make_cache_key(path, 0.0, cfg)
    assert key1 == key2
    assert len(key1) == 16


def test_make_cache_key_changes_with_params():
    path = Path("D:/data/file.edf")
    cfg1 = {"preprocessing": {"target_sfreq": 125, "bandpass": [0.5, 40.0]}}
    cfg2 = {"preprocessing": {"target_sfreq": 256, "bandpass": [0.5, 40.0]}}
    assert make_cache_key(path, 0.0, cfg1) != make_cache_key(path, 0.0, cfg2)


def test_make_cache_key_changes_with_tuab_duration_cap():
    path = Path("/data/file.edf")
    base = {
        "dataset": {"active": "tuab", "tuab": {"max_recording_sec": 1200.0}},
        "preprocessing": {
            "target_sfreq": 125, "bandpass": [0.5, 40.0],
            "epoch_length_sec": 20.0,
        },
    }
    changed = {
        **base,
        "dataset": {"active": "tuab", "tuab": {"max_recording_sec": 600.0}},
    }
    assert make_cache_key(path, 0.0, base) != make_cache_key(path, 0.0, changed)


def test_load_or_compute_saves_and_loads(tmp_path):
    cache_path = tmp_path / "test.npz"
    called = {"n": 0}

    def compute():
        called["n"] += 1
        return {"data": np.array([1.0, 2.0, 3.0]), "label": np.array([0])}

    result1 = load_or_compute(cache_path, compute)
    result2 = load_or_compute(cache_path, compute)

    assert called["n"] == 1  # compute called only once
    np.testing.assert_array_equal(result1["data"], result2["data"])


def test_load_or_compute_force_recompute(tmp_path):
    cache_path = tmp_path / "test.npz"
    called = {"n": 0}

    def compute():
        called["n"] += 1
        return {"data": np.zeros(5)}

    load_or_compute(cache_path, compute)
    load_or_compute(cache_path, compute, force_recompute=True)
    assert called["n"] == 2
