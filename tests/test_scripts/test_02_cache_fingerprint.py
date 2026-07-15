"""Tests for script 02 Wiener cache fingerprint validation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eeg_bg.io.cache import make_wiener_cache_fingerprint


def _load_script02():
    spec = importlib.util.spec_from_file_location(
        "script02_cache_fingerprint", Path("scripts/02_run_wiener.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg():
    return {
        "preprocessing": {"target_sfreq": 125.0},
        "channels": {"channel_groups": [["FP1", "FP2"]]},
        "wiener": {
            "nperseg": 500,
            "coherence_threshold": 0.15,
            "filter_magnitude_threshold": 50.0,
            "overlap_policy": "coherence_weighted",
            "phase_gate_threshold_rad": 0.392,
            "freq_band": [0.5, 40.0],
        },
    }


def _write_cached_output(tmp_path, cfg, schema_version=3):
    epoch_root = tmp_path / "epochs"
    out_root = tmp_path / "wiener_frequency"
    npz_path = epoch_root / "subject" / "data.npz"
    out_path = out_root / "subject" / "data.npz"
    out_path.parent.mkdir(parents=True)
    np.savez(
        out_path,
        schema_version=schema_version,
        wiener_config_fingerprint=make_wiener_cache_fingerprint(
            cfg, "frequency"
        ),
    )
    return npz_path, epoch_root, out_root


def test_matching_wiener_fingerprint_reuses_cache(tmp_path):
    script02 = _load_script02()
    cfg = _cfg()
    npz_path, epoch_root, out_root = _write_cached_output(tmp_path, cfg)

    result = script02._process_wiener_file(
        (str(npz_path), cfg, str(epoch_root), str(out_root), "frequency", False)
    )

    assert result == ("data.npz", "cached", None)


def test_new_wiener_cache_stores_schema_and_fingerprint(tmp_path):
    script02 = _load_script02()
    cfg = _cfg()
    cfg["wiener"]["nperseg"] = 64
    epoch_root = tmp_path / "epochs"
    out_root = tmp_path / "wiener_frequency"
    npz_path = epoch_root / "subject" / "data.npz"
    npz_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
    np.savez(
        npz_path,
        epochs=rng.standard_normal((1, 2, 128)),
        ch_names=np.array(["FP1", "FP2"], dtype=object),
        subject_id="subject",
        label=0,
        split="train",
    )

    result = script02._process_wiener_file(
        (str(npz_path), cfg, str(epoch_root), str(out_root), "frequency", True)
    )

    assert result == ("data.npz", "done", None)
    with np.load(out_root / "subject" / "data.npz", allow_pickle=True) as data:
        assert int(data["schema_version"]) == script02.WIENER_CACHE_SCHEMA_VERSION
        assert str(data["wiener_config_fingerprint"]) == (
            make_wiener_cache_fingerprint(cfg, "frequency")
        )


def test_changed_wiener_config_requires_force(tmp_path):
    script02 = _load_script02()
    cached_cfg = _cfg()
    npz_path, epoch_root, out_root = _write_cached_output(
        tmp_path, cached_cfg
    )
    current_cfg = _cfg()
    current_cfg["wiener"]["coherence_threshold"] = 0.45

    with pytest.raises(ValueError, match="configuration mismatch.*--force"):
        script02._process_wiener_file(
            (
                str(npz_path),
                current_cfg,
                str(epoch_root),
                str(out_root),
                "frequency",
                False,
            )
        )


def test_legacy_wiener_schema_requires_force(tmp_path):
    script02 = _load_script02()
    cfg = _cfg()
    npz_path, epoch_root, out_root = _write_cached_output(
        tmp_path, cfg, schema_version=2
    )

    with pytest.raises(ValueError, match="Old Wiener cache schema.*--force"):
        script02._process_wiener_file(
            (
                str(npz_path),
                cfg,
                str(epoch_root),
                str(out_root),
                "frequency",
                False,
            )
        )
