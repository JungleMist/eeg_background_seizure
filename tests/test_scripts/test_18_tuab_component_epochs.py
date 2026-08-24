"""Tests for Script 18's paired TUAB component epoch cache."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from eeg_bg.config.settings import load_config


SCRIPT_PATH = Path("scripts/18_extract_tuab_component_epochs.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("script18_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> dict:
    cfg = load_config("configs/tuab.yaml")
    cfg["paths"]["cache_dir"] = str(tmp_path / "cache")
    return cfg


def _write_script17_cache(
    root: Path,
    specific: np.ndarray,
    coherent: np.ndarray,
    *,
    evaluation_id: str = "aaaaa001_s001_t000",
    split: str = "train",
    label: int = 0,
    class_name: str = "abnormal",
    sfreq: float = 125.0,
) -> dict[str, Path]:
    paths = {
        "specific": root / "specific" / f"{evaluation_id}.npz",
        "coherent": root / "coherent" / f"{evaluation_id}.npz",
        "metadata": root / "metadata" / f"{evaluation_id}.json",
    }
    fingerprint = "source-fingerprint"
    common = {
        "ch_names": np.asarray(_config(root)["channels"]["standard_19"]),
        "sfreq": np.asarray(sfreq),
        "fingerprint": np.asarray(fingerprint),
        "schema_version": np.asarray(1),
        "patient_id": np.asarray("aaaaa001"),
        "recording_id": np.asarray(evaluation_id),
        "evaluation_id": np.asarray(evaluation_id),
        "class_name": np.asarray(class_name),
        "label": np.asarray(label),
        "split": np.asarray(split),
        "source_partition": np.asarray("eval" if split == "test" else "train"),
    }
    for component, sequence in (("specific", specific), ("coherent", coherent)):
        paths[component].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths[component], sequence=sequence.astype(np.float32),
            component=np.asarray(component), **common,
        )
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    paths["metadata"].write_text(json.dumps({
        "schema_version": 1,
        "fingerprint": fingerprint,
        "mode": "frequency",
        "n_times": int(specific.shape[1]),
        "sfreq": float(sfreq),
        "ch_names": _config(root)["channels"]["standard_19"],
    }), encoding="utf-8")
    return paths


def _paired_source() -> tuple[np.ndarray, np.ndarray]:
    n_channels = 19
    epoch_samples = 2500
    n_times = epoch_samples * 2 + 17
    specific = np.full((n_channels, n_times), 10.0, dtype=np.float32)
    coherent = np.zeros_like(specific)
    # This accepted raw window contains individually extreme components.
    specific[0, :epoch_samples] = -290.0
    coherent[0, :epoch_samples] = 300.0
    # The second raw window exceeds the shared 200 uV rejection threshold.
    specific[:, epoch_samples:2 * epoch_samples] = 250.0
    return specific, coherent


def test_extracts_paired_epochs_using_raw_shared_rejection(tmp_path):
    module = _load_script()
    cfg = _config(tmp_path)
    input_root = tmp_path / "script17"
    output_root = input_root / "epochs"
    specific, coherent = _paired_source()
    paths = _write_script17_cache(input_root, specific, coherent)

    result = module._process_one((paths, str(output_root), cfg, False))

    assert result["status"] == "processed"
    assert result["n_candidate_epochs"] == 2
    assert result["n_rejected_epochs"] == 1
    assert result["n_epochs"] == 1
    assert result["tail_samples"] == 17
    with np.load(output_root / "aaaaa001_s001_t000.npz", allow_pickle=False) as data:
        assert data["raw"].shape == (1, 19, 2500)
        assert data["specific"].dtype == np.float32
        assert data["coherent"].dtype == np.float32
        np.testing.assert_allclose(data["specific"] + data["coherent"], data["raw"])
        np.testing.assert_array_equal(data["epoch_start_samples"], [0])
        np.testing.assert_array_equal(data["epoch_start_sec"], [0.0])
        # The window survives because raw=10 uV even though both components
        # individually exceed 200 uV on FP1.
        np.testing.assert_allclose(data["specific"][0, 0], -290.0)
        np.testing.assert_allclose(data["coherent"][0, 0], 300.0)
        np.testing.assert_allclose(data["raw"][0, 0], 10.0)
        assert int(data["label"]) == 0
        assert str(data["class_name"]) == "abnormal"
        assert str(data["split"]) == "train"
        assert str(data["subject_id"]) == "aaaaa001_s001_t000"
        assert str(data["rejection_policy"]) == "raw_shared_mask"


@pytest.mark.parametrize(
    ("split", "label", "class_name"),
    [("train", 0, "abnormal"), ("val", 1, "normal"), ("test", 1, "normal")],
)
def test_preserves_split_and_tuab_label_mapping(
    tmp_path, split, label, class_name,
):
    module = _load_script()
    cfg = _config(tmp_path)
    input_root = tmp_path / split
    sequence = np.zeros((19, 2500), dtype=np.float32)
    paths = _write_script17_cache(
        input_root, sequence, sequence,
        evaluation_id=f"recording_{split}", split=split,
        label=label, class_name=class_name,
    )

    result = module._process_one((paths, str(input_root / "epochs"), cfg, False))

    assert result["status"] == "processed"
    with np.load(input_root / "epochs" / f"recording_{split}.npz") as data:
        assert str(data["split"]) == split
        assert int(data["label"]) == label
        assert str(data["class_name"]) == class_name


def test_cache_hit_mismatch_force_and_no_valid_epochs(tmp_path):
    module = _load_script()
    cfg = _config(tmp_path)
    input_root = tmp_path / "script17"
    output_root = input_root / "epochs"
    specific, coherent = _paired_source()
    paths = _write_script17_cache(input_root, specific, coherent)

    first = module._process_one((paths, str(output_root), cfg, False))
    second = module._process_one((paths, str(output_root), cfg, False))
    assert first["status"] == "processed"
    assert second["status"] == "cached"

    changed = deepcopy(cfg)
    changed["preprocessing"]["artifact_threshold_uv"] = 300.0
    mismatch = module._process_one((paths, str(output_root), changed, False))
    assert mismatch["status"] == "failed"
    assert "--force" in mismatch["error"]
    forced = module._process_one((paths, str(output_root), changed, True))
    assert forced["status"] == "processed"
    assert forced["n_epochs"] == 2

    rejected_root = tmp_path / "rejected"
    extreme = np.full((19, 2500), 250.0, dtype=np.float32)
    rejected_paths = _write_script17_cache(
        rejected_root, extreme, np.zeros_like(extreme),
        evaluation_id="all_rejected",
    )
    skipped = module._process_one(
        (rejected_paths, str(rejected_root / "epochs"), cfg, False)
    )
    assert skipped["status"] == "skipped"
    assert skipped["n_epochs"] == 0
    assert not (rejected_root / "epochs" / "all_rejected.npz").exists()


def test_discovery_reports_orphan_cache_in_manifest(tmp_path, monkeypatch):
    module = _load_script()
    cfg = _config(tmp_path)
    input_root = tmp_path / "orphan"
    for name in ("specific", "coherent", "metadata"):
        (input_root / name).mkdir(parents=True)
    np.savez(input_root / "specific" / "orphan.npz", sequence=np.zeros((1, 1)))
    output_root = tmp_path / "epochs"
    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))

    with pytest.raises(RuntimeError, match="1 TUAB epoch caches failed"):
        module.main(
            "ignored.yaml", input_dir=str(input_root),
            output_dir=str(output_root), workers=1,
        )

    manifest = pd.read_csv(output_root / "manifest.csv")
    assert manifest.loc[0, "status"] == "failed"
    assert manifest.loc[0, "evaluation_id"] == "orphan"
    assert "Incomplete Script 17 cache" in manifest.loc[0, "error"]
    assert not (output_root / "config_resolved.yaml").exists()


def test_main_uses_script17_default_root_and_writes_resolved_config(
    tmp_path, monkeypatch,
):
    module = _load_script()
    cfg = _config(tmp_path)
    input_root = (
        Path(cfg["paths"]["cache_dir"])
        / "tuab_continuous_wiener_frequency"
    )
    specific = np.zeros((19, 2500), dtype=np.float32)
    _write_script17_cache(input_root, specific, specific)
    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))

    manifest = module.main("ignored.yaml")

    assert manifest.loc[0, "status"] == "processed"
    output_root = input_root / "epochs"
    assert (output_root / "aaaaa001_s001_t000.npz").is_file()
    resolved = (output_root / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "rejection_policy: raw_shared_mask" in resolved
    assert "source_modes:" in resolved
    assert "- frequency" in resolved


def test_rejects_invalid_label_mapping_and_non_tuab_config(tmp_path, monkeypatch):
    module = _load_script()
    cfg = _config(tmp_path)
    root = tmp_path / "invalid"
    sequence = np.zeros((19, 2500), dtype=np.float32)
    paths = _write_script17_cache(
        root, sequence, sequence, label=0, class_name="normal"
    )
    result = module._process_one((paths, str(root / "epochs"), cfg, False))
    assert result["status"] == "failed"
    assert "Invalid TUAB label mapping" in result["error"]

    cfg["dataset"]["active"] = "tuep"
    monkeypatch.setattr(module, "load_config", lambda path: deepcopy(cfg))
    with pytest.raises(ValueError, match="dataset.active: tuab"):
        module.main("ignored.yaml")
