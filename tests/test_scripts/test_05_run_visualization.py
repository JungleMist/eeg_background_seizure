"""Regression tests for script 05 visualization/EDF export wiring."""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path("scripts/05_run_visualization.py")


def _load_script05(module_name: str = "script05_under_test"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(tmp_path: Path, *, export_edf: bool = False,
                  export_edf_max_epochs: int = 3) -> Path:
    cache_dir = tmp_path / "cache"
    results_dir = tmp_path / "results"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "test.yaml"
    config_path.write_text(
        f"""
paths:
  cache_dir: "{cache_dir}"
  results_dir: "{results_dir}"
preprocessing:
  target_sfreq: 125
visualization:
  psd_target_channels: ["FP1"]
  n_subjects: 1
  epoch_idx: 0
  export_edf: {str(export_edf).lower()}
  export_edf_max_epochs: {export_edf_max_epochs}
wiener:
  nperseg: 32
  freq_band: [0.5, 40.0]
""",
        encoding="utf-8",
    )
    return config_path


def _write_epoch_cache(tmp_path: Path, *, n_epochs: int = 3) -> Path:
    subj_dir = tmp_path / "cache" / "epochs" / "00_subject"
    subj_dir.mkdir(parents=True)
    epochs = np.arange(n_epochs * 2 * 64, dtype=np.float64).reshape(n_epochs, 2, 64)
    np.savez(subj_dir / "sample.npz", epochs=epochs, ch_names=np.array(["FP1", "FP2"]))
    return subj_dir / "sample.npz"


def test_help_does_not_require_mne_import():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--export-edf-max-epochs" in result.stdout


def test_import_does_not_import_mne_or_edf_writer(monkeypatch):
    monkeypatch.delitem(sys.modules, "mne", raising=False)
    monkeypatch.delitem(sys.modules, "eeg_bg.io.edf_writer", raising=False)

    _load_script05("script05_no_mne_import")

    assert "mne" not in sys.modules
    assert "eeg_bg.io.edf_writer" not in sys.modules


@pytest.mark.parametrize("bad_cap", [0, -1])
def test_export_edf_max_epochs_rejects_non_positive_values(tmp_path, bad_cap):
    config_path = _write_config(tmp_path, export_edf=True, export_edf_max_epochs=bad_cap)
    mod = _load_script05(f"script05_bad_cap_{bad_cap}")

    with pytest.raises(mod.VisualizationConfigError, match="positive integer"):
        mod.main(str(config_path), None, None, None, None, None)


def test_png_only_path_does_not_import_edf_writer(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, export_edf=False)
    _write_epoch_cache(tmp_path)
    mod = _load_script05("script05_png_only")

    monkeypatch.delitem(sys.modules, "eeg_bg.io.edf_writer", raising=False)
    monkeypatch.setattr(mod, "_save", lambda fig, path: None)
    monkeypatch.setattr(mod, "plot_multichannel_comparison", lambda *a, **kw: object())
    monkeypatch.setattr(mod, "plot_psd_comparison", lambda *a, **kw: object())

    mod.main(str(config_path), None, None, None, None, None)

    assert "eeg_bg.io.edf_writer" not in sys.modules


def test_positive_export_cap_limits_epoch_directories(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, export_edf=True, export_edf_max_epochs=1)
    _write_epoch_cache(tmp_path, n_epochs=3)
    mod = _load_script05("script05_export_cap")
    calls = []

    fake_writer = types.ModuleType("eeg_bg.io.edf_writer")

    def fake_export_epoch_edf(epoch, ch_names, sfreq, out_path):
        calls.append(Path(out_path))

    fake_writer.export_epoch_edf = fake_export_epoch_edf
    monkeypatch.setitem(sys.modules, "eeg_bg.io.edf_writer", fake_writer)
    monkeypatch.setattr(mod, "_save", lambda fig, path: None)
    monkeypatch.setattr(mod, "plot_multichannel_comparison", lambda *a, **kw: object())
    monkeypatch.setattr(mod, "plot_psd_comparison", lambda *a, **kw: object())

    mod.main(str(config_path), None, None, None, None, None)

    assert [path.name for path in calls] == ["raw.edf"]
    edf_root = tmp_path / "results" / "figures" / "00_subject" / "edf"
    assert (edf_root / "epoch_0").is_dir()
    assert not (edf_root / "epoch_1").exists()
