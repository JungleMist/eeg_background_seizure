# tests/test_config.py
from pathlib import Path
import pytest
from eeg_bg.config.settings import load_config


def test_load_config_returns_dict(tmp_path):
    cfg_text = """
paths:
  data_root: "D:/EEGdata"
  cache_dir: "cache"
  results_dir: "results"
preprocessing:
  target_sfreq: 125
"""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)
    assert isinstance(cfg, dict)
    assert cfg["preprocessing"]["target_sfreq"] == 125


def test_load_config_resolves_relative_paths(tmp_path):
    cfg_text = """
paths:
  data_root: "D:/EEGdata"
  cache_dir: "cache"
  results_dir: "results"
"""
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(cfg_file)
    # cache_dir and results_dir should be resolved to absolute paths
    assert Path(cfg["paths"]["cache_dir"]).is_absolute()
    assert Path(cfg["paths"]["results_dir"]).is_absolute()


def test_load_config_default_path_works():
    """Loading from the actual default config should not raise."""
    cfg = load_config("configs/default.yaml")
    assert "paths" in cfg
    assert "wiener" in cfg
    assert "channels" in cfg
