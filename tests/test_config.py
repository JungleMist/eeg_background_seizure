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


def test_load_config_extends_and_merges_nested_dicts(tmp_path):
    base_text = """
paths:
  cache_dir: "cache"
  results_dir: "results"
wiener:
  mode: "frequency"
  nperseg: 500
  coherence_threshold: 0.15
"""
    child_text = """
extends: base.yaml
paths:
  results_dir: "results/child"
wiener:
  coherence_threshold: 0.45
"""
    (tmp_path / "base.yaml").write_text(base_text)
    child_file = tmp_path / "child.yaml"
    child_file.write_text(child_text)

    cfg = load_config(child_file)

    assert cfg["wiener"]["mode"] == "frequency"
    assert cfg["wiener"]["nperseg"] == 500
    assert cfg["wiener"]["coherence_threshold"] == 0.45
    assert Path(cfg["paths"]["cache_dir"]).is_absolute()
    assert Path(cfg["paths"]["results_dir"]).is_absolute()
    assert cfg["paths"]["results_dir"].endswith("results/child")


def test_load_config_default_path_works():
    """Loading from the actual default config should not raise."""
    cfg = load_config("configs/default.yaml")
    assert "paths" in cfg
    assert "wiener" in cfg
    assert "channels" in cfg
