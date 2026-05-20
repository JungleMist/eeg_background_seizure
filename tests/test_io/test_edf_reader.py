import pytest
import numpy as np

# EDF reader requires a real EDF file — mark as integration test
pytestmark = pytest.mark.integration


def test_load_edf_shape_and_units():
    """Run with: pytest tests/test_io/test_edf_reader.py -m integration"""
    from pathlib import Path
    from eeg_bg.io.edf_reader import load_edf
    from eeg_bg.config.settings import load_config

    cfg = load_config("configs/default.yaml")
    edf_path = next(
        Path(cfg["paths"]["data_root"]).glob("00_epilepsy/**/01_tcp_ar/*.edf")
    )
    data, ch_names, sfreq = load_edf(edf_path, cfg)
    assert sfreq == cfg["preprocessing"]["target_sfreq"]
    assert data.ndim == 2
    assert data.shape[0] <= 19
    assert np.max(np.abs(data)) < 5000  # uV, not raw volts
