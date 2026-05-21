import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Unit tests — no real EDF file required
# ---------------------------------------------------------------------------

def test_normalize_channel_name():
    """normalize_channel_name strips EEG prefix, reference suffix, uppercases."""
    from eeg_bg.io.edf_reader import normalize_channel_name

    # Standard TUEP average-reference naming
    assert normalize_channel_name("EEG FP1-REF") == "FP1"
    assert normalize_channel_name("EEG FP2-REF") == "FP2"
    assert normalize_channel_name("EEG FZ-REF")  == "FZ"
    assert normalize_channel_name("EEG CZ-REF")  == "CZ"
    assert normalize_channel_name("EEG PZ-REF")  == "PZ"
    # Linked-ears variant
    assert normalize_channel_name("EEG FP1-LE")  == "FP1"
    # Non-EEG channels stripped correctly
    assert normalize_channel_name("EMG-REF")      == "EMG"
    assert normalize_channel_name("IBI")          == "IBI"
    assert normalize_channel_name("BURSTS")       == "BURSTS"
    # Bare names are a no-op (already canonical, modulo uppercasing)
    assert normalize_channel_name("FP1")          == "FP1"
    assert normalize_channel_name("Fz")           == "FZ"


def test_load_edf_normalizes_tuep_channels(tmp_path, monkeypatch):
    """load_edf maps TUEP-style 'EEG FP1-REF' names to canonical 'FP1' names."""
    import mne
    from eeg_bg.io.edf_reader import load_edf
    from eeg_bg.config.settings import load_config

    cfg = load_config("configs/default.yaml")

    # 19 EEG channels in TUEP tcp_ar style + 3 non-EEG channels that must be dropped
    tuep_names = [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF",  "EEG F4-REF",
        "EEG F7-REF",  "EEG F8-REF",  "EEG C3-REF",  "EEG C4-REF",
        "EEG T3-REF",  "EEG T4-REF",  "EEG T5-REF",  "EEG T6-REF",
        "EEG P3-REF",  "EEG P4-REF",  "EEG O1-REF",  "EEG O2-REF",
        "EEG FZ-REF",  "EEG CZ-REF",  "EEG PZ-REF",
        "EEG EKG1-REF", "EMG-REF", "IBI",
    ]
    sfreq = 256.0
    n_times = int(sfreq * 10)
    info = mne.create_info(
        ch_names=tuep_names,
        sfreq=sfreq,
        ch_types=["eeg"] * 19 + ["ecg", "emg", "misc"],
    )
    fake_raw = mne.io.RawArray(
        np.random.default_rng(0).standard_normal((len(tuep_names), n_times)) * 1e-5,
        info,
        verbose=False,
    )

    # Patch mne.io.read_raw_edf so no real file is needed
    monkeypatch.setattr(mne.io, "read_raw_edf", lambda *a, **kw: fake_raw)

    data, ch_names, out_sfreq = load_edf(tmp_path / "fake.edf", cfg)

    canonical_19 = cfg["channels"]["standard_19"]
    assert set(ch_names) == set(canonical_19), (
        f"Expected {canonical_19}, got {ch_names}"
    )
    assert data.shape[0] == 19
    assert out_sfreq == cfg["preprocessing"]["target_sfreq"]
    # Confirm units are µV (not raw Volts — max should be well under 5000 µV)
    assert np.max(np.abs(data)) < 5000


# ---------------------------------------------------------------------------
# Integration test — requires real EDF files on disk
# ---------------------------------------------------------------------------

@pytest.mark.integration
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
