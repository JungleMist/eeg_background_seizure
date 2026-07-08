import mne
import numpy as np
import pytest


CH_NAMES = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
SFREQ = 125.0


@pytest.fixture
def epoch():
    """19-channel, 2500-sample (20s @ 125Hz) synthetic epoch, uV, realistic amplitude."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((len(CH_NAMES), 2500)) * 30.0).astype(np.float64)


def test_export_epoch_edf_roundtrip_shape_and_names(tmp_path, epoch):
    """Exported EDF round-trips to the same shape, channel names, and sfreq."""
    from eeg_bg.io.edf_writer import export_epoch_edf

    out_path = tmp_path / "raw.edf"
    export_epoch_edf(epoch, CH_NAMES, SFREQ, out_path)

    assert out_path.exists()
    raw = mne.io.read_raw_edf(str(out_path), preload=True, verbose=False)
    assert raw.get_data().shape == epoch.shape
    assert raw.info["sfreq"] == SFREQ
    assert list(raw.ch_names) == CH_NAMES


def test_export_epoch_edf_roundtrip_values_within_quantization_tolerance(tmp_path, epoch):
    """Exported/re-read values match the original within EDF's 16-bit quantization step.

    EDF stores samples as signed 16-bit integers scaled to each channel's
    physical min/max range ("auto" physical_range). Do not assert exact
    equality — empirically the max absolute round-trip error for a ~30 uV-std
    synthetic epoch is ~0.002 uV, so atol=0.01 gives comfortable headroom.
    """
    from eeg_bg.io.edf_writer import export_epoch_edf

    out_path = tmp_path / "raw.edf"
    export_epoch_edf(epoch, CH_NAMES, SFREQ, out_path)

    raw = mne.io.read_raw_edf(str(out_path), preload=True, verbose=False)
    data_uv = raw.get_data() * 1e6
    np.testing.assert_allclose(data_uv, epoch, atol=0.01, rtol=0)
