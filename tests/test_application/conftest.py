import mne
import numpy as np
import pytest


CHANNELS = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
]


@pytest.fixture
def synthetic_fif(tmp_path):
    sfreq = 125.0
    times = np.arange(int(8 * sfreq)) / sfreq
    rng = np.random.default_rng(7)
    shared = 20e-6 * np.sin(2 * np.pi * 8.0 * times)
    data = np.stack([
        shared + rng.normal(scale=2e-6, size=times.size) for _ in CHANNELS
    ])
    raw = mne.io.RawArray(
        data,
        mne.create_info(CHANNELS, sfreq, ch_types="eeg"),
        verbose=False,
    )
    raw.set_annotations(mne.Annotations([1.0], [0.2], ["marker"]))
    path = tmp_path / "synthetic-raw.fif"
    raw.save(path, overwrite=True, verbose=False)
    return path
