import numpy as np
import pytest
from eeg_bg.preprocessing.epoch import slice_epochs, bandpass_filter


def test_slice_epochs_correct_count():
    sfreq = 125.0
    data = np.zeros((19, int(24 * sfreq)))
    intervals = [(0.0, 24.0)]
    epochs = slice_epochs(data, sfreq, intervals, epoch_len_sec=8.0,
                          artifact_threshold_uv=200.0)
    assert epochs.shape == (3, 19, 1000)


def test_slice_epochs_rejects_artifact():
    sfreq = 125.0
    data = np.zeros((19, 3000))
    data[0, 1000:2000] = 300.0   # epoch index 1 has artifact
    intervals = [(0.0, 24.0)]
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape[0] == 2  # epoch 1 rejected


def test_slice_epochs_empty_when_all_artifact():
    sfreq = 125.0
    data = np.full((19, 2000), 300.0)
    intervals = [(0.0, 16.0)]
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape == (0, 19, 1000)


def test_slice_epochs_multiple_intervals():
    sfreq = 125.0
    data = np.zeros((19, int(40 * sfreq)))
    intervals = [(0.0, 8.0), (16.0, 24.0)]
    epochs = slice_epochs(data, sfreq, intervals, 8.0, 200.0)
    assert epochs.shape[0] == 2


def test_bandpass_filter_preserves_shape():
    data = np.random.randn(19, 1000)
    filtered = bandpass_filter(data, sfreq=125.0, low=0.5, high=40.0)
    assert filtered.shape == data.shape


def test_bandpass_filter_attenuates_high_freq():
    sfreq = 125.0
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 60 * t)
    data = signal[None, :]
    filtered = bandpass_filter(data, sfreq, low=0.5, high=40.0)
    assert np.std(filtered) < 0.1 * np.std(data)
