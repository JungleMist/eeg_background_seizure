import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    data: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    sos = butter(5, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def slice_epochs(
    data: np.ndarray,
    sfreq: float,
    intervals: list[tuple[float, float]],
    epoch_len_sec: float,
    artifact_threshold_uv: float,
) -> np.ndarray:
    epoch_len = int(epoch_len_sec * sfreq)
    n_ch = data.shape[0]
    epochs: list[np.ndarray] = []

    for start_sec, stop_sec in intervals:
        start_sample = int(start_sec * sfreq)
        stop_sample = int(stop_sec * sfreq)
        pos = start_sample
        while pos + epoch_len <= stop_sample:
            epoch = data[:, pos : pos + epoch_len]
            if np.max(np.abs(epoch)) <= artifact_threshold_uv:
                epochs.append(epoch.copy())
            pos += epoch_len

    if not epochs:
        return np.empty((0, n_ch, epoch_len))
    return np.stack(epochs)
