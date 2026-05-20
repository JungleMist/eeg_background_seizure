import mne
import numpy as np
from pathlib import Path


def load_edf(
    edf_path: Path, cfg: dict
) -> tuple[np.ndarray, list[str], float]:
    target_sfreq = cfg["preprocessing"]["target_sfreq"]
    standard_19 = cfg["channels"]["standard_19"]
    low, high = cfg["preprocessing"]["bandpass"]

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    available = [ch for ch in standard_19 if ch in raw.ch_names]
    if not available:
        raise ValueError(
            f"No standard channels found in {edf_path}. "
            f"Available: {raw.ch_names}"
        )
    raw.pick_channels(available)
    raw.filter(low, high, verbose=False)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    data = raw.get_data() * 1e6  # V → uV
    return data, list(raw.ch_names), float(raw.info["sfreq"])
