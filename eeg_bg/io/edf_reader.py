import mne
import numpy as np
from pathlib import Path


def normalize_channel_name(raw_name: str) -> str:
    """Normalise a raw EDF channel name to a bare electrode label (uppercase).

    Examples
    --------
    >>> normalize_channel_name("EEG FP1-REF")
    'FP1'
    >>> normalize_channel_name("EEG FZ-REF")
    'FZ'
    >>> normalize_channel_name("EMG-REF")
    'EMG'
    >>> normalize_channel_name("IBI")
    'IBI'
    >>> normalize_channel_name("FP1")
    'FP1'
    """
    name = raw_name.strip()
    # Strip leading "EEG " prefix (case-insensitive)
    if name.upper().startswith("EEG "):
        name = name[4:].strip()
    # Strip reference suffix like "-REF", "-AR", "-LE"
    if "-" in name:
        name = name.split("-")[0].strip()
    return name.upper()


def load_edf(
    edf_path: Path, cfg: dict
) -> tuple[np.ndarray, list[str], float]:
    target_sfreq = cfg["preprocessing"]["target_sfreq"]
    standard_19 = cfg["channels"]["standard_19"]
    low, high = cfg["preprocessing"]["bandpass"]

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Build a case-insensitive reverse lookup so that e.g. 'FZ' maps to config
    # name 'Fz' and 'FP1' maps to 'FP1'.
    canonical = {ch.upper(): ch for ch in standard_19}

    # Map each raw EDF channel name to its canonical config name.
    # normalize_channel_name strips the 'EEG ' prefix and '-REF'/'-AR'/'-LE'
    # suffix that TUEP recordings carry, then uppercases.
    name_map: dict[str, str] = {}
    for raw_ch in raw.ch_names:
        key = normalize_channel_name(raw_ch)
        if key in canonical:
            name_map[raw_ch] = canonical[key]

    if not name_map:
        raise ValueError(
            f"No standard channels found in {edf_path}. "
            f"Available: {raw.ch_names}"
        )

    raw.pick_channels(list(name_map.keys()))
    raw.rename_channels(name_map)   # raw.ch_names now uses canonical names
    raw.filter(low, high, verbose=False)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    data = raw.get_data() * 1e6  # V → uV
    return data, list(raw.ch_names), float(raw.info["sfreq"])
