"""Export reconstructed epoch arrays to real .edf files (inverse of edf_reader.load_edf)."""
from pathlib import Path

import numpy as np


def export_epoch_edf(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    out_path: Path,
) -> None:
    """Write a single (n_channels, n_times) uV epoch to *out_path* as an EDF file."""
    import mne

    out_path = Path(out_path)
    info = mne.create_info(ch_names=list(ch_names), sfreq=float(sfreq),
                            ch_types=["eeg"] * len(ch_names))
    raw = mne.io.RawArray(np.asarray(epoch, dtype=np.float64) * 1e-6, info, verbose=False)
    try:
        raw.export(str(out_path), fmt="edf", overwrite=True, verbose=False)
    except AttributeError:
        mne.export.export_raw(str(out_path), raw, fmt="edf", overwrite=True, verbose=False)
