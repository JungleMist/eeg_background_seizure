"""Generate synthetic TUEP-format EDF data for smoke testing."""
import mne
import numpy as np
from pathlib import Path

DATA_ROOT = Path("D:/EEGdata/TUEP/v3.1.0")
SFREQ = 256.0
DURATION_SEC = 90.0

CH_NAMES = [
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
    "EEG F7-REF", "EEG F8-REF", "EEG C3-REF", "EEG C4-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF",
    "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF",
    "EEG Fz-REF", "EEG Cz-REF", "EEG Pz-REF",
]

SUBJECTS = [
    ("00_epilepsy", "smoke_ep01"), ("00_epilepsy", "smoke_ep02"),
    ("00_epilepsy", "smoke_ep03"), ("00_epilepsy", "smoke_ep04"),
    ("01_no_epilepsy", "smoke_ct01"), ("01_no_epilepsy", "smoke_ct02"),
    ("01_no_epilepsy", "smoke_ct03"), ("01_no_epilepsy", "smoke_ct04"),
]


def create_subject(class_folder, subject_id, seed):
    montage_dir = DATA_ROOT / class_folder / subject_id / "s001_smoke" / "01_tcp_ar"
    montage_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{subject_id}_s001_t001"
    edf_path = montage_dir / f"{stem}.edf"
    csv_path = montage_dir / f"{stem}.csv_bi"

    # Create EDF
    n_samples = int(DURATION_SEC * SFREQ)
    data = np.random.default_rng(seed).standard_normal((19, n_samples)) * 15.0 * 1e-6  # MNE expects Volts
    info = mne.create_info(ch_names=CH_NAMES, sfreq=SFREQ, ch_types=["eeg"] * 19)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Try raw.export first; fall back to mne.export.export_raw if needed
    try:
        raw.export(str(edf_path), fmt="edf", overwrite=True, verbose=False)
    except AttributeError:
        mne.export.export_raw(str(edf_path), raw, fmt="edf", overwrite=True, verbose=False)

    # Create csv_bi — plain text, no # header; annotation.py skips lines starting with #
    csv_path.write_text(
        "channel,start_time,stop_time,label,confidence\n"
        f"EEG FP1-REF,0.0000,{DURATION_SEC:.4f},bckg,0.9500\n"
    )
    print(f"  Created: {edf_path}")


if __name__ == "__main__":
    print("Creating synthetic TUEP smoke test data...")
    for seed, (class_folder, subject_id) in enumerate(SUBJECTS):
        create_subject(class_folder, subject_id, seed)
    print("Done.")
