"""EEG handcrafted feature extraction sub-package.

Public API
----------
extract_epoch_features : callable
    Convert one ``(n_ch, n_times)`` epoch → ``(171,)`` feature vector.
build_dataset : callable
    Load all NPZ epochs for a condition / split and return ``(X, y, subject_ids)``.
FEATURE_NAMES : list[str]
    Stable ordered list of 171 feature names (e.g. ``"FP1_delta_power"``).
BANDS : dict[str, tuple[float, float]]
    Standard EEG frequency-band definitions.
"""
from eeg_bg.features.extraction import (
    extract_epoch_features,
    build_dataset,
    FEATURE_NAMES,
)
from eeg_bg.features.band_power import BANDS

__all__ = [
    "extract_epoch_features",
    "build_dataset",
    "FEATURE_NAMES",
    "BANDS",
]
