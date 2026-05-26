"""Epoch-level feature extraction and dataset builder.

``extract_epoch_features`` converts a single ``(n_ch, n_times)`` epoch into a
fixed-length 211-dimensional feature vector.

``build_dataset`` iterates over an NPZ cache directory, applies the extractor
to every epoch that belongs to the requested split, and returns a feature
matrix together with labels and subject IDs.

Feature vector layout (211 = 171 per-channel + 40 hemispheric asymmetry)
-------------------------------------------------------------------------
First 171 features — 19 channels × 9 features each (in ``_STANDARD_19`` order):

    FP1_delta_power, FP1_theta_power, FP1_alpha_power, FP1_beta_power,
    FP1_gamma_power, FP1_hjorth_activity, FP1_hjorth_mobility,
    FP1_hjorth_complexity, FP1_spectral_entropy,
    FP2_delta_power, ...  (continues for all 19 channels)

Last 40 features — 8 symmetric pairs × 5 bands hemispheric asymmetry:

    asym_FP1_FP2_delta, asym_FP1_FP2_theta, ..., asym_O1_O2_gamma
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.features.band_power import relative_band_power, BANDS
from eeg_bg.features.hjorth import hjorth_parameters
from eeg_bg.features.spectral_entropy import spectral_entropy as _spectral_entropy
from eeg_bg.features.asymmetry import hemispheric_asymmetry, ASYMMETRY_NAMES

# Standard 19 channels in canonical order (matches configs/default.yaml)
_STANDARD_19 = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]

# Feature name suffixes in inner-loop order
_FEAT_SUFFIXES: list[str] = (
    [f"{b}_power" for b in BANDS]          # 5 band-power features
    + ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]  # 3 Hjorth
    + ["spectral_entropy"]                 # 1 spectral entropy
)  # total 9 per channel

# FEATURE_NAMES is built at import time and must remain stable.
# First 171: per-channel statistics; last 40: hemispheric asymmetry.
FEATURE_NAMES: list[str] = (
    [
        f"{ch}_{suffix}"
        for ch in _STANDARD_19
        for suffix in _FEAT_SUFFIXES
    ]
    + ASYMMETRY_NAMES
)  # length == 211

# Cache subdirectory names keyed by condition label
_CONDITION_TO_SUBDIR: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "wiener_frequency",
    "ica":    "ica",
}

# NPZ array key that holds the signal data for each condition
_CONDITION_TO_KEY: dict[str, str] = {
    "raw":    "epochs",
    "wiener": "specific",
    "ica":    "specific",
}


def extract_epoch_features(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    nperseg: int = 250,
    freq_band: tuple[float, float] = (0.5, 40.0),
) -> np.ndarray:
    """Extract a 171-dimensional feature vector from one epoch.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names corresponding to axis 0 of *epoch*.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch window length (default 250, pipeline-consistent).
    freq_band : tuple[float, float]
        Analysis band in Hz.

    Returns
    -------
    np.ndarray
        Shape ``(211,)``, ordered according to :data:`FEATURE_NAMES`.
        First 171 entries are per-channel statistics; last 40 are hemispheric
        asymmetry features.
    """
    features: list[float] = []
    for ch in _STANDARD_19:
        try:
            idx = ch_names.index(ch)
        except ValueError:
            # Channel absent — fill with zeros so vector length is always 171
            features.extend([0.0] * 9)
            continue

        sig = epoch[idx].astype(np.float64)

        # 5 relative band powers
        bp = relative_band_power(sig, sfreq=sfreq, nperseg=nperseg,
                                  freq_band=freq_band)
        for band in BANDS:
            features.append(bp[band])

        # 3 Hjorth parameters
        act, mob, cplx = hjorth_parameters(sig)
        features.extend([act, mob, cplx])

        # 1 spectral entropy
        features.append(_spectral_entropy(sig, sfreq=sfreq, nperseg=nperseg,
                                           freq_band=freq_band))

    # 40 hemispheric asymmetry features (appended after per-channel block)
    asym = hemispheric_asymmetry(epoch, ch_names, sfreq=sfreq,
                                  nperseg=nperseg, freq_band=freq_band)
    return np.concatenate(
        [np.asarray(features, dtype=np.float64), asym]
    ).astype(np.float64)


def _extract_one_file(args: tuple) -> tuple[list, list, list]:
    """Worker: load one subject npz and extract features for the target split."""
    npz_path_str, array_key, target_split, sfreq, nperseg, freq_band = args
    data = np.load(npz_path_str, allow_pickle=True)
    if str(data["split"]) != target_split:
        return [], [], []

    epochs_arr = data[array_key]
    ch_names   = (list(data["ch_names"])
                  if "ch_names" in data.files
                  else list(_STANDARD_19))
    label      = int(data["label"])
    subject_id = str(data["subject_id"])

    rows = [
        extract_epoch_features(ep, ch_names, sfreq=sfreq,
                                nperseg=nperseg, freq_band=freq_band)
        for ep in epochs_arr
    ]
    n = len(rows)
    return rows, [label] * n, [subject_id] * n


def build_dataset(
    cache_root: Path,
    condition: str,
    split: str,
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all cached epochs for *condition* / *split* and extract features.

    Parameters
    ----------
    cache_root : Path
        Project-level cache directory (e.g. ``Path("cache")`` resolved to abs).
    condition : str
        One of ``"raw"``, ``"wiener"``, ``"ica"``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch segment length.
    freq_band : tuple[float, float]
        Analysis band in Hz.
    max_workers : int | None
        Worker processes for parallel file processing.  ``None`` uses
        ``os.cpu_count()``.

    Returns
    -------
    X : np.ndarray, shape ``(n_epochs, 171)``
    y : np.ndarray, shape ``(n_epochs,)``, dtype int  (0=epilepsy, 1=control)
    subject_ids : list[str], length ``n_epochs``
        One entry per epoch; used for subject-level aggregation downstream.

    Raises
    ------
    ValueError
        If *condition* is not one of the recognised keys.
    FileNotFoundError
        If the condition subdirectory does not exist under *cache_root*.
    """
    if condition not in _CONDITION_TO_SUBDIR:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Expected one of {list(_CONDITION_TO_SUBDIR)}."
        )

    subdir = cache_root / _CONDITION_TO_SUBDIR[condition]
    if not subdir.exists():
        raise FileNotFoundError(
            f"Cache directory not found: {subdir}.  "
            f"Run the corresponding preprocessing script first."
        )

    array_key = _CONDITION_TO_KEY[condition]
    npz_files = sorted(subdir.rglob("*.npz"))

    args_list = [
        (str(p), array_key, split, sfreq, nperseg, freq_band)
        for p in npz_files
    ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    sid_list: list[str] = []

    n_workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_one_file, args): args
                   for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Features [{condition}/{split}]", leave=False):
            rows, ys, sids = future.result()
            X_list.extend(rows)
            y_list.extend(ys)
            sid_list.extend(sids)

    if not X_list:
        return (np.empty((0, 211), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                [])

    return (np.vstack(X_list).astype(np.float64),
            np.asarray(y_list, dtype=np.int64),
            sid_list)
