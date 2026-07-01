"""Epoch-level feature extraction and dataset builder.

``extract_epoch_features`` converts a single ``(n_ch, n_times)`` epoch into a
fixed-length 1070-dimensional feature vector.

``build_dataset`` iterates over an NPZ cache directory, applies the extractor
to every epoch that belongs to the requested split, and returns a feature
matrix together with labels and subject IDs.

Feature vector layout (1070 dims)
----------------------------------
- [0:171]   Per-channel statistics — 19 channels × 9 features each
            (delta/theta/alpha/beta/gamma power, Hjorth activity/mobility/complexity,
            spectral entropy) in ``_STANDARD_19`` order.
- [171:211] Hemispheric asymmetry — 8 pairs × 5 bands (see ``ASYMMETRY_NAMES``).
- [211:724] Wavelet DWT — 19 channels × 27 features (detail energy, modulus-maxima
            mean, reconstructed-band stats). Cut from 66/channel — see
            ``eeg_bg/features/wavelet.py`` module docstring for the physiological
            rationale behind which of the original 7 subgroups survived.
- [724:804] Connectivity — 8 homotopic pairs × 5 bands × 2 metrics (coherence, PLV).
            Restricted from all C(19,2)=171 pairs to the 8 bilateral pairs in
            ``asymmetry.SYMMETRIC_PAIRS`` — see ``eeg_bg/features/connectivity.py``.
- [804:842] Complexity — 19 channels × 2 features (SampEn, LZC).
- [842:1070] Temporal multi-scale stats — 19 channels × 3 scales × 4 stats.

Positions 0–210 are positionally stable; reordering them invalidates saved
SHAP ``.npy`` arrays. Positions past 210 changed as part of the 2026-07-01
dimensionality reduction (3441 → 1070 dims) — any SHAP/feature-cache
artifacts from before that change are not comparable to new runs.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.signal import welch
from tqdm import tqdm

from eeg_bg.features.band_power import relative_band_power, BANDS
from eeg_bg.features.hjorth import hjorth_parameters
from eeg_bg.features.spectral_entropy import spectral_entropy as _spectral_entropy
from eeg_bg.features.asymmetry import hemispheric_asymmetry, ASYMMETRY_NAMES
from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.wavelet import wavelet_features, WAVELET_NAMES
from eeg_bg.features.connectivity import connectivity_features, CONNECTIVITY_NAMES
from eeg_bg.features.complexity import complexity_features, COMPLEXITY_NAMES
from eeg_bg.features.temporal_stats import epoch_temporal_stats, TEMPORAL_NAMES

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
    + WAVELET_NAMES
    + CONNECTIVITY_NAMES
    + COMPLEXITY_NAMES
    + TEMPORAL_NAMES
)  # 211 + 513 + 80 + 38 + 228 = 1070

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
    """Extract a fixed-length feature vector (``len(FEATURE_NAMES)`` dims) from one epoch.

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
        Shape ``(len(FEATURE_NAMES),)`` = ``(1070,)``, ordered according to
        :data:`FEATURE_NAMES`.  Missing channels are zero-padded.
    """
    # O(1) channel lookup — avoids O(n) list.index() inside the loop.
    ch_map = {name: i for i, name in enumerate(ch_names)}

    features: list[float] = []
    # psd_cache[ch] = (freqs, psd) computed once per channel; reused by
    # relative_band_power, spectral_entropy, and hemispheric_asymmetry so that
    # welch() is called at most once per unique channel per epoch.
    psd_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ch in _STANDARD_19:
        idx = ch_map.get(ch)
        if idx is None:
            features.extend([0.0] * len(_FEAT_SUFFIXES))
            continue

        sig = epoch[idx].astype(np.float64)

        # Compute PSD once; share with all spectral feature functions below.
        fp = welch(sig, fs=sfreq, nperseg=nperseg, window="boxcar")
        psd_cache[ch] = fp

        # 5 relative band powers
        bp = relative_band_power(sig, sfreq=sfreq, nperseg=nperseg,
                                  freq_band=freq_band, freqs_psd=fp)
        for band in BANDS:
            features.append(bp[band])

        # 3 Hjorth parameters (time-domain only — no PSD needed)
        act, mob, cplx = hjorth_parameters(sig)
        features.extend([act, mob, cplx])

        # 1 spectral entropy — reuses the already-computed PSD
        features.append(_spectral_entropy(sig, sfreq=sfreq, nperseg=nperseg,
                                           freq_band=freq_band, freqs_psd=fp))

    # 40 hemispheric asymmetry features — pass psd_cache so asymmetry pairs
    # use the PSDs already computed above instead of calling welch() again.
    asym = hemispheric_asymmetry(epoch, ch_names, sfreq=sfreq,
                                  nperseg=nperseg, freq_band=freq_band,
                                  psd_cache=psd_cache)
    # ── New feature blocks ────────────────────────────────────────────────────
    # Wavelet: 3 feature groups per channel (27 dims per channel, 513 total)
    wavelet_vec = np.concatenate([
        wavelet_features(epoch[ch_map[ch]] if ch_map.get(ch) is not None
                         else np.zeros(epoch.shape[1]))
        for ch in _STANDARD_19
    ])

    # Connectivity: coherence + PLV — 8 homotopic pairs × 5 bands × 2 metrics = 80 dims
    conn_vec = connectivity_features(epoch, ch_names, sfreq=sfreq, nperseg=nperseg)

    # Complexity: SampEn + LZC per channel (38 dims)
    compl_vec = complexity_features(epoch, ch_names)

    # Temporal: multi-scale stats per channel (228 dims)
    temp_vec = epoch_temporal_stats(epoch, ch_names)

    return np.concatenate([
        np.asarray(features, dtype=np.float64),
        asym,
        wavelet_vec,
        conn_vec,
        compl_vec,
        temp_vec,
    ]).astype(np.float64)


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
    X : np.ndarray, shape ``(n_epochs, len(FEATURE_NAMES))``
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_one_file, args): args
                   for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=f"Features [{condition}/{split}]", leave=False):
            rows, ys, sids = future.result()
            X_list.extend(rows)
            y_list.extend(ys)
            sid_list.extend(sids)

    if not X_list:
        return (np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
                [])

    return (np.vstack(X_list).astype(np.float64),
            np.asarray(y_list, dtype=np.int64),
            sid_list)
