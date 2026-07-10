"""Feature-profile registry for script 06 ``--feature-set``.

Each profile defines a named collection of features with a fixed dimension,
extraction function, and ordered name list.  The default ``base211`` preserves
exact backward compatibility with the existing 211-dim per-channel +
asymmetry vector.  ``base211_conn80`` appends the 80-dim connectivity
features (coherence + PLV for 8 homotopic pairs × 5 bands).

Adding a new profile requires:
  1. A new key in ``PROFILES``.
  2. A corresponding extraction function.
  3. Positional stability of ``.names`` — saved SHAP ``.npy`` arrays are
     indexed by position, not name.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eeg_bg.features.connectivity import connectivity_features, CONNECTIVITY_NAMES
from eeg_bg.features.extraction import extract_epoch_features, FEATURE_NAMES


@dataclass
class FeatureProfile:
    name: str
    dim: int
    extract_fn: Callable[[np.ndarray, list[str], float, int, tuple[float, float], int | None], np.ndarray]
    names: list[str]
    hash: str = ""

    def __post_init__(self):
        # Pre-compute a position- and parameter-sensitive schema hash.
        if not self.hash:
            raw = f"{self.names}"
            self.hash = hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Base 211 (default, backward-compatible) ──────────────────────────────────

def _extract_base211(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
    connectivity_nperseg: int | None = None,
) -> np.ndarray:
    return extract_epoch_features(epoch, ch_names, sfreq, nperseg, freq_band)


# ── Base 211 + connectivity 80 (291 dims) ────────────────────────────────────

def _extract_base211_conn80(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    nperseg: int,
    freq_band: tuple[float, float],
    connectivity_nperseg: int | None = None,
) -> np.ndarray:
    """Extract 211 base features + 80 connectivity features → 291-dim."""
    base = extract_epoch_features(epoch, ch_names, sfreq, nperseg, freq_band)
    # Connectivity uses its own nperseg (from ml.features.connectivity.nperseg),
    # passed in by the caller.
    conn = connectivity_features(
        epoch, ch_names, sfreq,
        nperseg=int(connectivity_nperseg or nperseg),
    )
    return np.concatenate([base, conn], axis=0)


# ── Registry ─────────────────────────────────────────────────────────────────

PROFILES: dict[str, FeatureProfile] = {
    "base211": FeatureProfile(
        name="base211",
        dim=211,
        extract_fn=_extract_base211,
        names=list(FEATURE_NAMES),
    ),
    "base211_conn80": FeatureProfile(
        name="base211_conn80",
        dim=291,
        extract_fn=_extract_base211_conn80,
        names=list(FEATURE_NAMES) + list(CONNECTIVITY_NAMES),
    ),
}
