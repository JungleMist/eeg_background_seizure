"""Fixtures for test_features — independent of the root conftest."""
import numpy as np
import pytest

CH_NAMES_19 = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
SFREQ = 125.0


@pytest.fixture
def ch_names_19() -> list[str]:
    return list(CH_NAMES_19)


@pytest.fixture
def sfreq() -> float:
    return SFREQ


@pytest.fixture
def synthetic_epoch() -> np.ndarray:
    """19-channel, 1000-sample broadband random epoch."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((19, 1000)).astype(np.float64) * 20.0


@pytest.fixture
def pure_sine_signal() -> np.ndarray:
    """1000-sample 10 Hz pure sine at 125 Hz (alpha band)."""
    t = np.arange(1000) / SFREQ
    return (np.sin(2 * np.pi * 10.0 * t) * 50.0).astype(np.float64)


@pytest.fixture
def constant_signal() -> np.ndarray:
    """1000-sample constant (flat) signal."""
    return np.full(1000, 5.0, dtype=np.float64)
