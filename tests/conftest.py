import numpy as np
import pytest


CH_NAMES_19 = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]

BASE_CFG = {
    "preprocessing": {"target_sfreq": 125.0, "bandpass": [0.5, 40.0]},
    "wiener": {
        "mode": "frequency",
        "nperseg": 500,
        "coherence_threshold": 0.05,   # Low so decomposition always runs
        "coherent_gate_enabled": False,
        "coherent_gate_threshold_uv": 100.0,
        "overlap_policy": "coherence_weighted",
        "phase_gate_threshold_rad": 0.392,
        "freq_band": [0.5, 40.0],
        "protected_band_hz": [5.0, 20.0],
    },
    "ica": {
        "n_components": 19,
        "method": "fastica",
        "artifact_corr_threshold": 0.8,
        "random_state": 42,
        "max_iter": 1000,
    },
    "channels": {
        "standard_19": CH_NAMES_19,
        "channel_groups": [               # G1–G6 movement-artifact conduction paths
            ["FP1", "FP2"],               # G1 – symmetric facial
            ["F7", "T3"],                 # G2 – left SCM
            ["T3", "T5", "O1"],           # G3 – left posterior neck (3-ch chain)
            ["O1", "O2"],                 # G4 – bilateral occipitalis
            ["F8", "T4"],                 # G5 – right SCM
            ["T4", "T6", "O2"],           # G6 – right posterior neck (3-ch chain)
        ],
        "passthrough": ["F3", "F4", "C3", "C4", "P3", "P4", "Fz", "Cz", "Pz"],
    },
}


@pytest.fixture
def cfg():
    """Default test configuration."""
    import copy
    return copy.deepcopy(BASE_CFG)


@pytest.fixture
def synthetic_epoch(cfg):
    """
    19-channel, 1000-sample (8s @ 125Hz) synthetic EEG.
    A single broadband point source is mixed into all channels
    with known scalar gains. Independent noise is added.
    High SNR (source_std=50 uV, noise_std=1 uV) so Wiener
    decomposition can reliably remove the shared component.
    """
    rng = np.random.default_rng(42)
    n_ch = len(CH_NAMES_19)
    n_times = 1000

    source = rng.standard_normal(n_times) * 50.0
    gains = rng.uniform(0.5, 1.0, n_ch)
    noise = rng.standard_normal((n_ch, n_times)) * 1.0
    epoch = gains[:, None] * source[None, :] + noise

    return epoch.astype(np.float64), CH_NAMES_19, cfg, gains, source


@pytest.fixture
def synthetic_epochs_batch(cfg):
    """Batch of 5 synthetic epochs for subject-level tests."""
    rng = np.random.default_rng(99)
    n_epochs, n_ch, n_times = 5, 19, 1000
    source = rng.standard_normal((n_epochs, n_times)) * 50.0
    gains = rng.uniform(0.5, 1.0, n_ch)
    noise = rng.standard_normal((n_epochs, n_ch, n_times)) * 1.0
    epochs = gains[None, :, None] * source[:, None, :] + noise
    return epochs.astype(np.float64), CH_NAMES_19, cfg


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
