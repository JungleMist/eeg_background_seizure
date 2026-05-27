import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt
from eeg_bg.visualization.psd_plots import plot_psd_comparison


CH_NAMES = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
SFREQ = 125.0


@pytest.fixture
def epoch():
    rng = np.random.default_rng(0)
    return rng.standard_normal((len(CH_NAMES), 1000)).astype(np.float64) * 20.0


def test_psd_comparison_returns_figure(epoch):
    fig = plot_psd_comparison(epoch, CH_NAMES, SFREQ, channels=["FP1", "FP2"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_psd_comparison_with_all_signals(epoch):
    wiener = epoch * 0.8
    ica = epoch * 0.7
    fig = plot_psd_comparison(
        epoch, CH_NAMES, SFREQ,
        channels=["FP1", "FP2"],
        wiener_specific=wiener,
        ica_specific=ica,
    )
    assert isinstance(fig, plt.Figure)
    # 3-column grid (Raw | Wiener Specific | ICA Cleaned) × 2 channel rows = 6 axes
    assert len(fig.axes) == 6
    plt.close(fig)


def test_psd_comparison_channel_subset(epoch):
    fig = plot_psd_comparison(epoch, CH_NAMES, SFREQ, channels=["FP1"])
    # 3-column grid × 1 channel row = 3 axes
    assert len(fig.axes) == 3
    plt.close(fig)


def test_psd_comparison_default_channels_from_config():
    """Verify visualization config keys exist with correct types."""
    from eeg_bg.config.settings import load_config
    cfg = load_config("configs/default.yaml")
    assert cfg["visualization"]["psd_target_channels"] == ["FP1", "FP2"]
    assert isinstance(cfg["visualization"]["n_subjects"], int)
    assert isinstance(cfg["visualization"]["epoch_idx"], int)
