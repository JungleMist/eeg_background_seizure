import numpy as np
import pytest
from eeg_bg.decomposition.wiener import WienerResult, decompose_epoch as freq_decompose
from eeg_bg.decomposition.wiener_scalar import decompose_epoch as scalar_decompose


def test_scalar_returns_wiener_result(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = scalar_decompose(epoch, ch_names, cfg)
    assert isinstance(result, WienerResult)
    assert result.specific.shape == epoch.shape


def test_scalar_specific_plus_coherent_equals_raw(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = scalar_decompose(epoch, ch_names, cfg)
    np.testing.assert_allclose(
        result.specific + result.coherent, result.raw, atol=1e-8
    )


def test_scalar_residual_coherence_higher_than_freq(synthetic_epoch):
    """Frequency-dependent mode should leave lower or equal residual coherence."""
    from scipy.signal import coherence as scipy_coherence
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]

    freq_result = freq_decompose(epoch, ch_names, cfg)
    scalar_result = scalar_decompose(epoch, ch_names, cfg)

    fp1_idx = ch_names.index("FP1")
    fp2_idx = ch_names.index("FP2")

    _, coh_freq = scipy_coherence(
        freq_result.specific[fp1_idx], freq_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    _, coh_scalar = scipy_coherence(
        scalar_result.specific[fp1_idx], scalar_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    assert np.mean(coh_scalar) >= np.mean(coh_freq) - 0.05
