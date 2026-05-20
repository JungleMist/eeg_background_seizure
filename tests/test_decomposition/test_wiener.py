import numpy as np
import pytest
from scipy.signal import coherence as scipy_coherence
from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
    apply_wiener_filter,
    decompose_epoch,
    decompose_subject,
)


def test_estimate_cross_psd_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    freqs, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    assert freqs.shape == (nperseg // 2 + 1,)
    assert S.shape == (2, 2, nperseg // 2 + 1)
    assert np.all(S[0, 0].real > 0)
    assert np.allclose(S[0, 0].imag, 0, atol=1e-10)
    np.testing.assert_allclose(S[0, 1], np.conj(S[1, 0]))


def test_compute_wiener_filter_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h = compute_wiener_filter(S, target_idx=0)
    assert h.shape == (1, nperseg // 2 + 1)
    assert h.dtype == complex


def test_apply_wiener_filter_output_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    pair_data = epoch[:2]
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h = compute_wiener_filter(S, target_idx=0)
    specific, coherent = apply_wiener_filter(pair_data, h, target_idx=0, n_times=pair_data.shape[1])
    assert specific.shape == (pair_data.shape[1],)
    assert coherent.shape == (pair_data.shape[1],)
    np.testing.assert_allclose(specific + coherent, pair_data[0], atol=1e-8)


def test_decompose_epoch_reduces_coherence(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    fp1_idx = ch_names.index("FP1")
    fp2_idx = ch_names.index("FP2")
    _, coh_pre = scipy_coherence(epoch[fp1_idx], epoch[fp2_idx], fs=sfreq, nperseg=nperseg)
    _, coh_post = scipy_coherence(result.specific[fp1_idx], result.specific[fp2_idx], fs=sfreq, nperseg=nperseg)
    assert np.mean(coh_pre) > np.mean(coh_post)


def test_decompose_epoch_midline_unchanged(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    for ch in cfg["channels"]["midline"]:
        idx = ch_names.index(ch)
        np.testing.assert_array_equal(result.specific[idx], epoch[idx])


def test_decompose_epoch_result_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    assert result.raw.shape == epoch.shape
    assert result.specific.shape == epoch.shape
    assert result.coherent.shape == epoch.shape
    assert isinstance(result.filters, dict)
    assert isinstance(result.skipped_pairs, list)


def test_decompose_subject_returns_list(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    results = decompose_subject(epochs, ch_names, "test_subject", cfg)
    assert len(results) == epochs.shape[0]
    assert all(isinstance(r, WienerResult) for r in results)
