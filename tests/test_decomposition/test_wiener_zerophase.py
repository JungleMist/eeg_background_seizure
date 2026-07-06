import numpy as np
from eeg_bg.decomposition.wiener import (
    WienerResult,
    estimate_cross_psd,
    compute_wiener_filter,
    decompose_epoch as freq_decompose,
)
from eeg_bg.decomposition.wiener_zerophase import (
    compute_zerophase_filter,
    decompose_epoch as zerophase_decompose,
    decompose_subject as zerophase_decompose_subject,
)


def test_zerophase_returns_wiener_result(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = zerophase_decompose(epoch, ch_names, cfg)
    assert isinstance(result, WienerResult)
    assert result.specific.shape == epoch.shape


def test_zerophase_specific_plus_coherent_equals_raw(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = zerophase_decompose(epoch, ch_names, cfg)
    np.testing.assert_allclose(
        result.specific + result.coherent, result.raw, atol=1e-8
    )


def test_zerophase_decompose_subject_returns_list(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    results = zerophase_decompose_subject(epochs, ch_names, "subj01", cfg)
    assert len(results) == epochs.shape[0]
    assert all(isinstance(r, WienerResult) for r in results)


def test_zerophase_filter_is_real(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    fp1_idx, fp2_idx = ch_names.index("FP1"), ch_names.index("FP2")
    pair_data = epoch[[fp1_idx, fp2_idx]]

    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h = compute_zerophase_filter(S, target_idx=0)
    assert np.isrealobj(h)
    assert h.shape == (1, nperseg // 2 + 1)


def test_zerophase_matches_real_part_for_single_reference(synthetic_epoch):
    """For a 2-channel (single-reference) group, the real-constrained solve
    is mathematically identical to Re(h_complex), since both reduce to
    Re(S_ij)/S_jj (S_jj is already real). Not true in general for
    2-reference (3-channel chain) groups.
    """
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    fp1_idx, fp2_idx = ch_names.index("FP1"), ch_names.index("FP2")
    pair_data = epoch[[fp1_idx, fp2_idx]]

    _, S = estimate_cross_psd(pair_data, sfreq, nperseg)
    h_complex = compute_wiener_filter(S, target_idx=0)
    h_real = compute_zerophase_filter(S, target_idx=0)

    np.testing.assert_allclose(h_real, h_complex.real, atol=1e-10)


def test_zerophase_passthrough_unchanged(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = zerophase_decompose(epoch, ch_names, cfg)
    for ch in cfg["channels"]["passthrough"]:
        idx = ch_names.index(ch)
        np.testing.assert_array_equal(result.specific[idx], epoch[idx])
        np.testing.assert_array_equal(result.coherent[idx], np.zeros_like(epoch[idx]))


def test_zerophase_residual_coherence_close_to_freq(synthetic_epoch):
    """The synthetic_epoch fixture is an exactly zero-phase (real scalar-gain,
    no delay/filtering) mixture, so zerophase should decompose it about as
    well as the unconstrained frequency mode -- there's no genuine phase
    content to lose here.
    """
    from scipy.signal import coherence as scipy_coherence
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]

    freq_result = freq_decompose(epoch, ch_names, cfg)
    zerophase_result = zerophase_decompose(epoch, ch_names, cfg)

    fp1_idx, fp2_idx = ch_names.index("FP1"), ch_names.index("FP2")

    _, coh_freq = scipy_coherence(
        freq_result.specific[fp1_idx], freq_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    _, coh_zerophase = scipy_coherence(
        zerophase_result.specific[fp1_idx], zerophase_result.specific[fp2_idx],
        fs=sfreq, nperseg=nperseg
    )
    assert np.mean(coh_zerophase) <= np.mean(coh_freq) + 0.05
