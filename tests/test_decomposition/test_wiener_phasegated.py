import numpy as np

from eeg_bg.decomposition.wiener import (
    WienerResult,
    decompose_epoch as freq_decompose,
)
from eeg_bg.decomposition.wiener_phasegated import (
    compute_phasegated_filter,
    decompose_epoch as phasegated_decompose,
    decompose_subject as phasegated_decompose_subject,
)


def _two_channel_spectrum(cross_value: complex) -> np.ndarray:
    S = np.zeros((2, 2, 1), dtype=complex)
    S[0, 0, 0] = 1.0
    S[1, 1, 0] = 1.0
    S[0, 1, 0] = cross_value
    S[1, 0, 0] = np.conj(cross_value)
    return S


def test_phasegated_returns_wiener_result(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = phasegated_decompose(epoch, ch_names, cfg)
    assert isinstance(result, WienerResult)
    assert result.specific.shape == epoch.shape


def test_phasegated_specific_plus_coherent_equals_raw(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = phasegated_decompose(epoch, ch_names, cfg)
    np.testing.assert_allclose(
        result.specific + result.coherent, result.raw, atol=1e-8
    )


def test_phasegated_decompose_subject_returns_list(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    results = phasegated_decompose_subject(epochs, ch_names, "subj01", cfg)
    assert len(results) == epochs.shape[0]
    assert all(isinstance(r, WienerResult) for r in results)


def test_phasegated_threshold_pi_matches_frequency(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    cfg["wiener"]["phase_gate_threshold_rad"] = np.pi

    freq_result = freq_decompose(epoch, ch_names, cfg)
    phasegated_result = phasegated_decompose(epoch, ch_names, cfg)

    np.testing.assert_allclose(
        phasegated_result.specific, freq_result.specific, atol=1e-8
    )
    np.testing.assert_allclose(
        phasegated_result.coherent, freq_result.coherent, atol=1e-8
    )


def test_phasegated_blocks_antiphase_filter_below_pi_threshold():
    S = _two_channel_spectrum(-1.0 + 0.0j)

    h_low = compute_phasegated_filter(
        S,
        target_idx=0,
        phase_gate_threshold_rad=0.392,
        reg_factor=0.0,
    )
    h_all = compute_phasegated_filter(
        S,
        target_idx=0,
        phase_gate_threshold_rad=np.pi,
        reg_factor=0.0,
    )

    np.testing.assert_array_equal(h_low, np.zeros((1, 1), dtype=complex))
    np.testing.assert_allclose(h_all, np.array([[-1.0 + 0.0j]]))
