import numpy as np
import pytest
from scipy.signal import coherence as scipy_coherence
from eeg_bg.decomposition.wiener import (
    CANDIDATE_PROCESSED,
    CANDIDATE_SOLVE_FAILED,
    WienerResult,
    WienerSolveError,
    estimate_cross_psd,
    compute_wiener_filter,
    apply_wiener_filter,
    decompose_epoch,
    decompose_epoch_with_fusion,
    decompose_subject,
)
from eeg_bg.decomposition.wiener_phasegated import (
    decompose_epoch as phasegated_decompose,
)
from eeg_bg.decomposition.wiener_scalar import (
    decompose_epoch as scalar_decompose,
)
from eeg_bg.decomposition.wiener_zerophase import (
    decompose_epoch as zerophase_decompose,
)


def _fusion_cfg(
    channel_groups,
    coherence_threshold=0.0,
    filter_magnitude_threshold=50.0,
):
    return {
        "preprocessing": {"target_sfreq": 125.0},
        "wiener": {
            "nperseg": 250,
            "coherence_threshold": coherence_threshold,
            "filter_magnitude_threshold": filter_magnitude_threshold,
            "phase_gate_threshold_rad": np.pi,
            "freq_band": [0.5, 40.0],
            "overlap_policy": "coherence_weighted",
        },
        "channels": {
            "channel_groups": channel_groups,
            "passthrough": [],
        },
    }


def _overlap_epoch():
    rng = np.random.default_rng(7)
    n_times = 1000
    left_source = rng.standard_normal(n_times)
    right_source = rng.standard_normal(n_times)
    return np.vstack([
        left_source + 0.05 * rng.standard_normal(n_times),
        0.8 * left_source + 1.2 * right_source
        + 0.05 * rng.standard_normal(n_times),
        right_source + 0.05 * rng.standard_normal(n_times),
    ]).astype(np.float64)


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


def test_compute_wiener_filter_raises_on_singular_system():
    S = np.zeros((3, 3, 1), dtype=complex)
    S[1:, 1:, 0] = np.ones((2, 2))

    with pytest.raises(WienerSolveError, match="frequency bin 0"):
        compute_wiener_filter(S, target_idx=0, reg_factor=0.0)


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


def test_wiener_filter_reconstructs_delayed_target():
    """A complex Wiener filter must preserve the transfer-function phase."""
    rng = np.random.default_rng(42)
    n_times = 1024
    reference = rng.standard_normal(n_times)
    reference -= reference.mean()
    target = np.roll(reference, 7)
    pair_data = np.vstack([target, reference])

    _, S = estimate_cross_psd(pair_data, sfreq=128.0, nperseg=n_times)
    h = compute_wiener_filter(S, target_idx=0, reg_factor=0.0)
    specific, coherent = apply_wiener_filter(
        pair_data,
        h,
        target_idx=0,
        n_times=n_times,
    )

    np.testing.assert_allclose(coherent, target, atol=1e-10)
    np.testing.assert_allclose(specific, np.zeros_like(target), atol=1e-10)


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


def test_decompose_epoch_passthrough_unchanged(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg)
    for ch in cfg["channels"]["passthrough"]:
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
    assert isinstance(result.channel_sources, dict)
    assert isinstance(result.channel_weights, dict)


def test_overlapping_channel_uses_weighted_fusion():
    ch_names = ["A", "B", "C"]
    epoch = _overlap_epoch()
    cfg = _fusion_cfg(
        [["A", "B"], ["B", "C"]],
        filter_magnitude_threshold=1e6,
    )

    result = decompose_epoch(epoch, ch_names, cfg)
    left_only = decompose_epoch(
        epoch,
        ch_names,
        _fusion_cfg([["A", "B"]], filter_magnitude_threshold=1e6),
    )
    right_only = decompose_epoch(
        epoch,
        ch_names,
        _fusion_cfg([["B", "C"]], filter_magnitude_threshold=1e6),
    )

    assert result.channel_sources["B"] == ["A-B", "B-C"]
    assert set(result.filters["A-B"]) >= {"B"}
    assert set(result.filters["B-C"]) >= {"B"}

    weights = result.channel_weights["B"]
    assert 0.0 < weights["A-B"] < 1.0
    assert 0.0 < weights["B-C"] < 1.0

    b_idx = ch_names.index("B")
    expected = (
        weights["A-B"] * left_only.coherent[b_idx]
        + weights["B-C"] * right_only.coherent[b_idx]
    )
    np.testing.assert_allclose(result.coherent[b_idx], expected, atol=1e-8)
    assert not np.allclose(result.coherent[b_idx], right_only.coherent[b_idx])

    assert result.candidate_fusion_weight is not None
    keys = list(result.candidate_keys or [])
    b_weights = [
        result.candidate_fusion_weight[i]
        for i, key in enumerate(keys)
        if key.endswith("::B")
    ]
    np.testing.assert_allclose(sum(b_weights), 1.0, atol=1e-12)


def test_overlapping_channel_fusion_is_group_order_independent():
    ch_names = ["A", "B", "C"]
    epoch = _overlap_epoch()

    forward = decompose_epoch(
        epoch,
        ch_names,
        _fusion_cfg(
            [["A", "B"], ["B", "C"]],
            filter_magnitude_threshold=1e6,
        ),
    )
    reversed_order = decompose_epoch(
        epoch,
        ch_names,
        _fusion_cfg(
            [["B", "C"], ["A", "B"]],
            filter_magnitude_threshold=1e6,
        ),
    )

    np.testing.assert_allclose(forward.coherent, reversed_order.coherent)
    np.testing.assert_allclose(forward.specific, reversed_order.specific)
    assert forward.channel_weights == reversed_order.channel_weights


def test_target_level_gate_does_not_process_low_coherence_channel():
    sfreq = 125.0
    t = np.arange(1000) / sfreq
    shared = np.sin(2 * np.pi * 10.0 * t)
    rng = np.random.default_rng(123)
    epoch = np.vstack([
        shared,
        shared,
        rng.standard_normal(t.shape[0]),
    ]).astype(np.float64)
    ch_names = ["A", "B", "C"]
    cfg = _fusion_cfg([["A", "B", "C"]], coherence_threshold=0.999)
    cfg["wiener"]["freq_band"] = [8.0, 12.0]

    result = decompose_epoch(epoch, ch_names, cfg)

    c_idx = ch_names.index("C")
    np.testing.assert_array_equal(result.coherent[c_idx], np.zeros_like(epoch[c_idx]))
    np.testing.assert_array_equal(result.specific[c_idx], epoch[c_idx])
    assert "C" not in result.filters["A-B-C"]
    assert "A" in result.filters["A-B-C"]
    assert "B" in result.filters["A-B-C"]


def test_unstable_filter_skips_only_that_target_candidate():
    n_times = 1000
    t = np.arange(n_times) / 125.0
    epoch = np.vstack([
        np.sin(2 * np.pi * 10.0 * t),
        np.cos(2 * np.pi * 10.0 * t),
    ]).astype(np.float64)
    ch_names = ["A", "B"]
    cfg = _fusion_cfg([["A", "B"]])

    def candidate_fn(group_data, S, target_idx, freq_mask, n_times):
        if target_idx == 1:
            return np.array([[100.0]]), np.ones(n_times), {}
        return np.array([[0.5]]), np.ones(n_times), {}

    result = decompose_epoch_with_fusion(
        epoch,
        ch_names,
        cfg,
        candidate_fn=candidate_fn,
    )

    assert set(result.filters["A-B"]) == {"A"}
    np.testing.assert_allclose(result.coherent[0], np.ones(n_times))
    np.testing.assert_array_equal(result.coherent[1], np.zeros(n_times))
    np.testing.assert_allclose(
        result.specific + result.coherent,
        result.raw,
        atol=1e-12,
    )


def test_solve_failure_is_reported_and_only_skips_that_candidate():
    epoch = _overlap_epoch()[:2]
    ch_names = ["A", "B"]
    cfg = _fusion_cfg([["A", "B"]])

    def candidate_fn(group_data, S, target_idx, freq_mask, n_times):
        if target_idx == 1:
            raise WienerSolveError("synthetic solve failure")
        return np.array([[0.5]]), np.ones(n_times), {}

    result = decompose_epoch_with_fusion(
        epoch,
        ch_names,
        cfg,
        candidate_fn=candidate_fn,
    )

    status_by_key = dict(zip(result.candidate_keys, result.candidate_status))
    assert status_by_key["A-B::A"] == CANDIDATE_PROCESSED
    assert status_by_key["A-B::B"] == CANDIDATE_SOLVE_FAILED
    assert set(result.filters["A-B"]) == {"A"}
    np.testing.assert_array_equal(result.coherent[1], np.zeros_like(epoch[1]))
    np.testing.assert_array_equal(result.specific[1], epoch[1])


@pytest.mark.parametrize(
    "decomposer",
    [decompose_epoch, phasegated_decompose, zerophase_decompose, scalar_decompose],
)
def test_all_wiener_modes_preserve_raw_identity(synthetic_epoch, decomposer):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decomposer(epoch, ch_names, cfg)
    np.testing.assert_allclose(result.specific + result.coherent, result.raw)


def test_decompose_subject_returns_list(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    results = decompose_subject(epochs, ch_names, "test_subject", cfg)
    assert len(results) == epochs.shape[0]
    assert all(isinstance(r, WienerResult) for r in results)
