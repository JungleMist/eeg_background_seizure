import numpy as np
import pytest
import pandas as pd
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.verification.coherence import compute_pairwise_coherence, run_v1


def test_compute_pairwise_coherence_shape(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    sfreq = cfg["preprocessing"]["target_sfreq"]
    nperseg = cfg["wiener"]["nperseg"]
    coh = compute_pairwise_coherence(epoch, sfreq, nperseg,
                                      tuple(cfg["wiener"]["freq_band"]))
    assert coh.shape == (len(ch_names), len(ch_names))
    assert np.all(coh >= 0) and np.all(coh <= 1)
    np.testing.assert_allclose(coh, coh.T, atol=1e-10)


def test_run_v1_output_schema(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v1([result], cfg)
    assert isinstance(df, pd.DataFrame)
    required_cols = {"subject_id", "epoch_idx", "ch_i", "ch_j",
                     "coh_pre", "coh_post", "reduction"}
    assert required_cols.issubset(df.columns)
    assert (df["coh_pre"] >= 0).all() and (df["coh_pre"] <= 1).all()


def test_run_v1_reduction_positive_for_channel_groups(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v1([result], cfg)
    grouped_flat = [ch for group in cfg["channels"]["channel_groups"] for ch in group]
    grouped_df = df[df["ch_i"].isin(grouped_flat) & df["ch_j"].isin(grouped_flat)]
    if len(grouped_df):
        assert grouped_df["reduction"].mean() > 0
