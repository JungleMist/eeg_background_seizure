import numpy as np
import pytest
import pandas as pd
from eeg_bg.decomposition.wiener import decompose_epoch
from eeg_bg.verification.transitivity import run_v2, run_v3


def test_run_v2_error_nonnegative(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v2([result], cfg)
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert (df["eps_amp"] >= 0).all()
        assert (df["eps_phase"] >= 0).all()


def test_run_v3_variation_nonnegative(synthetic_epoch):
    epoch, ch_names, cfg, *_ = synthetic_epoch
    result = decompose_epoch(epoch, ch_names, cfg, subject_id="s1")
    df = run_v3([result], cfg)
    assert isinstance(df, pd.DataFrame)
    if len(df):
        assert (df["freq_variation"] >= 0).all()
