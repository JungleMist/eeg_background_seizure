"""Unit tests for eeg_bg.features.extraction."""
import numpy as np
import pytest

from eeg_bg.features.extraction import (
    extract_epoch_features,
    build_dataset,
    FEATURE_NAMES,
    _STANDARD_19,
)
from eeg_bg.features.band_power import BANDS


# ── FEATURE_NAMES sanity checks ───────────────────────────────────────────────

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 211


def test_feature_names_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_feature_names_band_power_count():
    """Exactly 19 × 5 = 95 names should contain '_power'."""
    power_names = [n for n in FEATURE_NAMES if "_power" in n]
    assert len(power_names) == 95


def test_feature_names_per_channel():
    """Each channel should have exactly 9 feature names."""
    for ch in _STANDARD_19:
        ch_feats = [n for n in FEATURE_NAMES if n.startswith(f"{ch}_")]
        assert len(ch_feats) == 9, f"Expected 9 features for {ch}, got {len(ch_feats)}"


# ── extract_epoch_features ────────────────────────────────────────────────────

def test_extract_epoch_features_shape(synthetic_epoch, ch_names_19, sfreq):
    feat = extract_epoch_features(synthetic_epoch, ch_names_19, sfreq=sfreq)
    assert feat.shape == (211,)


def test_extract_epoch_features_dtype(synthetic_epoch, ch_names_19, sfreq):
    feat = extract_epoch_features(synthetic_epoch, ch_names_19, sfreq=sfreq)
    assert feat.dtype == np.float64


def test_extract_epoch_features_finite(synthetic_epoch, ch_names_19, sfreq):
    feat = extract_epoch_features(synthetic_epoch, ch_names_19, sfreq=sfreq)
    assert np.all(np.isfinite(feat))


def test_extract_epoch_features_missing_channel(sfreq):
    """Epoch with only 1 of 19 channels — missing channels padded with zeros."""
    rng    = np.random.default_rng(7)
    epoch  = rng.standard_normal((1, 1000)).astype(np.float64) * 10.0
    feat   = extract_epoch_features(epoch, ch_names=["FP1"], sfreq=sfreq)
    assert feat.shape == (211,)
    # Per-channel features for channels other than FP1, and all asymmetry
    # features (asym_* prefix), should be zero-padded because FP2 and all
    # other symmetric-pair partners are absent.
    fp1_indices = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("FP1_")]
    other_indices = [i for i in range(211) if i not in fp1_indices]
    assert np.all(feat[other_indices] == 0.0)


# ── build_dataset ─────────────────────────────────────────────────────────────

def _write_fake_npz(path, epochs_arr, split_val, label_val, subject_id):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        epochs=epochs_arr,
        ch_names=np.array(["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
                            "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
                            "Fz", "Cz", "Pz"], dtype=object),
        label=label_val,
        subject_id=subject_id,
        split=split_val,
    )


def test_build_dataset_shapes(tmp_path):
    """build_dataset returns correct shapes for train split."""
    rng = np.random.default_rng(1)
    ep_arr = rng.standard_normal((3, 19, 1000)).astype(np.float64)
    _write_fake_npz(tmp_path / "epochs" / "s001" / "ep.npz",
                    ep_arr, "train", 0, "s001")

    X, y, sids = build_dataset(tmp_path, "raw", "train",
                                sfreq=125.0, nperseg=250,
                                freq_band=(0.5, 40.0))
    assert X.shape == (3, 211)
    assert y.shape == (3,)
    assert len(sids) == 3


def test_build_dataset_respects_split(tmp_path):
    """Files labelled 'test' are excluded when split='train' is requested."""
    rng = np.random.default_rng(2)
    ep_arr = rng.standard_normal((2, 19, 1000)).astype(np.float64)
    _write_fake_npz(tmp_path / "epochs" / "s001" / "train.npz",
                    ep_arr, "train", 0, "s001")
    _write_fake_npz(tmp_path / "epochs" / "s002" / "test.npz",
                    ep_arr, "test",  1, "s002")

    X, y, sids = build_dataset(tmp_path, "raw", "train",
                                sfreq=125.0, nperseg=250,
                                freq_band=(0.5, 40.0))
    assert X.shape[0] == 2
    assert set(sids) == {"s001"}


def test_build_dataset_label_consistency(tmp_path):
    """All epochs from a single NPZ share the same label."""
    rng = np.random.default_rng(3)
    ep_arr = rng.standard_normal((4, 19, 1000)).astype(np.float64)
    _write_fake_npz(tmp_path / "epochs" / "s001" / "ep.npz",
                    ep_arr, "train", 0, "s001")

    X, y, sids = build_dataset(tmp_path, "raw", "train",
                                sfreq=125.0, nperseg=250,
                                freq_band=(0.5, 40.0))
    assert np.all(y == 0)


def test_build_dataset_empty_split(tmp_path):
    """Returns empty arrays when no files match the requested split."""
    rng = np.random.default_rng(4)
    ep_arr = rng.standard_normal((2, 19, 1000)).astype(np.float64)
    _write_fake_npz(tmp_path / "epochs" / "s001" / "ep.npz",
                    ep_arr, "test", 0, "s001")

    X, y, sids = build_dataset(tmp_path, "raw", "train",
                                sfreq=125.0, nperseg=250,
                                freq_band=(0.5, 40.0))
    assert X.shape == (0, 211)
    assert y.shape == (0,)
    assert sids == []


def test_build_dataset_unknown_condition_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown condition"):
        build_dataset(tmp_path, "bogus", "train",
                      sfreq=125.0, nperseg=250, freq_band=(0.5, 40.0))
