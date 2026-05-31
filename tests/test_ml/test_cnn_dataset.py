"""Unit tests for EEGEpochDataset."""
import numpy as np
import pytest
import torch

from eeg_bg.ml.cnn_dataset import EEGEpochDataset


@pytest.fixture()
def fake_cache(tmp_path):
    """Create a minimal epoch cache with 2 subjects × 3 epochs."""
    rng = np.random.default_rng(0)

    # epilepsy subject in train split
    subdir0 = tmp_path / "epochs" / "00_subj0"
    subdir0.mkdir(parents=True)
    np.savez(
        subdir0 / "abc123.npz",
        epochs=rng.standard_normal((3, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj0",
        split="train",
    )

    # control subject in train split
    subdir1 = tmp_path / "epochs" / "01_subj1"
    subdir1.mkdir(parents=True)
    np.savez(
        subdir1 / "def456.npz",
        epochs=rng.standard_normal((3, 19, 1000)).astype(np.float32),
        label=np.int64(1),
        subject_id="01_subj1",
        split="train",
    )

    # subject in val split — must be excluded from train dataset
    subdir2 = tmp_path / "epochs" / "00_subj2"
    subdir2.mkdir(parents=True)
    np.savez(
        subdir2 / "ghi789.npz",
        epochs=rng.standard_normal((2, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj2",
        split="val",
    )

    return tmp_path


def test_dataset_length(fake_cache):
    """Dataset length equals total epochs across matching subjects."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    assert len(ds) == 6  # 3 + 3 epochs from two train subjects


def test_dataset_excludes_other_splits(fake_cache):
    """Val-split subject is not included in the train dataset."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    subject_ids = [ds[i][2] for i in range(len(ds))]
    assert "00_subj2" not in subject_ids


def test_epoch_tensor_shape(fake_cache):
    """Each item returns an epoch tensor of shape (1, 19, 1000)."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    epoch_tensor, label, subject_id = ds[0]
    assert epoch_tensor.shape == (1, 19, 1000)
    assert epoch_tensor.dtype == torch.float32


def test_epoch_is_z_scored(fake_cache):
    """Each channel of the returned tensor is approximately zero-mean, unit-variance."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    epoch_tensor, _, _ = ds[0]  # shape (1, 19, 1000)
    channel_means = epoch_tensor[0].mean(dim=-1)   # (19,)
    channel_stds  = epoch_tensor[0].std(dim=-1)    # (19,)
    assert torch.all(channel_means.abs() < 1e-5), "Channels not zero-mean after z-score"
    assert torch.all((channel_stds - 1.0).abs() < 0.01), "Channels not unit-var after z-score"


def test_label_values(fake_cache):
    """Labels are 0 or 1 integers."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    labels = {ds[i][1] for i in range(len(ds))}
    assert labels == {0, 1}


def test_subject_id_is_string(fake_cache):
    """subject_id field is a string."""
    ds = EEGEpochDataset(cache_root=fake_cache, condition="raw", split="train")
    _, _, subject_id = ds[0]
    assert isinstance(subject_id, str)


def test_wiener_condition_uses_specific_key(tmp_path):
    """Wiener condition reads 'specific' array key from wiener_frequency subdir."""
    rng = np.random.default_rng(1)
    subdir = tmp_path / "wiener_frequency" / "00_subj0"
    subdir.mkdir(parents=True)
    np.savez(
        subdir / "key.npz",
        specific=rng.standard_normal((2, 19, 1000)).astype(np.float32),
        label=np.int64(0),
        subject_id="00_subj0",
        split="train",
    )
    ds = EEGEpochDataset(cache_root=tmp_path, condition="wiener", split="train")
    assert len(ds) == 2
    epoch_tensor, label, sid = ds[0]
    assert epoch_tensor.shape == (1, 19, 1000)
