import numpy as np
import pytest
from eeg_bg.decomposition.ica import fit_ica, apply_ica


def test_fit_ica_returns_model_and_indices(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, artifact_indices = fit_ica(epochs, ch_names, cfg)
    assert artifact_indices == [] or all(
        0 <= i < cfg["ica"]["n_components"] for i in artifact_indices
    )


def test_apply_ica_preserves_shape(synthetic_epochs_batch):
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, artifact_indices = fit_ica(epochs, ch_names, cfg)
    cleaned = apply_ica(epochs, ica_model, artifact_indices, ch_names, cfg)
    assert cleaned.shape == epochs.shape


def test_apply_ica_no_artifacts_unchanged(synthetic_epochs_batch):
    """With no artifact components removed, reconstruction should be near-identical."""
    epochs, ch_names, cfg = synthetic_epochs_batch
    ica_model, _ = fit_ica(epochs, ch_names, cfg)
    cleaned = apply_ica(epochs, ica_model, [], ch_names, cfg)
    rel_error = np.linalg.norm(cleaned - epochs) / np.linalg.norm(epochs)
    assert rel_error < 0.01
