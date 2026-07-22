import mne
import numpy as np

from eeg_bg.application.artifacts import artifact_mask, summarize_raw_artifacts
from eeg_bg.application.models import ArtifactSettings


def test_artifact_mask_is_per_channel_and_strictly_greater():
    data_uv = np.array([[0.0, 200.0, 201.0], [-250.0, 0.0, 0.0]])
    assert artifact_mask(data_uv, 200.0).tolist() == [
        [False, False, True],
        [True, False, False],
    ]


def test_artifact_summary_counts_contiguous_regions_per_channel():
    data_uv = np.array([
        [0.0, 250.0, 260.0, 0.0, -300.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [220.0, 0.0, 230.0, 240.0, 0.0],
    ])
    raw = mne.io.RawArray(
        data_uv * 1e-6,
        mne.create_info(["FP1", "FP2", "F3"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    summary = summarize_raw_artifacts(raw, ArtifactSettings(threshold_uv=200.0))
    assert summary["affected_channels"] == ["FP1", "F3"]
    assert summary["channel_region_counts"] == {"FP1": 2, "F3": 2}
    assert summary["exceedance_region_count"] == 4
    assert summary["max_abs_uv"] == 300.0
