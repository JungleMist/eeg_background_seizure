from pathlib import Path

import mne

from eeg_bg.application.models import OutputFormat
from eeg_bg.application.recording import RecordingService, recording_format


def test_recording_format_accepts_only_edf_and_fif():
    assert recording_format("a.edf") == "edf"
    assert recording_format("a.fif") == "fif"
    assert recording_format("a.fif.gz") == "fif"


def test_inspect_and_fif_atomic_roundtrip(synthetic_fif, tmp_path):
    service = RecordingService()
    info = service.inspect(synthetic_fif)
    assert info.format == "fif"
    assert info.n_times == 1000
    raw, _ = service.load_eeg(synthetic_fif)
    out = tmp_path / "processed.fif"
    service.write(raw, out, OutputFormat.FIF)
    reread = mne.io.read_raw_fif(out, preload=True, verbose=False)
    assert reread.n_times == raw.n_times
    assert reread.annotations.description.tolist() == ["marker"]
    assert not list(tmp_path.glob(".*.tmp*"))


def test_edf_atomic_roundtrip(synthetic_fif, tmp_path):
    service = RecordingService()
    raw, _ = service.load_eeg(synthetic_fif)
    out = tmp_path / "processed.edf"
    service.write(raw, out, OutputFormat.EDF)
    reread = mne.io.read_raw_edf(out, preload=True, verbose=False)
    assert reread.n_times == raw.n_times
    assert not list(tmp_path.glob(".*.tmp*"))
