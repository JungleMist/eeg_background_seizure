from pathlib import Path

import mne
import pytest

from eeg_bg.application.models import OutputFormat
from eeg_bg.application.recording import RecordingService, recording_format


def test_recording_format_accepts_edf_fif_and_eeglab_set():
    assert recording_format("a.edf") == "edf"
    assert recording_format("a.fif") == "fif"
    assert recording_format("a.fif.gz") == "fif"
    assert recording_format("a.set") == "set"
    with pytest.raises(ValueError, match="不支持"):
        recording_format("a.fdt")


def test_eeglab_set_uses_eeglab_reader_and_marks_eog(monkeypatch, tmp_path):
    path = tmp_path / "subject.set"
    path.write_bytes(b"placeholder")
    raw = mne.io.RawArray(
        [[0.0] * 100, [0.0] * 100],
        mne.create_info(["FP1", "HEOG_left"], 100.0, ch_types="eeg"),
        verbose=False,
    )
    calls = []

    def fake_read_raw_eeglab(source, *, preload, verbose):
        calls.append((source, preload, verbose))
        return raw.copy()

    monkeypatch.setattr(mne.io, "read_raw_eeglab", fake_read_raw_eeglab)
    info = RecordingService().inspect(path)

    assert calls == [(str(path), False, False)]
    assert info.format == "set"
    assert info.ch_names == ["FP1"]


def test_eeglab_set_reports_missing_fdt(monkeypatch, tmp_path):
    path = tmp_path / "subject.set"
    path.write_bytes(b"placeholder")

    def missing_sidecar(*args, **kwargs):
        raise FileNotFoundError("subject.fdt")

    monkeypatch.setattr(mne.io, "read_raw_eeglab", missing_sidecar)
    with pytest.raises(FileNotFoundError, match=r"\.fdt 数据文件缺失"):
        RecordingService().open_raw(path)


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
