import json

from eeg_bg.application.sidecars import read_recording_sidecars


def test_read_recording_sidecars_loads_bids_metadata_and_events(tmp_path):
    recording = tmp_path / "sub-001_task-ERN_eeg.set"
    recording.write_bytes(b"placeholder")
    (tmp_path / "sub-001_task-ERN_eeg.json").write_text(
        json.dumps({"TaskName": "ERN", "SamplingFrequency": 1024}),
        encoding="utf-8",
    )
    (tmp_path / "sub-001_task-ERN_channels.tsv").write_text(
        "name\ttype\tunits\nFCz\tEEG\tmicroV\nVEOG\tEOG\tmicroV\n",
        encoding="utf-8",
    )
    (tmp_path / "sub-001_task-ERN_events.tsv").write_text(
        "onset\tduration\tsample\ttrial_type\tvalue\n"
        "2.5\t0.2\t2561\tstimulus\t11\n"
        "2.9\t0.0\t2970\tresponse\t111\n",
        encoding="utf-8",
    )
    (tmp_path / "sub-001_task-ERN_electrodes.tsv").write_text(
        "name\tx\ty\tz\nFCz\t1\t2\t3\n",
        encoding="utf-8",
    )
    (tmp_path / "sub-001_task-ERN_coordsystem.json").write_text(
        json.dumps({"EEGCoordinateSystem": "ARS"}), encoding="utf-8"
    )

    sidecars, warnings = read_recording_sidecars(recording)

    assert warnings == []
    assert sidecars.eeg["TaskName"] == "ERN"
    assert sidecars.channels[1]["type"] == "EOG"
    assert sidecars.electrodes[0]["name"] == "FCz"
    assert sidecars.coordsystem["EEGCoordinateSystem"] == "ARS"
    assert [event.onset_sec for event in sidecars.events] == [2.5, 2.9]
    assert sidecars.events[0].trial_type == "stimulus"
    assert sidecars.events[0].value == "11"
    assert sidecars.events[0].sample == 2561
    assert set(sidecars.paths) == {
        "eeg", "channels", "events", "electrodes", "coordsystem"
    }


def test_read_recording_sidecars_skips_invalid_event_onset(tmp_path):
    recording = tmp_path / "subject_eeg.set"
    recording.write_bytes(b"placeholder")
    (tmp_path / "subject_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\n"
        "bad\t0\tstimulus\t11\n"
        "1.25\tn/a\tresponse\t111\n",
        encoding="utf-8",
    )

    sidecars, warnings = read_recording_sidecars(recording)

    assert len(sidecars.events) == 1
    assert sidecars.events[0].duration_sec == 0.0
    assert warnings == ["events.tsv 中有 1 行缺少有效 onset，已跳过"]


def test_non_set_recording_has_no_sidecars(tmp_path):
    sidecars, warnings = read_recording_sidecars(tmp_path / "recording.fif")
    assert sidecars.events == []
    assert sidecars.paths == {}
    assert warnings == []
