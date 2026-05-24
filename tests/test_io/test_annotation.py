import pytest
from pathlib import Path
from eeg_bg.io.annotation import extract_bckg_intervals

CSV_BI_NO_SEIZ = """\
# version = csv_v1.0.0
# bname = test
# duration = 100 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,100.0000,bckg,1.0000
"""

CSV_BI_WITH_SEIZ = """\
# version = csv_v1.0.0
# bname = test
# duration = 120 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,50.0000,bckg,1.0000
TERM,50.0000,60.0000,seiz,1.0000
TERM,60.0000,120.0000,bckg,1.0000
"""

# Mimics TUEP v3.1.0 control subjects: 1190-second recording, only ~2s annotated.
CSV_BI_TUEP_CONTROL = """\
# version = csv_v1.0.0
# bname = test_control
# duration = 1190 secs
# montage_file = 01_tcp_ar_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,2.0020,bckg,1.0000
"""

CFG_30 = {"preprocessing": {"seizure_buffer_sec": 30.0}}
CFG_5 = {"preprocessing": {"seizure_buffer_sec": 5.0}}

# Keep the old name so any external references still work.
CFG = CFG_30


# ── Legacy path (recording_duration=None) ─────────────────────────────────────

def test_no_seizure_returns_full_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_NO_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG_30)
    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.0, 100.0))


def test_seiz_buffer_clips_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_WITH_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG_30)
    # seiz [50,60], buffer 30s → exclude [20,90]
    # bckg [0,50] → clip to [0,20]
    # bckg [60,120] → clip to [90,120]
    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((0.0, 20.0))
    assert intervals[1] == pytest.approx((90.0, 120.0))


def test_seiz_fully_covering_bckg_returns_empty(tmp_path):
    content = """\
# version = csv_v1.0.0
# bname = test
# duration = 30 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,0.0000,10.0000,bckg,1.0000
TERM,10.0000,20.0000,seiz,1.0000
TERM,20.0000,30.0000,bckg,1.0000
"""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(content)
    intervals = extract_bckg_intervals(csv_path, CFG_30)
    # seiz [10,20], buffer 30 → exclude [-20,50] → covers all bckg
    assert intervals == []


# ── New path (recording_duration provided) ────────────────────────────────────

def test_full_recording_returned_when_no_seiz(tmp_path):
    """No seizure rows → entire recording is background."""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_NO_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG_30, recording_duration=100.0)
    assert intervals == pytest.approx([(0.0, 100.0)])


def test_seiz_subtracted_from_full_recording(tmp_path):
    """Seizure interval is cut from the full-recording base."""
    content = """\
# version = csv_v1.0.0
# bname = test
# duration = 100 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,20.0000,30.0000,seiz,1.0000
"""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(content)
    # seiz [20,30], buffer 5 → exclude [15,35]
    intervals = extract_bckg_intervals(csv_path, CFG_5, recording_duration=100.0)
    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((0.0, 15.0))
    assert intervals[1] == pytest.approx((35.0, 100.0))


def test_tuep_control_pattern_uses_full_recording(tmp_path):
    """TUEP control subjects have only ~2 s annotated; full recording must be returned."""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_TUEP_CONTROL)
    intervals = extract_bckg_intervals(csv_path, CFG_30, recording_duration=1190.0)
    assert intervals == pytest.approx([(0.0, 1190.0)])


def test_all_recording_excluded_by_seiz_with_duration(tmp_path):
    """Single large seizure that, after buffering, covers the whole recording."""
    content = """\
# version = csv_v1.0.0
# bname = test
# duration = 30 secs
# montage_file = test
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,20.0000,seiz,1.0000
"""
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(content)
    # seiz [10,20], buffer 30 → exclude [0,50] → covers full 30s recording
    intervals = extract_bckg_intervals(csv_path, CFG_30, recording_duration=30.0)
    assert intervals == []
