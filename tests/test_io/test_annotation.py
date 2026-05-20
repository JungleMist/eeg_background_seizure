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

CFG = {"preprocessing": {"seizure_buffer_sec": 30.0}}


def test_no_seizure_returns_full_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_NO_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG)
    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.0, 100.0))


def test_seiz_buffer_clips_bckg(tmp_path):
    csv_path = tmp_path / "test.csv_bi"
    csv_path.write_text(CSV_BI_WITH_SEIZ)
    intervals = extract_bckg_intervals(csv_path, CFG)
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
    intervals = extract_bckg_intervals(csv_path, CFG)
    # seiz [10,20], buffer 30 → exclude [-20,50] → covers all bckg
    assert intervals == []
