import pandas as pd
import pytest
from eeg_bg.preprocessing.reference import detect_reference, filter_by_reference


def test_detect_reference_ar():
    assert detect_reference("01_tcp_ar") == "ar"
    assert detect_reference("03_tcp_ar_a") == "ar"


def test_detect_reference_le():
    assert detect_reference("02_tcp_le") == "le"
    assert detect_reference("04_tcp_le_a") == "le"


def test_filter_by_reference_keeps_matching():
    df = pd.DataFrame({
        "subject_id": ["s1", "s2", "s3"],
        "reference": ["ar", "le", "ar"],
    })
    result = filter_by_reference(df, "ar")
    assert list(result["subject_id"]) == ["s1", "s3"]
    assert result.index.tolist() == [0, 1]  # reset index
