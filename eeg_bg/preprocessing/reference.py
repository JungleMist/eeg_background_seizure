import pandas as pd


def detect_reference(montage_dir: str) -> str:
    lower = montage_dir.lower()
    if "tcp_ar" in lower:
        return "ar"
    if "tcp_le" in lower:
        return "le"
    return "unknown"


def filter_by_reference(index: pd.DataFrame, scheme: str) -> pd.DataFrame:
    return index[index["reference"] == scheme].reset_index(drop=True)
