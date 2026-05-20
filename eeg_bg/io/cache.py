import hashlib
import numpy as np
from pathlib import Path
from typing import Callable


def make_cache_key(edf_path: Path, start_sec: float, cfg: dict) -> str:
    sfreq = cfg["preprocessing"]["target_sfreq"]
    bandpass = cfg["preprocessing"]["bandpass"]
    raw = f"{edf_path}|{start_sec:.4f}|{sfreq}|{bandpass}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_or_compute(
    cache_path: Path,
    compute_fn: Callable[[], dict],
    force_recompute: bool = False,
) -> dict:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_recompute:
        return dict(np.load(cache_path, allow_pickle=True))
    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result
