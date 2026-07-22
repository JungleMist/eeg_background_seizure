"""PySide6 desktop interface for ECMAD Studio."""

import os
from pathlib import Path
import tempfile


# Keep the frozen desktop runtime independent of Numba/llvmlite. The Wiener
# implementation uses NumPy/SciPy directly, and MNE treats Numba as optional.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
_mpl_cache = Path(tempfile.gettempdir()) / "eeg_bg_studio_mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))
