"""Hjorth complexity parameters for a single EEG channel.

Reference: Hjorth, B. (1970). EEG analysis based on time domain properties.
Electroencephalography and Clinical Neurophysiology, 29(3), 306–310.
"""
from __future__ import annotations

import numpy as np


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth activity, mobility, and complexity.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-series, shape ``(n_times,)``.

    Returns
    -------
    tuple[float, float, float]
        ``(activity, mobility, complexity)`` where

        * **activity** = ``var(x)``
        * **mobility** = ``sqrt(var(x') / (var(x) + eps))``
        * **complexity** = ``mobility(x') / (mobility(x) + eps)``

        The ``+ eps`` guards against division by zero for flat signals.
    """
    diff1 = np.diff(signal.astype(np.float64))
    diff2 = np.diff(diff1)

    var_x  = float(np.var(signal))
    var_d1 = float(np.var(diff1))
    var_d2 = float(np.var(diff2))

    activity   = var_x
    mobility   = float(np.sqrt(var_d1 / (var_x  + 1e-30)))
    mob_d1     = float(np.sqrt(var_d2 / (var_d1 + 1e-30)))
    complexity = mob_d1 / (mobility + 1e-30)

    return activity, mobility, complexity
