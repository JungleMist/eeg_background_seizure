"""Shared constants for the features package.

Keeping _STANDARD_19 here prevents a circular import: connectivity.py
needs this list to define ALL_PAIRS at import time, but extraction.py
also imports connectivity.py — so both must import from a third module.
"""

_STANDARD_19: list[str] = [
    "FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4",
    "T3", "T4", "T5", "T6", "P3", "P4", "O1", "O2",
    "Fz", "Cz", "Pz",
]
