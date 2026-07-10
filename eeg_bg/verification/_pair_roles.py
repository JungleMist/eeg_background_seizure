"""Pair role classification for verification analyses.

Each electrode pair is assigned one of three mutually-exclusive roles:

* ``targeted_edge`` — adjacent electrodes within a Wiener channel group
  (G1-G6).  Coherence here is expected to drop sharply because the filter
  explicitly models these pairs.

* ``processed_untargeted_homotopic`` — cross-hemisphere homotopic pairs
  whose individual electrodes are processed (appear in some group) but
  are not in the *same* group.  These are the key indicators: if neural
  coupling is preserved, their lagged connectivity should stay close to 1.

* ``passthrough_control`` — pairs of electrodes that are never filtered
  (not in any ``channel_groups`` entry).  These serve as a pipeline
  quality-control: raw ≈ post should hold almost exactly.

The remaining pairs (combinations of processed and passthrough electrodes)
fall into ``other`` and are typically excluded from statistical comparison.
"""
from __future__ import annotations

from eeg_bg.features._constants import _STANDARD_19

# ── Role tag constants ──────────────────────────────────────────────────────
ROLE_TARGETED_EDGE             = "targeted_edge"
ROLE_PROCESSED_UNTARGETED_HOMO = "processed_untargeted_homotopic"
ROLE_PASSTHROUGH_CONTROL       = "passthrough_control"
ROLE_OTHER                     = "other"

# ── Channel-group edges ------------------------------------------------------
# Adjacent electrode pairs within each default G1-G6 channel group.
# Defined in the same order as configs/default.yaml ``channels.channel_groups``.
_TARGETED_EDGES: set[tuple[str, str]] = {
    # G1 — symmetric facial (frontalis)
    ("FP1", "FP2"),
    # G2 — left SCM
    ("F7", "T3"),
    # G3 — left posterior neck (3-channel chain → 2 adjacent edges)
    ("T3", "T5"),
    ("T5", "O1"),
    # G4 — bilateral occipitalis
    ("O1", "O2"),
    # G5 — right SCM
    ("F8", "T4"),
    # G6 — right posterior neck (3-channel chain → 2 adjacent edges)
    ("T4", "T6"),
    ("T6", "O2"),
}

# Canonicalise direction
_TARGETED_EDGES = {
    tuple(sorted(pair)) for pair in _TARGETED_EDGES  # type: ignore[misc]
}

# ── Processed-untargeted homotopic pairs -------------------------------------
# Cross-hemisphere pairs whose individual electrodes are processed (appear in
# at least one group) but not together in the same group.
_PROCESSED_CHANNELS: set[str] = {
    "FP1", "FP2", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2",
}
_PASSTHROUGH_CONTROL_PAIRS: set[tuple[str, str]] = {
    ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
}

_PROCESSED_UNTARGETED_HOMOTOPIC: set[tuple[str, str]] = {
    ("F7", "F8"), ("T3", "T4"), ("T5", "T6"),
}


_HOMOTOPIC_PAIRS: set[tuple[str, str]] = {
    tuple(sorted(pair)) for pair in (("F7", "F8"), ("T3", "T4"), ("T5", "T6"))
}


def classify_pair(ch_i: str, ch_j: str, channel_groups: list[list[str]] | None = None) -> str:
    """Return the role tag for an unordered electrode pair.

    Returns one of ``ROLE_TARGETED_EDGE``, ``ROLE_PROCESSED_UNTARGETED_HOMO``,
    ``ROLE_PASSTHROUGH_CONTROL``, or ``ROLE_OTHER``.
    """
    pair = tuple(sorted((ch_i, ch_j)))
    groups = channel_groups
    if groups is None:
        targeted_edges = _TARGETED_EDGES
        processed_channels = _PROCESSED_CHANNELS
    else:
        targeted_edges = {
            tuple(sorted((a, b)))
            for group in groups
            for a, b in zip(group, group[1:])
        }
        processed_channels = {ch for group in groups for ch in group}

    if pair in targeted_edges:
        return ROLE_TARGETED_EDGE
    if pair in _HOMOTOPIC_PAIRS and pair[0] in processed_channels and pair[1] in processed_channels:
        return ROLE_PROCESSED_UNTARGETED_HOMO
    if pair in _PASSTHROUGH_CONTROL_PAIRS and not (pair[0] in processed_channels or pair[1] in processed_channels):
        return ROLE_PASSTHROUGH_CONTROL
    return ROLE_OTHER


def all_pairs_with_roles() -> list[tuple[str, str, str]]:
    """Return every C(19,2) pair from ``_STANDARD_19`` with its role tag."""
    result: list[tuple[str, str, str]] = []
    for i, chi in enumerate(_STANDARD_19):
        for chj in _STANDARD_19[i + 1:]:
            result.append((chi, chj, classify_pair(chi, chj)))
    return result
