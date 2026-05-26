"""
V2: Transitivity constraint for single-point-source verification.
V3: Frequency variation of |h_ij(f)| across the target band.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd


def _build_pairwise_lookup(
    filters: dict,
    mask: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    """Build a ``(target, ref) → h(f)[mask]`` lookup from a filter dict.

    For a group ``[A, B, C]`` the stored filters are multi-reference
    (``h_A`` uses both B and C simultaneously).  We extract each individual
    reference coefficient as an approximation of the pairwise single-reference
    Wiener filter — this is exact under the point-source model and
    approximately correct otherwise.

    Reference order for a group of channels ``[c0, c1, ..., ck]`` with target
    at local index ``t`` is ``[c0, ..., c_{t-1}, c_{t+1}, ..., ck]``, so
    ``h[r_idx]`` corresponds to the ``r_idx``-th element of that list.
    """
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for _pair_key, ch_dict in filters.items():
        group = list(ch_dict.keys())       # ordered channel list for this group
        for t_idx, target_ch in enumerate(group):
            h = ch_dict[target_ch]         # shape (n_ref, n_freqs)
            refs = [c for c in group if c != target_ch]
            for r_idx, ref_ch in enumerate(refs):
                if h.shape[0] > r_idx:
                    lookup[(target_ch, ref_ch)] = h[r_idx][mask]
    return lookup


def run_v2(results: list, cfg: dict) -> pd.DataFrame:
    """V2: transitivity constraint — h_ij * h_jk ≈ h_ik for a single source.

    Handles both 2-channel and 3-channel groups.  For 3-channel groups each
    individual reference coefficient is used as a proxy for the corresponding
    pairwise single-reference Wiener filter.  Only triplets where all three
    pairwise filters are simultaneously available are tested.
    """
    freq_band = cfg["wiener"]["freq_band"]
    rows = []
    for result in results:
        freqs = result.freqs
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        filters = result.filters

        # Build (target, ref) → h(f) lookup across all groups
        lookup = _build_pairwise_lookup(filters, mask)
        chs = sorted({ch for (ch, _) in lookup})

        for ch_i, ch_j, ch_k in combinations(chs, 3):
            h_ij = lookup.get((ch_i, ch_j))
            h_jk = lookup.get((ch_j, ch_k))
            h_ik = lookup.get((ch_i, ch_k))

            if h_ij is None or h_jk is None or h_ik is None:
                continue

            eps_amp = float(
                np.mean(np.abs(np.abs(h_ij) * np.abs(h_jk) - np.abs(h_ik)))
            )
            phase_sum = np.angle(h_ij) + np.angle(h_jk) - np.angle(h_ik)
            eps_phase = float(
                np.mean(np.abs(np.angle(np.exp(1j * phase_sum))))
            )
            rows.append({
                "subject_id": result.subject_id,
                "epoch_idx":  result.epoch_idx,
                "triplet":    f"{ch_i}-{ch_j}-{ch_k}",
                "eps_amp":    eps_amp,
                "eps_phase":  eps_phase,
            })

    cols = ["subject_id", "epoch_idx", "triplet", "eps_amp", "eps_phase"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def run_v3(results: list, cfg: dict) -> pd.DataFrame:
    freq_band = cfg["wiener"]["freq_band"]
    rows = []
    for result in results:
        freqs = result.freqs
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        for pair_key, ch_dict in result.filters.items():
            for ch_name, h in ch_dict.items():
                amp = np.abs(h[0, mask])
                if amp.size == 0:
                    continue
                mean_amp = float(amp.mean())
                variation = float((amp.max() - amp.min()) / (mean_amp + 1e-12))
                rows.append({
                    "subject_id": result.subject_id,
                    "epoch_idx": result.epoch_idx,
                    "pair": pair_key,
                    "channel": ch_name,
                    "freq_variation": variation,
                    "amp_mean": mean_amp,
                    "amp_std": float(amp.std()),
                })
    return pd.DataFrame(rows)
