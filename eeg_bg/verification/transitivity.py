"""
V2: Transitivity constraint for single-point-source verification.
V3: Frequency variation of |h_ij(f)| across the target band.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd


def run_v2(results: list, cfg: dict) -> pd.DataFrame:
    freq_band = cfg["wiener"]["freq_band"]
    rows = []
    for result in results:
        freqs = result.freqs
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        filters = result.filters

        # Collect all channels that appear in filters
        all_chs: dict[str, tuple[str, str, np.ndarray]] = {}
        for pair_key, ch_dict in filters.items():
            for ch_name, h in ch_dict.items():
                all_chs[ch_name] = (pair_key, ch_name, h[0])  # h[0] = (n_freqs,)

        chs = list(all_chs.keys())
        for ch_i, ch_j, ch_k in combinations(chs, 3):
            # Try to find h_ij, h_jk, h_ik from the filters dict
            # We look for pairs where ch_i uses ch_j as reference, etc.
            h_ij = h_jk = h_ik = None
            for pair_key, ch_dict in filters.items():
                chs_in_pair = list(ch_dict.keys())
                if ch_i in ch_dict and len(chs_in_pair) == 2:
                    # The reference for ch_i in this pair is the other channel
                    other = [c for c in chs_in_pair if c != ch_i][0]
                    if other == ch_j:
                        h_ij = ch_dict[ch_i][0, mask] if ch_dict[ch_i].shape[0] >= 1 else None
                if ch_j in ch_dict and len(chs_in_pair) == 2:
                    other = [c for c in chs_in_pair if c != ch_j][0]
                    if other == ch_k:
                        h_jk = ch_dict[ch_j][0, mask] if ch_dict[ch_j].shape[0] >= 1 else None
                if ch_i in ch_dict and len(chs_in_pair) == 2:
                    other = [c for c in chs_in_pair if c != ch_i][0]
                    if other == ch_k:
                        h_ik = ch_dict[ch_i][0, mask] if ch_dict[ch_i].shape[0] >= 1 else None

            if h_ij is None or h_jk is None or h_ik is None:
                continue

            eps_amp = float(np.mean(np.abs(np.abs(h_ij) * np.abs(h_jk) - np.abs(h_ik))))
            phase_sum = np.angle(h_ij) + np.angle(h_jk) - np.angle(h_ik)
            eps_phase = float(np.mean(np.abs(np.angle(np.exp(1j * phase_sum)))))
            rows.append({
                "subject_id": result.subject_id,
                "epoch_idx": result.epoch_idx,
                "triplet": f"{ch_i}-{ch_j}-{ch_k}",
                "eps_amp": eps_amp,
                "eps_phase": eps_phase,
            })
    return pd.DataFrame(rows)


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
