"""Run physical verification experiments and save subject-level + summary CSVs.

Two modes:

* ``--source cache`` (default) — reads ``cache/wiener_{mode}/`` for the
  decomposed ``specific`` signal and ``cache/epochs/`` for the raw signal.
  Validates matching subject/label/split, then computes V1 coherence
  (pre/post, per-band with pair roles), gate diagnostics (from the
  target-level arrays saved by script 02), and lagged connectivity
  (imaginary coherence, wPLI).  Aggregates per-recording, then per-subject.

* ``--source recompute`` — legacy mode.  Re-runs Wiener decomposition
  (frequency mode only) from ``cache/epochs/`` and runs wide-band V1/V2/V3.
  Kept for backward compatibility.

Usage
-----
# Cache-driven verification of a phasegated decomposition
python scripts/04_run_verification.py --source cache --mode phasegated

# Run only gate diagnostics + connectivity
python scripts/04_run_verification.py --checks gate,connectivity

# Fast exploratory verification: at most 10 deterministic epochs per recording
python scripts/04_run_verification.py --checks v1,gate,connectivity \
    --max-epochs-per-recording 10 --sample-seed 42 --workers 4

# Legacy recompute mode, single-threaded
python scripts/04_run_verification.py --source recompute --workers 1

Output
------
results/verification/
    verification_metadata.json     — source, mode, config_hash, checks executed
    v1_subject.csv                 — subject × pair × band × role
    v1_summary.csv                 — pair × band × role (mean, Wilcoxon p, Cohen d)
    gate_subject.csv               — subject × candidate_key
    gate_summary.csv               — candidate_key summary
    fusion_subject.csv             — subject × overlapping channel/source
    fusion_summary.csv             — overlap fusion rates and weights
    connectivity_subject.csv       — subject × pair × band × role × metric (pre/post)
    connectivity_summary.csv       — pair × band × role × metric summary

With ``--max-epochs-per-recording K``, all cache-mode checks use the same
deterministic K-epoch sample per recording. The sampling cap and seed are
recorded in ``verification_metadata.json``; omit the option for full data.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch, WienerResult
from eeg_bg.features.band_power import BANDS
from eeg_bg.verification._pair_roles import (
    classify_pair,
    ROLE_TARGETED_EDGE,
    ROLE_PROCESSED_UNTARGETED_HOMO,
    ROLE_PASSTHROUGH_CONTROL,
)
from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.verification.coherence import run_v1_per_band
from eeg_bg.verification.transitivity import run_v2, run_v3

# ── Legacy imports (recompute mode) ──────────────────────────────────────────
# keep old coherence for recompute mode
from eeg_bg.verification.coherence import run_v1 as _legacy_run_v1


# ═══════════════════════════════════════════════════════════════════════════════
# Cache-mode worker
# ═══════════════════════════════════════════════════════════════════════════════

def _verify_recording_cache(args: tuple) -> dict:
    """Process one recording from cached Wiener + epochs NPZ files.

    Returns a dict of recording-level aggregates keyed by subject_id.
    """
    (
        wiener_npz_str,
        epoch_root_str,
        wiener_root_str,
        cfg,
        checks,
        max_epochs_per_recording,
        sample_seed,
    ) = args
    wiener_path = Path(wiener_npz_str)
    epoch_path = Path(epoch_root_str) / wiener_path.relative_to(Path(wiener_root_str))

    # ── Load ──────────────────────────────────────────────────────────────
    wdata = np.load(wiener_path, allow_pickle=True)
    edata = np.load(epoch_path, allow_pickle=True)

    # Validate consistency
    w_evaluation_id = str(wdata.get("evaluation_id", wdata["subject_id"]))
    e_evaluation_id = str(edata.get("evaluation_id", edata["subject_id"]))
    if w_evaluation_id != e_evaluation_id:
        return {"error": f"evaluation_id mismatch: {wiener_path.name}"}
    if int(wdata["label"]) != int(edata["label"]):
        return {"error": f"label mismatch: {wiener_path.name}"}

    subject_id = w_evaluation_id
    label = int(wdata["label"])
    split = str(wdata.get("split", ""))
    recording_id = str(wdata.get(
        "recording_id",
        wiener_path.relative_to(Path(wiener_root_str)).parent / wiener_path.stem,
    ))
    ch_names = list(edata["ch_names"])

    sfreq = float(cfg["preprocessing"]["target_sfreq"])
    freq_res = float(cfg["wiener"].get("freq_resolution_hz", 0.5))

    specific = wdata["specific"]   # (n_epochs, n_ch, n_times)
    raw_epochs = edata["epochs"]   # (n_epochs, n_ch, n_times)
    n_epochs = specific.shape[0]
    epoch_indices = _sample_epoch_indices(
        n_epochs, max_epochs_per_recording, sample_seed, recording_id
    )
    n_sampled = len(epoch_indices)

    rec = {
        "subject_id":   subject_id,
        "recording_id": recording_id,
        "split":        split,
        "label":        label,
        "n_epochs":     n_sampled,
        "n_epochs_total": n_epochs,
    }

    # Preserve group-level skip semantics separately from target diagnostics.
    skipped_counts: dict[str, int] = defaultdict(int)
    if "skipped_pairs" in wdata:
        skipped_arr = wdata["skipped_pairs"]
        for epoch_idx in epoch_indices:
            epoch_items = skipped_arr[epoch_idx]
            for pair_key in list(epoch_items or []):
                skipped_counts[str(pair_key)] += 1
    rec["skipped_pairs"] = [
        {"subject_id": subject_id, "recording_id": recording_id,
         "split": split, "label": label, "pair_key": pair_key,
         "n_epochs_skipped": n}
        for pair_key, n in sorted(skipped_counts.items())
    ]

    # ── Gate diagnostics (from cache) ─────────────────────────────────────
    if "gate" in checks and "candidate_keys" in wdata:
        ck = list(wdata["candidate_keys"])
        cs = wdata.get("candidate_status", None)
        cc = wdata.get("candidate_coherence", None)
        pf = wdata.get("phase_gate_pass_fraction", None)

        gate_rows = []
        for ci, key in enumerate(ck):
            if cs is not None and ci < cs.shape[1]:
                status_epochs = cs[epoch_indices, ci]
                n_acc = int(np.sum(status_epochs == 0))
                n_below = int(np.sum(status_epochs == 1))
                n_unstable = int(np.sum(status_epochs == 2))
                mean_coh = float(cc[epoch_indices, ci].mean()) if cc is not None and ci < cc.shape[1] else 0.0
                mean_pf = float(pf[epoch_indices, ci].mean()) if pf is not None and ci < pf.shape[1] else 1.0
            else:
                n_acc = n_below = n_unstable = 0
                mean_coh = 0.0
                mean_pf = 1.0

            gate_rows.append({
                "candidate_key":     key,
                "n_epochs":          n_sampled,
                "n_accepted":        n_acc,
                "n_below_coherence": n_below,
                "n_unstable_filter": n_unstable,
                "acceptance_rate":   n_acc / n_sampled if n_sampled else 0.0,
                "mean_coherence":    mean_coh,
                "mean_pass_fraction": mean_pf,
            })
        rec["gate"] = gate_rows

    # ── Overlap fusion diagnostics (from cache) ───────────────────────────
    if "gate" in checks and "candidate_fusion_weight" in wdata:
        ck = [str(x) for x in list(wdata["candidate_keys"])]
        fw = np.asarray(wdata["candidate_fusion_weight"], dtype=float)
        fusion_rows = []
        candidate_channels = [key.split("::", 1)[1] for key in ck if "::" in key]
        overlap_channels = sorted({ch for ch in candidate_channels if candidate_channels.count(ch) > 1})
        for channel in overlap_channels:
            indices = [i for i, key in enumerate(ck) if key.endswith(f"::{channel}")]
            if not indices or fw.ndim != 2:
                continue
            weights = fw[epoch_indices][:, indices]
            n_sources = np.sum(weights > 0.0, axis=1)
            multi = n_sources >= 2
            effective = np.divide(
                1.0, np.sum(weights * weights, axis=1),
                out=np.zeros(weights.shape[0], dtype=float),
                where=np.sum(weights * weights, axis=1) > 0.0,
            )
            for local_idx, candidate_idx in enumerate(indices):
                active = weights[:, local_idx] > 0.0
                active_multi = active & multi
                fusion_rows.append({
                    "subject_id": subject_id,
                    "recording_id": recording_id,
                    "split": split,
                    "label": label,
                    "channel": channel,
                    "source_key": ck[candidate_idx].split("::", 1)[0],
                    "n_epochs": n_epochs,
                    "n_no_source": int(np.sum(n_sources == 0)),
                    "n_single_source": int(np.sum(n_sources == 1)),
                    "n_multi_source": int(np.sum(multi)),
                    "multi_source_rate": float(np.mean(multi)) if n_epochs else 0.0,
                    "mean_weight_when_active": float(np.mean(weights[active, local_idx])) if np.any(active) else 0.0,
                    "mean_weight_when_multisource": float(np.mean(weights[active_multi, local_idx])) if np.any(active_multi) else 0.0,
                    "mean_effective_source_count": float(np.mean(effective)),
                })
        rec["fusion"] = fusion_rows

    # ── V1 per-band coherence ─────────────────────────────────────────────
    v1_rows: list[dict] = []
    conn_rows: list[dict] = []

    if "v1" in checks or "connectivity" in checks:
        for ei in epoch_indices:
            raw_ep = raw_epochs[ei]
            spc_ep = specific[ei]

            if "v1" in checks:
                df_v1 = run_v1_per_band(
                    raw_ep, spc_ep, ch_names, sfreq,
                    freq_resolution_hz=freq_res,
                    subject_id=subject_id,
                    recording_id=recording_id,
                    epoch_idx=ei,
                    split=split,
                    label=label,
                    channel_groups=cfg.get("channels", {}).get("channel_groups"),
                )
                v1_rows.extend(df_v1.to_dict("records"))

            if "connectivity" in checks:
                from eeg_bg.verification.connectivity import compute_connectivity_metrics
                metrics = compute_connectivity_metrics(
                    raw_ep, spc_ep, ch_names, sfreq,
                    nperseg=int(sfreq / freq_res),
                )
                i1 = metrics.pop("i1")
                i2 = metrics.pop("i2")
                pair_present = metrics.pop("pair_present")
                freqs_arr = metrics.pop("freqs")

                for pair_idx in range(len(i1)):
                    if not pair_present[pair_idx]:
                        continue
                    chi = _STANDARD_19[i1[pair_idx]]
                    chj = _STANDARD_19[i2[pair_idx]]
                    role = classify_pair(
                        chi, chj,
                        cfg.get("channels", {}).get("channel_groups"),
                    )
                    for band_name in BANDS:
                        pre_coh = metrics.get(f"{band_name}_coh_pre", None)
                        post_coh = metrics.get(f"{band_name}_coh_post", None)
                        pre_plv = metrics.get(f"{band_name}_plv_pre", None)
                        post_plv = metrics.get(f"{band_name}_plv_post", None)
                        pre_icoh = metrics.get(f"{band_name}_icoh_pre", None)
                        post_icoh = metrics.get(f"{band_name}_icoh_post", None)
                        pre_wpli = metrics.get(f"{band_name}_wpli_pre", None)
                        post_wpli = metrics.get(f"{band_name}_wpli_post", None)
                        conn_rows.append({
                            "subject_id":    subject_id,
                            "recording_id":  recording_id,
                            "epoch_idx":     ei,
                            "split":         split,
                            "label":         label,
                            "ch_i":          chi,
                            "ch_j":          chj,
                            "pair_role":     role,
                            "band":          band_name,
                            "coh_pre":       float(pre_coh[pair_idx]) if pre_coh is not None else np.nan,
                            "coh_post":      float(post_coh[pair_idx]) if post_coh is not None else np.nan,
                            "plv_pre":       float(pre_plv[pair_idx]) if pre_plv is not None else np.nan,
                            "plv_post":      float(post_plv[pair_idx]) if post_plv is not None else np.nan,
                            "icoh_pre":      float(pre_icoh[pair_idx]) if pre_icoh is not None else np.nan,
                            "icoh_post":     float(post_icoh[pair_idx]) if post_icoh is not None else np.nan,
                            "wpli_pre":      float(pre_wpli[pair_idx]) if pre_wpli is not None else np.nan,
                            "wpli_post":     float(post_wpli[pair_idx]) if post_wpli is not None else np.nan,
                        })

    if v1_rows:
        rec["v1"] = v1_rows
    if conn_rows:
        rec["connectivity"] = conn_rows

    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic epoch sampling
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_epoch_indices(
    n_epochs: int,
    max_epochs_per_recording: int | None,
    sample_seed: int,
    recording_id: str,
) -> np.ndarray:
    """Return deterministic, recording-specific epoch indices.

    SHA-256 is used instead of Python's process-randomised ``hash()`` so all
    worker processes and Wiener conditions select the same epochs.
    """
    if max_epochs_per_recording is None or max_epochs_per_recording >= n_epochs:
        return np.arange(n_epochs, dtype=int)
    if max_epochs_per_recording < 1:
        raise ValueError("max_epochs_per_recording must be >= 1")
    digest = hashlib.sha256(
        f"{sample_seed}:{recording_id}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_epochs, size=max_epochs_per_recording, replace=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation test helper
# ═══════════════════════════════════════════════════════════════════════════════

def _paired_wilcoxon_paired(
    values: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Wilcoxon signed-rank test against *threshold* (default 0).

    Returns NaN when fewer than 3 non-tied pairs are available.
    """
    diffs = values - threshold
    nonzero = diffs[diffs != 0.0]
    if len(nonzero) < 3:
        return np.nan
    try:
        _, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except Exception:
        return np.nan


def _cohens_d(values: np.ndarray, threshold: float = 0.0) -> float:
    diffs = values - threshold
    sd = np.std(diffs, ddof=1)
    if sd < 1e-30:
        return 0.0
    return float(np.mean(diffs) / sd)


def _bh_qvalues(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values for a Series that may contain NaN."""
    arr = values.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    out = np.full(len(arr), np.nan)
    if valid.any():
        order = np.argsort(arr[valid])
        ranked = arr[valid][order]
        q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        restored = np.empty(len(ranked))
        restored[order] = np.minimum(q, 1.0)
        out[valid] = restored
    return pd.Series(out, index=values.index)


# ═══════════════════════════════════════════════════════════════════════════════
# Subject-level aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_v1_subject(recs: list[dict]) -> pd.DataFrame:
    """Aggregate V1 epoch rows → subject-level means per (pair, band, role)."""
    all_rows = []
    for rec in recs:
        all_rows.extend(rec.get("v1", []))
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    grp = df.groupby(["subject_id", "ch_i", "ch_j", "pair_role", "band", "split", "label"])
    subj = grp.agg(
        mean_coh_pre=("coh_pre", "mean"),
        mean_coh_post=("coh_post", "mean"),
        mean_reduction=("reduction", "mean"),
        n_epochs=("epoch_idx", "nunique"),
    ).reset_index()
    return subj


def _aggregate_v1_summary(subj_df: pd.DataFrame) -> pd.DataFrame:
    if subj_df.empty:
        return pd.DataFrame()
    rows = []
    grp = subj_df.groupby(["ch_i", "ch_j", "pair_role", "band"])
    for (chi, chj, role, band), g in grp:
        n = len(g)
        vals = g["mean_reduction"].values
        mean_red = float(np.mean(vals))
        wilc_p = _paired_wilcoxon_paired(vals, 0.0)
        cd = _cohens_d(vals, 0.0)
        rows.append({
            "ch_i": chi, "ch_j": chj, "pair_role": role, "band": band,
            "n_subjects": n, "mean_reduction": mean_red,
            "wilcoxon_p": wilc_p, "cohens_d": cd,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = _bh_qvalues(out["wilcoxon_p"])
    return out


def _aggregate_gate_subject(recs: list[dict]) -> pd.DataFrame:
    all_rows = []
    for rec in recs:
        for g in rec.get("gate", []):
            g["subject_id"] = rec["subject_id"]
            all_rows.append(g)
    return pd.DataFrame(all_rows)


def _aggregate_gate_summary(subj_df: pd.DataFrame) -> pd.DataFrame:
    if subj_df.empty:
        return pd.DataFrame()
    grp = subj_df.groupby("candidate_key")
    rows = []
    for key, g in grp:
        rows.append({
            "candidate_key": key,
            "n_subjects": len(g),
            "mean_acceptance_rate": float(g["acceptance_rate"].mean()),
            "median_acceptance_rate": float(g["acceptance_rate"].median()),
            "mean_coherence": float(g["mean_coherence"].mean()),
            "mean_pass_fraction": float(g["mean_pass_fraction"].mean()),
        })
    out = pd.DataFrame(rows)
    return out


def _aggregate_fusion_subject(recs: list[dict]) -> pd.DataFrame:
    """Aggregate per-recording overlap fusion diagnostics to subject rows."""
    rows = [row for rec in recs for row in rec.get("fusion", [])]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["subject_id", "channel", "source_key", "split", "label"]
    numeric = [
        "n_epochs", "n_no_source", "n_single_source", "n_multi_source",
        "multi_source_rate", "mean_weight_when_active",
        "mean_weight_when_multisource", "mean_effective_source_count",
    ]
    return df.groupby(group_cols, as_index=False)[numeric].mean()


def _aggregate_fusion_summary(subj_df: pd.DataFrame) -> pd.DataFrame:
    if subj_df.empty:
        return pd.DataFrame()
    group_cols = ["channel", "source_key"]
    out = subj_df.groupby(group_cols, as_index=False).agg(
        n_subjects=("subject_id", "nunique"),
        mean_multi_source_rate=("multi_source_rate", "mean"),
        mean_weight_when_active=("mean_weight_when_active", "mean"),
        mean_weight_when_multisource=("mean_weight_when_multisource", "mean"),
        mean_effective_source_count=("mean_effective_source_count", "mean"),
    )
    return out


def _aggregate_skipped_summary(recs: list[dict]) -> pd.DataFrame:
    rows = [row for rec in recs for row in rec.get("skipped_pairs", [])]
    if not rows:
        return pd.DataFrame(columns=["pair_key", "n_subjects", "n_recordings", "n_epochs_skipped"])
    df = pd.DataFrame(rows)
    return (df.groupby("pair_key", as_index=False)
              .agg(n_subjects=("subject_id", "nunique"),
                   n_recordings=("recording_id", "nunique"),
                   n_epochs_skipped=("n_epochs_skipped", "sum")))


def _aggregate_conn_subject(recs: list[dict]) -> pd.DataFrame:
    all_rows = []
    for rec in recs:
        all_rows.extend(rec.get("connectivity", []))
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _aggregate_conn_summary(subj_df: pd.DataFrame) -> pd.DataFrame:
    if subj_df.empty:
        return pd.DataFrame()
    rows = []
    grp = subj_df.groupby(["ch_i", "ch_j", "pair_role", "band"])
    metrics = ["coh", "plv", "icoh", "wpli"]
    for (chi, chj, role, band), g in grp:
        n = len(g["subject_id"].unique())
        row = {"ch_i": chi, "ch_j": chj, "pair_role": role, "band": band, "n_subjects": n}
        for m in metrics:
            pre = g[f"{m}_pre"].values
            post = g[f"{m}_post"].values
            diff = pre - post
            valid = ~np.isnan(diff)
            d = diff[valid]
            row[f"{m}_mean_pre"] = float(np.mean(pre[valid])) if valid.any() else np.nan
            row[f"{m}_mean_post"] = float(np.mean(post[valid])) if valid.any() else np.nan
            row[f"{m}_mean_diff"] = float(np.mean(d)) if len(d) > 0 else np.nan
            row[f"{m}_wilcoxon_p"] = _paired_wilcoxon_paired(d, 0.0) if len(d) >= 3 else np.nan
            row[f"{m}_cohens_d"] = _cohens_d(d, 0.0) if len(d) >= 3 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for metric in metrics:
            out[f"{metric}_q_value"] = _bh_qvalues(out[f"{metric}_wilcoxon_p"])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy recompute worker
# ═══════════════════════════════════════════════════════════════════════════════

def _decompose_one_file(args):
    """Worker: decompose all epochs in one subject .npz, return list[WienerResult]."""
    npz_path_str, cfg = args
    data = np.load(npz_path_str, allow_pickle=True)
    epochs     = data["epochs"]
    ch_names   = list(data["ch_names"])
    subject_id = str(data.get("evaluation_id", data["subject_id"]))
    return [
        decompose_epoch(ep, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
        for i, ep in enumerate(epochs)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(config_path: str, source: str, mode: str, checks: list[str],
         max_workers: int | None = None,
         max_epochs_per_recording: int | None = None,
         sample_seed: int = 42) -> None:
    cfg = load_config(config_path)
    cache_dir  = Path(cfg["paths"]["cache_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    verif_dir = results_dir / "verification"
    verif_dir.mkdir(parents=True, exist_ok=True)

    # Hash of relevant config sections for metadata
    config_hash = hashlib.sha256(
        json.dumps({
            "source": source, "mode": mode, "checks": checks,
            "max_epochs_per_recording": max_epochs_per_recording,
            "sample_seed": sample_seed,
            "wiener": cfg.get("wiener", {}),
            "preprocessing": cfg.get("preprocessing", {}),
        }, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    n_workers = max_workers or os.cpu_count()
    all_checks = set(checks)

    if source == "cache":
        wiener_root = cache_dir / f"wiener_{mode}"
        if not wiener_root.exists():
            raise FileNotFoundError(
                f"Wiener cache not found at {wiener_root}. "
                f"Run scripts/02_run_wiener.py --mode {mode} first."
            )
        epoch_root = cache_dir / "epochs"
        wiener_paths = sorted(wiener_root.rglob("*.npz"))
        args_list = [
            (
                str(p), str(epoch_root), str(wiener_root), cfg, all_checks,
                max_epochs_per_recording, sample_seed,
            )
            for p in wiener_paths
        ]

        recordings: list[dict] = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_verify_recording_cache, a): a for a in args_list}
            with tqdm(total=len(futures), desc="Verifying (cache)") as pbar:
                for future in as_completed(futures):
                    rec = future.result()
                    if "error" not in rec:
                        recordings.append(rec)
                    else:
                        tqdm.write(f"  SKIP {rec['error']}")
                    pbar.update(1)

        # ── Aggregate to subject level ───────────────────────────────────
        _write_cache_outputs(verif_dir, recordings, all_checks)

        # ── Metadata ─────────────────────────────────────────────────────
        n_subjects = len({r["subject_id"] for r in recordings})
        _write_metadata(
            verif_dir, source, mode, checks, config_hash, n_subjects,
            n_recordings=len(recordings),
            max_epochs_per_recording=max_epochs_per_recording,
            sample_seed=sample_seed,
        )

        print(f"Verification complete. Results saved to {verif_dir}")

    elif source == "recompute":
        epoch_root = cache_dir / "epochs"
        verif_dir.mkdir(parents=True, exist_ok=True)

        npz_paths = sorted(epoch_root.rglob("*.npz"))
        args_list = [(str(p), cfg) for p in npz_paths]

        all_results: list[WienerResult] = []
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            futures = {executor.submit(_decompose_one_file, args): args
                       for args in args_list}
            with tqdm(total=len(futures), desc="Decomposing epochs") as pbar:
                for future in as_completed(futures):
                    all_results.extend(future.result())
                    pbar.update(1)
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            tqdm.write("\nInterrupted.")
            executor.shutdown(wait=False, cancel_futures=True)
            os._exit(1)

        print(f"Running V1/V2/V3 on {len(all_results)} epochs...")
        with ThreadPoolExecutor(max_workers=3) as tex:
            fut_v1 = tex.submit(_legacy_run_v1, all_results, cfg)
            fut_v2 = tex.submit(run_v2, all_results, cfg)
            fut_v3 = tex.submit(run_v3, all_results, cfg)
            v1_df = fut_v1.result()
            v2_df = fut_v2.result()
            v3_df = fut_v3.result()

        v1_df.to_csv(verif_dir / "v1_coherence.csv", index=False)
        v2_df.to_csv(verif_dir / "v2_transitivity.csv", index=False)
        v3_df.to_csv(verif_dir / "v3_frequency_variation.csv", index=False)

        print(f"V1: {len(v1_df)} rows | V2: {len(v2_df)} rows | V3: {len(v3_df)} rows")
        print(f"Results saved to {verif_dir}")
    else:
        raise ValueError(f"Unknown --source: {source}")


def _write_cache_outputs(verif_dir: Path, recordings: list[dict],
                         checks: set[str]) -> None:
    """Write subject-level and summary CSVs for cache-mode verification."""

    if "v1" in checks:
        v1_subj = _aggregate_v1_subject(recordings)
        v1_subj.to_csv(verif_dir / "v1_subject.csv", index=False)
        if not v1_subj.empty:
            (v1_subj.groupby(
                ["subject_id", "pair_role", "band", "split", "label"],
                as_index=False,
            )[["mean_coh_pre", "mean_coh_post", "mean_reduction", "n_epochs"]]
             .mean().to_csv(verif_dir / "v1_role_subject.csv", index=False))
        v1_sum = _aggregate_v1_summary(v1_subj)
        v1_sum.to_csv(verif_dir / "v1_summary.csv", index=False)
        print(f"V1: {len(v1_subj)} subject rows → {verif_dir / 'v1_subject.csv'}")

    if "gate" in checks:
        gate_subj = _aggregate_gate_subject(recordings)
        gate_subj.to_csv(verif_dir / "gate_subject.csv", index=False)
        gate_sum = _aggregate_gate_summary(gate_subj)
        gate_sum.to_csv(verif_dir / "gate_summary.csv", index=False)
        print(f"Gate: {len(gate_subj)} subject rows → {verif_dir / 'gate_subject.csv'}")

        fusion_subj = _aggregate_fusion_subject(recordings)
        fusion_subj.to_csv(verif_dir / "fusion_subject.csv", index=False)
        fusion_sum = _aggregate_fusion_summary(fusion_subj)
        fusion_sum.to_csv(verif_dir / "fusion_summary.csv", index=False)
        print(f"Fusion: {len(fusion_subj)} subject rows → {verif_dir / 'fusion_subject.csv'}")

    if "gate" in checks:
        skipped = [row for rec in recordings for row in rec.get("skipped_pairs", [])]
        pd.DataFrame(skipped).to_csv(verif_dir / "skipped_pairs_subject.csv", index=False)
        _aggregate_skipped_summary(recordings).to_csv(
            verif_dir / "skipped_pairs_summary.csv", index=False
        )

    if "connectivity" in checks:
        conn_raw = _aggregate_conn_subject(recordings)
        if not conn_raw.empty:
            # Per-epoch → subject-level (mean per subject × pair × band × role)
            grp_cols = ["subject_id", "ch_i", "ch_j", "pair_role", "band", "split", "label"]
            metric_cols = [c for c in conn_raw.columns
                          if c not in grp_cols + ["recording_id", "epoch_idx"]]
            subj_agg = conn_raw.groupby(grp_cols, dropna=False)[metric_cols].mean().reset_index()
            subj_agg.to_csv(verif_dir / "connectivity_subject.csv", index=False)
            role_cols = ["subject_id", "pair_role", "band", "split", "label"]
            metric_cols = [c for c in subj_agg.columns if c not in role_cols + ["ch_i", "ch_j"]]
            subj_agg.groupby(role_cols, as_index=False)[metric_cols].mean().to_csv(
                verif_dir / "connectivity_role_subject.csv", index=False
            )
            conn_sum = _aggregate_conn_summary(subj_agg)
            conn_sum.to_csv(verif_dir / "connectivity_summary.csv", index=False)
            print(f"Connectivity: {len(subj_agg)} subject rows → {verif_dir / 'connectivity_subject.csv'}")
        else:
            print("Connectivity: no pair data found — skipping connectivity output.")


def _write_metadata(verif_dir: Path, source: str, mode: str,
                    checks: list[str], config_hash: str,
                    n_subjects: int, n_recordings: int,
                    max_epochs_per_recording: int | None = None,
                    sample_seed: int = 42) -> None:
    meta = {
        "source": source,
        "mode": mode,
        "checks": checks,
        "config_hash": config_hash,
        "n_subjects": n_subjects,
        "n_recordings": n_recordings,
        "max_epochs_per_recording": max_epochs_per_recording,
        "sample_seed": sample_seed,
    }
    with open(verif_dir / "verification_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run physical verification experiments"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--source", choices=["cache", "recompute"], default="cache",
        help="'cache' reads decomposed signal from cache/wiener_{mode}/; "
             "'recompute' re-runs Wiener from epochs (frequency only, legacy).",
    )
    parser.add_argument(
        "--mode", choices=["frequency", "phasegated", "scalar", "zerophase"],
        default="frequency",
        help="Wiener cache subdirectory to read from (with --source cache).",
    )
    parser.add_argument(
        "--checks", default="v1,gate,connectivity",
        help="Comma-separated list of checks: v1, gate, connectivity (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    parser.add_argument(
        "--max-epochs-per-recording", type=int, default=None,
        help="Randomly sample at most this many epochs per recording; "
             "default processes all epochs",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Seed for deterministic per-recording epoch sampling (default: 42)",
    )
    args = parser.parse_args()
    main(
        args.config, args.source, args.mode,
        [c.strip() for c in args.checks.split(",")],
        args.workers,
        args.max_epochs_per_recording,
        args.sample_seed,
    )
