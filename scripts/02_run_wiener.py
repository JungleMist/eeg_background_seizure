"""Run Wiener decomposition on cached epochs."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.config.settings import load_config


def _process_wiener_file(args):
    npz_path_str, cfg, epoch_root_str, out_root_str, mode, force = args
    npz_path = Path(npz_path_str)
    out_path = Path(out_root_str) / npz_path.relative_to(Path(epoch_root_str))

    if out_path.exists() and not force:
        data = np.load(out_path, allow_pickle=True)
        sv = data.get("schema_version", None)
        if sv is None or int(sv) < 2:
            raise ValueError(
                f"Old Wiener cache without target-level diagnostics "
                f"(schema_version={sv}). Re-run script 02 with --force."
            )
        return (npz_path.name, "cached", None)

    from eeg_bg.decomposition import (
        wiener as wiener_freq,
        wiener_phasegated,
        wiener_scalar,
        wiener_zerophase,
    )
    _mode_dispatch = {
        "frequency": wiener_freq.decompose_epoch,
        "phasegated": wiener_phasegated.decompose_epoch,
        "scalar": wiener_scalar.decompose_epoch,
        "zerophase": wiener_zerophase.decompose_epoch,
    }
    decompose = _mode_dispatch[mode]

    try:
        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
    except Exception as e:
        return (npz_path.name, "skip", f"corrupt cache file: {e}")
    ch_names = list(data["ch_names"])
    subject_id = str(data["subject_id"])

    results = []
    for i, epoch in enumerate(epochs):
        r = decompose(epoch, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
        results.append(
            {"specific": r.specific, "coherent": r.coherent,
             "candidate_keys": r.candidate_keys,
             "candidate_status": r.candidate_status,
             "candidate_coherence": r.candidate_coherence,
             "candidate_max_abs_h": r.candidate_max_abs_h,
             "phase_gate_pass_fraction": r.phase_gate_pass_fraction,
             "candidate_fusion_weight": r.candidate_fusion_weight,
             "skipped_pairs": r.skipped_pairs,
             }
        )

    # ── Build diagnostic arrays from per-epoch dicts ──────────────────────
    _diag_keys = ["candidate_status", "candidate_coherence",
                  "candidate_max_abs_h", "phase_gate_pass_fraction",
                  "candidate_fusion_weight"]
    _diag_arrays: dict[str, np.ndarray] = {}
    for k in _diag_keys:
        arrs = [r[k] for r in results]
        # All None → skip this key; all ndarray → stack; mix → error.
        if all(a is None for a in arrs):
            continue
        elif all(isinstance(a, np.ndarray) for a in arrs):
            _diag_arrays[k] = np.stack(arrs, axis=0)
        else:
            _diag_arrays[k] = np.array(arrs, dtype=object)

    skipped_arr = np.empty(len(results), dtype=object)
    for i, r in enumerate(results):
        skipped_arr[i] = list(r.get("skipped_pairs") or [])

    # candidate_keys are identical across epochs — take from first result.
    _first = results[0] if results else {}
    ck = _first.get("candidate_keys")
    _ck_arr = np.array(ck, dtype=object) if ck else np.array([], dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        specific=np.stack([r["specific"] for r in results]),
        coherent=np.stack([r["coherent"] for r in results]),
        label=data["label"],
        subject_id=data["subject_id"],
        split=data["split"],
        schema_version=2,
        candidate_keys=_ck_arr,
        skipped_pairs=skipped_arr,
        **_diag_arrays,
    )
    return (npz_path.name, "done", None)


def main(config_path: str, mode: str = "frequency", force: bool = False,
         max_workers: int | None = None) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / f"wiener_{mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(epoch_root.rglob("*.npz"))
    args_list = [
        (str(p), cfg, str(epoch_root), str(out_root), mode, force)
        for p in npz_paths
    ]

    n_workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_wiener_file, args): args for args in args_list}
        with tqdm(total=len(futures), desc="Wiener") as pbar:
            for future in as_completed(futures):
                fname, status, msg = future.result()
                if status == "skip":
                    tqdm.write(f"  SKIP {fname}: {msg}")
                pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wiener decomposition on cached epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["frequency", "phasegated", "scalar", "zerophase"],
        default="frequency",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()
    main(args.config, args.mode, args.force, args.workers)
