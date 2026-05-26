"""Run physical verification experiments V1/V2/V3 and save CSV reports."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch, WienerResult
from eeg_bg.verification.coherence import run_v1
from eeg_bg.verification.transitivity import run_v2, run_v3


def _decompose_one_file(args):
    """Worker: decompose all epochs in one subject .npz, return list[WienerResult]."""
    npz_path_str, cfg = args
    data = np.load(npz_path_str, allow_pickle=True)
    epochs     = data["epochs"]
    ch_names   = list(data["ch_names"])
    subject_id = str(data["subject_id"])
    return [
        decompose_epoch(ep, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
        for i, ep in enumerate(epochs)
    ]


def main(config_path: str, max_workers: int | None = None) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    verif_dir = Path(cfg["paths"]["results_dir"]) / "verification"
    verif_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: parallel Wiener decomposition over all subject files ──────────
    npz_paths = sorted(epoch_root.rglob("*.npz"))
    args_list = [(str(p), cfg) for p in npz_paths]

    all_results: list[WienerResult] = []
    n_workers = max_workers or os.cpu_count()
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

    # ── Phase 2: V1/V2/V3 run concurrently on the collected results ────────────
    print(f"Running V1/V2/V3 on {len(all_results)} epochs...")
    with ThreadPoolExecutor(max_workers=3) as tex:
        fut_v1 = tex.submit(run_v1, all_results, cfg)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physical verification experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()
    main(args.config, args.workers)
