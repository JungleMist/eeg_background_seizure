"""Run ICA decomposition on cached epochs."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.ica import apply_ica, fit_ica


def _init_worker():
    import mne
    mne.set_log_level("ERROR")


def _process_ica_file(args):
    npz_path_str, cfg, epoch_root_str, out_root_str, force = args
    npz_path = Path(npz_path_str)
    out_path = Path(out_root_str) / npz_path.relative_to(Path(epoch_root_str))

    if out_path.exists() and not force:
        return (npz_path.name, "cached", None)

    data = np.load(npz_path, allow_pickle=True)
    epochs = data["epochs"]
    ch_names = list(data["ch_names"])

    try:
        ica_model, artifact_idx = fit_ica(epochs, ch_names, cfg)
        cleaned = apply_ica(epochs, ica_model, artifact_idx, ch_names, cfg)
    except Exception as e:
        return (npz_path.name, "skip", str(e))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        specific=cleaned,
        n_artifacts_removed=np.array(len(artifact_idx)),
        label=data["label"],
        subject_id=data["subject_id"],
        split=data["split"],
    )
    return (npz_path.name, "done", None)


def main(config_path: str, force: bool = False, max_workers: int | None = None) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / "ica"
    out_root.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(epoch_root.rglob("*.npz"))
    args_list = [
        (str(p), cfg, str(epoch_root), str(out_root), force)
        for p in npz_paths
    ]

    n_workers = max_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_process_ica_file, args): args for args in args_list}
        with tqdm(total=len(futures), desc="ICA") as pbar:
            for future in as_completed(futures):
                fname, status, msg = future.result()
                if status == "skip":
                    tqdm.write(f"  SKIP {fname}: {msg}")
                pbar.update(1)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICA on cached epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count())",
    )
    args = parser.parse_args()
    main(args.config, args.force, args.workers)
