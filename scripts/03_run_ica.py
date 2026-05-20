"""Run ICA decomposition on cached epochs."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.ica import fit_ica, apply_ica


def main(config_path: str, force: bool = False) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / "ica"
    out_root.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="ICA"):
        out_path = out_root / npz_path.relative_to(epoch_root)
        if out_path.exists() and not force:
            continue

        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
        ch_names = list(data["ch_names"])

        try:
            ica_model, artifact_idx = fit_ica(epochs, ch_names, cfg)
            cleaned = apply_ica(epochs, ica_model, artifact_idx, ch_names, cfg)
        except Exception as e:
            print(f"  SKIP {npz_path.name}: {e}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            specific=cleaned,
            n_artifacts_removed=np.array(len(artifact_idx)),
            label=data["label"],
            subject_id=data["subject_id"],
            split=data["split"],
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICA on cached epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.force)
