"""Run Wiener decomposition on cached epochs."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition import wiener as wiener_freq
from eeg_bg.decomposition import wiener_scalar


def main(config_path: str, mode: str = "frequency", force: bool = False) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    out_root = Path(cfg["paths"]["cache_dir"]) / f"wiener_{mode}"
    out_root.mkdir(parents=True, exist_ok=True)

    decompose = wiener_freq.decompose_epoch if mode == "frequency" else wiener_scalar.decompose_epoch

    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="Wiener"):
        out_path = out_root / npz_path.relative_to(epoch_root)
        if out_path.exists() and not force:
            continue

        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
        ch_names = list(data["ch_names"])
        subject_id = str(data["subject_id"])

        results = []
        for i, epoch in enumerate(epochs):
            r = decompose(epoch, ch_names, cfg, subject_id=subject_id, epoch_idx=i)
            results.append({
                "specific": r.specific,
                "coherent": r.coherent,
                "skipped_pairs": np.array(r.skipped_pairs),
            })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            specific=np.stack([r["specific"] for r in results]),
            coherent=np.stack([r["coherent"] for r in results]),
            label=data["label"],
            subject_id=data["subject_id"],
            split=data["split"],
        )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wiener decomposition on cached epochs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["frequency", "scalar"], default="frequency")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.config, args.mode, args.force)
