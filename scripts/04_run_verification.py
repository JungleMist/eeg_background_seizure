"""Run physical verification experiments V1/V2/V3 and save CSV reports."""
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from eeg_bg.config.settings import load_config
from eeg_bg.decomposition.wiener import decompose_epoch, WienerResult
from eeg_bg.verification.coherence import run_v1
from eeg_bg.verification.transitivity import run_v2, run_v3


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    epoch_root = Path(cfg["paths"]["cache_dir"]) / "epochs"
    verif_dir = Path(cfg["paths"]["results_dir"]) / "verification"
    verif_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[WienerResult] = []
    for npz_path in tqdm(sorted(epoch_root.rglob("*.npz")), desc="V1/V2/V3"):
        data = np.load(npz_path, allow_pickle=True)
        epochs = data["epochs"]
        ch_names = list(data["ch_names"])
        subject_id = str(data["subject_id"])
        for i, epoch in enumerate(epochs):
            r = decompose_epoch(epoch, ch_names, cfg,
                                subject_id=subject_id, epoch_idx=i)
            all_results.append(r)

    print(f"Running V1 on {len(all_results)} epochs...")
    v1_df = run_v1(all_results, cfg)
    v1_df.to_csv(verif_dir / "v1_coherence.csv", index=False)

    print("Running V2...")
    v2_df = run_v2(all_results, cfg)
    v2_df.to_csv(verif_dir / "v2_transitivity.csv", index=False)

    print("Running V3...")
    v3_df = run_v3(all_results, cfg)
    v3_df.to_csv(verif_dir / "v3_frequency_variation.csv", index=False)

    print(f"V1: {len(v1_df)} rows | V2: {len(v2_df)} rows | V3: {len(v3_df)} rows")
    print(f"Results saved to {verif_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physical verification experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
