"""End-to-end smoke test: runs scripts 01-07 with configs/smoke_test.yaml.

Usage
-----
conda run -n eeg_pipeline python scripts/run_smoke_test.py
"""
import subprocess
import sys
import time
from pathlib import Path

CONFIG = "configs/smoke_test.yaml"
PYTHON = sys.executable

def run(cmd: list[str], label: str, timeout: int = 600) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=timeout)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{status}] in {elapsed:.1f}s")
    return result.returncode == 0


steps = [
    (["python", "scripts/01_extract_epochs.py", "--config", CONFIG, "--force", "--workers", "1"],
     "01 — Extract epochs"),
    (["python", "scripts/02_run_wiener.py",     "--config", CONFIG, "--force", "--workers", "1"],
     "02 — Wiener decomposition"),
    (["python", "scripts/03_run_ica.py",        "--config", CONFIG, "--force", "--workers", "1"],
     "03 — ICA"),
    (["python", "scripts/04_run_verification.py", "--config", CONFIG, "--workers", "1"],
     "04 — Verification"),
    (["python", "scripts/05_run_visualization.py", "--config", CONFIG, "--n-subjects", "2"],
     "05 — Visualization"),
    (["python", "scripts/06_train_xgboost.py",  "--config", CONFIG, "--force", "--workers", "1"],
     "06 — XGBoost + SHAP"),
    (["python", "scripts/07_organize_experiment.py", "--config", CONFIG, "--name", "smoke"],
     "07 — Archive experiment"),
]

failed = []
for cmd, label in steps:
    ok = run(cmd, label)
    if not ok:
        failed.append(label)
        print(f"\n  STOPPING — {label} failed.")
        break

print("\n" + "="*60)
if failed:
    print(f"SMOKE TEST FAILED at: {failed[0]}")
    sys.exit(1)
else:
    print("SMOKE TEST PASSED — all scripts 01-07 completed successfully.")
