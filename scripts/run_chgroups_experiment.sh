#!/usr/bin/env bash
# run_chgroups_experiment.sh — Channel-group ablation experiment
#
# Runs scripts 01→02→03→06 for each of the 5 channel-group configurations.
# After each experiment the variable caches (wiener_frequency, ica, features)
# are removed to keep disk usage low.  The epoch cache (cache/epochs/) is
# preserved throughout — it is identical across all experiments and expensive
# to regenerate.
#
# Usage
# -----
#   bash scripts/run_chgroups_experiment.sh [--workers N] [--from N]
#
# Options
#   --workers N   Worker processes for scripts 01/02/03/06 (default: 1)
#   --from N      Resume from config N (1–5); skips earlier configs
#
# Cache layout after completion:
#   cache/epochs/          — kept (shared across all experiments)
#   cache/wiener_frequency — removed after each experiment
#   cache/ica/             — removed after each experiment
#   cache/features/        — removed after each experiment
#   results/exp_chgroups/{1..5}/  — permanent, one per config
#
# Requirements: conda env eeg_pipeline must exist.

set -euo pipefail

CONDA_RUN="conda run -n eeg_pipeline"
WORKERS=1
FROM=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --from)    FROM="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Project root = directory containing this script's parent
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIGS=(
    ""                                                          # placeholder so index starts at 1
    "configs/exp_chgroups_1.yaml|Frontal Only"
    "configs/exp_chgroups_2.yaml|Frontal + Temporal"
    "configs/exp_chgroups_3.yaml|Frontal + Temporal + Occipital"
    "configs/exp_chgroups_4.yaml|Default G1-G6"
    "configs/exp_chgroups_5.yaml|All Bilateral Pairs"
)

# ---------------------------------------------------------------------------
# Helper: run one command, print label and timing, abort on failure
# ---------------------------------------------------------------------------
run_step() {
    local label="$1"; shift
    echo ""
    echo "  -- $label"
    echo "  cmd: $*"
    local t0=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - t0 ))
    echo "  [OK] ${elapsed}s"
}

# ---------------------------------------------------------------------------
# Helper: remove a cache subdirectory if it exists
# ---------------------------------------------------------------------------
clear_cache() {
    local dir="cache/$1"
    if [[ -d "$dir" ]]; then
        echo "  Removing $dir ..."
        rm -rf "$dir"
    fi
}

# ---------------------------------------------------------------------------
# Script 01: run once up front (epochs are shared across all experiments).
# Uses the first config; preprocessing params are identical across all configs.
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  PRE-STEP: Extract epochs (shared across all experiments)"
echo "============================================================"
run_step "01 — Extract epochs" \
    $CONDA_RUN python scripts/01_extract_epochs.py \
        --config configs/exp_chgroups_1.yaml \
        --workers "$WORKERS"

# ---------------------------------------------------------------------------
# Per-experiment loop
# ---------------------------------------------------------------------------
for IDX in 1 2 3 4 5; do
    if (( IDX < FROM )); then
        echo ""
        echo "Skipping config $IDX (--from $FROM)"
        continue
    fi

    ENTRY="${CONFIGS[$IDX]}"
    CFG="${ENTRY%%|*}"
    LABEL="${ENTRY##*|}"

    echo ""
    echo "============================================================"
    echo "  Config $IDX / 5 : $LABEL"
    echo "  Config file     : $CFG"
    echo "============================================================"

    # Clear variable caches before this experiment
    clear_cache "wiener_frequency"
    clear_cache "ica"
    clear_cache "features"

    run_step "02 — Wiener decomposition" \
        $CONDA_RUN python scripts/02_run_wiener.py \
            --config "$CFG" --force --workers "$WORKERS"

    run_step "03 — ICA" \
        $CONDA_RUN python scripts/03_run_ica.py \
            --config "$CFG" --force --workers "$WORKERS"

    run_step "06 — XGBoost + SHAP" \
        $CONDA_RUN python scripts/06_train_xgboost.py \
            --config "$CFG" --force --workers "$WORKERS"

    run_step "07 — Archive experiment" \
        $CONDA_RUN python scripts/07_organize_experiment.py \
            --config "$CFG" --name "chgroups-$IDX"

    echo ""
    echo "  Config $IDX complete — results in results/exp_chgroups/$IDX/"
done

# ---------------------------------------------------------------------------
# Final cleanup: remove remaining variable caches
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Final cache cleanup"
echo "============================================================"
clear_cache "wiener_frequency"
clear_cache "ica"
clear_cache "features"

# ---------------------------------------------------------------------------
# Summary: collect test_metrics.json for the wiener condition from each run
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  CHANNEL-GROUP ABLATION — RESULTS SUMMARY (wiener condition)"
echo "============================================================"
printf "  %-2s  %-40s  %-8s  %-8s  %-8s\n" "#" "Config" "AUROC" "F1" "Acc"
echo "  ----------------------------------------------------------------------------"

for IDX in 1 2 3 4 5; do
    ENTRY="${CONFIGS[$IDX]}"
    LABEL="${ENTRY##*|}"
    METRICS="results/exp_chgroups/$IDX/xgboost/wiener/test_metrics.json"
    if [[ -f "$METRICS" ]]; then
        AUROC=$(python -c "import json; d=json.load(open('$METRICS')); print(f\"{d['auroc']:.4f}\")")
        F1=$(python    -c "import json; d=json.load(open('$METRICS')); print(f\"{d['f1']:.4f}\")")
        ACC=$(python   -c "import json; d=json.load(open('$METRICS')); print(f\"{d['accuracy']:.4f}\")")
        printf "  %-2s  %-40s  %-8s  %-8s  %-8s\n" "$IDX" "$LABEL" "$AUROC" "$F1" "$ACC"
    else
        printf "  %-2s  %-40s  %-8s\n" "$IDX" "$LABEL" "no results"
    fi
done

echo ""
echo "Done."
