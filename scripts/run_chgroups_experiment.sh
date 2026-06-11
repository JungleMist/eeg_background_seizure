#!/usr/bin/env bash
# run_chgroups_experiment.sh — Channel-group ablation experiment
#
# ICA and raw are INDEPENDENT of channel_groups, so they are computed once
# as pre-steps.  Only the Wiener decomposition varies across experiments.
#
# Pre-steps (run once):
#   01 — Extract epochs       (key-based cache; identical across all configs)
#   03 — ICA decomposition    (does not use channel_groups)
#
# Per experiment (×5):
#   Clear cache/wiener_frequency/ and cache/features/wiener_*.npz only.
#   Keep cache/ica/ and cache/features/{raw,ica}_*.npz across experiments so
#   that 06 --condition raw/ica are feature-cache hits from experiment 2 onward.
#   02 — Wiener decomposition  (varies by channel_groups)
#   06 — XGBoost + SHAP        (all conditions; raw/ica reuse cached features)
#   07 — Archive experiment
#
# Usage
# -----
#   bash scripts/run_chgroups_experiment.sh [--workers N] [--from N]
#
# Options
#   --workers N   Worker processes for scripts 01/02/03/06 (default: 1)
#   --from N      Resume from config N (1–5); skips pre-steps and earlier configs.
#                 Assumes 01 and 03 have already completed successfully.
#
# Requirements: conda env eeg_pipeline must exist.

set -euo pipefail

CONDA_RUN="conda run -n eeg_pipeline"
WORKERS=1
FROM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --from)    FROM="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIGS=(
    ""
    "configs/exp_chgroups_1.yaml|Frontal Only"
    "configs/exp_chgroups_2.yaml|Frontal + Temporal"
    "configs/exp_chgroups_3.yaml|Frontal + Temporal + Occipital"
    "configs/exp_chgroups_4.yaml|Default G1-G6"
    "configs/exp_chgroups_5.yaml|All Bilateral Pairs"
)

run_step() {
    local label="$1"; shift
    echo ""
    echo "  -- $label"
    echo "  cmd: $*"
    local t0=$SECONDS
    "$@"
    echo "  [OK] $(( SECONDS - t0 ))s"
}

# ---------------------------------------------------------------------------
# PRE-STEPS: epochs and ICA — run once, shared across all experiments
# ---------------------------------------------------------------------------
if (( FROM == 1 )); then
    echo "============================================================"
    echo "  PRE-STEPS: epochs + ICA (independent of channel groups)"
    echo "============================================================"

    run_step "01 — Extract epochs" \
        $CONDA_RUN python scripts/01_extract_epochs.py \
            --config configs/exp_chgroups_1.yaml \
            --workers "$WORKERS"

    run_step "03 — ICA decomposition" \
        $CONDA_RUN python scripts/03_run_ica.py \
            --config configs/exp_chgroups_1.yaml \
            --force --workers "$WORKERS"
else
    echo "  --from $FROM: skipping pre-steps (01 + 03 assumed done)"
fi

# ---------------------------------------------------------------------------
# Per-experiment loop: Wiener → XGBoost (all conditions) → Archive
# ---------------------------------------------------------------------------
for IDX in 1 2 3 4 5; do
    if (( IDX < FROM )); then
        echo ""
        echo "  Skipping config $IDX (--from $FROM)"
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

    # Clear only Wiener cache and Wiener feature cache.
    # Raw and ICA feature caches (cache/features/{raw,ica}_*.npz) are kept so
    # that 06 --condition raw/ica are cache hits from experiment 2 onward.
    echo "  Clearing wiener caches..."
    rm -rf cache/wiener_frequency/
    rm -f  cache/features/wiener_*.npz

    run_step "02 — Wiener decomposition" \
        $CONDA_RUN python scripts/02_run_wiener.py \
            --config "$CFG" --force --workers "$WORKERS"

    run_step "06 — XGBoost + SHAP (all conditions)" \
        $CONDA_RUN python scripts/06_train_xgboost.py \
            --config "$CFG" --condition all --force --workers "$WORKERS"

    run_step "07 — Archive experiment" \
        $CONDA_RUN python scripts/07_organize_experiment.py \
            --config "$CFG" --name "chgroups-$IDX"

    echo ""
    echo "  Config $IDX complete — results in results/exp_chgroups/$IDX/"
done

# ---------------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Final cache cleanup"
echo "============================================================"
rm -rf cache/wiener_frequency/
rm -rf cache/ica/
rm -rf cache/features/

# ---------------------------------------------------------------------------
# Summary table (wiener condition test metrics)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  CHANNEL-GROUP ABLATION — RESULTS SUMMARY"
echo "============================================================"
printf "  %-2s  %-40s  %s\n" "#" "Config" "AUROC(raw / ica / wiener)"
echo "  ------------------------------------------------------------------------"

for IDX in 1 2 3 4 5; do
    ENTRY="${CONFIGS[$IDX]}"
    LABEL="${ENTRY##*|}"
    get_auroc() {
        local f="results/exp_chgroups/$IDX/xgboost/$1/test_metrics.json"
        [[ -f "$f" ]] && python -c "import json; print(f\"{json.load(open('$f'))['auroc']:.4f}\")" || echo "n/a"
    }
    RAW=$(get_auroc raw)
    ICA=$(get_auroc ica)
    WIE=$(get_auroc wiener)
    printf "  %-2s  %-40s  %s / %s / %s\n" "$IDX" "$LABEL" "$RAW" "$ICA" "$WIE"
done

echo ""
echo "Done."
