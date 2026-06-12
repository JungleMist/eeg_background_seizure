#!/usr/bin/env bash
# run_chgroups_experiment.sh — Channel-group ablation experiment
#
# Minimises total runtime by separating invariant from variable work:
#
# PRE-STEPS (run once, --from 1 only):
#   01 — Extract epochs          (key-based cache; instant cache hit on replay)
#   03 — ICA decomposition       (independent of channel_groups)
#   06 raw + ica                 (features and models are channel_groups-independent;
#                                 raw/ica results land in results/exp_chgroups/1/xgboost/)
#
# PER-EXPERIMENT (×5):
#   Clear cache/wiener_frequency/ and cache/features/wiener_*.npz only.
#   02 — Wiener decomposition    (varies by channel_groups)
#   06 wiener only               (reads shared raw/ica feature cache; writes to
#                                 results/exp_chgroups/N/xgboost/wiener/)
#   Copy raw/ and ica/ results from exp 1 → exp N  (so 07 archives all three)
#   07 — Archive experiment
#
# Usage
# -----
#   bash scripts/run_chgroups_experiment.sh [--workers N] [--from N]
#
# Options
#   --workers N   Worker processes for scripts 01/02/03 (default: 1)
#   --from N      Resume from config N (1–5); skips pre-steps and earlier configs.
#                 Assumes 01, 03, and 06 raw+ica have already completed.
#
# Requirements: conda env eeg_pipeline must exist.

set -euo pipefail

CONDA_RUN="conda run -n eeg_pipeline"
WORKERS=12
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

# Runtime log — one line per step: timestamp | elapsed | label
LOG_DIR="results/exp_chgroups"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/runtime_$(date +%Y-%m-%d_%H%M%S).log"
EXPERIMENT_START=$SECONDS

log_step() {
    # log_step <elapsed_s> <label>
    local elapsed="$1"
    local label="$2"
    local ts
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    printf "%s | %6ds | %s\n" "$ts" "$elapsed" "$label" | tee -a "$LOG_FILE"
}

echo "# run_chgroups_experiment.sh  started $(date)" | tee "$LOG_FILE"
echo "# workers=$WORKERS  from=$FROM" | tee -a "$LOG_FILE"
echo "#" | tee -a "$LOG_FILE"

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
    local elapsed=$(( SECONDS - t0 ))
    echo "  [OK] ${elapsed}s"
    log_step "$elapsed" "$label"
}

# ---------------------------------------------------------------------------
# PRE-STEPS: epochs, ICA, and raw/ica XGBoost — run once
# ---------------------------------------------------------------------------
if (( FROM == 1 )); then
    echo "============================================================"
    echo "  PRE-STEPS: epochs + ICA + XGBoost raw/ica (run once)"
    echo "  Using config: configs/exp_chgroups_1.yaml"
    echo "  Results land in: results/exp_chgroups/1/xgboost/{raw,ica}/"
    echo "============================================================"

    run_step "01 — Extract epochs" \
        $CONDA_RUN python scripts/01_extract_epochs.py \
            --config configs/exp_chgroups_1.yaml \
            --workers "$WORKERS"

    run_step "03 — ICA decomposition" \
        $CONDA_RUN python scripts/03_run_ica.py \
            --config configs/exp_chgroups_1.yaml \
            --workers "$WORKERS"

    # Train raw and ica conditions once — channel_groups has no effect on these.
    # Feature caches (cache/features/raw_*.npz, cache/features/ica_*.npz) and
    # models are written using config 1; models are copied to configs 2-5 below.
    run_step "06 — XGBoost raw + ica (once)" \
        $CONDA_RUN python scripts/06_train_xgboost.py \
            --config configs/exp_chgroups_1.yaml \
            --condition raw

    run_step "06 — XGBoost ica (once)" \
        $CONDA_RUN python scripts/06_train_xgboost.py \
            --config configs/exp_chgroups_1.yaml \
            --condition ica
else
    echo "  --from $FROM: skipping pre-steps (01 + 03 + 06 raw/ica assumed done)"
fi

# ---------------------------------------------------------------------------
# Per-experiment loop: Wiener → XGBoost wiener → copy raw/ica → Archive
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
    # Raw and ICA feature caches (cache/features/{raw,ica}_*.npz) are preserved
    # so that 06 --condition wiener is the only expensive step per experiment.
    # cache_dir is read from the config so this works for both relative and
    # absolute paths (e.g. /root/autodl-tmp/cache on a remote server).
    CACHE_DIR=$($CONDA_RUN python -c "
from eeg_bg.config.settings import load_config
import sys
cfg = load_config('$CFG')
print(cfg['paths']['cache_dir'])
")
    echo "  Clearing wiener caches in: $CACHE_DIR"
    rm -rf  "$CACHE_DIR/wiener_frequency"
    rm -f   "$CACHE_DIR/features/wiener_"*.npz

    run_step "02 — Wiener decomposition" \
        $CONDA_RUN python scripts/02_run_wiener.py \
            --config "$CFG" --workers "$WORKERS"

    run_step "06 — XGBoost wiener" \
        $CONDA_RUN python scripts/06_train_xgboost.py \
            --config "$CFG" --condition wiener

    # For configs 2–5: copy raw/ica results from config 1 so that script 07
    # can archive all three conditions together for this experiment.
    if (( IDX > 1 )); then
        echo ""
        echo "  Copying raw/ica results from exp 1 → exp $IDX..."
        mkdir -p "results/exp_chgroups/$IDX/xgboost"
        cp -r "results/exp_chgroups/1/xgboost/raw" "results/exp_chgroups/$IDX/xgboost/"
        cp -r "results/exp_chgroups/1/xgboost/ica" "results/exp_chgroups/$IDX/xgboost/"
    fi

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
FINAL_CACHE_DIR=$($CONDA_RUN python -c "
from eeg_bg.config.settings import load_config
cfg = load_config('configs/exp_chgroups_1.yaml')
print(cfg['paths']['cache_dir'])
")
echo "  Cache dir: $FINAL_CACHE_DIR"
rm -rf "$FINAL_CACHE_DIR/wiener_frequency"
rm -rf "$FINAL_CACHE_DIR/ica"
rm -rf "$FINAL_CACHE_DIR/features"

# ---------------------------------------------------------------------------
# Summary table
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

TOTAL=$(( SECONDS - EXPERIMENT_START ))
echo ""
printf "Total wall time: %dh %dm %ds\n" \
    $(( TOTAL / 3600 )) $(( (TOTAL % 3600) / 60 )) $(( TOTAL % 60 ))
echo ""
echo "Runtime log: $LOG_FILE"
echo ""
echo "Done."

# Write totals footer to log
echo "#" | tee -a "$LOG_FILE"
printf "# total wall time: %dh %dm %ds\n" \
    $(( TOTAL / 3600 )) $(( (TOTAL % 3600) / 60 )) $(( TOTAL % 60 )) | tee -a "$LOG_FILE"
echo "# finished $(date)" | tee -a "$LOG_FILE"
