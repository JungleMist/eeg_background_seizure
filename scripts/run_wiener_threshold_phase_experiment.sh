#!/usr/bin/env bash
# run_wiener_threshold_phase_experiment.sh
#
# Runs the 8-way Wiener coherence-threshold / phase-gate ablation.
#
# Pipeline:
#   01 once -> exp1-8 Wiener+XGBoost -> baseline raw/ICA -> copy baseline
#   into each experiment result directory -> 07 archive each experiment.
#
# XGBoost condition names follow the codebase API:
#   02 --mode frequency   -> 06 --condition wiener
#   02 --mode phasegated  -> 06 --condition wiener_phasegated

set -euo pipefail

CONDA_RUN="${CONDA_RUN:-conda run -n eeg_pipeline}"
PROGRAM_NAME="${PROGRAM_NAME:-scripts/run_wiener_threshold_phase_experiment.sh}"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-TUEP WIENER THRESHOLD/PHASE ABLATION}"
CONFIG_PREFIX="${CONFIG_PREFIX:-configs/exp_wiener_phase}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/exp_wiener_phase_baseline.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-results/exp_wiener_phase}"
ARCHIVE_PREFIX="${ARCHIVE_PREFIX:-wiener-phase}"
EXPECTED_DATASET="${EXPECTED_DATASET:-tuep}"
RUN_GRID_ANALYSIS="${RUN_GRID_ANALYSIS:-0}"
WORKERS=16
FROM=1
CLEAR_CACHE=0
SKIP_EPOCHS=0
SKIP_BASELINE=0
CHECK_CONFIG=0

usage() {
    cat <<EOF
Usage:
  bash ${PROGRAM_NAME} [options]

Options:
  --workers N       Worker processes for scripts 01/02/03 (default: 16)
  --from N          Resume per-experiment loop from exp N, 1-8 (default: 1)
  --clear-cache     Remove the configured cache_dir before script 01
  --skip-epochs     Skip script 01; assumes cache/epochs already exists
  --skip-baseline   Skip script 03 and raw/ICA XGBoost; assumes baseline exists
  --check-config    Validate the experiment configs and exit without running
  -h, --help        Show this help

Cache policy:
  epochs/ is retained as shared input. wiener_frequency/, wiener_phasegated/,
  and ica/ are mutually exclusive and are deleted after feature extraction.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --from) FROM="$2"; shift 2 ;;
        --clear-cache) CLEAR_CACHE=1; shift ;;
        --skip-epochs) SKIP_EPOCHS=1; shift ;;
        --skip-baseline) SKIP_BASELINE=1; shift ;;
        --check-config) CHECK_CONFIG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if (( FROM < 1 || FROM > 8 )); then
    echo "--from must be between 1 and 8"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$RESULTS_ROOT"

CONFIGS=(
    ""
    "${CONFIG_PREFIX}_1.yaml"
    "${CONFIG_PREFIX}_2.yaml"
    "${CONFIG_PREFIX}_3.yaml"
    "${CONFIG_PREFIX}_4.yaml"
    "${CONFIG_PREFIX}_5.yaml"
    "${CONFIG_PREFIX}_6.yaml"
    "${CONFIG_PREFIX}_7.yaml"
    "${CONFIG_PREFIX}_8.yaml"
)
MODES=("" "frequency" "frequency" "frequency" "phasegated" "phasegated" "phasegated" "phasegated" "phasegated")
CONDITIONS=("" "wiener" "wiener" "wiener" "wiener_phasegated" "wiener_phasegated" "wiener_phasegated" "wiener_phasegated" "wiener_phasegated")
DERIVED_CACHE_SUBDIRS=("wiener_frequency" "wiener_phasegated" "ica")
LABELS=(
    ""
    "coh=0.15 phase=pi frequency"
    "coh=0.45 phase=pi frequency"
    "coh=0.75 phase=pi frequency"
    "coh=0.15 phase=pi/2 phasegated"
    "coh=0.15 phase=pi/5 phasegated"
    "coh=0.15 phase=pi/10 phasegated"
    "coh=0.45 phase=pi/10 phasegated"
    "coh=0.75 phase=pi/10 phasegated"
)

log_step() {
    local elapsed="$1"
    local label="$2"
    local ts
    ts=$(date +"%Y-%m-%d %H:%M:%S")
    printf "%s | %6ds | %s\n" "$ts" "$elapsed" "$label" | tee -a "$LOG_FILE"
}

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

config_value() {
    local cfg="$1"
    local dotted_key="$2"
    $CONDA_RUN python -c "
from eeg_bg.config.settings import load_config
cfg = load_config('$cfg')
node = cfg
for part in '$dotted_key'.split('.'):
    node = node[part]
print(node)
"
}

safe_clear_dir() {
    local target="$1"
    local label="$2"
    if [[ -z "$target" || "$target" == "/" ]]; then
        echo "Refusing to clear unsafe $label path: '$target'"
        exit 1
    fi
    echo "  Clearing $label: $target"
    rm -rf "$target"
}

clear_derived_caches() {
    local cache_dir="$1"
    local subdir
    for subdir in "${DERIVED_CACHE_SUBDIRS[@]}"; do
        safe_clear_dir "$cache_dir/$subdir" "$subdir cache"
    done
}

assert_derived_cache_exclusive() {
    local cache_dir="$1"
    local count=0
    local present=()
    local subdir
    for subdir in "${DERIVED_CACHE_SUBDIRS[@]}"; do
        if [[ -d "$cache_dir/$subdir" ]]; then
            count=$(( count + 1 ))
            present+=("$subdir")
        fi
    done
    if (( count > 1 )); then
        echo "Derived cache invariant violated: ${present[*]} coexist in $cache_dir"
        exit 1
    fi
}

clear_baseline_feature_caches() {
    local cache_dir="$1"
    rm -f "$cache_dir/features/raw_"*.npz
    rm -f "$cache_dir/features/ica_"*.npz
    rm -f "$cache_dir/features/base211_conn80/raw_"*.npz
    rm -f "$cache_dir/features/base211_conn80/ica_"*.npz
}

clean_condition_cache() {
    local cfg="$1"
    local condition="$2"
    local cache_dir
    cache_dir="$(config_value "$cfg" "paths.cache_dir")"
    clear_derived_caches "$cache_dir"
    rm -f "$cache_dir/features/${condition}_"*.npz
    rm -f "$cache_dir/features/base211_conn80/${condition}_"*.npz
}

prepare_condition_results() {
    local cfg="$1"
    local condition="$2"
    local results_dir
    results_dir="$(config_value "$cfg" "paths.results_dir")"
    for profile in base211 base211_conn80; do
        safe_clear_dir "$results_dir/xgboost/$profile/$condition" "$profile/$condition results"
    done
    # Script 04 is intentionally skipped for this experiment. Remove any
    # stale verification output so script 07 cannot archive results from an
    # earlier run as if they belonged to the current condition.
    safe_clear_dir "$results_dir/verification" "stale verification results"
}

clean_baseline_cache_and_results() {
    local cfg="$BASELINE_CONFIG"
    local cache_dir results_dir
    cache_dir="$(config_value "$cfg" "paths.cache_dir")"
    results_dir="$(config_value "$cfg" "paths.results_dir")"

    clear_derived_caches "$cache_dir"
    clear_baseline_feature_caches "$cache_dir"
    for profile in base211 base211_conn80; do
        safe_clear_dir "$results_dir/xgboost/$profile/raw" "$profile baseline raw results"
        safe_clear_dir "$results_dir/xgboost/$profile/ica" "$profile baseline ica results"
    done
}

copy_baseline_results() {
    local idx="$1"
    local cfg="${CONFIGS[$idx]}"
    local results_dir baseline_dir
    results_dir="$(config_value "$cfg" "paths.results_dir")"
    baseline_dir="$(config_value "$BASELINE_CONFIG" "paths.results_dir")"

    for profile in base211 base211_conn80; do
        if [[ ! -d "$baseline_dir/xgboost/$profile/raw" || ! -d "$baseline_dir/xgboost/$profile/ica" ]]; then
            echo "Missing $profile baseline raw/ica results in $baseline_dir/xgboost"
            exit 1
        fi
        mkdir -p "$results_dir/xgboost/$profile"
        safe_clear_dir "$results_dir/xgboost/$profile/raw" "exp${idx} $profile raw baseline copy"
        safe_clear_dir "$results_dir/xgboost/$profile/ica" "exp${idx} $profile ica baseline copy"
        cp -R "$baseline_dir/xgboost/$profile/raw" "$results_dir/xgboost/$profile/"
        cp -R "$baseline_dir/xgboost/$profile/ica" "$results_dir/xgboost/$profile/"
    done
}

BASE_CFG="${CONFIGS[1]}"
CACHE_DIR="$(config_value "$BASE_CFG" "paths.cache_dir")"
for CFG in "$BASELINE_CONFIG" "${CONFIGS[@]:1}"; do
    ACTIVE_DATASET="$(config_value "$CFG" "dataset.active")"
    CFG_CACHE_DIR="$(config_value "$CFG" "paths.cache_dir")"
    if [[ "$ACTIVE_DATASET" != "$EXPECTED_DATASET" ]]; then
        echo "Expected dataset.active=$EXPECTED_DATASET, got $ACTIVE_DATASET from $CFG"
        exit 1
    fi
    if [[ "$CFG_CACHE_DIR" != "$CACHE_DIR" ]]; then
        echo "Expected shared cache_dir=$CACHE_DIR, got $CFG_CACHE_DIR from $CFG"
        exit 1
    fi
done

if (( CHECK_CONFIG )); then
    echo "Validated 8 grid entries for dataset=$EXPECTED_DATASET"
    echo "Shared cache_dir: $CACHE_DIR"
    echo "Results root: $RESULTS_ROOT"
    exit 0
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/runtime_$(date +%Y-%m-%d_%H%M%S).log"
EXPERIMENT_START=$SECONDS
echo "# $PROGRAM_NAME started $(date)" | tee "$LOG_FILE"
echo "# workers=$WORKERS from=$FROM clear_cache=$CLEAR_CACHE skip_epochs=$SKIP_EPOCHS skip_baseline=$SKIP_BASELINE" | tee -a "$LOG_FILE"
echo "#" | tee -a "$LOG_FILE"

if (( CLEAR_CACHE )); then
    safe_clear_dir "$CACHE_DIR" "full cache_dir"
fi

# epochs/ is the shared input to scripts 02 and 03. Keep it, but ensure the
# three large derived caches never coexist, including stale data from a prior run.
clear_derived_caches "$CACHE_DIR"
assert_derived_cache_exclusive "$CACHE_DIR"

if (( SKIP_EPOCHS )); then
    echo ""
    echo "  -- Skipping 01; assuming epochs already exist in $CACHE_DIR/epochs"
else
    run_step "01 - Extract epochs" \
        $CONDA_RUN python scripts/01_extract_epochs.py \
            --config "$BASE_CFG" \
            --workers "$WORKERS"
fi

for IDX in 1 2 3 4 5 6 7 8; do
    if (( IDX < FROM )); then
        echo ""
        echo "  Skipping exp$IDX (--from $FROM)"
        continue
    fi

    CFG="${CONFIGS[$IDX]}"
    MODE="${MODES[$IDX]}"
    CONDITION="${CONDITIONS[$IDX]}"
    LABEL="${LABELS[$IDX]}"

    echo ""
    echo "============================================================"
    echo "  exp$IDX / 8 : $LABEL"
    echo "  Config      : $CFG"
    echo "  02 mode     : $MODE"
    echo "  06 condition: $CONDITION"
    echo "============================================================"

    clean_condition_cache "$CFG" "$CONDITION"
    prepare_condition_results "$CFG" "$CONDITION"

    run_step "exp$IDX - 02 Wiener $MODE" \
        $CONDA_RUN python scripts/02_run_wiener.py \
            --config "$CFG" \
            --mode "$MODE" \
            --force \
            --workers "$WORKERS"
    assert_derived_cache_exclusive "$CACHE_DIR"

    for FEATURE_SET in base211 base211_conn80; do
        run_step "exp$IDX - 06 XGBoost $CONDITION ($FEATURE_SET)" \
            $CONDA_RUN python scripts/06_train_xgboost.py \
                --config "$CFG" \
                --condition "$CONDITION" \
                --feature-set "$FEATURE_SET" \
                --force
    done

    clean_condition_cache "$CFG" "$CONDITION"
    assert_derived_cache_exclusive "$CACHE_DIR"
done

if (( SKIP_BASELINE )); then
    echo ""
    echo "  -- Skipping baseline; assuming baseline raw/ica results already exist"
else
    echo ""
    echo "============================================================"
    echo "  Baseline: ICA + XGBoost raw/ica"
    echo "============================================================"

    clean_baseline_cache_and_results

    run_step "03 - ICA decomposition" \
        $CONDA_RUN python scripts/03_run_ica.py \
            --config "$BASELINE_CONFIG" \
            --force \
            --workers "$WORKERS"
    assert_derived_cache_exclusive "$CACHE_DIR"

    for FEATURE_SET in base211 base211_conn80; do
        run_step "06 - XGBoost raw baseline ($FEATURE_SET)" \
            $CONDA_RUN python scripts/06_train_xgboost.py \
                --config "$BASELINE_CONFIG" \
                --condition raw \
                --feature-set "$FEATURE_SET" \
                --force

        run_step "06 - XGBoost ica baseline ($FEATURE_SET)" \
            $CONDA_RUN python scripts/06_train_xgboost.py \
                --config "$BASELINE_CONFIG" \
                --condition ica \
                --feature-set "$FEATURE_SET" \
                --force
    done

    # Models and predictions are already persisted in results/. The ICA tensor
    # cache and the temporary raw/ICA feature matrices are no longer needed.
    clear_derived_caches "$CACHE_DIR"
    clear_baseline_feature_caches "$CACHE_DIR"
    assert_derived_cache_exclusive "$CACHE_DIR"
fi

echo ""
echo "============================================================"
echo "  Copy baseline into each experiment"
echo "============================================================"
for IDX in 1 2 3 4 5 6 7 8; do
    run_step "exp$IDX - copy raw/ica baseline" copy_baseline_results "$IDX"
done

echo ""
echo "============================================================"
echo "  Archive experiments with script 07"
echo "============================================================"
for IDX in 1 2 3 4 5 6 7 8; do
    CFG="${CONFIGS[$IDX]}"
    run_step "exp$IDX - 07 archive" \
        $CONDA_RUN python scripts/07_organize_experiment.py \
            --config "$CFG" \
            --name "${ARCHIVE_PREFIX}-exp$IDX"
done

if (( RUN_GRID_ANALYSIS )); then
    run_step "09 - Analyze phase/coherence grid" \
        $CONDA_RUN python scripts/09_analyze_wiener_phase_grid.py \
            --results-root "$RESULTS_ROOT" \
            --config-dir "$(dirname "$CONFIG_PREFIX")" \
            --config-prefix "$(basename "$CONFIG_PREFIX")"
fi

echo ""
echo "============================================================"
echo "  $EXPERIMENT_LABEL - RESULTS SUMMARY"
echo "============================================================"
printf "  %-4s %-36s %s\n" "Exp" "Setting" "Test AUROC(raw / ica / wiener-condition)"
echo "  --------------------------------------------------------------------------------"

for IDX in 1 2 3 4 5 6 7 8; do
    CFG="${CONFIGS[$IDX]}"
    CONDITION="${CONDITIONS[$IDX]}"
    RESULTS_DIR="$(config_value "$CFG" "paths.results_dir")"
    get_auroc() {
        local f="$RESULTS_DIR/xgboost/base211/$1/test_metrics.json"
        [[ -f "$f" ]] && $CONDA_RUN python -c "import json; print(f\"{json.load(open('$f'))['auroc']:.4f}\")" || echo "n/a"
    }
    RAW=$(get_auroc raw)
    ICA=$(get_auroc ica)
    WIE=$(get_auroc "$CONDITION")
    printf "  %-4s %-36s %s / %s / %s\n" "exp$IDX" "${LABELS[$IDX]}" "$RAW" "$ICA" "$WIE"
done

TOTAL=$(( SECONDS - EXPERIMENT_START ))
echo ""
printf "Total wall time: %dh %dm %ds\n" \
    $(( TOTAL / 3600 )) $(( (TOTAL % 3600) / 60 )) $(( TOTAL % 60 ))
echo "Runtime log: $LOG_FILE"
echo "Done."

echo "#" | tee -a "$LOG_FILE"
printf "# total wall time: %dh %dm %ds\n" \
    $(( TOTAL / 3600 )) $(( (TOTAL % 3600) / 60 )) $(( TOTAL % 60 )) | tee -a "$LOG_FILE"
echo "# finished $(date)" | tee -a "$LOG_FILE"
