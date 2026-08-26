#!/usr/bin/env bash
# Run TUAB continuous ECMAD preprocessing and component EEGNet training for
# phase-gate thresholds 0.01, 0.05, 0.1, 0.5, and 1.

set -euo pipefail

WORKERS=1
CHECK_CONFIG=0
FROM_PHASE=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_tuab_component_eegnet_phase_sweep.sh [options]

Options:
  --workers N       Worker processes for scripts 17, 18, and 19 (default: 1)
  --from PHASE      Start from this phase (0.01, 0.05, 0.1, 0.5, or 1)
  --check-config    Validate configs and cleanup boundaries, then exit
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            if [[ $# -lt 2 ]]; then
                echo "--workers requires an integer argument" >&2
                exit 1
            fi
            WORKERS="$2"
            shift 2
            ;;
        --from)
            if [[ $# -lt 2 ]]; then
                echo "--from requires a phase argument" >&2
                exit 1
            fi
            FROM_PHASE="$2"
            shift 2
            ;;
        --check-config)
            CHECK_CONFIG=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--workers must be a positive integer" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_RUN=(conda run -n eeg_pipeline --no-capture-output)
CONFIGS=(
    "configs/tuab_phase001.yaml"
    "configs/tuab_phase005.yaml"
    "configs/tuab_phase01.yaml"
    "configs/tuab_phase05.yaml"
    "configs/tuab_phase1.yaml"
)
PHASES=("0.01" "0.05" "0.1" "0.5" "1")
RESULT_DIR_NAMES=(
    "results_tuab_phase001"
    "results_tuab_phase005"
    "results_tuab_phase01"
    "results_tuab_phase05"
    "results_tuab_phase1"
)
START_INDEX=0
if [[ -n "$FROM_PHASE" ]]; then
    START_INDEX=-1
    for index in "${!PHASES[@]}"; do
        if [[ "${PHASES[$index]}" == "$FROM_PHASE" ]]; then
            START_INDEX="$index"
            break
        fi
    done
    if (( START_INDEX < 0 )); then
        echo "--from must be one of: ${PHASES[*]}" >&2
        exit 1
    fi
fi
EXPECTED_DATA_ROOT="/root/autodl-tmp/data/v3.0.1"
EXPECTED_CACHE_DIR="/root/autodl-tmp/cache"

config_value() {
    local config="$1"
    local dotted_key="$2"
    "${CONDA_RUN[@]}" python - "$config" "$dotted_key" <<'PY'
import sys

from eeg_bg.config.settings import load_config

config = load_config(sys.argv[1])
value = config
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

validate_configs() {
    local shared_cache=""
    local seen_results="|"
    local index config phase result_name
    local dataset mode gate data_root cache_dir results_dir resolved_phase

    for index in "${!CONFIGS[@]}"; do
        config="${CONFIGS[$index]}"
        phase="${PHASES[$index]}"
        result_name="${RESULT_DIR_NAMES[$index]}"
        if [[ ! -f "$config" ]]; then
            echo "Missing config: $config" >&2
            return 1
        fi

        dataset="$(config_value "$config" "dataset.active")"
        mode="$(config_value "$config" "wiener.mode")"
        gate="$(config_value "$config" "wiener.coherent_gate_enabled")"
        resolved_phase="$(config_value "$config" "wiener.phase_gate_threshold_rad")"
        data_root="$(config_value "$config" "paths.data_root")"
        cache_dir="$(config_value "$config" "paths.cache_dir")"
        results_dir="$(config_value "$config" "paths.results_dir")"

        [[ "$dataset" == "tuab" ]] || {
            echo "Expected dataset.active=tuab in $config; got $dataset" >&2
            return 1
        }
        [[ "$mode" == "phasegated" ]] || {
            echo "Expected wiener.mode=phasegated in $config; got $mode" >&2
            return 1
        }
        [[ "$gate" == "False" ]] || {
            echo "Expected coherent_gate_enabled=false in $config; got $gate" >&2
            return 1
        }
        [[ "$resolved_phase" == "$phase" ]] || {
            echo "Expected phase $phase in $config; got $resolved_phase" >&2
            return 1
        }
        [[ "$data_root" == "$EXPECTED_DATA_ROOT" ]] || {
            echo "Expected data_root=$EXPECTED_DATA_ROOT in $config; got $data_root" >&2
            return 1
        }
        [[ "$cache_dir" == "$EXPECTED_CACHE_DIR" ]] || {
            echo "Expected cache_dir=$EXPECTED_CACHE_DIR in $config; got $cache_dir" >&2
            return 1
        }
        [[ "$(basename "$results_dir")" == "$result_name" ]] || {
            echo "Expected results_dir ending in $result_name in $config; got $results_dir" >&2
            return 1
        }
        if [[ "$seen_results" == *"|$results_dir|"* ]]; then
            echo "Duplicate results_dir: $results_dir" >&2
            return 1
        fi
        seen_results="${seen_results}${results_dir}|"

        if [[ -z "$shared_cache" ]]; then
            shared_cache="$cache_dir"
        elif [[ "$cache_dir" != "$shared_cache" ]]; then
            echo "Configs do not share one cache_dir" >&2
            return 1
        fi
        echo "Validated phase=$phase config=$config results=$results_dir"
    done

    CACHE_DIR="$shared_cache"
    PHASE_CACHE_DIR="$CACHE_DIR/tuab_continuous_wiener_phasegated"
    if [[ "$PHASE_CACHE_DIR" != "$EXPECTED_CACHE_DIR/tuab_continuous_wiener_phasegated" ]]; then
        echo "Unsafe cleanup target: $PHASE_CACHE_DIR" >&2
        return 1
    fi
    echo "Validated cleanup target: $PHASE_CACHE_DIR"
}

safe_clear_phase_cache() {
    if [[ -z "${PHASE_CACHE_DIR:-}" || "$PHASE_CACHE_DIR" == "/" ||
          "$PHASE_CACHE_DIR" == "$CACHE_DIR" ||
          "$PHASE_CACHE_DIR" != "$EXPECTED_CACHE_DIR/tuab_continuous_wiener_phasegated" ]]; then
        echo "Refusing to clear unsafe cache path: ${PHASE_CACHE_DIR:-<empty>}" >&2
        return 1
    fi
    if [[ -e "$PHASE_CACHE_DIR" ]]; then
        echo "Clearing phase cache: $PHASE_CACHE_DIR"
        rm -rf -- "$PHASE_CACHE_DIR"
        echo "Cleared phase cache: $PHASE_CACHE_DIR"
    else
        echo "Phase cache already absent: $PHASE_CACHE_DIR"
    fi
}

print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

run_step() {
    local label="$1"
    shift
    local started_at elapsed status
    started_at=$SECONDS
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $label"
    print_command "$@"
    if "$@"; then
        status=0
    else
        status=$?
    fi
    elapsed=$((SECONDS - started_at))
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $label status=$status elapsed=${elapsed}s"
    return "$status"
}

validate_configs

if (( CHECK_CONFIG )); then
    echo "Configuration check passed; no cache was removed and no log was created."
    exit 0
fi

LOG_DIR="$PROJECT_ROOT/results_tuab_phase_sweep/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/tuab_component_eegnet_phase_sweep_$(date '+%Y-%m-%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1

SWEEP_STARTED_AT=$SECONDS
on_exit() {
    local status=$?
    local elapsed=$((SECONDS - SWEEP_STARTED_AT))
    trap - EXIT
    if (( status == 0 )); then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SWEEP COMPLETE status=0 elapsed=${elapsed}s"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SWEEP FAILED status=$status elapsed=${elapsed}s"
        echo "Failed-run cache was retained at: $PHASE_CACHE_DIR"
    fi
    echo "Log file: $LOG_FILE"
    exit "$status"
}
trap on_exit EXIT

echo "# TUAB component EEGNet phase sweep"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Host: $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Git commit: $(git rev-parse HEAD)"
echo "Workers for scripts 17/18/19: $WORKERS"
echo "Phase order: ${PHASES[*]}"
echo "Starting phase: ${PHASES[$START_INDEX]}"
echo "Configs: ${CONFIGS[*]}"
echo "Cleanup target: $PHASE_CACHE_DIR"
echo "Log file: $LOG_FILE"

for ((index = START_INDEX; index < ${#CONFIGS[@]}; index++)); do
    config="${CONFIGS[$index]}"
    phase="${PHASES[$index]}"
    results_dir="$(config_value "$config" "paths.results_dir")"
    output_dir="$results_dir/tuab_component_eegnet_phasegated"

    echo ""
    echo "============================================================"
    echo "Phase $phase ($((index + 1))/${#CONFIGS[@]}): $config"
    echo "============================================================"

    safe_clear_phase_cache
    run_step "phase=$phase script=17 continuous ECMAD cache" \
        "${CONDA_RUN[@]}" python scripts/17_cache_tuab_continuous_wiener.py \
        --config "$config" --workers "$WORKERS"
    run_step "phase=$phase script=18 paired epoch cache" \
        "${CONDA_RUN[@]}" python scripts/18_extract_tuab_component_epochs.py \
        --config "$config" --workers "$WORKERS"
    run_step "phase=$phase script=19 component EEGNet" \
        "${CONDA_RUN[@]}" python scripts/19_train_tuab_component_eegnet.py \
        --config "$config" --workers "$WORKERS" --force

    for artifact in run_summary.json condition_metrics.csv; do
        if [[ ! -s "$output_dir/$artifact" ]]; then
            echo "Missing or empty result artifact: $output_dir/$artifact" >&2
            exit 1
        fi
        echo "Verified result artifact: $output_dir/$artifact"
    done
    safe_clear_phase_cache
    echo "Phase $phase completed successfully."
done
