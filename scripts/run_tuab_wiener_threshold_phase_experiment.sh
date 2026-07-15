#!/usr/bin/env bash
# TUAB entry point for the shared eight-cell Wiener phase/coherence driver.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PROGRAM_NAME="scripts/run_tuab_wiener_threshold_phase_experiment.sh"
export EXPERIMENT_LABEL="TUAB WIENER THRESHOLD/PHASE ABLATION"
export CONFIG_PREFIX="configs/exp_tuab_wiener_phase"
export BASELINE_CONFIG="configs/exp_tuab_wiener_phase_baseline.yaml"
export RESULTS_ROOT="results_tuab/exp_wiener_phase"
export ARCHIVE_PREFIX="tuab-wiener-phase"
export EXPECTED_DATASET="tuab"
export RUN_GRID_ANALYSIS=1

exec bash "$SCRIPT_DIR/run_wiener_threshold_phase_experiment.sh" "$@"
