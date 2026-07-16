#!/usr/bin/env bash
# Run the fixed eight-cell Wiener phase/coherence grid on MNE ERP-CORE Flankers.

set -euo pipefail

CONDA_RUN="${CONDA_RUN:-conda run -n eeg_pipeline}"
FIF_PATH=""

usage() {
    cat <<EOF
Usage: bash scripts/run_erp_core_wiener_phase_grid.sh [--fif PATH]

If --fif is omitted, script 10 locates/downloads the MNE ERP-CORE Flankers FIF.
Each cell writes results/erp_core_flankers/phase_grid/exp{1..8}/ern_fcz_difference.png.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fif) FIF_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LABELS=(
    "coh=0.15 phase=pi frequency"
    "coh=0.45 phase=pi frequency"
    "coh=0.75 phase=pi frequency"
    "coh=0.15 phase=pi/2 phasegated"
    "coh=0.15 phase=pi/5 phasegated"
    "coh=0.15 phase=pi/10 phasegated"
    "coh=0.45 phase=pi/10 phasegated"
    "coh=0.75 phase=pi/10 phasegated"
)

for idx in {1..8}; do
    config="configs/exp_erp_core_wiener_phase_${idx}.yaml"
    echo "[$idx/8] ${LABELS[$((idx - 1))]}"
    command=(
        $CONDA_RUN python scripts/10_benchmark_erp_core_flankers.py
        --config "$config" --force
    )
    if [[ -n "$FIF_PATH" ]]; then
        command+=(--fif "$FIF_PATH")
    fi
    "${command[@]}"
done

echo "Generated ERN figures:"
for idx in {1..8}; do
    echo "  results/erp_core_flankers/phase_grid/exp${idx}/ern_fcz_difference.png"
done
