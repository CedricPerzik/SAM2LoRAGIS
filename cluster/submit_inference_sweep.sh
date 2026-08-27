#!/bin/bash
# Submit one SLURM inference job per dataset size for each requested run.
# Reads checkpoints from sam2_logs/<RUN_ID>/n<N>/checkpoints/ and saves
# predictions to /home/s2145588/thesis/data/predictions/<RUN_ID>/n<N>/<area>/,
# where <area> is santa_madalena (OOD, always) plus one subdir per ID region
# (ceu_paz, cantidio_sampaio) restricted to that run's held-out test-split
# tiles -- see /home/s2145588/thesis/data/test_split/<RUN_ID>/test_tiles.json
# and B3_custom_predict_cluster.py. Falls back to santa_madalena-only if that
# manifest is missing (e.g. an older run, or test_fraction=0 for that run).
#
# Usage (from cluster, after copying predict_cluster.sh to the same dir):
#   bash submit_inference_sweep.sh --run-ids run_001
#   bash submit_inference_sweep.sh --run-ids run_001 run_002 run_003
#
# Requires: predict_cluster.sh in the same directory as this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/logs"

# --- Parse flags ---
RUN_IDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-ids)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RUN_IDS+=("$1"); shift
            done ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#RUN_IDS[@]} -eq 0 ]]; then
    echo "Error: --run-ids is required. Example: --run-ids run_001 run_002" >&2
    exit 1
fi

# Must match the sizes used in submit_scarcity_sweep.sh
SIZES=(5 10 15 25 40 65 108)

for RUN_ID in "${RUN_IDS[@]}"; do
    echo "Submitting inference sweep for ${RUN_ID}"
    mkdir -p "${LOG_DIR}/${RUN_ID}"

    for N in "${SIZES[@]}"; do
        JOB_ID=$(sbatch \
            --export=DATASET_SIZE=${N},RUN_ID=${RUN_ID} \
            --job-name=sam2infer_${RUN_ID}_n${N} \
            --output="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}_infer.out" \
            --error="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}_infer.err" \
            "${SCRIPT_DIR}/predict_cluster.sh" | awk '{print $NF}')
        echo "  Submitted inference job ${JOB_ID} for n${N}"
    done
done
