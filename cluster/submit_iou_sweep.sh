#!/bin/bash
# Submit one SLURM IoU-calculation job per dataset size for each requested run.
# Reads predictions from /home/s2145588/thesis/data/predictions/<RUN_ID>/n<N>/<area>/
# (written by predict_cluster.sh / B3_custom_predict_cluster.py) and ground truth
# from /home/s2145588/thesis/data/ground_truth_npz/<area>/, and writes a unified
# metrics CSV per area to
# /home/s2145588/thesis/data/iou_results/<RUN_ID>/n<N>/n<N>_<area>_unified_metrics.csv.
# <area> is whatever subdirectories exist under the run's predictions dir --
# santa_madalena (OOD, always) plus any ID region (ceu_paz, cantidio_sampaio)
# restricted to that run's held-out test-split tiles. CPU-only, no GPU needed.
#
# Usage (from cluster, after copying iou_calculation.sh to the same dir):
#   bash submit_iou_sweep.sh --run-ids run_001
#   bash submit_iou_sweep.sh --run-ids run_001 run_002 run_003
#
# Requires: iou_calculation.sh in the same directory as this script. Typically
# run after submit_inference_sweep.sh's jobs have finished for the same runs.

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

# Must match the sizes used in submit_scarcity_sweep.sh / submit_inference_sweep.sh
SIZES=(5 10 15 25 40 65 108)

for RUN_ID in "${RUN_IDS[@]}"; do
    echo "Submitting IoU-calculation sweep for ${RUN_ID}"
    mkdir -p "${LOG_DIR}/${RUN_ID}"

    for N in "${SIZES[@]}"; do
        JOB_ID=$(sbatch \
            --export=DATASET_SIZE=${N},RUN_ID=${RUN_ID} \
            --job-name=sam2iou_${RUN_ID}_n${N} \
            --output="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}_iou.out" \
            --error="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}_iou.err" \
            "${SCRIPT_DIR}/iou_calculation.sh" | awk '{print $NF}')
        echo "  Submitted IoU-calculation job ${JOB_ID} for n${N}"
    done
done
