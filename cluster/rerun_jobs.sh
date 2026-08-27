#!/bin/bash
# Re-submit failed scarcity sweep jobs defined in logs/seeds.txt.
#
# Usage (from cluster, after cd /home/s2145588/thesis/):
#   bash rerun_jobs.sh
#
# Edit logs/seeds.txt to append the sizes to re-run on a line:
#   run_001 42 5 10
#   run_002 137
#   run_003 999 25 40 65
# Lines with no sizes are skipped. run_002 above would be skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/logs"
SEEDS_FILE="${LOG_DIR}/seeds.txt"

if [ ! -f "${SEEDS_FILE}" ]; then
    echo "Error: ${SEEDS_FILE} not found." >&2
    exit 1
fi

while read -r RUN_ID SEED RERUN_SIZES_STR; do
    # Skip blank lines and comment lines.
    [[ -z "${RUN_ID}" || "${RUN_ID}" == \#* ]] && continue

    # Build array of sizes from remaining fields on the line.
    read -ra SIZES <<< "${RERUN_SIZES_STR}"

    if [ ${#SIZES[@]} -eq 0 ]; then
        echo "Skipping ${RUN_ID} (no sizes listed)"
        continue
    fi

    echo "Re-submitting ${RUN_ID} (seed=${SEED}) for sizes: ${SIZES[*]}"
    mkdir -p "${LOG_DIR}/${RUN_ID}"

    for N in "${SIZES[@]}"; do
        JOB_ID=$(sbatch \
            --export=DATASET_SIZE=${N},RUN_ID=${RUN_ID},SEED=${SEED} \
            --job-name=sam2lora_${RUN_ID}_n${N} \
            --output="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}.out" \
            --error="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}.err" \
            "${SCRIPT_DIR}/train_cluster.sh" | awk '{print $NF}')
        echo "  Submitted job ${JOB_ID} for N=${N} tiles"
    done
done < "${SEEDS_FILE}"
