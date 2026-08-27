#!/bin/bash
# Submit one SLURM job per dataset size for the data scarcity study.
# Each call creates one or more numbered runs (run_001, run_002, ...).
# All jobs within one run share the same RUN_ID and SEED so results land under
# sam2_logs/<RUN_ID>/n<N>/ and logs/<RUN_ID>/slurm-<jobid>-n<N>.{out,err}.
#
# Usage (from cluster, after cd /home/s2145588/thesis/):
#   bash submit_scarcity_sweep.sh                        # 1 run, random seed
#   bash submit_scarcity_sweep.sh --runs 5               # 5 runs, random seeds
#   bash submit_scarcity_sweep.sh --seeds 42 137 999     # 3 runs, explicit seeds
#
# --seeds overrides --runs; the number of runs equals the number of seeds given.
# Requires: train_cluster.sh in the same directory as this script.
#
# Split: 50% train / 25% val / 25% test (tile-level, seeded by SEED below), per
# scratch.train_fraction / scratch.test_fraction in sam2.1_hiera_favela_lora.yaml.
# Each train_cluster.sh job writes the held-out test tile names to
# ${TEST_SPLIT_DIR}/<RUN_ID>/test_tiles.json as a side effect of training
# (see training/dataset/split_utils.py:dump_test_manifest, called from
# train.py's main() via --test-split-dir) -- nothing to do here but wait for
# the jobs to start.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_LOGS_DIR="/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs"
LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/logs"
TEST_SPLIT_DIR="/home/s2145588/thesis/data/test_split"

# --- Parse flags ---
RUNS=1
SEEDS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)
            RUNS="$2"; shift 2 ;;
        --seeds)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SEEDS+=("$1"); shift
            done ;;
        *)
            echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# If explicit seeds were given they define how many runs to submit.
if [[ ${#SEEDS[@]} -gt 0 ]]; then
    RUNS=${#SEEDS[@]}
else
    # Generate the requested number of random seeds (0-99999).
    for (( i=0; i<RUNS; i++ )); do
        SEEDS+=( $(shuf -i 0-99999 -n 1) )
    done
fi

# Dataset sizes = number of training tiles.
# Annotated ID tiles: 67 (ceu_paz) + 149 (cantidio_sampaio) = 216 total.
# After 50/25/25 split (seed varies per run): ~108 train / ~54 val / ~54 test
# (exact counts vary slightly per seed once artifact-only tiles are dropped;
# see the dumped test_tiles.json for the true count of a given run).
# 108 is the maximum.
SIZES=(5 10 15 25 40 65 108)

# --- Determine starting run number ---
NEXT_ID=1
if [ -d "${SAM2_LOGS_DIR}" ]; then
    LAST=$(ls -d "${SAM2_LOGS_DIR}"/run_[0-9][0-9][0-9] 2>/dev/null \
           | sed 's/.*run_//' | sort -n | tail -1)
    if [ -n "${LAST}" ]; then
        NEXT_ID=$(( 10#${LAST} + 1 ))
    fi
fi

# --- Submit one sweep per run ---
for (( r=0; r<RUNS; r++ )); do
    RUN_ID=$(printf "run_%03d" $(( NEXT_ID + r )))
    SEED="${SEEDS[$r]}"

    echo "Submitting sweep ${RUN_ID} (seed=${SEED})"
    mkdir -p "${LOG_DIR}/${RUN_ID}"
    echo "${RUN_ID} ${SEED}" >> "${LOG_DIR}/seeds.txt"

    for N in "${SIZES[@]}"; do
        JOB_ID=$(sbatch \
            --export=DATASET_SIZE=${N},RUN_ID=${RUN_ID},SEED=${SEED} \
            --job-name=sam2lora_${RUN_ID}_n${N} \
            --output="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}.out" \
            --error="${LOG_DIR}/${RUN_ID}/slurm-%j-n${N}.err" \
            "${SCRIPT_DIR}/train_cluster.sh" | awk '{print $NF}')
        echo "  Submitted job ${JOB_ID} for N=${N} tiles"
    done

    echo "  Results -> ${SAM2_LOGS_DIR}/${RUN_ID}/"
    echo "  Test-split tile manifest -> ${TEST_SPLIT_DIR}/${RUN_ID}/test_tiles.json (written once the first job starts)"
done
