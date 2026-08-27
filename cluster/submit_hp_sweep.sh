#!/bin/bash
# Rationale:
# Parallel jobs break Bayesian optimization for hyperparameter tuning, need sequential jobs to find optimums.
# Submit N parallel HP tuning trials to SLURM.
# Each trial is one training run with Optuna-suggested hyperparameters.
# All trials share a single SQLite study DB on the NFS share.
#
# Usage (from cluster home dir, eduVPN):
#   bash /home/s2145588/thesis/submit_hp_sweep.sh [N_TRIALS]
#
# Default: 30 trials (minimum for BO to outperform random search).
# Each trial takes ~2 h on Lovelace (162 png tiles, early_stop_patience=5).

N_TRIALS="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_DB="/home/s2145588/thesis/sam2loraboracluster/hp_study.db"
LOG_BASE="/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs"

mkdir -p /home/s2145588/thesis/sam2loraboracluster/logs

echo "Submitting ${N_TRIALS} HP trials (study DB: ${STUDY_DB})"

for i in $(seq 1 "${N_TRIALS}"); do
    JOB_ID=$(sbatch \
        --job-name="hp_trial_${i}" \
        --export="STUDY_DB=${STUDY_DB}" \
        "${SCRIPT_DIR}/hp_trial.sh" | awk '{print $NF}')
    # TRIAL_DIR uses SLURM_JOB_ID (set inside the job), not the loop index.
    echo "  Submitted trial ${i}/${N_TRIALS} → SLURM job ${JOB_ID}"
done

echo ""
echo "Monitor with: squeue -u s2145588"
echo "Results DB:   ${STUDY_DB}"
echo ""
echo "To inspect results after jobs complete:"
echo "  python -c \""
echo "  import optuna"
echo "  s = optuna.load_study(study_name='hp_sweep', storage='sqlite:///${STUDY_DB}')"
echo "  print(s.best_params)"
echo "  \""
