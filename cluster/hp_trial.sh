#!/bin/bash
# SLURM job script for a single HP tuning trial.
# Submitted by submit_hp_sweep.sh — do not run directly.
#
# Environment variables injected by submit_hp_sweep.sh:
#   STUDY_DB   — absolute path to the shared SQLite Optuna DB
#   TRIAL_DIR  — per-trial log dir (uses SLURM_JOB_ID for uniqueness)
#
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH -c 8
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --output=/home/s2145588/thesis/sam2loraboracluster/logs/hp-%j.out
#SBATCH --error=/home/s2145588/thesis/sam2loraboracluster/logs/hp-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=c.e.s.omtzigt@student.utwente.nl

module load anaconda3/2024.02
module load nvidia/cuda-11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam2lora

mkdir -p /home/s2145588/thesis/sam2loraboracluster/logs

STUDY_DB="${STUDY_DB:-/home/s2145588/thesis/sam2loraboracluster/hp_study.db}"
TRIAL_DIR="${TRIAL_DIR:-/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs/hp_trial_${SLURM_JOB_ID}}"

cd /home/s2145588/thesis/sam2loraboracluster/sam2

/home/s2145588/.conda/envs/sam2lora/bin/python hp_sweep.py \
  --study-db "${STUDY_DB}" \
  --study-name "hp_sweep" \
  --log-dir "${TRIAL_DIR}"
