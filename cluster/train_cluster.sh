#!/bin/bash
#SBATCH -J sam2lora
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH -c 4
#SBATCH --mem=64gb
#SBATCH --time=50:00:00
#SBATCH --output=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.out
#SBATCH --error=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=c.e.s.omtzigt@student.utwente.nl

module load anaconda3/2024.02
module load nvidia/cuda-11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam2lora

# RUN_ID, DATASET_SIZE, and SEED are all set by submit_scarcity_sweep.sh via --export.
EXPERIMENT_LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs/${RUN_ID}/n${DATASET_SIZE}"

mkdir -p /home/s2145588/thesis/sam2loraboracluster/logs
mkdir -p "${EXPERIMENT_LOG_DIR}"

cd /home/s2145588/thesis/sam2loraboracluster/sam2

/home/s2145588/.conda/envs/sam2lora/bin/python training/train.py \
  -c configs/sam2.1_training/sam2.1_hiera_favela_lora \
  --num-gpus 1 \
  --use-cluster 0 \
  --model-size l \
  --tiles-root /home/s2145588/thesis/data/tiles \
  --masks-root /home/s2145588/thesis/data/ground_truth_npz \
  --png-cache-dir /home/s2145588/thesis/data/favela_png \
  --max-tiles ${DATASET_SIZE} \
  --experiment-log-dir "${EXPERIMENT_LOG_DIR}" \
  --split-seed ${SEED} \
  --trainer-seed ${SEED} \
  --test-split-dir "/home/s2145588/thesis/data/test_split/${RUN_ID}"
