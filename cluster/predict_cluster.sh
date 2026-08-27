#!/bin/bash
#SBATCH -J sam2infer
#SBATCH --partition=main-gpu
#SBATCH --gres=gpu:lovelace:1
#SBATCH -c 4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.out
#SBATCH --error=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=c.e.s.omtzigt@student.utwente.nl

# Requires on cluster before first run:
#   mkdir -p /home/s2145588/thesis/sam2loraboracluster/sam2/custom_notebooks
# Copy to that directory:
#   B3_custom_predict_cluster.py
#   B_functions.py

module load anaconda3/2024.02
module load nvidia/cuda-11.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam2lora

# RUN_ID and DATASET_SIZE are set by submit_inference_sweep.sh via --export.

mkdir -p /home/s2145588/thesis/sam2loraboracluster/logs
mkdir -p /home/s2145588/thesis/data/predictions

CUSTOM_NB=/home/s2145588/thesis/sam2loraboracluster/sam2/custom_notebooks
cd "${CUSTOM_NB}"

/home/s2145588/.conda/envs/sam2lora/bin/python B3_custom_predict_cluster.py
