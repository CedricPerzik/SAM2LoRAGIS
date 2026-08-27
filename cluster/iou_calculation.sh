#!/bin/bash
#SBATCH -J sam2iou
#SBATCH --partition=main-gpu
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH --time=05:00:00
#SBATCH --output=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.out
#SBATCH --error=/home/s2145588/thesis/sam2loraboracluster/logs/slurm-%j-%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=c.e.s.omtzigt@student.utwente.nl

# CPU-only job: matches GT masks to predicted masks and exports a unified
# metrics CSV per area. No GPU, no torch, no SAM2 model -- just numpy/pandas/
# opencv over the .npz files B3_custom_predict_cluster.py already wrote.

# Requires on cluster before first run:
#   mkdir -p /home/s2145588/thesis/sam2loraboracluster/sam2/custom_notebooks
# Copy to that directory:
#   D2_dataset_iou_cluster.py
#   D_functions.py

module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam2lora

# RUN_ID and DATASET_SIZE are set by submit_iou_sweep.sh via --export.

mkdir -p /home/s2145588/thesis/sam2loraboracluster/logs
mkdir -p /home/s2145588/thesis/data/evaluation_npz
mkdir -p /home/s2145588/thesis/data/iou_results

CUSTOM_NB=/home/s2145588/thesis/sam2loraboracluster/sam2/custom_notebooks
cd "${CUSTOM_NB}"

/home/s2145588/.conda/envs/sam2lora/bin/python D2_dataset_iou_cluster.py
