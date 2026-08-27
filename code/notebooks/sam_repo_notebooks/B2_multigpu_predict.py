# ---
# jupyter:
#   jupytext:
#     formats: ipynb, py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: daksam
#     language: python
#     name: python3
# ---

# %%
import os
import glob
import json
import torch
import threading
import numpy as np
from PIL import Image
import concurrent.futures

from tqdm.auto import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import B_functions as bf

# %%
# directories are from the perspective of this notebook file
DATASETS = {
    "ceupaz": {
        "input": "../../../output_data/tiles/ceu_paz",
        "output": "../../../output_data/masks/ceu_paz_masks"
    },
    "cansam": {
        "input": "../../../output_data/tiles/cantidio_sampaio",
        "output": "../../../output_data/masks/cantidio_sampaio_masks"
    },
    "sanmad": {
        "input": "../../../output_data/tiles/santa_madalena",
        "output": "../../../output_data/masks/santa_madalena_masks"
    }
}

ACTIVE_KEY = "sanmad"  # Change this to switch datasets

INPUT_FOLDER = DATASETS[ACTIVE_KEY]["input"]
OUTPUT_FOLDER = DATASETS[ACTIVE_KEY]["output"]
# %%
# --- Model Configuration ---
MODELS = {
    "T": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_tiny.pt"
    },
    "S": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_small.pt"
    },
    "B": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_base_plus.pt"
    },
    "L": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_large.pt"
    }
}

ACTIVE_MODEL = "L"  # Change this to switch models
model_cfg = MODELS[ACTIVE_MODEL]["config"]
sam2_checkpoint = MODELS[ACTIVE_MODEL]["checkpoint"]
print(f"Using model {ACTIVE_MODEL} with config {model_cfg} and checkpoint {sam2_checkpoint}")

# %%
image_available_check = bf.image_available_check(INPUT_FOLDER)

# %%
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# %%
# this needs to run otherwise multithreading will fail 
# due to JIT compiler issues with jupytext format
if __name__ == '__main__':
    print("Warming up JIT compiler...")
    dummy_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    _ = SAM2AutomaticMaskGenerator(model=dummy_model)
    del dummy_model 
    print("Warm-up complete.")

    # Now it is safe to start multi-threading
    bf.two_gpu_predictor(INPUT_FOLDER, OUTPUT_FOLDER, model_cfg, sam2_checkpoint)

# %% --- Visual Inspection of Random Samples ---
bf.visualize_mask_samples(INPUT_FOLDER, OUTPUT_FOLDER, num_samples=4, alpha=0.5, draw_borders=True)

# %%
