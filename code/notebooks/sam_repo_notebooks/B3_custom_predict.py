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
import re
import glob

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import B_functions as bf

# %%
# === Model Configuration ===
# Base directory where all custom model training logs are stored.
CUSTOM_MODELS_BASE_DIR = (
    "/home/ced/Documents/master_thesis/workingdir/stage_3_models/"
    "models/SAM2LoRaBoRaDD/sam2/sam2_logs"
)

# MODELS dict:
#   Base SAM2.1 entries → require "config" and "checkpoint" keys.
#   Custom LoRA entries → require only "checkpoint_number"
#       (int → load that epoch's checkpoint; None → auto-select highest-numbered checkpoint).
MODELS = {
    "T": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_tiny.pt",
    },
    "S": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_small.pt",
    },
    "B": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_base_plus.pt",
    },
    "L": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "../checkpoints/sam2.1_hiera_large.pt",
    },
    # --- Custom fine-tuned LoRA models ---
    # Add new entries here as more models are trained.
    "l_r8_v1":   {"checkpoint_number": 20},
    "b+_r16_v1": {"checkpoint_number": None},
    "b+_r16_v2": {"checkpoint_number": 16},
    "b+_r16_v3": {"checkpoint_number": 25},
    "favela_lora_improved": {"checkpoint_number": 35},
}

ACTIVE_MODEL = "l_r8_v1"  # ← change this to switch models

# %%
# === Dataset Configuration ===
# Results are saved under: PREDICTIONS_BASE / <model_name> / <area_name> /
PREDICTIONS_BASE = "../../../output_data/predictions"

DATASETS = {
    "ceupaz": {
        "input": "../../../output_data/tiles/ceu_paz",
        "area": "ceu_paz",
    },
    "cansam": {
        "input": "../../../output_data/tiles/cantidio_sampaio",
        "area": "cantidio_sampaio",
    },
    "sanmad": {
        "input": "../../../output_data/tiles/santa_madalena",
        "area": "santa_madalena",
    },
}

ACTIVE_DATASET = "sanmad"  # ← change this to switch datasets

# %%
# === Resolve Model Paths ===
def _resolve_model(model_name):
    """Return (config_path_or_None, checkpoint_path, is_lora)."""
    entry = MODELS[model_name]
    if "config" in entry:
        return entry["config"], entry["checkpoint"], False

    ckpt_dir = os.path.join(CUSTOM_MODELS_BASE_DIR, model_name, "checkpoints")
    n = entry.get("checkpoint_number")

    if n is not None:
        ckpt = os.path.join(ckpt_dir, f"checkpoint_{n}.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return None, ckpt, True

    # Auto-select: highest-numbered checkpoint_N.pt
    pattern = re.compile(r"checkpoint_(\d+)\.pt$")
    numbered = [
        (int(pattern.search(f).group(1)), f)
        for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
        if pattern.search(f)
    ]
    if numbered:
        ckpt = max(numbered, key=lambda x: x[0])[1]
    else:
        ckpt = os.path.join(ckpt_dir, "checkpoint.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return None, ckpt, True


model_cfg, sam2_checkpoint, IS_LORA = _resolve_model(ACTIVE_MODEL)
INPUT_FOLDER  = DATASETS[ACTIVE_DATASET]["input"]
OUTPUT_FOLDER = os.path.join(PREDICTIONS_BASE, ACTIVE_MODEL, DATASETS[ACTIVE_DATASET]["area"])

print(f"Model:      {ACTIVE_MODEL}  (LoRA={IS_LORA})")
print(f"Checkpoint: {sam2_checkpoint}")
if model_cfg:
    print(f"Config:     {model_cfg}")
print(f"Input:      {INPUT_FOLDER}")
print(f"Output:     {OUTPUT_FOLDER}")

# %%
bf.image_available_check(INPUT_FOLDER)

# %%
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# %%
# this needs to run otherwise multithreading will fail
# due to JIT compiler issues with jupytext format
if __name__ == '__main__':
    print("Warming up JIT compiler...")
    if IS_LORA:
        dummy_model = bf.build_sam2_lora(sam2_checkpoint, device="cpu")
    else:
        dummy_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    _ = SAM2AutomaticMaskGenerator(model=dummy_model)
    del dummy_model
    print("Warm-up complete.")

    bf.two_gpu_predictor(INPUT_FOLDER, OUTPUT_FOLDER, model_cfg, sam2_checkpoint, is_lora=IS_LORA)


# %% --- Visual Inspection of Random Samples ---
# Guard prevents this from running in spawned subprocesses (which re-execute this file).
# In Jupyter, __name__ is '__main__', so the cell behaves normally when run interactively.
if __name__ == '__main__':
    bf.visualize_mask_samples(INPUT_FOLDER, OUTPUT_FOLDER, 
                              num_samples=4, 
                              alpha=0.5, 
                              draw_borders=True, 
                              min_masks=50)

# %%
