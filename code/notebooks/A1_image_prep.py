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
# In vscode Press CTRL + Shift + P or (⌘ + Shift + P on macOS) 
# to open the command palette. Then type 
# Python select interpreter 
# in the search field and choose the right version (daksam).

# %%
import os
import json
import rasterio
import numpy as np
from rasterio.windows import Window

import A_functions as af

# %%
#### REPROJECTION CONFIG ####
INPUT_FOLDER      = "../input_data/original_tifs/"
NORMALIZED_FOLDER = "../input_data/normalized_tifs/" # Where reprojected TIFs will go
TARGET_CRS        = 'EPSG:31983'

# filenames without extension or suffix
my_datasets = [
    "ceu_paz", 
    "santa_madalena", 
    "cantidio_sampaio"
    ]

# %%
#### Standardizing CRS ####
# Sanity check original settings
af.check_tif_settings(INPUT_FOLDER + my_datasets[2] + ".tif")

# %%
# Reproject all TIFs in list
af.reproject_tif_list(my_datasets, INPUT_FOLDER, NORMALIZED_FOLDER, TARGET_CRS)

# %%
# Sanity check reprojected files
af.check_tif_settings(NORMALIZED_FOLDER + my_datasets[2] + "_normalized.tif")

# %%
#### Tiling tifs ####
# Set up folder structure and metadata file
configs = af.setup_tiling_environment(my_datasets)

# %%
#### Execute Tiling ####
af.generate_georeferenced_tiles(configs)

# %%
#### Quick Verify Tiling Output ####
af.verify_random_tiles(configs, num_samples=2)

# %%
