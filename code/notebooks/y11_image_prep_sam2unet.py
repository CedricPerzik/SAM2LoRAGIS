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
# %%
# -------------------------
# SAM 2 UNet image prep
# -------------------------
# This section is for preparing the image tiles for input into the SAM 2 UNet model

# %%
#### Tiling tifs ####
# 1. CHANGE INPUT: Pass tile_size=352 to override the 1024 default, new output path too.
# This determines the settings for configs, used in the next steps
# input_root is the entire area tif.
configs = af.setup_tiling_environment(my_datasets, output_root="../output_data/models/sam2unet/tiles/", tile_size=352)

# %%
#### Execute Tiling ####
af.generate_georeferenced_tiles(configs)

# %%
#### Quick Verify Tiling Output ####
af.verify_random_tiles(configs, num_samples=2)

# %%
#### Format for SAM2-UNet ####
# 2. Convert generated .tif tiles into standard .png images
af.sam2unet_format_tiles_to_png(configs)

# %%
# 3. Process the ground truth masks
# 3.1 Define Base Directories (Set these once)
BASE_TILES_DIR = "../output_data/models/sam2unet/tiles/"  # Assuming all areas are in the same base directory structure
BASE_MASKS_DIR = "../output_data/models/sam2unet/masks_5cm/"
BASE_GPKG_DIR = "/home/ced/Documents/master_thesis/workingdir/qgis_exports/"

# 3.2 Define Area Configurations
# Dictionary mapping the area_name to its specific GeoPackage file.
# Note: If all areas are actually inside a single master "rooftops_v2.gpkg", 
# you can just set all three values to "rooftops_v2.gpkg".
area_configs = {
    "ceu_paz": "rooftops_v1_ceupaz_sam2unet_5cm_dorphans.gpkg",
    "santa_madalena": "rooftops_v1_santamadalena_sam2unet_5cm_dorphans.gpkg",
    "cantidio_sampaio": "rooftops_v1_cantidiosampaio_sam2unet_5cm_dorphans.gpkg"
}
# %%
# 3.3 Execution Loop
for area_name, gpkg_filename in area_configs.items():
    print(f"\n{'='*50}")
    print(f"🌟 STARTING MASK GENERATION: {area_name.upper()}")
    print(f"{'='*50}")
    
    # Dynamically build the exact paths for the current area
    tiles_dir = os.path.join(BASE_TILES_DIR, area_name, "tiles_tif")
    gpkg_path = os.path.join(BASE_GPKG_DIR, gpkg_filename)
    output_mask_dir = os.path.join(BASE_MASKS_DIR, area_name)
    
    # Run the parallel processor
    af.sam2unet_generate_masks_parallel(tiles_dir, gpkg_path, output_mask_dir)

print("\n🎉 All geographic areas have been successfully processed!")