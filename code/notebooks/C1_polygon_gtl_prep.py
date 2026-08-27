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
import fiona
import random
import rasterio
import numpy as np
from tqdm.auto import tqdm
import geopandas as gpd
from matplotlib import image
from rasterio import features
import matplotlib.pyplot as plt
from shapely.geometry import box

import sam_repo_notebooks.B_functions as bf
import C_functions as cf

# %%
label_gpkg_path = "/home/ced/Documents/master_thesis/workingdir/qgis_exports/rooftops_v1_cantidiosampaio.gpkg"
tif_path = "/home/ced/Documents/master_thesis/workingdir/code/output_data/tiles/cantidio_sampaio/cantidio_sampaio_row_6_col_7.tif"

# %%
import fiona
# This looks at the file schema directly
with fiona.open(label_gpkg_path) as layer:
    print("--- SCHEMA PROPERTIES ---")
    print(layer.schema['properties'])

# %%
gt_data = cf.format_ground_truth(label_gpkg_path, tif_path)
# %%
# Visualize Ground Truth to see if they align with the rooftops in the image
plt.figure(figsize=(10, 10))
plt.imshow(image.imread(tif_path)) # Show the original image
bf.show_anns(gt_data) # Using your custom function
plt.title("Visual Check: Ground Truth Masks on Image")
plt.axis('off')
plt.show()

# %%
# Audit the GT data to check for any anomalies or inconsistencies
cf.audit_gt_data(gt_data)

# %%
idx = random.randint(0, len(gt_data) - 1)
sample = gt_data[idx]

print(f"--- INSPECTING GROUND TRUTH INDEX: {idx} ---")
print(f"Segmentation:       {sample['segmentation'][0]}") # first row
print(f"Area (Pixels):      {sample['area']} pixels")
print(f"bbox (Global):      {sample['bbox']} (x, y, w, h)")
print(f"Area (GIS):         {sample['area_meters_sq']} m²")
print(f"Perimeter (GIS):    {sample['perimeter_meters']} m")
print(f"Feature ID:         {sample['feature_id']}")
print(f"Is Artifact:        {sample['artifact_intersect']}")
print(f"Belongs To:         {sample['belongs_to']}")
print(f"Is Cut:             {sample['is_cut']}")

# %%
print(f"{len(gt_data)}\n")
print(f"{gt_data[0].keys()}\n")
print(gt_data[np.random.randint(0, len(gt_data))])

# %%
with fiona.open(label_gpkg_path) as f:
    feats = list(f)
    print(feats)

# %%
# --- CONFIGURATION Linux---
area_name = "ceu_paz"
gpkg_name = "rooftops_v1_ceupaz.gpkg"
# area_name = "cantidio_sampaio"
# gpkg_name = "rooftops_v1_cantidiosampaio.gpkg"
# area_name = "santa_madalena"
# gpkg_name = "rooftops_v1_santamadalena.gpkg"

TILES_DIR = f"/home/ced/Documents/master_thesis/workingdir/code/output_data/tiles/{area_name}/"
GPKG_PATH = f"/home/ced/Documents/master_thesis/workingdir/qgis_exports/{gpkg_name}"
BASE_OUTPUT_DIR = "/home/ced/Documents/master_thesis/workingdir/code/output_data/ground_truth_npz/"
# Grabs 'area name' from the tiles path string
folder_name = os.path.basename(TILES_DIR.strip("/")) 
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, folder_name)

# %%
# -------------WINDOWS JANK START----------------

# --- CONFIGURATION Windows ---
# area_name = "ceu_paz"
# gpkg_name = "rooftops_v1_ceupaz.gpkg"
# area_name = "cantidio_sampaio"
# gpkg_name = "rooftops_v1_cantidiosampaio.gpkg"
area_name = "santa_madalena"
gpkg_name = "rooftops_v1_santamadalena.gpkg"

CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR)
TILES_DIR = os.path.join(CODE_DIR, "output_data", "tiles", area_name)
GPKG_PATH_WIN = rf"C:\Users\Cedric\Desktop\utwente\interaction_technology\master_thesis\workingdir\qgis_exports\{gpkg_name}"

# 5. Build OUTPUT_DIR
BASE_OUTPUT_DIR = os.path.join(CODE_DIR, "output_data", "ground_truth_npz")
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, area_name)

# --- DEBUG CHECK ---
print(f"📍 Script is running from: {CURRENT_DIR}")
print(f"📁 Base code dir resolved to: {CODE_DIR}")
print(f"🔍 Looking for tiles in: {TILES_DIR}")
tif_count = len(glob.glob(os.path.join(TILES_DIR, '*.tif')))
print(f"📄 File count check: {tif_count}")

# %%
# --- main WINDOWS function ---
if tif_count > 0:
    print("🚀 Paths are perfectly aligned! Starting processing...")
    cf.process_all_gt_tiles(TILES_DIR, GPKG_PATH_WIN, OUTPUT_DIR)
else:
    print("❌ Still 0 files! Something else is wrong.")

# -------------WINDOWS JANK END----------------

# %%
import rasterio

# Grab just the first TIF to test
test_tif = glob.glob(os.path.join(TILES_DIR, '*.tif'))[0]

# Load GPKG to check its metadata
# gdf_full = gpd.read_file(GPKG_PATH_WIN)  # Use the Windows path make condition later
gdf_full = gpd.read_file(GPKG_PATH)

print("--- SPATIAL DIAGNOSTIC ---")
print(f"🌍 GPKG CRS: {gdf_full.crs}")
print(f"📦 GPKG Total Bounds: {gdf_full.total_bounds}")

with rasterio.open(test_tif) as src:
    print(f"🖼️ TIFF CRS: {src.crs}")
    print(f"📏 TIFF Bounds: {src.bounds}")

# %%
# --- main function ---
cf.process_all_gt_tiles(TILES_DIR, GPKG_PATH, OUTPUT_DIR)

# %%
cf.print_npz_premium(BASE_OUTPUT_DIR, area_name)

# %%
