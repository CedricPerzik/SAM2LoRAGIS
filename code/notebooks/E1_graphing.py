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
import E_functions as ef

# %%
# Setup your directories
CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR

# area_name = "ceu_paz"
# area_name = "cantidio_sampaio"
area_name = "santa_madalena"

# Define where your files live
GT_DIR = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", area_name)
PRED_DIR = os.path.join(CODE_DIR, "output_data", "masks", f"{area_name}_masks")
EVAL_DIR = os.path.join(CODE_DIR, "output_data", "evaluation_npz", area_name)
csv_path = os.path.join(CODE_DIR, "output_data", "iou_results", f"{area_name}_unified_metrics.csv")

# Choose between 'overlap_pixels', 'area_px', or 'area_meters_sq' for the area column in visualizations
ef.generate_thesis_visualizations(csv_path, clip_outliers=True, area_col='overlap_pixels', quantile=0.99)
# %%
