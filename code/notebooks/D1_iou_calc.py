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

import D_functions as df
# %%
# Setup your directories
CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR

# area_name = "ceu_paz"
# area_name = "cantidio_sampaio"
area_name = "santa_madalena"

# Set to None for the base model, or a model name string for a fine-tuned model (e.g. "l_r8_v1")
model_name = None
# model_name = "l_r8_v1"
# model_name = "l_r8_v2"
# model_name = "b+_r16_trial"
# model_name = "b+_r16_v2"
# model_name = "b+_r16_v3"


# Label used for output directories; set automatically from model_name for LoRa,
# or manually to "base_tiny", "base_small", "base_plus", "base_large" for base models
model_label = model_name if model_name else "base_large"
# model_label = "base_tiny"
# model_label = "base_small"
# model_label = "base_plus"

# Define where your files live
GT_DIR = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", area_name)
if model_name:
    PRED_DIR = os.path.join(CODE_DIR, "output_data", "predictions", model_name, area_name)
    EVAL_DIR = os.path.join(CODE_DIR, "output_data", "evaluation_npz", model_name, area_name)
else:
    PRED_DIR = os.path.join(CODE_DIR, "output_data", "masks", f"{area_name}_masks")
    EVAL_DIR = os.path.join(CODE_DIR, "output_data", "evaluation_npz", area_name)

# %%
df.run_evaluation_pipeline_concurrent(GT_DIR, PRED_DIR, EVAL_DIR)

# %%
# Run the inspection (Assuming EVAL_DIR is still defined from your previous cell)
image_stats, global_ious, strict_ious = df.inspect_evaluation_metrics(EVAL_DIR, area_name=area_name)

# %%
# save evaluation data to csv
csv_prefix = f"{model_name}_{area_name}" if model_name else area_name
csv_path = os.path.join(CODE_DIR, "output_data", "iou_results", f"{csv_prefix}_unified_metrics.csv")
df_results = df.export_unified_metrics_to_csv_concurrent(EVAL_DIR, GT_DIR, csv_path)

# %%
img_directory = os.path.join(CODE_DIR, "output_data", "tiles", area_name)
csv_file_path = os.path.join(CODE_DIR, "output_data", "iou_results", f"{csv_prefix}_unified_metrics.csv")

# df.analyze_and_visualize_iou_range(
#     csv_path=csv_file_path,
#     gt_dir=GT_DIR,
#     pred_dir=PRED_DIR,
#     img_dir=img_directory,
#     area_name=area_name,
#     model_label=model_label,
#     min_iou=0.40,        # Minimum IoU to include
#     max_iou=0.60         # Maximum IoU to include
# )

# df.analyze_and_visualize_iou_range_tqdm(
#     csv_path=csv_file_path,
#     gt_dir=GT_DIR,
#     pred_dir=PRED_DIR,
#     img_dir=img_directory,
#     area_name=area_name,
#     model_label=model_label,
#     min_iou=0.9,        # Minimum IoU to include
#     max_iou=1.0         # Maximum IoU to include
# )

df.visualize_iou_range_concurrent(
    csv_path=csv_file_path,
    gt_dir=GT_DIR,
    pred_dir=PRED_DIR,
    img_dir=img_directory,
    area_name=area_name,
    model_label=model_label,
    min_iou=0.0,        # Minimum IoU to include
    max_iou=1.0         # Maximum IoU to include
)

# Analyze and visualize the worst-performing roofs (lowest IoU)
# df.analyze_and_visualize_worst_roofs(csv_file_path, GT_DIR, PRED_DIR, img_directory, model_label=model_label, top_n=10)
# %%
