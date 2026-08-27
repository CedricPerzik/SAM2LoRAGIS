# https://github.com/sayanmndl/SAM2LoRA#
# https://arxiv.org/abs/2510.10288

# dataset prep
# %%
import os
import cv2
import json
import glob
import json
import tifffile
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
# Define base paths
base_dir = "/home/ced/Documents/master_thesis/workingdir/code/output_data"
tiles_dir = os.path.join(base_dir, "tiles")
tiles_png_dir = os.path.join(base_dir, "tiles_png")

# Find all .tif files recursively
search_pattern = os.path.join(tiles_dir, "**", "*.tif")
tif_files = glob.glob(search_pattern, recursive=True)

print(f"Found {len(tif_files)} TIF files. Continue to start conversion...")

# %%
for tif_path in tqdm(tif_files, desc="Converting TIF to PNG"):
    # Determine the relative path to maintain folder structure (e.g., ceu_paz/ceu_paz_row_8_col_7.tif)
    rel_path = os.path.relpath(tif_path, tiles_dir)
    
    # Create the new destination path, changing extension to .png
    png_rel_path = str(Path(rel_path).with_suffix('.png'))
    png_path = os.path.join(tiles_png_dir, png_rel_path)
    
    # Ensure the target sub-directory exists
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    # Read TIF and save as PNG
    try:
        # tifffile is generally safer for scientific TIFs than cv2.imread
        img = tifffile.imread(tif_path)
        
        # Ensure image is 8-bit RGB for standard PNG saving
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] > 3:
            img = img[..., :3] # Drop alpha or multispectral bands if present
            
        # OpenCV expects BGR for saving, but if we use PIL/imageio it expects RGB.
        # Assuming tifffile reads it as RGB, we convert to BGR for cv2.imwrite
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(png_path, img_bgr)
    except Exception as e:
        print(f"Failed to convert {tif_path}: {e}")

print(f"Conversion complete. PNGs saved to {tiles_png_dir}")
# %%
#######################
# Split dataset into train and test
#######################
# Define base paths
base_dir = "/home/ced/Documents/master_thesis/workingdir/code/output_data"
mask_dir = os.path.join(base_dir, "ground_truth_npz")
png_dir = os.path.join(base_dir, "tiles_png")
sam2lora_datasets = os.path.join(base_dir, "datasets", "sam2lora")
splits_file = os.path.join(sam2lora_datasets, "dataset_splits.json")

print(f"Preparing dataset splits using masks from {mask_dir} and images from {png_dir}.")
print(f"Splits will be saved to {sam2lora_datasets}. If it already exists, it will not be overwritten.")
print("Continue for matching and splitting process...")

# %%
# We only want to generate splits if they don't exist, or if we force it.
if os.path.exists(splits_file):
    print(f"Splits file already exists at {splits_file}. Delete it if you want to regenerate.")
else:
    # Find all .npz masks recursively (Mask-first approach)
    mask_files = glob.glob(os.path.join(mask_dir, "**", "*_gt.npz"), recursive=True)
    
    print(f"Found {len(mask_files)} mask files. Matching to PNGs...")
    
    # Dictionary to hold matched pairs per area
    area_data = {}
    valid_pairs_count = 0
    
    for mask_path in mask_files:
        # Extract the area name (the folder name just above the file)
        area_name = os.path.basename(os.path.dirname(mask_path)) # e.g., 'ceu_paz'
        
        # Determine the relative path to maintain folder structure
        rel_path = os.path.relpath(mask_path, mask_dir) 
        
        # Reconstruct the expected PNG path by removing '_gt' and changing to .png
        expected_png_name = str(Path(rel_path).name).replace('_gt.npz', '.png')
        png_path = os.path.join(png_dir, area_name, expected_png_name)
        
        # Only add to our list if the matching PNG actually exists
        if os.path.exists(png_path):
            if area_name not in area_data:
                area_data[area_name] = []
                
            area_data[area_name].append({
                "image": png_path,
                "annotation": mask_path
            })
            valid_pairs_count += 1
        else:
            print(f"Warning: Mask found but PNG missing for: {png_path}")

    print(f"Successfully matched {valid_pairs_count} valid Image-Mask pairs across {len(area_data)} areas.")
    
    # Generate the train/test splits per area
    final_splits = {}
    for area, pairs in area_data.items():
        # If an area has too few samples, you might want to handle it differently, 
        # but train_test_split handles it normally.
        trainset, testset = train_test_split(pairs, test_size=0.25, random_state=42)
        
        final_splits[area] = {
            "train": trainset,
            "test": testset
        }
        print(f"Area '{area}': {len(trainset)} train, {len(testset)} test.")
        
    # Save to a JSON file
    with open(splits_file, 'w') as f:
        json.dump(final_splits, f, indent=4)
        
    print(f"Splits successfully saved to {splits_file}")
# %%
##########################
# Validating training data
##########################

# 1. Load the JSON file
base_eval_dir = "/home/ced/Documents/master_thesis/workingdir/stage_3_models/models/SAM2LoRA/checkpoints"
json_path = os.path.join(base_eval_dir, 'fundus_favela_sam2_l_r256_a512_e6.json')
# json_path = os.path.join(base_eval_dir, 'fundus_favela_sam2_l_r256_a512_e1.json')

with open(json_path, 'r') as f:
    eval_data = json.load(f)

# 2. Extract the image-level data
dataset_name = 'favela'
image_data = eval_data[dataset_name]['data']

# 3. Parse nested JSON into a flat Pandas DataFrame
records = []
for item in image_data:
    # We grab 'class_1' which contains all the math
    metrics = item['metrics']['class_1'].copy()
    metrics['image_idx'] = item['idx']
    records.append(metrics)

df = pd.DataFrame(records)

# 4. Calculate overall means
mean_metrics = df.drop(columns=['image_idx']).mean()
print("=== Final Aggregated Scores ===")
for metric, score in mean_metrics.items():
    print(f"{metric.upper()}: {score:.4f}")

# 5. Visualizations!
# Set seaborn style for clean, academic-looking charts
sns.set_theme(style="whitegrid")

# Separate metrics by scale (0-1 ratios vs Distance metrics)
ratio_metrics = mean_metrics.drop(['hd95', 'assd'])
dist_metrics = mean_metrics[['hd95', 'assd']]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Chart A: 0-1 Metrics (IoU, AUC, F1, etc.)
sns.barplot(x=ratio_metrics.index, y=ratio_metrics.values, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Average Classification Metrics', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Score (0 to 1)')
axes[0, 0].set_ylim(0, 1.1)
axes[0, 0].tick_params(axis='x', rotation=45)
for index, value in enumerate(ratio_metrics.values):
    axes[0, 0].text(index, value + 0.02, f'{value:.3f}', ha='center', fontsize=9)

# Chart B: Distance Metrics (Hausdorff 95 & ASSD)
sns.barplot(x=dist_metrics.index, y=dist_metrics.values, ax=axes[0, 1], palette='flare')
axes[0, 1].set_title('Average Distance Metrics (Lower is Better)', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Distance')
for index, value in enumerate(dist_metrics.values):
    axes[0, 1].text(index, value + (value * 0.02), f'{value:.1f}', ha='center', fontsize=10)

# Chart C: Distribution of IoU Scores Across All Images
sns.histplot(df['iou'], bins=10, kde=True, ax=axes[1, 0], color='royalblue')
axes[1, 0].set_title('Distribution of IoU Scores (Per Image)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('IoU Score')
axes[1, 0].set_ylabel('Number of Images')

# Chart D: Distribution of F1 (Dice) Scores Across All Images
sns.histplot(df['f1'], bins=10, kde=True, ax=axes[1, 1], color='seagreen')
axes[1, 1].set_title('Distribution of F1 / Dice Scores (Per Image)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('F1 Score')
axes[1, 1].set_ylabel('Number of Images')

plt.tight_layout()
plt.show()
# %%
############################
# Visualize some sample predictions
############################

import sys
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. System Path Injection ---
SAM2LORA_REPO_PATH = "/home/ced/Documents/master_thesis/workingdir/stage_3_models/models/SAM2LoRA"
SAM2_BACKEND_PATH = os.path.join(SAM2LORA_REPO_PATH, "segment-anything-2")

if SAM2LORA_REPO_PATH not in sys.path:
    sys.path.append(SAM2LORA_REPO_PATH)
if SAM2_BACKEND_PATH not in sys.path:
    sys.path.append(SAM2_BACKEND_PATH)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from model.sam2lorabase import SAM2LoRABase
from dataloader.retina_datasets import get_dataset 

# --- 2. Configuration & Absolute Paths ---
BASE_PATH = "/home/ced/Documents/master_thesis/workingdir/code/output_data"
CHECKPOINT_PATH = os.path.join(SAM2LORA_REPO_PATH, "checkpoints/fundus_favela_vanilla_sam2_l_r256_a512_best.ckpt")
BASE_CKPT = "/home/ced/Documents/master_thesis/workingdir/stage_3_models/models/SAM2.1/sam2.1_hiera_large.pt"

DATASET_NAME = 'favela'
# REGION = 'ceu_paz'
# REGION = 'cantidio_sampaio'
REGION = 'santa_madalena'

def show_mask(mask, ax, color=None, alpha=0.5):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, alpha])
    else:
        color = np.concatenate([color, [alpha]])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if len(pos_points) > 0:
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    if len(neg_points) > 0:
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='X', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    if box is not None and len(box) == 4 and sum(box) > 0:
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0,0,0,0), lw=2))

print("Loading model and checkpoint...")

original_cwd = os.getcwd()
os.chdir(SAM2LORA_REPO_PATH)

try:
    sam2_base = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", BASE_CKPT)

    for param in sam2_base.parameters():
        param.requires_grad = False

    # THE FIX: use_high_res_features_in_sam is now True
    sam_lora_model = SAM2LoRABase(sam2_base, rank=256, alpha=512, use_high_res_features_in_sam=True).to('cuda')
    predictor = SAM2ImagePredictor(sam_lora_model)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
    predictor.model.load_state_dict(checkpoint['model_state_dict'])
    predictor.model.eval()
    print("Model loaded successfully!")
finally:
    os.chdir(original_cwd)

print("Loading dataset...")
dataset = get_dataset(
    dataset_name=DATASET_NAME,
    mode='test', 
    region=REGION,
    num_pos_points=20,
    num_neg_points=0,
    num_boxes=0,
    transform=None 
)

idx = random.randint(0, len(dataset) - 1)
img, gt_mask, pos_pts, neg_pts, boxes, _ = dataset[idx]
print(f"Running inference on Image Index: {idx}")

# --- 3. Format Prompts ---
pos_pts_np = np.array(pos_pts) if len(pos_pts) > 0 else np.empty((0, 2))
neg_pts_np = np.array(neg_pts) if len(neg_pts) > 0 else np.empty((0, 2))
box_np = np.array(boxes) if len(boxes) > 0 else None

points = np.concatenate((pos_pts_np, neg_pts_np), axis=0) if len(neg_pts_np) > 0 else pos_pts_np
labels = np.array([1] * len(pos_pts_np) + [0] * len(neg_pts_np))

if len(points) == 0:
    points, labels = None, None

# --- 4. Predict using Native API ---
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        predictor.set_image(img)
        masks_pred, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box_np,
            multimask_output=False 
        )

predicted_mask = masks_pred.squeeze() 

# --- 5. Visualize! ---
fig, axes = plt.subplots(1, 3, figsize=(20, 10))

axes[0].imshow(img)
if points is not None:
    show_points(points, labels, axes[0])
show_box(box_np, axes[0])
axes[0].set_title(f"Input Image & Prompts (Idx: {idx})", fontsize=14, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(img)
show_mask(gt_mask, axes[1], color=np.array([0, 1, 0])) 
axes[1].set_title("Ground Truth Mask", fontsize=14, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(img)
show_mask(predicted_mask, axes[2], color=np.array([1, 0, 0])) 
axes[2].set_title(f"Model Prediction (Conf: {scores[0]:.3f})", fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
# %%
