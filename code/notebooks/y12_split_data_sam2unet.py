# RUN IN TERMINAL: python y12_split_data_sam2unet.py
import os
import shutil
import random

# --- CONFIGURATION ---
CREATE_TEST_SET = False  # Toggle this flag to True/False
TRAIN_PCT = 0.70        # e.g., 70% for training, else 80% if no test set is created
VAL_PCT = 0.15          # e.g., 15% for validation, else 20% if no test set is created
# Test set automatically gets the remainder (15%)

# Absolute paths
images_dir = "/home/ced/Documents/master_thesis/workingdir/code/output_data/models/sam2unet/tiles/ceu_paz/tiles_png/"
masks_dir = "/home/ced/Documents/master_thesis/workingdir/code/output_data/models/sam2unet/masks_5cm/ceu_paz/"
output_base = "/home/ced/Documents/master_thesis/workingdir/code/output_data/models/sam2unet/split_data/ceu_paz/"

# Dynamically set up directories based on the flag
dirs = {
    "train_img": os.path.join(output_base, "train", "images"),
    "train_mask": os.path.join(output_base, "train", "masks"),
    "val_img": os.path.join(output_base, "val", "images"),
    "val_mask": os.path.join(output_base, "val", "masks")
}

if CREATE_TEST_SET:
    dirs["test_img"] = os.path.join(output_base, "test", "images")
    dirs["test_mask"] = os.path.join(output_base, "test", "masks")

# Create directories
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Look at the MASKS directory to get our list of valid (labeled) files
all_masks = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
valid_files = [f for f in all_masks if os.path.exists(os.path.join(images_dir, f))]

random.seed(69)  # For reproducibility
random.shuffle(valid_files)

# Calculate splits
total_files = len(valid_files)

if CREATE_TEST_SET:
    train_end = int(total_files * TRAIN_PCT)
    val_end = train_end + int(total_files * VAL_PCT)
    
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:val_end]
    test_files = valid_files[val_end:]
else:
    # Standard 80/20 fallback
    train_end = int(total_files * 0.8)
    train_files = valid_files[:train_end]
    val_files = valid_files[train_end:]
    test_files = []

def copy_files(file_list, img_dest, mask_dest):
    for f in file_list:
        shutil.copy(os.path.join(images_dir, f), os.path.join(img_dest, f))
        shutil.copy(os.path.join(masks_dir, f), os.path.join(mask_dest, f))

# Execute the copying
copy_files(train_files, dirs["train_img"], dirs["train_mask"])
copy_files(val_files, dirs["val_img"], dirs["val_mask"])

# Print appropriate summary
if CREATE_TEST_SET:
    copy_files(test_files, dirs["test_img"], dirs["test_mask"])
    print(f"Clean split complete! Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
else:
    print(f"Clean split complete! Train: {len(train_files)}, Val: {len(val_files)}")