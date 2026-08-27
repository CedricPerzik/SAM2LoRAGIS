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
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import B_functions as bf
 
# %%
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
 
if torch.cuda.get_device_properties(0).major >= 8:
    print("Using Ampere or newer GPU -> enabling tfloat32")
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# %%
ceupaz_sample = "/home/ced/Documents/master_thesis/workingdir/code/output_data/tiles/ceu_paz/ceu_paz_row_2_col_4.tif"
image = Image.open(ceupaz_sample)
image = np.array(image.convert("RGB"))

# %%
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# %%
masks = mask_generator.generate(image)

# %%
print(f"{len(masks)}\n")
print(f"{masks[0].keys()}\n")
print(masks[np.random.randint(0, len(masks))])

# %%
# visualize masks
plt.figure(figsize=(20, 20))
plt.imshow(image)
bf.show_anns(masks)
plt.axis('off')
plt.show() 

# %%
# visualize masks on darkened image
shaded_image = bf.image_overlay(image, masks[0]['segmentation'])
plt.figure(figsize=(20, 20))
plt.imshow(shaded_image)
bf.show_anns(masks)
plt.axis('off')
plt.show() 

# %%
# visualize masks and their prompt on darkened image
plt.figure(figsize=(20, 20))
plt.imshow(shaded_image)
bf.show_anns(masks)
bf.show_points(masks, plt.gca())
plt.axis('off')
plt.show()

# %%
