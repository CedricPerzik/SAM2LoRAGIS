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
import numpy as np
import pandas as pd
import cv2
import matplotlib
import torch
import torchvision

# %%
# Check library versions
# CUDA home shows N/A if not set in environment variables
# Not sure if this is a problem.
libraries = [
    ("numpy", np.__version__),
    ("pandas", pd.__version__),
    ("opencv-python", cv2.__version__),
    ("matplotlib", matplotlib.__version__),
    ("torch", torch.__version__),
    ("torchvision", torchvision.__version__),
    ("CUDA available", torch.cuda.is_available()),
    ("PyTorch CUDA version", torch.version.cuda),
    ('CUDA_PATH', os.environ.get("CUDA_PATH", "N/A")),
    ('CUDA_HOME', os.environ.get("CUDA_HOME", "N/A")),
    ("cuDNN version", torch.backends.cudnn.version()),
    ("GPU_0 device name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"),
    ("GPU_1 device name", torch.cuda.get_device_name(1) if torch.cuda.is_available() else "N/A"),
]

max_len = max(len(name) for name, _ in libraries)

for name, version in libraries:
    print(f"{name:>{max_len}} - {version}")

# %%
# !nvcc --version
# %%
# !nvidia-smi
# %%
