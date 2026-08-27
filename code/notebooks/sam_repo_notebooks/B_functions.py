import logging
import math
import os
import re
import cv2
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import queue
import random
import threading
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# --- LoRA model helpers ---

_lora_logger = logging.getLogger(__name__)


class _LoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with Low-Rank Adaptation (inference-ready)."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.lora_A  = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B  = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base  = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + delta


def _inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
    target_hiera: bool = True,
    target_mask_decoder: bool = True,
    target_hiera_mlp: bool = False,
    target_mask_decoder_mlp: bool = False,
) -> nn.Module:
    """Inject LoRA layers into SAM2 model in-place."""
    if target_hiera:
        for block in model.image_encoder.trunk.blocks:
            block.attn.qkv  = _LoRALinear(block.attn.qkv,  rank, alpha)
            block.attn.proj = _LoRALinear(block.attn.proj, rank, alpha)

    if target_hiera_mlp:
        for block in model.image_encoder.trunk.blocks:
            block.mlp.layers[0] = _LoRALinear(block.mlp.layers[0], rank, alpha)
            block.mlp.layers[1] = _LoRALinear(block.mlp.layers[1], rank, alpha)

    if target_mask_decoder:
        transformer = model.sam_mask_decoder.transformer
        for layer in transformer.layers:
            for attn_name in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
                attn = getattr(layer, attn_name)
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    setattr(attn, proj, _LoRALinear(getattr(attn, proj), rank, alpha))
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(
                transformer.final_attn_token_to_image, proj,
                _LoRALinear(getattr(transformer.final_attn_token_to_image, proj), rank, alpha),
            )

    if target_mask_decoder_mlp:
        for layer in model.sam_mask_decoder.transformer.layers:
            layer.mlp.layers[0] = _LoRALinear(layer.mlp.layers[0], rank, alpha)
            layer.mlp.layers[1] = _LoRALinear(layer.mlp.layers[1], rank, alpha)

    return model


def _infer_model_size(state_dict):
    """Infer SAM2 model size from the patch-embedding weight's output channels."""
    key = "image_encoder.trunk.patch_embed.proj.weight"
    if key not in state_dict:
        return "b+"
    embed_dim = int(state_dict[key].shape[0])
    if embed_dim == 144:
        return "l"
    if embed_dim == 112:
        return "b+"
    if embed_dim == 96:
        block_keys = {k for k in state_dict if k.startswith("image_encoder.trunk.blocks.")}
        n_blocks = max((int(k.split(".")[4]) for k in block_keys), default=0) + 1
        return "t" if n_blocks <= 12 else "s"
    return "b+"


def _infer_lora_config(state_dict):
    """Auto-detect LoRA rank and injection targets from checkpoint state dict."""
    rank = None
    target_hiera = target_hiera_mlp = target_mask_decoder = target_mask_decoder_mlp = False
    for k, v in state_dict.items():
        if not k.endswith("lora_A"):
            continue
        if rank is None:
            rank = int(v.shape[0])
        if "image_encoder.trunk" in k:
            if ".mlp." in k:
                target_hiera_mlp = True
            else:
                target_hiera = True
        elif "sam_mask_decoder" in k:
            if ".mlp." in k:
                target_mask_decoder_mlp = True
            else:
                target_mask_decoder = True
    return {
        "rank": rank if rank is not None else 4,
        "target_hiera": target_hiera,
        "target_hiera_mlp": target_hiera_mlp,
        "target_mask_decoder": target_mask_decoder,
        "target_mask_decoder_mlp": target_mask_decoder_mlp,
    }


def build_sam2_lora(checkpoint_path, device="cuda"):
    """Load a LoRA fine-tuned SAM2 checkpoint for inference.

    Infers model size and LoRA topology from the checkpoint itself, so no
    separate config file is needed.
    """
    import sam2 as _sam2_pkg
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    size = _infer_model_size(state_dict)
    lora_cfg = _infer_lora_config(state_dict)
    effective_alpha = float(lora_cfg["rank"] * 2)

    _SIZE_TO_CFG = {
        "t":  ("sam2_hiera_t.yaml",   "sam2.1_hiera_t.yaml"),
        "s":  ("sam2_hiera_s.yaml",   "sam2.1_hiera_s.yaml"),
        "b+": ("sam2_hiera_b+.yaml",  "sam2.1_hiera_b+.yaml"),
        "l":  ("sam2_hiera_l.yaml",   "sam2.1_hiera_l.yaml"),
    }
    flat_name, v21_name = _SIZE_TO_CFG[size]
    pkg_dir = _sam2_pkg.__path__[0]
    flat_path = os.path.join(pkg_dir, flat_name)
    v21_path  = os.path.join(pkg_dir, "configs", "sam2.1", v21_name)

    if os.path.exists(flat_path):
        cfg_path = flat_path
    elif os.path.exists(v21_path):
        cfg_path = v21_path
    else:
        raise FileNotFoundError(
            f"SAM2 config for size '{size}' not found in {pkg_dir}"
        )

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    _inject_lora(
        model,
        rank=lora_cfg["rank"],
        alpha=effective_alpha,
        target_hiera=lora_cfg["target_hiera"],
        target_mask_decoder=lora_cfg["target_mask_decoder"],
        target_hiera_mlp=lora_cfg["target_hiera_mlp"],
        target_mask_decoder_mlp=lora_cfg["target_mask_decoder_mlp"],
    )

    missing, _ = model.load_state_dict(state_dict, strict=False)
    lora_missing = [k for k in missing if "lora" not in k]
    if lora_missing:
        print(f"  Warning: non-LoRA keys missing from checkpoint: {lora_missing[:5]}")

    model.eval()
    model.to(device)
    return model


# Courtesy of sam2 automatic_mask_generator.py show_anns function
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

# overlay segmented image on dimmed original image
def image_overlay(image, segmented_image):
    # from https://learnopencv.com/sam-2/#aioseo-running-inferencce-on-videos-using-sam-2
    alpha = 0.6 # transparency for the original image
    beta = 0.4 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
 
    segmented_image = np.array(segmented_image, dtype=np.float32)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
 
    image = np.array(image, dtype=np.float32) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

# create predicted points overlay
def show_points(masks, ax, point_size=50):
    """
    Plots points on top of the current axis using the crop_box.
    """
    for mask in masks:
        points = np.array(mask['point_coords']) # Shape: [N, 2]
        
        for i, point in enumerate(points):
            ax.scatter(point[0], point[1], color='blue', marker='s', 
                       s=point_size, edgecolors='white', linewidth=1)

# --- Helper Function: Save Masks to npz ---
def save_masks_task(file_name, masks, output_dir):
    """
    Saves the entire masks list (list of dicts) directly into a 
    compressed .npz file.
    """
    try:
        base_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}_masks.npz")
        
        # Convert the list of dictionaries into a single NumPy object array
        masks_array = np.array(masks, dtype=object)
        np.savez_compressed(save_path, masks=masks_array)
        
    except Exception as e:
        print(f"Error saving {file_name}: {e}")

# --- Helper: Background Image Loader ---
def image_prefetcher(image_files, prefetch_queue):
    """Loads images from disk into memory in the background to hide IO latency."""
    for img_path in image_files:
        try:
            # cv2 is faster for TIFs; forces 8-bit RGB for SAM2 compatibility
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None: raise ValueError("Could not read image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prefetch_queue.put((os.path.basename(img_path), image))
        except Exception as e:
            print(f"Prefetch error on {img_path}: {e}")
            prefetch_queue.put(None) 
    prefetch_queue.put("DONE")

# --- initialize gpu worker ---
def gpu_worker(gpu_id, image_files, config_path, checkpoint_path, output_dir, progress_queue, is_lora=False):
    """
    Worker function to run on a specific GPU thread.
    Supports both base SAM2.1 models and custom LoRA fine-tuned checkpoints.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    # Setup Device
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Optimization Settings
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(device).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if is_lora:
        model = build_sam2_lora(checkpoint_path, device=device)
    else:
        from sam2.build_sam import build_sam2
        model = build_sam2(config_path, checkpoint_path, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )

    # Start the Prefetch Thread (Buffering 5-10 images)
    prefetch_queue = queue.Queue(maxsize=10) 
    prefetch_thread = threading.Thread(target=image_prefetcher, args=(image_files, prefetch_queue))
    prefetch_thread.start()

    # Processing Loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as local_save_executor:
        while True:
            data = prefetch_queue.get()
            if data == "DONE": break
            if data is None:
                progress_queue.put(1)
                continue

            img_name, image_array = data
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks = mask_generator.generate(image_array)
                
                local_save_executor.submit(save_masks_task, img_name, masks, output_dir)
                progress_queue.put(1) # Notify success

            except Exception as e:
                print(f"[GPU {gpu_id}] Inference error: {e}")
                progress_queue.put(1) # Notify skip

    prefetch_thread.join()
    print(f"[GPU {gpu_id}] Finished.")


def image_available_check(INPUT_FOLDER):
    """
    Check if an image can be opened successfully.
    """
    print(f"Checking images in: {INPUT_FOLDER}")

    # 1. Collect Images
    image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.png")) + \
                glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
                glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    image_files.sort() # Sort to ensure deterministic splitting

    if len(image_files) == 0:
        print("No images found, check the input folder path")
    else:
        print(f"Total images found: {len(image_files)}")
    
# parallel processing with gpu_worker
def two_gpu_predictor(INPUT_FOLDER, OUTPUT_FOLDER, model_cfg, sam2_checkpoint, is_lora=False):
    """
    Main function to run multi-GPU prediction.
    Splits data between 2 GPUs and uses a shared CPU thread pool for saving.
    Pass is_lora=True for LoRA fine-tuned checkpoints (model_cfg is ignored).
    """
    # 1. Collect Images
    image_files = glob.glob(os.path.join(INPUT_FOLDER, "*.png")) + \
                glob.glob(os.path.join(INPUT_FOLDER, "*.jpg")) + \
                glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    image_files.sort() # Sort to ensure deterministic splitting

    if len(image_files) == 0:
        print("No images found!")
    else:
        print(f"Total images found: {len(image_files)}")
    
    # 2. Prepare Multiprocessing
    mp.set_start_method('spawn', force=True)
    progress_queue = mp.Queue()
    mid = len(image_files) // 2

    # 3. Create Processes
    p1 = mp.Process(target=gpu_worker, args=(0, image_files[:mid],  model_cfg, sam2_checkpoint, OUTPUT_FOLDER, progress_queue, is_lora))
    p2 = mp.Process(target=gpu_worker, args=(1, image_files[mid:], model_cfg, sam2_checkpoint, OUTPUT_FOLDER, progress_queue, is_lora))

    p1.start()
    p2.start()

    # 4. UI: Shared Progress Bar (Main Process)
    pbar = tqdm(total=len(image_files), desc="Total Segmentation Progress", unit="img")
    
    completed = 0
    while completed < len(image_files):
        try:
            # Blocks until a result comes in from either GPU
            _ = progress_queue.get(timeout=60) 
            pbar.update(1)
            completed += 1
        except queue.Empty:
            # Safety break if GPUs hang
            break

    p1.join()
    p2.join()
    pbar.close()
    print("All processing and saving complete.")

# ---------- Visual Inspection Function ----------
def visualize_mask_samples(input_folder, output_folder, num_samples=4, alpha=0.5, draw_borders=True, min_masks=0):
    """
    Randomly selects .npz files (list of dicts), matches with images,
    and displays them with borders and prompt points.
    min_masks: skip files whose mask count is below this threshold.
    """
    processed_files = glob.glob(os.path.join(output_folder, "*_masks.npz"))
    if not processed_files:
        print(f"No mask files found in {output_folder}")
        return
    if min_masks > 0:
        filtered = []
        for f in processed_files:
            data = np.load(f, allow_pickle=True)
            if len(data['masks']) >= min_masks:
                filtered.append(f)
        skipped = len(processed_files) - len(filtered)
        if skipped:
            print(f"Skipped {skipped} files with fewer than {min_masks} masks.")
        processed_files = filtered
    if not processed_files:
        print(f"No files remaining after applying min_masks={min_masks}.")
        return
    sampled_files = random.sample(processed_files, min(len(processed_files), num_samples))

    cols = 2
    rows = max(1, (len(sampled_files) + 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < len(sampled_files):
            test_file_path = sampled_files[i]
            base_name = os.path.basename(test_file_path).replace("_masks.npz", "")
            
            # Match image
            img_path = None
            for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".TIF", ".TIFF"]:
                p = os.path.join(input_folder, base_name + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path:
                # Load Data, using allow_pickle=True for list of dicts 'masks'
                image = Image.open(img_path).convert("RGB")
                img_np = np.array(image)
                data = np.load(test_file_path, allow_pickle=True)
                masks = data['masks'] 

                # Process Overlay (partly apply logic from show_anns)
                # Sort by area descending to draw larger masks first
                sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                
                h, w = img_np.shape[:2]
                overlay = np.zeros((h, w, 4), dtype=np.float32)
                
                for ann in sorted_masks:
                    m = ann['segmentation']
                    color = np.concatenate([np.random.random(3), [alpha]])
                    overlay[m] = color
                    
                    if draw_borders:
                        # OpenCV border logic from your snippet
                        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
                        cv2.drawContours(overlay, contours, -1, (1, 1, 1, 0.8), thickness=1) # White borders

                # 4. Plotting
                ax.imshow(img_np)
                ax.imshow(overlay)
                
                # 5. Add Points (Logic from show_points)
                for ann in masks:
                    if 'point_coords' in ann:
                        pts = np.array(ann['point_coords'])
                        ax.scatter(pts[:, 0], pts[:, 1], color='blue', marker='s', s=40, edgecolors='white')

                ax.set_title(f"{base_name} ({len(masks)} masks)")
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

