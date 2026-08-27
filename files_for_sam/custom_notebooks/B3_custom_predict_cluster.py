import os
import re
import glob
import json
import cv2
import torch
import concurrent.futures
from tqdm.auto import tqdm
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import B_functions as bf

# RUN_ID and DATASET_SIZE are injected by SLURM via --export
_raw_n = os.environ.get("DATASET_SIZE", "")
_raw_run = os.environ.get("RUN_ID", "")
if not _raw_n:
    raise RuntimeError("DATASET_SIZE environment variable not set")
if not _raw_run:
    raise RuntimeError("RUN_ID environment variable not set")
N = int(_raw_n)
RUN_ID = _raw_run

# Cluster paths
CHECKPOINT_DIR = (
    f"/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs/{RUN_ID}/n{N}/checkpoints"
)
PREDICTIONS_BASE = "/home/s2145588/thesis/data/predictions"
PNG_IMAGES_ROOT = "/home/s2145588/thesis/data/favela_png/images"
TEST_SPLIT_MANIFEST = f"/home/s2145588/thesis/data/test_split/{RUN_ID}/test_tiles.json"
OOD_AREA_NAME = "santa_madalena"
OOD_INPUT_FOLDER = os.path.join(PNG_IMAGES_ROOT, OOD_AREA_NAME)
MODEL_NAME = f"{RUN_ID}/n{N}"


def _find_best_checkpoint(ckpt_dir):
    """Return *_best.pt if present, otherwise the highest-numbered checkpoint."""
    best_pattern = re.compile(r"checkpoint_(\d+)_best\.pt$")
    best = [
        (int(best_pattern.search(f).group(1)), f)
        for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_*_best.pt"))
        if best_pattern.search(f)
    ]
    if best:
        return max(best, key=lambda x: x[0])[1]
    pattern = re.compile(r"checkpoint_(\d+)\.pt$")
    numbered = [
        (int(pattern.search(f).group(1)), f)
        for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_*.pt"))
        if pattern.search(f)
    ]
    if numbered:
        return max(numbered, key=lambda x: x[0])[1]
    raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")


def _build_id_test_jobs():
    """Return [(area_name, image_files), ...] for the ID held-out test tiles.

    Reads this run's test-split manifest (written by train.py during training
    via split_utils.dump_test_manifest, see --test-split-dir). Returns [] if
    the manifest is missing (older run, or test_fraction=0 for this run) --
    a missing manifest must never crash the inference job, just skip ID
    test-tile inference and fall back to OOD-only.
    """
    if not os.path.exists(TEST_SPLIT_MANIFEST):
        print(f"No test-split manifest at {TEST_SPLIT_MANIFEST}, skipping ID test-tile inference")
        return []
    try:
        with open(TEST_SPLIT_MANIFEST) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Could not read test-split manifest {TEST_SPLIT_MANIFEST}: {e}")
        print("Skipping ID test-tile inference")
        return []

    jobs = []
    for region, tile_names in manifest.get("test_tiles", {}).items():
        image_files = sorted(
            os.path.join(PNG_IMAGES_ROOT, region, f"{tile_name}.png")
            for tile_name in tile_names
        )
        image_files = [p for p in image_files if os.path.exists(p)]
        if not image_files:
            print(f"No test-tile images found for region {region}, skipping")
            continue
        jobs.append((region, image_files))
    return jobs


def run_inference(image_files, checkpoint_path, output_dir, n, area_name):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Loading LoRA model...")
    model = bf.build_sam2_lora(checkpoint_path, device=device)
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
    print(f"Model loaded. Running inference on {len(image_files)} images...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as save_pool:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for img_path in tqdm(image_files, desc=f"n{n} {area_name} inference", unit="img"):
                img_name = os.path.basename(img_path)
                try:
                    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if image is None:
                        print(f"  Could not read {img_path}, skipping")
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    masks = mask_generator.generate(image)
                    save_pool.submit(bf.save_masks_task, img_name, masks, output_dir)
                except Exception as e:
                    import traceback
                    print(f"  Error on {img_name}: {e}")
                    traceback.print_exc()

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    checkpoint_path = _find_best_checkpoint(CHECKPOINT_DIR)

    print(f"Run ID       : {RUN_ID}")
    print(f"Dataset size : n{N}")
    print(f"Checkpoint   : {checkpoint_path}")

    # OOD job: santa_madalena is never part of the train/val/test split, so it
    # is always evaluated whole-region.
    ood_image_files = (
        glob.glob(os.path.join(OOD_INPUT_FOLDER, "*.png"))
        + glob.glob(os.path.join(OOD_INPUT_FOLDER, "*.jpg"))
        + glob.glob(os.path.join(OOD_INPUT_FOLDER, "*.tif"))
    )
    ood_image_files.sort()
    if not ood_image_files:
        raise RuntimeError(f"No images found in {OOD_INPUT_FOLDER}")

    jobs = [(OOD_AREA_NAME, ood_image_files)]
    jobs.extend(_build_id_test_jobs())

    for area_name, image_files in jobs:
        output_folder = os.path.join(PREDICTIONS_BASE, MODEL_NAME, area_name)
        print(f"\nArea         : {area_name}")
        print(f"Images found : {len(image_files)}")
        print(f"Output       : {output_folder}")

        os.makedirs(output_folder, exist_ok=True)
        run_inference(image_files, checkpoint_path, output_folder, N, area_name)
