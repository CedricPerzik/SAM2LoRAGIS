"""Evaluate a fine-tuned SAM2LoRA checkpoint on favela rooftop segmentation.

For each instance in every region:
  1. Sample a single positive click from the mask centroid.
  2. Run SAM2ImagePredictor → 3 candidate masks.
  3. Select the mask with highest predicted IoU score.
  4. Compute actual IoU vs ground truth.

Reports per-region and overall mean IoU, mean Dice, and recall.

Optionally saves side-by-side visualisations (image | GT mask | predicted mask),
cropped to each instance's bounding box with padding.

Two data loading modes:

  PNG cache mode (recommended, --png_cache_dir):
    <png_cache_dir>/images/<region>/  ->  *.png  (8-bit RGB, matches training)
    <png_cache_dir>/masks/<region>/<tile_name>/<i:04d>.png + _meta.json

  TIF+NPZ mode (--tiles_root / --masks_root):
    <tiles_root>/<region>/  ->  *.tif / *.png
    <masks_root>/<region>/  ->  *_gt.npz

Usage (from the sam2/ directory):
    python ../scripts/eval_lora.py \
        --checkpoint sam2_logs/l_r8_v1/checkpoints/checkpoint_45.pt \
        --regions ceu_paz cantidio_sampaio santa_madalena \
        --png_cache_dir /home/ced/Documents/master_thesis/workingdir/code/output_data/favela_png \
        --test_split_manifest /path/to/test_split/run_001/test_tiles.json \
        --vis_dir ./eval_vis \
        --max_vis 10

--test_split_manifest restricts any region that appears in the manifest's
"test_tiles" (i.e. the ID regions used for the train/val/test split, e.g.
ceu_paz/cantidio_sampaio) to just the held-out test tiles, so the reported ID
metrics do not leak train/val tiles. Regions absent from the manifest (e.g.
the OOD region santa_madalena, which is never part of the split) are always
evaluated whole-region, same as before. Omit the flag entirely to evaluate
every region whole-region (the old, pre-split-aware behaviour).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/sam2")

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

from sam2.modeling.lora import inject_lora
from sam2.sam2_image_predictor import SAM2ImagePredictor
from omegaconf import OmegaConf
from hydra.utils import instantiate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_model_size(state_dict: dict) -> str:
    """Infer SAM2 model size from the patch-embedding weight's output channels (embed_dim).

    embed_dim=144 → large; 112 → base+; 96 → small or tiny (resolved by block count).
    Falls back to 'b+' if the key is absent.
    """
    key = "image_encoder.trunk.patch_embed.proj.weight"
    if key not in state_dict:
        return "b+"
    embed_dim = int(state_dict[key].shape[0])
    if embed_dim == 144:
        return "l"
    if embed_dim == 112:
        return "b+"
    if embed_dim == 96:
        # tiny: stages [1,2,7,2] = 12 blocks; small: [1,2,11,2] = 16 blocks
        block_keys = {k for k in state_dict if k.startswith("image_encoder.trunk.blocks.")}
        n_blocks = max((int(k.split(".")[4]) for k in block_keys), default=0) + 1
        return "t" if n_blocks <= 12 else "s"
    return "b+"


def _infer_lora_config(state_dict: dict) -> dict:
    """Auto-detect LoRA rank and injection targets from a checkpoint state dict.

    Reads the shape of the first lora_A tensor to get rank, then scans all
    lora_A keys to figure out which modules were adapted.
    """
    rank = None
    target_hiera = False
    target_hiera_mlp = False
    target_mask_decoder = False
    target_mask_decoder_mlp = False

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


_SIZE_TO_CFG = {
    "t":  "sam2_hiera_t.yaml",
    "s":  "sam2_hiera_s.yaml",
    "b+": "sam2_hiera_b+.yaml",
    "l":  "sam2_hiera_l.yaml",
}


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    model_size: str | None = None,
) -> SAM2ImagePredictor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    size = model_size if model_size is not None else _infer_model_size(state_dict)
    cfg_name = _SIZE_TO_CFG.get(size, "sam2_hiera_b+.yaml")
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../sam2/sam2",
        cfg_name,
    )
    print(f"  Model size: {size}  (config: {cfg_name})")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    lora_cfg = _infer_lora_config(state_dict)
    if lora_rank is not None:
        lora_cfg["rank"] = lora_rank
    # alpha is not stored in the checkpoint; default to 2 * rank (matches the
    # training convention: alpha=32 for rank=16 → scaling = 2.0).
    effective_alpha = lora_alpha if lora_alpha is not None else float(lora_cfg["rank"] * 2)

    print(
        f"  LoRA config: rank={lora_cfg['rank']}, alpha={effective_alpha:.1f}  "
        f"(hiera={lora_cfg['target_hiera']}, hiera_mlp={lora_cfg['target_hiera_mlp']}, "
        f"decoder={lora_cfg['target_mask_decoder']}, decoder_mlp={lora_cfg['target_mask_decoder_mlp']})"
    )

    inject_lora(
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
    return SAM2ImagePredictor(model)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def centroid_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (col, row) centroid of a binary mask, or None if mask is empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def mask_bbox(mask: np.ndarray, padding: int = 80) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) bounding box of mask pixels, expanded by padding."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    H, W = mask.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(W, int(xs.max()) + padding)
    y1 = min(H, int(ys.max()) + padding)
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter) / float(denom) if denom > 0 else 1.0


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _mask_overlay(image_crop: np.ndarray, mask_crop: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    """Blend a semi-transparent coloured mask onto an RGB crop."""
    out = image_crop.astype(np.float32).copy()
    for c, val in enumerate(color):
        out[..., c] = np.where(mask_crop, out[..., c] * (1 - alpha) + val * alpha, out[..., c])
    return np.clip(out, 0, 255).astype(np.uint8)


def save_vis(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    iou: float,
    region: str,
    tile_name: str,
    inst_idx: int,
    out_path: str,
    padding: int = 80,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    bbox = mask_bbox(gt_mask, padding=padding)
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox

    img_crop  = image_rgb[y0:y1, x0:x1]
    gt_crop   = gt_mask[y0:y1, x0:x1]
    pred_crop = pred_mask[y0:y1, x0:x1]

    gt_vis   = _mask_overlay(img_crop, gt_crop,   color=(0, 220, 80),  alpha=0.50)
    pred_vis = _mask_overlay(img_crop, pred_crop, color=(30, 144, 255), alpha=0.50)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"[{region}]  {tile_name}  —  instance {inst_idx}  |  IoU = {iou:.3f}",
        fontsize=13,
    )

    axes[0].imshow(img_crop)
    axes[0].set_title("Image", fontsize=11)

    axes[1].imshow(gt_vis)
    gt_patch = mpatches.Patch(color=(0, 220/255, 80/255), alpha=0.7, label="Ground truth")
    axes[1].legend(handles=[gt_patch], loc="lower right", fontsize=9)
    axes[1].set_title("Ground Truth", fontsize=11)

    axes[2].imshow(pred_vis)
    pred_patch = mpatches.Patch(color=(30/255, 144/255, 1.0), alpha=0.7, label="Prediction")
    axes[2].legend(handles=[pred_patch], loc="lower right", fontsize=9)
    axes[2].set_title("Prediction", fontsize=11)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _iter_instances_png(
    region: str, png_cache_dir: str, filter_artifacts: bool, allowed_tiles: set[str] | None = None
):
    """Yield (tile_name, image_rgb, gt_mask) from the PNG cache.

    allowed_tiles: if given, tiles whose name is not in this set are skipped
    (used to restrict ID regions to their held-out test split).
    """
    img_dir = os.path.join(png_cache_dir, "images", region)
    msk_dir = os.path.join(png_cache_dir, "masks", region)
    tile_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    for tile_file in tqdm(tile_files, desc=f"[{region}]", unit="tile"):
        tile_name = os.path.splitext(tile_file)[0]
        if allowed_tiles is not None and tile_name not in allowed_tiles:
            continue
        meta_path = os.path.join(msk_dir, tile_name + "_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        with PILImage.open(os.path.join(img_dir, tile_file)) as im:
            image_rgb = np.array(im.convert("RGB"))
        for i, inst_meta in enumerate(metadata):
            if filter_artifacts and inst_meta.get("is_artifact", False):
                continue
            mask_path = os.path.join(msk_dir, tile_name, f"{i:04d}.png")
            if not os.path.exists(mask_path):
                continue
            with PILImage.open(mask_path) as m:
                gt_mask = np.array(m.convert("L")) > 0
            yield tile_name, i, image_rgb, gt_mask


def _iter_instances_npz(
    region: str, tiles_root: str, masks_root: str, filter_artifacts: bool, allowed_tiles: set[str] | None = None
):
    """Yield (tile_name, inst_idx, image_rgb, gt_mask) from TIF+NPZ files.

    allowed_tiles: if given, tiles whose name is not in this set are skipped
    (used to restrict ID regions to their held-out test split).
    """
    t_dir = os.path.join(tiles_root, region)
    m_dir = os.path.join(masks_root, region)
    tile_files = sorted(f for f in os.listdir(t_dir) if f.endswith((".tif", ".png")))
    for tile_file in tqdm(tile_files, desc=f"[{region}]", unit="tile"):
        tile_name = os.path.splitext(tile_file)[0]
        if allowed_tiles is not None and tile_name not in allowed_tiles:
            continue
        mask_path = os.path.join(m_dir, tile_name + "_gt.npz")
        if not os.path.exists(mask_path):
            continue
        with PILImage.open(os.path.join(t_dir, tile_file)) as im:
            image_rgb = np.array(im.convert("RGB"))
        masks_arr = np.load(mask_path, allow_pickle=True)["masks"]
        for inst_idx, mask_dict in enumerate(masks_arr):
            if filter_artifacts and mask_dict.get("is_artifact", False):
                continue
            gt_mask = mask_dict["segmentation"].astype(bool)
            yield tile_name, inst_idx, image_rgb, gt_mask


def evaluate_region(
    predictor: SAM2ImagePredictor,
    region: str,
    tiles_root: str,
    masks_root: str,
    filter_artifacts: bool = True,
    device: str = "cuda",
    vis_dir: str | None = None,
    max_vis: int = 10,
    vis_padding: int = 80,
    png_cache_dir: str | None = None,
    allowed_tiles: set[str] | None = None,
) -> dict:
    if vis_dir:
        region_vis_dir = os.path.join(vis_dir, region)
        os.makedirs(region_vis_dir, exist_ok=True)

    ious, dices, recalls = [], [], []
    skipped = 0
    vis_count = 0
    prev_tile_name = None

    if png_cache_dir is not None:
        instances = _iter_instances_png(region, png_cache_dir, filter_artifacts, allowed_tiles)
    else:
        instances = _iter_instances_npz(region, tiles_root, masks_root, filter_artifacts, allowed_tiles)

    for tile_name, inst_idx, image_rgb, gt_mask in instances:
        if gt_mask.sum() == 0:
            skipped += 1
            continue

        pt = centroid_point(gt_mask)
        if pt is None:
            skipped += 1
            continue

        if tile_name != prev_tile_name:
            predictor.set_image(image_rgb)
            prev_tile_name = tile_name
            saved_this_tile = False

        with torch.inference_mode():
            pred_masks, scores, _ = predictor.predict(
                point_coords=np.array([[pt[0], pt[1]]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

        best_idx = int(scores.argmax())
        pred_mask = pred_masks[best_idx].astype(bool)

        iou    = compute_iou(pred_mask, gt_mask)
        dice   = compute_dice(pred_mask, gt_mask)
        recall = compute_recall(pred_mask, gt_mask)

        ious.append(iou)
        dices.append(dice)
        recalls.append(recall)

        if vis_dir and vis_count < max_vis and not saved_this_tile:
            fname = f"vis{vis_count + 1:03d}_{tile_name}.png"
            out_path = os.path.join(region_vis_dir, fname)
            save_vis(
                image_rgb, gt_mask, pred_mask, iou,
                region, tile_name, inst_idx, out_path, padding=vis_padding,
            )
            vis_count += 1
            saved_this_tile = True

    return {
        "n": len(ious),
        "skipped": skipped,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "iou_vals": ious,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="/home/ced/Documents/master_thesis/workingdir/stage_3_models/models/SAM2LoRaBoRaDD/sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to checkpoint. Defaults to the base SAM2.1-L checkpoint (no LoRA).",
    )
    parser.add_argument(
        "--regions", nargs="+",
        default=["ceu_paz", "cantidio_sampaio", "santa_madalena"],
    )
    parser.add_argument(
        "--tiles_root",
        default="/home/ced/Documents/master_thesis/workingdir/code/output_data/tiles",
    )
    parser.add_argument(
        "--masks_root",
        default="/home/ced/Documents/master_thesis/workingdir/code/output_data/ground_truth_npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help=(
            "Override LoRA rank. Auto-detected from checkpoint by default "
            "(reads the shape of the first lora_A tensor)."
        ),
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help=(
            "Override LoRA alpha scaling factor. Defaults to 2 × detected rank "
            "(matching the training convention alpha=32 for rank=16)."
        ),
    )
    parser.add_argument(
        "--model_size",
        choices=["t", "s", "b+", "l"],
        default=None,
        help="Override model size (auto-detected from checkpoint by default).",
    )
    parser.add_argument(
        "--png_cache_dir",
        default=None,
        help=(
            "Use pre-converted PNG cache for eval (recommended). "
            "Matches the image preprocessing used during training and avoids "
            "geospatial TIFF colour-space issues. "
            "Example: /path/to/favela_png"
        ),
    )
    parser.add_argument(
        "--test_split_manifest",
        default=None,
        help=(
            "Path to a test_tiles.json manifest written by "
            "training/dataset/split_utils.dump_test_manifest (see train.py's "
            "--test-split-dir). When given, any region present in the manifest's "
            "test_tiles (the ID regions, e.g. ceu_paz/cantidio_sampaio) is restricted "
            "to just its held-out test tiles, avoiding train/val leakage in the "
            "reported ID metrics. Regions absent from the manifest (e.g. the OOD "
            "region santa_madalena) are always evaluated whole-region."
        ),
    )
    parser.add_argument(
        "--vis_dir", default=None,
        help="Directory to save visualisations. Omit to skip.",
    )
    parser.add_argument(
        "--max_vis", type=str, default="10",
        help="Max visualisations saved per region. Use 'all' to save one per tile (default 10).",
    )
    parser.add_argument(
        "--vis_padding", type=int, default=80,
        help="Pixel padding around each mask bounding box (default 80).",
    )
    args = parser.parse_args()
    if args.max_vis.lower() == "all":
        args.max_vis = float("inf")
    else:
        args.max_vis = int(args.max_vis)

    test_tiles_by_region: dict = {}
    if args.test_split_manifest is not None:
        with open(args.test_split_manifest) as f:
            manifest = json.load(f)
        test_tiles_by_region = manifest.get("test_tiles", {})
        counts = {k: len(v) for k, v in test_tiles_by_region.items()}
        print(f"Loaded test-split manifest: {args.test_split_manifest}  (test tile counts: {counts})")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    predictor = load_model(
        args.checkpoint, args.device, args.lora_rank, args.lora_alpha, args.model_size
    )
    print("Model loaded.\n")

    all_ious = []
    for region in args.regions:
        if args.png_cache_dir is not None:
            region_img_dir = os.path.join(args.png_cache_dir, "images", region)
            if not os.path.isdir(region_img_dir):
                print(f"Skipping {region}: not found in png_cache_dir")
                continue
        elif not os.path.isdir(os.path.join(args.tiles_root, region)):
            print(f"Skipping {region}: not found")
            continue
        allowed_tiles = test_tiles_by_region.get(region)
        allowed_tiles = set(allowed_tiles) if allowed_tiles is not None else None
        res = evaluate_region(
            predictor, region, args.tiles_root, args.masks_root,
            device=args.device,
            vis_dir=args.vis_dir,
            max_vis=args.max_vis,
            vis_padding=args.vis_padding,
            png_cache_dir=args.png_cache_dir,
            allowed_tiles=allowed_tiles,
        )
        all_ious.extend(res["iou_vals"])
        if region == "santa_madalena":
            tag = "(OOD)"
        elif allowed_tiles is not None:
            tag = "(ID-test)"
        else:
            tag = "(train)"
        print(
            f"[{region}] {tag}  n={res['n']}  "
            f"mIoU={res['mean_iou']:.4f}  "
            f"mDice={res['mean_dice']:.4f}  "
            f"Recall={res['mean_recall']:.4f}  "
            f"skipped={res['skipped']}"
        )

    print(f"\nOverall  n={len(all_ious)}  mIoU={float(np.mean(all_ious)):.4f}")
    print("Baseline: zero-shot SAM2.1-L = 0.730 mIoU")
    if args.vis_dir:
        print(f"Visualisations saved to: {args.vis_dir}")


if __name__ == "__main__":
    main()
