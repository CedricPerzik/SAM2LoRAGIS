"""Convert favela tile .tif and .npz mask files to PNG format.

Outputs per region:
    <out_dir>/images/<region>/<tile_name>.png          8-bit RGB tile
    <out_dir>/masks/<region>/<tile_name>/<i:04d>.png   binary mask (0/255), one per instance
    <out_dir>/masks/<region>/<tile_name>_meta.json     per-tile instance metadata

The metadata JSON is a list indexed by the original NPZ instance index:
    [{"is_artifact": bool, "is_cut": bool}, ...]
All instances (including artifacts) are recorded so FavelaDataset can reproduce
the same filtering logic (filter_artifacts, filter_cut) without ever opening an NPZ.

Usage:
    pyenv activate lorabora
    python scripts/prepare_favela_data.py \\
        --regions ceu_paz cantidio_sampaio santa_madalena \\
        --verify_tile ceu_paz_row_8_col_7

    # Compute dataset-specific normalisation stats after conversion:
    python scripts/prepare_favela_data.py \\
        --compute_stats --stats_regions ceu_paz cantidio_sampaio
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_region(
    region: str,
    tiles_root: str,
    masks_root: str,
    out_dir: str,
) -> dict:
    """Convert one region's tiles and masks. Returns conversion statistics."""
    t_dir = os.path.join(tiles_root, region)
    m_dir = os.path.join(masks_root, region)
    img_out = os.path.join(out_dir, "images", region)
    msk_out = os.path.join(out_dir, "masks", region)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)

    stats = {"tiles": 0, "instances": 0, "artifacts": 0, "errors": 0}

    tile_files = sorted(
        f for f in os.listdir(t_dir) if f.endswith((".tif", ".png"))
    )
    for tile_file in tqdm(tile_files, desc=f"[{region}] Converting", unit="tile"):
        tile_name = os.path.splitext(tile_file)[0]
        tile_src = os.path.join(t_dir, tile_file)
        tile_dst = os.path.join(img_out, tile_name + ".png")

        # Tile TIF → PNG (8-bit RGB, no geospatial metadata)
        try:
            with PILImage.open(tile_src) as im:
                img_rgb = im.convert("RGB")
            img_rgb.save(tile_dst, format="PNG")
            stats["tiles"] += 1
        except Exception as exc:
            print(f"  ERROR converting tile {tile_file}: {exc}", file=sys.stderr)
            stats["errors"] += 1
            continue

        # Masks NPZ → per-instance binary PNGs + metadata JSON
        mask_path = os.path.join(m_dir, tile_name + "_gt.npz")
        if not os.path.exists(mask_path):
            continue

        try:
            masks_arr = np.load(mask_path, allow_pickle=True)["masks"]
        except Exception as exc:
            print(f"  ERROR loading masks {mask_path}: {exc}", file=sys.stderr)
            stats["errors"] += 1
            continue

        tile_mask_dir = os.path.join(msk_out, tile_name)
        os.makedirs(tile_mask_dir, exist_ok=True)

        metadata = []
        for i, mask_dict in enumerate(masks_arr):
            is_artifact = bool(mask_dict.get("is_artifact", False))
            is_cut = bool(mask_dict.get("is_cut", False))
            metadata.append({"is_artifact": is_artifact, "is_cut": is_cut})

            if is_artifact:
                stats["artifacts"] += 1
                # Still record in metadata but skip PNG (artifact masks have no valid segmentation)
                continue

            seg = mask_dict["segmentation"].astype(np.uint8) * 255  # 0 / 255
            PILImage.fromarray(seg, mode="L").save(
                os.path.join(tile_mask_dir, f"{i:04d}.png"), format="PNG"
            )
            stats["instances"] += 1

        # Write metadata JSON for this tile (covers ALL instances including artifacts)
        meta_path = os.path.join(msk_out, tile_name + "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f)

    return stats


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_stats(regions: list, out_dir: str) -> dict:
    """Compute per-channel mean and std over all PNG tiles in the given regions.

    Statistics are computed in float64 in two passes (mean then std) to avoid
    numeric overflow. Values are in the [0, 1] range (after dividing uint8 by 255)
    to match the ToTensorAPI convention used in training.
    """
    print("\nComputing dataset normalisation statistics...")

    # Pass 1: accumulate per-channel sums for mean
    n_pixels = 0
    channel_sum = np.zeros(3, dtype=np.float64)

    for region in regions:
        img_dir = os.path.join(out_dir, "images", region)
        if not os.path.isdir(img_dir):
            print(f"  WARNING: {img_dir} not found, skipping", file=sys.stderr)
            continue
        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        for fname in tqdm(files, desc=f"[{region}] mean pass", unit="tile"):
            with PILImage.open(os.path.join(img_dir, fname)) as im:
                arr = np.array(im.convert("RGB"), dtype=np.float64) / 255.0
            channel_sum += arr.reshape(-1, 3).sum(axis=0)
            n_pixels += arr.shape[0] * arr.shape[1]

    mean = channel_sum / n_pixels

    # Pass 2: accumulate squared deviations for std
    channel_sq_sum = np.zeros(3, dtype=np.float64)

    for region in regions:
        img_dir = os.path.join(out_dir, "images", region)
        if not os.path.isdir(img_dir):
            continue
        files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
        for fname in tqdm(files, desc=f"[{region}] std  pass", unit="tile"):
            with PILImage.open(os.path.join(img_dir, fname)) as im:
                arr = np.array(im.convert("RGB"), dtype=np.float64) / 255.0
            diff = arr.reshape(-1, 3) - mean
            channel_sq_sum += (diff ** 2).sum(axis=0)

    std = np.sqrt(channel_sq_sum / n_pixels)

    result = {
        "mean": mean.round(4).tolist(),
        "std": std.round(4).tolist(),
        "n_tiles": n_pixels // (1024 * 1024),
        "n_pixels": int(n_pixels),
        "regions": regions,
    }

    stats_path = os.path.join(out_dir, "normalisation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nNormalisation stats ({', '.join(regions)}):")
    print(f"  mean = {result['mean']}")
    print(f"  std  = {result['std']}")
    print(f"  Saved to {stats_path}")
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_sample(tile_name: str, out_dir: str, region: str = None) -> None:
    if region is None:
        region = tile_name.rsplit("_row_", 1)[0]

    img_path = os.path.join(out_dir, "images", region, tile_name + ".png")
    mask_dir = os.path.join(out_dir, "masks", region, tile_name)
    meta_path = os.path.join(out_dir, "masks", region, tile_name + "_meta.json")

    print(f"\n--- Verification: {tile_name} ---")
    if not os.path.exists(img_path):
        print(f"  ERROR: image not found at {img_path}")
        return
    with PILImage.open(img_path) as im:
        arr = np.array(im)
    print(f"  Image : shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        n_art = sum(1 for m in meta if m["is_artifact"])
        n_cut = sum(1 for m in meta if m["is_cut"])
        print(f"  Meta  : {len(meta)} instances, {n_art} artifacts, {n_cut} cut")
    else:
        print(f"  Meta  : not found at {meta_path}")

    if not os.path.isdir(mask_dir):
        print(f"  Masks : directory not found at {mask_dir}")
        return
    mask_files = sorted(os.listdir(mask_dir))
    print(f"  Masks : {len(mask_files)} PNG files")
    for mf in mask_files[:3]:
        with PILImage.open(os.path.join(mask_dir, mf)) as m:
            ma = np.array(m)
        print(f"    {mf}: shape={ma.shape}, dtype={ma.dtype}, unique={np.unique(ma)}")
    if len(mask_files) > 3:
        print(f"    ... and {len(mask_files) - 3} more")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert favela data to PNG")
    parser.add_argument(
        "--tiles_root",
        default="/home/ced/Documents/master_thesis/workingdir/code/output_data/tiles",
    )
    parser.add_argument(
        "--masks_root",
        default="/home/ced/Documents/master_thesis/workingdir/code/output_data/ground_truth_npz",
    )
    parser.add_argument(
        "--out_dir",
        default="/home/ced/Documents/master_thesis/workingdir/code/output_data/favela_png",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["ceu_paz", "cantidio_sampaio", "santa_madalena"],
    )
    parser.add_argument(
        "--verify_tile",
        default="ceu_paz_row_8_col_7",
        help="Tile name to spot-check after conversion (set to empty string to skip)",
    )
    parser.add_argument(
        "--compute_stats",
        action="store_true",
        help="Compute per-channel mean/std over the converted PNG tiles",
    )
    parser.add_argument(
        "--stats_regions",
        nargs="+",
        default=["ceu_paz", "cantidio_sampaio"],
        help="Regions to use for stats (training regions only, no OOD test set)",
    )
    parser.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip conversion and only compute stats over already-converted PNGs",
    )
    args = parser.parse_args()

    if not args.skip_conversion:
        total_stats = {"tiles": 0, "instances": 0, "artifacts": 0, "errors": 0}
        for region in args.regions:
            t_dir = os.path.join(args.tiles_root, region)
            if not os.path.isdir(t_dir):
                print(f"Skipping {region}: directory not found at {t_dir}")
                continue
            stats = convert_region(
                region=region,
                tiles_root=args.tiles_root,
                masks_root=args.masks_root,
                out_dir=args.out_dir,
            )
            for k in total_stats:
                total_stats[k] += stats[k]
            print(
                f"[{region}] tiles={stats['tiles']}, instances={stats['instances']}, "
                f"artifacts_skipped={stats['artifacts']}, errors={stats['errors']}"
            )

        print(
            f"\nTotal: tiles={total_stats['tiles']}, instances={total_stats['instances']}, "
            f"artifacts_skipped={total_stats['artifacts']}, errors={total_stats['errors']}"
        )

        if args.verify_tile:
            verify_sample(args.verify_tile, args.out_dir)

    if args.compute_stats:
        compute_stats(args.stats_regions, args.out_dir)


if __name__ == "__main__":
    main()
