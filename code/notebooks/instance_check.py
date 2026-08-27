"""Standalone simulation: rebuild the 75/25 train/val split (test_fraction=0.0)
for every sweep seed against the real local PNG cache, and count both tiles and
surviving mask instances per split. Read-only -- does not touch the real repo,
just imports the existing split_utils helpers.
"""

import json
import os
import sys

sys.path.insert(0, "/home/ced/Documents/unicluster/sam2loraboracluster/sam2/training/dataset")
from split_utils import compute_split, list_annotated_tiles_png

REGIONS = ["ceu_paz", "cantidio_sampaio"]
PNG_CACHE_DIR = "/home/ced/Documents/unicluster/data/favela_png"
SEEDS = [1, 26, 42, 99, 1234]
TRAIN_FRACTION = 0.75
TEST_FRACTION = 0.0


def count_instances(tile_set, regions, png_cache_dir):
    """Same survival criterion as FavelaDataset._build_index_png: one count per
    surviving (non-artifact) instance that has a written mask PNG."""
    total = 0
    for region in regions:
        img_dir = os.path.join(png_cache_dir, "images", region)
        msk_dir = os.path.join(png_cache_dir, "masks", region)
        for tile_file in sorted(f for f in os.listdir(img_dir) if f.endswith(".png")):
            tile_path = os.path.join(img_dir, tile_file)
            if tile_path not in tile_set:
                continue
            tile_name = os.path.splitext(tile_file)[0]
            meta_path = os.path.join(msk_dir, tile_name + "_meta.json")
            mask_tile_dir = os.path.join(msk_dir, tile_name)
            with open(meta_path) as f:
                metadata = json.load(f)
            for i, inst_meta in enumerate(metadata):
                if inst_meta.get("is_artifact", False):
                    continue
                mask_path = os.path.join(mask_tile_dir, f"{i:04d}.png")
                if os.path.exists(mask_path):
                    total += 1
    return total


def main():
    tile_paths = list_annotated_tiles_png(REGIONS, PNG_CACHE_DIR)
    print(f"Total annotated tile pool: {len(tile_paths)} tiles\n")

    header = (
        f"{'seed':>6} | {'train_tiles':>11} | {'train_inst':>10} | "
        f"{'val_tiles':>9} | {'val_inst':>8} | {'test_tiles':>10} | {'test_inst':>9}"
    )
    print(header)
    print("-" * len(header))

    for seed in SEEDS:
        splits = compute_split(tile_paths, TRAIN_FRACTION, TEST_FRACTION, seed)
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        train_inst = count_instances(train_set, REGIONS, PNG_CACHE_DIR)
        val_inst = count_instances(val_set, REGIONS, PNG_CACHE_DIR)
        test_inst = count_instances(test_set, REGIONS, PNG_CACHE_DIR)

        print(
            f"{seed:>6} | {len(train_set):>11} | {train_inst:>10} | "
            f"{len(val_set):>9} | {val_inst:>8} | {len(test_set):>10} | {test_inst:>9}"
        )


if __name__ == "__main__":
    main()