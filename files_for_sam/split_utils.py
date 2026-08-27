"""Tile-level train/val/test split helpers, shared by FavelaDataset and
train.py's held-out test-tile manifest dump.

Kept dependency-light (stdlib only).
"""

import json
import os
import random
from typing import Dict, List


def compute_split(
    tile_paths: List[str],
    train_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> Dict[str, List[str]]:
    """Partition tile_paths into train/val/test using a deterministic shuffle.

    val gets whatever is left over: 1 - train_fraction - test_fraction.
    test_fraction=0.0 (the default everywhere) collapses this back to the
    original 2-way train/val split with no special-casing.
    """
    shuffled = sorted(tile_paths)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_fraction)
    n_test = int(n * test_fraction)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n - n_test],
        "test": shuffled[n - n_test :],
    }


def list_annotated_tiles_png(
    regions: List[str],
    png_cache_dir: str,
    filter_artifacts: bool = True,
    filter_cut: bool = False,
) -> List[str]:
    """Return sorted absolute paths of every tile's PNG image that has at
    least one surviving instance, across the given regions (PNG cache mode
    only). Mirrors FavelaDataset._build_index_png's tile-level inclusion
    criterion exactly (same filter_artifacts/filter_cut defaults as the
    training config) -- this must produce the same tile pool that
    FavelaDataset's train/val/test split is computed over, since a
    differently-sized input list shuffles to a different permutation even
    with the same split_seed.
    """
    tile_paths = []
    for region in regions:
        img_dir = os.path.join(png_cache_dir, "images", region)
        msk_dir = os.path.join(png_cache_dir, "masks", region)
        if not os.path.isdir(img_dir) or not os.path.isdir(msk_dir):
            continue
        for tile_file in sorted(f for f in os.listdir(img_dir) if f.endswith(".png")):
            tile_name = os.path.splitext(tile_file)[0]
            meta_path = os.path.join(msk_dir, tile_name + "_meta.json")
            mask_tile_dir = os.path.join(msk_dir, tile_name)
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except Exception:
                continue

            has_surviving_instance = False
            for i, inst_meta in enumerate(metadata):
                if filter_artifacts and inst_meta.get("is_artifact", False):
                    continue
                if filter_cut and inst_meta.get("is_cut", False):
                    continue
                mask_path = os.path.join(mask_tile_dir, f"{i:04d}.png")
                if os.path.exists(mask_path):
                    has_surviving_instance = True
                    break

            if has_surviving_instance:
                tile_paths.append(os.path.join(img_dir, tile_file))
    return sorted(tile_paths)


def dump_test_manifest(
    regions: List[str],
    png_cache_dir: str,
    train_fraction: float,
    test_fraction: float,
    split_seed: int,
    out_dir: str,
) -> str:
    """Compute the train/val/test split and write the held-out test tile
    names (per region) to <out_dir>/test_tiles.json.

    Writes via a temp file + os.replace so concurrent callers (e.g. multiple
    SLURM jobs from the same sweep run, sharing the same split_seed and thus
    producing byte-identical output) never observe a partially-written file.
    No-op (returns "") if test_fraction <= 0 or png_cache_dir is unset (TIF+NPZ
    mode isn't supported here -- see list_annotated_tiles_png), since this is a
    side effect that must never be able to break the actual training run.
    """
    if test_fraction <= 0 or not png_cache_dir:
        return ""

    tile_paths = list_annotated_tiles_png(regions, png_cache_dir)
    splits = compute_split(tile_paths, train_fraction, test_fraction, split_seed)

    test_tiles = {region: [] for region in regions}
    for tile_path in splits["test"]:
        region = os.path.basename(os.path.dirname(tile_path))
        tile_name = os.path.splitext(os.path.basename(tile_path))[0]
        test_tiles.setdefault(region, []).append(tile_name)
    for region in test_tiles:
        test_tiles[region].sort()

    manifest = {
        "regions": regions,
        "png_cache_dir": png_cache_dir,
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "split_seed": split_seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "test_tiles": test_tiles,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_tiles.json")
    tmp_path = os.path.join(out_dir, f".test_tiles.json.tmp.{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, out_path)
    return out_path
