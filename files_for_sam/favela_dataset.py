"""Dataset for SAM2 LoRA fine-tuning on favela rooftop segmentation.

Each sample is a single 1024x1024 drone-image tile paired with one binary
instance mask (one rooftop polygon).  We follow SAM2's single-frame video
convention so the existing collate_fn and SAM2Train machinery work unchanged.

Two loading modes are supported:

  TIF+NPZ mode (default, png_cache_dir=None):
    <tiles_root>/<region>/  ->  *.tif
    <masks_root>/<region>/  ->  *_gt.npz

  PNG cache mode (png_cache_dir set):
    <png_cache_dir>/images/<region>/  ->  *.png  (8-bit RGB)
    <png_cache_dir>/masks/<region>/<tile_name>/<i:04d>.png  (binary 0/255)
    <png_cache_dir>/masks/<region>/<tile_name>_meta.json    (per-instance metadata)

  Generate the PNG cache with scripts/prepare_favela_data.py.
  PNG mode eliminates the NPZ-reload-per-sample bottleneck and avoids any
  silent colour-space issues from geospatial TIFF metadata.
"""

import json
import logging
import os
import random
from typing import List, Optional

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.split_utils import compute_split
from training.utils.data_utils import Frame, Object, VideoDatapoint

logger = logging.getLogger(__name__)

MAX_RETRIES = 20


class FavelaDataset(VisionDataset):
    """
    Static-image dataset returning single-frame VideoDatapoint objects
    compatible with SAM2Train.

    Each element is a VideoDatapoint with exactly 1 frame and 1 object
    (one rooftop polygon instance).

    Args:
        tiles_root:      Root dir with per-region subdirs containing .tif tiles.
                         Unused when png_cache_dir is set.
        masks_root:      Root dir with per-region subdirs containing .npz masks.
                         Unused when png_cache_dir is set.
        regions:         List of region names to include.
        transforms:      SAM2-compatible transform callables (applied in order).
        training:        Enables retry-on-load-failure for training splits.
        filter_artifacts: Skip instances flagged is_artifact=True (default True).
        filter_cut:      Skip instances clipped at tile edges (default False).
        multiplier:      Per-sample repeat factor for RepeatFactorWrapper (default 1).
        png_cache_dir:   If set, load from pre-converted PNG cache instead of
                         TIF+NPZ. See scripts/prepare_favela_data.py.
    """

    def __init__(
        self,
        tiles_root: str,
        masks_root: str,
        regions: List[str],
        transforms,
        training: bool,
        filter_artifacts: bool = True,
        filter_cut: bool = False,
        multiplier: int = 1,
        png_cache_dir: Optional[str] = None,
        split: Optional[str] = None,
        train_fraction: float = 0.75,
        test_fraction: float = 0.0,
        split_seed: int = 42,
        max_tiles: int = -1,
    ):
        """
        split: one of "train", "val", "test", or None (use all instances).
        train_fraction: fraction of tiles assigned to the train split (default 0.75).
        test_fraction: fraction of tiles held out as a test split (default 0.0, i.e. no
                       test split -- val gets the remaining 1 - train_fraction, matching
                       the original 2-way behaviour). Set > 0 to carve out a third,
                       in-distribution test partition.
        split_seed: random seed for the tile-level shuffle (default 42).
        max_tiles: if > 0 and split="train", limit training to the first N tiles from
                   the shuffled training pool (deterministic via split_seed). Used for
                   data scarcity experiments. -1 means no limit (use all tiles).
        The split is computed at tile level (not instance level) to prevent data
        leakage across train/val/test. All splits must use the same train_fraction,
        test_fraction, and split_seed to be complementary.
        """
        super().__init__(root=tiles_root)
        self._transforms = transforms
        self.training = training
        self.curr_epoch = 0
        self.png_cache_dir = png_cache_dir

        if png_cache_dir is not None:
            self.samples = self._build_index_png(
                regions, png_cache_dir, filter_artifacts, filter_cut
            )
        else:
            self.samples = self._build_index_npz(
                regions, tiles_root, masks_root, filter_artifacts, filter_cut
            )

        if split in ("train", "val", "test"):
            tile_paths = sorted(set(s[0] for s in self.samples))
            splits = compute_split(tile_paths, train_fraction, test_fraction, split_seed)
            full_samples = self.samples
            keep_set = set(splits[split])
            n_train = len(splits["train"])
            before_count = len(self.samples)
            self.samples = [s for s in self.samples if s[0] in keep_set]
            logger.info(
                "FavelaDataset [%s split]: %d/%d instances from %d/%d tiles",
                split, len(self.samples), before_count, len(keep_set), len(tile_paths),
            )

            if split == "train" and 0 < max_tiles < n_train:
                limited_tiles = set(splits["train"][:max_tiles])
                before_count = len(self.samples)
                self.samples = [s for s in self.samples if s[0] in limited_tiles]
                logger.info(
                    "FavelaDataset [max_tiles=%d]: %d/%d instances from %d/%d training tiles",
                    max_tiles, len(self.samples), before_count, max_tiles, n_train,
                )

            if split == "train":
                n_val = len(splits["val"])
                n_test = len(splits["test"])
                wanted_train = max_tiles if max_tiles > 0 else n_train
                true_train = min(max_tiles, n_train) if max_tiles > 0 else n_train
                flag = " CAPPED (pool too small)" if true_train < wanted_train else ""
                print(
                    f"[dataset split] pool={len(tile_paths)} tiles "
                    f"(train_fraction={train_fraction}, test_fraction={test_fraction}, seed={split_seed})"
                )
                print(
                    f"[dataset split] wanted -> true tiles: "
                    f"train {wanted_train} -> {true_train}{flag} | "
                    f"val {n_val} -> {n_val} | "
                    f"test {n_test} -> {n_test}"
                )
                val_tile_set = set(splits["val"])
                test_tile_set = set(splits["test"])
                val_inst = sum(1 for s in full_samples if s[0] in val_tile_set)
                test_inst = sum(1 for s in full_samples if s[0] in test_tile_set)
                print(
                    f"[dataset split] instances -- "
                    f"train {len(self.samples)} (dataloader length) | "
                    f"val {val_inst} | test {test_inst}"
                )

        self.repeat_factors = torch.ones(len(self.samples), dtype=torch.float32) * multiplier

        mode = "PNG" if png_cache_dir else "TIF+NPZ"
        logger.info(
            "FavelaDataset [%s]: %d instances across regions %s",
            mode, len(self.samples), regions,
        )
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid instances found. Check paths, regions={regions}, split={split}."
            )

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------

    def _build_index_npz(self, regions, tiles_root, masks_root, filter_artifacts, filter_cut):
        """Build sample index from TIF tiles + NPZ mask files."""
        samples = []
        for region in regions:
            t_dir = os.path.join(tiles_root, region)
            m_dir = os.path.join(masks_root, region)
            if not os.path.isdir(t_dir):
                logger.warning("Tiles directory not found: %s", t_dir)
                continue
            if not os.path.isdir(m_dir):
                logger.warning("Masks directory not found: %s", m_dir)
                continue

            tile_files = sorted(
                f for f in os.listdir(t_dir) if f.endswith((".tif", ".png"))
            )
            for tile_file in tile_files:
                tile_name = os.path.splitext(tile_file)[0]
                mask_path = os.path.join(m_dir, tile_name + "_gt.npz")
                if not os.path.exists(mask_path):
                    continue

                tile_path = os.path.join(t_dir, tile_file)
                try:
                    masks_arr = np.load(mask_path, allow_pickle=True)["masks"]
                except Exception as e:
                    logger.warning("Failed to load %s: %s", mask_path, e)
                    continue

                for i, mask_dict in enumerate(masks_arr):
                    if filter_artifacts and mask_dict.get("is_artifact", False):
                        continue
                    if filter_cut and mask_dict.get("is_cut", False):
                        continue
                    # (tile_path, mask_path, instance_idx)
                    samples.append((tile_path, mask_path, i))
        return samples

    def _build_index_png(self, regions, png_cache_dir, filter_artifacts, filter_cut):
        """Build sample index from pre-converted PNG cache."""
        samples = []
        for region in regions:
            img_dir = os.path.join(png_cache_dir, "images", region)
            msk_dir = os.path.join(png_cache_dir, "masks", region)
            if not os.path.isdir(img_dir):
                logger.warning("PNG images directory not found: %s", img_dir)
                continue
            if not os.path.isdir(msk_dir):
                logger.warning("PNG masks directory not found: %s", msk_dir)
                continue

            tile_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
            for tile_file in tile_files:
                tile_name = os.path.splitext(tile_file)[0]
                tile_path = os.path.join(img_dir, tile_file)
                meta_path = os.path.join(msk_dir, tile_name + "_meta.json")
                mask_tile_dir = os.path.join(msk_dir, tile_name)

                if not os.path.exists(meta_path):
                    # Tile exists in images/ but has no annotations - silently skip.
                    logger.debug("No metadata JSON for unannotated tile: %s", meta_path)
                    continue

                try:
                    with open(meta_path) as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", meta_path, e)
                    continue

                for i, inst_meta in enumerate(metadata):
                    if filter_artifacts and inst_meta.get("is_artifact", False):
                        continue
                    if filter_cut and inst_meta.get("is_cut", False):
                        continue
                    mask_path = os.path.join(mask_tile_dir, f"{i:04d}.png")
                    if not os.path.exists(mask_path):
                        # Artifact masks are not written as PNGs - skip silently.
                        continue
                    # (tile_png_path, mask_png_path, None) - None signals PNG mode
                    samples.append((tile_path, mask_path, None))
        return samples

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> VideoDatapoint:
        for attempt in range(MAX_RETRIES):
            try:
                return self._load(idx)
            except Exception as exc:
                if not self.training:
                    raise
                logger.warning(
                    "Load failed idx=%d (attempt %d/%d): %s",
                    idx, attempt + 1, MAX_RETRIES, exc,
                )
                idx = random.randrange(len(self.samples))
        return self._load(idx)

    def _load(self, idx: int) -> VideoDatapoint:
        tile_path, mask_path, inst_idx = self.samples[idx]

        # Image
        with PILImage.open(tile_path) as im:
            image = im.convert("RGB")
        w, h = image.size

        # Mask
        if inst_idx is None:
            # PNG mode: mask is a single binary PNG (0 / 255)
            with PILImage.open(mask_path) as m:
                seg_np = np.array(m.convert("L"))
            segment = torch.from_numpy((seg_np > 0).astype(np.uint8))
        else:
            # TIF+NPZ mode
            masks_arr = np.load(mask_path, allow_pickle=True)["masks"]
            seg_np = masks_arr[inst_idx]["segmentation"]  # bool [H, W]
            segment = torch.from_numpy(seg_np).to(torch.uint8)

        frame = Frame(
            data=image,
            objects=[Object(object_id=1, frame_index=0, segment=segment)],
        )
        datapoint = VideoDatapoint(
            frames=[frame],
            video_id=idx,
            size=(h, w),
        )

        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)

        return datapoint
