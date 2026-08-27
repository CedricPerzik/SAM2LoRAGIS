# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from tqdm import tqdm

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer

from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import barrier, get_rank

from training.utils.logger import Logger, setup_logging

from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"

# Keys in epoch output dicts that are metadata, not model-quality metrics.
# They go to the JSON logs and best_stats.json but are excluded from TensorBoard.
_TB_SKIP_PREFIXES = ("Trainer/",)

# Maps the trainer's internal accumulator key names to clean TensorBoard tag names.
# Train and val share the same first-level prefix so TensorBoard overlays them in
# one panel per component, making the train/val gap immediately visible.
_LOSS_KEY_MAP = {
    "Losses/train_all_loss":         "Loss/total/train",
    "Losses/train_all_loss_bce":     "Loss/bce/train",
    "Losses/train_all_loss_dice":    "Loss/dice/train",
    "Losses/train_all_loss_tversky": "Loss/tversky/train",
    "Losses/train_all_loss_iou":     "Loss/iou/train",
    "Losses/train_all_loss_class":   "Loss/class/train",
    "Losses/val_all_loss":           "Loss/total/val",
    "Losses/val_all_loss_bce":       "Loss/bce/val",
    "Losses/val_all_loss_dice":      "Loss/dice/val",
    "Losses/val_all_loss_tversky":   "Loss/tversky/val",
    "Losses/val_all_loss_iou":       "Loss/iou/val",
    "Losses/val_all_loss_class":     "Loss/class/val",
    "Metrics/miou":       "Metrics/val_miou",
    "Metrics/precision":  "Metrics/val_precision",
    "Metrics/recall":     "Metrics/val_recall",
    "Metrics/accuracy":   "Metrics/val_accuracy",
    "Metrics/dice":       "Metrics/val_dice",
    "Metrics/ap50":       "Metrics/val_ap50",
    "Metrics/ap75":       "Metrics/val_ap75",
    "Metrics/map":        "Metrics/val_map",
}


def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    save_start_epoch: int = 0  # skip numbered checkpoint saves before this epoch
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        early_stop_patience: int = 0,
        early_stop_start_epoch: int = 20,
        early_stop_min_delta: float = 0.0,
        early_stop_metric: str = "Loss/total/val",
        log_mem: bool = False,
    ):

        self.log_mem = log_mem
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0
        self.early_stop_patience = early_stop_patience
        self.early_stop_start_epoch = early_stop_start_epoch
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_metric = early_stop_metric
        self._es_maximize = "Loss" not in early_stop_metric  # dice/IoU go up; loss goes down
        self._es_no_improve = 0
        self._best_val_loss = float("inf")
        self._best_val_dice = 0.0
        self._global_best_es = float("-inf") if self._es_maximize else float("inf")
        self._es_best_metric = float("-inf") if self._es_maximize else float("inf")
        self._stopped_early = False

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):

        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            epoch_n = int(epoch)
            if epoch_n >= self.checkpoint_conf.save_start_epoch:
                if (
                    self.checkpoint_conf.save_freq > 0
                    and (epoch_n % self.checkpoint_conf.save_freq == 0)
                ) or epoch_n in self.checkpoint_conf.save_list:
                    checkpoint_names.append(f"checkpoint_{epoch_n}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
            "best_val_loss": self._best_val_loss,
            "best_val_dice": self._best_val_dice,
            "global_best_es": self._global_best_es,
            "es_best_metric": self._es_best_metric,
            "es_no_improve": self._es_no_improve,
            "early_stop_min_delta": self.early_stop_min_delta,
            "early_stop_metric": self.early_stop_metric,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success
        tqdm.write(f"  [ckpt] Saved -> {checkpoint_path}")

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.best_meter_values = checkpoint.get("best_meter_values", {})
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self._best_val_dice = checkpoint.get("best_val_dice", 0.0)
        default_es_best = float("-inf") if self._es_maximize else float("inf")
        self._global_best_es = checkpoint.get("global_best_es", default_es_best)
        # backward compat: old checkpoints used "es_best_val_loss" (minimize only)
        self._es_best_metric = checkpoint.get("es_best_metric",
            checkpoint.get("es_best_val_loss", default_es_best))
        self._es_no_improve = checkpoint.get("es_no_improve", 0)
        self.early_stop_min_delta = checkpoint.get("early_stop_min_delta", self.early_stop_min_delta)
        self.early_stop_metric = checkpoint.get("early_stop_metric", self.early_stop_metric)

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def _check_early_stop(self, val_outs: dict) -> bool:
        """Track best metrics, save _best checkpoint when early_stop_metric hits new global optimum, return True to stop.

        Called after self.epoch has been incremented, so self.epoch is the
        1-based count of completed epochs (matching checkpoint_N filenames).

        Four separate trackers:
          _best_val_loss   - global val_loss minimum; logged only.
          _best_val_dice   - global dice maximum; logged only.
          _global_best_es  - global optimum of early_stop_metric (strict, no min_delta);
                             drives _best checkpoint saving.
          _es_best_metric  - ES-local optimum; only starts updating from
                             early_stop_start_epoch so the patience counter is never
                             pre-filled by warm-up history.
        """
        if not val_outs:
            return False

        val_loss = val_outs.get("Loss/total/val")
        dice = val_outs.get("Metrics/val_dice")
        es_val = val_outs.get(self.early_stop_metric)

        # --- always track val_loss and val_dice globally for logging ---
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self.logger.log("Metrics/best_val_loss", val_loss, self.epoch)

        if dice is not None and float(dice) > self._best_val_dice:
            self._best_val_dice = float(dice)
            self.logger.log("Metrics/best_val_dice", self._best_val_dice, self.epoch)

        # --- _best checkpoint: driven by early_stop_metric, strict improvement, no min_delta ---
        if es_val is not None:
            es_val_f = float(es_val)
            new_global_best = (
                es_val_f > self._global_best_es
                if self._es_maximize
                else es_val_f < self._global_best_es
            )
            if new_global_best:
                self._global_best_es = es_val_f
                if self.epoch >= self.checkpoint_conf.save_start_epoch:
                    self.save_checkpoint(
                        self.epoch,
                        checkpoint_names=[f"checkpoint_{self.epoch}_best"],
                    )
                logging.info(
                    f"New best {self.early_stop_metric} {es_val_f:.6g} at epoch {self.epoch} "
                    f"-> checkpoint_{self.epoch}_best.pt"
                )

        # --- early stopping: configurable metric, fresh tracking from early_stop_start_epoch ---
        if es_val is None:
            return False

        if self.epoch >= self.early_stop_start_epoch:
            if self._es_maximize:
                improved = float(es_val) > self._es_best_metric + self.early_stop_min_delta
            else:
                improved = float(es_val) < self._es_best_metric - self.early_stop_min_delta
            if improved:
                self._es_best_metric = float(es_val)
                self._es_no_improve = 0
            else:
                self._es_no_improve += 1

        self.logger.log("Metrics/es_counter", self._es_no_improve, self.epoch)

        if self.early_stop_patience <= 0:
            return False
        if self.epoch < self.early_stop_start_epoch:
            return False
        return self._es_no_improve >= self.early_stop_patience

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):

        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"

        # Accumulate sub-components for epoch-level averages.
        # Pop core_loss first so it is not duplicated as a separate series.
        step_losses = {}
        if isinstance(loss, dict):
            core_loss = loss.pop(CORE_LOSS_KEY)
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = core_loss

        # Segmentation metrics: computed only during validation to avoid training overhead.
        if phase == Phase.VAL:
            step_losses.update(self._compute_batch_seg_metrics(outputs, targets))

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple

    def _compute_batch_seg_metrics(
        self,
        outputs: list,
        targets: torch.Tensor,
    ) -> dict:
        """Compute segmentation metrics for the batch (validation only).

        Selects the best mask candidate per sample (by predicted IoU score),
        then computes pixel-level and instance-level metrics.

        Pixel-level  : precision, recall, accuracy, dice, (implicit in mIoU)
        Instance-level: mIoU, AP@0.50, AP@0.75, mAP (COCO-style 0.50:0.05:0.95)

        Returns a dict of scalar tensors keyed by 'Metrics/<name>'.
        """
        device = targets.device
        all_ious = []
        tp = torch.zeros(1, device=device)
        fp = torch.zeros(1, device=device)
        fn = torch.zeros(1, device=device)
        tn = torch.zeros(1, device=device)

        for frame_outs, frame_targets in zip(outputs, targets):
            masks_list = frame_outs.get("multistep_pred_multimasks_high_res")
            ious_list  = frame_outs.get("multistep_pred_ious")
            if not masks_list or not ious_list:
                continue

            src_masks = masks_list[-1]                                        # [N, M, H, W]
            pred_ious = ious_list[-1]                                         # [N, M]

            best_inds  = pred_ious.argmax(dim=1)                             # [N]
            batch_inds = torch.arange(src_masks.size(0), device=device)
            best_masks = src_masks[batch_inds, best_inds]                    # [N, H, W]

            pred = (best_masks > 0).float().flatten(1)                       # [N, H*W]
            gt   = (frame_targets > 0).float().flatten(1)                    # [N, H*W]

            # Pixel counts
            tp += (pred * gt).sum()
            fp += (pred * (1 - gt)).sum()
            fn += ((1 - pred) * gt).sum()
            tn += ((1 - pred) * (1 - gt)).sum()

            # Per-instance IoU for AP metrics
            inter = (pred * gt).sum(1)
            union = (pred + gt - pred * gt).sum(1)
            all_ious.append(inter / union.clamp(min=1.0))                    # [N]

        eps = 1e-7
        precision = (tp / (tp + fp + eps)).squeeze()
        recall    = (tp / (tp + fn + eps)).squeeze()
        accuracy  = ((tp + tn) / (tp + tn + fp + fn + eps)).squeeze()
        dice      = (2 * tp / (2 * tp + fp + fn + eps)).squeeze()

        if all_ious:
            iou_all = torch.cat(all_ious)                                    # [total_instances]
            miou    = iou_all.mean()
            thresholds = torch.arange(0.50, 1.00, 0.05, device=device)
            aps     = torch.stack([(iou_all >= t).float().mean() for t in thresholds])
            ap50    = (iou_all >= 0.50).float().mean()
            ap75    = (iou_all >= 0.75).float().mean()
            map_val = aps.mean()
        else:
            z = torch.zeros(1, device=device).squeeze()
            miou = ap50 = ap75 = map_val = z

        return {
            "Metrics/precision": precision,
            "Metrics/recall":    recall,
            "Metrics/accuracy":  accuracy,
            "Metrics/dice":      dice,
            "Metrics/miou":      miou,
            "Metrics/ap50":      ap50,
            "Metrics/ap75":      ap75,
            "Metrics/map":       map_val,
        }

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            if not self._stopped_early:
                self.run_val()
            if self.distributed_rank == 0:
                out_path = os.path.join(self.logging_conf.log_dir, "best_val_loss.json")
                with open(out_path, "w") as f:
                    json.dump({"best_val_loss": self._best_val_loss,
                               "best_val_dice": self._best_val_dice}, f)
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None
        self._val_dataloader = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        rank0 = self.distributed_rank == 0
        epoch_bar = tqdm(
            total=self.max_epochs,
            initial=self.epoch,
            desc="Epochs",
            unit="ep",
            disable=not rank0,
            leave=True,
            position=0,
            dynamic_ncols=True,
        )

        dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))

        while self.epoch < self.max_epochs:
            self.train_dataset.update_epoch(int(self.epoch))
            barrier()
            train_outs = self.train_epoch(dataloader)
            self._mem_snapshot(f"E{self.epoch:03d} post-train")
            # Filter Trainer/* metadata keys - keep them in JSON but not TensorBoard.
            tb_outs = {k: v for k, v in train_outs.items()
                       if not any(k.startswith(p) for p in _TB_SKIP_PREFIXES)}
            self.logger.log_dict(tb_outs, self.epoch + 1)  # Logged only on rank 0
            # Log learning rate once per epoch (epoch-aligned, clean single series).
            self.logger.log("Train/lr", self.optim.optimizer.param_groups[0]["lr"], self.epoch + 1)

            # log train to text file.
            if rank0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(train_outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)
            self._mem_snapshot(f"E{self.epoch:03d} post-ckpt")

            gc.collect()
            # Return fragmented glibc heap pages to the OS.
            try:
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            self._mem_snapshot(f"E{self.epoch:03d} post-gc", cuda_summary=True)

            # Run val, not running on last epoch since will run after the
            # loop anyway
            val_outs = {}
            if self.is_intermediate_val_epoch(self.epoch):
                val_outs = self.run_val() or {}
            self._mem_snapshot(f"E{self.epoch:03d} post-val")

            # Combined train+val loss on a single chart for easy comparison.
            if val_outs and rank0:
                t_loss = train_outs.get("Loss/total/train")
                v_loss = val_outs.get("Loss/total/val")
                if t_loss is not None and v_loss is not None:
                    self.logger.log_scalars(
                        "Loss/train_vs_val",
                        {"train": t_loss, "val": v_loss},
                        self.epoch + 1,
                    )

            if rank0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            # Epoch summary line + epoch bar update (rank 0 only).
            if rank0:
                train_loss = train_outs.get("Loss/total/train", float("nan"))
                val_loss = val_outs.get("Loss/total/val", None)
                lr = self.optim.optimizer.param_groups[0]["lr"]
                eta = self._log_timers_eta()

                postfix = {"train": f"{train_loss:.4e}"}
                if val_loss is not None:
                    postfix["val"] = f"{val_loss:.4e}"
                postfix["lr"] = f"{lr:.2e}"
                if eta:
                    postfix["ETA"] = eta
                epoch_bar.set_postfix(postfix, refresh=True)
                epoch_bar.update(1)

                summary = (
                    f"  Epoch {self.epoch + 1:3d}/{self.max_epochs}"
                    f"  train={train_loss:.4e}"
                )
                if val_loss is not None:
                    summary += f"  val={val_loss:.4e}"
                summary += f"  lr={lr:.2e}"
                if eta:
                    summary += f"  ETA {eta}"
                tqdm.write(summary)
            else:
                epoch_bar.update(1)

            self.epoch += 1

            # Early stopping: must come after epoch increment so self.epoch matches
            # the checkpoint_N filenames already saved this iteration.
            should_stop = self._check_early_stop(val_outs)
            if should_stop:
                logging.info(
                    f"Early stopping: {self.early_stop_metric} has not improved for "
                    f"{self.early_stop_patience} consecutive epochs "
                    f"(monitoring from epoch {self.early_stop_start_epoch})."
                )
                if rank0:
                    tqdm.write(
                        f"\n  [early stop] No improvement in {self.early_stop_metric} for "
                        f"{self.early_stop_patience} epochs after best epoch "
                        f"{self.epoch - self.early_stop_patience}. "
                        f"Stopping at epoch {self.epoch}."
                    )
                self._stopped_early = True
                break

        epoch_bar.close()
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return {}

        if self._val_dataloader is None:
            self._val_dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        else:
            self.val_dataset.update_epoch(int(self.epoch))
        outs = self.val_epoch(self._val_dataloader, phase=Phase.VAL)
        gc.collect()
        tb_outs = {k: v for k, v in outs.items()
                   if not any(k.startswith(p) for p in _TB_SKIP_PREFIXES)}
        self.logger.log_dict(tb_outs, self.epoch + 1)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

        return outs

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch + 1),
        )

        end = time.time()

        val_step_bar = tqdm(
            total=iters_per_epoch,
            desc=f"E{self.epoch + 1:03d} val  ",
            unit="step",
            disable=(self.distributed_rank != 0),
            leave=False,
            position=1,
            dynamic_ncols=True,
        )

        for data_iter, batch in enumerate(val_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            batch = batch.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % 10 == 0:
                dist.barrier()

            # Update val step bar.
            avg_loss = list(loss_mts.values())[0].avg if loss_mts else float("nan")
            val_step_bar.set_postfix(
                {"loss": f"{avg_loss:.4e}"},
                refresh=False,
            )
            val_step_bar.update(1)

            if data_iter % 10 == 0:
                gc.collect()

        val_step_bar.close()
        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[_LOSS_KEY_MAP.get(k, k)] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[_LOSS_KEY_MAP.get(k, k)] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")

        if self.distributed_rank == 0:
            val_loss = out_dict.get("Loss/total/val", float("nan"))
            tqdm.write(f"  [val]  Epoch {self.epoch + 1}  loss={val_loss:.4e}")

        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch + 1,  # 1-based: epoch 1 == checkpoint_1.pt
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):

        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch + 1),
        )

        # Model training loop
        self.model.train()
        end = time.time()

        step_bar = tqdm(
            total=iters_per_epoch,
            desc=f"E{self.epoch + 1:03d} train",
            unit="step",
            disable=(self.distributed_rank != 0),
            leave=False,
            position=1,
            dynamic_ncols=True,
        )

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            batch = batch.to(
                self.device, non_blocking=True
            )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                # Update step bar.
                avg_loss = list(loss_mts.values())[0].avg if loss_mts else float("nan")
                step_bar.set_postfix(
                    {"loss": f"{avg_loss:.4e}", "mem": f"{mem_meter.val:.1f}GB"},
                    refresh=False,
                )
                step_bar.update(1)

                if data_iter % 50 == 0:
                    self._mem_snapshot(
                        f"E{self.epoch:03d} step {data_iter:04d}/{iters_per_epoch}"
                    )

                # The @tensorclass BatchedVideoDatapoint collation creates
                # cyclic references that the reference counter cannot free.
                # Without periodic GC these accumulate ~98 MiB/batch and
                # cause OOM within a single epoch.
                if data_iter % 10 == 0:
                    gc.collect()
                # Return fragmented glibc heap pages to the OS every 100 steps.
                if data_iter % 100 == 0 and data_iter > 0:
                    try:
                        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                step_bar.close()
                raise e

        step_bar.close()
        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[_LOSS_KEY_MAP.get(k, k)] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[_LOSS_KEY_MAP.get(k, k)] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda",
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(
                batch,
                self.model,
                phase,
            )

        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        self.scaler.scale(loss).backward()
        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        eta_str = human_readable_time(time_remaining)
        logging.info(f"Estimated time remaining: {eta_str}")
        return eta_str

    def _log_timers_eta(self) -> str:
        """Return ETA string based on current epoch time estimates (no logging)."""
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch + 1, self.max_epochs)
        )
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_remaining += 1
        time_remaining = (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_remaining * self.est_epoch_time[Phase.VAL]
        )
        return human_readable_time(time_remaining)

    def _mem_snapshot(self, label: str, cuda_summary: bool = False) -> None:
        """Print CPU RSS + GPU VRAM stats at a named point (rank 0 only)."""
        if not self.log_mem or self.distributed_rank != 0:
            return
        rss_kb = vmswap_kb = 0
        try:
            with open("/proc/self/status") as _f:
                for _line in _f:
                    if _line.startswith("VmRSS:"):
                        rss_kb = int(_line.split()[1])
                    elif _line.startswith("VmSwap:"):
                        vmswap_kb = int(_line.split()[1])
        except OSError:
            pass
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_str = f"GPU alloc={gpu_alloc:.2f} GB  reserved={gpu_reserved:.2f} GB"
        else:
            gpu_str = "GPU n/a"
        tqdm.write(
            f"  [MEM] {label}: "
            f"CPU RSS={rss_kb / 1e6:.2f} GB  swap={vmswap_kb / 1e6:.2f} GB  {gpu_str}"
        )
        if cuda_summary and torch.cuda.is_available():
            tqdm.write(torch.cuda.memory_summary(abbreviated=True))
        rss_kb = vmswap_kb = 0
        try:
            with open("/proc/self/status") as _f:
                for _line in _f:
                    if _line.startswith("VmRSS:"):
                        rss_kb = int(_line.split()[1])
                    elif _line.startswith("VmSwap:"):
                        vmswap_kb = int(_line.split()[1])
        except OSError:
            pass
        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_str = f"GPU alloc={gpu_alloc:.2f} GB  reserved={gpu_reserved:.2f} GB"
        else:
            gpu_str = "GPU n/a"
        tqdm.write(
            f"  [MEM] {label}: "
            f"CPU RSS={rss_kb / 1e6:.2f} GB  swap={vmswap_kb / 1e6:.2f} GB  {gpu_str}"
        )
        if cuda_summary and torch.cuda.is_available():
            tqdm.write(torch.cuda.memory_summary(abbreviated=True))

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys())
                assert all([k in loss_keys for k in val_keys]), (
                    f"Keys in val datasets do not match the keys in losses."
                    f"\nMissing in losses: {set(val_keys) - loss_keys}"
                    f"\nMissing in val datasets: {loss_keys - set(val_keys)}"
                )

    def _setup_components(self):

        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            self.device,
            enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
        )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")
        if self.distributed_rank == 0:
            self._print_run_banner()

    def _print_run_banner(self):
        """Print a compact startup summary to the terminal (rank 0 only)."""
        m = self.model
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen = total - trainable
        pct = 100.0 * trainable / total if total > 0 else 0.0

        def _fmt(n):
            return f"{n / 1e6:,.1f} M"

        # LoRA-specific info (gracefully skip if not a LoRA model).
        lora_lines = []
        if hasattr(m, "lora_rank"):
            scale = m.lora_alpha / m.lora_rank
            targets = []
            if getattr(m, "lora_target_hiera", False):
                targets.append("Hiera Attn")
            if getattr(m, "lora_target_hiera_mlp", False):
                targets.append("Hiera MLP")
            if getattr(m, "lora_target_mask_decoder", False):
                targets.append("Decoder Attn")
            if getattr(m, "lora_target_mask_decoder_mlp", False):
                targets.append("Decoder MLP")
            lora_lines = [
                f"  LoRA   rank={m.lora_rank}  alpha={m.lora_alpha}  scale={scale:.2f}",
                f"  Targets  {' + '.join(targets) if targets else 'none'}",
            ]

        # Checkpoint path (best-effort from config dict).
        try:
            ckpt_path = (
                self.checkpoint_conf.model_weight_initializer
                ["state_dict"]["checkpoint_path"]
            )
        except Exception:
            ckpt_path = "?"

        # TensorBoard dir sits one level above the logs dir.
        # abspath so the printed command works regardless of cwd.
        tb_dir = os.path.abspath(
            os.path.join(os.path.dirname(self.logging_conf.log_dir), "tensorboard/")
        )

        if self.early_stop_patience > 0:
            es_str = (
                f"patience={self.early_stop_patience}  "
                f"start_epoch={self.early_stop_start_epoch}  "
                f"min_delta={self.early_stop_min_delta}"
            )
        else:
            es_str = "disabled"

        lines = [
            "=" * 59,
            "  SAM2.1 + LoRA  Training",
            "=" * 59,
            f"  Checkpoint   {ckpt_path}",
            "-" * 59,
            f"  Frozen base     {_fmt(frozen):>9s}  ({100-pct:5.2f}%)",
            f"  LoRA trainable  {_fmt(trainable):>9s}  ({pct:5.2f}%)",
            f"  Total           {_fmt(total):>9s}",
            "-" * 59,
            *lora_lines,
            "-" * 59,
            f"  Epochs  {self.max_epochs}    Early stop  {es_str}",
            f"  Ckpt save from epoch {self.checkpoint_conf.save_start_epoch}  "
            f"  freq={self.checkpoint_conf.save_freq}",
            f"  Log dir  {self.logging_conf.log_dir}",
            f"  TensorBoard:  tensorboard --logdir {tb_dir}/",
            "=" * 59,
        ]
        print("\n".join(lines))
        print()

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
