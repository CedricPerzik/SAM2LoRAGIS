# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import traceback
import warnings
from argparse import ArgumentParser

import submitit
import torch

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.dataset.split_utils import dump_test_manifest
from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"

# Architecture parameters for each supported model size.
# These values are applied by --model-size and patch the YAML trunk/neck/checkpoint.
_MODEL_SIZE_CONFIGS = {
    "t": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": [1, 2, 7, 2],
        "global_att_blocks": [5, 7, 9],
        "window_pos_embed_bkg_spatial_size": [7, 7],
        "window_spec": [8, 4, 14, 7],
        "backbone_channel_list": [768, 384, 192, 96],
        "checkpoint": "./checkpoints/sam2.1_hiera_tiny.pt",
    },
    "s": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": [1, 2, 11, 2],
        "global_att_blocks": [7, 10, 13],
        "window_pos_embed_bkg_spatial_size": [7, 7],
        "window_spec": [8, 4, 14, 7],
        "backbone_channel_list": [768, 384, 192, 96],
        "checkpoint": "./checkpoints/sam2.1_hiera_small.pt",
    },
    "b+": {
        "embed_dim": 112,
        "num_heads": 2,
        "stages": [2, 3, 16, 3],
        "global_att_blocks": [12, 16, 20],
        "window_pos_embed_bkg_spatial_size": [14, 14],
        "window_spec": [8, 4, 14, 7],
        "backbone_channel_list": [896, 448, 224, 112],
        "checkpoint": "./checkpoints/sam2.1_hiera_base_plus.pt",
    },
    "l": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": [2, 6, 36, 4],
        "global_att_blocks": [23, 33, 43],
        "window_pos_embed_bkg_spatial_size": [7, 7],
        "window_spec": [8, 4, 16, 8],
        "backbone_channel_list": [1152, 576, 288, 144],
        "checkpoint": "./checkpoints/sam2.1_hiera_large.pt",
    },
}


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        # Need to add this again so the hydra.job.set_env PYTHONPATH
        # is also set when launching jobs.
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            # Log the exception. Then raise it again (as what SubmititRunner currently does).
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info"""
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }

        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def _apply_args_overrides(cfg, args):
    """Push CLI overrides into the Hydra config before it is saved or instantiated.

    Only non-None values are applied, so any flag that is left out simply falls
    back to whatever the YAML specifies.
    """
    s = cfg.scratch

    # --- training scale ---
    if args.num_epochs is not None:
        s.num_epochs = args.num_epochs
    if args.batch_size is not None:
        s.train_batch_size = args.batch_size
    if args.num_workers is not None:
        s.num_train_workers = args.num_workers

    # --- dataset ---
    if args.max_tiles is not None:
        s.max_tiles = args.max_tiles
    if args.tiles_root is not None:
        s.tiles_root = args.tiles_root
    if args.masks_root is not None:
        s.masks_root = args.masks_root
    if args.png_cache_dir is not None:
        s.png_cache_dir = args.png_cache_dir
    if args.split_seed is not None:
        s.split_seed = args.split_seed

    # --- trainer seed ---
    if args.trainer_seed is not None:
        OmegaConf.update(cfg, "trainer.seed_value", args.trainer_seed)

    # --- optimiser ---
    if args.base_lr is not None:
        s.base_lr = args.base_lr
        s.vision_lr = args.base_lr  # keep in sync; both default to the same value
    if args.weight_decay is not None:
        s.weight_decay = args.weight_decay

    # --- LoRA ---
    if args.lora_rank is not None:
        s.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        s.lora_alpha = float(args.lora_alpha)

    # --- early stopping ---
    if args.early_stop_patience is not None:
        s.early_stop_patience = args.early_stop_patience
    if args.early_stop_start is not None:
        s.early_stop_start_epoch = args.early_stop_start
    if args.early_stop_min_delta is not None:
        s.early_stop_min_delta = args.early_stop_min_delta

    # --- checkpointing ---
    if args.ckpt_save_start is not None:
        s.checkpoint_save_start_epoch = args.ckpt_save_start

    # --- model size (patches trunk + neck + checkpoint path) ---
    if args.model_size is not None:
        mc = _MODEL_SIZE_CONFIGS[args.model_size]
        trunk = "trainer.model.image_encoder.trunk"
        OmegaConf.update(cfg, f"{trunk}.embed_dim", mc["embed_dim"])
        OmegaConf.update(cfg, f"{trunk}.num_heads", mc["num_heads"])
        OmegaConf.update(cfg, f"{trunk}.stages", mc["stages"])
        OmegaConf.update(cfg, f"{trunk}.global_att_blocks", mc["global_att_blocks"])
        OmegaConf.update(
            cfg,
            f"{trunk}.window_pos_embed_bkg_spatial_size",
            mc["window_pos_embed_bkg_spatial_size"],
        )
        OmegaConf.update(cfg, f"{trunk}.window_spec", mc["window_spec"])
        OmegaConf.update(
            cfg,
            "trainer.model.image_encoder.neck.backbone_channel_list",
            mc["backbone_channel_list"],
        )
        OmegaConf.update(
            cfg,
            "trainer.checkpoint.model_weight_initializer.state_dict.checkpoint_path",
            mc["checkpoint"],
        )
        cfg.scratch.model_size = args.model_size

    # --- loss weights ---
    loss = cfg.trainer.loss["all"]
    if args.weight_bce is not None:
        loss.weight_bce = args.weight_bce
    if args.weight_dice is not None:
        loss.weight_dice = args.weight_dice
    if args.weight_tversky is not None:
        loss.weight_tversky = args.weight_tversky
    if args.weight_iou is not None:
        loss.weight_iou = args.weight_iou
    if args.tversky_alpha is not None:
        loss.tversky_alpha = args.tversky_alpha
    if args.tversky_beta is not None:
        loss.tversky_beta = args.tversky_beta
    if args.tversky_gamma is not None:
        loss.tversky_gamma = args.tversky_gamma


def main(args) -> None:
    cfg = compose(config_name=args.config)
    if args.experiment_log_dir is not None:
        cfg.launcher.experiment_log_dir = args.experiment_log_dir
    elif args.run_name is not None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.run_name
        )
    elif cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    _apply_args_overrides(cfg, args)

    if args.test_split_dir is not None:
        try:
            manifest_path = dump_test_manifest(
                regions=list(cfg.scratch.train_regions),
                png_cache_dir=cfg.scratch.png_cache_dir,
                train_fraction=cfg.scratch.train_fraction,
                test_fraction=cfg.scratch.test_fraction,
                split_seed=cfg.scratch.split_seed,
                out_dir=args.test_split_dir,
            )
            if manifest_path:
                print(f"  Test-split manifest -> {manifest_path}")
        except Exception as e:
            # This is a side effect (a convenience manifest for later, manual
            # inference) -- it must never be able to abort an otherwise-healthy
            # training run.
            logging.warning("Failed to write test-split manifest: %s", e)

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    config_path = os.path.join(cfg.launcher.experiment_log_dir, "config.yaml")
    with g_pathmgr.open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    config_resolved_path = os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml")
    with g_pathmgr.open(config_resolved_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))

    print(f"  Config saved -> {config_path}")

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    # Priotrize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": (
                submitit_conf.name if hasattr(submitit_conf, "name") else args.config
            ),
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,  # one task per GPU
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        # Supports only cpu-bind option within srun_args. New options can be added here
        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        runner = SubmititRunner(main_port, cfg)
        job = executor.submit(runner)
        print(f"Submitit Job ID: {job.job_id}")
        runner.setup_job_info(job.job_id, rank=0)
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg, main_port)


if __name__ == "__main__":

    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU IDs to use, e.g. '0', '1', '0,1'. "
            "Overrides CUDA_VISIBLE_DEVICES. Defaults to all available GPUs "
            "(single-GPU runs use GPU 0 unless overridden)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Short name for this run (e.g. 'b+_rank16_v2'). "
            "Sets experiment_log_dir to <cwd>/sam2_logs/<run-name>, "
            "overriding whatever is in the config YAML. "
            "Use this to keep each attempt in its own directory without "
            "editing the YAML."
        ),
    )

    # ------------------------------------------------------------------ #
    # Model size                                                           #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(_MODEL_SIZE_CONFIGS.keys()),
        help=(
            "SAM2.1 Hiera backbone size: t (Tiny, ~38 M), s (Small, ~46 M), "
            "b+ (Base+, ~80 M, default), l (Large, ~224 M). "
            "Patches embed_dim, stages, global_att_blocks, window_spec, "
            "backbone_channel_list, and the pretrained checkpoint path."
        ),
    )

    # ------------------------------------------------------------------ #
    # Training scale                                                       #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs (scratch.num_epochs). "
             "Early stopping may terminate earlier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU training batch size (scratch.train_batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (scratch.num_train_workers).",
    )

    # ------------------------------------------------------------------ #
    # Optimiser                                                            #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--base-lr",
        type=float,
        default=None,
        help="Peak learning rate for the cosine schedule "
             "(scratch.base_lr and scratch.vision_lr). "
             "End value is always base_lr / 10.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay (scratch.weight_decay).",
    )

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Cap training tiles for data scarcity experiments (scratch.max_tiles). -1 = all.",
    )
    parser.add_argument(
        "--tiles-root",
        type=str,
        default=None,
        help="Root directory of TIF tiles (scratch.tiles_root).",
    )
    parser.add_argument(
        "--masks-root",
        type=str,
        default=None,
        help="Root directory of NPZ ground-truth masks (scratch.masks_root).",
    )
    parser.add_argument(
        "--png-cache-dir",
        type=str,
        default=None,
        help="PNG tile cache directory (scratch.png_cache_dir).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Random seed for the train/val tile split (scratch.split_seed). "
             "Vary across repeat sweeps to get independent samples for CIs.",
    )
    parser.add_argument(
        "--trainer-seed",
        type=int,
        default=None,
        help="Random seed for weight init, augmentation, and sampling "
             "(trainer.seed_value). Vary alongside --split-seed for independent runs.",
    )
    parser.add_argument(
        "--experiment-log-dir",
        type=str,
        default=None,
        help="Absolute path for experiment logs (launcher.experiment_log_dir). "
             "Takes priority over --run-name.",
    )
    parser.add_argument(
        "--test-split-dir",
        type=str,
        default=None,
        help="If set and scratch.test_fraction > 0, write the held-out "
             "in-distribution test tile names to <dir>/test_tiles.json "
             "(training.dataset.split_utils.dump_test_manifest). Omit to skip.",
    )

    # ------------------------------------------------------------------ #
    # LoRA                                                                 #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA inner rank applied to Hiera + Mask Decoder "
             "(scratch.lora_rank).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA scaling factor alpha (scratch.lora_alpha). "
             "Effective scale = alpha / rank.",
    )

    # ------------------------------------------------------------------ #
    # Early stopping                                                       #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after this many consecutive epochs without val-loss "
             "improvement (scratch.early_stop_patience). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-start",
        type=int,
        default=None,
        help="Epoch from which early stopping begins monitoring "
             "(scratch.early_stop_start_epoch). "
             "Epochs before this are always allowed to complete.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Minimum improvement in val loss required to reset the ES counter "
             "(scratch.early_stop_min_delta). Default 0.0 = any improvement counts.",
    )

    # ------------------------------------------------------------------ #
    # Checkpointing                                                        #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--ckpt-save-start",
        type=int,
        default=None,
        help="Numbered checkpoint_N.pt files are saved from this epoch onward "
             "(scratch.checkpoint_save_start_epoch). "
             "checkpoint.pt (the rolling resume file) is always saved.",
    )

    # ------------------------------------------------------------------ #
    # Loss weights                                                         #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--weight-bce",
        type=float,
        default=None,
        help="Weight for the SegmentationBCE term (trainer.loss.all.weight_bce). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--weight-dice",
        type=float,
        default=None,
        help="Weight for the SoftDice term (trainer.loss.all.weight_dice).",
    )
    parser.add_argument(
        "--weight-tversky",
        type=float,
        default=None,
        help="Weight for the FocalTversky term "
             "(trainer.loss.all.weight_tversky).",
    )
    parser.add_argument(
        "--weight-iou",
        type=float,
        default=None,
        help="Weight for the IoU-prediction auxiliary loss "
             "(trainer.loss.all.weight_iou).",
    )

    # ------------------------------------------------------------------ #
    # Tversky parameters                                                   #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--tversky-alpha",
        type=float,
        default=None,
        help="Tversky FP penalty alpha (trainer.loss.all.tversky_alpha). "
             "Lower = less FP penalty. alpha + beta should equal ~1.",
    )
    parser.add_argument(
        "--tversky-beta",
        type=float,
        default=None,
        help="Tversky FN penalty beta (trainer.loss.all.tversky_beta). "
             "Higher = penalise missed detections more strongly.",
    )
    parser.add_argument(
        "--tversky-gamma",
        type=float,
        default=None,
        help="Focal exponent for FocalTversky loss "
             "(trainer.loss.all.tversky_gamma). "
             "Higher = more focus on hard examples.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print all INFO-level log messages to the terminal (default: WARNING+ only).",
    )
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    # Propagate verbose flag so logger.py can widen the console level.
    if args.verbose:
        os.environ["SAM2_VERBOSE"] = "1"

    # Suppress noisy-but-harmless warnings that clutter the terminal.
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*amp.autocast.*")
    warnings.filterwarnings("ignore", message=".*anti_aliasing.*")
    warnings.filterwarnings("ignore", message=".*interpolation.*")

    # Apply GPU selection before any CUDA initialisation occurs.
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        num_selected = len(args.gpu_ids.split(","))
        if args.num_gpus is not None and args.num_gpus != num_selected:
            parser.error(
                f"--num-gpus ({args.num_gpus}) does not match the number of "
                f"--gpu-ids provided ({num_selected}: {args.gpu_ids})"
            )
        if args.num_gpus is None:
            args.num_gpus = num_selected
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Default: expose all GPUs. PyTorch will use GPU 0 for single-process runs.
        pass

    # Reduce memory fragmentation unless the caller already configured this.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    register_omegaconf_resolvers()
    main(args)
