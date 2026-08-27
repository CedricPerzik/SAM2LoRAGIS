# SAM2LoRaBoRaDD — SLURM Pipeline

LoRA fine-tuning of SAM2.1-Large for favela rooftop segmentation, run as a
**data scarcity sweep** on the UT EWI SLURM cluster: one training job per
training-set size (5, 10, 15, 25, 40, 65, 108 tiles), repeated across 5 seeds,
followed by inference and IoU evaluation sweeps.

This README documents every `.sh` script in this directory in the order you
actually run them, every setting each one exposes, and where to change values
that aren't exposed as flags (the Hydra YAML config).

## Cluster connection

```bash
# Requires EduVPN (Institute Access profile, https://ut.eduvpn.nl/portal/home)
ssh s2145588@hpc-head1.ewi.utwente.nl
```

| Resource | Cluster path |
|---|---|
| Code (model repo) | `/home/s2145588/thesis/sam2loraboracluster/` |
| Data (PNG cache) | `/home/s2145588/thesis/data/favela_png/` |
| SLURM job logs | `/home/s2145588/thesis/sam2loraboracluster/logs/` |
| TensorBoard logs | `/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs/<RUN_ID>/n<N>/` |
| Test-split manifests | `/home/s2145588/thesis/data/test_split/<RUN_ID>/test_tiles.json` |
| Predictions | `/home/s2145588/thesis/data/predictions/<RUN_ID>/n<N>/<area>/` |
| IoU results (CSV) | `/home/s2145588/thesis/data/iou_results/<RUN_ID>/n<N>/n<N>_<area>_unified_metrics.csv` |
| Conda env | `sam2lora` |

## Where to change values

Two layers control every run:

1. **The Hydra training config** —
   `sam2loraboracluster/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_favela_lora.yaml`.
   This is the source of truth for every hyperparameter: model size, LoRA
   rank/alpha, learning rate, batch size, loss weights, augmentation, split
   fractions, early stopping, checkpoint cadence, etc. Edit it directly for a
   permanent change.
2. **CLI overrides on top of the YAML** — `training/train.py` exposes most
   `scratch.*` and `trainer.loss.all.*` keys as flags (`--base-lr`,
   `--lora-rank`, `--weight-tversky`, ...). These win over the YAML for a
   single run without editing the file. `train_cluster.sh` is where you'd add
   a new flag to a submitted job; `submit_scarcity_sweep.sh` is where you'd
   thread a new sweep-wide setting (like `SEED`) through to it via
   `sbatch --export=`.

Any `scratch.*` key not covered by a dedicated flag can still be overridden
with a raw Hydra dotlist arg appended to the `python training/train.py` call,
e.g. `scratch.max_tiles=5`.

---

## Chronological execution order

```
1. download_ckpts.sh          (one-time setup, inside sam2loraboracluster/sam2/checkpoints/)
2. submit_scarcity_sweep.sh   -> train_cluster.sh        (training)
3. submit_inference_sweep.sh  -> predict_cluster.sh       (mask generation)
4. submit_iou_sweep.sh        -> iou_calculation.sh       (GT<->pred matching, metrics CSV)
5. cleanup_dataset_checkpoints.sh   (optional, storage hygiene, run any time after step 2)
6. rerun_jobs.sh               (optional, re-submit failures found in steps 2-4)

Separate track (LoRA hyperparameter search, not part of the scarcity sweep):
   submit_hp_sweep.sh -> hp_trial.sh   (sequential Optuna trials)
   cleanup_hp_checkpoints.sh           (optional, storage hygiene)
```

Steps 2-4 are strictly sequential *between* stages (inference needs
checkpoints, IoU needs predictions) but every job **within** a stage runs in
parallel across all seeds x sizes.

---

### 1. `download_ckpts.sh`

Location: `sam2loraboracluster/sam2/checkpoints/download_ckpts.sh` (part of
the upstream SAM2 repo, run once after the first `rsync` of the code).

Downloads the four pretrained SAM2.1 checkpoints (`tiny`, `small`, `base_plus`,
`large`) into that directory. `sam2.1_hiera_large.pt` is the one the favela
LoRA config loads by default. No settings to tune — just run it once:

```bash
cd /home/s2145588/thesis/sam2loraboracluster/sam2/checkpoints
bash download_ckpts.sh
```

---

### 2. `submit_scarcity_sweep.sh` -> `train_cluster.sh`

**`submit_scarcity_sweep.sh`** submits one SLURM training job per dataset
size, for one or more seeded "runs". All jobs in a run share a `RUN_ID`
(`run_001`, `run_002`, ...) and a `SEED`, so results land under
`sam2_logs/<RUN_ID>/n<N>/`.

Usage:

```bash
bash submit_scarcity_sweep.sh                       # 1 run, random seed
bash submit_scarcity_sweep.sh --runs 5               # 5 runs, random seeds
bash submit_scarcity_sweep.sh --seeds 1 26 42 99 1234  # 5 runs, explicit seeds (thesis default)
```

| Flag | Meaning |
|---|---|
| `--runs N` | Submit N runs with random seeds (0-99999). Ignored if `--seeds` given. |
| `--seeds S1 S2 ...` | Explicit seed list; number of runs = number of seeds. Overrides `--runs`. |

Hard-coded settings you can edit at the top of the script if needed:

| Variable | Default | Meaning |
|---|---|---|
| `SIZES` | `(5 10 15 25 40 65 108)` | Training-set sizes swept per run (`scratch.max_tiles` per job). Must match `submit_inference_sweep.sh` / `submit_iou_sweep.sh`. |
| `SAM2_LOGS_DIR`, `LOG_DIR`, `TEST_SPLIT_DIR` | cluster paths | Where results/logs/manifests land. |

Each SLURM job it submits is `train_cluster.sh`, with `DATASET_SIZE`,
`RUN_ID`, `SEED` injected via `sbatch --export=`.

**`train_cluster.sh`** is the actual training job (not run directly — always
via the submit script or `rerun_jobs.sh`).

SLURM resource directives (edit in the script header if the job needs more/less):

| Directive | Default |
|---|---|
| `--partition` | `main-gpu` |
| `--gres` | `gpu:lovelace:1` (L40/L40s, ~44 GiB) |
| `-c` | `4` |
| `--mem` | `64gb` |
| `--time` | `50:00:00` |

Python invocation (`training/train.py`) — flags fed from env vars set by the
submit script, plus fixed flags:

| Flag | Source | Meaning |
|---|---|---|
| `-c configs/sam2.1_training/sam2.1_hiera_favela_lora` | fixed | The YAML config (see above) — nearly everything else in the run is driven from here. |
| `--model-size l` | fixed | SAM2.1-Large (~224M params). Change to `t`/`s`/`b+` for smaller backbones. |
| `--max-tiles ${DATASET_SIZE}` | sweep | Data-scarcity cap on training tiles (`scratch.max_tiles`). |
| `--split-seed ${SEED}` | sweep | Which tiles land in train/val/test. |
| `--trainer-seed ${SEED}` | sweep | LoRA init + batch order. |
| `--experiment-log-dir` | derived | `sam2_logs/${RUN_ID}/n${DATASET_SIZE}` |
| `--test-split-dir` | derived | `data/test_split/${RUN_ID}` — dumps the held-out ID-test tile manifest once per run. |
| `--tiles-root`, `--masks-root`, `--png-cache-dir` | fixed | Cluster data paths. |

Any other training hyperparameter (learning rate, LoRA rank, loss weights,
early stopping, batch size, ...) can be added here as an extra CLI flag — see
the full flag list in `training/train.py`'s `argparse` block — or changed
permanently in the YAML. Useful ones for future sweeps:

```bash
--base-lr 2e-4 --lora-rank 16 --lora-alpha 32 \
--weight-tversky 1.0 --weight-iou 1.0 --tversky-alpha 0.3 --tversky-beta 0.55 \
--early-stop-patience 5 --early-stop-start 5 --ckpt-save-start 1 \
--batch-size 4 --num-workers 4 --num-epochs 30
```

---

### 3. `submit_inference_sweep.sh` -> `predict_cluster.sh`

Runs `SAM2AutomaticMaskGenerator` for every `(RUN_ID, N)` checkpoint,
producing mask dumps (no GT comparison yet). Requires
`custom_notebooks/B3_custom_predict_cluster.py` and `B_functions.py` to
already be copied onto the cluster (`custom_notebooks/` next to `sam2/`).

Usage:

```bash
bash submit_inference_sweep.sh --run-ids run_001
bash submit_inference_sweep.sh --run-ids run_001 run_002 run_003
```

| Flag | Meaning |
|---|---|
| `--run-ids R1 R2 ...` | Required. Which completed training runs to run inference for. |

Hard-coded: `SIZES` (must match `submit_scarcity_sweep.sh`). Areas covered per
job: `santa_madalena` (OOD, always, full region) plus one subdir per ID
region (`ceu_paz`, `cantidio_sampaio`) restricted to that run's held-out
`test_tiles.json`. Falls back to OOD-only with a warning if the manifest is
missing (e.g. an older run, or a run trained with `test_fraction=0`).

**`predict_cluster.sh`** SLURM directives:

| Directive | Default |
|---|---|
| `--gres` | `gpu:lovelace:1` |
| `-c` | `4` |
| `--mem` | `32gb` |
| `--time` | `04:00:00` |

No CLI flags — `DATASET_SIZE`/`RUN_ID` come from `sbatch --export=`; mask
generator settings (points-per-side, IoU/stability thresholds, etc.) live
inside `B3_custom_predict_cluster.py` itself if you need to tune those.

---

### 4. `submit_iou_sweep.sh` -> `iou_calculation.sh`

CPU-only job: greedy 1-to-1 GT<->prediction IoU matching
(`D_functions.py`) and export of a unified metrics CSV per area. Requires
`custom_notebooks/D2_dataset_iou_cluster.py` and `D_functions.py` on the
cluster. Typically run after the inference sweep's jobs finish for the same
runs. Idempotent — re-running only fills in missing `_eval.npz` files.

Usage:

```bash
bash submit_iou_sweep.sh --run-ids run_001
bash submit_iou_sweep.sh --run-ids run_001 run_002 run_003
```

| Flag | Meaning |
|---|---|
| `--run-ids R1 R2 ...` | Required. Same runs you passed to `submit_inference_sweep.sh`. |

**`iou_calculation.sh`** SLURM directives:

| Directive | Default |
|---|---|
| `--partition` | `main-gpu` (no GPU reserved) |
| `-c` | `8` |
| `--mem` | `16gb` |
| `--time` | `05:00:00` |

Areas are discovered dynamically from whatever `predict_cluster.sh` produced
under `predictions/<RUN_ID>/n<N>/` — no separate area list to maintain.
Output: `iou_results/<RUN_ID>/n<N>/n<N>_<area>_unified_metrics.csv`.

---

### 5. `cleanup_dataset_checkpoints.sh` (optional, run any time after step 2)

Frees checkpoint storage from the scarcity sweep. Safe to run while other
runs are still training (skips any `nY` dir whose SLURM job name is still in
`squeue`).

```bash
bash cleanup_dataset_checkpoints.sh          # live run
bash cleanup_dataset_checkpoints.sh --dry    # show what would be deleted only
```

Rules: completed runs (`logs/best_val_loss.json` present) keep only the
highest-epoch `checkpoint_N_best.pt`; failed/abandoned runs (no
`best_val_loss.json`, not running) have their whole `checkpoints/` dir
deleted; running jobs are skipped entirely. No settings beyond `--dry`.

---

### 6. `rerun_jobs.sh` (optional)

Re-submits failed training jobs recorded in `logs/seeds.txt` (one line per
run, written by `submit_scarcity_sweep.sh`: `RUN_ID SEED`). To re-run
specific sizes for a run, append the sizes to its line:

```
run_001 42 5 10
run_002 137
run_003 999 25 40 65
```

`run_002` above has no sizes listed and is skipped. Then:

```bash
bash rerun_jobs.sh
```

No flags — edit `logs/seeds.txt` directly to control what gets resubmitted.
Submits through the same `train_cluster.sh` path as the original sweep.

---

## Separate track: hyperparameter search

Not part of the scarcity sweep — used earlier to find the LoRA/optimizer
values now baked into the YAML defaults (`base_lr`, `lora_rank`,
`weight_decay`, `tversky_beta`, ...). Kept here for re-tuning later (e.g. if
the dataset changes).

### `submit_hp_sweep.sh` -> `hp_trial.sh`

Submits N **sequential** Optuna Bayesian-optimization trials (parallel jobs
break BO's sequential exploitation of prior trials). All trials share one
SQLite study DB on the NFS share.

```bash
bash submit_hp_sweep.sh            # 10 trials (this script's default)
bash submit_hp_sweep.sh 30         # 30 trials (recommended minimum for BO to beat random search)
```

| Arg | Meaning |
|---|---|
| `$1` (positional) | Number of trials. Default `10`. |

`STUDY_DB` path is hard-coded at the top of the script
(`sam2loraboracluster/hp_study.db`) — change there if you want a fresh study.

**`hp_trial.sh`** SLURM directives:

| Directive | Default |
|---|---|
| `--gres` | `gpu:lovelace:1` |
| `-c` | `8` |
| `--mem` | `48gb` |
| `--time` | `24:00:00` |

Runs `hp_sweep.py --study-db ... --study-name hp_sweep --log-dir ...`. The
search space itself (which hyperparameters Optuna suggests and their ranges)
is defined inside `hp_sweep.py`, not in this script.

Inspect results after trials complete:

```python
import optuna
s = optuna.load_study(study_name="hp_sweep", storage="sqlite:///.../hp_study.db")
print(s.best_params)
```

### `cleanup_hp_checkpoints.sh` (optional)

Same idea as `cleanup_dataset_checkpoints.sh` but for `hp_trial_<jobid>/`
dirs, and keeps **all** `checkpoint_N_best.pt` snapshots (not just the
highest) since you may want to compare trial trajectories.

```bash
bash cleanup_hp_checkpoints.sh          # live run
bash cleanup_hp_checkpoints.sh --dry    # show what would be deleted only
```

---

## Monitoring

```bash
squeue -u s2145588           # running / queued jobs
seff <jobid>                 # efficiency report after completion
# Web dashboard: http://hpc-status.ewi.utwente.nl/slurm/
```

TensorBoard (SSH tunnel from local machine):

```bash
ssh -L 6006:localhost:6006 s2145588@hpc-head1.ewi.utwente.nl
# on the cluster:
tensorboard --logdir /home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs/ --port 6006
```

---

## End-to-end example (5 seeds x 7 sizes, full pipeline)

```bash
# 1. One-time setup
cd /home/s2145588/thesis/sam2loraboracluster/sam2/checkpoints && bash download_ckpts.sh
cd /home/s2145588/thesis/sam2loraboracluster/sam2 && pip install -e .

# 2. Train — submits 35 jobs (5 seeds x 7 sizes), creates run_001..run_005
cd /home/s2145588/thesis
bash submit_scarcity_sweep.sh --seeds 1 26 42 99 1234

# (wait for squeue to clear)

# 3. Inference — masks for all 5 runs
bash submit_inference_sweep.sh --run-ids run_001 run_002 run_003 run_004 run_005

# (wait)

# 4. IoU / metrics CSVs for all 5 runs
bash submit_iou_sweep.sh --run-ids run_001 run_002 run_003 run_004 run_005

# 5. Free checkpoint storage once you have what you need locally
bash cleanup_dataset_checkpoints.sh
```
