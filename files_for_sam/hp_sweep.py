"""
Optuna HP trial wrapper - one call = one trial.

Each SLURM job calls:
    python hp_sweep.py --study-db <path/to/study.db> --log-dir <path/to/trial_log_dir>

Search space (4 params):
  base_lr      - log-uniform [5e-5, 5e-4]
  lora_rank    - categorical {4, 8, 16}  (lora_alpha fixed at 2*rank)
  tversky_beta - uniform [0.55, 0.90]
  weight_decay - log-uniform [1e-3, 0.3]

Objective: minimise (1 - best_val_dice).  Using Dice rather than the
composite val_loss keeps the objective independent of tversky_beta/gamma,
so trials are comparable across the full search space.

SQLite on NFS: Optuna retries on lock contention automatically. Works for
~30 concurrent trials; beyond that consider a PostgreSQL backend.
"""

import argparse
import json
import os
import subprocess
import sys

import optuna

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/home/s2145588/thesis/data"


def run_trial(trial: optuna.Trial, log_dir: str) -> float:
    base_lr      = trial.suggest_float("base_lr", 5e-5, 5e-4, log=True)
    lora_rank    = trial.suggest_categorical("lora_rank", [4, 8, 16])
    lora_alpha   = lora_rank * 2.0  # fixed alpha=2*rank (standard LoRA recommendation)
    tversky_beta = trial.suggest_float("tversky_beta", 0.55, 0.90)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 0.3, log=True)

    print(f"[hp_sweep] Trial {trial.number} - params: {trial.params}")

    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        sys.executable, "training/train.py",
        "-c", "configs/sam2.1_training/sam2.1_hiera_favela_lora",
        "--num-gpus", "1",
        "--use-cluster", "0",
        "--model-size", "l",
        "--tiles-root", f"{DATA_ROOT}/tiles",
        "--masks-root", f"{DATA_ROOT}/ground_truth_npz",
        "--png-cache-dir", f"{DATA_ROOT}/favela_png",
        "--max-tiles", "-1",
        "--base-lr", str(base_lr),
        "--lora-rank", str(int(lora_rank)),
        "--lora-alpha", str(float(lora_alpha)),
        "--weight-decay", str(weight_decay),
        "--tversky-beta", str(tversky_beta),
        "--experiment-log-dir", log_dir,
    ]

    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Training subprocess exited with code {result.returncode}")

    result_path = os.path.join(log_dir, "logs", "best_val_loss.json")
    with open(result_path) as f:
        data = json.load(f)
    # Objective: maximise val Dice - independent of tversky_beta/gamma in the loss.
    # Return 1 - dice so Optuna minimises it.
    return 1.0 - float(data["best_val_dice"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-db", required=True, help="Path to SQLite DB file")
    parser.add_argument("--study-name", default="hp_sweep")
    parser.add_argument("--log-dir", required=True,
                        help="Experiment log dir for this trial (unique per SLURM job)")
    args = parser.parse_args()

    storage = f"sqlite:///{args.study_db}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    trial = study.ask()

    try:
        value = run_trial(trial, args.log_dir)
        study.tell(trial, value)
        print(f"[hp_sweep] Trial {trial.number} done - 1-dice={value:.6f}  (dice={1-value:.6f})")
        print(f"[hp_sweep] Best so far: {study.best_value:.6f} "
              f"(trial {study.best_trial.number})")
    except Exception as exc:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        print(f"[hp_sweep] Trial {trial.number} FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
