#!/bin/bash
# cleanup_dataset_checkpoints.sh
#
# Frees checkpoint storage for completed/failed dataset-size sweep runs.
# Dataset sweep log dirs are named nY (e.g. n5, n10, n162), optionally with
# run_NNN subdirectories for HP-sweep trials (e.g. n5/run_001).
# Safe to run while other runs are active -- skips dirs whose SLURM job name
# matches the nY directory name (assumes job submitted with --job-name=nY).
#
# Usage:
#   bash cleanup_dataset_checkpoints.sh          # live run
#   bash cleanup_dataset_checkpoints.sh --dry    # show what would be deleted, change nothing
#
# What it does:
#   COMPLETED run (has logs/best_val_loss.json):
#     - Keeps only the highest-epoch checkpoint_N_best.pt (each _best save is a
#       strict global improvement over the last, so the highest epoch number
#       among them is the final/actual best -- older _best snapshots are
#       superseded and safe to delete)
#     - Deletes checkpoint.pt (rolling latest) and checkpoint_N.pt (per-epoch)
#   FAILED/ABANDONED run (no best_val_loss.json, not running):
#     - Deletes the entire checkpoints/ directory
#   RUNNING run (job name matches nY dir name in squeue): skipped entirely

set -euo pipefail

LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs"
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1
[[ "$DRY" == "1" ]] && echo "[DRY RUN] No files will be deleted."

# Running SLURM job names (dataset sweep jobs are submitted with --job-name=nY)
RUNNING_JOB_NAMES=$(squeue -u s2145588 -h -o "%j" 2>/dev/null || true)

freed_bytes=0

# cleanup_run <run_dir> <job_name>
# Applies checkpoint cleanup rules to a single run directory.
# job_name is the SLURM --job-name used for this run (nY for top-level dirs,
# parent nY for run_NNN subdirs) -- used to detect whether the job is active.
cleanup_run() {
    local run_dir="$1"
    local job_name="$2"

    if echo "$RUNNING_JOB_NAMES" | grep -qx "$job_name"; then
        echo "[RUNNING ] $run_dir -- skipped"
        return
    fi

    local ckpt_dir="${run_dir}checkpoints"
    local best_json="${run_dir}logs/best_val_loss.json"

    # -- Completed run: keep only *_best.pt ----------------------------------
    if [[ -f "$best_json" ]]; then
        local best_dice
        best_dice=$(python3 -c "import json; d=json.load(open('$best_json')); print(d.get('best_val_dice','?'))" 2>/dev/null || echo "?")
        echo "[COMPLETE] $run_dir  (best_dice=$best_dice)"

        if [[ -d "$ckpt_dir" ]]; then
            # Pass 1: find the highest epoch number among checkpoint_N_best.pt
            # files -- each _best save strictly improved on the last, so this
            # is the final/actual best checkpoint for the run.
            local max_best_epoch=-1
            while IFS= read -r -d '' f; do
                local base
                base=$(basename "$f")
                if [[ "$base" =~ ^checkpoint_([0-9]+)_best\.pt$ ]]; then
                    local ep="${BASH_REMATCH[1]}"
                    (( ep > max_best_epoch )) && max_best_epoch=$ep
                fi
            done < <(find "$ckpt_dir" -maxdepth 1 -name "checkpoint_*_best.pt" -print0 2>/dev/null)

            # Pass 2: keep checkpoint_<max_best_epoch>_best.pt only; delete
            # checkpoint.pt, checkpoint_N.pt, and every superseded _best.pt.
            while IFS= read -r -d '' f; do
                local base
                base=$(basename "$f")
                local keep=1
                if [[ "$base" == "checkpoint.pt" ]] || \
                   { [[ "$base" =~ ^checkpoint_[0-9]+\.pt$ ]] && [[ "$base" != *_best.pt ]]; }; then
                    keep=0
                elif [[ "$base" =~ ^checkpoint_([0-9]+)_best\.pt$ ]]; then
                    [[ "${BASH_REMATCH[1]}" == "$max_best_epoch" ]] || keep=0
                fi

                if [[ "$keep" == "0" ]]; then
                    local size
                    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
                    freed_bytes=$((freed_bytes + size))
                    echo "  DELETE $f  ($(numfmt --to=iec $size 2>/dev/null || echo ${size}B))"
                    [[ "$DRY" == "1" ]] || rm -f "$f"
                else
                    echo "  keep   $f"
                fi
            done < <(find "$ckpt_dir" -maxdepth 1 -name "*.pt" -print0 2>/dev/null)
        fi

    # -- Failed/abandoned run: drop entire checkpoints/ dir ------------------
    else
        echo "[FAILED  ] $run_dir"
        if [[ -d "$ckpt_dir" ]]; then
            local size
            size=$(du -sb "$ckpt_dir" 2>/dev/null | cut -f1 || echo 0)
            freed_bytes=$((freed_bytes + size))
            echo "  DELETE dir $ckpt_dir  ($(du -sh "$ckpt_dir" 2>/dev/null | cut -f1 || echo '?'))"
            [[ "$DRY" == "1" ]] || rm -rf "$ckpt_dir"
        else
            echo "  (no checkpoints dir)"
        fi
    fi
}

# -- Old layout: sam2_logs/nY/ -----------------------------------------------
for run_dir in "$LOG_DIR"/n[0-9]*/; do
    [[ -d "$run_dir" ]] || continue
    dir_name="${run_dir%/}"; dir_name="${dir_name##*/}"  # e.g. "n10"
    cleanup_run "$run_dir" "$dir_name"
done

# -- New layout: sam2_logs/run_NNN/nY/ ---------------------------------------
for run_dir in "$LOG_DIR"/run_[0-9]*/n[0-9]*/; do
    [[ -d "$run_dir" ]] || continue
    tmp="${run_dir%/}"
    n_name="${tmp##*/}"                      # e.g. "n162"
    run_name="${tmp%/*}"; run_name="${run_name##*/}"  # e.g. "run_001"
    job_name="sam2lora_${run_name}_${n_name}"        # e.g. "sam2lora_run_001_n162"
    cleanup_run "$run_dir" "$job_name"
done

echo ""
freed_gb=$(echo "scale=1; $freed_bytes / 1073741824" | bc 2>/dev/null || echo "$((freed_bytes/1073741824))")
echo "Total freed: ~${freed_gb} GB"
[[ "$DRY" == "1" ]] && echo "[DRY RUN complete -- rerun without --dry to apply]"
