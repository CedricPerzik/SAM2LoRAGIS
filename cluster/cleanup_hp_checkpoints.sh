#!/bin/bash
# cleanup_hp_checkpoints.sh
#
# Frees checkpoint storage for completed/failed HP sweep trials.
# Safe to run while other trials are still running -- skips all active SLURM jobs.
#
# Usage:
#   bash cleanup_hp_checkpoints.sh          # live run
#   bash cleanup_hp_checkpoints.sh --dry    # show what would be deleted, change nothing
#
# What it does:
#   COMPLETED trial (has logs/best_val_loss.json):
#     - Keeps all checkpoint_N_best.pt (best-val-loss snapshots)
#     - Deletes checkpoint.pt (rolling latest) and checkpoint_N.pt (per-epoch)
#   FAILED/ABANDONED trial (no best_val_loss.json, not in squeue):
#     - Deletes the entire checkpoints/ directory
#   RUNNING trial (job still in squeue): skipped entirely

set -euo pipefail

LOG_DIR="/home/s2145588/thesis/sam2loraboracluster/sam2/sam2_logs"
DRY=0
[[ "${1:-}" == "--dry" ]] && DRY=1
[[ "$DRY" == "1" ]] && echo "[DRY RUN] No files will be deleted."

# Current running SLURM job IDs (one per line)
RUNNING_JOBS=$(squeue -u s2145588 -h -o "%i" 2>/dev/null || true)

freed_bytes=0

for trial_dir in "$LOG_DIR"/hp_trial_*/; do
    [[ -d "$trial_dir" ]] || continue
    job_id="${trial_dir%/}"
    job_id="${job_id##*hp_trial_}"

    # -- Skip running jobs ---------------------------------------------------
    if echo "$RUNNING_JOBS" | grep -qx "$job_id"; then
        echo "[RUNNING ] $trial_dir -- skipped"
        continue
    fi

    ckpt_dir="${trial_dir}checkpoints"
    best_json="${trial_dir}logs/best_val_loss.json"

    # -- Completed trial: keep only *_best.pt --------------------------------
    if [[ -f "$best_json" ]]; then
        best_dice=$(python3 -c "import json; d=json.load(open('$best_json')); print(d.get('best_val_dice','?'))" 2>/dev/null || echo "?")
        echo "[COMPLETE] $trial_dir  (best_dice=$best_dice)"

        if [[ -d "$ckpt_dir" ]]; then
            while IFS= read -r -d '' f; do
                base=$(basename "$f")
                # Keep checkpoint_N_best.pt; delete checkpoint.pt and checkpoint_N.pt
                if [[ "$base" == "checkpoint.pt" ]] || \
                   { [[ "$base" =~ ^checkpoint_[0-9]+\.pt$ ]] && [[ "$base" != *_best.pt ]]; }; then
                    size=$(stat -c%s "$f" 2>/dev/null || echo 0)
                    freed_bytes=$((freed_bytes + size))
                    echo "  DELETE $f  ($(numfmt --to=iec $size 2>/dev/null || echo ${size}B))"
                    [[ "$DRY" == "1" ]] || rm -f "$f"
                else
                    echo "  keep   $f"
                fi
            done < <(find "$ckpt_dir" -maxdepth 1 -name "*.pt" -print0 2>/dev/null)
        fi

    # -- Failed/abandoned trial: drop entire checkpoints/ dir ----------------
    else
        echo "[FAILED  ] $trial_dir"
        if [[ -d "$ckpt_dir" ]]; then
            size=$(du -sb "$ckpt_dir" 2>/dev/null | cut -f1 || echo 0)
            freed_bytes=$((freed_bytes + size))
            echo "  DELETE dir $ckpt_dir  ($(du -sh "$ckpt_dir" 2>/dev/null | cut -f1 || echo '?'))"
            [[ "$DRY" == "1" ]] || rm -rf "$ckpt_dir"
        else
            echo "  (no checkpoints dir)"
        fi
    fi
done

echo ""
freed_gb=$(echo "scale=1; $freed_bytes / 1073741824" | bc 2>/dev/null || echo "$((freed_bytes/1073741824))")
echo "Total freed: ~${freed_gb} GB"
[[ "$DRY" == "1" ]] && echo "[DRY RUN complete -- rerun without --dry to apply]"
