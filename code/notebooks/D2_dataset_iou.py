# ---
# jupyter:
#   jupytext:
#     formats: ipynb, py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: daksam
#     language: python
#     name: python3
# ---

# %%
# Skip to 144 for calculating IoU metrics for runs.
# %%
import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import D_functions as df
import sam_repo_notebooks.B_functions as bf

# %%
# === Configuration ===
# Pre-defined dataset sizes for the scarcity sweep.
# Edit this list to add new sizes, re-run a subset, or skip sizes.
# DATASET_SIZES = [5, 10, 20, 40, 60, 80, 100, 120, 140, 160]
DATASET_SIZES = [5, 10, 15, 25, 40, 65, 105, 162]  # 162 = full dataset (all tiles)

# Local directory where scarcity-sweep checkpoints live (rsync'd from SLURM).
# Path pattern: CKPT_BASE_DIR/n<N>/checkpoints/checkpoint_<epoch>_best.pt
DATASET_SWEEP_CKPT_DIR = "dataset_sweep_val_loss_delta"
CKPT_BASE_DIR = f"/home/ced/drives/dookiedisk/sam2_logs/{DATASET_SWEEP_CKPT_DIR}"

# Out-of-distribution area to evaluate on (not used during training).
AREA_NAME = "santa_madalena"

# === Path resolution ===
CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR

TILES_DIR        = os.path.join(CODE_DIR, "output_data", "tiles_png", AREA_NAME)
GT_DIR           = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", AREA_NAME)
PREDICTIONS_BASE = os.path.join(CODE_DIR, "output_data", "predictions", DATASET_SWEEP_CKPT_DIR)
EVAL_BASE        = os.path.join(CODE_DIR, "output_data", "evaluation_npz", DATASET_SWEEP_CKPT_DIR)
IOU_RESULTS_DIR  = os.path.join(CODE_DIR, "output_data", "iou_results", DATASET_SWEEP_CKPT_DIR)
BASE_IOU_RESULTS_DIR  = os.path.join(CODE_DIR, "output_data", "iou_results")

print(f"Tiles dir:    {TILES_DIR}")
print(f"Ground truth: {GT_DIR}")
print(f"Output base:  {IOU_RESULTS_DIR}")

# %%
def find_best_checkpoint(n):
    """Return (epoch, path) for the highest-numbered checkpoint_N_best.pt in n<N>."""
    ckpt_dir = os.path.join(CKPT_BASE_DIR, f"n{n}", "checkpoints")
    pattern = re.compile(r"checkpoint_(\d+)_best\.pt$")
    candidates = [
        (int(pattern.search(f).group(1)), f)
        for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_*_best.pt"))
        if pattern.search(f)
    ]
    if not candidates:
        raise FileNotFoundError(f"No _best.pt checkpoints found in {ckpt_dir}")
    return max(candidates, key=lambda x: x[0])

# %%
# Resolve all checkpoints upfront; skip sizes where the checkpoint directory is missing.
checkpoint_map = {}  # n -> (epoch, path)
for n in DATASET_SIZES:
    try:
        epoch, ckpt_path = find_best_checkpoint(n)
        checkpoint_map[n] = (epoch, ckpt_path)
        print(f"n{n:>4d}: epoch {epoch:>3d}  ->  {os.path.basename(ckpt_path)}")
    except FileNotFoundError as e:
        print(f"SKIP n{n}: {e}")

# %%
# JIT warmup - must run once in the main process before two_gpu_predictor is called.
# Spawned child processes re-import this file but have __name__ != '__main__',
# so the warmup and inference blocks are only executed in the top-level session.
if __name__ == '__main__' and checkpoint_map:
    first_ckpt = checkpoint_map[min(checkpoint_map)][1]
    print(f"Warming up JIT compiler...")
    _dummy = bf.build_sam2_lora(first_ckpt, device="cpu")
    _ = SAM2AutomaticMaskGenerator(model=_dummy)
    del _dummy
    print("Warm-up complete.")

# %%
# === Inference loop ===
# Run SAM2 automatic mask generation for each dataset size on the OOD area.
# A size is skipped if its prediction directory already contains mask files.
if __name__ == '__main__':
    for n, (epoch, ckpt_path) in sorted(checkpoint_map.items()):
        model_label = f"n{n}"
        pred_dir = os.path.join(PREDICTIONS_BASE, model_label, AREA_NAME)
        os.makedirs(pred_dir, exist_ok=True)

        existing = glob.glob(os.path.join(pred_dir, "*_masks.npz"))
        if existing:
            print(f"n{n}: {len(existing)} mask files already exist - skipping inference.")
            continue

        print(f"\n--- Inference: n{n} | epoch {epoch} | {os.path.basename(ckpt_path)} ---")
        bf.two_gpu_predictor(TILES_DIR, pred_dir, None, ckpt_path, is_lora=True)

# %%
# === Evaluation loop ===
# Match predictions to ground-truth annotations and compute per-size IoU metrics.
all_results = {}

for n, (epoch, ckpt_path) in sorted(checkpoint_map.items()):
    model_label = f"n{n}"
    pred_dir = os.path.join(PREDICTIONS_BASE, model_label, AREA_NAME)
    eval_dir = os.path.join(EVAL_BASE,        model_label, AREA_NAME)

    print(f"\n--- Evaluation: n{n} ---")
    df.run_evaluation_pipeline_concurrent(GT_DIR, pred_dir, eval_dir)

    result = df.inspect_evaluation_metrics(eval_dir, area_name=f"n{n} ({AREA_NAME})")
    if result is None:
        print(f"n{n}: No evaluation data produced - skipping.")
        continue
    image_stats, matched_ious, strict_ious = result

    csv_path = os.path.join(IOU_RESULTS_DIR, model_label, f"n{n}_{AREA_NAME}_unified_metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.export_unified_metrics_to_csv_concurrent(eval_dir, GT_DIR, csv_path)

    all_results[n] = {
        "epoch":        epoch,
        "matched_ious": matched_ious or [],
        "strict_ious":  strict_ious  or [],
    }

# %%
# === Multi-run evaluation (run_001..run_003) ===
# Evaluates predictions from three independent training runs so the summary
# plots can show a mean line with a min/max error band.
RUNS           = ["run_001", "run_002", "run_004", "run_005", "run_006"]
RUNS_PRED_BASE = "/home/ced/drives/dookiedisk/predictions"
RUNS_EVAL_BASE = os.path.join(CODE_DIR, "output_data", "evaluation_npz", "runs")
RUNS_IOU_BASE  = os.path.join(CODE_DIR, "output_data", "iou_results", "runs")

per_run_results = {}  # {run: {n: {"matched_ious": [...], "strict_ious": [...]}}}

for _run in RUNS:
    per_run_results[_run] = {}
    for _n in DATASET_SIZES:
        _pred_dir = os.path.join(RUNS_PRED_BASE, _run, f"n{_n}", AREA_NAME)
        _eval_dir = os.path.join(RUNS_EVAL_BASE, _run, f"n{_n}", AREA_NAME)

        if not os.path.isdir(_pred_dir):
            print(f"{_run} n{_n}: prediction dir not found - skipping")
            continue

        print(f"\n--- Evaluation: {_run} n{_n} ---")
        df.run_evaluation_pipeline_concurrent(GT_DIR, _pred_dir, _eval_dir)

        _result = df.inspect_evaluation_metrics(_eval_dir, area_name=f"{_run} n{_n} ({AREA_NAME})")
        if _result is None:
            print(f"{_run} n{_n}: no evaluation data - skipping")
            continue
        _, _matched, _strict = _result

        _csv_path = os.path.join(RUNS_IOU_BASE, _run, f"n{_n}", f"n{_n}_{AREA_NAME}_unified_metrics.csv")
        os.makedirs(os.path.dirname(_csv_path), exist_ok=True)
        df.export_unified_metrics_to_csv_concurrent(_eval_dir, GT_DIR, _csv_path)

        per_run_results[_run][_n] = {
            "matched_ious": _matched or [],
            "strict_ious":  _strict  or [],
        }
    print(f"{_run}: evaluated {len(per_run_results[_run])} sizes")

# %%
# === Summary: mIoU vs training-set size (3-run mean with min/max band) ===
ZERO_SHOT_MATCHED = 0.7744
ZERO_SHOT_STRICT  = 0.7665

_mr_sizes = sorted({n for r in per_run_results.values() for n in r})

def _per_run_miou(metric_key, sizes):
    """Return (n_runs × n_sizes) array of per-run mean mIoU; NaN where data is missing."""
    return np.array([
        [float(np.mean(per_run_results[r].get(s, {}).get(metric_key, None) or [np.nan]))
         for s in sizes]
        for r in sorted(per_run_results)
    ])

_mr_matched = _per_run_miou("matched_ious", _mr_sizes)
_mr_strict  = _per_run_miou("strict_ious",  _mr_sizes)

_mr_plot_x = [0] + _mr_sizes
_mr_m_mean = np.concatenate([[ZERO_SHOT_MATCHED], np.nanmean(_mr_matched, axis=0)])
_mr_m_lo   = np.concatenate([[ZERO_SHOT_MATCHED], np.nanmin( _mr_matched, axis=0)])
_mr_m_hi   = np.concatenate([[ZERO_SHOT_MATCHED], np.nanmax( _mr_matched, axis=0)])
_mr_s_mean = np.concatenate([[ZERO_SHOT_STRICT],  np.nanmean(_mr_strict,  axis=0)])
_mr_s_lo   = np.concatenate([[ZERO_SHOT_STRICT],  np.nanmin( _mr_strict,  axis=0)])
_mr_s_hi   = np.concatenate([[ZERO_SHOT_STRICT],  np.nanmax( _mr_strict,  axis=0)])

# --- Plot export settings ---
PAPER_MODE    = True    # True = use settings optimized for paper figures; False = larger, more readable
DOUBLE_COLUMN = True    # True = ~7" wide; False = ~3.5" wide (single column)
EXPORT_DPI    = None    # e.g. 300; None = display only, no file written
EXPORT_PATH   = None    # e.g. "figures/scarcity_sweep.pdf"; required when EXPORT_DPI is set

if PAPER_MODE:
    figsize = (7.0, 3.5) if DOUBLE_COLUMN else (3.5, 2.8)
    base_fs = 9
    lw, ms  = 1.5, 5
else:
    figsize = (10, 5)
    base_fs = plt.rcParams["font.size"]
    lw, ms  = 1.5, 6

with plt.rc_context({
    "font.size":       base_fs,
    "axes.labelsize":  base_fs,
    "xtick.labelsize": base_fs,
    "ytick.labelsize": base_fs,
    "legend.fontsize": base_fs,
    "axes.titlesize":  base_fs + 1,
}):
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(_mr_plot_x, _mr_m_lo, _mr_m_hi, alpha=0.2, color="steelblue")
    ax.plot(_mr_plot_x, _mr_m_mean, "o-",  color="steelblue", linewidth=lw, markersize=ms,
            label="Matched mIoU (found roofs only)")
    ax.fill_between(_mr_plot_x, _mr_s_lo, _mr_s_hi, alpha=0.2, color="tomato")
    ax.plot(_mr_plot_x, _mr_s_mean, "s--", color="tomato",    linewidth=lw, markersize=ms,
            label="Strict mIoU (incl. missed roofs)")
    ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
    ax.set_ylabel("mIoU")
    if not PAPER_MODE:
        ax.set_title(f"Data scarcity sweep - OOD performance on {AREA_NAME} ({len(RUNS)} runs)")
    ax.set_xticks(_mr_plot_x)
    if PAPER_MODE:
        ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if EXPORT_DPI and EXPORT_PATH:
        fig.savefig(EXPORT_PATH, dpi=EXPORT_DPI, bbox_inches="tight")
        print(f"Figure saved to {EXPORT_PATH}")
    plt.show()

# %%
# Summary table (single-run, from checkpoint_map evaluation)
sizes = sorted(all_results.keys())
print(f"\n{'n':>6}  {'epoch':>6}  {'matched mIoU':>14}  {'strict mIoU':>12}")
print("-" * 46)
print(f"{'zs':>6}  {'--':>6}  {ZERO_SHOT_MATCHED:>14.4f}  {ZERO_SHOT_STRICT:>12.4f}")
for n in sizes:
    ep = all_results[n]["epoch"]
    mm = float(np.mean(all_results[n]["matched_ious"])) if all_results[n]["matched_ious"] else 0.0
    sm = float(np.mean(all_results[n]["strict_ious"]))  if all_results[n]["strict_ious"]  else 0.0
    print(f"{n:>6}  {ep:>6}  {mm:>14.4f}  {sm:>12.4f}")

# %%
# =============================================================================
# === EXPERIMENTAL: mIoU vs training-set size (cut rooftops excluded) ===
# =============================================================================
import pandas as pd

ZEROSHOT_CSV = os.path.join(os.path.dirname(IOU_RESULTS_DIR), f"{AREA_NAME}_unified_metrics.csv")

# --- Plot export settings (cut-excluded variant) ---
CUT_PAPER_MODE    = True
CUT_DOUBLE_COLUMN = True
CUT_EXPORT_DPI    = None
CUT_EXPORT_PATH   = None

def load_filtered_miou(csv_path):
    """Return (matched_miou, strict_miou, n_excluded, n_included) from a unified metrics CSV, cut roofs removed."""
    _df          = pd.read_csv(csv_path)
    n_excluded   = int(_df["is_cut"].sum())
    _df          = _df[~_df["is_cut"]]
    n_included   = len(_df)
    strict_miou  = float(_df["iou_score"].fillna(0).mean())
    matched_miou = float(_df.loc[_df["matched"], "iou_score"].mean())
    return matched_miou, strict_miou, n_excluded, n_included

# Zero-shot baseline (single reference run)
zs_matched, zs_strict, zs_excl, zs_incl = load_filtered_miou(ZEROSHOT_CSV)
print(f"{'':>12}  {'excl':>6}  {'incl':>6}  {'matched mIoU':>14}  {'strict mIoU':>12}")
print("-" * 60)
print(f"{'zs':>12}  {zs_excl:>6}  {zs_incl:>6}  {zs_matched:>14.4f}  {zs_strict:>12.4f}")

# Load per-run cut-excluded results from multi-run CSVs
_cut_per_run = {}  # {run: {n: {"matched": ..., "strict": ...}}}
for _run in RUNS:
    _cut_per_run[_run] = {}
    for _n in sorted(per_run_results.get(_run, {}).keys()):
        _csv = os.path.join(RUNS_IOU_BASE, _run, f"n{_n}", f"n{_n}_{AREA_NAME}_unified_metrics.csv")
        if not os.path.exists(_csv):
            print(f"{_run} n{_n}: CSV not found - skipping")
            continue
        _m, _s, _ne, _ni = load_filtered_miou(_csv)
        _cut_per_run[_run][_n] = {"matched": _m, "strict": _s}
        print(f"{_run} n{_n:>4d}  {_ne:>6}  {_ni:>6}  {_m:>14.4f}  {_s:>12.4f}")

# %%
_cut_sizes = sorted({n for r in _cut_per_run.values() for n in r})

_cut_m_arr = np.array([
    [_cut_per_run[r].get(n, {}).get("matched", np.nan) for n in _cut_sizes]
    for r in sorted(_cut_per_run)
])
_cut_s_arr = np.array([
    [_cut_per_run[r].get(n, {}).get("strict", np.nan) for n in _cut_sizes]
    for r in sorted(_cut_per_run)
])

_cut_plot_x = [0] + _cut_sizes
_cut_m_mean = np.concatenate([[zs_matched], np.nanmean(_cut_m_arr, axis=0)])
_cut_m_lo   = np.concatenate([[zs_matched], np.nanmin( _cut_m_arr, axis=0)])
_cut_m_hi   = np.concatenate([[zs_matched], np.nanmax( _cut_m_arr, axis=0)])
_cut_s_mean = np.concatenate([[zs_strict],  np.nanmean(_cut_s_arr, axis=0)])
_cut_s_lo   = np.concatenate([[zs_strict],  np.nanmin( _cut_s_arr, axis=0)])
_cut_s_hi   = np.concatenate([[zs_strict],  np.nanmax( _cut_s_arr, axis=0)])

if CUT_PAPER_MODE:
    cut_figsize = (7.0, 3.5) if CUT_DOUBLE_COLUMN else (3.5, 2.8)
    cut_base_fs = 9
    cut_lw, cut_ms = 1.5, 5
else:
    cut_figsize = (10, 5)
    cut_base_fs = plt.rcParams["font.size"]
    cut_lw, cut_ms = 1.5, 6

with plt.rc_context({
    "font.size":       cut_base_fs,
    "axes.labelsize":  cut_base_fs,
    "xtick.labelsize": cut_base_fs,
    "ytick.labelsize": cut_base_fs,
    "legend.fontsize": cut_base_fs,
    "axes.titlesize":  cut_base_fs + 1,
}):
    fig, ax = plt.subplots(figsize=cut_figsize)
    ax.fill_between(_cut_plot_x, _cut_m_lo, _cut_m_hi, alpha=0.2, color="steelblue")
    ax.plot(_cut_plot_x, _cut_m_mean, "o-",  color="steelblue", linewidth=cut_lw, markersize=cut_ms,
            label="Matched mIoU (found roofs only)")
    ax.fill_between(_cut_plot_x, _cut_s_lo, _cut_s_hi, alpha=0.2, color="tomato")
    ax.plot(_cut_plot_x, _cut_s_mean, "s--", color="tomato",    linewidth=cut_lw, markersize=cut_ms,
            label="Strict mIoU (incl. missed roofs)")
    ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
    ax.set_ylabel("mIoU")
    if not CUT_PAPER_MODE:
        ax.set_title(f"Data scarcity sweep - OOD on {AREA_NAME} (cut rooftops excluded, {len(RUNS)} runs)")
    ax.set_xticks(_cut_plot_x)
    if CUT_PAPER_MODE:
        ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if CUT_EXPORT_DPI and CUT_EXPORT_PATH:
        fig.savefig(CUT_EXPORT_PATH, dpi=CUT_EXPORT_DPI, bbox_inches="tight")
        print(f"Figure saved to {CUT_EXPORT_PATH}")
    plt.show()

# %%
# =============================================================================
# === EXPERIMENTAL: Deep-dive roof-level and geometry analyses ===
# =============================================================================

# --- Load all per-n CSVs into one combined dataframe ---
_frames = []
for _n in sorted(all_results.keys()):
    _csv = os.path.join(IOU_RESULTS_DIR, f"n{_n}", f"n{_n}_{AREA_NAME}_unified_metrics.csv")
    if os.path.exists(_csv):
        _d = pd.read_csv(_csv)
        _d["n"] = _n
        _frames.append(_d)

_d0 = pd.read_csv(ZEROSHOT_CSV)
_d0["n"] = 0
df_all = pd.concat([_d0] + _frames, ignore_index=True)
df_all["iou_score"]   = df_all["iou_score"].fillna(0.0)
df_all["matched"]     = df_all["matched"].astype(bool)
df_all["is_cut"]      = df_all["is_cut"].astype(bool)
df_all["is_artifact"] = df_all["is_artifact"].astype(bool)

_ns_plot      = [_n for _n in sorted(df_all["n"].unique()) if _n != 162]
_best_n       = max(r for r in all_results.keys() if r != 162)
_xtick_labels = ["zs" if _n == 0 else str(_n) for _n in _ns_plot]

_n_roofs = int((df_all["n"] == 0).sum())
print(f"Loaded {len(df_all)} rows | {df_all['n'].nunique()} checkpoints | "
      f"{_n_roofs} roofs per checkpoint")

# %%
# === [EXP] 1. Cut vs non-cut IoU distribution ===
# Strict IoU (0 for missed) for zero-shot and best fine-tuned model side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

for ax, _n in zip(axes, [0, _best_n]):
    _d      = df_all[df_all["n"] == _n]
    _noncut = _d[~_d["is_cut"]]["iou_score"].values
    _cut    = _d[  _d["is_cut"]]["iou_score"].values
    _all    = _d["iou_score"].values
    _nc_mr  = _d[~_d["is_cut"]]["matched"].mean()
    _cut_mr = _d[  _d["is_cut"]]["matched"].mean()
    _all_mr = _d["matched"].mean()

    parts = ax.violinplot([_noncut, _cut, _all], positions=[0, 1, 2], showmedians=True, widths=0.7)
    parts["bodies"][0].set_facecolor("steelblue"); parts["bodies"][0].set_alpha(0.7)
    parts["bodies"][1].set_facecolor("tomato");    parts["bodies"][1].set_alpha(0.7)
    parts["bodies"][2].set_facecolor("mediumpurple"); parts["bodies"][2].set_alpha(0.7)
    for _key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if _key in parts:
            parts[_key].set_color("black"); parts[_key].set_linewidth(1.0)

    for _pos, _vals in zip([0, 1, 2], [_noncut, _cut, _all]):
        _med = float(np.median(_vals))
        ax.text(_pos + 0.16, _med, f"{_med:.3f}", va="center", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([
        f"Non-cut\n{_nc_mr:.0%} matched\nn={len(_noncut)}",
        f"Cut\n{_cut_mr:.0%} matched\nn={len(_cut)}",
        f"All\n{_all_mr:.0%} matched\nn={len(_all)}",
    ])
    ax.set_title("Zero-shot" if _n == 0 else f"Fine-tuned (n={_n})")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, alpha=0.3, axis="y")

axes[0].set_ylabel("IoU score (strict: 0 if missed)")
fig.suptitle(f"IoU distribution - cut vs non-cut roofs ({AREA_NAME})")
plt.tight_layout()
plt.show()

# %%
# === [EXP] 2. IoU percentile bands (non-cut, non-artifact) ===
_d_clean = df_all[~df_all["is_cut"] & ~df_all["is_artifact"] & df_all["n"].isin(_ns_plot)]
_pct     = _d_clean.groupby("n")["iou_score"].quantile([0.25, 0.50, 0.75]).unstack()
_pct.columns = ["p25", "p50", "p75"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(_pct.index, _pct["p25"], _pct["p75"],
                alpha=0.2, color="steelblue", label="p25-p75 range")
ax.plot(_pct.index, _pct["p50"], "o-", color="steelblue",
        linewidth=1.5, markersize=5, label="Median IoU")
ax.plot(_pct.index, _pct["p25"], "--", color="steelblue", linewidth=0.8, alpha=0.5)
ax.plot(_pct.index, _pct["p75"], "--", color="steelblue", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
ax.set_ylabel("IoU score (strict: 0 if missed)")
ax.set_title(f"IoU percentile bands - non-cut, non-artifact roofs ({AREA_NAME})")
ax.set_xticks(_ns_plot)
ax.set_xticklabels(_xtick_labels)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# === [EXP] 3. Matched vs missed stacked bar (all roofs) ===
_d_bar   = df_all[df_all["n"].isin(_ns_plot)]
_matched = _d_bar.groupby("n")["matched"].sum().reindex(_ns_plot)
_total   = _d_bar.groupby("n")["matched"].count().reindex(_ns_plot)
_missed  = _total - _matched
_prop_m  = (_matched / _total).values
_prop_x  = (_missed  / _total).values
_xpos    = list(range(len(_ns_plot)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(_xpos, _prop_m, color="steelblue", label="Matched", width=0.6)
ax.bar(_xpos, _prop_x, bottom=_prop_m,   color="tomato",   label="Missed",  width=0.6)
ax.set_xticks(_xpos)
ax.set_xticklabels(_xtick_labels)
ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
ax.set_ylabel("Proportion of ground-truth roofs")
ax.set_title(f"Matched vs missed roofs per training size ({AREA_NAME}, all roofs)")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.set_ylim(0, 1.07)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for xi, mis in zip(_xpos, _missed.values):
    ax.text(xi, 1.01, str(int(mis)), ha="center", va="bottom", fontsize=7, color="tomato")
ax.text(0.01, 1.03, "missed count ->", ha="left", va="bottom",
        fontsize=7, color="tomato", transform=ax.transAxes)
plt.tight_layout()
plt.show()

# %%
# === [EXP] 4. IoU vs roof size (area_px) - best model, non-cut, non-artifact ===
_d_sz = df_all[(df_all["n"] == _best_n) & ~df_all["is_cut"] & ~df_all["is_artifact"]].copy()
_d_sz["log_area"] = np.log10(_d_sz["area_px"].clip(lower=1))
_log_bins  = np.linspace(_d_sz["log_area"].min(), _d_sz["log_area"].max(), 11)
_d_sz["area_bin"] = pd.cut(_d_sz["log_area"], bins=_log_bins)
_bin_stats = _d_sz.groupby("area_bin", observed=True)["iou_score"].agg(["mean", "count"])
_bin_cx    = [10 ** iv.mid for iv in _bin_stats.index]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(_d_sz["area_px"], _d_sz["iou_score"],
           alpha=0.15, s=8, color="steelblue", label="Individual roofs")
ax.plot(_bin_cx, _bin_stats["mean"],
        "o-", color="tomato", linewidth=2, markersize=6, label="Bin mean IoU")
ax.set_xscale("log")
ax.set_xlabel("Roof area (px, log scale)")
ax.set_ylabel("IoU score (strict: 0 if missed)")
ax.set_title(f"IoU vs roof size (n={_best_n}, non-cut, non-artifact)")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.legend()
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.show()

# %%
# === [EXP] 5. Size-stratified scarcity curve ===
# Tertile bins from the pooled (all-n) non-cut, non-artifact area distribution.
_d_ref       = df_all[~df_all["is_cut"] & ~df_all["is_artifact"]]
_t33, _t67   = float(_d_ref["area_px"].quantile(1/3)), float(_d_ref["area_px"].quantile(2/3))
_size_bins   = [0.0, _t33, _t67, float("inf")]
_size_labels = ["Small", "Medium", "Large"]
_bin_leg     = {
    "Small":  f"Small  (< {_t33:.0f} px)",
    "Medium": f"Medium ({_t33:.0f} - {_t67:.0f} px)",
    "Large":  f"Large  (> {_t67:.0f} px)",
}

_d_strat = df_all[~df_all["is_cut"] & ~df_all["is_artifact"] & df_all["n"].isin(_ns_plot)].copy()
_d_strat["size_bin"] = pd.cut(_d_strat["area_px"], bins=_size_bins,
                               labels=_size_labels, include_lowest=True)
_strat_stats = _d_strat.groupby(["n", "size_bin"], observed=True)["iou_score"].mean().unstack()

_bin_styles = {
    "Small":  {"color": "steelblue",  "marker": "o", "ls": "-"},
    "Medium": {"color": "seagreen",   "marker": "s", "ls": "--"},
    "Large":  {"color": "darkorange", "marker": "^", "ls": "-."},
}
fig, ax = plt.subplots(figsize=(10, 5))
for _bin in _size_labels:
    if _bin in _strat_stats.columns:
        st = _bin_styles[_bin]
        ax.plot(_strat_stats.index, _strat_stats[_bin],
                marker=st["marker"], linestyle=st["ls"], color=st["color"],
                linewidth=1.5, markersize=5, label=_bin_leg[_bin])
ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
ax.set_ylabel("Mean IoU (strict)")
ax.set_title(f"Size-stratified scarcity curve ({AREA_NAME}, non-cut, non-artifact)")
ax.set_xticks(_ns_plot)
ax.set_xticklabels(_xtick_labels)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# === [EXP] 6. Artifact analysis: missed counts per category vs training size ===
# Match rates hover near 0.99 for all groups, so proportions are unreadable.
# Absolute missed counts reveal where the few detection failures actually fall.
_d_exp6    = df_all[df_all["n"].isin(_ns_plot)]
_ref_n     = df_all[df_all["n"] == 0]
_n_std_ref = int((~_ref_n["is_cut"] & ~_ref_n["is_artifact"]).sum())
_n_art_ref = int(_ref_n["is_artifact"].sum())

def _miss_count(grp):
    return int((~grp["matched"]).sum())

_std_missed = (_d_exp6[~_d_exp6["is_artifact"] & ~_d_exp6["is_cut"]]
               .groupby("n").apply(_miss_count).reindex(_ns_plot, fill_value=0))
_art_missed = (_d_exp6[  _d_exp6["is_artifact"]]
               .groupby("n").apply(_miss_count).reindex(_ns_plot, fill_value=0))

_w   = 0.35
_x   = np.arange(len(_ns_plot))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(_x - _w / 2, _std_missed.values, width=_w, color="steelblue",
       label=f"Standard - non-cut, non-artifact (n={_n_std_ref})")
ax.bar(_x + _w / 2, _art_missed.values, width=_w, color="tomato",
       label=f"Artifact (n={_n_art_ref})")
ax.set_xticks(_x)
ax.set_xticklabels(_xtick_labels, rotation=45, ha="right")
ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
ax.set_ylabel("Missed roofs (absolute count)")
ax.set_title(f"Missed roof counts per category vs training size ({AREA_NAME})")
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %%
# === [EXP] 7. IoU quality: standard vs artifact vs cut ===
_d_exp7    = df_all[df_all["n"].isin(_ns_plot)]
_n_cut_ref = int((_ref_n["is_cut"] & ~_ref_n["is_artifact"]).sum())

_n_all_ref = int(len(_ref_n))
_art_iou = _d_exp7[  _d_exp7["is_artifact"]].groupby("n")["iou_score"].mean()
_std_iou = _d_exp7[~_d_exp7["is_artifact"] & ~_d_exp7["is_cut"]].groupby("n")["iou_score"].mean()
_cut_iou = _d_exp7[  _d_exp7["is_cut"] & ~_d_exp7["is_artifact"]].groupby("n")["iou_score"].mean()
_all_iou = _d_exp7.groupby("n")["iou_score"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(_all_iou.index, _all_iou.values, "D-",  color="black",
        linewidth=1.5, markersize=5, label=f"All roofs (n={_n_all_ref})", zorder=5)
ax.plot(_std_iou.index, _std_iou.values, "o-",  color="steelblue",
        linewidth=1.5, markersize=5, label=f"Standard - non-cut, non-artifact (n={_n_std_ref})")
ax.plot(_art_iou.index, _art_iou.values, "s--", color="tomato",
        linewidth=1.5, markersize=5, label=f"Artifact (n={_n_art_ref})")
ax.plot(_cut_iou.index, _cut_iou.values, "^:",  color="seagreen",
        linewidth=1.5, markersize=5, label=f"Cut - non-artifact (n={_n_cut_ref})")
ax.set_xlabel("Training set size (nr of tiles), where 0 = zero-shot")
ax.set_ylabel("Mean IoU (strict)")
ax.set_title(f"IoU quality by roof category vs training size ({AREA_NAME})")
ax.set_xticks(_ns_plot)
ax.set_xticklabels(_xtick_labels, rotation=45, ha="right")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# === [EXP] 8. Size-class proportion per training split (ceu_paz + cantidio_sampaio) ===
# Mirrors EXP 5 but for the training data: for each dataset size n, shows what
# fraction of training instances fall into Small / Medium / Large bins.
# Tests whether the mIoU dips at n=60 and n=120 coincide with a shift in roof-size
# composition of the training set.
# Training uses filter_artifacts=True, filter_cut=False -- matches the computation here.
import random as _random
import json   as _json

_TRAIN_REGIONS = ["ceu_paz", "cantidio_sampaio"]
_png_cache     = os.path.join(CODE_DIR, "output_data", "favela_png")

# Reproduce FavelaDataset tile list (PNG mode, filter_artifacts=True, filter_cut=False)
_tile_seq = []
for _region in _TRAIN_REGIONS:
    _img_dir = os.path.join(_png_cache, "images", _region)
    _msk_dir = os.path.join(_png_cache, "masks",  _region)
    for _tf in sorted(f for f in os.listdir(_img_dir) if f.endswith(".png")):
        _tname     = os.path.splitext(_tf)[0]
        _meta_path = os.path.join(_msk_dir, _tname + "_meta.json")
        if not os.path.exists(_meta_path):
            continue
        _meta = _json.load(open(_meta_path))
        _has_valid = any(
            not _m.get("is_artifact", False) and
            os.path.exists(os.path.join(_msk_dir, _tname, f"{_i:04d}.png"))
            for _i, _m in enumerate(_meta)
        )
        if _has_valid:
            _tile_seq.append(_tname)

_rng = _random.Random(42)
_rng.shuffle(_tile_seq)
_n_train   = int(len(_tile_seq) * 0.75)
_train_seq = _tile_seq[:_n_train]
print(f"Train pool: {_n_train} tiles from {len(_tile_seq)} annotated")

# Build area_px lookup: tile_name -> [area_px, ...] (non-artifact, cut included)
_train_areas = {}
for _region in _TRAIN_REGIONS:
    _csv = os.path.join(BASE_IOU_RESULTS_DIR, f"{_region}_unified_metrics.csv")
    if not os.path.exists(_csv):
        print(f"WARNING: {_csv} not found")
        continue
    _d = pd.read_csv(_csv)
    _d = _d[~_d["is_artifact"].astype(bool)]
    for _row in _d.itertuples(index=False):
        _tname = os.path.splitext(_row.belongs_to)[0]
        _train_areas.setdefault(_tname, []).append(float(_row.area_px))

# --- Bin source ---
# False = tertiles of the training pool (ceu_paz + cantidio_sampaio)
# True  = tertiles of the santa_madalena eval set (same bins as EXP 5)
USE_EVAL_BINS = False

# Training-data tertiles (always computed; used when USE_EVAL_BINS=False)
_all_train_areas = [a for areas in _train_areas.values() for a in areas]
_tr33 = float(np.percentile(_all_train_areas, 100/3))
_tr67 = float(np.percentile(_all_train_areas, 200/3))

# Pick active thresholds
if USE_EVAL_BINS:
    _b33, _b67   = _t33, _t67
    _bin_source  = f"eval set ({AREA_NAME})"
else:
    _b33, _b67   = _tr33, _tr67
    _bin_source  = "training pool (ceu_paz + cantidio_sampaio)"

_exp8_size_labels = ["Small", "Medium", "Large"]
_exp8_bin_leg     = {
    "Small":  f"Small  (< {_b33:.0f} px)",
    "Medium": f"Medium ({_b33:.0f} - {_b67:.0f} px)",
    "Large":  f"Large  (> {_b67:.0f} px)",
}
print(f"Bin source: {_bin_source}")
print(f"Thresholds: p33={_b33:.0f} px, p67={_b67:.0f} px")

# For each n, compute proportion of each size class among training instances
_exp8_ns    = [n for n in DATASET_SIZES if n != 162]
_exp8_props = {lbl: [] for lbl in _exp8_size_labels}

print(f"\n{'n':>5}  {'Small%':>8}  {'Medium%':>8}  {'Large%':>8}  {'instances':>10}")
print("-" * 46)
for _n in _exp8_ns:
    _active = set(_train_seq[:_n])
    _areas  = [a for _t in _active for a in _train_areas.get(_t, [])]
    _total  = len(_areas)
    _small  = sum(1 for a in _areas if a <= _b33)
    _medium = sum(1 for a in _areas if _b33 < a <= _b67)
    _large  = sum(1 for a in _areas if a > _b67)
    _exp8_props["Small"].append(_small  / _total if _total else 0.0)
    _exp8_props["Medium"].append(_medium / _total if _total else 0.0)
    _exp8_props["Large"].append(_large  / _total if _total else 0.0)
    print(f"{_n:>5}  {_small/_total:>8.1%}  {_medium/_total:>8.1%}  {_large/_total:>8.1%}  {_total:>10}")

# %%
_exp8_bin_styles = {
    "Small":  {"color": "steelblue",  "marker": "o", "ls": "-"},
    "Medium": {"color": "seagreen",   "marker": "s", "ls": "--"},
    "Large":  {"color": "darkorange", "marker": "^", "ls": "-."},
}
fig, ax = plt.subplots(figsize=(10, 5))
for _lbl in _exp8_size_labels:
    _st = _exp8_bin_styles[_lbl]
    ax.plot(_exp8_ns, _exp8_props[_lbl],
            marker=_st["marker"], linestyle=_st["ls"], color=_st["color"],
            linewidth=1.5, markersize=5, label=_exp8_bin_leg[_lbl])
ax.set_xlabel("Training set size (nr of tiles)")
ax.set_ylabel("Proportion of training instances")
ax.set_title(
    f"Size-class composition per training split\n"
    f"(ceu_paz + cantidio_sampaio, cut incl. | bins from {_bin_source})"
)
ax.set_xticks(_exp8_ns)
ax.set_xticklabels([str(n) for n in _exp8_ns], rotation=45, ha="right")
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# === [EXP] 9. Training progress curves: n60 and n120 ===
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_pr_ns = [40, 160]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, _n in zip(axes, _pr_ns):
    _tb_dir  = os.path.join(CKPT_BASE_DIR, f"n{_n}", "tensorboard")
    _best_ep = checkpoint_map[_n][0]

    _train_ea = EventAccumulator(os.path.join(_tb_dir, "Loss_train_vs_val_train"))
    _val_ea   = EventAccumulator(os.path.join(_tb_dir, "Loss_train_vs_val_val"))
    _train_ea.Reload()
    _val_ea.Reload()

    _tr_steps = [e.step  for e in _train_ea.Scalars("Loss/train_vs_val")]
    _tr_vals  = [e.value for e in _train_ea.Scalars("Loss/train_vs_val")]
    _vl_steps = [e.step  for e in _val_ea.Scalars("Loss/train_vs_val")]
    _vl_vals  = [e.value for e in _val_ea.Scalars("Loss/train_vs_val")]

    # Find the epoch where val loss is minimum to mark the actual plateau start
    _min_ep = _vl_steps[_vl_vals.index(min(_vl_vals))]

    ax.plot(_tr_steps, _tr_vals, "-",  color="steelblue", linewidth=1.5, label="Train loss")
    ax.plot(_vl_steps, _vl_vals, "-",  color="tomato",    linewidth=1.5, label="Val loss")
    ax.axvline(_best_ep, color="black",   linestyle="--", linewidth=1.2,
               label=f"Selected checkpoint (epoch {_best_ep})")
    ax.axvline(_min_ep,  color="seagreen", linestyle=":",  linewidth=1.2,
               label=f"Val loss minimum (epoch {_min_ep})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"n={_n}  |  best epoch={_best_ep}, val min={_min_ep}")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Training progress - n60 and n120 (train vs val loss)")
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# === [EXP] 10. Mask composition per training region (whole / cut / artifact) ===
# =============================================================================
# For each training region shows how many ground-truth masks are whole (kept by
# all filter settings), cut-only (tile-edge overlap, no artifact), artifact-only,
# or both cut and artifact.  The new unicluster sweep uses filter_artifacts=True,
# so any mask with is_artifact=True is excluded regardless of is_cut status.

_exp10_regions = ["ceu_paz", "cantidio_sampaio", "santa_madalena"]

_exp10_counts = {}
for _reg in _exp10_regions:
    _csv = os.path.join(BASE_IOU_RESULTS_DIR, f"{_reg}_unified_metrics.csv")
    _d   = pd.read_csv(_csv)
    _art = _d["is_artifact"].fillna(False).astype(bool)
    _cut = _d["is_cut"].fillna(False).astype(bool)
    _exp10_counts[_reg] = {
        "whole":         int((~_art & ~_cut).sum()),
        "cut_only":      int((_cut  & ~_art).sum()),
        "artifact_only": int((_art  & ~_cut).sum()),
        "both":          int((_art  &  _cut).sum()),
    }
    _c = _exp10_counts[_reg]
    _n_artifact = _c["artifact_only"] + _c["both"]
    print(
        f"{_reg}: total={len(_d)}\n"
        f"whole = {_c['whole']}\tcut_only = {_c['cut_only']}"
        f"\tartifact_only = {_c['artifact_only']}\tboth = {_c['both']}\n"
        f"=> excluded by artifact filter: {_n_artifact}\n"
    )

_exp10_cats      = ["whole",   "cut_only", "artifact_only",          "both"]
_exp10_nicenames = ["Whole",   "Cut only", "Artifact only (excl.)",  "Cut + artifact (excl.)"]
_exp10_colors    = ["#4caf50", "#2196f3",  "#ff9800",                "#f44336"]

fig, ax = plt.subplots(figsize=(7, 4))
_x10      = np.arange(len(_exp10_regions))
_bottoms10 = np.zeros(len(_exp10_regions))

for _cat, _col, _nice in zip(_exp10_cats, _exp10_colors, _exp10_nicenames):
    _vals = np.array([_exp10_counts[_reg][_cat] for _reg in _exp10_regions], dtype=float)
    ax.bar(_x10, _vals, bottom=_bottoms10, color=_col, label=_nice, edgecolor="white", linewidth=0.5)
    for _xi, (_v, _b) in enumerate(zip(_vals, _bottoms10)):
        if _v >= 10:
            ax.text(_xi, _b + _v / 2, str(int(_v)),
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    _bottoms10 += _vals

ax.set_xticks(_x10)
ax.set_xticklabels(_exp10_regions)
ax.set_ylabel("Number of ground-truth masks")
ax.set_title("Mask composition per region\n(filter_artifacts=True excludes orange + red segments)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# === [EXP] 11. Training dynamics and LR schedule - n40 investigation ===
# =============================================================================
# All runs share identical HPs (base_lr, weight_decay, loss params).
# Only max_tiles differs.  This cell loads per-epoch val stats from the JSON
# logs and reconstructs the warmup+cosine LR schedule to see where each
# model's best checkpoint falls on the curve.
import json as _json_exp11
from matplotlib.lines import Line2D as _Line2D

_EXP11_BASE_LR    = 7.102526614544882e-05
_EXP11_MAX_EPOCHS = 30
_EXP11_HL         = 40   # highlighted n
_EXP11_NS         = sorted(n for n in DATASET_SIZES if n in checkpoint_map)

def _exp11_lr(ep):
    """Reconstruct warmup+cosine LR value at a given epoch (1-indexed)."""
    where = ep / _EXP11_MAX_EPOCHS
    if where <= 0.1:
        return (where / 0.1) * _EXP11_BASE_LR
    t   = (where - 0.1) / 0.9
    end = _EXP11_BASE_LR / 20.0
    return end + (_EXP11_BASE_LR - end) * 0.5 * (1.0 + np.cos(np.pi * t))

_exp11_val = {}
for _n in _EXP11_NS:
    _f = os.path.join(CKPT_BASE_DIR, f"n{_n}", "logs", "val_stats.json")
    if os.path.exists(_f):
        _rows = [_json_exp11.loads(l) for l in open(_f)]
        _exp11_val[_n] = {
            "epoch":    [r["Trainer/epoch"]   for r in _rows],
            "val_loss": [r["Loss/total/val"]   for r in _rows],
            "val_dice": [r["Metrics/val_dice"] for r in _rows],
        }

print(f"{'n':>5}  {'stopped':>7}  {'best_ep':>7}  {'dice@ckpt':>10}  {'lr@ckpt':>12}  {'lr@stop':>12}")
print("-" * 62)
for _n in _EXP11_NS:
    if _n not in _exp11_val:
        continue
    _v       = _exp11_val[_n]
    _stopped = _v["epoch"][-1]
    _bi      = _v["val_dice"].index(max(_v["val_dice"]))
    _best_ep = _v["epoch"][_bi]
    _dice    = _v["val_dice"][_bi]
    print(f"{_n:>5}  {_stopped:>7}  {_best_ep:>7}  {_dice:>10.4f}"
          f"  {_exp11_lr(_best_ep):>12.4e}  {_exp11_lr(_stopped):>12.4e}")

# %%
_exp11_cmap   = plt.cm.Blues
_exp11_n_cols = {_n: _exp11_cmap(0.35 + 0.60 * i / max(1, len(_EXP11_NS) - 1))
                 for i, _n in enumerate(_EXP11_NS)}
_EXP11_HL_COL = "tomato"
_EXP11_CKPT_C = "steelblue"

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax_loss, ax_dice = axes[0]
ax_lr,   ax_ood  = axes[1]

_ep_curve = np.linspace(0.01, _EXP11_MAX_EPOCHS, 400)
_lr_curve = np.array([_exp11_lr(e) for e in _ep_curve])

# Collect ax_lr points; drawn after loop so stacking levels can be pre-computed
_lr_pts = []   # (best_ep, n, is_hl)

for _n in _EXP11_NS:
    if _n not in _exp11_val:
        continue
    _v   = _exp11_val[_n]
    _col = _EXP11_HL_COL if _n == _EXP11_HL else _exp11_n_cols[_n]
    _lw  = 2.0            if _n == _EXP11_HL else 0.9
    _al  = 1.0            if _n == _EXP11_HL else 0.55
    _zo  = 5              if _n == _EXP11_HL else 2
    _ms  = 60             if _n == _EXP11_HL else 22

    ax_loss.plot(_v["epoch"], _v["val_loss"], color=_col, linewidth=_lw, alpha=_al, zorder=_zo)
    ax_dice.plot(_v["epoch"], _v["val_dice"], color=_col, linewidth=_lw, alpha=_al, zorder=_zo)

    _bi      = _v["val_dice"].index(max(_v["val_dice"]))
    _best_ep = _v["epoch"][_bi]
    _hl      = (_n == _EXP11_HL)

    _star_ms = 180 if _hl else 60
    ax_loss.scatter([_best_ep], [_v["val_loss"][_bi]], marker="*", color=_col,
                    s=_star_ms, zorder=_zo + 1,
                    edgecolors="black" if _hl else "none", linewidths=0.6)
    ax_dice.scatter([_best_ep], [_v["val_dice"][_bi]], marker="*", color=_col,
                    s=_star_ms, zorder=_zo + 1,
                    edgecolors="black" if _hl else "none", linewidths=0.6)

    _lr_pts.append((_best_ep, _n, _hl))

    if _n in all_results and all_results[_n]["matched_ious"]:
        _ood = float(np.mean(all_results[_n]["matched_ious"]))
        ax_ood.scatter([_exp11_lr(_best_ep)], [_ood],
                       color=_col, s=_ms, zorder=_zo, marker="o",
                       edgecolors="black" if _hl else "none", linewidths=1.0)
        ax_ood.annotate(str(_n), (_exp11_lr(_best_ep), _ood),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="black" if _hl else _col,
                        fontweight="bold" if _hl else "normal")

# --- ax_lr: lollipop chart ---
# Each stem starts at the actual LR value on the curve and rises a fixed height.
# The ball sits at the top of the stem, so its vertical position reflects the LR
# at that epoch.  When two models share the same epoch the second stem continues
# a small extra step so the balls stay distinct.  Labels to the right of each ball.
from collections import defaultdict as _defaultdict
_lr_pts.sort(key=lambda t: (t[0], t[1]))
_ep_rank = _defaultdict(int)
_STEM_H  = _EXP11_BASE_LR * 0.15   # fixed stem length
_COLL_H  = _EXP11_BASE_LR * 0.13   # tiny extra step per collision rank

_ball_tops = []
for _best_ep, _n, _hl in _lr_pts:
    _rank   = _ep_rank[_best_ep]
    _ep_rank[_best_ep] += 1
    _lr_val = _exp11_lr(_best_ep)
    _ball_y = _lr_val + _STEM_H + _rank * _COLL_H
    _ball_tops.append(_ball_y)
    _col    = _EXP11_HL_COL if _hl else _EXP11_CKPT_C
    # Stem from curve to ball
    ax_lr.plot([_best_ep, _best_ep], [_lr_val, _ball_y],
               color=_col, linewidth=1.5 if _hl else 0.8, alpha=0.7, zorder=2)
    # Ball
    ax_lr.scatter([_best_ep], [_ball_y], color=_col,
                  s=55 if _hl else 22, zorder=3, marker="o",
                  edgecolors="black" if _hl else "none", linewidths=1.0)
    # Label to the right of the ball
    ax_lr.text(_best_ep + 0.25, _ball_y, str(_n), va="center",
               fontsize=8 if _hl else 7,
               color="black" if _hl else "dimgray",
               fontweight="bold" if _hl else "normal")

ax_lr.plot(_ep_curve, _lr_curve, color="gray", linewidth=1.2, alpha=0.5, zorder=1)
ax_lr.set_xlim(left=0)
ax_lr.set_ylim(bottom=0, top=max(_ball_tops) + _EXP11_BASE_LR * 0.18)
ax_lr.set_xlabel("Epoch")
ax_lr.set_ylabel("Learning rate")
ax_lr.set_title("Best-checkpoint epoch on LR schedule")
ax_lr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_lr.grid(True, alpha=0.3)
ax_lr.legend(handles=[
    _Line2D([0], [0], color="gray",            linewidth=1.2,               label="LR schedule (30 ep)"),
    _Line2D([0], [0], marker="o", color=_EXP11_CKPT_C, linewidth=0, markersize=5, label="Best-ckpt epoch"),
    _Line2D([0], [0], marker="o", color=_EXP11_HL_COL,  linewidth=0, markersize=6,
            markeredgecolor="black", label=f"n={_EXP11_HL} (highlighted)"),
], fontsize=7)

ax_ood.set_xlabel("LR at best-checkpoint epoch")
ax_ood.set_ylabel("OOD matched mIoU")
ax_ood.set_title(f"OOD matched mIoU vs. LR at checkpoint ({AREA_NAME})")
ax_ood.grid(True, alpha=0.3)

_ld_legend = [
    _Line2D([0], [0], color=_EXP11_HL_COL, linewidth=2.0,             label=f"n={_EXP11_HL}"),
    _Line2D([0], [0], color="steelblue",   linewidth=0.9, alpha=0.55, label="Other sizes"),
]
for _ax, _title, _ylabel in [
    (ax_loss, "Val loss per epoch (training region)", "Loss/total/val"),
    (ax_dice, "Val dice per epoch (training region)", "Metrics/val_dice"),
]:
    _ax.set_xlabel("Epoch")
    _ax.set_ylabel(_ylabel)
    _ax.set_title(_title)
    _ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _ax.grid(True, alpha=0.3)
    _ax.legend(handles=_ld_legend, fontsize=8)

fig.suptitle(f"Training dynamics per dataset size  (n={_EXP11_HL} highlighted in red)")
plt.tight_layout()
plt.show()

# %%
# =============================================================================
# === [EXP] 12. IoU score vs ground-truth mask area ===
# =============================================================================
# Academic confirmation that smaller masks receive lower IoU scores.
# Only matched, non-cut, non-artifact detections are used so the signal reflects
# segmentation quality only, not detection failure.
# Left:  raw scatter + pooled binned mean +/- 95% CI (all training sizes).
# Right: zero-shot vs best fine-tuned side-by-side, same binning.
from scipy import stats as _scipy_stats

_d12 = df_all[
    df_all["is_cut"] & ~df_all["is_artifact"] & df_all["matched"] & (df_all["area_px"] > 0)
].copy()

# Log-space bins over the pooled area range
_log12      = np.log10(_d12["area_px"])
_N_BINS12   = 12
_log12_edges = np.linspace(_log12.min(), _log12.max(), _N_BINS12 + 1)
_bin12_cx    = np.array([10 ** ((_log12_edges[i] + _log12_edges[i+1]) / 2)
                         for i in range(_N_BINS12)])
_d12["area_bin12"] = pd.cut(_log12, bins=_log12_edges)

def _bin12_stats(sub):
    g   = sub.groupby("area_bin12", observed=False)["iou_score"]
    mn  = g.mean().values
    std = g.std().values
    cnt = g.count().values
    ci  = np.where(cnt > 1, 1.96 * std / np.sqrt(np.maximum(cnt, 1)), np.nan)
    return mn, ci, cnt

_mn_all,  _ci_all,  _cnt_all  = _bin12_stats(_d12)
_mn_zs,   _ci_zs,   _cnt_zs   = _bin12_stats(_d12[_d12["n"] == 0])
_mn_best, _ci_best, _cnt_best = _bin12_stats(_d12[_d12["n"] == _best_n])

_rho_all,  _p_all  = _scipy_stats.spearmanr(_d12["area_px"], _d12["iou_score"])
_rho_zs,   _p_zs   = _scipy_stats.spearmanr(_d12.loc[_d12["n"] == 0,      "area_px"],
                                              _d12.loc[_d12["n"] == 0,      "iou_score"])
_rho_best, _p_best = _scipy_stats.spearmanr(_d12.loc[_d12["n"] == _best_n, "area_px"],
                                              _d12.loc[_d12["n"] == _best_n, "iou_score"])

print(f"Spearman rho (pooled):    rho={_rho_all:.3f},  p={_p_all:.2e},  n={len(_d12)}")
print(f"Spearman rho (zero-shot): rho={_rho_zs:.3f},  p={_p_zs:.2e},  n={((_d12['n']==0).sum())}")
print(f"Spearman rho (n={_best_n}):  rho={_rho_best:.3f},  p={_p_best:.2e},  n={((_d12['n']==_best_n).sum())}")

# %%
_VALID_ALL = ~np.isnan(_mn_all)

fig, (ax12l, ax12r) = plt.subplots(1, 2, figsize=(13, 5))

# --- Left: scatter + pooled binned mean +/- 95% CI ---
_scat12 = _d12.sample(min(4000, len(_d12)), random_state=42)
ax12l.scatter(_scat12["area_px"], _scat12["iou_score"],
              alpha=0.04, s=4, color="steelblue", rasterized=True, zorder=1)
ax12l.fill_between(_bin12_cx[_VALID_ALL],
                   (_mn_all - _ci_all)[_VALID_ALL],
                   (_mn_all + _ci_all)[_VALID_ALL],
                   alpha=0.25, color="tomato", zorder=2)
ax12l.plot(_bin12_cx[_VALID_ALL], _mn_all[_VALID_ALL],
           "o-", color="tomato", linewidth=1.8, markersize=5, zorder=3,
           label="Binned mean +/- 95% CI")
ax12l.set_xscale("log")
ax12l.set_xlabel("Ground-truth mask area (px, log scale)")
ax12l.set_ylabel("IoU score (matched detections only)")
ax12l.set_title("IoU vs mask area (pooled, all training sizes)")
ax12l.text(0.97, 0.05,
           f"Spearman rho={_rho_all:.3f}  (p={_p_all:.1e},  n={len(_d12)})",
           transform=ax12l.transAxes, fontsize=8, ha="right",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor="lightgray", alpha=0.9))
ax12l.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax12l.legend(fontsize=8)
ax12l.grid(True, alpha=0.3, which="both")

# --- Right: zero-shot vs best fine-tuned ---
for _mn, _ci, _cnt, _lbl, _col, _rho, _p in [
    (_mn_zs,   _ci_zs,   _cnt_zs,
     "Zero-shot", "steelblue", _rho_zs, _p_zs),
    (_mn_best, _ci_best, _cnt_best,
     f"Fine-tuned (n={_best_n})", "tomato", _rho_best, _p_best),
]:
    _vld = ~np.isnan(_mn)
    ax12r.fill_between(_bin12_cx[_vld], (_mn - _ci)[_vld], (_mn + _ci)[_vld],
                       alpha=0.2, color=_col)
    ax12r.plot(_bin12_cx[_vld], _mn[_vld], "o-", color=_col,
               linewidth=1.8, markersize=5,
               label=f"{_lbl}  (rho={_rho:.3f}, p={_p:.1e})")
    for _cx, _m, _c in zip(_bin12_cx[_vld], _mn[_vld], _cnt[_vld]):
        if _c >= 5:
            ax12r.text(_cx, _m + 0.012, str(int(_c)),
                       ha="center", va="bottom", fontsize=6, color=_col, alpha=0.75)

ax12r.set_xscale("log")
ax12r.set_xlabel("Ground-truth mask area (px, log scale)")
ax12r.set_ylabel("IoU score (matched detections only)")
ax12r.set_title(f"IoU vs mask area: zero-shot vs fine-tuned ({AREA_NAME})")
ax12r.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax12r.legend(fontsize=8)
ax12r.grid(True, alpha=0.3, which="both")

fig.suptitle("IoU decreases for smaller ground-truth masks (non-artifact, matched only)")
plt.tight_layout()
plt.show()

# %%
