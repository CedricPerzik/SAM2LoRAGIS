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

# %% [markdown]
# # D3 - Final thesis figures (experimental graph section)
#
# This notebook rebuilds the "EXPERIMENTAL" graph section of `D2_dataset_iou.py`
# into a set of clean, independent, click-through-safe blocks.
#
# It does **not** re-run inference or IoU comparison - all metrics were already
# computed by `D2_dataset_iou.py` (via `D_functions.run_evaluation_pipeline_concurrent`
# / `export_unified_metrics_to_csv_concurrent`) and are read here straight from the
# per-run CSVs under `output_data/iou_results/`. This notebook is purely about
# loading that data and producing final, thesis-ready figures.
#
# Every figure is exported as PDF (vector, for LaTeX inclusion) + PNG (300 dpi
# raster preview) into `output_data/figures/exp_graphs/`.
#
# Five independent fine-tuning runs are available for every dataset size, each
# with its own `split_seed` (passed to `compute_split(tiles, train_fraction,
# test_fraction, split_seed)` - see `training/dataset/split_utils.py` on the
# cluster checkout). Each run therefore has **both** a different train/val/test
# tile split *and* a different model-init draw - there is no separate "model
# seed": `config.yaml`'s `seed_value` and the SLURM `SEED=` env var are the same
# number as `split_seed`.
#
# | run     | split_seed |
# |---------|------------|
# | run_007 | 111        |
# | run_008 | 222        |
# | run_009 | 333        |
# | run_010 | 444        |
# | run_011 | 555        |
#
# The split is train-50% / val-25% / test-25% of a 205-tile annotated pool
# (`ceu_paz` + `cantidio_sampaio`). `ceu_paz`/`cantidio_sampaio` therefore now
# serve double duty: source of training tiles *and*, via their held-out test
# subset, an **in-distribution (ID)** evaluation area to set against the
# **out-of-distribution (OOD)** `santa_madalena` evaluation (which is never
# split - the whole area is always held out).
#
# Wherever a figure summarizes model behaviour (not the dataset itself), all 5
# runs are used and run-to-run variation is shown explicitly (mean +/- min/max
# band, or all runs overlaid). EXP8 and EXP10 describe properties of the
# *dataset* rather than of a trained model; EXP10 stays single-seed by design
# (pure annotation counts, no split involved), but EXP8 now varies per run too
# (each run's train split differs) - see the note in that block.

# %%
import os
import re
import sys
import glob
import json
import functools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats
from adjustText import adjust_text

# %% [markdown]
# ## Configuration

# %%
# --- Dataset sizes used in the scarcity sweep ---
# 108 is a *requested* size, capped to ~102 actual train tiles (train_fraction
# 0.5 of the 205-tile pool, after val/test are carved out) - see the
# "[dataset split] wanted -> true tiles: train 108 -> 102 CAPPED" line in the
# SLURM logs. It's effectively "full dataset, given this split", the new
# analogue of the old sweep's n=162.
DATASET_SIZES = [5, 10, 15, 25, 40, 65, 108]

# --- Out-of-distribution evaluation area (never split - whole area held out) ---
AREA_NAME = "santa_madalena"
REAL_AREA_NAME = "Rio Claro II" # Real name of santa madalena region, confusion in original communication.

# --- In-distribution evaluation areas: former "training-only" regions, now ---
# --- also source their own held-out test split per run (see split_utils). ---
ID_REGIONS = TRAIN_REGIONS = ["ceu_paz", "cantidio_sampaio"]
TRAIN_FRACTION = 0.5
TEST_FRACTION  = 0.25

# --- The 5 independent fine-tuning runs, and their split_seed ---
RUNS = {
    "run_007": 111,
    "run_008": 222,
    "run_009": 333,
    "run_010": 444,
    "run_011": 555,
}
RUN_NAMES = sorted(RUNS)  # deterministic iteration order

# --- Path resolution ---
CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR

IOU_RESULTS_DIR = os.path.join(CODE_DIR, "output_data", "iou_results")
RUNS_IOU_BASE   = os.path.join(IOU_RESULTS_DIR, "runs")
ZEROSHOT_CSV    = os.path.join(IOU_RESULTS_DIR, f"{AREA_NAME}_unified_metrics.csv")

# Ground-truth pixel masks (used by EXP13/EXP14 to recover each roof's
# position within its tile - identical across every run/size, for both the
# OOD area and the two ID regions).
GT_DIR = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", AREA_NAME)
GT_DIRS_ID = {r: os.path.join(CODE_DIR, "output_data", "ground_truth_npz", r) for r in ID_REGIONS}

# Per-run training logs/checkpoints/tensorboard (rsync'd from SLURM), used by
# the training-dynamics blocks (EXP9, EXP11).
RUNS_LOG_BASE = "/home/ced/Documents/unicluster/sam2loraboracluster/sam2/sam2_logs"

# Cached PNG tiles/masks for the training regions (EXP8, and the train/val/test
# split reconstruction below).
FAVELA_PNG_DIR = os.path.join(CODE_DIR, "output_data", "favela_png")

FIGURES_DIR = os.path.join(CODE_DIR, "output_data", "figures", "exp_graphs")
os.makedirs(FIGURES_DIR, exist_ok=True)

# `compute_split`/`list_annotated_tiles_png` are the exact stdlib-only helpers
# the training pipeline itself uses to build the train/val/test split from a
# split_seed - vendored on the cluster checkout, imported here (not
# reimplemented) so the reconstructed split is guaranteed identical to what
# each run actually trained/tested on.
sys.path.insert(0, "/home/ced/Documents/unicluster/sam2loraboracluster/sam2/training/dataset")
from split_utils import compute_split, list_annotated_tiles_png  # noqa: E402


@functools.lru_cache(maxsize=None)
def get_split_tiles(run):
    """{'train': [...], 'val': [...], 'test': [...]} of tile *paths* for `run`,
    reconstructed from its split_seed - byte-identical to that run's actual
    training split (verified against test_tiles.json manifests)."""
    tile_paths = list_annotated_tiles_png(ID_REGIONS, FAVELA_PNG_DIR)
    return compute_split(tile_paths, TRAIN_FRACTION, TEST_FRACTION, RUNS[run])


@functools.lru_cache(maxsize=None)
def get_test_tiles(run):
    """{region: set(tile_name)} of `run`'s held-out ID test tiles (no extension)."""
    out = {r: set() for r in ID_REGIONS}
    for tile_path in get_split_tiles(run)["test"]:
        region = os.path.basename(os.path.dirname(tile_path))
        tile_name = os.path.splitext(os.path.basename(tile_path))[0]
        out.setdefault(region, set()).add(tile_name)
    return out


print(f"Code dir:     {CODE_DIR}")
print(f"Runs IoU dir: {RUNS_IOU_BASE}")
print(f"Figures dir:  {FIGURES_DIR}")
_id_test_counts = []
for _r in RUN_NAMES:
    _tt = get_test_tiles(_r)
    _total = sum(len(v) for v in _tt.values())
    _id_test_counts.append(_total)
    print(f"{_r} (seed {RUNS[_r]}): test tiles -> " + ", ".join(f"{k}={len(v)}" for k, v in _tt.items())
          + f"  (total={_total})")

# Every ID panel's "n=<size>" refers to *training*-set size, not eval-set size
# - unlike OOD, whose fixed holdout area makes "n" unambiguous. ID inference
# always runs on that run's much smaller held-out test split instead, so every
# ID plot title spells that out explicitly. int(205 * 0.25) = 51 is the same
# for every split_seed regardless of shuffle order, so this is a fixed count,
# not a real min/max range - min()==max() below just guards against it ever
# not being one (e.g. if the annotated pool size changes later).
ID_TEST_TILES_STR = (str(_id_test_counts[0]) if min(_id_test_counts) == max(_id_test_counts)
                      else f"{min(_id_test_counts)}-{max(_id_test_counts)}")
print(f"ID test-split size: {ID_TEST_TILES_STR} tiles/run (out of a 205-tile pool, test_fraction=0.25)")


# %% [markdown]
# ## Shared helpers
#
# - `run_csv_path` / `load_run_df` - locate and load a single (run, n, area) metrics CSV.
# - `load_all_runs_df` - the OOD (`santa_madalena`) long dataframe: every (run, n)
#   CSV + the single deterministic zero-shot CSV at n=0.
# - `load_all_runs_df_id` - the pooled in-distribution (`ceu_paz` + `cantidio_sampaio`)
#   long dataframe: every (run, n) CSV (already restricted to that run's own
#   test-split tiles) + each run's own *filtered* zero-shot subset at n=0
#   (`load_id_zeroshot_row`). Because every run's test split differs, ID
#   zero-shot genuinely varies run-to-run - unlike OOD's single deterministic
#   point - so n=0 behaves like any other n here and needs no special-casing
#   downstream.
# - Both dataframes share the same schema (`n`, `run`, `seed`, + all original
#   columns), so every plotting helper below works unmodified on either one.
# - `mean_band` - collapse a (n_runs x n_x) array into mean/lo/hi across runs,
#   used everywhere we want to show run-to-run variation as a shaded band.
# - `export_figure` - save a figure as PDF + PNG with thesis-appropriate sizing.

# %%
def run_csv_path(run, n, area=AREA_NAME):
    return os.path.join(RUNS_IOU_BASE, run, f"n{n}", f"n{n}_{area}_unified_metrics.csv")


def load_run_df(run, n, area=AREA_NAME):
    """Load one (run, n, area) unified-metrics CSV, or None if it doesn't exist."""
    path = run_csv_path(run, n, area)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d["n"]    = n
    d["run"]  = run
    d["seed"] = RUNS[run]
    return d


def _normalize_metrics_df(df):
    """Fill/cast the handful of columns every downstream helper relies on."""
    df["iou_score"]   = df["iou_score"].fillna(0.0)
    df["matched"]     = df["matched"].astype(bool)
    df["is_cut"]      = df["is_cut"].fillna(False).astype(bool)
    df["is_artifact"] = df["is_artifact"].fillna(False).astype(bool)
    return df


def load_all_runs_df(sizes=DATASET_SIZES):
    """Concatenate every (run, n) OOD CSV plus the zero-shot CSV into one long dataframe.

    Each row is one predicted-vs-ground-truth roof match. For a given n there are
    up to len(RUNS) rows per ground-truth roof (one per run) plus the zero-shot
    rows at n=0 (zero-shot is deterministic, so only one "run").
    """
    frames = []
    for run in RUN_NAMES:
        for n in sizes:
            d = load_run_df(run, n, AREA_NAME)
            if d is None:
                print(f"{run} n{n}: CSV not found - skipping")
                continue
            frames.append(d)

    zs = pd.read_csv(ZEROSHOT_CSV)
    zs["n"], zs["run"], zs["seed"] = 0, "zero_shot", None
    frames.append(zs)

    return _normalize_metrics_df(pd.concat(frames, ignore_index=True))


def load_id_zeroshot_row(run):
    """Zero-shot ID baseline for `run`: filter each region's full top-level
    zero-shot CSV (real zero-shot predictions on every annotated tile in that
    region, from before any train/val/test split existed) down to just
    `run`'s own held-out test tiles, tagged n=0.
    """
    test_tiles = get_test_tiles(run)
    frames = []
    for region in ID_REGIONS:
        d = pd.read_csv(os.path.join(IOU_RESULTS_DIR, f"{region}_unified_metrics.csv"))
        tile_names = d["belongs_to"].map(lambda p: os.path.splitext(p)[0])
        frames.append(d[tile_names.isin(test_tiles[region])])
    out = pd.concat(frames, ignore_index=True)
    out["n"], out["run"], out["seed"] = 0, run, RUNS[run]
    return out


def load_all_runs_df_id(sizes=DATASET_SIZES):
    """Concatenate every (run, n) ID CSV - already restricted to that run's own
    test-split tiles, pooling ceu_paz + cantidio_sampaio - plus each run's own
    filtered zero-shot subset at n=0, into one long dataframe with the same
    schema as `load_all_runs_df`.
    """
    frames = []
    for run in RUN_NAMES:
        frames.append(load_id_zeroshot_row(run))
        for n in sizes:
            for region in ID_REGIONS:
                d = load_run_df(run, n, region)
                if d is None:
                    print(f"{run} n{n} {region}: CSV not found - skipping")
                    continue
                frames.append(d)

    return _normalize_metrics_df(pd.concat(frames, ignore_index=True))


def mean_band(arr2d):
    """(n_runs x n_x) array -> (mean, lo, hi) across runs, NaN-safe."""
    return np.nanmean(arr2d, axis=0), np.nanmin(arr2d, axis=0), np.nanmax(arr2d, axis=0)


def stack_by_run(per_run_dict, x_values):
    """{run: {x: value}} -> (n_runs x n_x) array, aligned to x_values, run order = RUN_NAMES."""
    return np.array([[per_run_dict.get(r, {}).get(x, np.nan) for x in x_values] for r in RUN_NAMES])


# --- Figure sizing: tuned to sit at full thesis text width without rescaling ---
THESIS_WIDTH_IN = 6.3   # typical single-column thesis \textwidth
SINGLE_FIGSIZE  = (THESIS_WIDTH_IN, 3.8)   # one axes
DOUBLE_FIGSIZE  = (THESIS_WIDTH_IN, 3.4)   # 1x2 axes (side by side)
BASE_FONTSIZE   = 9

# Manual override for plot_miou_vs_size (01/02), the size-stratified scarcity
# curve (07), and IoU quality by category (09): those panels carry more x-tick
# labels ("zs" + every DATASET_SIZES value) than other DOUBLE_FIGSIZE plots, so
# the shared 6.3in width leaves them cramped. MIOU_FIGSIZE controls the overall
# exported image size in inches;
# MIOU_WSPACE controls how much of that width goes to the inter-panel gap vs.
# the two axes themselves (smaller = wider graphs, more whitespace squeezed
# out). Both are independent of THESIS_WIDTH_IN/DOUBLE_FIGSIZE, which stay
# tuned to the thesis's actual \textwidth for every other figure - widening
# those would make LaTeX rescale (and shrink) every other figure in the
# document when placed at \textwidth.
MIOU_WIDTH_IN = 9.5   # widen this to make 01/02/07/09 wider than every other figure

# Because MIOU_WIDTH_IN is wider than THESIS_WIDTH_IN, LaTeX shrinks this
# figure by MIOU_FONT_SCALE when it's placed at \textwidth alongside every
# other figure in the document (all exported at THESIS_WIDTH_IN, so they need
# no shrinking). Fonts/markers/lines are absolute point sizes, so without
# compensation that shrink makes this figure's text/markers read smaller than
# every other figure's - not a bug in the numbers, just uncorrected scaling.
# Bumping BASE_FONTSIZE-derived sizes by this same ratio up front cancels it
# back out so the printed result matches everything else.
#
# Critically, MIOU_FIGSIZE's *height* must scale by that same factor too, not
# just the width - scaling fonts/markers up while leaving the height alone
# (an earlier version of this) blows up the margins tight_layout reserves for
# labels/legend/title inside a canvas that didn't grow to match, which is what
# squeezed the plotting area, forced the legend into the data, and let the
# now-oversized markers/lines swallow the gap between the OOD/ID curves. A
# uniform scale (width and height together) keeps the exact same proportions
# as DOUBLE_FIGSIZE, just rendered bigger - so it looks identical to every
# other figure's layout once LaTeX shrinks it back down.
MIOU_FONT_SCALE = MIOU_WIDTH_IN / THESIS_WIDTH_IN
MIOU_FIGSIZE    = (MIOU_WIDTH_IN, DOUBLE_FIGSIZE[1] * MIOU_FONT_SCALE)
MIOU_WSPACE     = 0.28   # fraction of avg axes width reserved as inter-panel gap
                          # (both panels show their own y-tick numbers, so the
                          # gap needs more room than a bare axes boundary)

THESIS_RC = {
    "font.size":       BASE_FONTSIZE,
    "axes.titlesize":  BASE_FONTSIZE + 1,
    "axes.labelsize":  BASE_FONTSIZE,
    "xtick.labelsize": BASE_FONTSIZE - 1,
    "ytick.labelsize": BASE_FONTSIZE - 1,
    "legend.fontsize": BASE_FONTSIZE - 1,
    "figure.dpi":      120,   # on-screen preview only; export_figure controls file DPI
}

# THESIS_RC with every font size scaled up by MIOU_FONT_SCALE, so text prints
# at the same size as every other figure once LaTeX shrinks MIOU_FIGSIZE back
# down to THESIS_WIDTH_IN. Only the font.* / *tick.labelsize / legend.fontsize
# keys are meaningful to scale this way; figure.dpi is passed through as-is.
MIOU_RC = {
    **THESIS_RC,
    "font.size":       BASE_FONTSIZE * MIOU_FONT_SCALE,
    "axes.titlesize":  (BASE_FONTSIZE + 1) * MIOU_FONT_SCALE,
    "axes.labelsize":  BASE_FONTSIZE * MIOU_FONT_SCALE,
    "xtick.labelsize": (BASE_FONTSIZE - 3) * MIOU_FONT_SCALE,
    "ytick.labelsize": (BASE_FONTSIZE - 1) * MIOU_FONT_SCALE,
    "legend.fontsize": (BASE_FONTSIZE - 1) * MIOU_FONT_SCALE,
}


def export_figure(fig, name):
    """Save fig as PDF (vector, for LaTeX) and PNG (300dpi raster preview)."""
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Exported {name}.{{pdf,png}} -> {FIGURES_DIR}")


# %% [markdown]
# ## Load the combined multi-run dataframes
#
# This reads every already-computed CSV (no re-evaluation).
# - `df_all` (OOD, `santa_madalena`): one row per (ground-truth roof, run) pair,
#   for every dataset size, plus the single deterministic zero-shot rows at n=0.
# - `df_id_all` (ID, pooled `ceu_paz` + `cantidio_sampaio`): same shape, but the
#   zero-shot rows at n=0 vary per run too (each run's test split differs).

# %%
df_all    = load_all_runs_df()
df_id_all = load_all_runs_df_id()

_ns_plot      = [n for n in DATASET_SIZES] + [0]
_ns_plot      = sorted(set(_ns_plot))
_xtick_labels = ["zs" if n == 0 else str(n) for n in _ns_plot]

print(f"df_all (OOD):    {len(df_all)} rows    | sizes: {sorted(df_all['n'].unique())} | runs: {RUN_NAMES}")
print(f"df_id_all (ID):  {len(df_id_all)} rows | sizes: {sorted(df_id_all['n'].unique())} | runs: {RUN_NAMES}")

# OOD's holdout area is never split, so every run/n evaluates the exact same
# fixed set of tiles - unlike ID_TEST_TILES_STR, this is a single count, not a
# per-run range. groupby(n) (pooling all 5 runs' rows at each n) just guards
# that assumption instead of assuming it: nunique()==1 across every n means
# the tile set genuinely never changes.
_ood_tile_counts = df_all.groupby("n")["belongs_to"].nunique()
assert _ood_tile_counts.nunique() == 1, f"OOD tile count varies by n: {_ood_tile_counts.to_dict()}"
OOD_N_TILES = str(_ood_tile_counts.iloc[0])
print(f"OOD tile count: {OOD_N_TILES} tiles (fixed holdout area, same {REAL_AREA_NAME} tiles at every n)")

# %% [markdown]
# ## Block 1+2: mIoU vs. training-set size (headline summary plots)
#
# Two variants, both across all 5 runs (mean line, min/max band across runs),
# each rendered as two side-by-side panels - matched mIoU (found roofs only)
# left, strict mIoU (incl. missed roofs) right - with OOD (`santa_madalena`)
# and in-distribution (pooled `ceu_paz` + `cantidio_sampaio`) overlaid on both
# panels, so the OOD-vs-ID contrast is read directly off each panel instead of
# being split across two:
# 1. All roofs.
# 2. Cut rooftops (tile-boundary clipped) excluded, since a clipped ground-truth
#    polygon caps the best achievable IoU regardless of prediction quality.
#
# "Matched" mIoU only averages over ground-truth roofs the model actually found;
# "strict" mIoU scores a missed roof as 0, so it also penalizes recall.
#
# Zero-shot (n=0) is always a single point with no run-to-run band, computed
# separately from `compute_miou_per_run` and prepended - it's one deterministic
# model pass (no training, no seed), so it cannot have a run-to-run band on
# principle. This holds for both domains: the OOD area is never split, so
# there's only one zero-shot evaluation to begin with; the ID zero-shot rows
# are each run's own filtered test-tile subset, but that's still the *same*
# zero-shot pass, just evaluated on different tiles per run - pooling those 5
# subsets into one number (`compute_miou_zeroshot`, which ignores the `run`
# column entirely) is what "combine into one datapoint" means. Feeding n=0
# through `compute_miou_per_run` instead - keying on each run's real `run` tag
# - would treat "which tiles got pooled" as if it were model variance and draw
# a spurious band; that was an earlier bug in this notebook, since fixed.

# %%
def compute_miou_per_run(df, sizes, cut_excluded=False):
    """{run: {n: {"matched": ..., "strict": ...}}} of per-(run, n) mIoU."""
    out = {r: {} for r in RUN_NAMES}
    for run in RUN_NAMES:
        for n in sizes:
            sub = df[(df["run"] == run) & (df["n"] == n)]
            if cut_excluded:
                sub = sub[~sub["is_cut"]]
            if sub.empty:
                continue
            strict  = float(sub["iou_score"].mean())
            matched = sub.loc[sub["matched"], "iou_score"]
            out[run][n] = {
                "matched": float(matched.mean()) if len(matched) else float("nan"),
                "strict":  strict,
            }
    return out


def compute_miou_zeroshot(df, cut_excluded=False):
    """The single pooled zero-shot point for either domain - deliberately
    ignores the `run` column and just pools every n=0 row, so it works
    identically whether n=0 has one deterministic run (OOD) or 5 runs' worth
    of filtered-but-still-zero-shot rows (ID); either way there is exactly
    one model pass behind it, hence exactly one number, never a band."""
    sub = df[df["n"] == 0]
    if cut_excluded:
        sub = sub[~sub["is_cut"]]
    strict  = float(sub["iou_score"].mean())
    matched = float(sub.loc[sub["matched"], "iou_score"].mean())
    return matched, strict


def _miou_arrays(per_run, sizes):
    """per_run (compute_miou_per_run output) + sizes -> mean/lo/hi arrays for
    both the matched and strict mIoU series, aligned to `sizes`."""
    m_arr = stack_by_run({r: {n: v["matched"] for n, v in d.items()} for r, d in per_run.items()}, sizes)
    s_arr = stack_by_run({r: {n: v["strict"]  for n, v in d.items()} for r, d in per_run.items()}, sizes)
    return mean_band(m_arr) + mean_band(s_arr)   # (m_mean, m_lo, m_hi, s_mean, s_lo, s_hi)


def _draw_miou_panel(ax, plot_x_ood, ood_mean, ood_lo, ood_hi, ood_label,
                      plot_x_id, id_mean, id_lo, id_hi, id_label, title,
                      legend_loc="best"):
    """One metric (matched or strict) with OOD and ID overlaid, so the
    OOD-vs-ID contrast is the thing the panel reads out directly."""
    plot_x = sorted(set(plot_x_ood) | set(plot_x_id))
    plot_x_labels = ["zs" if x == 0 else str(x) for x in plot_x]
    lw, ms = 1.5 * MIOU_FONT_SCALE, 5 * MIOU_FONT_SCALE
    ax.fill_between(plot_x_ood, ood_lo, ood_hi, alpha=0.2, color="steelblue")
    ax.plot(plot_x_ood, ood_mean, "o-", color="steelblue", linewidth=lw, markersize=ms,
            label=ood_label)
    ax.fill_between(plot_x_id, id_lo, id_hi, alpha=0.2, color="tomato")
    ax.plot(plot_x_id, id_mean, "s--", color="tomato", linewidth=lw, markersize=ms,
            label=id_label)
    ax.set_xlabel("Training-set size (tiles), zs = zero-shot")
    ax.set_title(title)
    ax.set_xticks(plot_x)
    ax.set_xticklabels(plot_x_labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    # sharey=True (needed so both panels use the same y-scale) hides tick
    # labels on every axes but the first by default - force them back on so
    # the strict panel is readable without cross-referencing the matched one.
    ax.tick_params(labelleft=True)
    ax.legend(fontsize=(BASE_FONTSIZE - 2) * MIOU_FONT_SCALE, loc=legend_loc)
    ax.grid(True, alpha=0.3)


def _panel_arrays(df, cut_excluded):
    """Zero-shot (pooled, single point, no band) prepended to per-run
    fine-tuned mean/min/max bands. Identical computation for OOD and ID -
    `compute_miou_zeroshot` pools n=0 regardless of `run` tag, and
    `compute_miou_per_run` is only ever asked about the real (non-zero)
    dataset sizes, so it never sees n=0 at all."""
    zs_matched, zs_strict = compute_miou_zeroshot(df, cut_excluded=cut_excluded)
    per_run = compute_miou_per_run(df, DATASET_SIZES, cut_excluded=cut_excluded)
    sizes   = sorted({n for r in per_run.values() for n in r})
    m_mean, m_lo, m_hi, s_mean, s_lo, s_hi = _miou_arrays(per_run, sizes)
    # Numeric x so spacing reflects true dataset-size magnitude (e.g. 5->10 much
    # closer than 65->108); only the *displayed* tick label for x=0 becomes "zs".
    plot_x = [0] + sizes
    m_mean, m_lo, m_hi = (np.concatenate([[zs_matched], a]) for a in (m_mean, m_lo, m_hi))
    s_mean, s_lo, s_hi = (np.concatenate([[zs_strict],  a]) for a in (s_mean, s_lo, s_hi))
    return plot_x, m_mean, m_lo, m_hi, s_mean, s_lo, s_hi, per_run, (zs_matched, zs_strict)


def plot_miou_vs_size(cut_excluded, filename, title_suffix):
    plot_x_ood, m_mean, m_lo, m_hi, s_mean, s_lo, s_hi, per_run_ood, zs_ood = \
        _panel_arrays(df_all, cut_excluded)
    plot_x_id, m_mean_id, m_lo_id, m_hi_id, s_mean_id, s_lo_id, s_hi_id, per_run_id, zs_id = \
        _panel_arrays(df_id_all, cut_excluded)

    ood_label = f"OOD ({OOD_N_TILES} tiles/run)"
    id_label  = f"ID ({ID_TEST_TILES_STR} tiles/run)"

    with plt.rc_context(MIOU_RC):
        fig, (ax_matched, ax_strict) = plt.subplots(1, 2, figsize=MIOU_FIGSIZE, sharey=True)
        _draw_miou_panel(ax_matched, plot_x_ood, m_mean, m_lo, m_hi, ood_label,
                          plot_x_id, m_mean_id, m_lo_id, m_hi_id, id_label,
                          "Matched mIoU (found roofs only)")
        _draw_miou_panel(ax_strict, plot_x_ood, s_mean, s_lo, s_hi, ood_label,
                          plot_x_id, s_mean_id, s_lo_id, s_hi_id, id_label,
                          "Strict mIoU (incl. missed roofs)", legend_loc="lower right")
        ax_matched.set_ylabel("Mean IoU (mIoU)")
        fig.suptitle(f"{title_suffix} ({len(RUN_NAMES)} runs)")
        fig.tight_layout()
        fig.subplots_adjust(wspace=MIOU_WSPACE)
        export_figure(fig, filename)
        plt.show()

    return per_run_ood, zs_ood, per_run_id


# %%
per_run_all, zs_all, per_run_all_id = plot_miou_vs_size(
    cut_excluded=False,
    filename="01_miou_vs_trainingsize_all_roofs",
    title_suffix="Data-scarcity sweep - all roofs",
)

# %%
per_run_cut, zs_cut, per_run_cut_id = plot_miou_vs_size(
    cut_excluded=True,
    filename="02_miou_vs_trainingsize_cut_excluded",
    title_suffix="Data-scarcity sweep - cut rooftops excluded",
)

# %% [markdown]
# ### "Best" fine-tuned size
#
# The deep-dive blocks below (EXP1, EXP4, EXP12) contrast zero-shot against
# "the best fine-tuned model." That has to mean the dataset size with the
# highest actual mean matched mIoU across the 5 runs - not just the largest
# dataset size in the sweep, which is what an earlier version of this
# notebook (D2) mistakenly used (`max(n != 162)`, i.e. "biggest sub-full-
# dataset n," regardless of whether it scored best). Picking the true argmax
# over all sizes, including the full-dataset case n=108.
#
# Computed on the **Standard subset** (non-cut, non-artifact roofs) - matching
# `D4_performance_scores_tvt.py`'s `best_n_by_matched_iou`, not the "all
# roofs" data behind the Block 1/2 plots just above. Cut roofs cap the best
# achievable IoU regardless of model quality (the same reasoning that makes
# "Standard" the primary table in D4), so letting a cut-heavy size look
# artificially better/worse would pick a "best n" that doesn't actually
# reflect which model is best. (`per_run_all`/`per_run_all_id` above stay on
# "all roofs" - that's a deliberate, separate view of the *data*, not the
# criterion for picking a best model.)
#
# OOD and ID each get their own "best n" (`_best_n`, `_best_n_id`) - the two
# domains can favor different sizes, and conflating them would misrepresent
# whichever domain a given deep-dive panel is actually showing.

# %%
def best_n_by_matched_iou(df, sizes=DATASET_SIZES):
    """Mean matched mIoU per n, standard (non-cut, non-artifact) roofs only,
    averaged across the 5 runs - identical selection rule to D4's
    `best_n_by_matched_iou`."""
    matched_iou_by_n = {
        n: df[(df["run"].isin(RUN_NAMES)) & (df["n"] == n) & ~df["is_cut"] & ~df["is_artifact"] & df["matched"]]
            .groupby("run")["iou_score"].mean().mean()
        for n in sizes
    }
    return max(matched_iou_by_n, key=matched_iou_by_n.get), matched_iou_by_n


_best_n,    _mean_matched_by_n    = best_n_by_matched_iou(df_all)
_best_n_id, _mean_matched_by_n_id = best_n_by_matched_iou(df_id_all)

print(f"{'n':>5}  {'OOD mean matched mIoU':>24}  {'ID mean matched mIoU':>24}")
for n in DATASET_SIZES:
    ood_marker = " <-best" if n == _best_n else "       "
    id_marker  = " <-best" if n == _best_n_id else "       "
    print(f"{n:>5}  {_mean_matched_by_n[n]:>17.4f}{ood_marker}  {_mean_matched_by_n_id[n]:>17.4f}{id_marker}")
print(f"\n'Best' fine-tuned size (standard subset): OOD n={_best_n}, ID n={_best_n_id}")

# %% [markdown]
# ## EXP1: Cut vs. non-cut IoU distribution
#
# Strict IoU (0 for a missed roof) for zero-shot vs. the best fine-tuned size,
# split into non-cut / cut / all ground-truth roofs. The fine-tuned panel pools
# individual detections from all 5 runs (zero-shot has only one deterministic
# run for OOD, so its panel is n=1; for ID zero-shot already pools all 5 runs'
# filtered test-tile subsets).
#
# Both panel slots are already used (zero-shot | fine-tuned), so OOD and ID get
# their own separate exports rather than being crammed into one 4-panel figure.

# %%
def plot_iou_distribution(df, best_n, title, filename):
    panels = [(0, "Zero-shot"), (best_n, f"Fine-tuned (n={best_n}, {len(RUN_NAMES)} runs pooled)")]

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=True)

        for ax, (n, panel_title) in zip(axes, panels):
            d       = df[df["n"] == n]
            noncut  = d[~d["is_cut"]]["iou_score"].values
            cut     = d[ d["is_cut"]]["iou_score"].values
            allv    = d["iou_score"].values
            nc_mr   = d[~d["is_cut"]]["matched"].mean()
            cut_mr  = d[ d["is_cut"]]["matched"].mean()
            all_mr  = d["matched"].mean()

            parts = ax.violinplot([noncut, cut, allv], positions=[0, 1, 2], showmedians=True, widths=0.7)
            parts["bodies"][0].set_facecolor("steelblue");    parts["bodies"][0].set_alpha(0.7)
            parts["bodies"][1].set_facecolor("tomato");       parts["bodies"][1].set_alpha(0.7)
            parts["bodies"][2].set_facecolor("mediumpurple"); parts["bodies"][2].set_alpha(0.7)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_color("black"); parts[key].set_linewidth(1.0)

            for pos, vals in zip([0, 1, 2], [noncut, cut, allv]):
                med = float(np.median(vals))
                ax.text(pos + 0.16, med, f"{med:.3f}", va="center", ha="left", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels([
                f"Non-cut\n{nc_mr:.0%} matched\nn={len(noncut)}",
                f"Cut\n{cut_mr:.0%} matched\nn={len(cut)}",
                f"All\n{all_mr:.0%} matched\nn={len(allv)}",
            ])
            ax.set_xlabel("Roof subset")
            ax.set_title(panel_title)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.grid(True, alpha=0.3, axis="y")

        axes[0].set_ylabel("Mean IoU score (strict: 0 if missed)")
        fig.suptitle(title)
        fig.tight_layout()
        export_figure(fig, filename)
        plt.show()


# %%
plot_iou_distribution(
    df_all, _best_n,
    title=f"Mean IoU distribution by cut/non-cut roofs - OOD ({OOD_N_TILES} tiles/run)",
    filename="03_iou_distribution_cut_vs_noncut",
)

# %%
plot_iou_distribution(
    df_id_all, _best_n_id,
    title=f"Mean IoU distribution by cut/non-cut roofs - ID ({ID_TEST_TILES_STR} tiles/run)",
    filename="03_iou_distribution_cut_vs_noncut_id",
)

# %% [markdown]
# ## EXP2: IoU percentile bands (non-cut, non-artifact)
#
# Median IoU with a p25-p75 band vs. training size, OOD and ID overlaid on the
# same axes (only 2 median lines total, so no clutter). Pooling all 5 runs'
# rows per n directly widens/stabilizes the quantile estimate with run-to-run
# variation folded in alongside roof-to-roof variation.

# %%
def _pct_by_n(df):
    d = df[~df["is_cut"] & ~df["is_artifact"] & df["n"].isin(_ns_plot)]
    pct = d.groupby("n")["iou_score"].quantile([0.25, 0.50, 0.75]).unstack()
    pct.columns = ["p25", "p50", "p75"]
    return pct


_pct_ood = _pct_by_n(df_all)
_pct_id  = _pct_by_n(df_id_all)

with plt.rc_context(THESIS_RC):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for _pct, _col, _lbl in [(_pct_ood, "steelblue", "OOD"), (_pct_id, "tomato", "ID")]:
        ax.fill_between(_pct.index, _pct["p25"], _pct["p75"], alpha=0.15, color=_col)
        ax.plot(_pct.index, _pct["p50"], "o-", color=_col,
                linewidth=1.5, markersize=5, label=f"{_lbl} median (p25-p75 shaded)")
        ax.plot(_pct.index, _pct["p25"], "--", color=_col, linewidth=0.8, alpha=0.5)
        ax.plot(_pct.index, _pct["p75"], "--", color=_col, linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Training-set size (number of tiles), where zs = zero-shot")
    ax.set_ylabel("Median IoU score (strict: 0 if missed)")
    ax.set_title("IoU percentile bands vs. training size - non-cut, non-artifact roofs\n"
                 f"OOD vs. ID, {len(RUN_NAMES)} runs pooled")
    ax.set_xticks(_ns_plot)
    ax.set_xticklabels(_xtick_labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    export_figure(fig, "04_iou_percentile_bands")
    plt.show()

# %% [markdown]
# ## EXP3: Matched vs. missed roofs vs. training size
#
# Proportion of ground-truth roofs found (matched) vs. missed, OOD left / ID
# right. Each bar is the mean matched proportion across the 5 runs; the black
# error bar spans the run-to-run min/max range, and the label above each bar
# gives the min-max range of the *absolute* missed-roof count across runs.
# Zero-shot (n=0) is always a single pooled point with no band in either
# domain - one deterministic model pass, regardless of how many runs' worth of
# tiles it gets evaluated/pooled over (see the note in Block 1+2 above).

# %%
def _prop_matched_per_run(df, sizes):
    out = {r: {} for r in RUN_NAMES}
    for run in RUN_NAMES:
        for n in sizes:
            sub = df[(df["run"] == run) & (df["n"] == n)]
            if sub.empty:
                continue
            out[run][n] = float(sub["matched"].mean())
    return out


def _missed_count_per_run(df, sizes):
    out = {r: {} for r in RUN_NAMES}
    for run in RUN_NAMES:
        for n in sizes:
            sub = df[(df["run"] == run) & (df["n"] == n)]
            if sub.empty:
                continue
            out[run][n] = int((~sub["matched"]).sum())
    return out


_exp3_sizes = [n for n in _ns_plot if n != 0]   # non-zero sizes, reused by EXP6 below


def _exp3_arrays(df):
    """(prop_mean, prop_lo, prop_hi, miss_lo, miss_hi) aligned to `_ns_plot`.
    Zero-shot (n=0) is always pooled directly (`df[df["n"] == 0]`, ignoring
    the `run` column) into one point with no band - `_prop_matched_per_run`/
    `_missed_count_per_run` are only ever asked about the non-zero sizes."""
    sizes = _exp3_sizes
    p_mean, p_lo, p_hi = mean_band(stack_by_run(_prop_matched_per_run(df, sizes), sizes))
    miss_arr = stack_by_run(_missed_count_per_run(df, sizes), sizes)
    miss_lo, miss_hi = np.nanmin(miss_arr, axis=0).astype(int), np.nanmax(miss_arr, axis=0).astype(int)

    zs_sub  = df[df["n"] == 0]
    zs_prop = float(zs_sub["matched"].mean())
    zs_miss = int((~zs_sub["matched"]).sum())

    return (np.concatenate([[zs_prop], p_mean]), np.concatenate([[zs_prop], p_lo]),
            np.concatenate([[zs_prop], p_hi]), np.concatenate([[zs_miss], miss_lo]),
            np.concatenate([[zs_miss], miss_hi]))


# --- Tweak these to adjust the missed-count annotations above the bars ---
_EXP3_LABEL_COLOR = "firebrick"   # darker than the "Missed" bar color (tomato) for contrast on white
_EXP3_LABEL_Y      = 1.03         # axes-fraction y of the per-bar missed-count labels
_EXP3_HEADER_Y     = 1.10         # axes-fraction y of the "missed count range ->" caption
_EXP3_TITLE_PAD    = 46           # points between the axes top and the title; raise if it still touches the labels
_xpos = list(range(len(_ns_plot)))


def _draw_exp3_panel(ax, prop_m, prop_lo, prop_hi, miss_lo, miss_hi, title):
    ax.bar(_xpos, prop_m, color="steelblue", label="Matched", width=0.6)
    ax.bar(_xpos, 1 - prop_m, bottom=prop_m, color="tomato", label="Missed", width=0.6)
    ax.errorbar(_xpos, prop_m, yerr=[prop_m - prop_lo, prop_hi - prop_m],
                fmt="none", ecolor="black", elinewidth=1.0, capsize=3,
                label="Run-to-run range (matched proportion)")
    ax.set_xticks(_xpos)
    ax.set_xticklabels(_xtick_labels)
    ax.set_xlabel("Training-set size (tiles), zs = zero-shot")
    ax.set_title(title, pad=_EXP3_TITLE_PAD)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.set_ylim(0.95, 1.0)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(True, alpha=0.3, axis="y")
    # x in data coords, y in axes-fraction coords, so labels sit just above the
    # frame regardless of the y-axis data range.
    blend = ax.get_xaxis_transform()
    for xi, lo, hi in zip(_xpos, miss_lo, miss_hi):
        lbl = str(int(lo)) if lo == hi else f"{int(lo)}-{int(hi)}"
        ax.text(xi, _EXP3_LABEL_Y, lbl, ha="center", va="bottom", fontsize=6.5,
                color=_EXP3_LABEL_COLOR, transform=blend)
    ax.text(0.5, _EXP3_HEADER_Y, "missed count range", ha="center", va="bottom",
            fontsize=6.5, color=_EXP3_LABEL_COLOR, transform=ax.transAxes)


with plt.rc_context(THESIS_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=True)
    _draw_exp3_panel(ax_ood, *_exp3_arrays(df_all), f"OOD ({OOD_N_TILES} tiles/run)")
    _draw_exp3_panel(ax_id, *_exp3_arrays(df_id_all), f"ID ({ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("Proportion of ground-truth roofs")
    fig.suptitle(f"Matched vs. missed roofs by training size "
                 f"(zero-shot: 1 pooled point; fine-tuned: {len(RUN_NAMES)} runs)")
    fig.tight_layout()
    export_figure(fig, "05_matched_vs_missed_roofs")
    plt.show()

# %% [markdown]
# ## EXP4: IoU vs. roof size, best fine-tuned size
#
# Non-cut, non-artifact roofs at each domain's own best fine-tuned size
# (`_best_n` OOD, `_best_n_id` ID - these can differ), pooled across all 5 runs
# (so each ground-truth roof contributes up to 5 points, one per run). Rendered
# as two panels since the two domains' scatter clouds would be hard to tell
# apart overlaid on one axes, and they don't necessarily share the same n.

# %%
def _draw_iou_vs_size_panel(ax, df, n, title):
    d = df[(df["n"] == n) & ~df["is_cut"] & ~df["is_artifact"]].copy()
    d["log_area"] = np.log10(d["area_px"].clip(lower=1))
    log_bins = np.linspace(d["log_area"].min(), d["log_area"].max(), 11)
    d["area_bin"] = pd.cut(d["log_area"], bins=log_bins)
    bin_stats = d.groupby("area_bin", observed=True)["iou_score"].agg(["mean", "count"])
    bin_cx    = [10 ** iv.mid for iv in bin_stats.index]

    ax.scatter(d["area_px"], d["iou_score"],
               alpha=0.12, s=6, color="steelblue", rasterized=True, label="Individual roofs (all runs)")
    ax.plot(bin_cx, bin_stats["mean"], "o-", color="tomato", linewidth=2, markersize=6, label="Bin mean IoU")
    ax.set_xscale("log")
    ax.set_xlabel("Roof area (pixels, log scale)")
    ax.set_title(title)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")


with plt.rc_context(THESIS_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=True)
    _draw_iou_vs_size_panel(ax_ood, df_all, _best_n, f"OOD (n={_best_n} training tiles, {OOD_N_TILES} tiles/run)")
    _draw_iou_vs_size_panel(ax_id, df_id_all, _best_n_id,
                             f"ID (n={_best_n_id} training tiles, {ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("Mean IoU score (strict: 0 if missed)")
    fig.suptitle(f"Mean IoU vs. roof size - best fine-tuned size, non-cut, non-artifact\n"
                 f"({len(RUN_NAMES)} runs pooled)")
    fig.tight_layout()
    export_figure(fig, "06_iou_vs_roof_size")
    plt.show()

# %% [markdown]
# ## EXP5: Size-stratified scarcity curve
#
# Mean IoU vs. training size, split into Small/Medium/Large roof-area tertiles
# (tertile edges from the pooled OOD non-cut, non-artifact area distribution
# across all runs - shared with the ID panel so "Small/Medium/Large" mean the
# same physical roof size in both, keeping the two panels comparable). Each
# point pools individual roof detections from all 5 runs. OOD left, ID right.

# %%
_d_ref     = df_all[~df_all["is_cut"] & ~df_all["is_artifact"]]
_t33, _t67 = float(_d_ref["area_px"].quantile(1 / 3)), float(_d_ref["area_px"].quantile(2 / 3))
_size_bins   = [0.0, _t33, _t67, float("inf")]
_size_labels = ["Small", "Medium", "Large"]
_bin_leg     = {
    "Small":  f"Small  (< {_t33:.0f} px)",
    "Medium": f"Medium ({_t33:.0f} - {_t67:.0f} px)",
    "Large":  f"Large  (> {_t67:.0f} px)",
}

_bin_styles = {
    "Small":  {"color": "steelblue",  "marker": "o", "ls": "-"},
    "Medium": {"color": "seagreen",   "marker": "s", "ls": "--"},
    "Large":  {"color": "tomato", "marker": "^", "ls": "-."},
}


def _strat_stats_for(df):
    d = df[~df["is_cut"] & ~df["is_artifact"] & df["n"].isin(_ns_plot)].copy()
    d["size_bin"] = pd.cut(d["area_px"], bins=_size_bins, labels=_size_labels, include_lowest=True)
    return d.groupby(["n", "size_bin"], observed=True)["iou_score"].mean().unstack()


def _draw_strat_panel(ax, stats, title):
    lw, ms = 1.5 * MIOU_FONT_SCALE, 5 * MIOU_FONT_SCALE
    for _bin in _size_labels:
        if _bin in stats.columns:
            st = _bin_styles[_bin]
            ax.plot(stats.index, stats[_bin], marker=st["marker"], linestyle=st["ls"], color=st["color"],
                    linewidth=lw, markersize=ms, label=_bin_leg[_bin])
    ax.set_xlabel("Training-set size (tiles), zs = zero-shot")
    ax.set_title(title)
    ax.set_xticks(_ns_plot)
    ax.set_xticklabels(_xtick_labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    # sharey=True (needed so both panels use the same y-scale) hides tick
    # labels on every axes but the first by default - force them back on so
    # the ID panel is readable without cross-referencing the OOD one.
    ax.tick_params(labelleft=True)
    ax.legend(fontsize=7 * MIOU_FONT_SCALE)
    ax.grid(True, alpha=0.3)


with plt.rc_context(MIOU_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=MIOU_FIGSIZE, sharey=True)
    _draw_strat_panel(ax_ood, _strat_stats_for(df_all), f"OOD ({OOD_N_TILES} tiles/run)")
    _draw_strat_panel(ax_id, _strat_stats_for(df_id_all), f"ID ({ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("Mean IoU (strict: 0 if missed)")
    fig.suptitle(f"Size-stratified scarcity curve - non-cut, non-artifact roofs ({len(RUN_NAMES)} runs pooled)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=MIOU_WSPACE)
    export_figure(fig, "07_size_stratified_scarcity_curve")
    plt.show()

# %% [markdown]
# ## EXP6: Missed-roof counts by category vs. training size
#
# Match rates are all >= 0.98 (see EXP3), so proportions are unreadable at this
# scale; absolute missed counts reveal where the few detection failures land.
# Each bar is the mean missed count across the 5 runs, with a min/max error bar
# for run-to-run range. Zero-shot (n=0) is always a single pooled point with no
# band in either domain - one deterministic model pass (see the note in
# Block 1+2 above).

# %%
_std_mask = lambda d: d[~d["is_artifact"] & ~d["is_cut"]]
_art_mask = lambda d: d[d["is_artifact"]]
_cut_mask = lambda d: d[d["is_cut"] & ~d["is_artifact"]]


def _missed_count_per_run_grouped(df, sizes, mask_fn):
    out = {r: {} for r in RUN_NAMES}
    for run in RUN_NAMES:
        for n in sizes:
            sub = mask_fn(df[(df["run"] == run) & (df["n"] == n)])
            out[run][n] = int((~sub["matched"]).sum())
    return out


def _exp6_arrays(df, mask_fn):
    """(mean, lo, hi) missed-count arrays aligned to `_ns_plot`. Zero-shot is
    pooled directly (ignoring `run`) into one point with no band."""
    sizes = _exp3_sizes
    mean, lo, hi = mean_band(stack_by_run(_missed_count_per_run_grouped(df, sizes, mask_fn), sizes))
    zs_missed = int((~mask_fn(df[df["n"] == 0])["matched"]).sum())
    return (np.concatenate([[zs_missed], mean]), np.concatenate([[zs_missed], lo]),
            np.concatenate([[zs_missed], hi]))


def _draw_exp6_panel(ax, df, title):
    ref_n = df[df["n"] == 0]
    n_std_ref = int((~ref_n["is_cut"] & ~ref_n["is_artifact"]).sum())
    n_art_ref = int(ref_n["is_artifact"].sum())
    n_cut_ref = int((ref_n["is_cut"] & ~ref_n["is_artifact"]).sum())

    std_mean, std_lo, std_hi = _exp6_arrays(df, _std_mask)
    cut_mean, cut_lo, cut_hi = _exp6_arrays(df, _cut_mask)
    art_mean, art_lo, art_hi = _exp6_arrays(df, _art_mask)

    w = 0.27
    x = np.arange(len(_ns_plot))
    ax.bar(x - w, std_mean, width=w, color="steelblue",
           yerr=[std_mean - std_lo, std_hi - std_mean], capsize=3,
           label=f"Standard (n={n_std_ref})")
    ax.bar(x, cut_mean, width=w, color="seagreen",
           yerr=[cut_mean - cut_lo, cut_hi - cut_mean], capsize=3,
           label=f"Cut (n={n_cut_ref})")
    ax.bar(x + w, art_mean, width=w, color="tomato",
           yerr=[art_mean - art_lo, art_hi - art_mean], capsize=3,
           label=f"Artifact (n={n_art_ref})")
    ax.set_xticks(x)
    ax.set_xticklabels(_xtick_labels, rotation=0, ha="center")
    ax.set_xlabel("Training-set size (tiles), zs = zero-shot")
    ax.set_title(title)
    # Integer ticks, but let matplotlib pick a sensible step - a fixed step of
    # 1 (fine when OOD's max was ~30) produces ~70 unreadable overlapping
    # ticks on the ID panel, whose pooled zero-shot "cut" bar is genuinely
    # much larger (real data: fine-tuning helps a lot more on cut roofs
    # in-distribution than the ~30-max OOD range this locator was tuned for).
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=10))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


with plt.rc_context(THESIS_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=False)
    _draw_exp6_panel(ax_ood, df_all, f"OOD ({OOD_N_TILES} tiles/run)")
    _draw_exp6_panel(ax_id, df_id_all, f"ID ({ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("Missed roofs (mean absolute count across runs)")
    fig.suptitle(f"Missed-roof counts by category vs. training size\n"
                 f"error bars = run-to-run min/max ({len(RUN_NAMES)} runs)")
    fig.tight_layout()
    export_figure(fig, "08_missed_counts_by_category")
    plt.show()

# %% [markdown]
# ## EXP7: IoU quality by roof category vs. training size
#
# Mean strict IoU for all / standard / artifact / cut roofs. Each point pools
# individual detections from all 5 runs (denominators shown in the legend are
# the zero-shot counts, matching the original per-category groupings - a single
# run for OOD, all 5 runs' filtered test-tile subsets pooled for ID). OOD left,
# ID right.

# %%
def _draw_exp7_panel(ax, df, title):
    lw, ms = 1.5 * MIOU_FONT_SCALE, 5 * MIOU_FONT_SCALE
    d        = df[df["n"].isin(_ns_plot)]
    ref_n    = df[df["n"] == 0]
    art_iou  = d[  d["is_artifact"]].groupby("n")["iou_score"].mean()
    std_iou  = d[~d["is_artifact"] & ~d["is_cut"]].groupby("n")["iou_score"].mean()
    cut_iou  = d[  d["is_cut"] & ~d["is_artifact"]].groupby("n")["iou_score"].mean()
    all_iou  = d.groupby("n")["iou_score"].mean()
    n_all_ref = int(len(ref_n))
    n_std_ref = int((~ref_n["is_cut"] & ~ref_n["is_artifact"]).sum())
    n_art_ref = int(ref_n["is_artifact"].sum())
    n_cut_ref = int((ref_n["is_cut"] & ~ref_n["is_artifact"]).sum())

    ax.plot(all_iou.index, all_iou.values, "D-",  color="black",
            linewidth=lw, markersize=ms, label=f"All roofs (n={n_all_ref})", zorder=5)
    ax.plot(std_iou.index, std_iou.values, "o-",  color="steelblue",
            linewidth=lw, markersize=ms, label=f"Standard (n={n_std_ref})")
    ax.plot(art_iou.index, art_iou.values, "s--", color="tomato",
            linewidth=lw, markersize=ms, label=f"Artifact (n={n_art_ref})")
    ax.plot(cut_iou.index, cut_iou.values, "^:",  color="seagreen",
            linewidth=lw, markersize=ms, label=f"Cut (n={n_cut_ref})")
    ax.set_xlabel("Training-set size (tiles), zs = zero-shot")
    ax.set_title(title)
    ax.set_xticks(_ns_plot)
    ax.set_xticklabels(_xtick_labels)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.legend(fontsize=7 * MIOU_FONT_SCALE)
    ax.grid(True, alpha=0.3)


with plt.rc_context(MIOU_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=MIOU_FIGSIZE, sharey=False)
    _draw_exp7_panel(ax_ood, df_all, f"OOD ({OOD_N_TILES} tiles/run)")
    _draw_exp7_panel(ax_id, df_id_all, f"ID ({ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("Mean IoU (strict: 0 if missed)")
    fig.suptitle(f"IoU quality by roof category vs. training size ({len(RUN_NAMES)} runs pooled)")
    fig.tight_layout()
    fig.subplots_adjust(wspace=MIOU_WSPACE)
    export_figure(fig, "09_iou_quality_by_category")
    plt.show()

# %% [markdown]
# ## EXP8: Training-set size-class composition (dataset-level, but now varies per run)
#
# For each dataset size n, what fraction of *training* instances (not the eval
# sets) are Small/Medium/Large roofs. Which tiles go into a given n-tile subset
# is decided by `compute_split(tiles, train_fraction, test_fraction, split_seed)`
# (see `get_split_tiles` above) - and unlike the old train/val-only scheme,
# `split_seed` now differs per run (each run also gets its own train/val/test
# split, not just its own model-init draw). So this composition curve now
# genuinely varies run-to-run, and is shown as a mean +/- min/max band across
# the 5 runs, the same way every model-behaviour plot above is - not the single
# fixed curve of the old (pre-test-split) notebook.

# %%
# area_px lookup: tile_name -> [area_px, ...] (non-artifact, cut included)
_train_areas = {}
for _region in TRAIN_REGIONS:
    _csv = os.path.join(IOU_RESULTS_DIR, f"{_region}_unified_metrics.csv")
    if not os.path.exists(_csv):
        print(f"WARNING: {_csv} not found")
        continue
    _d = pd.read_csv(_csv)
    _d = _d[~_d["is_artifact"].astype(bool)]
    for _row in _d.itertuples(index=False):
        _tname = os.path.splitext(_row.belongs_to)[0]
        _train_areas.setdefault(_tname, []).append(float(_row.area_px))

# Reuse graph 07's OOD-based tertile edges (not the ID training pool's own
# tertiles) so Small/Medium/Large mean the same physical roof size across
# every stratified plot in this notebook.
_b33, _b67 = _t33, _t67

_exp8_size_labels = ["Small", "Medium", "Large"]
_exp8_bin_leg     = {
    "Small":  f"Small  (< {_b33:.0f} px)",
    "Medium": f"Medium ({_b33:.0f} - {_b67:.0f} px)",
    "Large":  f"Large  (> {_b67:.0f} px)",
}
print(f"Thresholds (shared with graph 07, OOD-based): p33={_b33:.0f} px, p67={_b67:.0f} px")

# {run: [n, ...] -> train tile names, first n of that run's own shuffled train split}
_train_seq_by_run = {}
for _run in RUN_NAMES:
    _train_paths = get_split_tiles(_run)["train"]
    _train_seq_by_run[_run] = [os.path.splitext(os.path.basename(p))[0] for p in _train_paths]

_exp8_ns = [n for n in DATASET_SIZES]
# {label: n_runs x n_sizes array of proportions}
_exp8_props_by_run = {lbl: np.full((len(RUN_NAMES), len(_exp8_ns)), np.nan) for lbl in _exp8_size_labels}

print(f"\n{'run':>8}  {'n':>5}  {'Small%':>8}  {'Medium%':>8}  {'Large%':>8}  {'instances':>10}")
print("-" * 58)
for _ri, _run in enumerate(RUN_NAMES):
    _train_seq = _train_seq_by_run[_run]
    for _ni, _n in enumerate(_exp8_ns):
        _active = set(_train_seq[:_n])
        _areas  = [a for _t in _active for a in _train_areas.get(_t, [])]
        _total  = len(_areas)
        _small  = sum(1 for a in _areas if a <= _b33)
        _medium = sum(1 for a in _areas if _b33 < a <= _b67)
        _large  = sum(1 for a in _areas if a > _b67)
        if _total:
            _exp8_props_by_run["Small"][_ri, _ni]  = _small  / _total
            _exp8_props_by_run["Medium"][_ri, _ni] = _medium / _total
            _exp8_props_by_run["Large"][_ri, _ni]  = _large  / _total
        print(f"{_run:>8}  {_n:>5}  {_small/_total if _total else 0:>8.1%}  "
              f"{_medium/_total if _total else 0:>8.1%}  {_large/_total if _total else 0:>8.1%}  {_total:>10}")

_exp8_bands = {lbl: mean_band(arr) for lbl, arr in _exp8_props_by_run.items()}

# %%
_exp8_bin_styles = {
    "Small":  {"color": "steelblue",  "marker": "o", "ls": "-"},
    "Medium": {"color": "seagreen",   "marker": "s", "ls": "--"},
    "Large":  {"color": "tomato", "marker": "^", "ls": "-."},
}

with plt.rc_context(THESIS_RC):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    for _lbl in _exp8_size_labels:
        _st = _exp8_bin_styles[_lbl]
        _mean, _lo, _hi = _exp8_bands[_lbl]
        ax.fill_between(_exp8_ns, _lo, _hi, alpha=0.15, color=_st["color"])
        ax.plot(_exp8_ns, _mean,
                marker=_st["marker"], linestyle=_st["ls"], color=_st["color"],
                linewidth=1.5, markersize=5, label=_exp8_bin_leg[_lbl])
    ax.set_xlabel("Training-set size (number of tiles)")
    ax.set_ylabel("Proportion of training instances")
    ax.set_title("Size-class composition of the training split vs. dataset size\n"
                  f"(ID training pool; mean +/- min/max across {len(RUN_NAMES)} runs' own splits)")
    ax.set_xticks(_exp8_ns)
    ax.set_xticklabels([str(n) for n in _exp8_ns])
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    export_figure(fig, "10_training_size_class_composition")
    plt.show()

# %% [markdown]
# ## EXP10: Mask composition per training region (dataset-level, single-seed by design)
#
# How many ground-truth masks per region are whole / cut-only / artifact-only /
# both, straight from the annotation CSVs. This describes the raw annotations,
# not a trained model, so - like EXP8 - it is single-seed by design.

# %%
_exp10_regions = ["ceu_paz", "cantidio_sampaio", "santa_madalena"]
# Display-only rename: files/paths still use "santa_madalena" (never renamed
# on disk when the region was renamed mid-project), but the graph should show
# its current name.
_exp10_display_names = {"santa_madalena": "rio_claro_ii"}

_exp10_counts = {}
for _reg in _exp10_regions:
    _csv = os.path.join(IOU_RESULTS_DIR, f"{_reg}_unified_metrics.csv")
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
    print(f"{_reg}: total={len(_d)}  whole={_c['whole']}  cut_only={_c['cut_only']}  "
          f"artifact_only={_c['artifact_only']}  both={_c['both']}  "
          f"excluded_by_artifact_filter={_c['artifact_only'] + _c['both']}")

_exp10_cats      = ["whole",   "cut_only", "artifact_only",          "both"]
_exp10_nicenames = ["Whole",   "Cut only", "Artifact only (excl.)",  "Cut + artifact (excl.)"]
_exp10_colors    = ["#4caf50", "#2196f3",  "#ff9800",                "#f44336"]

with plt.rc_context(THESIS_RC):
    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    _x10       = np.arange(len(_exp10_regions))
    _bottoms10 = np.zeros(len(_exp10_regions))

    for _cat, _col, _nice in zip(_exp10_cats, _exp10_colors, _exp10_nicenames):
        _vals = np.array([_exp10_counts[_reg][_cat] for _reg in _exp10_regions], dtype=float)
        ax.bar(_x10, _vals, bottom=_bottoms10, color=_col, label=_nice, edgecolor="white", linewidth=0.5)
        for _xi, (_v, _b) in enumerate(zip(_vals, _bottoms10)):
            if _v >= 10:
                ax.text(_xi, _b + _v / 2, str(int(_v)),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        _bottoms10 += _vals

    ax.set_xticks(_x10)
    ax.set_xticklabels([_exp10_display_names.get(_r, _r) for _r in _exp10_regions])
    ax.set_xlabel("Region")
    ax.set_ylabel("Number of ground-truth masks")
    ax.set_title("Ground-truth mask composition per region\n"
                      "filter_artifacts=True excludes orange + red segments\n"
                      "Every annotated tile in the region (train+val+test)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    export_figure(fig, "11_mask_composition_per_region")
    plt.show()

# %% [markdown]
# ## EXP9: Training progress curves (all 5 runs)
#
# Per-epoch train/val loss for two representative dataset sizes (`n=40`, a
# small/mid regime, and `n=108`, the largest sweep size - not the same as
# `_best_n` below, which is picked by mIoU, not by size). Every run is drawn
# as a thin, semi-transparent line;
# the bold line is the mean across runs. An "x" marks each run's selected
# (best-val-loss) checkpoint epoch on its own val-loss curve.
#
# Loss values come straight from the per-epoch `train_stats.json` /
# `val_stats.json` logs already written during training - no retraining or
# recomputation involved.

# %%
def find_best_checkpoint_epoch(run, n):
    """Highest-numbered checkpoint_<epoch>_best.pt for (run, n), i.e. the selected checkpoint."""
    ckpt_dir = os.path.join(RUNS_LOG_BASE, run, f"n{n}", "checkpoints")
    pattern  = re.compile(r"checkpoint_(\d+)_best\.pt$")
    epochs = [int(pattern.search(f).group(1)) for f in glob.glob(os.path.join(ckpt_dir, "checkpoint_*_best.pt"))
              if pattern.search(f)]
    return max(epochs) if epochs else None


def load_run_epoch_series(run, n, split):
    """split in {'train', 'val'} -> (epochs, loss) arrays from the per-epoch JSON log, or None.

    A resumed/requeued job can log the same epoch twice; keep the later entry.
    """
    f = os.path.join(RUNS_LOG_BASE, run, f"n{n}", "logs", f"{split}_stats.json")
    if not os.path.exists(f):
        return None
    rows = [json.loads(line) for line in open(f)]
    key  = f"Loss/total/{split}"
    s = pd.Series({r["Trainer/epoch"]: r[key] for r in rows}).sort_index()
    return s.index.to_numpy(), s.to_numpy()


_EXP9_NS = [40, 108]

with plt.rc_context(THESIS_RC):
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=False)

    for ax, n in zip(axes, _EXP9_NS):
        _train_means, _val_means = [], []
        for run in RUN_NAMES:
            _tr = load_run_epoch_series(run, n, "train")
            _vl = load_run_epoch_series(run, n, "val")
            if _tr is None or _vl is None:
                continue
            ax.plot(_tr[0], _tr[1], "-", color="steelblue", linewidth=0.8, alpha=0.35, zorder=2)
            ax.plot(_vl[0], _vl[1], "-", color="tomato",    linewidth=0.8, alpha=0.35, zorder=2)

            _best_ep = find_best_checkpoint_epoch(run, n)
            if _best_ep is not None and _best_ep in _vl[0]:
                _bi = list(_vl[0]).index(_best_ep)
                ax.scatter([_best_ep], [_vl[1][_bi]], marker="x", color="black",
                           s=30, linewidths=1.2, zorder=4)

            _train_means.append(pd.Series(_tr[1], index=_tr[0]))
            _val_means.append(pd.Series(_vl[1], index=_vl[0]))

        _tr_mean = pd.concat(_train_means, axis=1).mean(axis=1)
        _vl_mean = pd.concat(_val_means,   axis=1).mean(axis=1)
        ax.plot(_tr_mean.index, _tr_mean.values, "-", color="steelblue", linewidth=2.0,
                label="Train loss (mean of 5 runs)", zorder=3)
        ax.plot(_vl_mean.index, _vl_mean.values, "-", color="tomato", linewidth=2.0,
                label="Val loss (mean of 5 runs)", zorder=3)
        ax.scatter([], [], marker="x", color="black", s=30, linewidths=1.2,
                   label="Selected checkpoint (per run)")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (total)")
        ax.set_title(f"n={n}")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training progress - train vs. val loss ({len(RUN_NAMES)} runs overlaid)")
    fig.tight_layout()
    export_figure(fig, "12_training_progress_curves")
    plt.show()

# %% [markdown]
# ## EXP12: IoU score vs. ground-truth mask area
#
# Academic check that smaller masks receive lower IoU scores. Only matched,
# non-cut, non-artifact detections are used so the signal reflects segmentation
# quality only, not detection failure or tile-boundary clipping.
#
# Note: D2's version of this plot filtered on `is_cut == True` (cut roofs only)
# while its own comment said "non-cut, non-artifact" - the two disagreed. Cut
# roofs have an artificially truncated area *and* a capped achievable IoU
# (see EXP1/EXP3), so including them would confound the roof-size signal this
# plot is meant to isolate. D3 filters on `~is_cut`, matching the stated intent.
#
# A. Scatter + binned mean, one line per training-set size (5 runs pooled per
#    size), zero-shot excluded - so this is purely "how does fine-tuning
#    behave across dataset scale," not mixed with the zero-shot baseline.
#    Already an 8-line max-density plot on its own, so OOD and ID get their
#    own separate exports rather than being crammed into one figure.
# B. Zero-shot vs. best fine-tuned (each domain's own best n, 5 runs pooled).
#    Only 2 lines, so OOD | ID share one figure as side-by-side panels.
#
# Note on the Spearman rho/p pseudo-replication fix below: it assumes the 5
# runs re-evaluate the *same* rooftops, which holds for OOD (the whole area is
# always the eval set) but not for ID (each run holds out different test
# tiles). `_roof_level`'s groupby still does the right thing either way - it
# just collapses less for ID, since there's less to collapse.

# %%
_N_BINS12 = 12


def _prep_exp12(df):
    """Filter to matched, non-cut, non-artifact detections and bin log(area_px)
    into _N_BINS12 bins (edges fit to this df's own area range - fine since the
    x-axis is continuous area_px, not a categorical label that needs to mean
    the same thing across domains, unlike EXP5's Small/Medium/Large)."""
    d = df[~df["is_cut"] & ~df["is_artifact"] & df["matched"] & (df["area_px"] > 0)].copy()
    log = np.log10(d["area_px"])
    edges = np.linspace(log.min(), log.max(), _N_BINS12 + 1)
    bin_cx = np.array([10 ** ((edges[i] + edges[i + 1]) / 2) for i in range(_N_BINS12)])
    d["area_bin12"] = pd.cut(log, bins=edges)
    return d, bin_cx


_d12, _bin12_cx = _prep_exp12(df_all)
_d12_id, _bin12_cx_id = _prep_exp12(df_id_all)


def _bin12_stats(sub):
    """(mean, 95% CI, count) per area bin. A bin with only one sample gets a
    real mean but ci=NaN (no std is defined for n=1) - in the plots below this
    means that lone point still gets drawn, but fill_between's shaded CI band
    correctly stops one bin short of it instead of fabricating an interval.
    This shows up on the zero-shot line in the zero-shot-vs-fine-tuned plot,
    whose largest-area bin has exactly one sample (zero-shot is a single
    deterministic pass, so it never gets more than 1 sample in a sparse bin);
    the fine-tuned line's equivalent bin pools 5 runs so its band reaches the
    last point. `_mark_unreliable_bins` below flags this explicitly instead of
    leaving it as an unexplained gap in the shading."""
    g   = sub.groupby("area_bin12", observed=False)["iou_score"]
    mn  = g.mean().values
    std = g.std().values
    cnt = g.count().values
    ci  = np.where(cnt > 1, 1.96 * std / np.sqrt(np.maximum(cnt, 1)), np.nan)
    return mn, ci, cnt


_MIN_RELIABLE_N = 15   # below this, a bin's mean is 1-14 detections - real, but not
                       # trustworthy enough to draw with the same visual weight as
                       # bins backed by dozens/hundreds. Chosen so it only catches
                       # the genuinely thin zero-shot edge bins (counts 1-3) without
                       # flagging anything on the pooled or 5-run fine-tuned lines,
                       # whose smallest bin count is 15+.


def _mark_unreliable_bins(ax, x, y, cnt, color, min_n=_MIN_RELIABLE_N, xytext=(0, 8), annotate=True):
    """Overlay a hollow marker on any point backed by fewer than `min_n`
    detections, so a mean computed from a handful of samples doesn't read
    with the same confidence as one from hundreds. `annotate=False` still
    draws the ring but skips the 'n=<count>' text - for callers plotting
    several series that share near-identical bin counts (Figure A), where
    every series should still visually flag its own low-confidence points
    but only one needs to spell the count out in text. `xytext` is the
    label's offset from the point - pass a negative y to put it below the
    point instead of above. Returns how many points were marked, so callers
    can skip the "hollow marker" legend entry when none were drawn anywhere."""
    _lo = cnt <= min_n
    if not np.any(_lo):
        return 0
    ax.scatter(x[_lo], y[_lo], s=34, facecolors="white", edgecolors=color,
               linewidths=1.5, zorder=4)
    if annotate:
        _va = "top" if xytext[1] < 0 else "bottom"
        for _x, _y, _c in zip(x[_lo], y[_lo], cnt[_lo]):
            ax.annotate(f"n={_c}", (_x, _y), textcoords="offset points", xytext=xytext,
                        fontsize=6, ha="center", va=_va, color=color)
    return int(_lo.sum())


_mn_zs,   _ci_zs,   _cnt_zs   = _bin12_stats(_d12[_d12["n"] == 0])
_mn_best, _ci_best, _cnt_best = _bin12_stats(_d12[_d12["n"] == _best_n])
_mn_zs_id,   _ci_zs_id,   _cnt_zs_id   = _bin12_stats(_d12_id[_d12_id["n"] == 0])
_mn_best_id, _ci_best_id, _cnt_best_id = _bin12_stats(_d12_id[_d12_id["n"] == _best_n_id])


# The 5 fine-tuning runs re-evaluate the *same* OOD ground-truth rooftops
# (identical tile_source/feature_id/area_px across seeds - verified directly
# against the per-run CSVs), so a naive spearmanr on raw pooled rows treats
# ~5 repeated measurements of every rooftop as 5 independent observations.
# That pseudo-replication doesn't move rho much but massively overstates
# significance (checked: at n=5, pooled rows give p=1.5e-05 vs. p=0.91 once
# collapsed to one row per rooftop). `_bin12_stats` above is unaffected by
# this - it only ever produces a descriptive mean-IoU curve for plotting, not
# a significance claim - so it's left pooling raw rows for a smoother curve.
# rho/p, which *do* carry a significance claim, use only the two functions
# below instead.
def _roof_level(sub):
    """Collapse to one row per ground-truth rooftop by averaging iou_score
    across runs. area_px is identical across runs for a given rooftop (same
    OOD tiles/ground truth, only the fine-tuned model differs), so this
    removes seed-to-seed pseudo-replication without touching the area axis."""
    g = sub.groupby(["tile_source", "feature_id"])
    return pd.DataFrame({"area_px": g["area_px"].first(), "iou_score": g["iou_score"].mean()})


def _roof_spearman(sub):
    """Spearman rho/p on one row per rooftop (seed-averaged IoU) - the only
    valid independence unit here (see note above)."""
    roof = _roof_level(sub)
    rho, p = scipy_stats.spearmanr(roof["area_px"], roof["iou_score"])
    return rho, p, len(roof)


def _per_run_spearman(sub):
    """One Spearman rho per individual run (no pseudo-replication within a
    single run - every rooftop appears once there), returned as
    (mean, sd, rhos). Supplementary stability check: are the per-seed
    correlations consistent with each other, independent of the roof-
    averaging in `_roof_spearman`."""
    rhos = np.array([
        scipy_stats.spearmanr(s["area_px"], s["iou_score"])[0]
        for _, s in sub.groupby("run")
    ])
    sd = rhos.std(ddof=1) if len(rhos) > 1 else float("nan")
    return rhos.mean(), sd, rhos


_rho_zs,   _p_zs,   _nroof_zs   = _roof_spearman(_d12[_d12["n"] == 0])
_rho_best, _p_best, _nroof_best = _roof_spearman(_d12[_d12["n"] == _best_n])
_rho_mean_best, _rho_sd_best, _ = _per_run_spearman(_d12[_d12["n"] == _best_n])

_rho_zs_id,   _p_zs_id,   _nroof_zs_id   = _roof_spearman(_d12_id[_d12_id["n"] == 0])
_rho_best_id, _p_best_id, _nroof_best_id = _roof_spearman(_d12_id[_d12_id["n"] == _best_n_id])
_rho_mean_best_id, _rho_sd_best_id, _    = _per_run_spearman(_d12_id[_d12_id["n"] == _best_n_id])

# Figure A pools zero-shot OUT entirely (it's the deliberate contrast subject
# of Figure B, not part of "how fine-tuning behaves across scale") and instead
# of one line pooling every size together, gives each size its own line -
# still 5 runs per size, just not pooled across sizes too.
_d12_ft    = _d12[_d12["n"] != 0]
_d12_ft_id = _d12_id[_d12_id["n"] != 0]


def _exp12_size_stats(d12_ft, label):
    stats = {}
    for _n in DATASET_SIZES:
        _sub = d12_ft[d12_ft["n"] == _n]
        _mn_n, _ci_n, _cnt_n = _bin12_stats(_sub)
        _rho_n, _p_n, _nroof_n = _roof_spearman(_sub)
        _rho_mean_n, _rho_sd_n, _ = _per_run_spearman(_sub)
        stats[_n] = (_mn_n, _ci_n, _cnt_n, _rho_n, _p_n)
        print(f"[{label}] Spearman rho (n={_n}, roof-averaged over {len(RUN_NAMES)} seeds): "
              f"rho={_rho_n:.3f},  p={_p_n:.2e},  n_roofs={_nroof_n}  "
              f"| per-seed rho mean={_rho_mean_n:.3f}, sd={_rho_sd_n:.3f}")
    return stats


_size_stats    = _exp12_size_stats(_d12_ft, "OOD")
_size_stats_id = _exp12_size_stats(_d12_ft_id, "ID")

print(f"\nSpearman rho (OOD zero-shot, single pass - no pseudo-replication): "
      f"rho={_rho_zs:.3f},  p={_p_zs:.2e},  n_roofs={_nroof_zs}")
print(f"Spearman rho (OOD n={_best_n}, roof-averaged over {len(RUN_NAMES)} seeds): "
      f"rho={_rho_best:.3f},  p={_p_best:.2e},  n_roofs={_nroof_best}  "
      f"| per-seed rho mean={_rho_mean_best:.3f}, sd={_rho_sd_best:.3f}")
print(f"Spearman rho (ID zero-shot, {len(RUN_NAMES)} runs' filtered subsets pooled): "
      f"rho={_rho_zs_id:.3f},  p={_p_zs_id:.2e},  n_roofs={_nroof_zs_id}")
print(f"Spearman rho (ID n={_best_n_id}, roof-averaged over {len(RUN_NAMES)} seeds): "
      f"rho={_rho_best_id:.3f},  p={_p_best_id:.2e},  n_roofs={_nroof_best_id}  "
      f"| per-seed rho mean={_rho_mean_best_id:.3f}, sd={_rho_sd_best_id:.3f}")

# A one-hue light->dark ramp (the "correct" ordinal encoding for an ordered
# magnitude like dataset size) read as 8 near-identical shades of blue once
# actually rendered - not distinguishable at a glance even though the legend
# spells out each n and its rho. Swapped for a fixed 8-hue categorical
# palette instead (CVD-safe adjacent order), opening on the same blue the
# rest of the notebook already uses for "baseline"/steelblue context, so it
# still reads as belonging to this notebook's palette rather than a random
# rainbow.
_SIZE_RAMP = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_SIZE_COLOR = dict(zip(DATASET_SIZES, _SIZE_RAMP))

# --- Figure A: scatter + binned mean, one line per training-set size ---
# Each domain gets its own SINGLE_FIGSIZE=(6.3, 3.8) export (an 8-line max-
# density plot on its own; cramming OOD+ID side by side would double that).
def _draw_exp12_figA(d12_ft, size_stats, bin_cx, area_label, filename):
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)

        scat = d12_ft.sample(min(4000, len(d12_ft)), random_state=42)
        ax.scatter(scat["area_px"], scat["iou_score"],
                   alpha=0.04, s=4, color="steelblue", rasterized=True, zorder=1)

        # No CI shading here (unlike Figure B): 8 bands stacked on top of each
        # other would just be visual clutter regardless of hue. Reliability is
        # still shown, just via the hollow-marker mechanism below instead of a
        # shaded ribbon per line.
        #
        # Per-bin sample counts are near-identical across all 8 lines (recall is
        # uniformly high across dataset sizes - see EXP12 investigation), so every
        # line still gets its own hollow ring at a low-count bin (each one really
        # is backed by just as few detections), but only n=5 - which sits at or
        # near the bottom of the stack in every bin - also gets the "n=<count>"
        # text, placed below its point: one shared count per bin instead of up to
        # 8 overlapping copies of the same text.
        any_low_n = 0
        for n in DATASET_SIZES:
            mn_n, ci_n, cnt_n, rho_n, p_n = size_stats[n]
            vld = ~np.isnan(mn_n)
            col = _SIZE_COLOR[n]
            ax.plot(bin_cx[vld], mn_n[vld], "o-", color=col, linewidth=1.4,
                    markersize=4, zorder=3, label=f"n={n}  (rho={rho_n:.2f})")
            any_low_n += _mark_unreliable_bins(ax, bin_cx[vld], mn_n[vld], cnt_n[vld],
                                                col, xytext=(0, -10), annotate=(n == 5))

        ax.set_xscale("log")
        ax.set_xlabel("Ground-truth mask area (pixels, log scale)")
        ax.set_ylabel("IoU score (per rooftop)")
        ax.set_title("IoU score per ground-truth masks (non-cut, non-artifact, matched only)\n"
                     f"{area_label} - fine-tuned separated by training-set size ({len(RUN_NAMES)} runs pooled per size)",
                     pad=42)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        handles, labels = ax.get_legend_handles_labels()
        if any_low_n:
            # Only add this legend entry when a hollow marker actually got drawn -
            # per-size sample counts here rarely dip below _MIN_RELIABLE_N (recall
            # is uniformly high across sizes, unlike zero-shot's single pass in
            # Figure B), so a legend entry with nothing on the chart to back it up
            # is exactly the "broken" dangling marker this avoids.
            handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                                   markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5))
            labels.append(f"Hollow marker: bin has {u'\u2264'}{_MIN_RELIABLE_N} detections (n annotated)")
        # Legend sits in the gap opened by the title's extra `pad` above - between
        # the title and the axes - instead of off to the right, which left the
        # plot itself squeezed into a narrow strip of the figure.
        ax.legend(handles, labels, fontsize=6.5, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3, frameon=True)
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        export_figure(fig, filename)
        plt.show()


_draw_exp12_figA(_d12_ft, _size_stats, _bin12_cx, f"OOD ({OOD_N_TILES} tiles/run)", "13_iou_vs_mask_area_pooled_ood")
_draw_exp12_figA(_d12_ft_id, _size_stats_id, _bin12_cx_id, f"ID ({ID_TEST_TILES_STR} tiles/run)", "13_iou_vs_mask_area_pooled_id")


# --- Figure B: zero-shot vs. best fine-tuned, OOD | ID side by side ---
def _draw_exp12_figB_panel(ax, d12, best_n, bin_cx, mn_zs, ci_zs, cnt_zs, rho_zs, p_zs,
                            mn_best, ci_best, cnt_best, rho_best, p_best, title):
    # Same single blue as Figure A's background scatter, not one color per
    # series - both subsets' raw detections pooled into one cloud, since the
    # scatter is background context for where the data mass sits, not a third
    # way of encoding which series a point belongs to (the lines already do that).
    scat = pd.concat([d12[d12["n"] == 0], d12[d12["n"] == best_n]])
    scat = scat.sample(min(4000, len(scat)), random_state=42)
    ax.scatter(scat["area_px"], scat["iou_score"],
               alpha=0.04, s=4, color="steelblue", rasterized=True, zorder=1)

    any_low_n = 0
    for mn, ci, cnt, lbl, col, rho, p in [
        (mn_zs,   ci_zs,   cnt_zs,   "Zero-shot", "steelblue", rho_zs, p_zs),
        (mn_best, ci_best, cnt_best, f"Fine-tuned (n={best_n}, {len(RUN_NAMES)} run avg.)",
         "tomato", rho_best, p_best),
    ]:
        vld = ~np.isnan(mn)
        ax.fill_between(bin_cx[vld], (mn - ci)[vld], (mn + ci)[vld], alpha=0.2, color=col, zorder=2)
        ax.plot(bin_cx[vld], mn[vld], "o-", color=col, linewidth=1.8, markersize=5, zorder=3,
                label=f"{lbl}  (rho={rho:.3f}, p={p:.1e})")
        any_low_n += _mark_unreliable_bins(ax, bin_cx[vld], mn[vld], cnt[vld], col)

    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth mask area (pixels, log scale)")
    ax.set_title(title)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                           markerfacecolor="steelblue", markeredgecolor="none", alpha=0.5))
    labels.append("Individual roofs (ground-truth area vs. IoU)")
    if any_low_n:
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                               markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5))
        labels.append(f"Hollow marker: bin has {u'\u2264'}{_MIN_RELIABLE_N} detections (n annotated)")
    ax.legend(handles, labels, fontsize=6.5)
    ax.grid(True, alpha=0.3, which="both")


with plt.rc_context(THESIS_RC):
    fig, (ax_ood, ax_id) = plt.subplots(1, 2, figsize=DOUBLE_FIGSIZE, sharey=True)
    _draw_exp12_figB_panel(ax_ood, _d12, _best_n, _bin12_cx,
                            _mn_zs, _ci_zs, _cnt_zs, _rho_zs, _p_zs,
                            _mn_best, _ci_best, _cnt_best, _rho_best, _p_best,
                            f"OOD ({OOD_N_TILES} tiles/run)")
    _draw_exp12_figB_panel(ax_id, _d12_id, _best_n_id, _bin12_cx_id,
                            _mn_zs_id, _ci_zs_id, _cnt_zs_id, _rho_zs_id, _p_zs_id,
                            _mn_best_id, _ci_best_id, _cnt_best_id, _rho_best_id, _p_best_id,
                            f"ID ({ID_TEST_TILES_STR} tiles/run)")
    ax_ood.set_ylabel("IoU score (per rooftop)")
    fig.suptitle("IoU score per ground-truth masks (non-cut, non-artifact, matched only)\n"
                 "Zero-shot vs. fine-tuned")
    fig.tight_layout()
    export_figure(fig, "13_iou_vs_mask_area_zeroshot_vs_finetuned")
    plt.show()

# %% [markdown]
# ## EXP11: Training dynamics and LR schedule
#
# All runs share identical hyperparameters (base LR, weight decay, loss terms,
# 30-epoch warmup+cosine schedule) - only the dataset size `n` and the training
# seed differ. This figure is about the *mechanics* of checkpoint selection
# against the shared LR schedule, which don't vary meaningfully by seed; the
# actual run-to-run performance variance is already shown in EXP9 (loss
# curves) and the mIoU summary plots. So unlike the other blocks, this one is
# deliberately kept on a single illustrative reference run - `run_011`
# (split_seed 555, last of the new `RUN_NAMES`) - rather than overplotting 5x
# the lines onto an already-dense 4-panel figure. The one exception is panel
# (4), which is cheap to extend to a second series (only 1 series, no clutter):
# it now shows both OOD and ID matched mIoU for that reference run.
#
# Label placement notes (both in the plotting cell below):
# - "Best-checkpoint epoch on LR schedule" (ax_lr): same-epoch balls are
#   already vertically de-collided by the existing stem-stacking (`_ep_rank` /
#   `_COLL_H`), but that only catches *exact* epoch ties - two dataset sizes
#   landing on adjacent epochs can still be close enough on screen for their
#   labels to overlap a ball or each other. A hand-rolled fixed/pixel-cluster
#   offset was tried here first and looked worse (labels drifted from their
#   ball, arrangement felt arbitrary), so label placement now uses the
#   `adjustText` package (`from adjustText import adjust_text`, added as a
#   project dependency) instead: each `n=` label starts just right of its own
#   ball (`ax_lr.text(_best_ep + 0.3, _ball_y, ...)` - a small manual nudge so
#   no label starts exactly on top of its own marker), and a single
#   `adjust_text(...)` call after `xlim`/`ylim` are fixed iteratively repels
#   every label away from all ball *coordinates* (passed via `x=_ball_xs,
#   y=_ball_ys` - adjustText's `objects=` option, which repels from the actual
#   scatter artists' rendered extents, crashed with a NaN error in this
#   adjustText version, hence plain coordinates) and from every other label,
#   with `force_static` raised to (0.5, 0.8) so it pushes firmly away from
#   neighboring balls, not just its own. Draws a thin leader line
#   (`arrowprops`) back to its ball when a label had to move noticeably. This
#   is a general solution - it keeps working if the number/spacing of dataset
#   sizes changes later, unlike a hand-tuned offset.
# - "OOD matched mIoU vs. LR at checkpoint" (ax_ood): the `n=` labels are
#   offset straight right (`xytext=(5, 0), va="center"`) rather than up-right,
#   which used to sit diagonally above the marker.

# %%
def load_run_val_stats(run, n):
    """(epoch, val_loss, val_dice) arrays for (run, n), deduped by epoch (see EXP9)."""
    f = os.path.join(RUNS_LOG_BASE, run, f"n{n}", "logs", "val_stats.json")
    if not os.path.exists(f):
        return None
    rows = [json.loads(line) for line in open(f)]
    d = pd.DataFrame(rows).drop_duplicates(subset="Trainer/epoch", keep="last").sort_values("Trainer/epoch")
    return d["Trainer/epoch"].to_numpy(), d["Loss/total/val"].to_numpy(), d["Metrics/val_dice"].to_numpy()


_EXP11_REF_RUN    = "run_011"
_EXP11_BASE_LR    = 7.102526614544882e-05
_EXP11_MAX_EPOCHS = 30
_EXP11_HL         = 108   # highlighted n
_EXP11_NS         = [n for n in DATASET_SIZES]


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
    _v = load_run_val_stats(_EXP11_REF_RUN, _n)
    if _v is not None:
        _exp11_val[_n] = {"epoch": _v[0], "val_loss": _v[1], "val_dice": _v[2]}

_exp11_ood = {n: v["matched"] for n, v in per_run_all[_EXP11_REF_RUN].items()}
_exp11_id  = {n: v["matched"] for n, v in per_run_all_id[_EXP11_REF_RUN].items()}

print(f"Reference run: {_EXP11_REF_RUN}")
print(f"{'n':>5}  {'stopped':>7}  {'best_ep':>7}  {'dice@ckpt':>10}  {'lr@ckpt':>12}  {'lr@stop':>12}")
print("-" * 62)
for _n in _EXP11_NS:
    if _n not in _exp11_val:
        continue
    _v       = _exp11_val[_n]
    _stopped = int(_v["epoch"][-1])
    _bi      = int(np.argmax(_v["val_dice"]))
    _best_ep = int(_v["epoch"][_bi])
    _dice    = _v["val_dice"][_bi]
    print(f"{_n:>5}  {_stopped:>7}  {_best_ep:>7}  {_dice:>10.4f}"
          f"  {_exp11_lr(_best_ep):>12.4e}  {_exp11_lr(_stopped):>12.4e}")

# %%
_exp11_cmap   = plt.cm.Blues
_exp11_n_cols = {_n: _exp11_cmap(0.35 + 0.60 * i / max(1, len(_EXP11_NS) - 1))
                 for i, _n in enumerate(_EXP11_NS)}
_EXP11_HL_COL = "tomato"
_EXP11_CKPT_C = "steelblue"

with plt.rc_context(THESIS_RC):
    fig, axes = plt.subplots(2, 2, figsize=(THESIS_WIDTH_IN, THESIS_WIDTH_IN * 0.78))
    ax_loss, ax_dice = axes[0]
    ax_lr,   ax_ood  = axes[1]

    _ep_curve = np.linspace(0.01, _EXP11_MAX_EPOCHS, 400)
    _lr_curve = np.array([_exp11_lr(e) for e in _ep_curve])

    _lr_pts = []   # (best_ep, n, is_hl)
    _ood_xs, _ood_ys, _ood_texts = [], [], []   # handed to adjustText below

    for _n in _EXP11_NS:
        if _n not in _exp11_val:
            continue
        _v   = _exp11_val[_n]
        _col = _EXP11_HL_COL if _n == _EXP11_HL else _exp11_n_cols[_n]
        _lw  = 2.0 if _n == _EXP11_HL else 0.9
        _al  = 1.0 if _n == _EXP11_HL else 0.55
        _zo  = 5   if _n == _EXP11_HL else 2

        ax_loss.plot(_v["epoch"], _v["val_loss"], color=_col, linewidth=_lw, alpha=_al, zorder=_zo)
        ax_dice.plot(_v["epoch"], _v["val_dice"], color=_col, linewidth=_lw, alpha=_al, zorder=_zo)

        _bi      = int(np.argmax(_v["val_dice"]))
        _best_ep = int(_v["epoch"][_bi])
        _hl      = (_n == _EXP11_HL)
        _star_ms = 140 if _hl else 45

        ax_loss.scatter([_best_ep], [_v["val_loss"][_bi]], marker="*", color=_col,
                         s=_star_ms, zorder=_zo + 1, edgecolors="black" if _hl else "none", linewidths=0.6)
        ax_dice.scatter([_best_ep], [_v["val_dice"][_bi]], marker="*", color=_col,
                         s=_star_ms, zorder=_zo + 1, edgecolors="black" if _hl else "none", linewidths=0.6)

        _lr_pts.append((_best_ep, _n, _hl))

        _ckpt_lr = _exp11_lr(_best_ep)
        _ms      = 42 if _hl else 16
        # OOD = circle, ID = triangle, same per-n color for both - both series
        # feed the same adjustText call below so labels dodge every point,
        # whichever domain it belongs to.
        for _domain_vals, _marker in [(_exp11_ood, "o"), (_exp11_id, "^")]:
            if _n in _domain_vals and _domain_vals[_n] == _domain_vals[_n]:  # not NaN
                _val = _domain_vals[_n]
                ax_ood.scatter([_ckpt_lr], [_val], color=_col, s=_ms, zorder=_zo, marker=_marker,
                               edgecolors="black" if _hl else "none", linewidths=1.0)
                _ood_xs.append(_ckpt_lr)
                _ood_ys.append(_val)
                _ood_texts.append(ax_ood.text(_ckpt_lr, _val, str(_n), va="center", fontsize=6,
                                   color="black" if _hl else _col, fontweight="bold" if _hl else "normal"))

    # --- ax_lr: lollipop chart, one stem per n at its best-checkpoint epoch ---
    _lr_pts.sort(key=lambda t: (t[0], t[1]))
    _ep_rank = defaultdict(int)
    _STEM_H  = _EXP11_BASE_LR * 0.15
    _COLL_H  = _EXP11_BASE_LR * 0.13

    _ball_tops = []
    _ball_xs   = []   # ball coordinates, handed to adjustText below so labels dodge every ball
    _ball_ys   = []
    _lr_texts  = []   # matplotlib Text artists, handed to adjustText below
    for _best_ep, _n, _hl in _lr_pts:
        _rank   = _ep_rank[_best_ep]
        _ep_rank[_best_ep] += 1
        _lr_val = _exp11_lr(_best_ep)
        _ball_y = _lr_val + _STEM_H + _rank * _COLL_H
        _ball_tops.append(_ball_y)
        _ball_xs.append(_best_ep)
        _ball_ys.append(_ball_y)
        _col    = _EXP11_HL_COL if _hl else _EXP11_CKPT_C
        ax_lr.plot([_best_ep, _best_ep], [_lr_val, _ball_y],
                   color=_col, linewidth=1.5 if _hl else 0.8, alpha=0.7, zorder=2)
        ax_lr.scatter([_best_ep], [_ball_y], color=_col, s=42 if _hl else 18, zorder=3, marker="o",
                      edgecolors="black" if _hl else "none", linewidths=1.0)
        _lr_texts.append(ax_lr.text(_best_ep + 0.3, _ball_y, str(_n), va="center", fontsize=7,
                   color="black" if _hl else "dimgray", fontweight="bold" if _hl else "normal"))

    ax_lr.plot(_ep_curve, _lr_curve, color="gray", linewidth=1.2, alpha=0.5, zorder=1)
    ax_lr.set_xlim(left=0)
    ax_lr.set_ylim(bottom=0, top=max(_ball_tops) + _EXP11_BASE_LR * 0.60)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_title("(3) Best epoch on LR schedule")
    ax_lr.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_lr.grid(True, alpha=0.3)

    # auto-repel n= labels away from every ball (not just their own) and from each other
    adjust_text(_lr_texts, x=_ball_xs, y=_ball_ys, ax=ax_lr,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                only_move={"text": "xy"}, expand=(1.2, 1.2), force_static=(0.235, 0.4))
                # only_move={"text": "y"}, expand=(1.45, 1.5), force_static=(0.01, 0.2))

    ax_lr.legend(handles=[
        Line2D([0], [0], color="gray", linewidth=1.2, label="LR schedule (30 ep)"),
        Line2D([0], [0], marker="o", color=_EXP11_CKPT_C, linewidth=0, markersize=5, label="Best-ckpt epoch"),
        Line2D([0], [0], marker="o", color=_EXP11_HL_COL, linewidth=0, markersize=6,
               markeredgecolor="black", label=f"n={_EXP11_HL} (highlighted)"),
    ], fontsize=6)

    ax_ood.set_xlabel("LR at best-checkpoint epoch")
    ax_ood.set_ylabel("Matched mIoU")
    ax_ood.set_title("(4) OOD/ID matched mIoU vs. LR at checkpoint")
    ax_ood.grid(True, alpha=0.3)
    _ood_id_legend = ax_ood.legend(handles=[
        Line2D([0], [0], marker="o", color="dimgray", linewidth=0, markersize=5, label="OOD"),
        Line2D([0], [0], marker="^", color="dimgray", linewidth=0, markersize=5, label="ID"),
    ], fontsize=6, loc="lower right")
    ax_ood.add_artist(_ood_id_legend)

    # auto-repel n= labels away from every point and from each other; also
    # keeps labels inside the axes (fixes the one that used to hang off the
    # right edge) since adjust_text's ensure_inside_axes defaults to True
    _OOD_EXPAND       = (1.75, 1.75)   # tune & re-run: label-vs-label/point clearance multiplier
    _OOD_FORCE_STATIC = (0.5, 0.8)   # tune & re-run: how hard labels get pushed off other points
    adjust_text(_ood_texts, x=_ood_xs, y=_ood_ys, ax=ax_ood,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                only_move={"text": "xy"}, expand=_OOD_EXPAND, force_static=_OOD_FORCE_STATIC)

    # nudge the x-axis "1e-5"-style scale multiplier upward so it clears the
    # x-axis label below it - tune _OOD_UNIT_DY_PT (points) and re-run
    _OOD_UNIT_DY_PT = 14
    _ood_offset_txt = ax_ood.xaxis.get_offset_text()
    _ood_offset_txt.set_transform(mtransforms.offset_copy(
        _ood_offset_txt.get_transform(), fig=fig, x=0, y=_OOD_UNIT_DY_PT, units="points"))

    _ld_legend = [
        Line2D([0], [0], color=_EXP11_HL_COL, linewidth=2.0, label=f"n={_EXP11_HL}"),
        Line2D([0], [0], color="steelblue", linewidth=0.9, alpha=0.55, label="Other sizes"),
    ]
    for _ax, _title, _ylabel in [
        (ax_loss, "(1) Val loss per epoch", "Loss/total/val"),
        (ax_dice, "(2) Val dice per epoch", "Metrics/val_dice"),
    ]:
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_title)
        _ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        _ax.grid(True, alpha=0.3)
        _ax.legend(handles=_ld_legend, fontsize=7)

    fig.suptitle(f"Training dynamics per dataset size - reference run {_EXP11_REF_RUN} "
                 f"(seed {RUNS[_EXP11_REF_RUN]}, n={_EXP11_HL} highlighted)")
    fig.tight_layout()
    EXPORT_NAME_EXP11 = f"14_training_dynamics_lr_schedule_{_EXP11_REF_RUN}_seed{RUNS[_EXP11_REF_RUN]}_n{_EXP11_HL}"
    export_figure(fig, EXPORT_NAME_EXP11)
    plt.show()

# %% [markdown]
# ## EXP13: IoU heatmap by position within tile
#
# Tests whether detections near a tile's edge score lower IoU than ones near
# the center (a plausible effect of tile-boundary clipping and partial
# context). Ground-truth roof centroids are recovered directly from each
# tile's pixel-space segmentation mask in `output_data/ground_truth_npz/` -
# not re-derived from geo-coordinates - then binned onto a coarse grid over
# the tile footprint (normalized to [0,1] x [0,1]) and colored by mean strict
# IoU (0 for a missed roof).
#
# Ground truth is identical across every run and dataset size, so the
# position lookup is built once per domain and joined against `df_all`/
# `df_id_all`; every heatmap below therefore pools all 5 runs (~6000
# detections per size for OOD). One heatmap is drawn per dataset size, OOD and
# ID as two separate 3x3-grid figures (a full second grid, not side-by-side
# panels - nine small-multiples per domain already fill the figure width).
#
# `HEATMAP_BINS` sets the output grid resolution (cells per tile side) -
# raise it for a "sharper"-looking image, lower it to render faster.
# `HEATMAP_SIGMA_FRAC` is the Gaussian smoothing bandwidth (as a fraction of
# the tile side) that turns individual detections into a continuous field -
# this, not HEATMAP_BINS, is what actually controls how "zoomed in" the
# visible detail is; raise it for a smoother/blurrier picture, lower it for a
# sharper/noisier one. `HEATMAP_MIN_WEIGHT` blanks out (grey) any region with
# too little *effective* (smoothed) detection weight nearby to trust.

# %%
def build_tile_position_lookup(gt_dirs):
    """(tile_source, feature_id) -> normalized (cx, cy) within its tile, from GT
    pixel masks in one or more region directories (list of dirs, e.g. the two
    ID regions, or a single-element list for OOD)."""
    rows = []
    for gt_dir in gt_dirs:
        for f in sorted(glob.glob(os.path.join(gt_dir, "*_gt.npz"))):
            tile_source = os.path.basename(f).replace("_gt.npz", "_eval.npz")
            for m in np.load(f, allow_pickle=True)["masks"].tolist():
                seg  = m["segmentation"]
                h, w = seg.shape
                ys, xs = np.where(seg)
                if len(xs) == 0:
                    continue
                rows.append({
                    "tile_source": tile_source,
                    "feature_id":  int(m["feature_id"]),
                    "cx_norm":     float(xs.mean()) / w,
                    "cy_norm":     float(ys.mean()) / h,
                })
    return pd.DataFrame(rows)


_tile_pos = build_tile_position_lookup([GT_DIR])
print(f"OOD position lookup: {len(_tile_pos)} ground-truth roofs across "
      f"{_tile_pos['tile_source'].nunique()} tiles")

_d13 = df_all.merge(_tile_pos, on=["tile_source", "feature_id"], how="inner")
print(f"Matched {len(_d13)} / {len(df_all)} df_all rows to a tile position "
      f"({len(df_all) - len(_d13)} unmatched)")

_tile_pos_id = build_tile_position_lookup(list(GT_DIRS_ID.values()))
print(f"ID position lookup: {len(_tile_pos_id)} ground-truth roofs across "
      f"{_tile_pos_id['tile_source'].nunique()} tiles")

_d13_id = df_id_all.merge(_tile_pos_id, on=["tile_source", "feature_id"], how="inner")
print(f"Matched {len(_d13_id)} / {len(df_id_all)} df_id_all rows to a tile position "
      f"({len(df_id_all) - len(_d13_id)} unmatched)")

# %%
# A single roof centroid can only ever land in one cell, so a hard bin count
# collapses fast as HEATMAP_BINS grows: at 1024x1024 (>1M cells) with ~6000
# detections per size, ~99.4% of cells are empty regardless of
# HEATMAP_MIN_COUNT - that's why cranking HEATMAP_BINS up while keeping the
# old nearest-neighbor binning produced an all-grey plot, not a bug.
#
# To make a genuinely fine-resolution ("per-pixel") view meaningful, this
# uses kernel smoothing (a Gaussian-weighted local average, i.e. Nadaraya-
# Watson kernel regression) instead of a hard histogram: every detection
# contributes a soft Gaussian blob rather than a single cell, so nearby
# points blend into a continuous field no matter how fine the output grid is.
from scipy.ndimage import gaussian_filter

HEATMAP_BINS       = 1024   # output grid resolution per tile side - can go as fine as you like
HEATMAP_SIGMA_FRAC = 0.035  # Gaussian smoothing bandwidth, as a fraction of the tile side
                             # (auto-scales with HEATMAP_BINS; raise for smoother/blurrier, lower for sharper/noisier)
HEATMAP_MIN_WEIGHT = 3.0    # bins with less than this *effective* (smoothed) detection weight are left blank (grey)


def compute_heatmap(sub, bins, sigma_frac):
    """Gaussian-smoothed mean strict IoU over (cx_norm, cy_norm) -> (mean_iou, raw_count), each [bins, bins]."""
    edges = np.linspace(0, 1, bins + 1)
    sum_iou, _, _ = np.histogram2d(sub["cx_norm"], sub["cy_norm"], bins=[edges, edges],
                                    weights=sub["iou_score"])
    count, _, _   = np.histogram2d(sub["cx_norm"], sub["cy_norm"], bins=[edges, edges])

    sigma = max(sigma_frac * bins, 1e-6)   # bandwidth in bin units, scaled to the current resolution
    # gaussian_filter conserves total mass, so spreading a fixed number of
    # detections over ever-more cells (as `bins` grows) dilutes every cell's
    # value proportionally to 1/sigma^2. Rescale by 2*pi*sigma^2 (the inverse
    # of a 2D Gaussian kernel's peak height) to convert that density back
    # into "effective nearby detection count" units, so HEATMAP_MIN_WEIGHT
    # means the same thing regardless of HEATMAP_BINS/HEATMAP_SIGMA_FRAC.
    norm = 2 * np.pi * sigma ** 2
    sum_smooth    = gaussian_filter(sum_iou, sigma=sigma, mode="constant") * norm
    weight_smooth = gaussian_filter(count,   sigma=sigma, mode="constant") * norm

    mean_iou = np.divide(sum_smooth, weight_smooth, out=np.full_like(sum_smooth, np.nan),
                          where=weight_smooth > HEATMAP_MIN_WEIGHT)
    # imshow expects [row=y, col=x]; histogram2d returns [x, y] -> transpose.
    return mean_iou.T, count.T   # count returned raw (unsmoothed) - used only for the printed n_roofs annotation


_cmap = plt.cm.viridis.copy()
_cmap.set_bad("lightgray")
_nrows, _ncols = 3, 3


def _draw_position_heatmap_grid(d13, area_label, filename):
    heat_ns   = sorted(d13["n"].unique())
    heatmaps  = {n: compute_heatmap(d13[d13["n"] == n], HEATMAP_BINS, HEATMAP_SIGMA_FRAC) for n in heat_ns}
    all_valid = np.concatenate([hm[~np.isnan(hm)] for hm, _ in heatmaps.values()])
    vmin, vmax = float(np.percentile(all_valid, 2)), float(np.percentile(all_valid, 98))

    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(_nrows, _ncols, figsize=(THESIS_WIDTH_IN, THESIS_WIDTH_IN * 0.92),
                                  sharex=True, sharey=True, constrained_layout=True)

        im = None
        for ax, n in zip(axes.ravel(), heat_ns):
            mean_iou, count = heatmaps[n]
            im = ax.imshow(np.ma.masked_invalid(mean_iou), origin="lower", extent=(0, 1, 0, 1),
                            cmap=_cmap, vmin=vmin, vmax=vmax, aspect="equal")
            n_used = int(count.sum())
            label  = "Zero-shot" if n == 0 else f"n={n}"
            ax.set_title(f"{label}  (n_roofs={n_used})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

        for ax in axes.ravel()[len(heat_ns):]:
            ax.axis("off")

        fig.supxlabel("Position within tile (normalized, left -> right)", fontsize=9)
        fig.supylabel("Position within tile (normalized, bottom -> top)", fontsize=9)
        fig.suptitle(f"Gaussian smoothed Mean IoU by position within tile, per dataset size - {area_label}\n"
                     f"({HEATMAP_BINS}x{HEATMAP_BINS} grid, Gaussian sigma={HEATMAP_SIGMA_FRAC:.3f}x tile side, "
                     f"{len(RUN_NAMES)} runs pooled)", fontsize=10)
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=30)
        cbar.set_label("Mean IoU (strict: 0 if missed)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        export_figure(fig, filename)
        plt.show()


_draw_position_heatmap_grid(_d13, f"OOD ({OOD_N_TILES} tiles/run)", "15_iou_heatmap_by_tile_position_ood")
_draw_position_heatmap_grid(_d13_id, f"ID ({ID_TEST_TILES_STR} tiles/run)",
                             "15_iou_heatmap_by_tile_position_id")

# %% [markdown]
# ## EXP14: Missed-rooftop footprints by position within tile
#
# Same per-size, all-runs-pooled small-multiples layout as EXP13, but instead
# of binning/averaging into a heatmap this draws the *actual polygon
# footprint* of every missed ground-truth roof (`matched == False`) at its
# true position within its tile - individual shapes, not an aggregate. All
# categories are included (standard, cut, artifact - no filtering). Every
# polygon is drawn in tomato at low alpha and simply stacked, so a location
# missed by more than one of the 5 runs shows up as overlapping, more opaque
# tomato rather than being averaged away. OOD and ID each get their own full
# 3x3 grid (same reasoning as EXP13: nine small-multiples per domain already
# fill the figure).

# %%
import cv2


def build_tile_polygon_lookup(gt_dirs):
    """(tile_source, feature_id) -> Nx2 array of normalized [0,1] contour points,
    from GT pixel masks in one or more region directories.

    Also returns the tile pixel size (w, h), assumed constant across all tiles.
    """
    lookup = {}
    tile_size = None
    for gt_dir in gt_dirs:
        for f in sorted(glob.glob(os.path.join(gt_dir, "*_gt.npz"))):
            tile_source = os.path.basename(f).replace("_gt.npz", "_eval.npz")
            for m in np.load(f, allow_pickle=True)["masks"].tolist():
                seg  = m["segmentation"]
                h, w = seg.shape
                if tile_size is None:
                    tile_size = (w, h)
                contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea).squeeze(axis=1)   # [N, 2] in (x, y) pixel coords
                if contour.ndim != 2 or len(contour) < 3:
                    continue
                lookup[(tile_source, int(m["feature_id"]))] = contour.astype(float) / np.array([w, h])
    return lookup, tile_size


_tile_poly, TILE_SIZE_PX = build_tile_polygon_lookup([GT_DIR])
print(f"OOD polygon lookup: {len(_tile_poly)} ground-truth roof footprints, tile size {TILE_SIZE_PX}px")

_missed = df_all[~df_all["matched"]]
print(f"OOD missed detections pooled across all runs/sizes: {len(_missed)}")

_tile_poly_id, TILE_SIZE_PX_id = build_tile_polygon_lookup(list(GT_DIRS_ID.values()))
print(f"ID polygon lookup: {len(_tile_poly_id)} ground-truth roof footprints, tile size {TILE_SIZE_PX_id}px")

_missed_id = df_id_all[~df_id_all["matched"]]
print(f"ID missed detections pooled across all runs/sizes: {len(_missed_id)}")

# %%
from matplotlib.patches import Polygon as MplPolygon, Rectangle, Patch
from matplotlib.collections import PatchCollection

MISSED_ALPHA = 0.35   # per-polygon alpha; overlapping instances (missed by >1 run) stack up darker

# A missed roof right on a tile edge is often a sliver a few pixels wide - at
# small-multiples scale that's sub-pixel and effectively invisible, even
# though it's a real, correctly-placed detection. Rather than inflating the
# polygon itself (which would misrepresent its true extent), any polygon
# whose bounding box is smaller than CALLOUT_VISIBILITY_FRAC of the tile side
# also gets a tomato callout box drawn around it, at exactly the true
# CALLOUT_VISIBILITY_FRAC size - this is the actual prompt bounding-box size
# used by the algorithm, so it must be drawn at true scale (no inflation for
# visibility, which would misrepresent whether the prompt box could have
# covered the roof).
CALLOUT_VISIBILITY_FRAC = 0.03125    # true bbox smaller than this (either dimension) triggers a callout box; also the drawn callout-box side, as a fraction of the tile side
CALLOUT_BOX_LW          = 1.1
PIXEL_SIZE_TILE         = 1024
BOX_SIZE_TILE           = int(CALLOUT_VISIBILITY_FRAC * PIXEL_SIZE_TILE)   # 32px - true prompt-box side at 1024x1024 tile size

def draw_missed_footprints(ax, sub, tile_w, tile_h, tile_poly=None):
    """Draw each missed roof's true polygon (tomato fill) plus a tomato callout
    box around any that are too small to reliably see at this scale.
    Returns (n_polygons_drawn, n_callout_boxes)."""
    if tile_poly is None:
        tile_poly = _tile_poly
    patches, boxes, n_drawn = [], [], 0
    for row in sub.itertuples(index=False):
        poly = tile_poly.get((row.tile_source, row.feature_id))
        if poly is None:
            continue
        poly_px = poly * [tile_w, tile_h]
        patches.append(MplPolygon(poly_px, closed=True))
        n_drawn += 1

        x0, y0 = poly_px.min(axis=0)
        x1, y1 = poly_px.max(axis=0)
        w, h = x1 - x0, y1 - y0
        if min(w, h) < CALLOUT_VISIBILITY_FRAC * min(tile_w, tile_h):
            bw, bh = max(w, CALLOUT_VISIBILITY_FRAC * tile_w), max(h, CALLOUT_VISIBILITY_FRAC * tile_h)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bx0 = min(max(cx - bw / 2, 0), tile_w - bw)
            by0 = min(max(cy - bh / 2, 0), tile_h - bh)
            boxes.append(Rectangle((bx0, by0), bw, bh))

    if patches:
        ax.add_collection(PatchCollection(patches, facecolor="tomato", edgecolor="black",
                                           linewidth=0.8, alpha=MISSED_ALPHA, zorder=2))
    if boxes:
        ax.add_collection(PatchCollection(boxes, facecolor="none", edgecolor="tomato",
                                           linewidth=CALLOUT_BOX_LW, zorder=3))
    return n_drawn, len(boxes)


# Shared legend handles for both the small-multiples grid and the detail plot below.
EXP14_LEGEND_HANDLES = [
    Patch(facecolor="tomato", edgecolor="black", linewidth=0.8, alpha=MISSED_ALPHA,
          label="Missed ground-truth roof (true polygon)"),
    Patch(facecolor="none", edgecolor="tomato", linewidth=CALLOUT_BOX_LW,
          label=f"Callout box (bbox narrower than {BOX_SIZE_TILE}px side)"),
]


# --- human-tunable header spacing, all in inches (re-run the cell after editing) ---
# fig.tight_layout(rect=...) was tried here first: it adds its own automatic
# padding around the suptitle that ignores whatever rect you give it, so the
# gap it produced didn't track these knobs at all. Instead we measured, once,
# exactly what plain fig.tight_layout() (no legend, no rect - the original,
# correctly-sized layout) settles on for the grid's own position
# (EXP14_BASELINE_* below, from fig.subplotpars after that call) and now
# reproduce that EXACT geometry in absolute inches via fig.subplots_adjust(),
# so the 3x3 grid is pixel-for-pixel the same size it was before the legend
# existed, no matter how the header knobs are tuned. The three knobs insert as
# extra height between the (also unchanged, absolute) suptitle and the grid:
#   EXP14_TITLE_LEGEND_GAP_IN - gap between the suptitle and the legend
#   EXP14_LEGEND_BLOCK_IN     - vertical space reserved for the legend row
#   EXP14_LEGEND_GRID_GAP_IN  - gap between the legend and the first subplot titles
# Note: EXP14_BASELINE_TOP (below) is the axes box edge, not the visible top of
# the subplot title text - the title itself pokes up above that edge by about
# 0.15-0.2in on its own (its own text height + matplotlib's default titlepad),
# so EXP14_LEGEND_GRID_GAP_IN needs to clear that on top of whatever blank
# space you actually want between the legend and the title text.
EXP14_TITLE_LEGEND_GAP_IN = 0.05
EXP14_LEGEND_BLOCK_IN     = 0.05
EXP14_LEGEND_GRID_GAP_IN  = 0.25

# Measured from fig.subplotpars after a plain fig.tight_layout() with no rect,
# no legend, at figsize=(THESIS_WIDTH_IN, THESIS_WIDTH_IN * 0.92) - re-measure
# and update these (see scratch script in the conversation) if THESIS_WIDTH_IN,
# the number of grid rows/cols, or the subplot title/suptitle text changes
# enough to shift the layout.
EXP14_BASELINE_FIG_H_IN = THESIS_WIDTH_IN * 0.92
EXP14_BASELINE_LEFT     = 0.0683
EXP14_BASELINE_RIGHT    = 0.9762
EXP14_BASELINE_BOTTOM   = 0.0742
EXP14_BASELINE_TOP      = 0.8680   # top of the axes grid (bottom edge of the old suptitle margin)
EXP14_BASELINE_HSPACE   = 0.25   # space between graphs 2365
EXP14_BASELINE_WSPACE   = 0.0   # space between graphs 0830
EXP14_BASELINE_TITLE_Y  = 0.98     # matplotlib's default fig.suptitle() y-anchor

_exp14_bottom_in     = EXP14_BASELINE_BOTTOM * EXP14_BASELINE_FIG_H_IN                        # unchanged: supxlabel margin
_exp14_grid_in       = (EXP14_BASELINE_TOP - EXP14_BASELINE_BOTTOM) * EXP14_BASELINE_FIG_H_IN     # unchanged: the 3x3 grid itself
_exp14_top_margin_in = (1 - EXP14_BASELINE_TOP) * EXP14_BASELINE_FIG_H_IN                     # unchanged: whole suptitle block (title text + its own top gap)
_exp14_title_gap_in  = (1 - EXP14_BASELINE_TITLE_Y) * EXP14_BASELINE_FIG_H_IN                 # unchanged: gap between fig top edge and the suptitle's own anchor

_exp14_fig_h_in = (_exp14_bottom_in + _exp14_grid_in
                   + EXP14_LEGEND_GRID_GAP_IN + EXP14_LEGEND_BLOCK_IN + EXP14_TITLE_LEGEND_GAP_IN
                   + _exp14_top_margin_in)
_exp14_bottom    = _exp14_bottom_in / _exp14_fig_h_in
_exp14_top       = (_exp14_bottom_in + _exp14_grid_in) / _exp14_fig_h_in
_exp14_legend_y  = (_exp14_bottom_in + _exp14_grid_in + EXP14_LEGEND_GRID_GAP_IN) / _exp14_fig_h_in
_exp14_title_y   = 1 - _exp14_title_gap_in / _exp14_fig_h_in

def _draw_missed_footprints_grid(missed_df, heat_ns, tile_poly, area_label, filename):
    with plt.rc_context(THESIS_RC):
        fig, axes = plt.subplots(_nrows, _ncols, figsize=(THESIS_WIDTH_IN, _exp14_fig_h_in),
                                  sharex=True, sharey=True)

        for ax, n in zip(axes.ravel(), heat_ns):
            sub = missed_df[missed_df["n"] == n]
            n_drawn, n_boxes = draw_missed_footprints(ax, sub, 1.0, 1.0, tile_poly)   # normalized [0,1] tile space

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            label = "Zero-shot" if n == 0 else f"n={n}"
            ax.set_title(f"{label}  (missed={n_drawn}, {n_boxes} flagged)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

        for ax in axes.ravel()[len(heat_ns):]:
            ax.axis("off")

        fig.supxlabel("Position within tile (normalized, left -> right)", fontsize=9)
        fig.supylabel("Position within tile (normalized, bottom -> top)", fontsize=9)
        fig.suptitle(f"Missed ground-truth roof polygons - {area_label}\n"
                     f"(all categories, {len(RUN_NAMES)} runs stacked, "
                     f"red frame = too small to see at true scale)", fontsize=10, y=_exp14_title_y)
        # loc="lower center" + bbox_to_anchor anchors the legend's BOTTOM edge at
        # _exp14_legend_y, so EXP14_LEGEND_GRID_GAP_IN directly controls the gap to
        # the grid below it, independent of EXP14_TITLE_LEGEND_GAP_IN above it.
        fig.legend(handles=EXP14_LEGEND_HANDLES, loc="lower center", bbox_to_anchor=(0.5, _exp14_legend_y),
                   bbox_transform=fig.transFigure, ncol=2, fontsize=8, frameon=False)
        fig.subplots_adjust(left=EXP14_BASELINE_LEFT, right=EXP14_BASELINE_RIGHT,
                             bottom=_exp14_bottom, top=_exp14_top,
                             hspace=EXP14_BASELINE_HSPACE, wspace=EXP14_BASELINE_WSPACE)

        export_figure(fig, filename)
        plt.show()


_draw_missed_footprints_grid(_missed, sorted(_d13["n"].unique()), _tile_poly,
                              f"OOD ({OOD_N_TILES} tiles/run)", "16_missed_rooftop_footprints_ood")
_draw_missed_footprints_grid(_missed_id, sorted(_d13_id["n"].unique()), _tile_poly_id,
                              f"ID ({ID_TEST_TILES_STR} tiles/run)",
                              "16_missed_rooftop_footprints_id")

# %% [markdown]
# ### EXP14 detail: single dataset size, enlarged
#
# Same data and style as the EXP14 grid above, but rendered as one large
# standalone panel with real axis ticks, for close inspection or direct
# inclusion in the thesis. Set `DETAIL_N` to any of the sizes shown above
# (0 = zero-shot) and re-run the cell.

# %%
DETAIL_N = 108   # dataset size to inspect closely - one of: 0, 5, 10, 15, 25, 40, 65, 108


def plot_missed_footprints_detail(n, missed_df, tile_size_px, tile_poly, area_label, filename=None):
    tile_w, tile_h = tile_size_px
    sub = missed_df[missed_df["n"] == n]

    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(THESIS_WIDTH_IN, THESIS_WIDTH_IN))
        n_drawn, n_boxes = draw_missed_footprints(ax, sub, tile_w, tile_h, tile_poly)
        ax.set_xlim(0, tile_w); ax.set_ylim(0, tile_h)
        ax.set_aspect("equal")
        ax.set_xticks(np.linspace(0, tile_w, 5))
        ax.set_yticks(np.linspace(0, tile_h, 5))
        ax.set_xlabel("Position within tile (pixels, left -> right)")
        ax.set_ylabel("Position within tile (pixels, bottom -> top)")
        label = "Zero-shot" if n == 0 else f"n={n}"
        ax.set_title(f"Missed ground-truth roof polygons - {area_label}, {label}\n"
                     f"(all categories, {len(RUN_NAMES)} runs stacked, "
                     f"missed={n_drawn}, {n_boxes} flagged with box {BOX_SIZE_TILE}x{BOX_SIZE_TILE}px)")
        ax.grid(True, alpha=0.2)
        # loc="best" doesn't reliably avoid the Patch/PatchCollection artists
        # here (its overlap heuristic favors Line2D data); missed roofs
        # cluster along the tile edges (top/bottom) across all dataset sizes,
        # so the vertical center is reliably empty - pin it there instead.
        ax.legend(handles=EXP14_LEGEND_HANDLES, fontsize=7, loc="center")
        fig.tight_layout()
        export_figure(fig, filename or f"17_missed_rooftop_footprints_detail_n{n}")
        plt.show()

    return n_drawn


plot_missed_footprints_detail(DETAIL_N, _missed, TILE_SIZE_PX, _tile_poly,
                               f"OOD ({OOD_N_TILES} tiles/run)", filename=f"17_missed_rooftop_footprints_detail_ood_n{DETAIL_N}")
plot_missed_footprints_detail(DETAIL_N, _missed_id, TILE_SIZE_PX_id, _tile_poly_id,
                               f"ID ({ID_TEST_TILES_STR} tiles/run)",
                               filename=f"17_missed_rooftop_footprints_detail_id_n{DETAIL_N}")

# %%
