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
# # D4 - Performance score tables (thesis results section), train/val/test split
#
# This notebook loads the already-computed `unified_metrics.csv` files (same
# source as `D3_clean_final_graphs_tvt.py`, no re-inference, no re-matching)
# and turns them into **copy-paste-ready LaTeX tables** for the thesis:
# Precision, Recall, F1, Dice, mean IoU, median IoU and Hausdorff Distance.
#
# Unlike the original `D4_performance_scores.py` (single OOD-only scarcity
# sweep, runs `run_001..run_006`), the 5 runs analysed here (`run_007..011`)
# each carve a train/50% - val/25% - test/25% split out of a 205-tile
# annotated pool (`ceu_paz` + `cantidio_sampaio`), reconstructed on the fly
# from each run's `split_seed` via `training/dataset/split_utils.py` (the
# exact stdlib-only helper the training pipeline itself uses - vendored, not
# reimplemented, so the split is guaranteed byte-identical to what each run
# actually trained/tested on). That gives every table here **two evaluation
# areas** instead of one:
#
# - **OOD** (out-of-distribution) - `santa_madalena`, never split, the whole
#   area always held out. Same role as in the original sweep.
# - **ID** (in-distribution) - the pooled held-out **test** subset of
#   `ceu_paz`/`cantidio_sampaio` (~51 tiles/run out of the 205-tile pool, a
#   different 51 for every run since the split_seed differs). `n` still means
#   *training*-set size here, but the eval set is always that run's own fixed
#   test split, not a function of `n` - so, unlike OOD, ID's `n=0` ("zero-shot")
#   genuinely varies run-to-run too (each run's zero-shot predictions are
#   filtered down to a different set of test tiles), and gets a real +/- std
#   across the 5 seeds instead of being a single deterministic point.
#
# Every table below reports OOD and ID **side by side as rows** (an "Area"
# column), not as extra columns - the two domains aren't really comparable
# point-for-point (different tile counts, different train/test relationship),
# and best/second-best highlighting is computed independently within each
# Area block. This keeps column count identical to the original single-area
# tables regardless of how much data goes in.
#
# ## What each metric means here, and why
#
# The underlying evaluation (`D_functions.match_annotations_to_predictions` +
# `resolve_double_claims`) does a **global greedy 1-to-1 IoU matching** between
# every ground-truth roof and every candidate SAM2 proposal mask on a tile, with
# no minimum-IoU acceptance threshold - any positive overlap can "win" a match.
# That single design choice shapes which metrics are meaningful here:
#
# - **Object-level Recall** (fraction of ground-truth roofs matched at all) is
#   directly meaningful - this is exactly what D3/EXP3 already plots.
# - **Object-level Precision** (of everything the model *proposed*, how much was
#   a real roof) is **not** computed here. SAM2's automatic mask generator
#   proposes many candidate regions per tile that have nothing to do with
#   buildings (roads, vegetation, shadows, yards) - the ground truth only
#   annotates roofs. Counting every one of those as a "false positive" would
#   conflate "correctly not-a-roof" with genuine false detections and produce a
#   near-meaningless, near-zero number, unless scored properly as a
#   confidence-threshold sweep (see the COCO-style AP note at the bottom).
# - Instead, **Precision / Recall / F1 (= Dice) / IoU are reported at the
#   pixel level, pooled over matched roofs**: TP = overlapping pixels, FP =
#   predicted-mask pixels outside the ground-truth roof, FN = ground-truth
#   pixels the prediction missed (a fully-missed roof contributes FN = its
#   whole area, TP = FP = 0). This is the standard formulation used throughout
#   segmentation literature (e.g. medical-imaging Dice/precision/recall) and it
#   *is* well-defined here, because both masks being compared already refer to
#   the same roof.
# - **Dice and pixel-level F1 are the identical formula**
#   (`2*TP / (2*TP + FP + FN)`); both names are reported since the thesis may
#   want to cite either convention, but they are the same number.
# - **Hausdorff Distance** additionally needs the actual mask geometry (not
#   just areas), so it is computed straight from the saved `*_gt.npz` /
#   `*_masks.npz` boundary pixels, converted from pixels to metres using each
#   tile's own ground-sample distance (`area_meters_sq / area_px`). Plain
#   Hausdorff Distance is well known to be dominated by single-pixel outliers,
#   so alongside it we also report **HD95** (95th-percentile boundary
#   distance) and **ASSD** (Average Symmetric Surface Distance, the mean
#   boundary-to-boundary distance) - both are near-universal companions to HD
#   in the segmentation literature for exactly that reason.
# - Cut (tile-boundary-clipped) roofs cap the best achievable IoU/Dice/HD
#   regardless of prediction quality (same point D3 makes), so the primary
#   "Standard" table below excludes them; a secondary "All roofs" table
#   includes them for completeness. Artifact-flagged annotations are excluded
#   throughout, matching the rest of the pipeline (`filter_artifacts=True`).

# %% [markdown]
# ## Configuration

# %%
import os
import sys
import functools
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm

# --- Dataset sizes used in the scarcity sweep (108 = full dataset, given the
# --- train/val/test split; see the "wanted -> true tiles: 108 -> 102 CAPPED"
# --- note in D3_clean_final_graphs_tvt.py) ---
DATASET_SIZES = [5, 10, 15, 25, 40, 65, 108]
FULL_N = max(DATASET_SIZES)

# --- Out-of-distribution evaluation area (never split - whole area held out) ---
AREA_NAME = "santa_madalena"
REAL_AREA_NAME = "Rio Claro II"  # Real name of santa madalena region, confusion in original communication.

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

# --- IoU acceptance threshold for the stricter "Recall@IoU" detection metric ---
IOU_ACCEPT_THRESHOLD = 0.5

# --- Path resolution ---
CURRENT_DIR = os.getcwd()
CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR

IOU_RESULTS_DIR = os.path.join(CODE_DIR, "output_data", "iou_results")
RUNS_IOU_BASE   = os.path.join(IOU_RESULTS_DIR, "runs")
ZEROSHOT_CSV    = os.path.join(IOU_RESULTS_DIR, f"{AREA_NAME}_unified_metrics.csv")

# Ground-truth pixel masks - fixed across every run/size, used for the
# Hausdorff/ASSD boundary computation. One dir per evaluation area/region.
GT_DIR     = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", AREA_NAME)
GT_DIRS_ID = {r: os.path.join(CODE_DIR, "output_data", "ground_truth_npz", r) for r in ID_REGIONS}
GT_DIRS    = {AREA_NAME: GT_DIR, **GT_DIRS_ID}

# Predicted proposal masks (SAM2 automatic mask generator output), per
# run/size/area, also used for the Hausdorff/ASSD boundary computation.
# (Not rsync'd to the same drive as the old run_001-006 sweep - these newer
# runs' predictions live on the cluster checkout instead.)
RUNS_PRED_BASE       = "/home/ced/Documents/unicluster/data/predictions"
ZEROSHOT_PRED_DIR    = os.path.join(CODE_DIR, "output_data", "masks", f"{AREA_NAME}_masks")
ZEROSHOT_PRED_DIR_ID = {r: os.path.join(CODE_DIR, "output_data", "masks", f"{r}_masks") for r in ID_REGIONS}

# --- Output locations ---
TABLES_DIR = os.path.join(CODE_DIR, "output_data", "tables", "perf_scores")
os.makedirs(TABLES_DIR, exist_ok=True)

HD_CACHE_DIR = os.path.join(IOU_RESULTS_DIR, "hausdorff_cache")
os.makedirs(HD_CACHE_DIR, exist_ok=True)
# Separate (and separately-named) from the old run_001-006 cache: different
# runs, different row schema (adds a "region" column), so keeping them apart
# avoids mixing stale rows from the old sweep into these tables.
HD_CACHE_PATH_OOD = os.path.join(HD_CACHE_DIR, f"{AREA_NAME}_tvt_hd_cache.csv")
HD_CACHE_PATH_ID  = os.path.join(HD_CACHE_DIR, "id_tvt_hd_cache.csv")

# `compute_split`/`list_annotated_tiles_png` are the exact stdlib-only helpers
# the training pipeline itself uses to build the train/val/test split from a
# split_seed - vendored on the cluster checkout, imported here (not
# reimplemented) so the reconstructed split is guaranteed identical to what
# each run actually trained/tested on.
FAVELA_PNG_DIR = os.path.join(CODE_DIR, "output_data", "favela_png")
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
print(f"Tables dir:   {TABLES_DIR}")

_id_test_counts = [sum(len(v) for v in get_test_tiles(r).values()) for r in RUN_NAMES]
ID_TEST_TILES_STR = (str(_id_test_counts[0]) if min(_id_test_counts) == max(_id_test_counts)
                      else f"{min(_id_test_counts)}-{max(_id_test_counts)}")
print(f"ID test-split size: {ID_TEST_TILES_STR} tiles/run (out of a 205-tile pool, test_fraction={TEST_FRACTION})")


# %% [markdown]
# ## Load the combined multi-run dataframes
#
# Identical loading strategy to `D3_clean_final_graphs_tvt.py`: reads every
# already-computed CSV (no re-evaluation) into one long dataframe per area.
#
# - `df_all` (OOD, `santa_madalena`): one row per (ground-truth roof, run)
#   pair for every training size, plus the single deterministic zero-shot rows
#   at n=0 (tagged `run="zero_shot"`).
# - `df_id_all` (ID, pooled `ceu_paz` + `cantidio_sampaio`): same shape, but
#   the zero-shot rows at n=0 are tagged with the *real* run name and vary
#   run-to-run, since each run's own held-out test split differs.
#
# Both carry a `region` column (`AREA_NAME` for OOD; the actual source region
# for ID) used later to pick the right ground-truth/prediction directory for
# the Hausdorff computation.

# %%
def run_csv_path(run, n, area=AREA_NAME):
    return os.path.join(RUNS_IOU_BASE, run, f"n{n}", f"n{n}_{area}_unified_metrics.csv")


def load_run_df(run, n, area=AREA_NAME):
    """Load one (run, n, area) unified-metrics CSV, or None if it doesn't exist."""
    path = run_csv_path(run, n, area)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d["n"]      = n
    d["run"]    = run
    d["seed"]   = RUNS[run]
    d["region"] = area
    return d


def _normalize_metrics_df(df):
    df["iou_score"]   = df["iou_score"].fillna(0.0)
    df["matched"]     = df["matched"].astype(bool)
    df["is_cut"]      = df["is_cut"].fillna(False).astype(bool)
    df["is_artifact"] = df["is_artifact"].fillna(False).astype(bool)
    return df


def load_all_runs_df(sizes=DATASET_SIZES):
    """Concatenate every (run, n) OOD CSV plus the zero-shot CSV into one long dataframe."""
    frames = []
    for run in RUN_NAMES:
        for n in sizes:
            d = load_run_df(run, n, AREA_NAME)
            if d is None:
                print(f"{run} n{n} {AREA_NAME}: CSV not found - skipping")
                continue
            frames.append(d)

    zs = pd.read_csv(ZEROSHOT_CSV)
    zs["n"], zs["run"], zs["seed"], zs["region"] = 0, "zero_shot", None, AREA_NAME
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
        d = d[tile_names.isin(test_tiles[region])].copy()
        d["region"] = region
        frames.append(d)
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


# %%
df_all    = load_all_runs_df()
df_id_all = load_all_runs_df_id()

print(f"df_all (OOD):    {len(df_all)} rows    | sizes: {sorted(df_all['n'].unique())} | runs: {RUN_NAMES}")
print(f"df_id_all (ID):  {len(df_id_all)} rows | sizes: {sorted(df_id_all['n'].unique())} | runs: {RUN_NAMES}")

# %% [markdown]
# ## Derived per-roof pixel confusion counts
#
# From `area_px` (ground-truth area), `overlap_pixels` (intersection with the
# matched prediction) and `iou_score`, the matched prediction's own area falls
# out algebraically (`union = overlap / iou`, `pred_area = union - gt_area +
# overlap`), so no extra mask loading is needed for the pixel-level
# Precision/Recall/F1/Dice/IoU tables below - only the Hausdorff/ASSD section
# further down needs to touch the raw `.npz` mask arrays.
#
# - `TP_px` = overlap pixels (correctly predicted roof pixels).
# - `FN_px` = ground-truth pixels not covered by the prediction (the whole roof
#   area for a missed roof).
# - `FP_px` = predicted pixels outside the ground-truth roof (`union - gt_area`);
#   0 for a missed roof, since no prediction can be attributed to it.
#
# Applied identically to both `df_all` (OOD) and `df_id_all` (ID).

# %%
def add_confusion_columns(df):
    df["TP_px"] = np.where(df["matched"], df["overlap_pixels"], 0)
    df["FN_px"] = np.where(df["matched"], df["area_px"] - df["overlap_pixels"], df["area_px"])
    df["FP_px"] = np.where(
        df["matched"],
        df["overlap_pixels"] / df["iou_score"].replace(0, np.nan) - df["area_px"],
        0.0,
    )
    df["FP_px"] = df["FP_px"].fillna(0.0).clip(lower=0.0)
    df["pred_area_px"] = df["TP_px"] + df["FP_px"]
    return df


df_all    = add_confusion_columns(df_all)
df_id_all = add_confusion_columns(df_id_all)


# %% [markdown]
# ## Metric aggregation helpers

# %%
def confusion_metrics(sub):
    """Pixel-level Precision/Recall/F1(=Dice) + object-level Recall/Recall@thr + mIoU, for one (run, n) subset."""
    tp, fp, fn = sub["TP_px"].sum(), sub["FP_px"].sum(), sub["FN_px"].sum()
    n_gt      = len(sub)
    n_matched = int(sub["matched"].sum())
    n_matched_thr = int((sub["matched"] & (sub["iou_score"] >= IOU_ACCEPT_THRESHOLD)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall_px = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1        = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

    return {
        "n_gt":              n_gt,
        "n_matched":         n_matched,
        "n_missed":          n_gt - n_matched,
        "recall_obj":        n_matched / n_gt if n_gt else np.nan,
        "recall_obj_at_thr": n_matched_thr / n_gt if n_gt else np.nan,
        "precision_px":      precision,
        "recall_px":         recall_px,
        "f1_dice":           f1,
        "mean_iou":          float(sub["iou_score"].mean()),
        "median_iou":        float(sub["iou_score"].median()),
    }


def build_metrics_per_run(df, mask_fn, sizes=DATASET_SIZES, zero_shot_mode="pseudo_run"):
    """Tidy dataframe: one row per (run, n) with all metrics.

    `zero_shot_mode`:
    - "pseudo_run" (OOD): zero-shot is a single deterministic extra "run"
      (`run="zero_shot"`) evaluated only at n=0; real runs are evaluated at
      every size in `sizes`.
    - "per_run" (ID): every run has its own n=0 zero-shot subset (filtered to
      that run's test split), so n=0 is just another size alongside `sizes`
      for each of the 5 real runs - no separate pseudo-run needed.
    """
    rows = []
    if zero_shot_mode == "pseudo_run":
        run_iter = RUN_NAMES + ["zero_shot"]
        ns_for = lambda run: [0] if run == "zero_shot" else sizes
    elif zero_shot_mode == "per_run":
        run_iter = RUN_NAMES
        ns_for = lambda run: [0] + list(sizes)
    else:
        raise ValueError(zero_shot_mode)

    for run in run_iter:
        for n in ns_for(run):
            sub = mask_fn(df[(df["run"] == run) & (df["n"] == n)])
            if sub.empty:
                continue
            row = {"run": run, "n": n}
            row.update(confusion_metrics(sub))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_n(metrics_per_run):
    """mean +/- std across the 5 seeds per n."""
    agg = metrics_per_run.drop(columns="run").groupby("n").agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index().sort_values("n")


METRIC_LABELS = {
    "recall_obj":        "Recall (IoU>0)",
    "recall_obj_at_thr": f"Recall (IoU>={IOU_ACCEPT_THRESHOLD})",
    "precision_px":      "Precision (px)",
    "recall_px":         "Recall (px)",
    "f1_dice":           "F1 / Dice",
    "mean_iou":          "Mean IoU",
    "median_iou":        "Median IoU",
    "hd_m":              "Hausdorff (m)",
    "hd95_m":             "HD95 (m)",
    "assd_m":            "ASSD (m)",
}

# Whether a higher value is better for each metric - controls which end of a
# column gets bold/underline in rank_highlight(). Distance metrics (Hausdorff
# family) are the only "lower is better" ones here.
METRIC_HIGHER_IS_BETTER = {
    "recall_obj":        True,
    "recall_obj_at_thr": True,
    "precision_px":      True,
    "recall_px":         True,
    "f1_dice":           True,
    "mean_iou":          True,
    "median_iou":        True,
    "hd_m":              False,
    "hd95_m":            False,
    "assd_m":            False,
}


def fmt_cell(mean, std, decimals=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        # No +/- part (e.g. zero-shot, a single deterministic run) - pad with
        # an invisible block the same width as " $\pm$ 0.000" so the digits
        # still line up under rows that do have one, despite right-alignment.
        pad = f"{0:.{decimals}f}"
        return f"{mean:.{decimals}f}\\phantom{{ $\\pm$ {pad}}}"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def rank_highlight(cells, means, higher_is_better=True):
    """Bold the best value in a column, underline the runner-up.

    `means` drives the ranking (point estimate only, std is ignored for
    ranking purposes); `cells` are the already-formatted display strings to
    wrap. Ties share the best rank (both get bolded); NaN/"--" cells are left
    untouched.
    """
    s = pd.Series(list(means), dtype=float)
    ranks = s.rank(method="min", ascending=not higher_is_better)
    out = []
    for cell, r in zip(cells, ranks):
        if pd.isna(r) or cell == "--":
            out.append(cell)
        elif r == 1:
            out.append(f"\\textbf{{{cell}}}")
        elif r == 2:
            out.append(f"\\underline{{{cell}}}")
        else:
            out.append(cell)
    return out


def rank_highlight_closest(cells, values, target=1.0):
    """Like rank_highlight, but "best" means closest to `target` (e.g. a
    predicted/ground-truth area ratio of 1.0), not highest or lowest."""
    dist = (pd.Series(list(values), dtype=float) - target).abs()
    return rank_highlight(cells, -dist, higher_is_better=True)


def build_display_table(summary, metric_keys, n_gt_col="n_gt_mean"):
    x_labels = ["zs" if n == 0 else str(n) for n in summary["n"]]
    out = pd.DataFrame({"$T$size": x_labels})
    out["# GT roofs"] = summary[n_gt_col].round().astype(int).values
    for key in metric_keys:
        means = summary[f"{key}_mean"]
        stds  = summary.get(f"{key}_std", pd.Series(np.nan, index=summary.index))
        cells = [fmt_cell(m, s) for m, s in zip(means, stds)]
        out[METRIC_LABELS[key]] = rank_highlight(cells, means, METRIC_HIGHER_IS_BETTER[key])
    return out


def build_display_table_by_area(summaries_by_area, metric_keys, n_gt_col="n_gt_mean"):
    """Wrap `build_display_table`, called independently per area (so best/
    second-best highlighting never compares OOD against ID - the two aren't
    really commensurable), then stack the results with an "Area" row label.
    """
    parts = []
    for area_label, summary in summaries_by_area.items():
        t = build_display_table(summary, metric_keys, n_gt_col)
        t.insert(0, "Area", area_label)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def build_headline_table(rows, keys):
    out = pd.DataFrame({"Model": [r["Model"] for r in rows]})
    out["# GT roofs"] = [r["n_gt"] for r in rows]
    for key in keys:
        means = [r[key][0] for r in rows]
        stds  = [r[key][1] for r in rows]
        cells = [fmt_cell(m, s) for m, s in zip(means, stds)]
        out[METRIC_LABELS[key]] = rank_highlight(cells, means, METRIC_HIGHER_IS_BETTER[key])
    return out


def build_headline_table_by_area(rows_by_area, keys):
    """Same "independent ranking per area" wrapping as `build_display_table_by_area`."""
    parts = []
    for area_label, rows in rows_by_area.items():
        t = build_headline_table(rows, keys)
        t.insert(0, "Area", area_label)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def export_table(df_display, name, caption, label, scale_to_width=True):
    """Save a display-ready dataframe as .tex (booktabs) + .csv, and print it.

    `scale_to_width=True` wraps the tabular in \\resizebox{\\textwidth}{!}{...}
    so wide tables (many metric columns) shrink to fit the page instead of
    overflowing the margin - requires \\usepackage{graphicx} in the thesis
    preamble (near-universally already there). For narrow tables this
    stretches them to an oversized, distracting font instead, so pass
    `scale_to_width=False` to just center them at their natural size.
    \\centering is always added regardless.
    """
    tex_path = os.path.join(TABLES_DIR, f"{name}.tex")
    csv_path = os.path.join(TABLES_DIR, f"{name}.csv")
    df_display.to_csv(csv_path, index=False)
    latex = df_display.to_latex(
        index=False, escape=False, na_rep="--",
        caption=caption, label=f"tab:{label}",
        column_format="l" + "r" * (len(df_display.columns) - 1),
    )
    latex = latex.replace("\\begin{table}", "\\begin{table}\n\\centering", 1)
    if scale_to_width:
        latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}", 1)
        latex = latex.replace("\\end{tabular}", "\\end{tabular}}", 1)
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"Exported {name}.{{tex,csv}} -> {TABLES_DIR}")
    print(latex)
    return df_display


# %% [markdown]
# ## Subset definitions
#
# - **Standard** = non-cut & non-artifact roofs. Primary "model quality" table -
#   matches the "clean" subset D3/EXP2-EXP5 uses for its quality analysis.
# - **All roofs** = non-artifact roofs (cut roofs included). Secondary
#   "operational" table - a deployment scenario also has to deal with roofs
#   that straddle a tile boundary, even though their IoU ceiling is capped by
#   the annotation clipping rather than by model quality.
#
# Artifact-flagged annotations are excluded from both, matching the rest of
# the pipeline (`filter_artifacts=True`).

# %%
mask_standard = lambda d: d[~d["is_cut"] & ~d["is_artifact"]]
mask_all      = lambda d: d[~d["is_artifact"]]

summary_std_ood = summarize_per_n(build_metrics_per_run(df_all,    mask_standard, DATASET_SIZES, "pseudo_run"))
summary_std_id  = summarize_per_n(build_metrics_per_run(df_id_all, mask_standard, DATASET_SIZES, "per_run"))
summary_all_ood = summarize_per_n(build_metrics_per_run(df_all,    mask_all,      DATASET_SIZES, "pseudo_run"))
summary_all_id  = summarize_per_n(build_metrics_per_run(df_id_all, mask_all,      DATASET_SIZES, "per_run"))

# Change this list for output
# _METRIC_ORDER = ["recall_obj", "recall_obj_at_thr", "precision_px", "recall_px", "f1_dice", "mean_iou", "median_iou"]
_METRIC_ORDER = ["precision_px", "recall_px", "f1_dice", "mean_iou", "median_iou"]

# %% [markdown]
# ## Table 1: Segmentation performance - Standard roofs (non-cut, non-artifact)
#
# The headline "how good is the model" table, OOD and ID stacked as an "Area"
# row group (independently ranked - see `build_display_table_by_area`):
# pixel-level Precision/Recall/F1(=Dice), then mean/median IoU. Every cell
# pools pixels across all matched roofs for that (Area, n), then reports mean
# +/- std across the 5 seeds. For OOD, "zs" is a single deterministic run
# (no +/-); for ID, "zs" already varies across the 5 seeds' own held-out test
# splits, so it gets a real +/- too.

# %%
table1 = build_display_table_by_area({"OOD": summary_std_ood, "ID": summary_std_id}, _METRIC_ORDER)
export_table(
    table1, "table1_segmentation_standard",
    caption=(f"Segmentation performance vs. training-set size, standard roofs only (non-cut, non-artifact). "
             f"OOD = {REAL_AREA_NAME} (fixed holdout area, evaluated in full at every n); "
             f"ID = pooled {', '.join(ID_REGIONS)} (each run's own held-out test split, "
             f"{ID_TEST_TILES_STR} tiles/run). Mean $\\pm$ std across {len(RUN_NAMES)} training seeds; OOD "
             f"zero-shot is a single deterministic run, ID zero-shot already varies per seed's own test split. "
             f"Best/second-best highlighted independently within each Area."),
    label="perf-standard",
)

# %% [markdown]
# ## Table 2: Segmentation performance - All roofs (incl. cut, excl. artifacts)

# %%
table2 = build_display_table_by_area({"OOD": summary_all_ood, "ID": summary_all_id}, _METRIC_ORDER)
export_table(
    table2, "table2_segmentation_all",
    caption=(f"Segmentation performance vs. training-set size, all roofs including tile-boundary-clipped ones "
             f"(artifacts excluded). OOD = {REAL_AREA_NAME}; ID = pooled {', '.join(ID_REGIONS)} held-out test "
             f"split ({ID_TEST_TILES_STR} tiles/run). Mean $\\pm$ std across {len(RUN_NAMES)} training seeds; "
             f"best/second-best highlighted independently within each Area."),
    label="perf-all",
)

# %% [markdown]
# ## "Best" fine-tuned size
#
# Same selection rule as D3: the dataset size with the highest actual mean
# matched mIoU across the 5 runs (not just the largest sub-full-dataset size).
# OOD and ID get their own "best n" - the two domains can favor different
# sizes, and conflating them would misrepresent whichever one a given table
# is actually reporting.

# %%
def best_n_by_matched_iou(df, sizes=DATASET_SIZES):
    matched_iou_by_n = {
        n: df[(df["run"].isin(RUN_NAMES)) & (df["n"] == n) & ~df["is_cut"] & ~df["is_artifact"] & df["matched"]]
            .groupby("run")["iou_score"].mean().mean()
        for n in sizes
    }
    return max(matched_iou_by_n, key=matched_iou_by_n.get), matched_iou_by_n


_best_n,    _matched_iou_by_n    = best_n_by_matched_iou(df_all)
_best_n_id, _matched_iou_by_n_id = best_n_by_matched_iou(df_id_all)
print(f"'Best' fine-tuned size (highest mean matched mIoU, standard roofs): OOD n={_best_n}, ID n={_best_n_id}")

# %% [markdown]
# ## Hausdorff Distance & ASSD (boundary metrics, metres)
#
# Unlike the metrics above, Hausdorff Distance needs the actual mask geometry,
# not just areas, so this section reads the raw `*_gt.npz` / `*_masks.npz`
# boundary pixels directly. Only computed for **matched, standard (non-cut,
# non-artifact) roofs** - a missed roof has no predicted boundary to compare
# against, and a cut roof's ground-truth boundary is truncated by the tile
# edge, which would corrupt the distance (that truncation is an annotation
# artifact, not a model error).
#
# `region` (set at load time - `AREA_NAME` for OOD rows, the source region for
# ID rows) picks the right ground-truth/prediction directory per roof, so one
# code path serves both areas. Results are cached to disk (`HD_CACHE_PATH_OOD`
# / `HD_CACHE_PATH_ID`) since this touches every mask array on disk;
# re-running the notebook after the first pass is instant.

# %%
def _boundary_mask(seg_u8):
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    b = np.zeros(seg_u8.shape, dtype=bool)
    for c in contours:
        b[c[:, 0, 1], c[:, 0, 0]] = True
    return b


def _hd_metrics_px(gt_seg, pred_seg, pad=10):
    """Symmetric boundary distances (pixels): (Hausdorff, HD95, ASSD)."""
    ys, xs = np.where(gt_seg | pred_seg)
    if ys.size == 0:
        return np.nan, np.nan, np.nan
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gt_seg.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gt_seg.shape[1])
    g = gt_seg[y0:y1, x0:x1].astype(np.uint8)
    p = pred_seg[y0:y1, x0:x1].astype(np.uint8)
    gb, pb = _boundary_mask(g), _boundary_mask(p)
    if not gb.any() or not pb.any():
        return np.nan, np.nan, np.nan
    dt_from_gt   = distance_transform_edt(~gb)
    dt_from_pred = distance_transform_edt(~pb)
    all_d = np.concatenate([dt_from_gt[pb], dt_from_pred[gb]])
    return float(all_d.max()), float(np.percentile(all_d, 95)), float(all_d.mean())


def pred_dir_for(run, n, region):
    """n=0 always means zero-shot (OOD's deterministic pseudo-run, or ID's
    per-run-filtered subset of the same static zero-shot masks) - both read
    from the fixed zero-shot mask dir for that region rather than a
    per-(run,n) training checkpoint's predictions."""
    if n == 0:
        return ZEROSHOT_PRED_DIR if region == AREA_NAME else ZEROSHOT_PRED_DIR_ID[region]
    return os.path.join(RUNS_PRED_BASE, run, f"n{n}", region)


def compute_hd_for_run_n(df, run, n):
    """Standard, matched roofs for one (run, n): list of dicts with HD/HD95/ASSD in px and metres."""
    sub = mask_standard(df[(df["run"] == run) & (df["n"] == n) & df["matched"]])
    if sub.empty:
        return []

    rows = []
    for (region, tile_src), g in sub.groupby(["region", "tile_source"]):
        base = tile_src.replace("_eval.npz", "")
        gt_path   = os.path.join(GT_DIRS[region], base + "_gt.npz")
        pred_path = os.path.join(pred_dir_for(run, n, region), base + "_masks.npz")
        if not (os.path.exists(gt_path) and os.path.exists(pred_path)):
            continue
        gt_masks   = np.load(gt_path, allow_pickle=True)["masks"].tolist()
        pred_masks = np.load(pred_path, allow_pickle=True)["masks"].tolist()
        gt_lookup  = {m["feature_id"]: m for m in gt_masks}

        for row in g.itertuples():
            gt_m = gt_lookup.get(row.feature_id)
            if gt_m is None or row.pred_mask_index < 0 or row.pred_mask_index >= len(pred_masks):
                continue
            pred_m = pred_masks[int(row.pred_mask_index)]
            hd_px, hd95_px, assd_px = _hd_metrics_px(gt_m["segmentation"], pred_m["segmentation"])
            if np.isnan(hd_px):
                continue
            px_size_m = float(np.sqrt(row.area_meters_sq / row.area_px)) if row.area_px > 0 else np.nan
            rows.append({
                "run": run, "n": n, "region": region, "tile_source": tile_src, "feature_id": row.feature_id,
                "hd_m": hd_px * px_size_m, "hd95_m": hd95_px * px_size_m, "assd_m": assd_px * px_size_m,
            })
    return rows


def get_or_build_hd_cache(cache_path, df, expected_pairs, rebuild=False):
    if os.path.exists(cache_path) and not rebuild:
        hd_df = pd.read_csv(cache_path)
        cached_pairs = set(zip(hd_df["run"], hd_df["n"]))
    else:
        hd_df = None
        cached_pairs = set()

    missing = sorted(expected_pairs - cached_pairs)
    if missing:
        print(f"Computing Hausdorff/ASSD for {len(missing)} missing (run, n) pairs -> {os.path.basename(cache_path)}")
        new_rows = []
        for run, n in tqdm(missing, desc=f"Hausdorff sweep ({os.path.basename(cache_path)})"):
            new_rows.extend(compute_hd_for_run_n(df, run, n))
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            hd_df = pd.concat([hd_df, new_df], ignore_index=True) if hd_df is not None and len(hd_df) else new_df
        elif hd_df is None:
            hd_df = pd.DataFrame(columns=["run", "n", "region", "tile_source", "feature_id", "hd_m", "hd95_m", "assd_m"])
        hd_df.to_csv(cache_path, index=False)
        print(f"Cache updated -> {cache_path}")
    else:
        print(f"Loaded Hausdorff/ASSD cache ({len(hd_df)} roofs) -> {cache_path}")
    return hd_df


# %%
REBUILD_HD_CACHE = False  # set True to force a full recompute instead of reusing the cache

_expected_pairs_ood = {(r, n) for r in RUN_NAMES for n in DATASET_SIZES} | {("zero_shot", 0)}
_expected_pairs_id  = {(r, n) for r in RUN_NAMES for n in [0] + DATASET_SIZES}

hd_df_ood = get_or_build_hd_cache(HD_CACHE_PATH_OOD, df_all,    _expected_pairs_ood, REBUILD_HD_CACHE)
hd_df_id  = get_or_build_hd_cache(HD_CACHE_PATH_ID,  df_id_all, _expected_pairs_id,  REBUILD_HD_CACHE)

# %% [markdown]
# ### Table 3: Hausdorff Distance & ASSD vs. training-set size

# %%
def summarize_hd(hd_df):
    per_run = hd_df.groupby(["run", "n"]).agg(
        n_roofs=("hd_m", "size"), hd_m=("hd_m", "mean"),
        hd95_m=("hd95_m", "mean"), assd_m=("assd_m", "mean"),
    ).reset_index()
    agg = per_run.drop(columns="run").groupby("n").agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index().sort_values("n")


def build_hd_table(hd_summary):
    t = pd.DataFrame({"$T$size": ["zs" if n == 0 else str(n) for n in hd_summary["n"]]})
    t["# roofs"] = hd_summary["n_roofs_mean"].round().astype(int)
    for key, disp in [("hd_m", "Hausdorff (m)"), ("hd95_m", "HD95 (m)"), ("assd_m", "ASSD (m)")]:
        means = hd_summary[f"{key}_mean"]
        stds  = hd_summary[f"{key}_std"]
        cells = [fmt_cell(m, s) for m, s in zip(means, stds)]
        t[disp] = rank_highlight(cells, means, higher_is_better=False)
    return t


hd_summary_ood = summarize_hd(hd_df_ood)
hd_summary_id  = summarize_hd(hd_df_id)

table3 = pd.concat([
    build_hd_table(hd_summary_ood).assign(Area="OOD"),
    build_hd_table(hd_summary_id).assign(Area="ID"),
], ignore_index=True)
table3 = table3[["Area"] + [c for c in table3.columns if c != "Area"]]

export_table(
    table3, "table3_hausdorff",
    caption=(f"Boundary distance metrics vs. training-set size, standard (non-cut, non-artifact), matched "
             f"roofs only. OOD = {REAL_AREA_NAME}; ID = pooled {', '.join(ID_REGIONS)} held-out test split "
             f"({ID_TEST_TILES_STR} tiles/run). Mean $\\pm$ std across {len(RUN_NAMES)} training seeds; "
             f"best/second-best highlighted independently within each Area."),
    label="hausdorff",
    scale_to_width=False,
)

# %% [markdown]
# ## Table 4: Headline comparison - zero-shot vs. best fine-tuned vs. full dataset
#
# Same "zero-shot vs. best fine-tuned vs. full dataset" comparison as before,
# standard roofs only, OOD and ID stacked as an "Area" row group (each ranked
# independently), split into three narrower tables (4a: pixel
# Precision/Recall/F1-Dice, 4b: mean/median IoU, 4c: boundary distances) so
# each one reads comfortably at its natural size instead of one wide table.
# "Best" = highest mean matched mIoU across the 5 seeds (n=`_best_n`/
# `_best_n_id`, see above, own value per area); the full-dataset row
# (n=`FULL_N`) is included separately whenever it differs from that area's
# best size, since "use all the data" is itself a relevant operating point
# even if it isn't the empirical optimum.

# %%
def _row_data_for_n(summary_std, hd_summary, n, label):
    """Raw (mean, std) per metric for one n - kept numeric so the headline
    tables below can rank/highlight across their rows before formatting."""
    s = summary_std[summary_std["n"] == n].iloc[0]
    data = {"Model": label, "n_gt": int(round(s["n_gt_mean"]))}
    for key in _METRIC_ORDER:
        data[key] = (s[f"{key}_mean"], s.get(f"{key}_std", np.nan))
    hd_row = hd_summary[hd_summary["n"] == n]
    if len(hd_row):
        hd_row = hd_row.iloc[0]
        data["hd_m"]   = (hd_row["hd_m_mean"], hd_row["hd_m_std"])
        data["hd95_m"] = (hd_row["hd95_m_mean"], hd_row["hd95_m_std"])
        data["assd_m"] = (hd_row["assd_m_mean"], hd_row["assd_m_std"])
    else:
        data["hd_m"] = data["hd95_m"] = data["assd_m"] = (np.nan, np.nan)
    return data


def _headline_rows_for(summary_std, hd_summary, best_n):
    rows = [
        _row_data_for_n(summary_std, hd_summary, 0, "Zero-shot"),
        _row_data_for_n(summary_std, hd_summary, best_n, f"Best fine-tuned (n={best_n})"),
    ]
    if best_n != FULL_N:
        rows.append(_row_data_for_n(summary_std, hd_summary, FULL_N, f"Full dataset (n={FULL_N})"))
    return rows


_headline_rows_ood = _headline_rows_for(summary_std_ood, hd_summary_ood, _best_n)
_headline_rows_id  = _headline_rows_for(summary_std_id,  hd_summary_id,  _best_n_id)

_headline_caption_base = (f"standard roofs only (non-cut, non-artifact). OOD = {REAL_AREA_NAME}; "
                           f"ID = pooled {', '.join(ID_REGIONS)} held-out test split ({ID_TEST_TILES_STR} "
                           f"tiles/run). Mean $\\pm$ std across {len(RUN_NAMES)} training seeds. Bold = best "
                           f"in column, underlined = second-best, ranked independently within each Area.")

# %% [markdown]
# ### Table 4a: Precision / Recall / F1-Dice

# %%
table4a = build_headline_table_by_area({"OOD": _headline_rows_ood, "ID": _headline_rows_id},
                                        ["precision_px", "recall_px", "f1_dice"])
export_table(
    table4a, "table4a_headline_precision_recall_f1",
    caption=f"Headline pixel Precision/Recall/F1(=Dice), {_headline_caption_base}",
    label="headline-prf", scale_to_width=False,
)

# %% [markdown]
# ### Table 4b: Mean / Median IoU

# %%
table4b = build_headline_table_by_area({"OOD": _headline_rows_ood, "ID": _headline_rows_id},
                                        ["mean_iou", "median_iou"])
export_table(
    table4b, "table4b_headline_iou",
    caption=f"Headline mean/median IoU, {_headline_caption_base}",
    label="headline-iou", scale_to_width=False,
)

# %% [markdown]
# ### Table 4c: Hausdorff / HD95 / ASSD

# %%
table4c = build_headline_table_by_area({"OOD": _headline_rows_ood, "ID": _headline_rows_id},
                                        ["hd_m", "hd95_m", "assd_m"])
export_table(
    table4c, "table4c_headline_hausdorff",
    caption=f"Headline boundary distance metrics, {_headline_caption_base}",
    label="headline-hd", scale_to_width=False,
)

# %% [markdown]
# ## Table 5: Size-stratified quality - zero-shot vs. best fine-tuned
#
# Splits each area's own "Standard" subset into Small/Medium/Large tertiles by
# roof area (same tertile scheme as D3/EXP5) to see *where* zero-shot's lower
# overall pixel recall (Table 1/4a) actually comes from - whether it's spread
# evenly across roof sizes or concentrated at one end. OOD and ID get their
# **own** tertile thresholds (computed separately from each area's own
# standard-roof area distribution - the two roof-size distributions aren't
# assumed comparable) and their own best-n, stacked as an outer "Area" group
# on top of the existing Size-bin/Model rows.
#
# Alongside pixel Precision/Recall/Mean IoU, "Median area ratio" is
# median(predicted mask area / ground-truth area) over matched roofs in that
# bin: near 1.0 means the mask's extent matches the annotation; well below
# 1.0 means systematic under-coverage (well above 1.0 would mean
# over-coverage). Best value per column is highlighted **within each
# (Area, size bin) pair** (zero-shot vs. fine-tuned only, not compared across
# bins or areas); for the area-ratio column, "best" means closest to 1.0
# rather than highest or lowest.

# %%
_SIZE_LABELS = ["Small", "Medium", "Large"]


def add_size_bins(df):
    df = df.copy()
    t33, t67 = df["area_px"].quantile(1 / 3), df["area_px"].quantile(2 / 3)
    df["size_bin"] = pd.cut(df["area_px"], bins=[0.0, t33, t67, float("inf")],
                             labels=_SIZE_LABELS, include_lowest=True)
    return df, t33, t67


_std_all_ood, _t33_ood, _t67_ood = add_size_bins(mask_standard(df_all))
_std_all_id,  _t33_id,  _t67_id  = add_size_bins(mask_standard(df_id_all))


def _size_bin_metrics(sub):
    tp, fp, fn = sub["TP_px"].sum(), sub["FP_px"].sum(), sub["FN_px"].sum()
    matched = sub[sub["matched"]]
    return {
        "n_gt":         len(sub),
        "precision_px": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "recall_px":    tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "mean_iou":     float(sub["iou_score"].mean()),
        "area_ratio":   float((matched["pred_area_px"] / matched["area_px"]).median()) if len(matched) else np.nan,
    }


def _zs_sub_ood(df, bin_):
    return df[(df["run"] == "zero_shot") & (df["size_bin"] == bin_)]


def _zs_sub_id(df, bin_):
    return df[(df["run"].isin(RUN_NAMES)) & (df["n"] == 0) & (df["size_bin"] == bin_)]


def build_size_rows(std_all, best_n, zs_selector):
    rows = []
    for bin_ in _SIZE_LABELS:
        zs_sub = zs_selector(std_all, bin_)
        ft_sub = std_all[(std_all["run"].isin(RUN_NAMES)) & (std_all["n"] == best_n) & (std_all["size_bin"] == bin_)]
        rows.append({"Size bin": bin_, "Model": "Zero-shot", **_size_bin_metrics(zs_sub)})
        rows.append({"Size bin": bin_, "Model": f"Best fine-tuned (n={best_n})", **_size_bin_metrics(ft_sub)})
    return pd.DataFrame(rows)


def format_size_table(size_df):
    t = pd.DataFrame({"Size bin": size_df["Size bin"], "Model": size_df["Model"], "# roofs": size_df["n_gt"]})
    for key, disp, higher in [
        ("precision_px", "Precision (px)", True),
        ("recall_px",    "Recall (px)",    True),
        ("mean_iou",     "Mean IoU",       True),
    ]:
        t[disp] = ""
        for bin_ in _SIZE_LABELS:
            mask  = size_df["Size bin"] == bin_
            cells = [f"{v:.3f}" if pd.notna(v) else "--" for v in size_df.loc[mask, key]]
            t.loc[mask, disp] = rank_highlight(cells, size_df.loc[mask, key], higher)

    t["Median area ratio (pred/GT)"] = ""
    for bin_ in _SIZE_LABELS:
        mask  = size_df["Size bin"] == bin_
        cells = [f"{v:.3f}" if pd.notna(v) else "--" for v in size_df.loc[mask, "area_ratio"]]
        t.loc[mask, "Median area ratio (pred/GT)"] = rank_highlight_closest(cells, size_df.loc[mask, "area_ratio"])
    return t


_size_df_ood = build_size_rows(_std_all_ood, _best_n,    _zs_sub_ood)
_size_df_id  = build_size_rows(_std_all_id,  _best_n_id, _zs_sub_id)

table5 = pd.concat([
    format_size_table(_size_df_ood).assign(Area="OOD"),
    format_size_table(_size_df_id).assign(Area="ID"),
], ignore_index=True)
table5 = table5[["Area"] + [c for c in table5.columns if c != "Area"]]

export_table(
    table5, "table5_size_stratified",
    caption=(f"Pixel-level Precision/Recall/mean IoU and median predicted/ground-truth area ratio, stratified "
             f"by roof size tertile, zero-shot vs. best fine-tuned ({len(RUN_NAMES)} seeds pooled), standard "
             f"roofs (non-cut, non-artifact). OOD tertiles: Small $<${_t33_ood:.0f}\\,px, Large "
             f"$>${_t67_ood:.0f}\\,px, on {REAL_AREA_NAME}, best n={_best_n}. ID tertiles (own distribution): "
             f"Small $<${_t33_id:.0f}\\,px, Large $>${_t67_id:.0f}\\,px, on the pooled {', '.join(ID_REGIONS)} "
             f"held-out test split, best n={_best_n_id}. Bold/underline highlight the better/worse of the two "
             f"models within each (Area, size bin) pair; for the area-ratio column, bold/underline mark "
             f"whichever is closer to/farther from 1.0."),
    label="size-stratified", scale_to_width=False,
)

# %% [markdown]
# ## Table 6: Train/val/test split composition vs. training-set size
#
# The ID annotated pool (pooled `ceu_paz` + `cantidio_sampaio`) is carved into
# a dedicated **50% train / 25% val / 25% test** split per run
# (`TRAIN_FRACTION`=0.5, `TEST_FRACTION`=0.25, so val gets the 25% remainder -
# see `split_utils.compute_split`), at the *tile* level. Unlike tile counts
# (which are identical across every seed - only which tiles land where
# differs), the number of **rooftop instances** (individual ground-truth roof
# annotations, artifacts excluded, cut roofs included - the same "all roofs"
# convention as Table 2) those tiles carry genuinely varies seed-to-seed,
# since different runs draw different specific tiles into train/val/test and
# roof density isn't uniform across tiles. So this table reports actual
# instance counts, mean $\pm$ std across the 5 seeds, and each partition's
# share of the total instance pool.
#
# The scarcity-sweep `n` draws its first `n` *tiles* from that run's own train
# partition (capped at the partition's actual size for the "full dataset" row,
# `n=108` requested -> 102 actual tiles - see `DATASET_SIZES`'s docstring in
# the config cell above); the instance count reported here is just whatever
# rooftops happen to live on those `n` tiles. Val/test never depend on `n`, so
# they're repeated on every row for reference. There's no OOD equivalent of
# this table - the OOD area is never split, it's always evaluated on its whole
# fixed tile set (see the earlier "OOD tile count" print).

# %%
def _tile_names(tile_paths):
    return [os.path.splitext(os.path.basename(p))[0] for p in tile_paths]


# Per-tile rooftop instance counts (artifacts excluded, cut roofs included),
# pooled over both ID regions - one row per annotated tile in the 205-tile pool.
_instance_counts_by_tile = {}
for _region in ID_REGIONS:
    _d = pd.read_csv(os.path.join(IOU_RESULTS_DIR, f"{_region}_unified_metrics.csv"))
    _d = _d[~_d["is_artifact"].fillna(False).astype(bool)]
    for _tile_fname, _cnt in _d.groupby("belongs_to").size().items():
        _instance_counts_by_tile[os.path.splitext(_tile_fname)[0]] = int(_cnt)

_TOTAL_POOL_INSTANCES = sum(_instance_counts_by_tile.values())


def _instances_on(tile_names):
    return sum(_instance_counts_by_tile.get(t, 0) for t in tile_names)


_split_tiles_by_run = {
    run: {part: _tile_names(tiles) for part, tiles in get_split_tiles(run).items()}
    for run in RUN_NAMES
}

_instance_rows = []
for n in DATASET_SIZES:
    row = {"n": n}
    train_counts = [_instances_on(_split_tiles_by_run[r]["train"][:n]) for r in RUN_NAMES]
    val_counts   = [_instances_on(_split_tiles_by_run[r]["val"])       for r in RUN_NAMES]
    test_counts  = [_instances_on(_split_tiles_by_run[r]["test"])      for r in RUN_NAMES]
    for part, counts in [("train", train_counts), ("val", val_counts), ("test", test_counts)]:
        row[f"{part}_mean"] = float(np.mean(counts))
        row[f"{part}_std"]  = float(np.std(counts, ddof=1))
    _instance_rows.append(row)

_instance_df = pd.DataFrame(_instance_rows)
for part in ["train", "val", "test"]:
    _instance_df[f"{part}_pct_mean"] = _instance_df[f"{part}_mean"] / _TOTAL_POOL_INSTANCES * 100
    _instance_df[f"{part}_pct_std"]  = _instance_df[f"{part}_std"]  / _TOTAL_POOL_INSTANCES * 100

table6 = pd.DataFrame({"$T$size": [str(n) for n in _instance_df["n"]]})
for part, disp in [("train", "Train"), ("val", "Val"), ("test", "Test")]:
    table6[f"{disp} rooftops"] = [fmt_cell(m, s, decimals=0)
                                   for m, s in zip(_instance_df[f"{part}_mean"], _instance_df[f"{part}_std"])]
    table6[f"{disp} \\%"] = [fmt_cell(m, s, decimals=1)
                              for m, s in zip(_instance_df[f"{part}_pct_mean"], _instance_df[f"{part}_pct_std"])]

export_table(
    table6, "table6_split_composition",
    caption=(f"Rooftop-instance composition of the ID train/val/test split (pooled {', '.join(ID_REGIONS)}, "
             f"{_TOTAL_POOL_INSTANCES} ground-truth rooftops total, artifacts excluded) vs. training-set size "
             f"$n$, from a dedicated 50/25/25 tile-level split (train\\_fraction={TRAIN_FRACTION}, "
             f"test\\_fraction={TEST_FRACTION}, val gets the remainder). Mean $\\pm$ std across "
             f"{len(RUN_NAMES)} training seeds - unlike tile counts, instance counts genuinely vary by seed, "
             f"since different seeds draw different tiles and roof density isn't uniform across tiles. "
             f"Val/test counts don't depend on $n$ and are repeated on every row for reference; the "
             f"full-dataset row ($n={FULL_N}$, requested) is capped to the train partition's actual tile "
             f"count (102 of 205)."),
    label="split-composition", scale_to_width=False,
)

# %% [markdown]
# ## Suggested extensions (not computed here)
#
# A few metrics that would meaningfully extend this table set but need more
# machinery than a single evaluation pass, worth flagging for the thesis'
# future-work / limitations discussion:
#
# - **COCO-style AP / AP50 / AP75.** SAM2's automatic mask generator already
#   emits a `stability_score` and `predicted_iou` confidence per proposal
#   (visible in the raw `*_masks.npz` files). Sweeping a confidence threshold
#   over those scores would turn the "is object-level Precision meaningful"
#   problem from this notebook's intro into a proper precision-recall curve,
#   and its integral (Average Precision) is the standard, threshold-free way
#   the instance-segmentation literature (COCO, PASCAL VOC) reports exactly
#   this trade-off.
# - **Boundary F-score at a fixed pixel tolerance** (e.g. "what fraction of
#   predicted/GT boundary pixels lie within k pixels of the other mask's
#   boundary"), which is a threshold-free relative of HD95 popular in
#   semantic-segmentation benchmarks (e.g. Cityscapes' boundary IoU).
# - **Calibration** of `predicted_iou` against the actually achieved IoU
#   (reliability diagram / ECE) - relevant if the trained confidence score is
#   ever used downstream to auto-filter predictions.
# - **Statistical test for OOD vs. ID gap.** The tables here report both
#   domains side by side but don't test whether the gap between them is
#   significant at a given n - a paired test across the 5 seeds (same seed's
#   OOD run vs. its own ID run) would be the natural choice, mirroring the
#   pseudo-replication fix already applied to the EXP12 Spearman correlations
#   in `D3_clean_final_graphs_tvt.py`.

# %%
