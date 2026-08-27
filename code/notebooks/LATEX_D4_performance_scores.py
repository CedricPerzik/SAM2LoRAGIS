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
# # D4 - Performance score tables (thesis results section)
#
# This notebook loads the same already-computed `unified_metrics.csv` files as
# `D3_clean_final_graphs.py` (no re-inference, no re-matching) and turns them
# into **copy-paste-ready LaTeX tables** for the thesis: Precision, Recall, F1,
# Dice, mean IoU, median IoU and Hausdorff Distance, computed across all 5
# fine-tuning runs and every dataset size in the scarcity sweep, plus the
# zero-shot baseline. Output is LaTeX (`\usepackage{booktabs}` required), so it
# can be pasted straight into the report.
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
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm

# --- Dataset sizes used in the scarcity sweep (162 = full dataset, all tiles) ---
DATASET_SIZES = [5, 10, 15, 25, 40, 65, 105, 162]

# --- Out-of-distribution evaluation area ---
AREA_NAME = "santa_madalena"
REAL_AREA_NAME = "Rio Claro II"  # Real name of santa madalena region, confusion in original communication.

# --- The 5 independent fine-tuning runs, and their training seed ---
RUNS = {
    "run_001": 42,
    "run_002": 1234,
    "run_004": 1,
    "run_005": 26,
    "run_006": 99,
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

# Ground-truth pixel masks for the OOD area - fixed across every run/size, used
# for the Hausdorff/ASSD boundary computation.
GT_DIR = os.path.join(CODE_DIR, "output_data", "ground_truth_npz", AREA_NAME)

# Predicted proposal masks (SAM2 automatic mask generator output), per run/size,
# also used for the Hausdorff/ASSD boundary computation.
RUNS_PRED_BASE     = "/home/ced/drives/dookiedisk/predictions"
ZEROSHOT_PRED_DIR  = os.path.join(CODE_DIR, "output_data", "masks", f"{AREA_NAME}_masks")

# --- Output locations ---
TABLES_DIR = os.path.join(CODE_DIR, "output_data", "tables", "perf_scores")
os.makedirs(TABLES_DIR, exist_ok=True)

HD_CACHE_PATH = os.path.join(CODE_DIR, "output_data", "iou_results", "hausdorff_cache", f"{AREA_NAME}_hd_cache.csv")
os.makedirs(os.path.dirname(HD_CACHE_PATH), exist_ok=True)

print(f"Code dir:     {CODE_DIR}")
print(f"Runs IoU dir: {RUNS_IOU_BASE}")
print(f"Tables dir:   {TABLES_DIR}")


# %% [markdown]
# ## Load the combined multi-run dataframe
#
# Identical loader to `D3_clean_final_graphs.py`: reads every already-computed
# CSV (no re-evaluation) into one long dataframe, one row per (ground-truth
# roof, run) pair, for every dataset size, plus the zero-shot rows at n=0.

# %%
def run_csv_path(run, n):
    return os.path.join(RUNS_IOU_BASE, run, f"n{n}", f"n{n}_{AREA_NAME}_unified_metrics.csv")


def load_run_df(run, n):
    """Load one (run, n) unified-metrics CSV, or None if it doesn't exist."""
    path = run_csv_path(run, n)
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d["n"]    = n
    d["run"]  = run
    d["seed"] = RUNS[run]
    return d


def load_all_runs_df(sizes=DATASET_SIZES):
    """Concatenate every (run, n) CSV plus the zero-shot CSV into one long dataframe.

    Each row is one predicted-vs-ground-truth roof match. For a given n there are
    up to len(RUNS) rows per ground-truth roof (one per run) plus the zero-shot
    rows at n=0 (zero-shot is deterministic, so only one "run").
    """
    frames = []
    for run in RUN_NAMES:
        for n in sizes:
            d = load_run_df(run, n)
            if d is None:
                print(f"{run} n{n}: CSV not found - skipping")
                continue
            frames.append(d)

    zs = pd.read_csv(ZEROSHOT_CSV)
    zs["n"], zs["run"], zs["seed"] = 0, "zero_shot", None
    frames.append(zs)

    out = pd.concat(frames, ignore_index=True)
    out["iou_score"]   = out["iou_score"].fillna(0.0)
    out["matched"]     = out["matched"].astype(bool)
    out["is_cut"]      = out["is_cut"].fillna(False).astype(bool)
    out["is_artifact"] = out["is_artifact"].fillna(False).astype(bool)
    return out


# %%
df_all = load_all_runs_df()

_ns_plot = sorted(set(DATASET_SIZES) | {0})
_run_and_zs = RUN_NAMES + ["zero_shot"]

print(f"Loaded {len(df_all)} rows | sizes: {sorted(df_all['n'].unique())} | runs: {RUN_NAMES}")

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

# %%
df_all["TP_px"] = np.where(df_all["matched"], df_all["overlap_pixels"], 0)
df_all["FN_px"] = np.where(df_all["matched"], df_all["area_px"] - df_all["overlap_pixels"], df_all["area_px"])
df_all["FP_px"] = np.where(
    df_all["matched"],
    df_all["overlap_pixels"] / df_all["iou_score"].replace(0, np.nan) - df_all["area_px"],
    0.0,
)
df_all["FP_px"] = df_all["FP_px"].fillna(0.0).clip(lower=0.0)


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


def build_metrics_per_run(df, mask_fn, sizes=DATASET_SIZES):
    """Tidy dataframe: one row per (run incl. zero_shot, n) with all metrics."""
    rows = []
    for run in _run_and_zs:
        ns = [0] if run == "zero_shot" else sizes
        for n in ns:
            sub = mask_fn(df[(df["run"] == run) & (df["n"] == n)])
            if sub.empty:
                continue
            row = {"run": run, "n": n}
            row.update(confusion_metrics(sub))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_n(metrics_per_run):
    """mean +/- std across the 5 seeds per n; zero-shot (n=0, 1 run) has std=NaN."""
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

metrics_std_per_run = build_metrics_per_run(df_all, mask_standard)
metrics_all_per_run = build_metrics_per_run(df_all, mask_all)

summary_std = summarize_per_n(metrics_std_per_run)
summary_all = summarize_per_n(metrics_all_per_run)

# Change this list for output
# _METRIC_ORDER = ["recall_obj", "recall_obj_at_thr", "precision_px", "recall_px", "f1_dice", "mean_iou", "median_iou"]
_METRIC_ORDER = ["precision_px", "recall_px", "f1_dice", "mean_iou", "median_iou"]

# %% [markdown]
# ## Table 1: Segmentation performance - Standard roofs (non-cut, non-artifact)
#
# The headline "how good is the model" table: object-level recall (any overlap,
# and the stricter IoU>=0.5 acceptance criterion), then pixel-level
# Precision/Recall/F1(=Dice), then mean/median IoU. Every cell pools pixels
# across all matched roofs for that (run, n), then reports mean +/- std across
# the 5 training seeds (zero-shot is a single deterministic run, no +/-).

# %%
table1 = build_display_table(summary_std, _METRIC_ORDER)
export_table(
    table1, "table1_segmentation_standard",
    caption=(f"Segmentation performance vs. training-set size on the OOD area ({REAL_AREA_NAME}), "
             f"standard roofs only (non-cut, non-artifact). Mean $\\pm$ std across {len(RUN_NAMES)} "
             f"training seeds; zero-shot is a single deterministic run."),
    label="perf-standard",
)

# %% [markdown]
# ## Table 2: Segmentation performance - All roofs (incl. cut, excl. artifacts)

# %%
table2 = build_display_table(summary_all, _METRIC_ORDER)
export_table(
    table2, "table2_segmentation_all",
    caption=(f"Segmentation performance vs. training-set size on the OOD area ({REAL_AREA_NAME}), "
             f"all roofs including tile-boundary-clipped ones (artifacts excluded). Mean $\\pm$ std "
             f"across {len(RUN_NAMES)} training seeds; zero-shot is a single deterministic run."),
    label="perf-all",
)

# %% [markdown]
# ## "Best" fine-tuned size
#
# Same selection rule as D3: the dataset size with the highest actual mean
# matched mIoU across the 5 runs (not just the largest sub-full-dataset size).

# %%
_matched_iou_by_n = {
    n: df_all[(df_all["run"].isin(RUN_NAMES)) & (df_all["n"] == n) & ~df_all["is_cut"] & ~df_all["is_artifact"] & df_all["matched"]]
        .groupby("run")["iou_score"].mean().mean()
    for n in DATASET_SIZES
}
_best_n = max(_matched_iou_by_n, key=_matched_iou_by_n.get)
print(f"'Best' fine-tuned size (highest mean matched mIoU, standard roofs): n={_best_n}")

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
# Results are cached to disk (`HD_CACHE_PATH`) since this touches every mask
# array on disk; re-running the notebook after the first pass is instant.

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


def pred_dir_for(run, n):
    if run == "zero_shot":
        return ZEROSHOT_PRED_DIR
    return os.path.join(RUNS_PRED_BASE, run, f"n{n}", AREA_NAME)


def compute_hd_for_run_n(df, run, n):
    """Standard, matched roofs for one (run, n): list of dicts with HD/HD95/ASSD in px and metres."""
    sub = mask_standard(df[(df["run"] == run) & (df["n"] == n) & df["matched"]])
    if sub.empty:
        return []

    pred_dir = pred_dir_for(run, n)
    rows = []
    for tile_src, g in sub.groupby("tile_source"):
        base = tile_src.replace("_eval.npz", "")
        gt_path   = os.path.join(GT_DIR, base + "_gt.npz")
        pred_path = os.path.join(pred_dir, base + "_masks.npz")
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
                "run": run, "n": n, "tile_source": tile_src, "feature_id": row.feature_id,
                "hd_m": hd_px * px_size_m, "hd95_m": hd95_px * px_size_m, "assd_m": assd_px * px_size_m,
            })
    return rows


# %%
REBUILD_HD_CACHE = False  # set True to force a full recompute instead of reusing the cache

_expected_pairs = {(r, n) for r in RUN_NAMES for n in DATASET_SIZES} | {("zero_shot", 0)}

if os.path.exists(HD_CACHE_PATH) and not REBUILD_HD_CACHE:
    hd_df = pd.read_csv(HD_CACHE_PATH)
    _cached_pairs = set(zip(hd_df["run"], hd_df["n"]))
else:
    hd_df = None
    _cached_pairs = set()

_missing_pairs = sorted(_expected_pairs - _cached_pairs)
if _missing_pairs:
    print(f"Computing Hausdorff/ASSD for {len(_missing_pairs)} missing (run, n) pairs...")
    new_rows = []
    for run, n in tqdm(_missing_pairs, desc="Hausdorff sweep"):
        new_rows.extend(compute_hd_for_run_n(df_all, run, n))
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        hd_df = pd.concat([hd_df, new_df], ignore_index=True) if hd_df is not None and len(hd_df) else new_df
    hd_df.to_csv(HD_CACHE_PATH, index=False)
    print(f"Cache updated -> {HD_CACHE_PATH}")
else:
    print(f"Loaded Hausdorff/ASSD cache ({len(hd_df)} roofs) -> {HD_CACHE_PATH}")

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


hd_summary = summarize_hd(hd_df)

table3 = pd.DataFrame({"$T$size": ["zs" if n == 0 else str(n) for n in hd_summary["n"]]})
table3["# roofs"] = hd_summary["n_roofs_mean"].round().astype(int)
for _key, _disp in [("hd_m", "Hausdorff (m)"), ("hd95_m", "HD95 (m)"), ("assd_m", "ASSD (m)")]:
    _means = hd_summary[f"{_key}_mean"]
    _stds  = hd_summary[f"{_key}_std"]
    _cells = [fmt_cell(m, s) for m, s in zip(_means, _stds)]
    table3[_disp] = rank_highlight(_cells, _means, higher_is_better=False)

export_table(
    table3, "table3_hausdorff",
    caption=(f"Boundary distance metrics vs. training-set size on the OOD area ({REAL_AREA_NAME}), "
             f"standard (non-cut, non-artifact), matched roofs only. Mean $\\pm$ std across "
             f"{len(RUN_NAMES)} training seeds; zero-shot is a single deterministic run."),
    label="hausdorff",
    scale_to_width=False,
)

# %% [markdown]
# ## Table 4: Headline comparison - zero-shot vs. best fine-tuned vs. full dataset
#
# Same "zero-shot vs. best fine-tuned vs. full dataset" comparison as before,
# standard roofs only, but split into three narrower tables (4a: pixel
# Precision/Recall/F1-Dice, 4b: mean/median IoU, 4c: boundary distances) so
# each one reads comfortably at its natural size instead of one wide table.
# "Best" = highest mean matched mIoU across the 5 seeds (n=`_best_n`, see
# above); the full-dataset row (n=162) is included separately whenever it
# differs from the best size, since "use all the data" is itself a relevant
# operating point even if it isn't the empirical optimum.

# %%
def _row_data_for_n(n, label):
    """Raw (mean, std) per metric for one n - kept numeric so the 3 headline
    tables below can rank/highlight across their (2-3) rows before formatting."""
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


def build_headline_table(rows, keys):
    out = pd.DataFrame({"Model": [r["Model"] for r in rows]})
    out["# GT roofs"] = [r["n_gt"] for r in rows]
    for key in keys:
        means = [r[key][0] for r in rows]
        stds  = [r[key][1] for r in rows]
        cells = [fmt_cell(m, s) for m, s in zip(means, stds)]
        out[METRIC_LABELS[key]] = rank_highlight(cells, means, METRIC_HIGHER_IS_BETTER[key])
    return out


_headline_rows = [
    _row_data_for_n(0, "Zero-shot"),
    _row_data_for_n(_best_n, f"Best fine-tuned (n={_best_n})"),
]
if _best_n != 162:
    _headline_rows.append(_row_data_for_n(162, "Full dataset (n=162)"))

_headline_caption_base = (f"on the OOD area ({REAL_AREA_NAME}), standard roofs only (non-cut, non-artifact). "
                           f"Mean $\\pm$ std across {len(RUN_NAMES)} training seeds; zero-shot is a single "
                           f"deterministic run. Bold = best in column, underlined = second-best.")

# %% [markdown]
# ### Table 4a: Precision / Recall / F1-Dice

# %%
table4a = build_headline_table(_headline_rows, ["precision_px", "recall_px", "f1_dice"])
export_table(
    table4a, "table4a_headline_precision_recall_f1",
    caption=f"Headline pixel Precision/Recall/F1(=Dice) {_headline_caption_base}",
    label="headline-prf", scale_to_width=False,
)

# %% [markdown]
# ### Table 4b: Mean / Median IoU

# %%
table4b = build_headline_table(_headline_rows, ["mean_iou", "median_iou"])
export_table(
    table4b, "table4b_headline_iou",
    caption=f"Headline mean/median IoU {_headline_caption_base}",
    label="headline-iou", scale_to_width=False,
)

# %% [markdown]
# ### Table 4c: Hausdorff / HD95 / ASSD

# %%
table4c = build_headline_table(_headline_rows, ["hd_m", "hd95_m", "assd_m"])
export_table(
    table4c, "table4c_headline_hausdorff",
    caption=f"Headline boundary distance metrics {_headline_caption_base}",
    label="headline-hd", scale_to_width=False,
)

# %% [markdown]
# ## Table 5: Size-stratified quality - zero-shot vs. best fine-tuned
#
# Splits the "Standard" subset into Small/Medium/Large tertiles by roof area
# (same tertile scheme as D3/EXP5, computed from the pooled standard-roof
# area distribution) to see *where* zero-shot's much lower overall pixel
# recall (Table 1/4a) actually comes from - whether it's spread evenly across
# roof sizes or concentrated at one end.
#
# Alongside pixel Precision/Recall/Mean IoU, "Median area ratio" is
# median(predicted mask area / ground-truth area) over matched roofs in that
# bin: near 1.0 means the mask's extent matches the annotation; well below
# 1.0 means systematic under-coverage (well above 1.0 would mean
# over-coverage). Best value per column is highlighted **within each size
# bin** (zero-shot vs. fine-tuned for that bin only, not compared across
# bins); for the area-ratio column, "best" means closest to 1.0 rather than
# highest or lowest.

# %%
df_all["pred_area_px"] = df_all["TP_px"] + df_all["FP_px"]

_std_all = mask_standard(df_all).copy()
_t33, _t67 = _std_all["area_px"].quantile(1 / 3), _std_all["area_px"].quantile(2 / 3)
_SIZE_BINS   = [0.0, _t33, _t67, float("inf")]
_SIZE_LABELS = ["Small", "Medium", "Large"]
_std_all["size_bin"] = pd.cut(_std_all["area_px"], bins=_SIZE_BINS, labels=_SIZE_LABELS, include_lowest=True)


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


_size_rows = []
for _bin in _SIZE_LABELS:
    _zs_sub = _std_all[(_std_all["run"] == "zero_shot") & (_std_all["size_bin"] == _bin)]
    _ft_sub = _std_all[(_std_all["run"].isin(RUN_NAMES)) & (_std_all["n"] == _best_n) & (_std_all["size_bin"] == _bin)]
    _size_rows.append({"Size bin": _bin, "Model": "Zero-shot", **_size_bin_metrics(_zs_sub)})
    _size_rows.append({"Size bin": _bin, "Model": f"Best fine-tuned (n={_best_n})", **_size_bin_metrics(_ft_sub)})

_size_df = pd.DataFrame(_size_rows)

table5 = pd.DataFrame({"Size bin": _size_df["Size bin"], "Model": _size_df["Model"], "# roofs": _size_df["n_gt"]})
for _key, _disp, _higher in [
    ("precision_px", "Precision (px)", True),
    ("recall_px",    "Recall (px)",    True),
    ("mean_iou",     "Mean IoU",       True),
]:
    table5[_disp] = ""
    for _bin in _SIZE_LABELS:
        _mask  = _size_df["Size bin"] == _bin
        _cells = [f"{v:.3f}" if pd.notna(v) else "--" for v in _size_df.loc[_mask, _key]]
        table5.loc[_mask, _disp] = rank_highlight(_cells, _size_df.loc[_mask, _key], _higher)

table5["Median area ratio (pred/GT)"] = ""
for _bin in _SIZE_LABELS:
    _mask  = _size_df["Size bin"] == _bin
    _cells = [f"{v:.3f}" if pd.notna(v) else "--" for v in _size_df.loc[_mask, "area_ratio"]]
    table5.loc[_mask, "Median area ratio (pred/GT)"] = rank_highlight_closest(_cells, _size_df.loc[_mask, "area_ratio"])

export_table(
    table5, "table5_size_stratified",
    caption=(f"Pixel-level Precision/Recall/mean IoU and median predicted/ground-truth area ratio, "
             f"stratified by roof size tertile (Small $<${_t33:.0f}\\,px, Medium, Large $>${_t67:.0f}\\,px), "
             f"zero-shot vs. best fine-tuned (n={_best_n}, {len(RUN_NAMES)} seeds pooled) on the OOD area "
             f"({REAL_AREA_NAME}), standard roofs (non-cut, non-artifact). Bold/underline highlight the "
             f"better/worse of the two models within each size bin; for the area-ratio column, bold/"
             f"underline mark whichever is closer to/farther from 1.0."),
    label="size-stratified", scale_to_width=False,
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

# %%
