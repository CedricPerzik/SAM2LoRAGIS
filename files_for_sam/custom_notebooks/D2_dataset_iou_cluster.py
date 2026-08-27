import os

import D_functions as df

# RUN_ID and DATASET_SIZE are injected by SLURM via --export
_raw_n = os.environ.get("DATASET_SIZE", "")
_raw_run = os.environ.get("RUN_ID", "")
if not _raw_n:
    raise RuntimeError("DATASET_SIZE environment variable not set")
if not _raw_run:
    raise RuntimeError("RUN_ID environment variable not set")
N = int(_raw_n)
RUN_ID = _raw_run

# Cluster paths
GT_ROOT = "/home/s2145588/thesis/data/ground_truth_npz"
PREDICTIONS_BASE = "/home/s2145588/thesis/data/predictions"
EVAL_NPZ_BASE = "/home/s2145588/thesis/data/evaluation_npz"
IOU_RESULTS_BASE = "/home/s2145588/thesis/data/iou_results"
MODEL_NAME = f"{RUN_ID}/n{N}"


def _discover_areas(pred_root):
    """Return the sorted list of area subdirectories that
    B3_custom_predict_cluster.py wrote predictions for -- santa_madalena
    (OOD, always) plus any ID region it found held-out test tiles for.
    """
    if not os.path.isdir(pred_root):
        return []
    return sorted(
        name for name in os.listdir(pred_root)
        if os.path.isdir(os.path.join(pred_root, name))
    )


def evaluate_area(area):
    """Match this area's predictions to ground truth and export a unified CSV.

    Note: for the ID regions (ceu_paz, cantidio_sampaio), gt_dir contains every
    tile in the region (train+val+test), but pred_dir only contains the
    held-out test-split tiles (B3_custom_predict_cluster.py restricts ID
    predictions to those). df.run_evaluation_pipeline_concurrent only scores a
    GT tile when a matching prediction file exists, so this already restricts
    ID evaluation to the held-out test tiles with no extra manifest-reading
    logic needed here.
    """
    gt_dir = os.path.join(GT_ROOT, area)
    pred_dir = os.path.join(PREDICTIONS_BASE, MODEL_NAME, area)
    eval_dir = os.path.join(EVAL_NPZ_BASE, MODEL_NAME, area)
    csv_path = os.path.join(IOU_RESULTS_BASE, MODEL_NAME, f"n{N}_{area}_unified_metrics.csv")

    if not os.path.isdir(gt_dir):
        print(f"No ground truth found for area {area} at {gt_dir}, skipping")
        return

    print(f"\n--- Evaluating: {RUN_ID} n{N} ({area}) ---")
    df.run_evaluation_pipeline_concurrent(gt_dir, pred_dir, eval_dir)

    result = df.inspect_evaluation_metrics(eval_dir, area_name=f"{RUN_ID} n{N} ({area})", plot=False)
    if result is None:
        print(f"{area}: no evaluation data produced, skipping CSV export")
        return
    _, matched_ious, strict_ious = result

    df.export_unified_metrics_to_csv_concurrent(eval_dir, gt_dir, csv_path)

    matched_miou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    strict_miou = sum(strict_ious) / len(strict_ious) if strict_ious else 0.0
    n_found = len(matched_ious)
    n_total = len(strict_ious)
    n_missed = n_total - n_found
    recall = n_found / n_total if n_total else 0.0
    print(
        f"{RUN_ID} n{N} ({area}): "
        f"matched mIoU={matched_miou:.4f}  strict mIoU={strict_miou:.4f}  "
        f"recall={recall:.4f}  found={n_found}  missed={n_missed}"
    )


if __name__ == "__main__":
    pred_root = os.path.join(PREDICTIONS_BASE, MODEL_NAME)
    areas = _discover_areas(pred_root)

    print(f"Run ID       : {RUN_ID}")
    print(f"Dataset size : n{N}")
    print(f"Predictions  : {pred_root}")
    print(f"Areas found  : {areas}")

    if not areas:
        print(f"No prediction areas found under {pred_root}, nothing to evaluate.")
    else:
        for area in areas:
            evaluate_area(area)
