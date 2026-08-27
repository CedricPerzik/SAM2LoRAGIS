import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import concurrent.futures
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def match_annotations_to_predictions(gt_list, pred_list):
    """
    Finds overlapping predictions for each ground truth annotation and packages 
    the candidates with their IoU scores for complete traceability.
    """
    all_matches = []

    for gt in gt_list:
        gt_seg = gt.get('segmentation')
        feature_id = gt.get('feature_id')
        area_pixels_annotation = int(np.sum(gt_seg)) 
        
        candidates_data = []

        for pred_idx, pred in enumerate(pred_list):
            pred_seg = pred.get('segmentation')
            
            intersection = (gt_seg & pred_seg)
            overlap_pixels = int(np.sum(intersection))

            if overlap_pixels > 0:
                union = (gt_seg | pred_seg)
                union_pixels = int(np.sum(union))
                iou = overlap_pixels / union_pixels if union_pixels > 0 else 0
                area_pixels_mask = int(np.sum(pred_seg))

                candidates_data.append({
                    'mask_nr': pred_idx,
                    'iou': float(iou),
                    'area_pixels_mask': area_pixels_mask,
                    'area_pixels_overlap': overlap_pixels
                })

        # Rank by IoU (highest first)
        ranked_candidates = sorted(candidates_data, key=lambda x: x['iou'], reverse=True)
        
        # Unsorted candidates for baseline traceability
        candidates = [{'mask_nr': c['mask_nr'], 'iou': c['iou']} for c in candidates_data]

        # Naive best fit (will be scrutinized by the resolver function)
        best_fit = None
        if ranked_candidates:
            top = ranked_candidates[0]
            best_fit = {
                'feature_id': feature_id,
                'mask_nr': top['mask_nr'],
                'area_pixels_annotation': area_pixels_annotation,
                'area_pixels_mask': top['area_pixels_mask'],
                'area_pixels_overlap': top['area_pixels_overlap'],
                'iou_value': top['iou']
            }

        all_matches.append({
            'feature_id': feature_id,
            'area_pixels_annotation': area_pixels_annotation,
            'candidates': candidates,
            'ranked_candidates': ranked_candidates,
            'best_fit': best_fit
        })

    return all_matches


def resolve_double_claims(all_matches):
    """
    Scans the match data and enforces a 1-to-1 relationship between Ground Truths 
    and Predictions using a global greedy sorting algorithm.
    """
    resolved_matches = []
    
    # 1. Flatten all possible pairings into a single arena
    all_pairings = []
    for gt_data in all_matches:
        f_id = gt_data['feature_id']
        ann_area = gt_data['area_pixels_annotation']
        
        for cand in gt_data['ranked_candidates']:
            all_pairings.append({
                'feature_id': f_id,
                'area_pixels_annotation': ann_area,
                'mask_nr': cand['mask_nr'],
                'iou_value': cand['iou'],
                'area_pixels_mask': cand['area_pixels_mask'],
                'area_pixels_overlap': cand['area_pixels_overlap']
            })
            
    # 2. Sort the entire arena by IoU (highest first)
    # This guarantees the strongest overlaps get first pick
    all_pairings.sort(key=lambda x: x['iou_value'], reverse=True)
    
    # 3. Greedily assign masks to GTs
    assigned_gts = set()
    assigned_masks = set()
    final_best_fits = {}
    
    for pair in all_pairings:
        f_id = pair['feature_id']
        m_nr = pair['mask_nr']
        
        # If neither the Ground Truth nor the Prediction Mask has been claimed yet: lock it in!
        if f_id not in assigned_gts and m_nr not in assigned_masks:
            assigned_gts.add(f_id)
            assigned_masks.add(m_nr)
            final_best_fits[f_id] = pair
    
    # 4. Rebuild the output with the resolved best fits
    for gt_data in all_matches:
        f_id = gt_data['feature_id']
        
        new_gt_data = {
            'feature_id': f_id,
            'candidates': gt_data['candidates'],
            'ranked_candidates': gt_data['ranked_candidates'],
            'best_fit': None # Default to None if it lost all claims and had no fallbacks
        }
        
        # If this GT won a mask in the arena, assign it
        if f_id in final_best_fits:
            won = final_best_fits[f_id]
            new_gt_data['best_fit'] = {
                'feature_id': won['feature_id'],
                'mask_nr': won['mask_nr'],
                'area_pixels_annotation': won['area_pixels_annotation'],
                'area_pixels_mask': won['area_pixels_mask'],
                'area_pixels_overlap': won['area_pixels_overlap'],
                'iou_value': won['iou_value']
            }
            
        resolved_matches.append(new_gt_data)
        
    return resolved_matches

def _evaluate_single_tile(gt_path, pred_dir, output_dir):
    """
    Worker function: Loads a single GT and Prediction file, calculates the IoU matches, 
    resolves double claims, and saves the evaluation results to disk.
    """
    base_name = os.path.basename(gt_path)
    pred_name = base_name.replace("_gt.npz", "_masks.npz") 
    pred_path = os.path.join(pred_dir, pred_name)
    
    save_name = base_name.replace("_gt.npz", "_eval.npz")
    save_path = os.path.join(output_dir, save_name)
    
    # 1. Skip if already processed
    if os.path.exists(save_path):
        return True
        
    # 2. Skip if there is no corresponding prediction file
    if not os.path.exists(pred_path):
        return False
        
    try:
        # 3. Load the data
        gt_data = np.load(gt_path, allow_pickle=True)
        pred_data = np.load(pred_path, allow_pickle=True)
        
        gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []
        pred_masks = pred_data['masks'].tolist() if 'masks' in pred_data else []
        
        # 4. If no ground truth roofs exist, save empty result and move on
        if not gt_masks:
            np.savez_compressed(save_path, matches=np.array([], dtype=object))
            return True
            
        # 5. Run the Core Logic
        raw_matches = match_annotations_to_predictions(gt_masks, pred_masks)
        resolved_matches = resolve_double_claims(raw_matches)
        
        # 6. Save the results
        np.savez_compressed(save_path, matches=np.array(resolved_matches, dtype=object))
        return True
        
    except Exception as e:
        print(f"Warning: error evaluating {base_name}: {e}")
        return False


def run_evaluation_pipeline_concurrent(gt_dir, pred_dir, output_dir):
    """
    Manager function: Grabs all Ground Truth tiles and distributes the 
    IoU evaluation math across all available CPU threads.
    """
    # 1. Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Grab all Ground Truth files
    gt_files = glob.glob(os.path.join(gt_dir, "*_gt.npz"))
    
    if not gt_files:
        print(f"Soft fail: no files ending in '_gt.npz' found in {gt_dir}")
        return
        
    print(f"Found {len(gt_files)} ground truth tiles. Starting parallel evaluation...")
    
    max_workers = max(1, os.cpu_count() - 2)
    
    # 3. Use ThreadPoolExecutor to bypass Jupyter freezing and speed up NumPy I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tiles to the thread pool
        futures = [executor.submit(_evaluate_single_tile, gt_path, pred_dir, output_dir) for gt_path in gt_files]
        
        # Update the progress bar as soon as any thread finishes a tile
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(gt_files), desc="Evaluating Tiles"):
            # We don't need to do anything with the result here because 
            # the worker function saves the .npz file directly to the hard drive.
            pass
            
    print("All tiles evaluated and saved successfully.")

# def run_evaluation_pipeline(gt_dir, pred_dir, output_dir):
#     """
#     Iterates through all Ground Truth .npz files, finds the corresponding 
#     Prediction .npz file, calculates the resolved IoU matches, and saves 
#     the results to a new .npz file.
#     """
#     # 1. Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 2. Grab all Ground Truth files
#     gt_files = glob.glob(os.path.join(gt_dir, "*_gt.npz"))
    
#     if not gt_files:
#         print(f"Soft fail: No files ending in '_gt.npz' found in {gt_dir}")
#         return
        
#     print(f"Found {len(gt_files)} Ground Truth tiles. Starting evaluation...")
    
#     # 3. Iterate through every tile
#     for gt_path in tqdm(gt_files, desc="Evaluating Tiles"):
#         base_name = os.path.basename(gt_path)
        
#         # --- FILENAME MATCHING LOGIC ---
#         # Assuming your prediction files have a predictable name based on the GT name.
#         pred_name = base_name.replace("_gt.npz", "_masks.npz") 
#         pred_path = os.path.join(pred_dir, pred_name)
        
#         # Set up the save path for the final evaluated data
#         save_name = base_name.replace("_gt.npz", "_eval.npz")
#         save_path = os.path.join(output_dir, save_name)
        
#         # Skip if already processed
#         if os.path.exists(save_path):
#             continue
            
#         # Skip if there is no corresponding prediction file
#         if not os.path.exists(pred_path):
#             # print(f"\nMissing prediction for {base_name}. Skipping...")
#             continue
            
#         # 4. Load the data
#         gt_data = np.load(gt_path, allow_pickle=True)
#         pred_data = np.load(pred_path, allow_pickle=True)
        
#         # Safely extract the lists (fallback to empty list if key is missing)
#         gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []
#         pred_masks = pred_data['masks'].tolist() if 'masks' in pred_data else []
        
#         # If there are no ground truth roofs in this tile, save an empty result and move on
#         if not gt_masks:
#             np.savez_compressed(save_path, matches=np.array([], dtype=object))
#             continue
            
#         # 5. Run the Core Logic
#         raw_matches = match_annotations_to_predictions(gt_masks, pred_masks)
#         resolved_matches = resolve_double_claims(raw_matches)
        
#         # 6. Save the results back to .npz format under the 'matches' key
#         np.savez_compressed(save_path, matches=np.array(resolved_matches, dtype=object))


def inspect_evaluation_metrics(eval_dir, area_name="", plot=True):
    eval_files = glob.glob(os.path.join(eval_dir, "*_eval.npz"))
    
    if not eval_files:
        print(f"No evaluation files found in {eval_dir}")
        return

    all_matched_ious = []
    all_strict_ious = []
    per_image_metrics = {}

    total_gts = 0
    total_misses = 0

    print(f"Analyzing {len(eval_files)} evaluated tiles for area '{area_name}'...\n")

    for file_path in eval_files:
        filename = os.path.basename(file_path)
        data = np.load(file_path, allow_pickle=True)
        
        # Extract the matches list
        matches = data['matches'].tolist() if 'matches' in data else []
        
        if not matches:
            continue

        image_matched_ious = []
        image_strict_ious = []

        for gt in matches:
            total_gts += 1
            best_fit = gt.get('best_fit')
            
            if best_fit is not None:
                iou = best_fit['iou_value']
                image_matched_ious.append(iou)
                image_strict_ious.append(iou)
                all_matched_ious.append(iou)
                all_strict_ious.append(iou)
            else:
                # The model completely missed this Ground Truth
                total_misses += 1
                image_strict_ious.append(0.0)
                all_strict_ious.append(0.0)

        # Calculate per-image mIoU
        matched_miou = np.mean(image_matched_ious) if image_matched_ious else 0.0
        strict_miou = np.mean(image_strict_ious) if image_strict_ious else 0.0
        
        per_image_metrics[filename] = {
            'matched_miou': matched_miou,
            'strict_miou': strict_miou,
            'gts_found': len(image_matched_ious),
            'gts_missed': len(image_strict_ious) - len(image_matched_ious)
        }

    # --- GLOBAL METRICS ---
    global_matched_miou = np.mean(all_matched_ious) if all_matched_ious else 0.0
    global_strict_miou = np.mean(all_strict_ious) if all_strict_ious else 0.0
    recall = (total_gts - total_misses) / total_gts if total_gts > 0 else 0.0

    print(f"=== GLOBAL METRICS ({total_gts} Total Ground Truths) ===")
    print(f"Roofs Found:  {total_gts - total_misses}")
    print(f"Roofs Missed: {total_misses} (False Negatives)")
    print(f"Matched mIoU: {global_matched_miou:.4f}")
    print(f"Strict mIoU:  {global_strict_miou:.4f}\n")

    if plot:
        # --- UPDATED VISUALIZATION ---
        plt.figure(figsize=(12, 6))

        # 1. Histogram of ALL IoUs (Strict)
        # We use a slightly transparent color so you can see the density
        plt.hist(all_strict_ious, bins=20, color='gray', alpha=0.3, edgecolor='black', label=f'Strict Distribution (incl. {total_misses} Misses)')

        # 2. Histogram of Matched IoUs (Successful detections)
        plt.hist(all_matched_ious, bins=20, color='skyblue', edgecolor='black', label='Matched Distribution (Found only)')

        # Add vertical lines for the means
        plt.axvline(global_matched_miou, color='green', linestyle='--', linewidth=2,
                    label=f'Matched mIoU: {global_matched_miou:.4f}')
        plt.axvline(global_strict_miou, color='red', linestyle=':', linewidth=2,
                    label=f'Strict mIoU: {global_strict_miou:.4f}')

        plt.title(f'IoU Performance Distribution: {area_name}')
        plt.xlabel('IoU Score (0.0 = Missed Roof)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return per_image_metrics, all_matched_ious, all_strict_ious


# def export_unified_metrics_to_csv(eval_dir, gt_dir, output_path):
#     """
#     Merges evaluation results with original Ground Truth metadata 
#     into a single comprehensive CSV file.
#     """
#     output_folder = os.path.dirname(output_path)
#     if output_folder and not os.path.exists(output_folder):
#         os.makedirs(output_folder, exist_ok=True)
        
#     eval_files = glob.glob(os.path.join(eval_dir, "*_eval.npz"))
#     all_rows = []

#     print(f"Compiling data from {len(eval_files)} tiles...")

#     for eval_path in eval_files:
#         # 1. Load Evaluation Data
#         eval_data = np.load(eval_path, allow_pickle=True)
#         matches = eval_data['matches'].tolist() if 'matches' in eval_data else []
        
#         # 2. Load corresponding GT Data to get metadata (area, bbox, etc.)
#         # Mapping back: area_name_tile_eval.npz -> area_name_tile_gt.npz
#         gt_path = os.path.join(gt_dir, os.path.basename(eval_path).replace("_eval.npz", "_gt.npz"))
        
#         if not os.path.exists(gt_path):
#             continue
            
#         gt_data_file = np.load(gt_path, allow_pickle=True)
#         gt_metadata_list = gt_data_file['masks'].tolist() if 'masks' in gt_data_file else []
        
#         # Create a lookup dictionary for GT metadata using feature_id
#         gt_lookup = {str(m['feature_id']): m for m in gt_metadata_list}

#         for match in matches:
#             f_id = str(match['feature_id'])
#             best_fit = match.get('best_fit')
            
#             # Start with the Performance Metrics
#             row = {
#                 'tile_source': os.path.basename(eval_path),
#                 'feature_id': f_id,
#                 'matched': best_fit is not None,
#                 'iou_score': best_fit['iou_value'] if best_fit else 0.0,
#                 'pred_mask_index': best_fit['mask_nr'] if best_fit else -1,
#                 'overlap_pixels': best_fit['area_pixels_overlap'] if best_fit else 0
#             }
            
#             # Add the Metadata from the original GT keys
#             if f_id in gt_lookup:
#                 meta = gt_lookup[f_id]
#                 row.update({
#                     'area_px': meta.get('area'),
#                     'vertices': meta.get('vertices'),
#                     'num_vertices': len(meta.get('vertices')[0]) if meta.get('vertices') and len(meta.get('vertices')) > 0 else 0, 
#                     'area_meters_sq': meta.get('area_meters_sq'),
#                     'perimeter_meters': meta.get('perimeter_meters'),
#                     'is_artifact': meta.get('is_artifact'),
#                     'is_cut': meta.get('is_cut'),
#                     'bbox': str(meta.get('bbox')), # Convert list to string for CSV
#                     'belongs_to': meta.get('belongs_to')
#                 })
            
#             all_rows.append(row)

#     # Create DataFrame and Export
#     df = pd.DataFrame(all_rows)
#     df.to_csv(output_path, index=False)
#     print(f"Unified metrics exported to: {output_path}")
#     return df

# ---------------- TESTING THREADPOOLING FOR CSV ANALYSIS ----------------
def _process_single_eval_file(eval_path, gt_dir):
    """
    Worker function: Loads a single eval file and its matching GT file,
    extracts the metrics, and returns a list of dictionary rows.
    """
    rows = []
    try:
        # 1. Load Evaluation Data
        eval_data = np.load(eval_path, allow_pickle=True)
        matches = eval_data['matches'].tolist() if 'matches' in eval_data else []
        
        if not matches:
            return rows

        # 2. Load corresponding GT Data to get metadata
        gt_path = os.path.join(gt_dir, os.path.basename(eval_path).replace("_eval.npz", "_gt.npz"))
        
        if not os.path.exists(gt_path):
            return rows
            
        gt_data_file = np.load(gt_path, allow_pickle=True)
        gt_metadata_list = gt_data_file['masks'].tolist() if 'masks' in gt_data_file else []
        
        # Create a lookup dictionary for GT metadata using feature_id
        gt_lookup = {str(m['feature_id']): m for m in gt_metadata_list}

        for match in matches:
            f_id = str(match['feature_id'])
            best_fit = match.get('best_fit')
            
            # Start with the Performance Metrics
            row = {
                'tile_source': os.path.basename(eval_path),
                'feature_id': f_id,
                'matched': best_fit is not None,
                'iou_score': best_fit['iou_value'] if best_fit else 0.0,
                'pred_mask_index': best_fit['mask_nr'] if best_fit else -1,
                'overlap_pixels': best_fit['area_pixels_overlap'] if best_fit else 0
            }
            
            # Add the Metadata from the original GT keys
            if f_id in gt_lookup:
                meta = gt_lookup[f_id]
                row.update({
                    'area_px': meta.get('area'),
                    'area_meters_sq': meta.get('area_meters_sq'),
                    'perimeter_meters': meta.get('perimeter_meters'),
                    'is_artifact': meta.get('is_artifact', False), # Fallback to False if key is missing
                    'is_cut': meta.get('is_cut', False), # Fallback to False if key is missing
                    'belongs_to': meta.get('belongs_to'),
                    'num_vertices': len(meta.get('vertices')[0]) if meta.get('vertices') and len(meta.get('vertices')) > 0 else 0,
                    'bbox': str(meta.get('bbox')), 
                    'building_orientation': meta.get('building_orientation'),
                    'vertices': meta.get('vertices')

                })
            
            rows.append(row)
            
    except Exception as e:
        print(f"Warning: error processing {eval_path}: {e}")
        
    return rows


def export_unified_metrics_to_csv_concurrent(eval_dir, gt_dir, output_path):
    """
    Manager function: Distributes the .npz file reading across all CPU threads
    and compiles the returned data into a single CSV.
    """
    output_folder = os.path.dirname(output_path)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    eval_files = glob.glob(os.path.join(eval_dir, "*_eval.npz"))
    
    if not eval_files:
        print(f"No evaluation files found in {eval_dir}")
        return pd.DataFrame()

    print(f"Compiling data from {len(eval_files)} tiles using multithreading...")
    
    all_rows = []
    max_workers = max(1, os.cpu_count() - 2)

    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files to the thread pool
        futures = [executor.submit(_process_single_eval_file, path, gt_dir) for path in eval_files]
        
        # Use tqdm with as_completed to update the progress bar as threads finish their files
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(eval_files), desc="Exporting to CSV"):
            # future.result() returns the `rows` list from the worker function
            all_rows.extend(future.result())

    # Create DataFrame and Export
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"Unified metrics exported to: {output_path}")
    
    return df

# ---------------- END OF THREADPOOLING TEST ----------------

def draw_masks_and_labels_cv2(img_np, masks, id_key='feature_id', prefix="", draw_borders=True):
    """
    Directly modifies a numpy image array to add colored masks, borders, and text.
    Bypasses Matplotlib entirely.
    """
    overlay = img_np.copy()
    output = img_np.copy()
    
    sorted_masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
    
    for i, ann in enumerate(sorted_masks):
        m = ann['segmentation'].astype(bool)
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay[m] = color
        
        if draw_borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(output, contours, -1, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    for i, ann in enumerate(masks):
        mask_id = str(ann.get(id_key, i))
        
        # FIX: Always use the pixel mask to find the center.
        # This completely ignores the geospatial bounding box in your GT data.
        y_coords, x_coords = np.where(ann['segmentation'])
        if len(x_coords) == 0: 
            continue
            
        cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
            
        text = f"{prefix}{mask_id}"
        
        cv2.putText(output, text, (cx, cy), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(output, text, (cx, cy), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return output


def analyze_and_visualize_worst_roofs(csv_path, gt_dir, pred_dir, img_dir, model_label="base", top_n=10):
    """
    Finds the worst performing roofs and uses OpenCV to stitch and save visuals.
    """
    df = pd.read_csv(csv_path)
    worst_roofs = df.sort_values(by=['iou_score', 'area_meters_sq'], ascending=[True, False]).head(top_n)

    print(f"\nTop {top_n} worst performing roofs (saving without Matplotlib):")

    CURRENT_DIR = os.getcwd()
    CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR
    out_dir = os.path.join(CODE_DIR, "output_data", "debug_visuals", model_label)
    os.makedirs(out_dir, exist_ok=True)

    for index, row in worst_roofs.iterrows():
        eval_name = row['tile_source']
        f_id = row['feature_id']
        img_name = row['belongs_to']
        iou = row['iou_score']

        gt_name = eval_name.replace("_eval.npz", "_gt.npz")
        pred_name = eval_name.replace("_eval.npz", "_masks.npz")

        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = os.path.join(pred_dir, pred_name)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        
        gt_data = np.load(gt_path, allow_pickle=True)
        gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []

        pred_data = np.load(pred_path, allow_pickle=True) if os.path.exists(pred_path) else None
        pred_masks = pred_data['masks'].tolist() if (pred_data and 'masks' in pred_data) else []

        # FIX: Grab the actual pixel mask of the target, not the geospatial bbox
        target_mask = None
        for m in gt_masks:
            if str(m.get('feature_id')) == str(f_id):
                target_mask = m.get('segmentation') 
                break

        gt_drawn = draw_masks_and_labels_cv2(img_np, gt_masks, id_key='feature_id', prefix="ID: ")
        pred_drawn = draw_masks_and_labels_cv2(img_np, pred_masks, id_key='feature_id', prefix="#")

        # FIX: Draw the red target bounding box using the pixel coordinates of the mask
        if target_mask is not None:
            y_coords, x_coords = np.where(target_mask)
            if len(x_coords) > 0:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                pad = 5 # Add a little breathing room around the roof
                cv2.rectangle(gt_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)
                cv2.rectangle(pred_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)

        combined_img = np.hstack((gt_drawn, pred_drawn))

        h, w, c = combined_img.shape
        header = np.zeros((60, w, c), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header, f"Ground Truth | Tile: {img_name}", (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, f"Predictions | IoU: {iou:.4f} | Target ID: {f_id}", (w//2 + 20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        final_output = np.vstack((header, combined_img))

        save_file = os.path.join(out_dir, f"rank_{index}_{eval_name.replace('.npz', '.jpg')}")
        print(f"Saving: {save_file}")
        Image.fromarray(final_output).save(save_file, format="JPEG", quality=85)


def analyze_and_visualize_iou_range(csv_path, gt_dir, pred_dir, img_dir, area_name, model_label="base", min_iou=0.0, max_iou=1.0):
    """
    Finds roofs within a specific IoU range across all tiles and uses OpenCV
    to stitch and save visuals into a dedicated area folder.
    """
    df = pd.read_csv(csv_path)

    # 1. Filter by the specified IoU range
    filtered_roofs = df[(df['iou_score'] >= min_iou) & (df['iou_score'] <= max_iou)]

    # Sort them by IoU so you review them in a logical order
    filtered_roofs = filtered_roofs.sort_values(by=['iou_score'])

    print(f"\nFound {len(filtered_roofs)} roofs in IoU range [{min_iou}, {max_iou}] for '{area_name}'.")

    # 2. Add model_label and area_name to the output directory
    CURRENT_DIR = os.getcwd()
    CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR
    out_dir = os.path.join(CODE_DIR, "output_data", "debug_visuals", model_label, area_name)

    # exist_ok=True ensures it won't crash if the folder is already there
    os.makedirs(out_dir, exist_ok=True)

    for index, row in filtered_roofs.iterrows():
        eval_name = row['tile_source']
        f_id = row['feature_id']
        img_name = row['belongs_to']
        iou = row['iou_score']

        gt_name = eval_name.replace("_eval.npz", "_gt.npz")
        pred_name = eval_name.replace("_eval.npz", "_masks.npz")

        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = os.path.join(pred_dir, pred_name)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        
        gt_data = np.load(gt_path, allow_pickle=True)
        gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []

        pred_data = np.load(pred_path, allow_pickle=True) if os.path.exists(pred_path) else None
        pred_masks = pred_data['masks'].tolist() if (pred_data and 'masks' in pred_data) else []

        target_mask = None
        for m in gt_masks:
            if str(m.get('feature_id')) == str(f_id):
                target_mask = m.get('segmentation') 
                break

        # Draw overlays using your existing cv2 function
        gt_drawn = draw_masks_and_labels_cv2(img_np, gt_masks, id_key='feature_id', prefix="ID: ")
        pred_drawn = draw_masks_and_labels_cv2(img_np, pred_masks, id_key='feature_id', prefix="#")

        if target_mask is not None:
            y_coords, x_coords = np.where(target_mask)
            if len(x_coords) > 0:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                pad = 5
                cv2.rectangle(gt_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)
                cv2.rectangle(pred_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)

        combined_img = np.hstack((gt_drawn, pred_drawn))

        h, w, c = combined_img.shape
        header = np.zeros((60, w, c), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header, f"Ground Truth | Tile: {img_name}", (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, f"Predictions | IoU: {iou:.4f} | Target ID: {f_id}", (w//2 + 20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        final_output = np.vstack((header, combined_img))

        # 3. Descriptive filenames (e.g., iou_0.45_id_1054_ceu_paz_row_0_col_0.jpg)
        clean_eval_name = eval_name.replace('_eval.npz', '.jpg')
        file_name = f"iou_{iou:.2f}_id_{f_id}_{clean_eval_name}"
        
        save_file = os.path.join(out_dir, file_name)
        print(f"Saving: {save_file}")
        Image.fromarray(final_output).save(save_file, format="JPEG", quality=85)

def analyze_and_visualize_iou_range_tqdm(csv_path, gt_dir, pred_dir, img_dir, area_name, model_label="base", min_iou=0.0, max_iou=1.0):
    """
    Finds roofs within a specific IoU range across all tiles and uses OpenCV
    to stitch and save visuals into a dedicated area folder, with a progress bar.
    """
    df = pd.read_csv(csv_path)

    # 1. Filter by the specified IoU range
    filtered_roofs = df[(df['iou_score'] >= min_iou) & (df['iou_score'] <= max_iou)]

    # Sort them by IoU so you review them in a logical order
    filtered_roofs = filtered_roofs.sort_values(by=['iou_score'])

    print(f"\nFound {len(filtered_roofs)} roofs in IoU range [{min_iou}, {max_iou}] for '{area_name}'.")

    # Early exit if the filter is too strict and returns nothing
    if len(filtered_roofs) == 0:
        print("Skipping visualization: No roofs match this criteria.")
        return

    # 2. Add model_label and area_name to the output directory
    CURRENT_DIR = os.getcwd()
    CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR
    out_dir = os.path.join(CODE_DIR, "output_data", "debug_visuals", model_label, area_name)

    os.makedirs(out_dir, exist_ok=True)

    # 3. Wrap the loop in tqdm
    for index, row in tqdm(filtered_roofs.iterrows(), total=len(filtered_roofs), desc=f"Visualizing {area_name}"):
        eval_name = row['tile_source']
        f_id = row['feature_id']
        img_name = row['belongs_to']
        iou = row['iou_score']

        mask_nr = int(row['pred_mask_index'])

        gt_name = eval_name.replace("_eval.npz", "_gt.npz")
        pred_name = eval_name.replace("_eval.npz", "_masks.npz")

        gt_path = os.path.join(gt_dir, gt_name)
        pred_path = os.path.join(pred_dir, pred_name)
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"))
        
        gt_data = np.load(gt_path, allow_pickle=True)
        gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []

        pred_data = np.load(pred_path, allow_pickle=True) if os.path.exists(pred_path) else None
        pred_masks = pred_data['masks'].tolist() if (pred_data and 'masks' in pred_data) else []

        target_mask = None
        for m in gt_masks:
            if str(m.get('feature_id')) == str(f_id):
                target_mask = m.get('segmentation') 
                break

        # Draw overlays using your existing cv2 function
        gt_drawn = draw_masks_and_labels_cv2(img_np, gt_masks, id_key='feature_id', prefix="ID: ")
        pred_drawn = draw_masks_and_labels_cv2(img_np, pred_masks, id_key='feature_id', prefix="#")

        if target_mask is not None:
            y_coords, x_coords = np.where(target_mask)
            if len(x_coords) > 0:
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                y_min, y_max = np.min(y_coords), np.max(y_coords)
                
                pad = 5
                cv2.rectangle(gt_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)
                cv2.rectangle(pred_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)

        combined_img = np.hstack((gt_drawn, pred_drawn))

        h, w, c = combined_img.shape
        header = np.zeros((60, w, c), dtype=np.uint8)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(header, f"Ground Truth | Tile: {img_name}", (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, f"Predictions | IoU: {iou:.4f} | Target ID: {f_id} | Mask nr: {mask_nr}", (w//2 + 20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        
        final_output = np.vstack((header, combined_img))

        clean_eval_name = eval_name.replace('_eval.npz', '.jpg')
        file_name = f"iou_{iou:.2f}_id_{f_id}_{clean_eval_name}"
        
        save_file = os.path.join(out_dir, file_name)
        
        # We save silently here to avoid breaking the visual flow of the tqdm progress bar
        Image.fromarray(final_output).save(save_file, format="JPEG", quality=85)

# Prevent OpenCV's internal multithreading from fighting with Python's multiprocessing
cv2.setNumThreads(0)

def process_single_roof(task_data):
    """
    Worker function executed by individual CPU cores.
    Processes exactly one roof and saves the visual to disk.
    """
    row, gt_dir, pred_dir, img_dir, out_dir = task_data
    
    eval_name = row['tile_source']
    f_id = row['feature_id']
    img_name = row['belongs_to']
    iou = row['iou_score']
    mask_nr = int(row['pred_mask_index'])

    gt_name = eval_name.replace("_eval.npz", "_gt.npz")
    pred_name = eval_name.replace("_eval.npz", "_masks.npz")

    gt_path = os.path.join(gt_dir, gt_name)
    pred_path = os.path.join(pred_dir, pred_name)
    img_path = os.path.join(img_dir, img_name)

    # Skip if missing fundamental files
    if not os.path.exists(img_path) or not os.path.exists(gt_path):
        return False

    img_np = np.array(Image.open(img_path).convert("RGB"))
    
    gt_data = np.load(gt_path, allow_pickle=True)
    gt_masks = gt_data['masks'].tolist() if 'masks' in gt_data else []

    pred_data = np.load(pred_path, allow_pickle=True) if os.path.exists(pred_path) else None
    pred_masks = pred_data['masks'].tolist() if (pred_data and 'masks' in pred_data) else []

    target_mask = None
    for m in gt_masks:
        if str(m.get('feature_id')) == str(f_id):
            target_mask = m.get('segmentation') 
            break

    # Draw overlays
    gt_drawn = draw_masks_and_labels_cv2(img_np, gt_masks, id_key='feature_id', prefix="ID: ")
    pred_drawn = draw_masks_and_labels_cv2(img_np, pred_masks, id_key='feature_id', prefix="#")

    # Draw target bounding box
    if target_mask is not None:
        y_coords, x_coords = np.where(target_mask)
        if len(x_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            pad = 5
            cv2.rectangle(gt_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)
            cv2.rectangle(pred_drawn, (x_min-pad, y_min-pad), (x_max+pad, y_max+pad), (255, 0, 0), 4)

    # Combine and add header
    combined_img = np.hstack((gt_drawn, pred_drawn))
    h, w, c = combined_img.shape
    header = np.zeros((60, w, c), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, f"Ground Truth | Tile: {img_name}", (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(header, f"Predictions | IoU: {iou:.4f} | Target ID: {f_id} | Mask nr: {mask_nr}", (w//2 + 20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    final_output = np.vstack((header, combined_img))

    # Save to disk
    clean_eval_name = eval_name.replace('_eval.npz', '.jpg')
    file_name = f"iou_{iou:.2f}_id_{f_id}_{clean_eval_name}"
    save_file = os.path.join(out_dir, file_name)
    
    Image.fromarray(final_output).save(save_file, format="JPEG", quality=85)
    return True


def visualize_iou_range_concurrent(csv_path, gt_dir, pred_dir, img_dir, area_name, model_label="base", min_iou=0.0, max_iou=1.0):
    """
    Manager function: Filters data and distributes tasks across all CPU cores.
    """
    df = pd.read_csv(csv_path)
    filtered_roofs = df[(df['iou_score'] >= min_iou) & (df['iou_score'] <= max_iou)]
    filtered_roofs = filtered_roofs.sort_values(by=['iou_score'])

    print(f"\nFound {len(filtered_roofs)} roofs in IoU range [{min_iou}, {max_iou}] for '{area_name}'.")
    if len(filtered_roofs) == 0:
        return

    CURRENT_DIR = os.getcwd()
    CODE_DIR = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "notebooks" else CURRENT_DIR
    out_dir = os.path.join(CODE_DIR, "output_data", "debug_visuals", model_label, area_name)
    os.makedirs(out_dir, exist_ok=True)

    # Package the data for the worker processes (converting pandas rows to dicts for safe pickling)
    tasks = []
    for _, row in filtered_roofs.iterrows():
        tasks.append((row.to_dict(), gt_dir, pred_dir, img_dir, out_dir))

    # Determine optimal number of cores (leaves one free so your OS doesn't freeze)
    max_workers = max(1, os.cpu_count() - 2)
    print(f"Firing up {max_workers} CPU cores...")

    # Execute in parallel with a shared progress bar
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_single_roof, tasks), total=len(tasks), desc=f"Processing {area_name}"))