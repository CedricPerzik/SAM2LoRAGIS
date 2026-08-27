import os
import glob
import random
import rasterio
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import geopandas as gpd
import concurrent.futures
from matplotlib import image
from rasterio import features
import matplotlib.pyplot as plt
from shapely.geometry import box

import sam_repo_notebooks.B_functions as bf



def format_ground_truth(gpkg_path, tif_path):
    """
    Formats QGIS GeoPackage rooftops into SAM 2 compatible dictionaries 
    while preserving global coordinate context and all GIS metadata.
    """
    with rasterio.open(tif_path) as src:
        tif_bounds = src.bounds
        transform = src.transform
        out_shape = src.shape
        tile_geom = box(*tif_bounds)
    
    # Load and spatially filter: only process what touches this tile
    gdf_full = gpd.read_file(gpkg_path)
    gdf = gdf_full[gdf_full.intersects(tile_geom)].copy()
    
    gt_masks = []

    for _, row in gdf.iterrows():
        # 1. LOCAL MASK (1024x1024)
        # We rasterize the geometry relative to the tile's transform.
        # This keeps the binary array size correct for SAM 2.
        mask = features.rasterize(
            [(row.geometry, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        ).astype(bool)

        # Skip features that are geometrically in range but have no pixel footprint
        if not np.any(mask):
            continue

        # 2. GLOBAL BOUNDS & VERTICES
        # We keep the raw geographic/global values for stitching
        minx, miny, maxx, maxy = row.geometry.bounds
        global_bbox = [minx, miny, maxx - minx, maxy - miny]
        
        if row.geometry.geom_type == 'Polygon':
            vertices = [list(row.geometry.exterior.coords)]
        else:
            vertices = [list(part.exterior.coords) for part in row.geometry.geoms]

        # 3. COMPLETE DATA STRUCTURE
        data = {
            # SAM 2 Standard Keys (Mask is Local, BBox is Global)
            "segmentation": mask,
            "area": int(mask.sum()), # Area in pixels 
            "bbox": global_bbox,
            
            # Custom Geospatial/Metadata Keys (Preserved from GPKG)
            # perimeter backup is geopandas length: https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.length.html
            "area_meters_sq": row.get('area_meters_sq', row.geometry.area),
            "perimeter_meters": row.get('perimeter_meters', row.geometry.length),
            "vertices": vertices,
            "feature_id": row.get('feature_id'),
            "is_artifact": row.get('is_artifact', False),
            "belongs_to": os.path.basename(tif_path),
            "is_cut": not row.geometry.within(tile_geom) # Dynamic check against tile boundary
        }
        
        gt_masks.append(data)
        
    return gt_masks


def audit_gt_data(gt_data):
    print(f"Total Features Found: {len(gt_data)}")
    
    if len(gt_data) == 0:
        print("❌ ERROR: No features found in this tile!")
        return

    # Check a random sample
    sample = gt_data[0]
    required_keys = [
        'segmentation', 
        'area',
        'bbox', 
        'area_meters_sq',
        'perimeter_meters',
        'vertices',
        'feature_id', 
        'is_artifact',
        'belongs_to',
        'is_cut'
        ]
    
    print("\n--- Field Check ---")
    for key in required_keys:
        if key in sample:
            print(f"✅ {key}: Found (Type: {type(sample[key])})")
        else:
            print(f"❌ {key}: MISSING!")

    # Value Range Checks
    areas = [d['area'] for d in gt_data]
    print(f"\n--- Value Stats ---")
    print(f"Average Mask Area: {np.mean(areas):.2f} pixels")
    print(f"Min Area: {min(areas)} | Max Area: {max(areas)}")
    
    # Check for empty masks
    empty_masks = sum(1 for d in gt_data if d['area'] == 0)
    if empty_masks > 0:
        print(f"⚠️ WARNING: {empty_masks} features have an area of 0 pixels.")

def _process_single_gt_tile(tif_path, gdf_full, output_dir):
    """
    Worker function: Opens one TIF, filters the global GeoDataFrame for that 
    specific tile's bounds, rasterizes the roofs, and saves the .npz file.
    """
    filename = os.path.basename(tif_path).replace(".tif", "_gt.npz")
    save_path = os.path.join(output_dir, filename)

    # 1. Skip if already processed
    if os.path.exists(save_path): 
        return True
        
    try:
        # 2. Extract bounds and transform from the TIF
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            transform = src.transform
            out_shape = src.shape
            tile_geom = box(*tif_bounds)
        
        # 3. Spatially filter: only process roofs that touch this specific tile
        gdf = gdf_full[gdf_full.intersects(tile_geom)].copy()
        gt_masks = []

        # 4. Rasterize and extract metadata
        for _, row in gdf.iterrows():
            mask = features.rasterize(
                [(row.geometry, 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype='uint8'
            ).astype(bool)

            if not np.any(mask):
                continue

            minx, miny, maxx, maxy = row.geometry.bounds
            global_bbox = [minx, miny, maxx - minx, maxy - miny]
            
            if row.geometry.geom_type == 'Polygon':
                vertices = [list(row.geometry.exterior.coords)]
            else:
                vertices = [list(part.exterior.coords) for part in row.geometry.geoms]

            data = {
                "segmentation": mask,
                "area": int(mask.sum()), 
                "bbox": global_bbox,
                "area_meters_sq": row.get('area_meters_sq', row.geometry.area),
                "perimeter_meters": row.get('perimeter_meters', row.geometry.length),
                "vertices": vertices,
                "feature_id": row.get('feature_id'),
                "is_artifact": row.get('artifact_intersect', False),
                "belongs_to": os.path.basename(tif_path),
                "is_cut": not row.geometry.within(tile_geom) 
            }
            gt_masks.append(data)
            
        # 5. Save the data to disk
        if gt_masks:
            np.savez_compressed(save_path, masks=np.array(gt_masks, dtype=object))
            
        return True
        
    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(tif_path)}: {e}")
        return False


def process_all_gt_tiles(TILES_DIR, GPKG_PATH, OUTPUT_DIR):
    """
    Manager function: Loads the GeoPackage once, then distributes the 
    TIF files across all available CPU threads for fast rasterization.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📦 Loading master GeoPackage into memory (this may take a moment)...")
    gdf_full = gpd.read_file(GPKG_PATH)
    
    if 'feature_id' not in gdf_full.columns:
        print(f"⚠️ Warning: 'feature_id' column not found. Using Index instead.")

    tif_files = glob.glob(os.path.join(TILES_DIR, "*.tif"))
    
    if not tif_files:
        print(f"❌ No .tif files found in {TILES_DIR}")
        return

    # Use all cores minus 2 to keep the system responsive
    max_workers = max(1, os.cpu_count() - 2)
    print(f"🚀 Firing up {max_workers} threads for parallel Ground Truth generation...")

    # Execute using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks. We pass `gdf_full` to every thread.
        futures = [executor.submit(_process_single_gt_tile, tif_path, gdf_full, OUTPUT_DIR) for tif_path in tif_files]
        
        # Track progress dynamically
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tif_files), desc="Pre-computing Ground Truth"):
            pass
            
    print("✅ All Ground Truth files generated successfully!")


def print_npz_premium(base_npz_dir, area_name, specific_file=None):
    """
    Picks a random .npz from the area folder, loads the corresponding .tif,
    and displays the overlay.
    """
    # 1. Setup paths dynamically
    current_dir = os.getcwd()
    
    # If running inside 'notebooks', step up one level to 'code'. Otherwise, stay put.
    if os.path.basename(current_dir) == "notebooks":
        code_dir = os.path.dirname(current_dir)
    else:
        code_dir = current_dir

    # Safely build absolute paths
    npz_folder = os.path.join(base_npz_dir, area_name)
    tif_folder = os.path.join(code_dir, "output_data", "tiles", area_name)
    
    # 2. Pick a file
    if specific_file:
        target_npz = os.path.join(npz_folder, specific_file)
    else:
        all_files = glob.glob(os.path.join(npz_folder, "*.npz"))
        if not all_files:
            print(f"!!! No .npz files found in {npz_folder}")
            return
        target_npz = random.choice(all_files)
    
    # 3. Load Data correctly
    data = np.load(target_npz, allow_pickle=True)
    masks = data['masks'].tolist() 

    # Debug Prints for the first mask
    if masks:
        m0 = masks[0]
        print(f"--- INSPECTING: {os.path.basename(target_npz)} ---")
        print(f"Keys: {m0.keys()}")
        print(f"ID: {m0.get('feature_id')} | Area: {m0.get('area')}px | Is Cut: {m0.get('is_cut')}")
        print(f"Belongs to: {m0.get('belongs_to')}")
        # --- NEW INSPECTION CODE ---
        seg = m0.get('segmentation')
        print(f"--- SEGMENTATION DETAILS ---")
        print(f"Type: {type(seg)}")
        print(f"Shape: {seg.shape} (Should match image dimensions)")
        print(f"Data type (dtype): {seg.dtype}")
        print(f"Unique values: {np.unique(seg)}")
        # ---------------------------

        # 4. Match and Load Image
        img_name = m0.get('belongs_to')
        tif_path = os.path.join(tif_folder, img_name)
        
        if not os.path.exists(tif_path):
            print(f"!!! Image not found at absolute path: \n{tif_path}")
            return

        img_np = np.array(Image.open(tif_path).convert("RGB"))

        # 5. Visualization
        plt.figure(figsize=(12, 12))
        plt.imshow(img_np)
        
        # (Assuming bf.show_anns is defined in your environment)
        bf.show_anns(masks) 
        
        plt.title(f"Ground Truth: {img_name}")
        plt.axis('off')
        plt.show()
    else:
        print("!!! The loaded .npz file contains an empty mask list.")