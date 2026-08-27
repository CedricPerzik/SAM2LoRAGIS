
import os, json, random, glob, shutil
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from rasterio.plot import show
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling

from rasterio import features
import geopandas as gpd
from shapely.geometry import box
import concurrent.futures


def check_tif_settings(tif_path):
    with rasterio.open(tif_path) as src:
        print("--- Image Profile (Settings) ---")
        for key, value in src.profile.items():
            print(f"{key}: {value}")

def reproject_tif_list(file_names, input_folder, output_folder, target_crs, overwrite=False):
    """
    Reprojects a specific list of TIF files to a target CRS.
    
    Args:
        file_names (list): List of base names (e.g., ["ceu_paz", "santa_madalena"])
        input_folder (str): Where the original .tif files are stored.
        output_folder (str): Where the _normalized.tif files will be saved.
        target_crs (str): The EPSG code (e.g., 'EPSG:31983').
        overwrite (bool): If True, reprocess files even if they already exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    processed_files = []
    
    print(f"Starting reprojection for {len(file_names)} files...")

    for name in file_names:
        input_path = os.path.join(input_folder, f"{name}.tif")
        output_filename = f"{name}_normalized.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        # 1. Skip logic
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {name}: Normalized file already exists.")
            processed_files.append(output_filename)
            continue

        if not os.path.exists(input_path):
            print(f"Warning: Source file not found: {input_path}")
            continue
            
        with rasterio.open(input_path) as src:
            print(f"Processing: {name}.tif...")
            
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.profile.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'bigtiff': 'IF_NEEDED',
            })
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            processed_files.append(output_filename)

    # 2. Summary
    print("\n" + "="*40)
    print("REPROJECTION SUMMARY")
    print(f"    Total expected: {len(file_names)}")
    print(f"    Successfully ready: {len(processed_files)}")
    print(f"    Location: {os.path.abspath(output_folder)}")
    if processed_files:
        print(f"    Files: {', '.join(processed_files)}")
    print("="*40 + "\n")

def setup_tiling_environment(file_names, input_root="../input_data/normalized_tifs/", output_root="../output_data/tiles/", tile_size=1024):
    """
    Sets up the directory structure and paths for processing reprojected rasters.
    
    Args:
        file_names (list): List of base names (e.g., ["ceu_paz", "santa_madalena"])
        input_root (str): Root folder containing the reprojected .tif files.
        output_root (str): Root folder where the tile subfolders will be created.
        tile_size (int): Pixel dimensions for the square tiles.
        
    Returns:
        list: A list of dictionaries, each containing the specific paths for one image.
    """
    project_configs = []

    for name in file_names:
        # 1. Construct Input Path
        # Pattern: ../input_data/normalized_tifs/name_normalized.tif
        input_tif_path = os.path.join(input_root, f"{name}_normalized.tif")
        
        # 2. Construct Output Paths
        # Pattern: ../output_data/tiles/name/
        target_tile_dir = os.path.join(output_root, name)
        metadata_dir = os.path.join(target_tile_dir, "metadata")
        json_path = os.path.join(metadata_dir, f"{name}_output_metadata.json")

        # 3. Create Necessary Directories
        # This creates the tile folder AND the metadata subfolder in one go
        os.makedirs(metadata_dir, exist_ok=True)

        # 4. info Dictionary
        config = {
            "name": name,
            "input_path": input_tif_path,
            "output_dir": target_tile_dir,
            "metadata_path": json_path,
            "tile_size": tile_size
        }
        project_configs.append(config)

        # 5. Status Printout
        print(f"Setup complete for: {name.upper()}")
        print(f"    Source: {input_tif_path}")
        print(f"    Target: {target_tile_dir}")
        print("=" * 40)

    return project_configs

def pad_array(array, target_shape):
    """
    Part 1/2 for tiling execution
    Pads a (C, H, W) numpy array to target_shape with zeros.
    """
    channels, height, width = array.shape
    target_h, target_w = target_shape
    # Pad format: ((before_ch, after_ch), (before_h, after_h), (before_w, after_w))
    padding = ((0, 0), (0, target_h - height), (0, target_w - width))
    # Apply padding with values of 0
    return np.pad(array, padding, mode='constant', constant_values=0)

def generate_georeferenced_tiles(config_list):
    """
    Part 2/2 for tiling execution
    Takes the list of configurations and generates tiles for each dataset.
    
    Args:
        config_list (list): The list of dictionaries from setup_tiling_environment()
    """
    for config in config_list:
        # Extract variables from the config dictionary
        name = config["name"]
        input_path = config["input_path"]
        output_folder = config["output_dir"]
        json_path = config["metadata_path"]
        tile_size = config["tile_size"]
        
        # We initialize a NEW registry for EVERY dataset
        metadata_registry = {}
        
        if not os.path.exists(input_path):
            print(f"Skipping {name}: File not found at {input_path}")
            continue

        print(f"Tiling: {name} (Source: {os.path.basename(input_path)})")
        
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            width, height = src.width, src.height
            crs_str = src.crs.to_string() if src.crs else "Unknown"
            crs_wkt = src.crs.to_wkt() if src.crs else "Unknown"
            
            row_idx = 0
            for y in range(0, height, tile_size):
                col_idx = 0
                for x in range(0, width, tile_size):
                    
                    # 1. Define window and read
                    w_width = min(tile_size, width - x)
                    w_height = min(tile_size, height - y)
                    window = Window(col_off=x, row_off=y, width=w_width, height=w_height)
                    
                    data = src.read(window=window)
                    
                    # 2. Pad if necessary
                    if data.shape[1] != tile_size or data.shape[2] != tile_size:
                        data = pad_array(data, (tile_size, tile_size))
                    
                    # 3. Handle Transform and Metadata
                    tile_transform = src.window_transform(window)
                    tile_meta = meta.copy()
                    tile_meta.update({
                        "driver": "GTiff",
                        "height": tile_size,
                        "width": tile_size,
                        "transform": tile_transform
                    })
                    
                    # 4. Save the Tile
                    tile_filename = f"{name}_row_{row_idx}_col_{col_idx}.tif"
                    tile_out_path = os.path.join(output_folder, tile_filename)
                    
                    with rasterio.open(tile_out_path, "w", **tile_meta) as dest:
                        dest.write(data)
                        
                    # 5. Calculate Georeferenced Bounds
                    # (Top Left and Bottom Right)
                    tl_x, tl_y = tile_transform * (0, 0)
                    br_x, br_y = tile_transform * (tile_size, tile_size)
                    
                    # Register this tile's metadata
                    metadata_registry[tile_filename] = {
                        "original_image": os.path.basename(input_path),
                        "row": row_idx,
                        "col": col_idx,
                        "crs_string": crs_str,
                        "crs_wkt": crs_wkt,
                        "bounds": {
                            "x_min": min(tl_x, br_x),
                            "x_max": max(tl_x, br_x),
                            "y_min": min(tl_y, br_y),
                            "y_max": max(tl_y, br_y)
                        }
                    }
                    col_idx += 1
                row_idx += 1

        # 6. Save JSON for this specific dataset
        with open(json_path, 'w') as f:
            json.dump(metadata_registry, f, indent=4)
            
        print(f"Finished {name}: {len(metadata_registry)} tiles created.")
        print(f"    Metadata: {json_path}")
        print("=" * 40)

def verify_random_tiles(config_list, num_samples=1):
    """
    Picks random tiles across the processed datasets to verify padding and metadata.
    
    Args:
        config_list (list): The list of dictionaries from setup_tiling_environment()
        num_samples (int): How many random tiles to show.
    """
    if not config_list:
        print("No configurations found. Please run setup first.")
        return

    for _ in range(num_samples):
        # 1. Pick a random dataset from your list
        config = random.choice(config_list)
        name = config["name"]
        output_folder = config["output_dir"]
        json_path = config["metadata_path"]

        # 2. Load that specific dataset's metadata
        if not os.path.exists(json_path):
            print(f"Metadata for {name} not found at {json_path}. Skipping.")
            continue

        with open(json_path, 'r') as f:
            meta_registry = json.load(f)
        
        if not meta_registry:
            print(f"No tiles recorded for {name}.")
            continue

        # 3. Pick a random tile from that dataset
        random_tile_name = random.choice(list(meta_registry.keys()))
        tile_data = meta_registry[random_tile_name]
        tile_path = os.path.join(output_folder, random_tile_name)
        
        print(f"\n" + "="*40)
        print(f"INSPECTING DATASET: {name.upper()}")
        print(f"   Tile Name: {random_tile_name}")
        print("="*40)
        
        # 4. Open and plot the image
        if not os.path.exists(tile_path):
            print(f"Error: Tile file not found at {tile_path}")
            continue

        with rasterio.open(tile_path) as src:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Using show() to display the raster
            show(src, ax=ax, title=f"Dataset: {name}\nTile: {random_tile_name}\n(Shape: {src.shape})")
            plt.axis('on') # Keep coordinates visible for CRS check
            plt.show()
            
            # 5. Print the JSON Metadata Entry
            print("\n--- JSON Metadata Entry ---")
            print(json.dumps(tile_data, indent=4))
            
            # 6. Quick verify on dimensions
            expected_size = config["tile_size"]
            if src.width == expected_size and src.height == expected_size:
                print(f"\nPASS: Image is correctly padded to {expected_size}x{expected_size}.")
            else:
                print(f"\nERROR: Image dimensions are {src.width}x{src.height}.")

def sam2unet_format_tiles_to_png(config_list):
    """
    Reads the tiled TIFs, converts them to 8-bit RGB, and saves them as PNGs.
    Organizes the directory by putting the new .png files in 'tiles_png' 
    and moving the original .tif files into 'tiles_tif'.
    """
    for config in config_list:
        area_name = config["name"]
        tile_dir = config["output_dir"]
        os.makedirs(tile_dir, exist_ok=True)

        # Create the specific subfolders for sorting
        png_dir = os.path.join(tile_dir, "tiles_png")
        tif_dir = os.path.join(tile_dir, "tiles_tif")
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(tif_dir, exist_ok=True)
        
        # Grab all the fresh .tif files in the root of tile_dir
        tif_files = glob.glob(os.path.join(tile_dir, "*.tif"))
        
        if not tif_files:
            print(f"⚠️ No .tif files found to process in {tile_dir}")
            continue

        for tif_path in tif_files:
            filename = os.path.basename(tif_path)
            
            # --- 1. Process and Save the PNG ---
            with rasterio.open(tif_path) as src:
                # Read only the first 3 bands (RGB) to drop alpha/infrared bands if present
                img_array = src.read([1, 2, 3])
                
                # Transpose from Rasterio's (C, H, W) to PIL's (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))
                
                # Normalize to 0-255 uint8 format for standard image saving
                if img_array.max() <= 1.0 or img_array.dtype != np.uint8:
                    # Avoid divide by zero for completely black edge tiles
                    max_val = img_array.max() if img_array.max() > 0 else 1
                    img_array = (img_array / max_val * 255).astype(np.uint8)
            
            # Format the filename to .png and save it into the tiles_png folder
            png_filename = filename.replace(".tif", ".png")
            png_out_path = os.path.join(png_dir, png_filename)
            Image.fromarray(img_array).save(png_out_path)
            
            # --- 2. Move the TIF ---
            # Move the original .tif file into the tiles_tif folder to keep the root clean
            tif_out_path = os.path.join(tif_dir, filename)
            shutil.move(tif_path, tif_out_path)
                
    print("\n✅ SAM2-UNet Images formatting complete. TIFs and PNGs have been successfully sorted.")


# format masks, not with C_functions because it's specific to instance segmentation.
def _sam2unet_process_single_mask(tif_path, gdf_full, output_dir):
    """
    Worker function: Opens one TIF, filters the global GeoDataFrame for that 
    specific tile's bounds, rasterizes the roofs, and saves the .png file directly.
    """
    try:
        # 1. Read the spatial bounds of the tile
        with rasterio.open(tif_path) as src:
            tif_bounds = src.bounds
            transform = src.transform
            out_shape = src.shape 
            tile_geom = box(*tif_bounds)
            
        # 2. Spatially filter the GeoDataFrame for intersecting roofs
        gdf = gdf_full[gdf_full.intersects(tile_geom)].copy()
        
        # 3. Rasterize into a flat mask
        if gdf.empty:
            # If there are no labels in this area cut, continue. 
            # Do NOT save a black mask, so the evaluator later skips this tile.
            return False 
        else:
            mask = features.rasterize(
                [(geom, 255) for geom in gdf.geometry],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            
        # 4. Save directly as an 8-bit grayscale PNG
        png_name = os.path.basename(tif_path).replace(".tif", ".png")
        out_path = os.path.join(output_dir, png_name)
        
        Image.fromarray(mask, mode='L').save(out_path)
        return True
        
    except Exception as e:
        print(f"⚠️ Error processing {os.path.basename(tif_path)}: {e}")
        return False

def sam2unet_generate_masks_parallel(tiles_dir, gpkg_path, output_dir):
    """
    Manager function: Loads the GeoPackage once, then distributes the 
    TIF files across available CPU threads for fast direct-to-PNG rasterization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📦 Loading master GeoPackage into memory (this may take a moment)...")
    print(f"   Source: {os.path.basename(gpkg_path)}")
    gdf_full = gpd.read_file(gpkg_path)
    
    tif_files = glob.glob(os.path.join(tiles_dir, "*.tif"))
    if not tif_files:
        print(f"❌ No .tif files found in {tiles_dir}")
        return

    # Use all cores minus 2 to keep your system responsive
    max_workers = max(1, os.cpu_count() - 2)
    print(f"🚀 Firing up {max_workers} threads to rasterize {len(tif_files)} masks...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks. We pass the loaded `gdf_full` memory reference to every thread.
        futures = [
            executor.submit(_sam2unet_process_single_mask, tif_path, gdf_full, output_dir) 
            for tif_path in tif_files
        ]
        
        # Track progress dynamically with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tif_files), desc="Generating SAM2 Masks"):
            pass

    print(f"✅ All masks generated directly to: {os.path.abspath(output_dir)}")