import argparse
import sys
import os
import concurrent.futures

import numpy as np
import poseidon_core


# --- Camera Configuration ---
# Store camera parameters in dictionaries to be selected by name
# You can add more cameras to these configs as needed.

INTRINSICS_CONFIG = {
    'suds_cam': np.array([
        3040,      # number of pixel columns
        4056,      # number of pixel rows
        1503.0136, # U component of principal point
        2163.4301, # V component of principal point
        2330.4972, # U component of focal length
        2334.0017, # V component of focal length
        -0.3587,   # radial distortion
        0.1388,    # radial distortion
        -0.0266,   # radial distortion
        -0.0046,   # tangential distortion
        0.0003     # tangential distortion
    ])
}

EXTRINSICS_CONFIG = {
    'CB_03': np.array([
        712159.597863065, # camera x in world
        33136.9994153273, # camera y in world
        3.72446811607855, # camera elev in world
        1.30039127961854, # azimuth
        1.02781393967485, # tilt
        -0.160877893129538 # roll/swing
    ]),
    'DE_01': np.array([
        847955.4296, # camera x in world
        127408.728,  # camera y in world
        4.4922,      # camera elev in world
        4.38504,     # azimuth
        1.14484,     # tilt
        0.01305      # roll/swing
    ])
}

def process_event_folder(event_dir_path, rectifier, args):
    """
    Worker function to process a single event folder.
    Returns (status, message) tuple.
    """
    subfolder_name = os.path.basename(event_dir_path)
    
    # Check if it's a directory (scandir already did, but as a standalone func, we check)
    if not os.path.isdir(event_dir_path):
        return ('skip', f"Entry {subfolder_name} is not a directory.")

    orig_images_folder = os.path.join(event_dir_path, args.image_subfolder)
    labels_folder = os.path.join(event_dir_path, args.label_subfolder)

    # Check if the required input folders exist
    if not (os.path.exists(orig_images_folder) and os.path.exists(labels_folder)):
        return ('skip', f"Skipping {subfolder_name}: Missing '{args.image_subfolder}' or '{args.label_subfolder}'.")

    try:
        print(f"--- Processing event: {subfolder_name} ---")
        
        # Define paths for saving rectified images, creating zarr dir
        zarr_output_dir = os.path.join(event_dir_path, args.zarr_base)
        os.makedirs(zarr_output_dir, exist_ok=True)
        
        zarr_store_orig = os.path.join(zarr_output_dir, args.zarr_orig_name)
        zarr_store_labels = os.path.join(zarr_output_dir, args.zarr_label_name)

        # --- Run Rectification for Images ---
        print(f"[{subfolder_name}] Rectifying images from: {args.image_subfolder}")
        rectifier.merge_rectify_folder(orig_images_folder, zarr_store_orig)
        
        # --- Run Rectification for Labels ---
        print(f"[{subfolder_name}] Rectifying labels from: {args.label_subfolder}")
        rectifier.merge_rectify_folder(labels_folder, zarr_store_labels, labels=True)
        
        print(f"+++ Successfully processed event {subfolder_name} +++")
        return ('success', subfolder_name)
        
    except Exception as e:
        print(f"!!! ERROR processing {subfolder_name}: {e} !!!", file=sys.stderr)
        return ('error', f"Error in {subfolder_name}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run the full image rectification pipeline for flood events."
    )

    # --- Path Arguments ---
    parser.add_argument(
        "--lidar_file",
        type=str,
        required=True,
        help="Path to the .laz or .las LiDAR point cloud file.",
    )
    parser.add_argument(
        "--event_dir",
        type=str,
        required=True,
        help="Main directory containing event subfolders (e.g., '.../flood_events').",
    )

    # --- Grid Extent Arguments ---
    parser.add_argument(
        "--min_x", type=float, required=True, help="Minimum X extent for the grid."
    )
    parser.add_argument(
        "--max_x", type=float, required=True, help="Maximum X extent for the grid."
    )
    parser.add_argument(
        "--min_y", type=float, required=True, help="Minimum Y extent for the grid."
    )
    parser.add_argument(
        "--max_y", type=float, required=True, help="Maximum Y extent for the grid."
    )

    # --- Configuration Arguments ---
    parser.add_argument(
        "--camera_name",
        type=str,
        required=True,
        choices=EXTRINSICS_CONFIG.keys(),
        help="Name of the camera extrinsics config to use.",
    )
    parser.add_argument(
        "--intrinsics_name",
        type=str,
        default="suds_cam",
        choices=INTRINSICS_CONFIG.keys(),
        help="Name of the camera intrinsics config to use (default: 'suds_cam').",
    )
    parser.add_argument(
        "--grid_dir",
        type=str,
        required=True,
        help="Name of the directory to save generated grids to.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Grid resolution in meters (default: 0.05).",
    )
    parser.add_argument(
        "--lidar_units",
        type=str,
        default="meters",
        help="Grid units (default: 'meters').",
    )
    parser.add_argument(
        "--grid_descr",
        type=str,
        required=True,
        help="Grid descriptor.",
    )
    parser.add_argument(
        "--image_subfolder",
        type=str,
        default="orig_images",
        help="Name of the subfolder containing original images (default: 'orig_images').",
    )
    parser.add_argument(
        "--label_subfolder",
        type=str,
        default="labels",
        help="Name of the subfolder containing label images (default: 'labels').",
    )
    parser.add_argument(
        "--zarr_base",
        type=str,
        default="zarr",
        help="Name of the base folder to store zarr outputs in (default: 'zarr').",
    )
    parser.add_argument(
        "--zarr_orig_name",
        type=str,
        default="orig_image_rects",
        help="Name of the zarr store for rectified original images (default: 'orig_image_rects').",
    )
    parser.add_argument(
        "--zarr_label_name",
        type=str,
        default="labels_rects",
        help="Name of the zarr store for rectified label images (default: 'labels_rects').",
    )
    parser.add_argument(
        "--disable_gpu",
        action="store_false",
        dest="use_gpu",
        help="Disable GPU acceleration (default: GPU is enabled).",
    )
    # Set the default for use_gpu to True, --disable_gpu will set it to False
    parser.set_defaults(use_gpu=True) 
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4, # You can tune this number
        help="Number of parallel worker threads to use (default: 4)."
    )

    args = parser.parse_args()

    # !!! START OF NEW DEBUG CODE !!!
    print("--- 🕵️ DEBUGGING START 🕵️ ---")
    
    # 1. Verify the file path is correct
    lidar_path = args.lidar_file
    print(f"DEBUG: --lidar_file argument: {lidar_path}")
    
    # 2. Verify the absolute path
    abs_lidar_path = os.path.abspath(lidar_path)
    print(f"DEBUG: Absolute path: {abs_lidar_path}")
    
    # 3. CRITICAL: Check if the file exists from the script's perspective
    file_exists = os.path.exists(abs_lidar_path)
    print(f"DEBUG: Does path exist? {file_exists}")
    
    if not file_exists:
        print("!!! DEBUG: File not found at path. Exiting. !!!")
        sys.exit(1) # Fail fast

    print("--- Starting Rectification Pipeline ---")

    # --- Step 1: Load Camera Parameters ---
    try:
        intrinsics = INTRINSICS_CONFIG[args.intrinsics_name]
        extrinsics = EXTRINSICS_CONFIG[args.camera_name]
        print(f"Loaded intrinsics: '{args.intrinsics_name}'")
        print(f"Loaded extrinsics: '{args.camera_name}'")
    except KeyError as e:
        print(f"Error: Config name {e} not found. Check INTRINSICS_CONFIG and EXTRINSICS_CONFIG.", file=sys.stderr)
        sys.exit(1)

    # !!! START NEW PDAL DEBUGGING (v2) !!!
    import subprocess
    import json
    print("--- 🕵️ PDAL PIPELINE TEST (v2) 🕵️ ---")
    
    # Build the PDAL bounds string in PDAL's expected format
    # Note: No internal spaces, which was also a likely error in my last test
    pdal_bounds = f"([{args.min_x},{args.max_x}],[{args.min_y},{args.max_y}])"
    print(f"DEBUG: Testing with PDAL bounds: {pdal_bounds}")
    
    # Construct a JSON pipeline
    # 1. Read the file
    # 2. Crop to the bounds
    # 3. Get stats (which will give us the count *after* cropping)
    pipeline_json = f"""
    {{
      "pipeline": [
        {{
          "type": "readers.las",
          "filename": "{args.lidar_file}"
        }},
        {{
          "type": "filters.crop",
          "bounds": "{pdal_bounds}"
        }},
        {{
          "type": "filters.stats"
        }}
      ]
    }}
    """
    
    print(f"DEBUG: Testing with PDAL pipeline...")
    
    pdal_cmd = ["pdal", "pipeline", "--stdin"]

    try:
        # Run the command and pass the JSON via stdin
        result = subprocess.run(
            pdal_cmd,
            input=pipeline_json,  # Pass JSON string as stdin
            capture_output=True,
            text=True,
            check=True
        )
        
        # The output of a stats filter is a JSON metadata doc
        pdal_json = json.loads(result.stdout)
        
        # Navigate the JSON to find the count
        stats = pdal_json.get("metadata", {}).get("filters.stats", {})
        count = stats.get("statistic", {}).get("count", "COULD NOT FIND COUNT")
        print(f"DEBUG: PDAL filtered point count (v2): {count}")
        
    except FileNotFoundError:
        print("!!! DEBUG (v2): 'pdal' command not found. Is it in the $PATH? !!!")
    except subprocess.CalledProcessError as e:
        print(f"!!! DEBUG (v2): PDAL command failed: {e} !!!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
    except json.JSONDecodeError:
        print("!!! DEBUG (v2): Could not parse PDAL's JSON output. !!!")
        print(f"STDOUT: {result.stdout}")
    
    print("--- 🕵️ PDAL TEST (v2) COMPLETE 🕵️ ---")
    # !!! END NEW PDAL DEBUGGING (v2) !!!

    # --- Step 2: Initialize Grid Generator ---
    print(f"Loading LiDAR data from: {args.lidar_file}")
    grid_gen = poseidon_core.GridGenerator(
        args.lidar_file,
        args.min_x,
        args.max_x,
        args.min_y,
        args.max_y,
        args.lidar_units
    )

    print("Creating point array from LiDAR data...")
    pts_array = grid_gen.create_point_array()
    
    # !!! MORE DEBUG CODE !!!
    # 4. CRITICAL: Check if the pts_array is empty
    print(f"DEBUG: pts_array shape: {pts_array.shape}")
    print(f"DEBUG: Is pts_array empty? {pts_array.size == 0}")
    print("--- 🕵️ DEBUGGING END 🕵️ ---")
    # !!! END OF DEBUG CODE !!!
    
    print(f"Generating grid at {args.resolution}m resolution...")
    grid_x, grid_y, grid_z = grid_gen.gen_grid(
        args.resolution, 
        pts_array,
        dir=args.grid_dir,
        grid_descriptor=args.grid_descr)

    # --- Step 3: Initialize Image Rectifier ---
    print(f"Initializing ImageRectifier... (GPU Enabled: {args.use_gpu})")
    rectifier = poseidon_core.ImageRectifier(
        intrinsics, extrinsics, grid_x, grid_y, grid_z, use_gpu=args.use_gpu
    )

    # --- Step 4: Process Each Event Subfolder (in parallel) ---
    print(f"Iterating through event folders in: {args.event_dir} using {args.workers} workers")

    # Use os.scandir for efficient directory listing
    event_paths = []
    with os.scandir(args.event_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                event_paths.append(entry.path)

    print(f"Found {len(event_paths)} potential event directories.")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Use ThreadPoolExecutor to parallelize the work
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Create a "future" for each call to process_event_folder
        # We pass the rectifier and args objects to each thread
        future_to_path = {
            executor.submit(process_event_folder, path, rectifier, args): path 
            for path in event_paths
        }
        
        # As each future completes, get its result
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                status, message = future.result()
                if status == 'success':
                    processed_count += 1
                elif status == 'skip':
                    skipped_count += 1
                    print(message) # Print skip messages
                elif status == 'error':
                    error_count += 1
            except Exception as e:
                print(f"!!! FATAL ERROR for path {path}: {e} !!!", file=sys.stderr)
                error_count += 1

    print("\n--- Rectification Pipeline Complete ---")
    print(f"Total events processed: {processed_count}")
    print(f"Total events skipped:   {skipped_count}")
    print(f"Total events (errors):  {error_count}")

if __name__ == "__main__":
    main()