import argparse
import sys
import os
import logging
import numpy as np
import poseidon_core

# --- Camera Configuration ---
INTRINSICS_CONFIG = {
    "suds_cam": np.array(
        [3040, 4056, 1503.0136, 2163.4301, 2330.4972, 2334.0017, -0.3587, 0.1388, -0.0266, -0.0046, 0.0003]
    )
}

EXTRINSICS_CONFIG = {
    "CB_03": np.array([712159.597863065, 33136.9994153273, 3.72446811607855, 1.30039127961854, 1.02781393967485, -0.160877893129538]),
    "DE_01": np.array([847955.4296, 127408.728, 4.4922, 4.38504, 1.14484, 0.01305]),
}

def process_event_folder(event_dir_path, rectifier, args):
    logger = logging.getLogger(__name__)
    subfolder_name = os.path.basename(event_dir_path)

    orig_images_folder = os.path.join(event_dir_path, args.image_subfolder)
    labels_folder = os.path.join(event_dir_path, args.label_subfolder)

    if not (os.path.exists(orig_images_folder) and os.path.exists(labels_folder)):
        logger.info(f"Skipping {subfolder_name}: Missing '{args.image_subfolder}' or '{args.label_subfolder}'.")
        return ("skip", subfolder_name)

    try:
        logger.info(f"--- Processing event: {subfolder_name} ---")

        zarr_output_dir = os.path.join(event_dir_path, args.zarr_base)
        os.makedirs(zarr_output_dir, exist_ok=True)

        zarr_store_orig = os.path.join(zarr_output_dir, args.zarr_orig_name)
        zarr_store_labels = os.path.join(zarr_output_dir, args.zarr_label_name)

        logger.info(f"[{subfolder_name}] Rectifying images from: {args.image_subfolder}")
        rectifier.merge_rectify_folder(orig_images_folder, zarr_store_orig)

        logger.info(f"[{subfolder_name}] Rectifying labels from: {args.label_subfolder}")
        rectifier.merge_rectify_folder(labels_folder, zarr_store_labels, labels=True)

        logger.info(f"+++ Successfully processed event {subfolder_name} +++")
        return ("success", subfolder_name)

    except Exception as e:
        logger.error(f"!!! ERROR processing {subfolder_name}: {e} !!!")
        return ("error", subfolder_name)

def main():
    log_format = "[%(asctime)s] [%(processName)-12s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run single image rectification pipeline.")

    # --- Replaced event_dir with target_event_dir ---
    parser.add_argument("--lidar_file", type=str, required=True)
    parser.add_argument("--target_event_dir", type=str, required=True, help="Specific event folder to process.")
    
    parser.add_argument("--min_x", type=float, required=True)
    parser.add_argument("--max_x", type=float, required=True)
    parser.add_argument("--min_y", type=float, required=True)
    parser.add_argument("--max_y", type=float, required=True)
    
    parser.add_argument("--camera_name", type=str, required=True, choices=EXTRINSICS_CONFIG.keys())
    parser.add_argument("--intrinsics_name", type=str, default="suds_cam", choices=INTRINSICS_CONFIG.keys())
    parser.add_argument("--grid_dir", type=str, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--lidar_units", type=str, default="meters")
    parser.add_argument("--grid_descr", type=str, required=True)
    
    parser.add_argument("--image_subfolder", type=str, default="orig_images")
    parser.add_argument("--label_subfolder", type=str, default="labels")
    parser.add_argument("--zarr_base", type=str, default="zarr")
    parser.add_argument("--zarr_orig_name", type=str, default="orig_image_rects")
    parser.add_argument("--zarr_label_name", type=str, default="labels_rects")
    
    parser.add_argument("--disable_gpu", action="store_false", dest="use_gpu")
    parser.set_defaults(use_gpu=True)

    args = parser.parse_args()

    logger.info("--- Starting Rectification Pipeline ---")

    try:
        intrinsics = INTRINSICS_CONFIG[args.intrinsics_name]
        extrinsics = EXTRINSICS_CONFIG[args.camera_name]
    except KeyError as e:
        logger.error(f"Error: Config name {e} not found.")
        sys.exit(1)

    # --- Load Pre-computed Grid ---
    grid_file = os.path.join(args.grid_dir, f"shared_grid_{args.grid_descr}.npz")
    logger.info(f"Loading pre-computed grid from: {grid_file}")
    
    if not os.path.exists(grid_file):
        logger.error(f"Grid file not found! Did you run the prep script? Expected: {grid_file}")
        sys.exit(1)

    grid_data = np.load(grid_file)
    grid_x = grid_data['x']
    grid_y = grid_data['y']
    grid_z = grid_data['z']

    # --- Initialize Image Rectifier ---
    logger.info(f"Initializing ImageRectifier... (GPU Enabled: {args.use_gpu})")
    rectifier = poseidon_core.ImageRectifier(
        intrinsics, extrinsics, grid_x, grid_y, grid_z, use_gpu=args.use_gpu
    )

    # --- Process the Single Directory ---
    status, msg = process_event_folder(args.target_event_dir, rectifier, args)
    
    if status == "success":
        logger.info(f"Successfully finished rectification for {msg}.")
    else:
        logger.warning(f"Rectification ended with status '{status}' for {msg}.")
        if status == "error":
            sys.exit(1)

if __name__ == "__main__":
    main()
