import argparse
import sys
import os
import concurrent.futures
import logging

import numpy as np
import poseidon_core


def process_event_folder(event_dir_path, processor, args):
    """
    Worker function to process a single event folder.
    Returns (status, message) tuple.
    """
    logger = logging.getLogger(__name__)
    subfolder_name = os.path.basename(event_dir_path)

    # Check if it's a directory (scandir already did, but as a standalone func, we check)
    if not os.path.isdir(event_dir_path):
        logger.info(f"Entry {subfolder_name} is not a directory.")
        return ("skip", f"Entry {subfolder_name} is not a directory.")

    labels_folder = os.path.join(
        event_dir_path, args.zarr_base, args.zarr_label_dir
    )

    # Check if the required input folders exist
    if not os.path.exists(labels_folder):
        logger.info(
            f"Skipping {subfolder_name}: Missing '{args.zarr_label_dir}'."
        )
        return (
            "skip",
            f"Skipping {subfolder_name}: Missing '{args.zarr_label_dir}'.",
        )

    try:
        logger.info(f"--- Processing event: {subfolder_name} ---")

        # Define paths for saving rectified images, creating zarr dir
        zarr_output_dir = os.path.join(
            event_dir_path, args.zarr_base, args.zarr_depth_dir
        )
        os.makedirs(zarr_output_dir, exist_ok=True)

        # --- Run Calculation ---
        logger.info(
            f"[{subfolder_name}] Calculating depths from: {args.zarr_label_dir}"
        )
        processor.process_depth_maps(
            labels_folder,
            zarr_output_dir,
            pond_edge_elev_plot_dir=os.path.join(
                event_dir_path, args.plot_base_dir
            ),
        )

        logger.info(f"+++ Successfully processed event {subfolder_name} +++")
        return ("success", subfolder_name)

    except Exception as e:
        logger.error(f"!!! ERROR processing {subfolder_name}: {e} !!!")
        return ("error", f"Error in {subfolder_name}: {e}")


def main():
    log_format = "[%(asctime)s] [%(threadName)-12s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)  # Get logger for this script

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
        "--min_x",
        type=float,
        required=True,
        help="Minimum X extent for the grid.",
    )
    parser.add_argument(
        "--max_x",
        type=float,
        required=True,
        help="Maximum X extent for the grid.",
    )
    parser.add_argument(
        "--min_y",
        type=float,
        required=True,
        help="Minimum Y extent for the grid.",
    )
    parser.add_argument(
        "--max_y",
        type=float,
        required=True,
        help="Maximum Y extent for the grid.",
    )

    # --- Configuration Arguments ---
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
        "--zarr_base",
        type=str,
        default="zarr",
        help="Name of the base folder to store zarr outputs in (default: 'zarr').",
    )
    parser.add_argument(
        "--zarr_label_dir",
        type=str,
        default="labels_rects",
        help="Name of the zarr store containing rectified labels (default: 'labels_rects').",
    )
    parser.add_argument(
        "--zarr_depth_dir",
        type=str,
        default="depth_maps",
        help="Name of the zarr store for depth maps (default: 'depth_maps').",
    )
    parser.add_argument(
        "--plot_base_dir",
        type=str,
        default="plots",
        help="Name of the directory to save plots (default: 'plots').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,  # You can tune this number
        help="Number of parallel worker threads to use (default: 4).",
    )

    args = parser.parse_args()

    logger.info("--- Starting Depth Calculation Pipeline ---")

    # --- Step 1: Initialize Grid Generator ---
    logger.info(f"Loading LiDAR data from: {args.lidar_file}")
    grid_gen = poseidon_core.GridGenerator(
        args.lidar_file,
        args.min_x,
        args.max_x,
        args.min_y,
        args.max_y,
        extent_units="meters",
        lidar_units=args.lidar_units,
    )

    logger.info("Creating point array from LiDAR data...")
    pts_array = grid_gen.create_point_array()

    logger.info(f"Generating grid at {args.resolution}m resolution...")
    grid_x, grid_y, grid_z = grid_gen.gen_grid(
        args.resolution,
        pts_array,
        dir=args.grid_dir,
        grid_descriptor=args.grid_descr,
    )

    # --- Step 2: Initialize Depth Map Processor ---
    logger.info(f"Initializing DepthMapProcessor... ")
    processor = poseidon_core.DepthMapProcessor(
        grid_z,
    )

    # --- Step 3: Process Each Event Subfolder (in parallel) ---
    logger.info(
        f"Iterating through event folders in: {args.event_dir} using {args.workers} workers"
    )

    # Use os.scandir for efficient directory listing
    event_paths = []
    with os.scandir(args.event_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                event_paths.append(entry.path)

    logger.info(f"Found {len(event_paths)} potential event directories.")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Use ThreadPoolExecutor to parallelize the work
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="EventWorker"
    ) as executor:
        # Create a "future" for each call to process_event_folder
        # We pass the rectifier and args objects to each thread
        future_to_path = {
            executor.submit(process_event_folder, path, processor, args): path
            for path in event_paths
        }

        # As each future completes, get its result
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                status, message = future.result()
                if status == "success":
                    processed_count += 1
                elif status == "skip":
                    skipped_count += 1
                elif status == "error":
                    error_count += 1
            except Exception as e:
                logger.error(f"!!! FATAL ERROR for path {path}: {e} !!!")
                error_count += 1

    logger.info("\n--- Depth Calculation Pipeline Complete ---")
    logger.info(f"Total events processed: {processed_count}")
    logger.info(f"Total events skipped:   {skipped_count}")
    logger.info(f"Total events (errors):  {error_count}")


if __name__ == "__main__":
    main()
