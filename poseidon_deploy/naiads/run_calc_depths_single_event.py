import argparse
import sys
import os
import logging
import numpy as np
import poseidon_core

def main():
    log_format = "[%(asctime)s] [%(processName)-12s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run single event depth map calculation.")

    parser.add_argument("--target_event_dir", type=str, required=True, help="Specific event folder to process.")
    parser.add_argument("--grid_dir", type=str, required=True)
    parser.add_argument("--grid_descr", type=str, required=True)
    
    parser.add_argument("--zarr_base", type=str, default="zarr")
    parser.add_argument("--zarr_label_dir", type=str, default="labels_rects")
    parser.add_argument("--zarr_depth_dir", type=str, default="depth_maps")
    parser.add_argument("--plot_base_dir", type=str, default="plots")

    args = parser.parse_args()

    logger.info("--- Starting Depth Calculation Pipeline ---")

    # --- Step 1: Load Pre-computed Grid ---
    grid_file = os.path.join(args.grid_dir, f"shared_grid_{args.grid_descr}.npz")
    logger.info(f"Loading pre-computed grid from: {grid_file}")
    
    if not os.path.exists(grid_file):
        logger.error(f"Grid file not found! Expected: {grid_file}")
        sys.exit(1)

    grid_data = np.load(grid_file)
    grid_z = grid_data['z']

    # --- Step 2: Initialize Depth Map Processor ---
    logger.info("Initializing DepthMapProcessor...")
    processor = poseidon_core.DepthMapProcessor(elevation_grid=grid_z, plot_edges=True)

    # --- Step 3: Define Zip Paths and Process ---
    labels_zip = os.path.join(args.target_event_dir, args.zarr_base, f"{args.zarr_label_dir}.zip")
    depths_zip = os.path.join(args.target_event_dir, args.zarr_base, f"{args.zarr_depth_dir}.zip")
    plot_dir = os.path.join(args.target_event_dir, args.plot_base_dir)

    if not os.path.exists(labels_zip):
        logger.error(f"ERROR: Labels zip file not found at {labels_zip}. Cannot compute depths.")
        sys.exit(1)

    os.makedirs(os.path.dirname(depths_zip), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    logger.info(f"Processing event: {os.path.basename(args.target_event_dir)}")
    processor.process_depth_maps(
        labels_zarr_zip_path=labels_zip,
        depth_map_zarr_zip_path=depths_zip,
        pond_edge_elev_plot_dir=plot_dir
    )

    logger.info("--- Depth Calculation Pipeline Complete ---")

if __name__ == "__main__":
    main()
