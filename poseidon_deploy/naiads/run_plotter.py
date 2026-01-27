import argparse
import logging
import os
import sys
import numpy as np
from mpi4py import MPI
import poseidon_core

# --- Configuration ---
# Store sensor locations in a dictionary selectable by name.
# Note: I applied the (* 2) operation from your snippet directly here.
SENSOR_CONFIG = {
    "CB_03": np.array([
        [15, 717],
        [224, 781],
        [177, 905]
    ]),
    "DE_01": np.array([
        # Placeholder coords based on your commented out code in the prompt
        # You can update these with the correct Down East coordinates
        [1610, 1847], 
        [2682, 3176], 
        [1756, 1990], 
        [2120, 2414], 
        [2556, 2928]
    ])
}

def generate_time_series_parallel(plotter, logger):
    """
    Parallelizes the generation of time series plots for all images
    across all flood events.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 discovers the flood event folders
    if rank == 0:
        flood_event_folders = plotter._list_flood_event_folders()
    else:
        flood_event_folders = None

    # Broadcast folders to all ranks
    flood_event_folders = comm.bcast(flood_event_folders, root=0)

    if not flood_event_folders:
        if rank == 0:
            logger.warning("No flood event folders found for time series plotting.")
        return

    # Distribute folders among ranks
    n_folders = len(flood_event_folders)
    chunk_size = n_folders // size
    start_index = rank * chunk_size
    end_index = start_index + chunk_size if rank != size - 1 else n_folders
    
    my_folders = flood_event_folders[start_index:end_index]

    logger.info(f"Rank {rank}: Assigned {len(my_folders)} events for time series plotting.")

    for flood_event in my_folders:
        flood_event_path = os.path.join(plotter.main_dir, flood_event)
        orig_images_path = os.path.join(flood_event_path, "orig_images")
        
        # Output folder for time series
        ts_output_folder = os.path.join(flood_event_path, "plots", "time_series_using_depths")
        os.makedirs(ts_output_folder, exist_ok=True)

        if not os.path.exists(orig_images_path):
            continue

        images = sorted(os.listdir(orig_images_path))
        
        for filename in images:
            try:
                plotter.plot_water_level_time_series(
                    filename,
                    flood_event_path,
                    ts_output_folder
                )
            except Exception as e:
                logger.error(f"Rank {rank}: Error plotting time series for {filename} in {flood_event}: {e}")

def main():
    # MPI Setup
    
    # REQUIRED because MPI4PY_RC_INITIALIZE=False in the submission script
    MPI.Init()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Logging Setup
    log_format = f"[%(asctime)s] [Rank {rank}] %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the DepthMapPlotter pipeline via MPI.")

    # --- Path Arguments ---
    parser.add_argument(
        "--event_dir", type=str, required=True,
        help="Main directory containing flood event subfolders.",
    )

    # --- Extent Arguments ---
    parser.add_argument("--min_x", type=float, required=True, help="Min X extent.")
    parser.add_argument("--max_x", type=float, required=True, help="Max X extent.")
    parser.add_argument("--min_y", type=float, required=True, help="Min Y extent.")
    parser.add_argument("--max_y", type=float, required=True, help="Max Y extent.")

    # --- Configuration Arguments ---
    parser.add_argument(
        "--location", type=str, required=True, choices=SENSOR_CONFIG.keys(),
        help="Location key to select virtual sensor coordinates from SENSOR_CONFIG."
    )
    parser.add_argument(
        "--bbox_crs", type=str, default="EPSG:32119",
        help="CRS for the bounding box coords (default: EPSG:32119).",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="Grid resolution in meters.",
    )
    parser.add_argument(
        "--stats", nargs="+", default=["95_perc"],
        help="List of stats to plot (e.g., '95_perc' 'mean').",
    )
    
    args = parser.parse_args()

    # --- Select Sensors ---
    try:
        virtual_sensors = SENSOR_CONFIG[args.location]
        if rank == 0:
            logger.info(f"Loaded sensor config for: '{args.location}'")
    except KeyError:
        if rank == 0:
            logger.error(f"Location '{args.location}' not found in SENSOR_CONFIG.")
        sys.exit(1)

    if rank == 0:
        logger.info("--- Initializing DepthMapPlotter ---")

    # Initialize Plotter
    plotter = poseidon_core.DepthMapPlotter(
        main_dir=args.event_dir,
        min_x_extent=args.min_x,
        max_x_extent=args.max_x,
        min_y_extent=args.min_y,
        max_y_extent=args.max_y,
        resolution_m=args.resolution,
        bbox_crs=args.bbox_crs,
        virtual_sensor_locations=virtual_sensors,
        plot_sensors=True 
    )

    # --- Step 1: Preprocessing (Rank 0 Only) ---
    if rank == 0:
        logger.info("--- Starting Preprocessing (Rank 0) ---")
        try:
            plotter.preprocess_flood_events()
            logger.info("--- Preprocessing Complete ---")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            sys.exit(1)
    
    # Sync all ranks
    comm.Barrier()

    # --- Step 2: Plot Depth Maps (MPI) ---
    if rank == 0:
        logger.info(f"--- Plotting Depth Maps (MPI) for stats: {args.stats} ---")
    
    plotter.process_flood_events_HPC(stats_to_plot=args.stats)
    
    comm.Barrier()

    # --- Step 3: Plot Time Series (MPI) ---
    if rank == 0:
        logger.info("--- Plotting Water Level Time Series (MPI) ---")
        
    generate_time_series_parallel(plotter, logger)

    if rank == 0:
        logger.info("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()