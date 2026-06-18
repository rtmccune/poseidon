import argparse
import logging
import os
import sys
import numpy as np
import poseidon_core

SENSOR_CONFIG = {
    "CB_03": np.array([[15, 717], [224, 781], [177, 905]]),
    "DE_01": np.array([
        [1610, 1847], [2682, 3176], [1756, 1990], [2120, 2414], [2556, 2928]
    ])
}

def main():
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run single event plotting pipeline.")

    parser.add_argument("--target_event_dir", type=str, required=True)
    parser.add_argument("--min_x", type=float, required=True)
    parser.add_argument("--max_x", type=float, required=True)
    parser.add_argument("--min_y", type=float, required=True)
    parser.add_argument("--max_y", type=float, required=True)
    parser.add_argument("--location", type=str, required=True, choices=SENSOR_CONFIG.keys())
    parser.add_argument("--bbox_crs", type=str, default="EPSG:32119")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--stats", nargs="+", default=["95_perc"])
    parser.add_argument("--basemap", type=str, default=None)
    
    args = parser.parse_args()

    try:
        virtual_sensors = SENSOR_CONFIG[args.location]
        logger.info(f"Loaded sensor config for: '{args.location}'")
    except KeyError:
        logger.error(f"Location '{args.location}' not found.")
        sys.exit(1)

    logger.info("--- Initializing DepthMapPlotter ---")

    plotter = poseidon_core.DepthMapPlotter(
        target_event_dir=args.target_event_dir,
        min_x_extent=args.min_x,
        max_x_extent=args.max_x,
        min_y_extent=args.min_y,
        max_y_extent=args.max_y,
        resolution_m=args.resolution,
        bbox_crs=args.bbox_crs,
        virtual_sensor_locations=virtual_sensors,
        plot_sensors=True,
        basemap_path=args.basemap 
    )

    logger.info("--- Starting Preprocessing ---")
    plotter.preprocess_single_event()

    logger.info(f"--- Plotting Depth Maps for stats: {args.stats} ---")
    plotter.process_single_flood_event(stats_to_plot=args.stats)
    
    logger.info("--- Plotting Water Level Time Series ---")
    plotter.plot_all_time_series()

    logger.info("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()
