import argparse
import sys
import os
import logging
import numpy as np
import poseidon_core

def main():
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Pre-compute grid for rectification array jobs.")
    
    parser.add_argument("--lidar_file", type=str, required=True)
    parser.add_argument("--min_x", type=float, required=True)
    parser.add_argument("--max_x", type=float, required=True)
    parser.add_argument("--min_y", type=float, required=True)
    parser.add_argument("--max_y", type=float, required=True)
    parser.add_argument("--grid_dir", type=str, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--lidar_units", type=str, default="meters")
    parser.add_argument("--grid_descr", type=str, required=True)
    
    args = parser.parse_args()

    os.makedirs(args.grid_dir, exist_ok=True)
    out_file = os.path.join(args.grid_dir, f"shared_grid_{args.grid_descr}.npz")

    if os.path.exists(out_file):
        logger.warning(f"Grid file already exists at {out_file}. Overwriting...")

    logger.info(f"Loading LiDAR data from: {args.lidar_file}")
    grid_gen = poseidon_core.GridGenerator(
        args.lidar_file, args.min_x, args.max_x, args.min_y, args.max_y,
        extent_units="meters", lidar_units=args.lidar_units,
    )

    logger.info("Creating point array from LiDAR data...")
    pts_array = grid_gen.create_point_array()

    logger.info(f"Generating grid at {args.resolution}m resolution...")
    grid_x, grid_y, grid_z = grid_gen.gen_grid(
        args.resolution, pts_array, dir=args.grid_dir, grid_descriptor=args.grid_descr,
    )

    logger.info(f"Saving grid arrays to {out_file}...")
    np.savez_compressed(out_file, x=grid_x, y=grid_y, z=grid_z)
    
    logger.info("Grid prep complete!")

if __name__ == "__main__":
    main()
