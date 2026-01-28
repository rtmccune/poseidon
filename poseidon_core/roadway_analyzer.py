import os
import json
import zarr
import numpy as np
import pandas as pd
from datetime import datetime
from mpi4py import MPI
from scipy.interpolate import interp1d

# Helper for timestamp extraction (mirrors your existing utils)
import re

def _extract_timestamp(filename):
    pattern = r"\d{14}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None

def _log(message):
    """Helper for clean HPC logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

class RoadwayAnalyzer:
    def __init__(self, main_dir, labelme_json_path, line_label="roadway", step_size=1.0):
        """
        initializes the analyzer.

        Parameters
        ----------
        main_dir : str
            Path to the main directory containing flood events.
        labelme_json_path : str
            Path to the LabelMe JSON file defining the roadway line.
        line_label : str
            The label name given to the line in LabelMe (default: "roadway").
        step_size : float
            The spacing (in pixels) for interpolating points along the line.
            1.0 means every pixel along the line is sampled.
        """
        self.main_dir = main_dir
        self.labelme_json_path = labelme_json_path
        self.line_label = line_label
        self.step_size = step_size
        
        # Parse the JSON immediately to get the coordinates
        self.transect_coords = self._get_transect_from_labelme()
        
        if self.transect_coords is None:
            raise ValueError(f"Could not find a line labeled '{line_label}' in {labelme_json_path}")
            
        _log(f"Initialized RoadwayAnalyzer. Transect length: {len(self.transect_coords)} points.")

    def _get_transect_from_labelme(self):
        """
        Parses a LabelMe JSON file to extract coordinates along a polyline.
        Interpolates points to ensure dense sampling along the line.
        """
        try:
            with open(self.labelme_json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            _log(f"Error reading LabelMe JSON: {e}")
            return None

        # Find the shape with the matching label
        line_points = None
        for shape in data['shapes']:
            if shape['label'] == self.line_label and shape['shape_type'] == 'linestrip':
                line_points = np.array(shape['points'])
                break
        
        if line_points is None:
            return None

        # Interpolate points along the line
        # Calculate cumulative distance along the line
        dists = np.sqrt(np.sum(np.diff(line_points, axis=0)**2, axis=1))
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
        
        # Create interpolation functions for X and Y
        fx = interp1d(cumulative_dist, line_points[:, 0])
        fy = interp1d(cumulative_dist, line_points[:, 1])
        
        # Generate new distances at the specified step size
        new_dists = np.arange(0, cumulative_dist[-1], self.step_size)
        
        # Generate new coordinates
        new_x = fx(new_dists)
        new_y = fy(new_dists)
        
        # Stack and round to nearest integer indices (column, row) -> (x, y)
        # Note: Arrays are indexed [row, col], so we need [y, x] for array indexing later
        coords = np.column_stack((np.round(new_y), np.round(new_x))).astype(int)
        
        return coords

    def list_flood_event_folders(self):
        return sorted([
            f for f in os.listdir(self.main_dir)
            if os.path.isdir(os.path.join(self.main_dir, f))
        ])

    def gen_transect_depths(self, flood_event, flood_event_path):
        """
        Extracts depth values along the transect for every time step in a flood event.
        Saves the result to a Zarr store.
        """
        depth_maps_zarr_dir = os.path.join(flood_event_path, "zarr", "depth_maps")
        output_zarr_store = os.path.join(flood_event_path, "zarr", "roadway_transect_depths")

        if not os.path.exists(depth_maps_zarr_dir):
            return

        file_names = sorted([
            f for f in os.listdir(depth_maps_zarr_dir) 
            if f.endswith("_95_perc") # Filter for specific statistic maps
        ])
        
        num_files = len(file_names)
        if num_files == 0:
            return

        # Preallocate array: [Time, Transect_Points]
        transect_depth_array = np.empty((num_files, len(self.transect_coords)), dtype=np.float32)
        timestamp_list = []

        for idx, file_name in enumerate(file_names):
            timestamp = _extract_timestamp(file_name)
            timestamp_list.append(timestamp)

            file_zarr_store = os.path.join(depth_maps_zarr_dir, file_name)
            
            try:
                img_store = zarr.open(file_zarr_store, mode="r")
                depth_map = img_store[:] # Load into memory (it's usually small enough)

                # Extract values using advanced indexing
                # transect_coords is [N, 2] where col 0 is y (row), col 1 is x (col)
                ys = self.transect_coords[:, 0]
                xs = self.transect_coords[:, 1]
                
                # Check bounds
                h, w = depth_map.shape
                valid_mask = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                
                # Fill invalid points with NaN, valid points with data
                transect_depth_array[idx, :] = np.nan
                transect_depth_array[idx, valid_mask] = depth_map[ys[valid_mask], xs[valid_mask]]
                
            except Exception as e:
                _log(f"Error processing {file_name} in {flood_event}: {e}")
                transect_depth_array[idx, :] = np.nan

        # Save to Zarr
        datetimes = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30")
        
        try:
            root = zarr.open_group(output_zarr_store, mode="w")
            root.create_array("timestamps", data=datetimes)
            root.create_array("roadway_transect_depths", data=transect_depth_array)
        except Exception as e:
            _log(f"Failed to save Zarr for {flood_event}: {e}")

    def process_roadway_accessibility(self, flood_event):
        """
        Loads the transect depths and calculates summary statistics (CSV).
        """
        flood_event_path = os.path.join(self.main_dir, flood_event)
        zarr_store_path = os.path.join(flood_event_path, "zarr", "roadway_transect_depths")

        if not os.path.exists(zarr_store_path):
            return

        try:
            root = zarr.open(zarr_store_path, mode="r")
            timestamps = root["timestamps"][:]
            transect_depths = root["roadway_transect_depths"][:]
            
            datetimes = pd.to_datetime(timestamps, utc=True)
            
            # --- Statistics Calculation ---
            # 1. Impassable: 1 if ANY point on the line > 0 (or a threshold like 0.1m)
            impassable = np.any(transect_depths > 0.0, axis=1).astype(int)
            
            # 2. Depth Stats across the line for each timestep
            mean_depths = np.nanmean(transect_depths, axis=1)
            median_depths = np.nanmedian(transect_depths, axis=1)
            max_depths = np.nanmax(transect_depths, axis=1)
            min_depths = np.nanmin(transect_depths, axis=1)

            stats_df = pd.DataFrame({
                "Time": datetimes,
                "Impassable": impassable,
                "MeanDepth": mean_depths,
                "MedianDepth": median_depths,
                "MaxDepth": max_depths,
                "MinDepth": min_depths,
            }).sort_values(by="Time")

            output_path = os.path.join(flood_event_path, "roadway_accessibility_time_series.csv")
            stats_df.to_csv(output_path, index=False)
            
        except Exception as e:
            _log(f"Error calculating stats for {flood_event}: {e}")

    def run_hpc_pipeline(self):
        """
        MPI-enabled pipeline runner.
        Rank 0 lists folders.
        All ranks process their share of folders.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            _log(f"Starting Roadway Analysis on {size} ranks.")
            flood_event_folders = self.list_flood_event_folders()
        else:
            flood_event_folders = None

        # Broadcast folders
        flood_event_folders = comm.bcast(flood_event_folders, root=0)

        if not flood_event_folders:
            return

        # Distribute work
        chunk_size = len(flood_event_folders) // size
        start = rank * chunk_size
        end = start + chunk_size if rank != size - 1 else len(flood_event_folders)
        
        my_folders = flood_event_folders[start:end]
        
        _log(f"Rank {rank} processing {len(my_folders)} events.")

        for i, flood_event in enumerate(my_folders):
            # 1. Generate Zarr Data (Heavy lifting)
            self.gen_transect_depths(flood_event, os.path.join(self.main_dir, flood_event))
            
            # 2. Process Statistics (Fast)
            self.process_roadway_accessibility(flood_event)
            
            if i % 5 == 0:
                _log(f"Rank {rank}: Completed {i+1}/{len(my_folders)} events.")

        _log(f"Rank {rank} Finished.")