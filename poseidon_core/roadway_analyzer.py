import os
import json
import zarr
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import re

def _extract_timestamp(filename):
    pattern = r"\d{14}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None

def _log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

class RoadwayAnalyzer:
    def __init__(self, target_event_dir, labelme_json_path, line_label="roadway", step_size=1.0, statistic="95_perc"):
        self.event_path = target_event_dir
        self.event_name = os.path.basename(target_event_dir)
        self.labelme_json_path = labelme_json_path
        self.line_label = line_label
        self.step_size = step_size
        self.statistic = statistic
        
        self.transect_name = os.path.splitext(os.path.basename(labelme_json_path))[0]
        self.transect_coords = self._get_transect_from_labelme()
        
        if self.transect_coords is None:
            raise ValueError(f"Could not find a line labeled '{line_label}' in {labelme_json_path}")
            
        _log(f"Initialized RoadwayAnalyzer for '{self.statistic}' using transect '{self.transect_name}'. Length: {len(self.transect_coords)} points.")

    def _get_transect_from_labelme(self):
        try:
            with open(self.labelme_json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            _log(f"Error reading LabelMe JSON: {e}")
            return None

        line_points = None
        for shape in data['shapes']:
            if shape['label'] == self.line_label and shape['shape_type'] in ['line', 'linestrip']:
                line_points = np.array(shape['points'])
                break
        
        if line_points is None:
            available_labels = [s.get('label', 'unknown') for s in data.get('shapes', [])]
            _log(f"debug: Found labels in JSON: {available_labels}")
            return None

        dists = np.sqrt(np.sum(np.diff(line_points, axis=0)**2, axis=1))
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
        
        fx = interp1d(cumulative_dist, line_points[:, 0])
        fy = interp1d(cumulative_dist, line_points[:, 1])
        
        new_dists = np.arange(0, cumulative_dist[-1], self.step_size)
        new_x = fx(new_dists)
        new_y = fy(new_dists)
        
        coords = np.column_stack((np.round(new_y), np.round(new_x))).astype(int)
        return coords

    def process_single_event(self):
        """Orchestrates the extraction and CSV generation for a single event."""
        _log(f"--- Starting Roadway Analysis for {self.event_name} ---")
        self._gen_transect_depths()
        self._process_roadway_accessibility()
        _log(f"--- Finished Roadway Analysis for {self.event_name} ---")

    def _gen_transect_depths(self):
        in_zip_path = os.path.join(self.event_path, "zarr", "depth_maps.zip")
        out_zip_path = os.path.join(self.event_path, "zarr", f"{self.transect_name}_transect_depths.zip")

        if not os.path.exists(in_zip_path):
            _log(f"Source depth maps not found: {in_zip_path}")
            return

        # 1. Open Source ZipStore
        in_backend = zarr.storage.ZipStore(in_zip_path, mode="r")
        in_root = zarr.open_group(store=in_backend, mode="r")
        
        target_suffix = f"depth_map_{self.statistic}"
        
        # Filter for the specific statistic, explicitly excluding WSE maps
        file_names = sorted([
            f for f in in_root.keys() 
            if f.endswith(target_suffix) and "wse_map" not in f
        ])
        
        num_files = len(file_names)
        if num_files == 0:
            _log(f"No depth maps found matching suffix '{target_suffix}'")
            in_backend.close()
            return

        transect_depth_array = np.empty((num_files, len(self.transect_coords)), dtype=np.float32)
        timestamp_list = []

        # 2. Extract Data
        for idx, file_name in enumerate(file_names):
            timestamp_list.append(_extract_timestamp(file_name))
            
            try:
                depth_map = in_root[file_name][:] 
                
                ys = self.transect_coords[:, 0]
                xs = self.transect_coords[:, 1]
                
                h, w = depth_map.shape
                valid_mask = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
                
                transect_depth_array[idx, :] = np.nan
                transect_depth_array[idx, valid_mask] = depth_map[ys[valid_mask], xs[valid_mask]]
                
            except Exception as e:
                _log(f"Error processing {file_name}: {e}")
                transect_depth_array[idx, :] = np.nan

        # 3. Save to Output ZipStore
        datetimes = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30")
        
        try:
            out_backend = zarr.storage.ZipStore(out_zip_path, mode="w")
            out_root = zarr.open_group(store=out_backend, mode="w")
            
            # Using Zarr V3 chunks logic
            out_root.create_array("timestamps", data=datetimes, chunks=datetimes.shape, overwrite=True)
            out_root.create_array(f"{self.transect_name}_depths", data=transect_depth_array, chunks=transect_depth_array.shape, overwrite=True)
            
            out_backend.close()
            _log(f"Successfully generated {out_zip_path}")
        except Exception as e:
            _log(f"Failed to save ZipStore: {e}")
            
        in_backend.close()

    def _process_roadway_accessibility(self):
        zip_store_path = os.path.join(self.event_path, "zarr", f"{self.transect_name}_transect_depths.zip")

        if not os.path.exists(zip_store_path):
            return

        try:
            # 1. Read from the ZipStore
            backend = zarr.storage.ZipStore(zip_store_path, mode="r")
            root = zarr.open_group(store=backend, mode="r")
            
            timestamps = root["timestamps"][:]
            transect_depths = root[f"{self.transect_name}_depths"][:]
            backend.close()
            
            datetimes = pd.to_datetime(timestamps, utc=True)
            
            # 2. Statistics Calculation
            impassable = np.any(transect_depths > 0.0, axis=1).astype(int)
            
            with np.errstate(all='ignore'): # Suppress warnings for all-NaN slices
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

            # 3. Save CSV
            output_path = os.path.join(self.event_path, f"{self.transect_name}_accessibility_time_series.csv")
            stats_df.to_csv(output_path, index=False)
            _log(f"Accessibility CSV saved to {output_path}")
            
        except Exception as e:
            _log(f"Error calculating stats: {e}")