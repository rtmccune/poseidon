import os
import json
import zarr
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
import re
from skimage.draw import polygon  # <--- Added for polygon interior masking

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
        self.roi_label = line_label  # Can be a line or polygon label
        self.step_size = step_size
        self.statistic = statistic
        
        # Base name derived from the JSON (e.g., 'canal_dr_transect' or 'canal_dr_polygon')
        self.roi_name = os.path.splitext(os.path.basename(labelme_json_path))[0]
        self.geom_type = None # Will be set during parsing
        
        self.roi_coords = self._get_coords_from_labelme()
        
        if self.roi_coords is None:
            raise ValueError(f"Could not find a valid shape labeled '{self.roi_label}' in {labelme_json_path}")
            
        _log(f"Initialized RoadwayAnalyzer for '{self.statistic}' using {self.geom_type} '{self.roi_name}'. Area/Length: {len(self.roi_coords)} pixels.")

    def _get_coords_from_labelme(self):
        """Auto-detects lines vs polygons and extracts the correct pixel coordinates."""
        try:
            with open(self.labelme_json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            _log(f"Error reading LabelMe JSON: {e}")
            return None

        points = None
        shape_type = None
        for shape in data['shapes']:
            if shape['label'] == self.roi_label:
                points = np.array(shape['points'])
                shape_type = shape['shape_type']
                break
        
        if points is None:
            available_labels = [s.get('label', 'unknown') for s in data.get('shapes', [])]
            _log(f"debug: Found labels in JSON: {available_labels}")
            return None

        self.geom_type = shape_type

        # --- LINE EXTRACTION ---
        if shape_type in ['line', 'linestrip']:
            dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            cumulative_dist = np.insert(np.cumsum(dists), 0, 0)
            
            fx = interp1d(cumulative_dist, points[:, 0])
            fy = interp1d(cumulative_dist, points[:, 1])
            
            new_dists = np.arange(0, cumulative_dist[-1], self.step_size)
            new_x = fx(new_dists)
            new_y = fy(new_dists)
            
            coords = np.column_stack((np.round(new_y), np.round(new_x))).astype(int)
            return coords

        # --- POLYGON EXTRACTION ---
        elif shape_type == 'polygon':
            # polygon(r, c) takes Rows (Y) and Cols (X)
            rr, cc = polygon(points[:, 1], points[:, 0])
            coords = np.column_stack((rr, cc)).astype(int)
            return coords
            
        else:
            _log(f"Unsupported shape_type '{shape_type}' for label '{self.roi_label}'")
            return None

    def process_single_event(self):
        """Orchestrates the extraction and CSV generation for a single event."""
        _log(f"--- Starting Analysis for {self.event_name} ---")
        self._gen_roi_depths()
        self._process_roadway_accessibility()
        
        # --- NEW CALLS FOR WSE ---
        self._gen_roi_wse()
        self._process_roadway_wse()
        
        _log(f"--- Finished Analysis for {self.event_name} ---")

    def _gen_roi_depths(self):
        in_zip_path = os.path.join(self.event_path, "zarr", "depth_maps.zip")
        out_zip_path = os.path.join(self.event_path, "zarr", f"{self.roi_name}_depths.zip")

        if not os.path.exists(in_zip_path):
            _log(f"Source depth maps not found: {in_zip_path}")
            return

        # 1. Open Source ZipStore
        in_backend = zarr.storage.ZipStore(in_zip_path, mode="r")
        in_root = zarr.open_group(store=in_backend, mode="r")
        
        target_suffix = f"depth_map_{self.statistic}"
        
        file_names = sorted([
            f for f in in_root.keys() 
            if f.endswith(target_suffix) and "wse_map" not in f
        ])
        
        num_files = len(file_names)
        if num_files == 0:
            _log(f"No depth maps found matching suffix '{target_suffix}'")
            in_backend.close()
            return

        roi_depth_array = np.empty((num_files, len(self.roi_coords)), dtype=np.float32)
        timestamp_list = []

        # 2. Extract Data
        for idx, file_name in enumerate(file_names):
            timestamp_list.append(_extract_timestamp(file_name))
            
            try:
                depth_map = in_root[file_name][:] 
                
                ys = self.roi_coords[:, 0]
                xs = self.roi_coords[:, 1]
                
                h, w = depth_map.shape
                
                # Flip the Y-axis to match the Zarr array's geographic orientation
                ys_flipped = (h - 1) - ys
                
                valid_mask = (ys_flipped >= 0) & (ys_flipped < h) & (xs >= 0) & (xs < w)
                
                roi_depth_array[idx, :] = np.nan
                roi_depth_array[idx, valid_mask] = depth_map[ys_flipped[valid_mask], xs[valid_mask]]
                
            except Exception as e:
                _log(f"Error processing {file_name}: {e}")
                roi_depth_array[idx, :] = np.nan

        # 3. Save to Output ZipStore
        datetimes = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30")
        
        try:
            out_backend = zarr.storage.ZipStore(out_zip_path, mode="w")
            out_root = zarr.open_group(store=out_backend, mode="w")
            
            # Save the extracted array using the dynamic ROI name
            out_root.create_array("timestamps", data=datetimes, chunks=datetimes.shape, overwrite=True)
            out_root.create_array(f"{self.roi_name}_depths", data=roi_depth_array, chunks=roi_depth_array.shape, overwrite=True)
            
            out_backend.close()
            _log(f"Successfully generated {out_zip_path}")
        except Exception as e:
            _log(f"Failed to save ZipStore: {e}")
            
        in_backend.close()

    def _process_roadway_accessibility(self):
        zip_store_path = os.path.join(self.event_path, "zarr", f"{self.roi_name}_depths.zip")

        if not os.path.exists(zip_store_path):
            return

        try:
            # 1. Read from the ZipStore
            backend = zarr.storage.ZipStore(zip_store_path, mode="r")
            root = zarr.open_group(store=backend, mode="r")
            
            timestamps = root["timestamps"][:]
            roi_depths = root[f"{self.roi_name}_depths"][:]
            backend.close()
            
            datetimes = pd.to_datetime(timestamps, utc=True)
            
            # Count pixels > 0.0 to get the flooded area, and get the total polygon size
            flooded_pixels = np.sum(roi_depths > 0.0, axis=1)
            total_pixels = roi_depths.shape[1]
            
            # 2. Statistics Calculation
            # NOTE: For a polygon, 'impassable' means ANY pixel inside the polygon is > 0.
            impassable = np.any(roi_depths > 0.0, axis=1).astype(int)
            
            with np.errstate(all='ignore'):
                mean_depths = np.nanmean(roi_depths, axis=1)
                median_depths = np.nanmedian(roi_depths, axis=1)
                max_depths = np.nanmax(roi_depths, axis=1)
                min_depths = np.nanmin(roi_depths, axis=1)

            stats_df = pd.DataFrame({
                "Time": datetimes,
                "Impassable": impassable,
                "FloodedPixels": flooded_pixels,
                "TotalPixels": total_pixels,
                "MeanDepth": mean_depths,
                "MedianDepth": median_depths,
                "MaxDepth": max_depths,
                "MinDepth": min_depths,
            }).sort_values(by="Time")

            # 3. Save CSV
            output_path = os.path.join(self.event_path, f"{self.roi_name}_accessibility_time_series.csv")
            stats_df.to_csv(output_path, index=False)
            _log(f"Accessibility CSV saved to {output_path}")
            
        except Exception as e:
            _log(f"Error calculating stats: {e}")
            
    def _gen_roi_wse(self):
        """Extracts Water Surface Elevation (WSE) maps from the Zarr store."""
        in_zip_path = os.path.join(self.event_path, "zarr", "depth_maps.zip")
        out_zip_path = os.path.join(self.event_path, "zarr", f"{self.roi_name}_wse.zip")

        if not os.path.exists(in_zip_path):
            return

        in_backend = zarr.storage.ZipStore(in_zip_path, mode="r")
        in_root = zarr.open_group(store=in_backend, mode="r")
        
        target_suffix = f"wse_map_{self.statistic}"
        
        # Look specifically FOR the wse_map files
        file_names = sorted([
            f for f in in_root.keys() 
            if f.endswith(target_suffix)
        ])
        
        num_files = len(file_names)
        if num_files == 0:
            _log(f"No WSE maps found matching suffix '{target_suffix}'")
            in_backend.close()
            return
        
        roi_wse_array = np.empty((num_files, len(self.roi_coords)), dtype=np.float32)
        timestamp_list = []

        for idx, file_name in enumerate(file_names):
            timestamp_list.append(_extract_timestamp(file_name))
            try:
                wse_map = in_root[file_name][:] 
                ys = self.roi_coords[:, 0]
                xs = self.roi_coords[:, 1]
                h, w = wse_map.shape
                ys_flipped = (h - 1) - ys
                valid_mask = (ys_flipped >= 0) & (ys_flipped < h) & (xs >= 0) & (xs < w)
                
                roi_wse_array[idx, :] = np.nan
                roi_wse_array[idx, valid_mask] = wse_map[ys_flipped[valid_mask], xs[valid_mask]]
            except Exception as e:
                roi_wse_array[idx, :] = np.nan

        datetimes = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30")
        
        try:
            out_backend = zarr.storage.ZipStore(out_zip_path, mode="w")
            out_root = zarr.open_group(store=out_backend, mode="w")
            out_root.create_array("timestamps", data=datetimes, chunks=datetimes.shape, overwrite=True)
            out_root.create_array(f"{self.roi_name}_wse", data=roi_wse_array, chunks=roi_wse_array.shape, overwrite=True)
            out_backend.close()
        except Exception as e:
            _log(f"Failed to save WSE ZipStore: {e}")
            
        in_backend.close()

    def _process_roadway_wse(self):
        """Calculates statistics for WSE and saves to a new CSV."""
        zip_store_path = os.path.join(self.event_path, "zarr", f"{self.roi_name}_wse.zip")

        if not os.path.exists(zip_store_path):
            return

        try:
            backend = zarr.storage.ZipStore(zip_store_path, mode="r")
            root = zarr.open_group(store=backend, mode="r")
            timestamps = root["timestamps"][:]
            roi_wse = root[f"{self.roi_name}_wse"][:]
            backend.close()
            
            datetimes = pd.to_datetime(timestamps, utc=True)
            
            # For WSE, 0.0 often means dry/no data depending on your datum. 
            # We replace 0.0 with NaN so we only average actual water elevations.
            roi_wse_masked = np.where(roi_wse == 0.0, np.nan, roi_wse)
            
            with np.errstate(all='ignore'):
                mean_wse = np.nanmean(roi_wse_masked, axis=1)
                median_wse = np.nanmedian(roi_wse_masked, axis=1)
                max_wse = np.nanmax(roi_wse_masked, axis=1)

            stats_df = pd.DataFrame({
                "Time": datetimes,
                "MeanWSE": mean_wse,
                "MedianWSE": median_wse,
                "MaxWSE": max_wse,
            }).sort_values(by="Time")

            output_path = os.path.join(self.event_path, f"{self.roi_name}_wse_time_series.csv")
            stats_df.to_csv(output_path, index=False)
            _log(f"WSE CSV saved to {output_path}")
            
        except Exception as e:
            _log(f"Error calculating WSE stats: {e}")