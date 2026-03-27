import os
import re
import pytz
from pathlib import Path
from svgpath2mpl import parse_path
import pandas as pd
from datetime import datetime
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from pyproj import Transformer
import rioxarray
import dask.array as da
from datetime import timedelta

def _log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def _extract_camera_name(filename):
    pattern = r"CAM_[A-Z]{2}_[0-9]{2}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None

def _extract_timestamp(filename):
    pattern = r"\d{14}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None

class DepthMapPlotter:
    def __init__(
        self, target_event_dir, min_x_extent, max_x_extent, min_y_extent, max_y_extent,
        resolution_m=0.05, bbox_crs="EPSG:32119", virtual_sensor_locations=None,
        plot_sensors=False, basemap_path=None,
    ):
        self.event_path = Path(target_event_dir)
        self.event_name = self.event_path.name
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.resolution_m = resolution_m
        self.bbox_crs = bbox_crs
        self.water_level_color = cmocean.cm.balance(0.2)
        self.virtual_sensor_loc = virtual_sensor_locations
        self.plot_sensors = plot_sensors
        self.basemap_path = basemap_path
        
        if self.plot_sensors:
            self.sensor_marker_path = self._create_sensor_marker()

    def preprocess_single_event(self):
        self._match_measurements_to_images()
        self._gen_virtual_sensor_depths()

    def process_single_flood_event(self, stats_to_plot=None):
        _log(f"\n=== Processing Flood Event: {self.event_name} ===")
        
        zip_path = self.event_path / "zarr" / "depth_maps.zip"
        if not zip_path.exists():
            _log(f"  ERROR: Zip file not found at '{zip_path}'")
            return

        try:
            store_backend = zarr.storage.ZipStore(zip_path, mode="r")
            root = zarr.open_group(store=store_backend, mode="r")
            contents = list(root.keys())
        except Exception as e:
            _log(f"  ERROR: Could not open ZipStore: {e}")
            return

        total_potential_files = len(contents)

        if stats_to_plot is not None:
            files_to_process = [name for name in contents if any(name.endswith(stat) for stat in stats_to_plot)]
        else:
            files_to_process = contents

        total_files = len(files_to_process)
        if total_files == 0:
            _log("  WARNING: No matching maps found to plot.")
            store_backend.close()
            return

        _log(f"  Found {total_files} maps to process.")

        for i, zarr_name in enumerate(files_to_process):
            try:
                output_png_filename = f"{zarr_name}.png"
                if "wse" in zarr_name:
                    plot_type = "wse"
                    plotting_folder = self.event_path / "plots" / "WSE_maps"
                elif "depth" in zarr_name:
                    plot_type = "depth"
                    plotting_folder = self.event_path / "plots" / "depth_maps"
                else:
                    continue 

                self._plot_georeferenced_map(
                    dataset=root[zarr_name],
                    output_filename=output_png_filename,
                    plot_type=plot_type,
                    output_folder=plotting_folder,
                )
            except Exception as e:
                _log(f"    -> ERROR processing {zarr_name}: {e}")

        store_backend.close()
        _log(f"=== Processing Complete: {self.event_name} ===")

    def plot_all_time_series(self):
        orig_images_path = self.event_path / "orig_images"
        ts_output_folder = self.event_path / "plots" / "time_series_using_depths"
        os.makedirs(ts_output_folder, exist_ok=True)

        if not orig_images_path.exists():
            _log("  WARNING: orig_images directory not found. Skipping time series.")
            return

        images = sorted(os.listdir(orig_images_path))
        for filename in images:
            try:
                self.plot_water_level_time_series(filename, ts_output_folder)
            except Exception as e:
                _log(f"  ERROR plotting time series for {filename}: {e}")

    def plot_water_level_time_series(self, file_name, plotting_folder):
        datetimes, max_depths, avg_depths, vs_depths = self._load_virtual_sensor_depths()
        datetimes = pd.to_datetime(datetimes)

        obs_to_img_matches = pd.read_csv(self.event_path / "wtr_lvl_obs_to_image_matches.csv")
        obs_to_img_matches["closest_utc_time"] = pd.to_datetime(obs_to_img_matches["closest_utc_time"])

        timestamp = _extract_timestamp(file_name)
        current_timestamp = pd.to_datetime(timestamp, utc=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            obs_to_img_matches["closest_utc_time"],
            obs_to_img_matches["water_level"],
            label="Observed Water Level",
            color=self.water_level_color,
        )

        if self.plot_sensors:
            num_sensors = len(self.virtual_sensor_loc)
            colors = plt.cm.viridis(np.linspace(0, 1, num_sensors))
            for i in range(num_sensors):
                ax.scatter(
                    datetimes, vs_depths[:, i], label=f"Sensor {i+1} Depth",
                    marker=self.sensor_marker_path, color=colors[i], s=15, zorder=5,
                )

        ax.axvline(x=current_timestamp, color="k", linestyle="--", zorder=1)
        padding = timedelta(hours=1)
        ax.set_xlim(datetimes.min() - padding, datetimes.max() + padding)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Water Surface Elevation (m NAVD88)")
        ax.set_xlabel("Date and Time (UTC)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

        plt.tight_layout()
        save_path = os.path.join(plotting_folder, f"{os.path.splitext(file_name)[0]}_time_series.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)

    def _gen_virtual_sensor_depths(self):
        zip_path = self.event_path / "zarr" / "depth_maps.zip"
        output_zarr_store = self.event_path / "zarr" / "virtual_sensor_wses.zip"

        if not zip_path.exists():
            return

        in_backend = zarr.storage.ZipStore(zip_path, mode="r")
        in_root = zarr.open_group(store=in_backend, mode="r")
        
        file_names = [f for f in in_root.keys() if f.endswith("depth_map_95_perc")]
        num_files = len(file_names)

        max_depth_array = np.empty(num_files, dtype=np.float32)
        avg_depth_array = np.empty(num_files, dtype=np.float32)
        vs_depth_array = np.empty((num_files, len(self.virtual_sensor_loc)), dtype=np.float32)
        timestamp_list = []

        for idx, file_name in enumerate(file_names):
            timestamp_list.append(_extract_timestamp(file_name))
            depth_map = in_root[file_name][:]

            max_depth_array[idx] = np.nanmax(depth_map)
            avg_depth_array[idx] = np.nanmean(depth_map)

            for i, (x, y) in enumerate(self.virtual_sensor_loc):
                vs_depth_array[idx, i] = depth_map[y, x]

        datetimes_np = np.array(pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30")

        out_backend = zarr.storage.ZipStore(output_zarr_store, mode="w")
        out_root = zarr.open_group(store=out_backend, mode="w")
        
        out_root.create_array("timestamps", data=datetimes_np, chunks=datetimes_np.shape)
        out_root.create_array("max_depths", data=max_depth_array, chunks=max_depth_array.shape)
        out_root.create_array("avg_depths", data=avg_depth_array, chunks=avg_depth_array.shape)
        out_root.create_array("vs_depths", data=vs_depth_array, chunks=vs_depth_array.shape)

        in_backend.close()
        out_backend.close()

    def _load_virtual_sensor_depths(self):
        zarr_store_path = self.event_path / "zarr" / "virtual_sensor_wses.zip"
        if not zarr_store_path.exists():
            raise FileNotFoundError(f"Zip store not found: {zarr_store_path}")

        backend = zarr.storage.ZipStore(zarr_store_path, mode="r")
        root = zarr.open_group(store=backend, mode="r")

        timestamps = root["timestamps"][:]
        max_depths = root["max_depths"][:]
        avg_depths = root["avg_depths"][:]
        vs_depths = root["vs_depths"][:]
        backend.close()

        datetimes = pd.to_datetime(timestamps, utc=True)
        return datetimes, max_depths, avg_depths, vs_depths

    def _match_measurements_to_images(self):
        csv_path = self.event_path / f"{self.event_name}.csv"
        if not csv_path.exists():
            return
            
        sunnyd_data = pd.read_csv(csv_path)
        sunnyd_data["time_UTC"] = pd.to_datetime(sunnyd_data["time_UTC"])

        orig_images_path = self.event_path / "orig_images"
        if not orig_images_path.exists():
            return
            
        match = []
        for filename in sorted(os.listdir(orig_images_path)):
            sensor_id = _extract_camera_name(filename)[4:]
            timestamp_str = _extract_timestamp(filename)
            timestamp = pytz.utc.localize(datetime.strptime(timestamp_str, "%Y%m%d%H%M%S"))

            filtered_df = sunnyd_data[sunnyd_data["sensor_ID"] == sensor_id]
            closest_row = filtered_df.iloc[(filtered_df["time_UTC"] - timestamp).abs().argsort()[:1]]

            if not closest_row.empty:
                match.append({
                    "image_filename": filename,
                    "closest_utc_time": closest_row["time_UTC"].values[0],
                    "water_level": closest_row["sensor_water_level"].values[0] * 0.3048,
                })

        pd.DataFrame(match).to_csv(self.event_path / "wtr_lvl_obs_to_image_matches.csv")

    def _plot_georeferenced_map(self, dataset, output_filename, plot_type, output_folder):
        geodata = self._load_and_prepare_geodata(dataset)
        src_data = geodata["mercator_array"]

        fig, ax = plt.subplots(figsize=(10, 10))
        minx, miny, maxx, maxy = src_data.rio.bounds()
        data_to_plot = src_data.to_numpy()
        style = self._get_plot_style(plot_type, src_data)

        im = ax.imshow(
            data_to_plot, extent=(minx, maxx, miny, maxy),
            cmap=style["cmap"], vmin=style["vmin"], vmax=style["vmax"],
            alpha=0.7, interpolation="none", zorder=10,
        )

        if self.plot_sensors:
            self._plot_sensor_locations(
                ax=ax, marker_path=self.sensor_marker_path,
                original_data_shape=geodata["shape"],
                data_affine_transform=geodata["transform"],
            )

        self._finalize_and_save_plot(fig, ax, im, geodata, style["cbar_label"], output_folder, output_filename)

    def _load_and_prepare_geodata(self, dataset):
        H, W = dataset.shape
        lazy_data = da.from_array(dataset)
        
        valid_pixel_count = da.sum(~da.isnan(lazy_data)).compute()
        spatial_extent = round((valid_pixel_count) * (self.resolution_m**2), 2)

        da_hmax = xr.DataArray(data=lazy_data.astype(float), dims=["y", "x"], name="flood_data")
        da_hmax_flipped = da_hmax.isel(y=slice(None, None, -1))

        data_affine_transform = Affine(self.resolution_m, 0.0, self.min_x_extent, 0.0, -self.resolution_m, self.max_y_extent)

        da_hmax_flipped = da_hmax_flipped.rio.write_crs(self.bbox_crs)
        da_hmax_flipped = da_hmax_flipped.rio.write_transform(data_affine_transform)
        da_hmax_flipped = da_hmax_flipped.rio.write_nodata(np.nan)

        da_hmax_mercator = da_hmax_flipped.rio.reproject(3857)

        return {
            "mercator_array": da_hmax_mercator, "shape": (H, W),
            "transform": data_affine_transform, "spatial_extent": spatial_extent,
        }

    @staticmethod
    def _get_plot_style(plot_type, data_array):
        if plot_type == "depth":
            return {"vmin": 0, "vmax": 0.25, "cmap": cmocean.cm.dense, "cbar_label": "Depth (m)"}
        elif plot_type == "wse":
            vmin = data_array.min(skipna=True).compute()
            vmax = data_array.max(skipna=True).compute()
            return {"vmin": vmin, "vmax": vmax, "cmap": "Blues", "cbar_label": "Water Surface Elevation (m)"}
        else:
            raise ValueError("Unknown plot_type")

    def _finalize_and_save_plot(self, fig, ax, im, geodata, cbar_label, output_folder, output_filename):
        minx, miny, maxx, maxy = geodata["mercator_array"].rio.bounds()
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # --- FIX: REPROJECT BASEMAP ON THE FLY ---
        if self.basemap_path and os.path.exists(self.basemap_path):
            try:
                # Open basemap and ensure it matches EPSG:3857 so it aligns with the data
                basemap = rioxarray.open_rasterio(self.basemap_path)
                basemap = basemap.rio.reproject("EPSG:3857")
                
                # Clip to data bounds to prevent massive memory usage when plotting
                basemap = basemap.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
                
                # Plot basemap behind data (zorder=1)
                basemap.plot.imshow(ax=ax, add_colorbar=False, add_labels=False, zorder=1)
            except Exception as e:
                _log(f"    -> WARNING: Failed to load/reproject local basemap: {e}")

        spatial_extent = geodata["spatial_extent"]
        ax.text(
            0.05, 0.95, f"Spatial Extent ($m^2$): {spatial_extent}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"), zorder=30,
        )

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
        cbar.set_label(cbar_label)

        png_path = Path(output_folder) / output_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)

        ax.set_title("")
        ax.set_axis_off()
        plt.tight_layout()

        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        _log(f"    -> Plot saved: {png_path}")

    @staticmethod
    def _create_sensor_marker():
        svg_path_str = (
            "M0 0 C0.825 -0.00128906 1.65 -0.00257812 2.5 -0.00390625 C8.08645564 1.2617751 11.16527062 5.79806233 14.4375 10.1875 C15.31557349 11.35479356 16.19453012 12.52142315 17.07421875 13.6875 C17.77877197 14.62529297 17.77877197 14.62529297 18.49755859 15.58203125 C20.63003679 18.39299373 22.81066988 21.16524823 24.98876953 23.94091797 C32.87088989 33.99515578 40.48478691 44.23203019 48 54.5625 C48.55574707 55.32594727 49.11149414 56.08939453 49.68408203 56.87597656 C56.98821361 66.92210529 64.20967289 77.02547875 71.33984375 87.19580078 C72.50493339 88.85673886 73.67175131 90.51646531 74.83984375 92.17529297 C84.23635549 105.52561863 93.36395435 119.01987438 102.19287109 132.75244141 C103.95234767 135.48840134 105.72007622 138.21889483 107.48828125 140.94921875 C150.05895804 206.96147365 200.12438567 285.23012917 182.93359375 366.8671875 C175.39631755 399.94249141 159.3412289 429.9889702 136 454.5625 C135.51080078 455.08424805 135.02160156 455.60599609 134.51757812 456.14355469 C120.27329692 471.2508615 104.32723065 482.85206456 86 492.5625 C85.29407715 492.93745605 84.5881543 493.31241211 83.86083984 493.69873047 C42.55213774 515.24558359 -9.65633974 519.33070778 -54.06469727 505.54663086 C-56.04851041 504.90027389 -58.02462909 504.23416574 -60 503.5625 C-60.63744141 503.35431641 -61.27488281 503.14613281 -61.93164062 502.93164062 C-87.35443777 494.5604055 -109.59679551 479.78392291 -129 461.5625 C-129.53141602 461.06508301 -130.06283203 460.56766602 -130.61035156 460.05517578 C-165.73760723 426.90395362 -185.00181714 380.48876006 -187.28515625 332.51953125 C-188.96966452 274.30244773 -152.74099454 216.2166584 -123 168.5625 C-122.47180664 167.71042969 -121.94361328 166.85835937 -121.39941406 165.98046875 C-102.42600323 135.42588507 -81.73079704 105.87110301 -60.60449219 76.77587891 C-58.99495681 74.55554297 -57.39107213 72.33123958 -55.7890625 70.10546875 C-50.23587409 62.3920993 -44.64681757 54.70763842 -39 47.0625 C-24.93834436 28.1566023 -24.93834436 28.1566023 -12 8.48046875 C-6.07678149 -0.16141451 -6.07678149 -0.16141451 0 0 Z"
        )
        marker_path = parse_path(svg_path_str)
        marker_path.vertices -= marker_path.vertices.mean(axis=0)
        marker_path.vertices[:, 1] *= -1
        return marker_path

    def _plot_sensor_locations(self, ax, marker_path, original_data_shape, data_affine_transform):
        if not self.plot_sensors or self.virtual_sensor_loc is None:
            return 
        try:
            H, W = original_data_shape
            transformer = Transformer.from_crs(self.bbox_crs, "EPSG:3857", always_xy=True)
            mercator_x, mercator_y = [], []

            for y_orig, x_orig in self.virtual_sensor_loc:
                y_flipped = (H - 1) - y_orig
                x_flipped = x_orig 
                if not (0 <= y_flipped < H and 0 <= x_flipped < W):
                    continue
                x_crs, y_crs = data_affine_transform * (x_flipped + 0.5, y_flipped + 0.5)
                mx, my = transformer.transform(x_crs, y_crs)
                mercator_x.append(mx)
                mercator_y.append(my)

            if not mercator_x:
                return

            colors = plt.cm.viridis(np.linspace(0, 1, len(mercator_x)))
            ax.scatter(
                mercator_x, mercator_y, marker=marker_path, c=colors,
                edgecolors="white", s=300, linewidths=1, label="Virtual Sensors", zorder=20,
            )
        except Exception as e:
            _log(f"    -> WARNING: Failed to plot sensor locations: {e}")