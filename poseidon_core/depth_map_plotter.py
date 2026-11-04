"""Plot georeferenced depth/WSE maps and virtual sensor time series.

For one flood event folder this module:

1. Matches each image to the nearest SDFP water level observation
   (``wtr_lvl_obs_to_image_matches.csv``).
2. Samples the 95th-percentile depth and WSE maps at virtual sensor
   pixels (``zarr/virtual_sensor_data.zip``).
3. Plots selected maps over an optional basemap in Web Mercator
   (``plots/depth_maps`` and ``plots/WSE_maps``).
4. Plots observed versus virtual sensor time series for every image
   (``plots/time_series_using_depths``).
"""

import logging
import os
from datetime import timedelta
from pathlib import Path

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from affine import Affine
from pyproj import Transformer
from svgpath2mpl import parse_path

from poseidon_utils.constants import FEET_TO_METERS
from poseidon_utils.naming import (
    extract_camera_name,
    parse_timestamp,
    sensor_id_from_camera,
)
from poseidon_utils.zarr_io import (
    array_names,
    open_zip_group,
    read_zip_arrays,
    write_array,
    write_zip_group,
)

logger = logging.getLogger(__name__)

#: Map statistic sampled at the virtual sensors.
VIRTUAL_SENSOR_STAT = "95_perc"

#: Web Mercator, the CRS used for all map figures.
PLOT_CRS = "EPSG:3857"

#: SVG path of the virtual sensor marker glyph.
MARKER_FILE = Path(__file__).with_name("assets") / "sensor_marker.svgpath"


class DepthMapPlotter:
    """
    Plot depth products for one flood event folder.

    Parameters
    ----------
    target_event_dir : str or os.PathLike
        Event folder containing ``zarr/depth_maps.zip``, ``orig_images``
        and ``<event name>.csv`` with SDFP observations.
    min_x_extent, max_x_extent, min_y_extent, max_y_extent : float
        Grid bounds in `bbox_crs` (meters).
    resolution_m : float, optional
        Grid spacing in meters. Default is 0.05.
    bbox_crs : str, optional
        CRS of the grid. Default is ``"EPSG:32119"``.
    virtual_sensor_locations : array_like, optional
        ``(N, 2)`` ``(col, row)`` grid pixels of virtual sensors, row 0
        at the southern edge. Default is None (no virtual sensors).
    plot_sensors : bool, optional
        Draw virtual sensors on maps and time series. Default False.
    basemap_path : str or os.PathLike, optional
        GeoTIFF drawn beneath the maps. Default is None.

    Attributes
    ----------
    event_path : pathlib.Path
        Event folder.
    event_name : str
        Event folder name.
    """

    def __init__(self, target_event_dir, min_x_extent, max_x_extent,
                 min_y_extent, max_y_extent, resolution_m=0.05,
                 bbox_crs="EPSG:32119", virtual_sensor_locations=None,
                 plot_sensors=False, basemap_path=None):
        """
        Store the event location, grid geometry, and plot options.

        Parameters
        ----------
        target_event_dir : str or os.PathLike
            Event folder.
        min_x_extent, max_x_extent, min_y_extent, max_y_extent : float
            Grid bounds in `bbox_crs`.
        resolution_m : float, optional
            Grid spacing in meters. Default is 0.05.
        bbox_crs : str, optional
            CRS of the grid. Default is ``"EPSG:32119"``.
        virtual_sensor_locations : array_like, optional
            ``(N, 2)`` virtual sensor grid pixels. Default is None.
        plot_sensors : bool, optional
            Draw virtual sensors. Default is False.
        basemap_path : str or os.PathLike, optional
            Basemap GeoTIFF. Default is None.
        """
        self.event_path = Path(target_event_dir)
        self.event_name = self.event_path.name
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.resolution_m = resolution_m
        self.bbox_crs = bbox_crs
        self.virtual_sensor_loc = (
            None if virtual_sensor_locations is None
            else np.asarray(virtual_sensor_locations, dtype=int))
        has_sensors = self.virtual_sensor_loc is not None
        self.plot_sensors = plot_sensors and has_sensors
        self.basemap_path = basemap_path
        self.water_level_color = cmocean.cm.balance(0.2)
        self.sensor_marker_path = (self._create_sensor_marker()
                                   if self.plot_sensors else None)
        self._basemap_cache = None
        self._basemap_loaded = False

    @property
    def _depth_zip(self):
        """
        Path of the event's depth map store.

        Returns
        -------
        pathlib.Path
            ``<event>/zarr/depth_maps.zip``.
        """
        return self.event_path / "zarr" / "depth_maps.zip"

    @property
    def _sensor_zip(self):
        """
        Path of the event's virtual sensor store.

        Returns
        -------
        pathlib.Path
            ``<event>/zarr/virtual_sensor_data.zip``.
        """
        return self.event_path / "zarr" / "virtual_sensor_data.zip"

    @property
    def _matches_csv(self):
        """
        Path of the image-to-observation match table.

        Returns
        -------
        pathlib.Path
            ``<event>/wtr_lvl_obs_to_image_matches.csv``.
        """
        return self.event_path / "wtr_lvl_obs_to_image_matches.csv"

    def preprocess_single_event(self):
        """Build the observation matches and virtual sensor time series."""
        self._match_measurements_to_images()
        self._gen_virtual_sensor_depths()

    def process_single_flood_event(self, stats_to_plot=None):
        """
        Plot the event's depth and WSE maps.

        Parameters
        ----------
        stats_to_plot : sequence of str, optional
            Statistics to plot (e.g. ``["95_perc"]``). Default plots
            every map in the store.

        Returns
        -------
        int
            Number of maps plotted.
        """
        if not self._depth_zip.exists():
            logger.error("Depth maps not found: %s", self._depth_zip)
            return 0

        plotted = 0
        with open_zip_group(self._depth_zip) as group:
            suffixes = None if stats_to_plot is None else tuple(stats_to_plot)
            names = array_names(group, suffix=suffixes)
            logger.info("Plotting %d maps for %s", len(names),
                        self.event_name)
            for name in names:
                if "wse" in name:
                    plot_type, folder = "wse", "WSE_maps"
                elif "depth" in name:
                    plot_type, folder = "depth", "depth_maps"
                else:
                    continue
                try:
                    self._plot_georeferenced_map(
                        group[name][:], f"{name}.png", plot_type,
                        self.event_path / "plots" / folder)
                    plotted += 1
                except Exception:  # noqa: BLE001 - keep plotting the rest
                    logger.exception("Failed to plot %s", name)
        return plotted

    def plot_all_time_series(self):
        """
        Plot observed and virtual sensor time series for every image.

        Returns
        -------
        int
            Number of images plotted.
        """
        images_dir = self.event_path / "orig_images"
        if not images_dir.is_dir():
            logger.warning("No orig_images folder in %s; skipping time "
                           "series.", self.event_path)
            return 0

        output_dir = self.event_path / "plots" / "time_series_using_depths"
        output_dir.mkdir(parents=True, exist_ok=True)
        series = self._load_time_series_inputs()

        plotted = 0
        for file_name in sorted(os.listdir(images_dir)):
            try:
                self._plot_time_series_pair(file_name, output_dir, *series)
                plotted += 1
            except Exception:  # noqa: BLE001 - keep plotting the rest
                logger.exception("Failed to plot time series for %s",
                                 file_name)
        return plotted

    def plot_water_level_time_series(self, file_name, plotting_folder):
        """
        Plot the WSE and depth time series marking one image's time.

        Parameters
        ----------
        file_name : str
            Image filename whose timestamp is marked on the plot.
        plotting_folder : str or os.PathLike
            Output folder.
        """
        self._plot_time_series_pair(file_name, plotting_folder,
                                    *self._load_time_series_inputs())

    def _load_time_series_inputs(self):
        """
        Load the virtual sensor series and the observation matches.

        Returns
        -------
        datetimes : pandas.DatetimeIndex
            Virtual sensor sample times, or observation times when no
            virtual sensor data exist.
        vs_wse, vs_depth : numpy.ndarray or None
            ``(T, N)`` virtual sensor WSE and depth, or None.
        observations : pandas.DataFrame
            Image-matched observations with parsed times.
        """
        observations = pd.read_csv(self._matches_csv)
        observations["closest_utc_time"] = pd.to_datetime(
            observations["closest_utc_time"], utc=True)
        if self._sensor_zip.exists():
            datetimes, vs_wse, vs_depth = self._load_virtual_sensor_depths()
        else:
            datetimes = pd.DatetimeIndex(observations["closest_utc_time"])
            vs_wse = vs_depth = None
        return datetimes, vs_wse, vs_depth, observations

    def _plot_time_series_pair(self, file_name, plotting_folder, datetimes,
                               vs_wse, vs_depth, observations):
        """
        Save the WSE and depth time series figures for one image.

        Parameters
        ----------
        file_name : str
            Image filename whose timestamp is marked.
        plotting_folder : str or os.PathLike
            Output folder.
        datetimes : pandas.DatetimeIndex
            Virtual sensor sample times.
        vs_wse, vs_depth : numpy.ndarray
            ``(T, N)`` virtual sensor WSE and depth.
        observations : pandas.DataFrame
            Image-matched observations.

        Raises
        ------
        ValueError
            If `file_name` has no parsable timestamp.
        """
        current = parse_timestamp(file_name)
        if current is None:
            raise ValueError(f"No timestamp in filename '{file_name}'")

        for obs_col, vs_data, y_label, suffix in (
                ("obs_wse_meters", vs_wse,
                 "Water Surface Elevation (m NAVD88)", "wse"),
                ("obs_depth_meters", vs_depth,
                 "Depth above Roadway (m)", "depth")):
            self._create_ts_plot(datetimes, observations, current, file_name,
                                 plotting_folder, obs_col, vs_data, y_label,
                                 suffix)

    def _create_ts_plot(self, datetimes, obs_df, current_timestamp,
                        file_name, plotting_folder, obs_col, vs_data,
                        y_label, suffix):
        """
        Save one observed-vs-virtual-sensor time series figure.

        Parameters
        ----------
        datetimes : pandas.DatetimeIndex
            Virtual sensor sample times.
        obs_df : pandas.DataFrame
            Observations with ``closest_utc_time`` and `obs_col`.
        current_timestamp : datetime.datetime
            Time marked with a vertical line.
        file_name : str
            Image filename; its stem names the output file.
        plotting_folder : str or os.PathLike
            Output folder.
        obs_col : str
            Observation column to plot.
        vs_data : numpy.ndarray or None
            ``(T, N)`` virtual sensor values, or None to omit them.
        y_label : str
            Y-axis label.
        suffix : str
            Quantity name used in the legend and filename.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(obs_df["closest_utc_time"], obs_df[obs_col],
                label=f"Observed {suffix.upper()}",
                color=self.water_level_color)

        if self.plot_sensors and vs_data is not None:
            n_sensors = len(self.virtual_sensor_loc)
            colors = plt.cm.viridis(np.linspace(0, 1, n_sensors))
            for i in range(n_sensors):
                ax.scatter(datetimes, vs_data[:, i],
                           label=f"Sensor {i + 1} {suffix.title()}",
                           marker=self.sensor_marker_path, color=colors[i],
                           s=15, zorder=5)

        ax.axvline(x=current_timestamp, color="k", linestyle="--", zorder=1)
        padding = timedelta(hours=1)
        ax.set_xlim(datetimes.min() - padding, datetimes.max() + padding)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Date and Time (UTC)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

        fig.tight_layout()
        stem = os.path.splitext(file_name)[0]
        fig.savefig(os.path.join(plotting_folder,
                                 f"{stem}_{suffix}_time_series.png"),
                    bbox_inches="tight", pad_inches=0.1, dpi=300)
        plt.close(fig)

    def _gen_virtual_sensor_depths(self):
        """
        Sample the depth and WSE maps at each virtual sensor pixel.

        Writes ``zarr/virtual_sensor_data.zip`` with ``timestamps``,
        ``vs_wse`` and ``vs_depth`` arrays.

        Returns
        -------
        bool
            True if the store was written.
        """
        if self.virtual_sensor_loc is None:
            logger.info("No virtual sensors configured; skipping sampling.")
            return False
        if not self._depth_zip.exists():
            logger.warning("Depth maps not found: %s", self._depth_zip)
            return False

        cols, rows = self.virtual_sensor_loc.T
        with open_zip_group(self._depth_zip) as group:
            wse_names = array_names(group,
                                    suffix=f"wse_map_{VIRTUAL_SENSOR_STAT}")
            depth_names = array_names(
                group, suffix=f"depth_map_{VIRTUAL_SENSOR_STAT}")
            if not wse_names or len(wse_names) != len(depth_names):
                logger.warning("Expected matching WSE/depth '%s' maps in %s; "
                               "found %d and %d.", VIRTUAL_SENSOR_STAT,
                               self._depth_zip, len(wse_names),
                               len(depth_names))
                return False
            vs_wse = np.array([group[name][:][rows, cols]
                               for name in wse_names], dtype=np.float32)
            vs_depth = np.array([group[name][:][rows, cols]
                                 for name in depth_names], dtype=np.float32)

        times = [parse_timestamp(name) for name in wse_names]
        stamps = np.array(pd.DatetimeIndex(times).astype(str), dtype="U30")
        with write_zip_group(self._sensor_zip) as out:
            write_array(out, "timestamps", stamps)
            write_array(out, "vs_wse", vs_wse)
            write_array(out, "vs_depth", vs_depth)
        return True

    def _load_virtual_sensor_depths(self):
        """
        Load the virtual sensor store written by preprocessing.

        Returns
        -------
        datetimes : pandas.DatetimeIndex
            Sample times (UTC).
        vs_wse, vs_depth : numpy.ndarray
            ``(T, N)`` virtual sensor WSE and depth.

        Raises
        ------
        FileNotFoundError
            If the store does not exist.
        """
        arrays = read_zip_arrays(self._sensor_zip)
        datetimes = pd.to_datetime(arrays["timestamps"], utc=True)
        return datetimes, arrays["vs_wse"], arrays["vs_depth"]

    def _match_measurements_to_images(self):
        """
        Pair each image with the nearest SDFP observation in time.

        Reads ``<event name>.csv`` (``sensor_ID``, ``time_UTC``,
        ``sensor_water_level``, ``road_water_level`` in feet) and writes
        ``wtr_lvl_obs_to_image_matches.csv`` with levels in meters.

        Returns
        -------
        pandas.DataFrame or None
            The match table, or None if inputs are missing.
        """
        csv_path = self.event_path / f"{self.event_name}.csv"
        images_dir = self.event_path / "orig_images"
        if not csv_path.exists() or not images_dir.is_dir():
            logger.warning("Observations or images missing in %s; "
                           "skipping matching.", self.event_path)
            return None

        observations = pd.read_csv(csv_path)
        observations["time_UTC"] = pd.to_datetime(observations["time_UTC"],
                                                  utc=True)

        matches = []
        for file_name in sorted(os.listdir(images_dir)):
            camera = extract_camera_name(file_name)
            image_time = parse_timestamp(file_name)
            if camera is None or image_time is None:
                logger.debug("Skipping unparsable image name %s", file_name)
                continue
            sensor_obs = observations[observations["sensor_ID"]
                                      == sensor_id_from_camera(camera)]
            if sensor_obs.empty:
                continue
            nearest = sensor_obs.loc[
                (sensor_obs["time_UTC"] - image_time).abs().idxmin()]
            matches.append({
                "image_filename": file_name,
                "closest_utc_time": nearest["time_UTC"],
                "obs_wse_meters": (nearest["sensor_water_level"]
                                   * FEET_TO_METERS),
                "obs_depth_meters": (nearest["road_water_level"]
                                     * FEET_TO_METERS),
            })

        table = pd.DataFrame(matches)
        table.to_csv(self._matches_csv, index=False)
        return table

    def _plot_georeferenced_map(self, data, output_filename, plot_type,
                                output_folder):
        """
        Plot one map in Web Mercator over the optional basemap.

        Parameters
        ----------
        data : numpy.ndarray
            2-D map in grid orientation (row 0 south).
        output_filename : str
            PNG filename.
        plot_type : {'depth', 'wse'}
            Selects the color scale.
        output_folder : str or os.PathLike
            Output folder.
        """
        geodata = self._load_and_prepare_geodata(data)
        mercator = geodata["mercator_array"]
        minx, miny, maxx, maxy = mercator.rio.bounds()
        style = self._get_plot_style(plot_type, mercator)

        fig, ax = plt.subplots(figsize=(10, 10))
        image = ax.imshow(mercator.to_numpy(), extent=(minx, maxx, miny, maxy),
                          cmap=style["cmap"], vmin=style["vmin"],
                          vmax=style["vmax"], alpha=0.7, interpolation="none",
                          zorder=10)
        if self.plot_sensors:
            self._plot_sensor_locations(ax, self.sensor_marker_path,
                                        geodata["shape"],
                                        geodata["transform"])
        self._finalize_and_save_plot(fig, ax, image, geodata,
                                     style["cbar_label"], output_folder,
                                     output_filename)

    def _load_and_prepare_geodata(self, data):
        """
        Georeference a grid map and reproject it to Web Mercator.

        Parameters
        ----------
        data : numpy.ndarray
            2-D map in grid orientation (row 0 south).

        Returns
        -------
        dict
            ``mercator_array`` (xarray.DataArray), ``shape`` (grid
            shape), ``transform`` (north-up affine in `bbox_crs`), and
            ``spatial_extent`` (wetted area in square meters).
        """
        n_valid = int(np.count_nonzero(~np.isnan(data)))
        spatial_extent = round(n_valid * self.resolution_m**2, 2)

        # Flip to north-up so the affine origin is the top-left corner
        north_up = xr.DataArray(np.asarray(data, dtype=float)[::-1],
                                dims=["y", "x"], name="flood_data")
        transform = Affine(self.resolution_m, 0.0, self.min_x_extent,
                           0.0, -self.resolution_m, self.max_y_extent)
        north_up = (north_up.rio.write_crs(self.bbox_crs)
                    .rio.write_transform(transform)
                    .rio.write_nodata(np.nan))
        return {
            "mercator_array": north_up.rio.reproject(PLOT_CRS),
            "shape": data.shape,
            "transform": transform,
            "spatial_extent": spatial_extent,
        }

    @staticmethod
    def _get_plot_style(plot_type, data_array):
        """
        Choose the colormap and limits for a map type.

        Parameters
        ----------
        plot_type : {'depth', 'wse'}
            Map type.
        data_array : xarray.DataArray
            Map data, used for WSE color limits.

        Returns
        -------
        dict
            ``vmin``, ``vmax``, ``cmap``, and ``cbar_label``.

        Raises
        ------
        ValueError
            If `plot_type` is unknown.
        """
        if plot_type == "depth":
            return {"vmin": 0, "vmax": 0.25, "cmap": cmocean.cm.dense,
                    "cbar_label": "Depth (m)"}
        if plot_type == "wse":
            return {"vmin": float(data_array.min(skipna=True)),
                    "vmax": float(data_array.max(skipna=True)),
                    "cmap": "Blues",
                    "cbar_label": "Water Surface Elevation (m)"}
        raise ValueError(f"Unknown plot_type '{plot_type}'")

    def _load_basemap(self):
        """
        Load and reproject the basemap once, caching the result.

        Returns
        -------
        xarray.DataArray or None
            Basemap in Web Mercator, or None if unavailable.
        """
        if not self._basemap_loaded:
            self._basemap_loaded = True
            if self.basemap_path and os.path.exists(self.basemap_path):
                basemap = rioxarray.open_rasterio(self.basemap_path)
                self._basemap_cache = basemap.rio.reproject(PLOT_CRS)
            else:
                logger.info("Basemap not found or not provided: %s",
                            self.basemap_path)
        return self._basemap_cache

    def _draw_basemap(self, ax, bounds):
        """
        Draw the basemap clipped to the map bounds.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.
        bounds : tuple of float
            ``(minx, miny, maxx, maxy)`` in Web Mercator.
        """
        basemap = self._load_basemap()
        if basemap is None:
            return
        minx, miny, maxx, maxy = bounds
        try:
            basemap = basemap.rio.clip_box(minx=minx, miny=miny, maxx=maxx,
                                           maxy=maxy)
        except Exception as err:  # noqa: BLE001 - fall back to full extent
            logger.info("Could not clip basemap; drawing full extent (%s)",
                        err)

        left, bottom, right, top = basemap.rio.bounds()
        pixels = basemap.to_numpy()
        if pixels.ndim == 3:
            pixels = np.moveaxis(pixels, 0, -1)  # bands last for imshow
            if pixels.dtype != np.uint8 and np.max(pixels) > 1.0:
                pixels = (pixels / np.max(pixels)).astype(float)
        ax.imshow(pixels, extent=(left, right, bottom, top), zorder=1,
                  origin="upper", interpolation="none")

    def _finalize_and_save_plot(self, fig, ax, image, geodata, cbar_label,
                                output_folder, output_filename):
        """
        Add the basemap, annotations, and colorbar, then save the figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        ax : matplotlib.axes.Axes
            Map axes.
        image : matplotlib.image.AxesImage
            Map layer, used for the colorbar.
        geodata : dict
            Output of :meth:`_load_and_prepare_geodata`.
        cbar_label : str
            Colorbar label.
        output_folder : str or os.PathLike
            Output folder.
        output_filename : str
            PNG filename.
        """
        bounds = geodata["mercator_array"].rio.bounds()
        try:
            self._draw_basemap(ax, bounds)
        except Exception as err:  # noqa: BLE001 - a basemap is optional
            logger.warning("Failed to draw basemap: %s", err)

        minx, miny, maxx, maxy = bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.text(0.05, 0.95,
                f"Spatial Extent ($m^2$): {geodata['spatial_extent']}",
                transform=ax.transAxes, fontsize=12, verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 0.8,
                      "edgecolor": "black"}, zorder=30)
        fig.colorbar(image, ax=ax, shrink=0.6, aspect=30).set_label(
            cbar_label)

        png_path = Path(output_folder) / output_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)
        ax.set_title("")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        logger.info("Saved %s", png_path)

    @staticmethod
    def _create_sensor_marker():
        """
        Build the centered, upright virtual sensor marker glyph.

        Returns
        -------
        matplotlib.path.Path
            Marker path usable as a scatter ``marker``.
        """
        marker = parse_path(MARKER_FILE.read_text().strip())
        marker.vertices -= marker.vertices.mean(axis=0)
        marker.vertices[:, 1] *= -1  # SVG y points down
        return marker

    def _plot_sensor_locations(self, ax, marker_path, original_data_shape,
                               data_affine_transform):
        """
        Draw virtual sensors at their map locations.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Map axes.
        marker_path : matplotlib.path.Path
            Marker glyph.
        original_data_shape : tuple of int
            ``(rows, cols)`` of the grid map.
        data_affine_transform : affine.Affine
            North-up transform of the grid in `bbox_crs`.
        """
        height, width = original_data_shape
        transformer = Transformer.from_crs(self.bbox_crs, PLOT_CRS,
                                           always_xy=True)
        points = []
        for col, row in self.virtual_sensor_loc:
            row_north_up = (height - 1) - row  # grid rows run south-up
            if not (0 <= row_north_up < height and 0 <= col < width):
                continue
            # Sample at the pixel center
            x, y = data_affine_transform @ (col + 0.5, row_north_up + 0.5)
            points.append(transformer.transform(x, y))

        if not points:
            return
        xs, ys = zip(*points)
        ax.scatter(xs, ys, marker=marker_path,
                   c=plt.cm.viridis(np.linspace(0, 1, len(points))),
                   edgecolors="white", s=300, linewidths=1,
                   label="Virtual Sensors", zorder=20)
