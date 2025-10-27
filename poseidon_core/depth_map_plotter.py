import os
from pathlib import Path

# Third-party libraries
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from mpi4py import MPI
from tqdm import tqdm

# This import is necessary for xarray .rio extension methods
import rioxarray
import contextily as ctx


class DepthMapPlotter:
    """
    A class to process and plot flood depth and Water Surface Elevation (WSE)
    maps from Zarr datasets, with support for parallel processing via MPI.
    """

    def __init__(
        self,
        main_dir,
        min_x_extent,
        max_x_extent,
        min_y_extent,
        max_y_extent,
        bbox_crs="EPSG:32119",
        virtual_sensor_locations=None,
        plot_sensors=False,
    ):
        """
        Initializes the plotter with spatial configuration.

        Parameters
        ----------
        main_dir : str or pathlib.Path
            The main directory where flood event data is stored.
        min_x_extent : float
            Minimum x-coordinate (easting) of the spatial extent.
        max_x_extent : float
            Maximum x-coordinate (easting) of the spatial extent.
        min_y_extent : float
            Minimum y-coordinate (northing) of the spatial extent.
        max_y_extent : float
            Maximum y-coordinate (northing) of the spatial extent.
        bbox_crs : str, optional
            The Coordinate Reference System (CRS) string for the input
            bounding box, by default "EPSG:32119".
        virtual_sensor_locations : object, optional
            A mapping or DataFrame-like object containing coordinates for
            virtual sensors, by default None.
        plot_sensors : bool, optional
            A flag to control whether sensor locations should be plotted,
            by default False. (Note: plotting logic is not yet implemented).
        """
        self.main_dir = Path(main_dir)

        # Store spatial extent and CRS
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.bbox_crs = bbox_crs

        # Sensor attributes
        self.virtual_sensor_loc = virtual_sensor_locations
        self.plot_sensors = plot_sensors
        self.sensor_1_color = cmocean.cm.phase(0.1)
        self.sensor_2_color = cmocean.cm.phase(0.3)
        self.sensor_3_color = cmocean.cm.phase(0.5)

    def process_flood_events_HPC(self, stats_to_plot=None):
        """
        Processes and generates plots for flood events in parallel using MPI.

        Parameters
        ----------
        stats_to_plot : list of str, optional
            A list of statistic suffixes to plot (e.g., ['mean', '95_perc']).
            If None, all found maps are plotted.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            flood_event_folders = self.list_flood_event_folders()
            self.preprocess_flood_events()
        else:
            flood_event_folders = None

        flood_event_folders = comm.bcast(flood_event_folders, root=0)

        if not flood_event_folders:
            if rank == 0:
                print("No flood event folders found.")
            return

        n_folders = len(flood_event_folders)
        chunk_size = n_folders // size
        start_index = rank * chunk_size
        end_index = start_index + chunk_size if rank != size - 1 else n_folders

        for flood_event in tqdm(
            flood_event_folders[start_index:end_index],
            desc=f"Rank {rank} plotting events",
            unit="event",
        ):
            # Pass the filter list down
            self.process_single_flood_event(
                flood_event, stats_to_plot=stats_to_plot
            )

    def process_single_flood_event(self, flood_event, stats_to_plot=None):
        """
        Processes a single flood event folder, plotting specified Zarr maps.

        Parameters
        ----------
        flood_event : str
            The name of the flood event directory.
        stats_to_plot : list of str, optional
            A list of statistic suffixes to plot (e.g., ['mean', '95_perc']).
            If None, all found maps are plotted.
        """
        flood_event_path = self.main_dir / flood_event
        depth_maps_zarr_dir = flood_event_path / "zarr" / "depth_maps"

        if not depth_maps_zarr_dir.is_dir():
            print(
                f"ERROR: Directory not found at '{depth_maps_zarr_dir}'",
                flush=True,
            )
            return

        try:
            contents = sorted(os.listdir(depth_maps_zarr_dir))
        except OSError as e:
            print(
                f"ERROR: Could not list contents of '{depth_maps_zarr_dir}': {e}",
                flush=True,
            )
            return

        for zarr_name in contents:
            full_zarr_path = depth_maps_zarr_dir / zarr_name

            if not full_zarr_path.is_dir():
                continue

            # --- NEW: Filtering Logic ---
            if stats_to_plot is not None:
                # Check if the zarr_name ends with any of the specified stats
                if not any(zarr_name.endswith(stat) for stat in stats_to_plot):
                    continue  # Skip this file if it doesn't match the filter

            try:
                output_png_filename = f"{zarr_name}.png"
                plot_type = None
                plotting_folder = None

                if "wse" in zarr_name:
                    print(f" -> Processing WSE file: {zarr_name}", flush=True)
                    plot_type = "wse"
                    plotting_folder = flood_event_path / "plots" / "WSE_maps"

                elif "depth" in zarr_name:
                    print(f" -> Processing DEPTH file: {zarr_name}", flush=True)
                    plot_type = "depth"
                    plotting_folder = flood_event_path / "plots" / "depth_maps"

                if plot_type and plotting_folder:
                    # --- MODIFIED: Removed max_x and min_y from call ---
                    self._plot_georeferenced_map(
                        depth_array_path=full_zarr_path,
                        min_x=self.min_x_extent,
                        max_y=self.max_y_extent,
                        output_filename=output_png_filename,
                        bbox_crs=self.bbox_crs,
                        output_folder=plotting_folder,
                        plot_type=plot_type,
                    )

            except Exception as e:
                print(
                    f"   -> An ERROR occurred while processing {zarr_name}: {e}",
                    flush=True,
                )

    def _plot_georeferenced_map(
        self,
        depth_array_path,
        min_x,
        max_y,
        output_filename,
        plot_type,
        resolution_m=0.05,
        bbox_crs="EPSG:32119",
        output_folder="figures",
    ):
        """
        Generates and saves a georeferenced plot for a flood map (WSE or depth).

        Parameters
        ----------
        depth_array_path : str or pathlib.Path
            Path to the input Zarr store.
        min_x : float
            Minimum x-coordinate (easting) of the spatial extent (top-left).
        max_y : float
            Maximum y-coordinate (northing) of the spatial extent (top-left).
        output_filename : str
            The base name for the output PNG file.
        plot_type : str
            Type of plot to generate, either 'depth' or 'wse'.
        resolution_m : float, optional
            The resolution (pixel size) in meters, by default 0.05.
        bbox_crs : str, optional
            The CRS string for the input bounding box, by default "EPSG:32119".
        output_folder : str or pathlib.Path, optional
            The directory where the final PNG plot will be saved.
        """
        array_store = zarr.open(str(depth_array_path), mode="r")
        depth_array = array_store[:]

        print(
            f"    -> Number of valid (non-NaN) data points: {np.count_nonzero(~np.isnan(depth_array))}"
        )

        data_for_xarray = np.flipud(depth_array).astype(float)

        # --- MODIFIED: Affine transform uses min_x and max_y (top-left) ---
        transform = Affine(resolution_m, 0.0, min_x, 0.0, -resolution_m, max_y)
        da_hmax = xr.DataArray(
            data=data_for_xarray, dims=["y", "x"], name="flood_data"
        )
        da_hmax.rio.write_crs(bbox_crs, inplace=True)
        da_hmax.rio.write_transform(transform, inplace=True)
        da_hmax.rio.write_nodata(np.nan, inplace=True)

        # Reproject the full-resolution array
        da_hmax_mercator = da_hmax.rio.reproject(3857)

        fig, ax = plt.subplots(figsize=(10, 10))
        minx, miny, maxx, maxy = da_hmax_mercator.rio.bounds()
        data_to_plot = da_hmax_mercator.to_numpy()

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=1)

        if plot_type == "depth":
            vmin = 0
            vmax = 0.25
            cmap = cmocean.cm.dense
            cbar_label = "Depth (m)"
        elif plot_type == "wse":
            vmin = np.nanmin(data_to_plot)
            vmax = np.nanmax(data_to_plot)
            cmap = "Blues"
            cbar_label = "Water Surface Elevation (m)"
        else:
            raise ValueError(
                f"Unknown plot_type: '{plot_type}'. Must be 'depth' or 'wse'."
            )

        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            interpolation="none",
            zorder=10,
        )

        # if self.plot_sensors and self.virtual_sensor_loc is not None:
        #    ... logic to reproject and plot sensor locations ...

        ax.text(
            0.05,
            0.95,
            f"Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_array))) * (resolution_m**2), 2)}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
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
        print(f"Plot created successfully and saved to {png_path}.")
