import os
from pathlib import Path
from svgpath2mpl import parse_path

from datetime import datetime
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from mpi4py import MPI
from tqdm import tqdm
from pyproj import Transformer
import rioxarray
import contextily as ctx
import dask.array as da


def _log(message):
    """Helper function for timestamped logging."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


class DepthMapPlotter:
    """
    A class to process and plot flood depth and Water Surface Elevation
    (WSE) maps from Zarr datasets, with support for parallel processing
    via MPI.
    """

    def __init__(
        self,
        main_dir,
        min_x_extent,
        max_x_extent,
        min_y_extent,
        max_y_extent,
        resolution_m=0.05,
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
        resolution_m : float, optional
            The resolution (pixel size) in meters, by default 0.05.
        bbox_crs : str, optional
            The Coordinate Reference System (CRS) string for the input
            bounding box, by default "EPSG:32119".
        virtual_sensor_locations : np.ndarray, optional
            A NumPy array of shape (N, 2) containing [Y, X] array indices
            for sensor locations, by default None.
        plot_sensors : bool, optional
            A flag to control whether sensor locations should be plotted,
            by default False.
        """
        self.main_dir = Path(main_dir)

        # Store spatial extent, resolution, and CRS
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.resolution_m = resolution_m
        self.bbox_crs = bbox_crs

        # Store sensor attributes
        self.virtual_sensor_loc = virtual_sensor_locations
        self.plot_sensors = plot_sensors
        if self.plot_sensors:
            # Pre-create the custom marker if sensors will be plotted
            self.sensor_marker_path = self._create_sensor_marker()

    def process_single_flood_event(self, flood_event, stats_to_plot=None):
        """
        Processes a single flood event folder, plotting specified Zarr
        maps.

        Parameters
        ----------
        flood_event : str
            The name of the flood event directory.
        stats_to_plot : list of str, optional
            A list of statistic suffixes to plot
            (e.g., ['mean', '95_perc']).
            If None, all found maps are plotted.
        """
        _log(f"\n=== Processing Flood Event: {flood_event} ===")
        flood_event_path = self.main_dir / flood_event
        depth_maps_zarr_dir = flood_event_path / "zarr" / "depth_maps"
        _log(f"  Source Zarr directory: {depth_maps_zarr_dir}")

        if not depth_maps_zarr_dir.is_dir():
            _log(f"  ERROR: Directory not found at '{depth_maps_zarr_dir}'")
            _log(f"=== Processing Aborted: {flood_event} ===")
            return

        try:
            # Get a sorted list of all files/dirs in the zarr directory
            contents = sorted(os.listdir(depth_maps_zarr_dir))
        except OSError as e:
            _log(
                f"  ERROR: Could not list contents of "
                f"'{depth_maps_zarr_dir}': {e}"
            )
            _log(f"=== Processing Aborted: {flood_event} ===")
            return

        # Pre-filter for directories only, as Zarr stores are directories
        all_zarr_dirs = [
            name for name in contents if (depth_maps_zarr_dir / name).is_dir()
        ]
        total_potential_files = len(all_zarr_dirs)

        if total_potential_files == 0:
            _log(
                "  WARNING: No Zarr directories found in source folder. "
                f"Nothing to plot."
            )
            _log(f"=== Processing Complete: {flood_event} ===")
            return

        # --- NEW FILTERING LOGIC ---
        # Apply the stats_to_plot filter *before* the loop
        if stats_to_plot is not None:
            _log(f"  Filtering for stats: {stats_to_plot}")
            files_to_process = [
                name
                for name in all_zarr_dirs
                if any(name.endswith(stat) for stat in stats_to_plot)
            ]
            skipped_filter = total_potential_files - len(files_to_process)
        else:
            _log("  No filter provided, processing all found files.")
            files_to_process = all_zarr_dirs
            skipped_filter = 0
        # --- END NEW FILTERING LOGIC ---

        # The *actual* total files to process for the progress bar
        total_files = len(files_to_process)

        if total_files == 0:
            _log(
                f"  Found {total_potential_files} potential files, but 0 match"
                f"the filter. Nothing to plot."
            )
            _log(f"=== Processing Complete: {flood_event} ===")
            return

        _log(
            f"  Found {total_potential_files} potential Zarr stores. "
            f"{total_files} will be processed."
        )

        # Determine report interval (print ~every 5% or ~20 updates)
        report_interval = max(1, total_files // 20)
        processed_count = 0
        skipped_type = 0

        # Iterate over the *filtered* list
        for i, zarr_name in enumerate(files_to_process):
            # Log progress periodically (now accurate)
            if (i + 1) % report_interval == 0 and (i + 1) != total_files:
                percent_complete = ((i + 1) / total_files) * 100
                _log(
                    f"    ...progress: {percent_complete:.0f}% complete "
                    f"({i + 1}/{total_files})"
                )

            full_zarr_path = depth_maps_zarr_dir / zarr_name

            # The stats_to_plot filter is already applied, so we remove
            # it from here

            try:
                output_png_filename = f"{zarr_name}.png"
                plot_type = None
                plotting_folder = None

                # Determine plot type and output folder based on filename
                if "wse" in zarr_name:
                    plot_type = "wse"
                    plotting_folder = flood_event_path / "plots" / "WSE_maps"

                elif "depth" in zarr_name:
                    plot_type = "depth"
                    plotting_folder = flood_event_path / "plots" / "depth_maps"

                else:
                    # This check remains valid, as a file could match the
                    # filter (e.g. "depth_mean") but not be "wse" or
                    # "depth" if the logic was ever changed.
                    _log(f"    -> Skipping (unknown plot type): {zarr_name}")
                    skipped_type += 1
                    continue  # Skip if not a recognized type

                # If a valid plot type was identified, proceed with
                # plotting
                if plot_type and plotting_folder:
                    self._plot_georeferenced_map(
                        depth_array_path=full_zarr_path,
                        output_filename=output_png_filename,
                        plot_type=plot_type,
                        output_folder=plotting_folder,
                    )
                    processed_count += 1

            except Exception as e:
                _log(
                    f"    -> An ERROR occurred while processing "
                    f"{zarr_name}: {e}"
                )
                # Continue to the next file

        _log(f"  Successfully processed and plotted {processed_count} files.")
        if skipped_filter > 0:
            _log(f"  Skipped {skipped_filter} files due to name filter.")
        if skipped_type > 0:
            _log(f"  Skipped {skipped_type} files due to unknown type.")
        _log(f"=== Processing Complete: {flood_event} ===")

    def process_flood_events_HPC(self, stats_to_plot=None):
        """
        Processes and generates plots for flood events in parallel using
        MPI.

        Parameters
        ----------
        stats_to_plot : list of str, optional
            A list of statistic suffixes to plot
            (e.g., ['mean', '95_perc']).
            If None, all found maps are plotted.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Rank 0 discovers the flood event folders
        if rank == 0:
            flood_event_folders = self._list_flood_event_folders()
        else:
            flood_event_folders = None

        # Rank 0 broadcasts the list of folders to all other ranks
        flood_event_folders = comm.bcast(flood_event_folders, root=0)

        if not flood_event_folders:
            if rank == 0:
                _log("No flood event folders found.")
            return

        # Divide the work among the ranks
        n_folders = len(flood_event_folders)
        chunk_size = n_folders // size
        start_index = rank * chunk_size
        end_index = start_index + chunk_size if rank != size - 1 else n_folders

        # Each rank processes its assigned chunk of folders
        # Note: tqdm provides its own progress bar for this outer loop
        for flood_event in tqdm(
            flood_event_folders[start_index:end_index],
            desc=f"Rank {rank} plotting events",
            unit="event",
        ):
            # Pass the filter list to the single-event processor
            self.process_single_flood_event(
                flood_event, stats_to_plot=stats_to_plot
            )

    def _list_flood_event_folders(self):
        """
        Lists subdirectories in the main directory.

        Assumes each subdirectory represents a single flood event.

        Returns
        -------
        list of str
            A sorted list of folder names (flood events).
        """
        flood_event_folders = [
            flood_event
            for flood_event in os.listdir(self.main_dir)
            if os.path.isdir(os.path.join(self.main_dir, flood_event))
        ]
        return sorted(flood_event_folders)

    def _plot_georeferenced_map(
        self,
        depth_array_path,
        output_filename,
        plot_type,
        output_folder="figures",
    ):
        """
        Generates and saves a georeferenced plot for a single flood map.

        This method coordinates the loading of lazy data, triggering
        computation for plotting, adding a basemap, and saving the
        final figure.

        Parameters
        ----------
        depth_array_path : str or pathlib.Path
            Path to the Zarr array to be plotted.
        output_filename : str
            The name for the output PNG file (e.g., "plot.png").
        plot_type : {'depth', 'wse'}
            The type of data being plotted, which determines
            styling (colormap, vmin, vmax).
        output_folder : str or pathlib.Path, optional
            The directory to save the final plot, by default "figures".
        """
        # Load and prepare the data lazily
        geodata = self._load_and_prepare_geodata(depth_array_path)
        src_data = geodata["mercator_array"]

        # Setup the plot canvas
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get the bounds from the lazy xarray object *before* computing
        minx, miny, maxx, maxy = src_data.rio.bounds()

        # Trigger the entire lazy computation chain (flip, reproject,
        # etc.)
        # and load the final reprojected array into memory for plotting.
        data_to_plot = src_data.to_numpy()

        # Get plot-specific styling (cmap, vmin, vmax)
        style = self._get_plot_style(plot_type, src_data)

        # Plot the main data array
        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),  # Use the calculated bounds
            cmap=style["cmap"],
            vmin=style["vmin"],
            vmax=style["vmax"],
            alpha=0.7,
            interpolation="none",
            zorder=10,
        )

        # Plot sensor locations if enabled
        if self.plot_sensors:
            self._plot_sensor_locations(
                ax=ax,
                marker_path=self.sensor_marker_path,
                original_data_shape=geodata["shape"],
                data_affine_transform=geodata["transform"],
            )

        # Add basemap, text, colorbar, and save the figure
        self._finalize_and_save_plot(
            fig,
            ax,
            im,
            geodata,
            style["cbar_label"],
            output_folder,
            output_filename,
        )

    def _load_and_prepare_geodata(self, depth_array_path):
        """
        Loads a Zarr array lazily and prepares it for reprojection.

        This method opens the Zarr store, creates a lazy dask array,
        calculates the spatial extent, and sets up a georeferenced
        xarray DataArray. It uses .isel() to lazily flip the Y-axis,
        ensuring compatibility with rioxarray's reprojection.

        Parameters
        ----------
        depth_array_path : str or pathlib.Path
            Path to the input Zarr array.

        Returns
        -------
        dict
            A dictionary containing:
            - 'mercator_array': Lazy reprojected xr.DataArray
            (EPSG:3857).
            - 'shape': Original (H, W) shape of the array.
            - 'transform': Original 'top-left' affine transform.
            - 'spatial_extent': Calculated spatial extent in square
            meters.
        """
        # Open the Zarr store
        array_store = zarr.open(str(depth_array_path), mode="r")
        H, W = array_store.shape

        # Create a lazy dask array from the Zarr store (still
        # non-flipped)
        lazy_data = da.from_array(array_store, chunks=array_store.chunks)

        # Calculate spatial extent from the raw lazy data
        valid_pixel_count = da.sum(~da.isnan(lazy_data)).compute()
        spatial_extent = round((valid_pixel_count) * (self.resolution_m**2), 2)

        # Create an xarray DataArray from the non-flipped lazy data
        da_hmax = xr.DataArray(
            data=lazy_data.astype(float), dims=["y", "x"], name="flood_data"
        )

        # Lazily flip the array using xarray's .isel()
        da_hmax_flipped = da_hmax.isel(y=slice(None, None, -1))

        # Define the "top-left" affine transform based on class
        # attributes
        data_affine_transform = Affine(
            self.resolution_m,
            0.0,
            self.min_x_extent,
            0.0,
            -self.resolution_m,
            self.max_y_extent,
        )

        # Attach georeferencing to the FLIPPED array
        da_hmax_flipped = da_hmax_flipped.rio.write_crs(self.bbox_crs)
        da_hmax_flipped = da_hmax_flipped.rio.write_transform(
            data_affine_transform
        )
        da_hmax_flipped = da_hmax_flipped.rio.write_nodata(np.nan)

        # Lazily reproject the array to Web Mercator (EPSG:3857)
        da_hmax_mercator = da_hmax_flipped.rio.reproject(3857)

        return {
            "mercator_array": da_hmax_mercator,  # Lazy reprojected array
            "shape": (H, W),  # Original shape
            "transform": data_affine_transform,
            # Original top-left transform
            "spatial_extent": spatial_extent,  # Pre-calculated extent
        }

    @staticmethod
    def _get_plot_style(plot_type, data_array):
        """
        Returns a dictionary of plotting parameters based on the plot
        type.

        For 'wse', it computes vmin/vmax from the data_array.

        Parameters
        ----------
        plot_type : {'depth', 'wse'}
            The type of plot.
        data_array : xr.DataArray
            The data array, used to calculate limits for 'wse' plots.

        Returns
        -------
        dict
            Plotting parameters (vmin, vmax, cmap, cbar_label).

        Raises
        ------
        ValueError
            If plot_type is not 'depth' or 'wse'.
        """
        if plot_type == "depth":
            # Use fixed limits for depth plots
            return {
                "vmin": 0,
                "vmax": 0.25,
                "cmap": cmocean.cm.dense,
                "cbar_label": "Depth (m)",
            }
        elif plot_type == "wse":
            # Compute dynamic limits for WSE plots
            vmin = data_array.min(skipna=True).compute()
            vmax = data_array.max(skipna=True).compute()

            return {
                "vmin": vmin,
                "vmax": vmax,
                "cmap": "Blues",
                "cbar_label": "Water Surface Elevation (m)",
            }
        else:
            raise ValueError(
                f"Unknown plot_type: '{plot_type}'. Must be 'depth' or 'wse'."
            )

    def _finalize_and_save_plot(
        self, fig, ax, im, geodata, cbar_label, output_folder, output_filename
    ):
        """
        Adds final touches (basemap, text, colorbar) and saves the
        figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        im : matplotlib.image.AxesImage
            The image object returned by imshow (for the colorbar).
        geodata : dict
            The dictionary returned by _load_and_prepare_geodata.
        cbar_label : str
            The label for the colorbar.
        output_folder : str or pathlib.Path
            The directory to save the final plot.
        output_filename : str
            The name for the output PNG file.
        """
        # Get bounds and CRS from the reprojected mercator array
        minx, miny, maxx, maxy = geodata["mercator_array"].rio.bounds()
        crs = geodata["mercator_array"].rio.crs

        # Set map limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Add Esri World Imagery basemap
        ctx.add_basemap(
            ax,
            crs=crs,
            source=ctx.providers.Esri.WorldImagery,
            zorder=1,
        )

        # Use the pre-calculated spatial extent
        spatial_extent = geodata["spatial_extent"]

        # Add spatial extent text box
        ax.text(
            0.05,
            0.95,
            f"Spatial Extent ($m^2$): {spatial_extent}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            zorder=30,
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
        cbar.set_label(cbar_label)

        # Create output directory and save figure
        png_path = Path(output_folder) / output_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)

        ax.set_title("")
        ax.set_axis_off()
        plt.tight_layout()

        plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        _log(f"    -> Plot saved: {png_path}")

    # -----------------------------------------------------------------
    # HELPER METHODS
    # -----------------------------------------------------------------

    @staticmethod
    def _create_sensor_marker():
        """
        Creates a custom matplotlib Path object for the sensor marker.

        This is a static method as it doesn't rely on 'self'. It
        parses a hardcoded SVG path string, centers it, and flips
        it vertically to create a custom marker.

        Returns
        -------
        matplotlib.path.Path
            A Path object to be used as a scatter plot marker.
        """
        # SVG path string for the custom marker
        svg_path_str = (
            "M0 0 C0.825 -0.00128906 1.65 -0.00257812 2.5 -0.00390625 "
            "C8.08645564 1.2617751 11.16527062 5.79806233 14.4375 10.1875 "
            "C15.31557349 11.35479356 16.19453012 12.52142315 17.07421875 13.6875 "
            "C17.77877197 14.62529297 17.77877197 14.62529297 18.49755859 15.58203125 "
            "C20.63003679 18.39299373 22.81066988 21.16524823 24.98876953 23.94091797 "
            "C32.87088989 33.99515578 40.48478691 44.23203019 48 54.5625 "
            "C48.55574707 55.32594727 49.11149414 56.08939453 49.68408203 56.87597656 "
            "C56.98821361 66.92210529 64.20967289 77.02547875 71.33984375 87.19580078 "
            "C72.50493339 88.85673886 73.67175131 90.51646531 74.83984375 92.17529297 "
            "C84.23635549 105.52561863 93.36395435 119.01987438 102.19287109 132.75244141 "
            "C103.95234767 135.48840134 105.72007622 138.21889483 107.48828125 140.94921875 "
            "C150.05895804 206.96147365 200.12438567 285.23012917 182.93359375 366.8671875 "
            "C175.39631755 399.94249141 159.3412289 429.9889702 136 454.5625 "
            "C135.51080078 455.08424805 135.02160156 455.60599609 134.51757812 456.14355469 "
            "C120.27329692 471.2508615 104.32723065 482.85206456 86 492.5625 "
            "C85.29407715 492.93745605 84.5881543 493.31241211 83.86083984 493.69873047 "
            "C42.55213774 515.24558359 -9.65633974 519.33070778 -54.06469727 505.54663086 "
            "C-56.04851041 504.90027389 -58.02462909 504.23416574 -60 503.5625 "
            "C-60.63744141 503.35431641 -61.27488281 503.14613281 -61.93164062 502.93164062 "
            "C-87.35443777 494.5604055 -109.59679551 479.78392291 -129 461.5625 "
            "C-129.53141602 461.06508301 -130.06283203 460.56766602 -130.61035156 460.05517578 "
            "C-165.73760723 426.90395362 -185.00181714 380.48876006 -187.28515625 332.51953125 "
            "C-188.96966452 274.30244773 -152.74099454 216.2166584 -123 168.5625 "
            "C-122.47180664 167.71042969 -121.94361328 166.85835937 -121.39941406 165.98046875 "
            "C-102.42600323 135.42588507 -81.73079704 105.87110301 -60.60449219 76.77587891 "
            "C-58.99495681 74.55554297 -57.39107213 72.33123958 -55.7890625 70.10546875 "
            "C-50.23587409 62.3920993 -44.64681757 54.70763842 -39 47.0625 "
            "C-24.93834436 28.1566023 -24.93834436 28.1566023 -12 8.48046875 "
            "C-6.07678149 -0.16141451 -6.07678149 -0.16141451 0 0 Z"
        )
        # Parse the SVG string into a matplotlib Path
        marker_path = parse_path(svg_path_str)
        # Center the marker at (0, 0)
        marker_path.vertices -= marker_path.vertices.mean(axis=0)
        # Flip the marker vertically (SVGs and matplotlib have
        # different Y-axes)
        marker_path.vertices[:, 1] *= -1
        return marker_path

    def _plot_sensor_locations(
        self, ax, marker_path, original_data_shape, data_affine_transform
    ):
        """
        Calculates sensor coordinates and plots them on the given axes.

        This method transforms sensor locations from their [Y, X]
        array indices (relative to the *original*, non-flipped array)
        into the map's CRS (EPSG:3857) and plots them.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        marker_path : matplotlib.path.Path
            The custom marker object to use.
        original_data_shape : tuple
            The (H, W) shape of the *original* data array.
        data_affine_transform : affine.Affine
            The 'top-left' affine transform for the *flipped* array.
        """
        if not self.plot_sensors or self.virtual_sensor_loc is None:
            return  # Do nothing if plotting is disabled or no sensors

        try:
            H, W = original_data_shape

            # Create a transformer for sensor coordinates
            transformer = Transformer.from_crs(
                self.bbox_crs, "EPSG:3857", always_xy=True
            )

            mercator_x = []
            mercator_y = []

            # Assumes self.virtual_sensor_loc is (N, 2) array of [Y, X]
            # indices from the *original* (non-flipped) array
            for y_orig, x_orig in self.virtual_sensor_loc:

                # Convert original Y index to the flipped Y index
                y_flipped = (H - 1) - y_orig
                x_flipped = x_orig  # X index stays the same

                # Check if the sensor is within the array bounds
                if not (0 <= y_flipped < H and 0 <= x_flipped < W):
                    _log(
                        f"    -> WARNING: Sensor index (Y={y_orig}, X={x_orig}) "
                        f"is out of bounds for array shape {H, W}."
                    )
                    continue

                # Get CRS coords from *flipped* indices
                x_crs, y_crs = data_affine_transform * (
                    x_flipped + 0.5,
                    y_flipped + 0.5,
                )

                # Transform from data CRS to map CRS (Mercator)
                mx, my = transformer.transform(x_crs, y_crs)

                mercator_x.append(mx)
                mercator_y.append(my)

            num_sensors = len(mercator_x)
            if num_sensors == 0:
                _log("    -> No valid sensor locations found to plot.")
                return

            # Generate distinct colors for each sensor
            colors = plt.cm.viridis(np.linspace(0, 1, num_sensors))

            # Plot sensors on the map
            ax.scatter(
                mercator_x,
                mercator_y,
                marker=marker_path,
                c=colors,
                edgecolors="white",
                s=300,
                linewidths=1,
                label="Virtual Sensors",
                zorder=20,
            )
        except Exception as e:
            _log(f"    -> WARNING: Failed to plot sensor locations: {e}")
