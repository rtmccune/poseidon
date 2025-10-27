import os
from pathlib import Path
import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle
from svgpath2mpl import parse_path

# Third-party libraries
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from mpi4py import MPI
from tqdm import tqdm
from pyproj import Transformer  # <-- ADDED THIS IMPORT

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

        # Store spatial extent and CRS
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.resolution_m = resolution_m
        self.bbox_crs = bbox_crs

        # Sensor attributes
        self.virtual_sensor_loc = virtual_sensor_locations
        self.plot_sensors = plot_sensors

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

            # --- Filtering Logic ---
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
                    self._plot_georeferenced_map(
                        depth_array_path=full_zarr_path,
                        output_filename=output_png_filename,
                        plot_type=plot_type,
                        output_folder=plotting_folder,
                        
                    )

            except Exception as e:
                print(
                    f"   -> An ERROR occurred while processing {zarr_name}: {e}",
                    flush=True,
                )

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
            flood_event_folders = self._list_flood_event_folders()
            # self.preprocess_flood_events() # This method was called but not defined, commented out
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

        # Check if list_flood_event_folders was defined, if not, this will fail
        if flood_event_folders is None:
            if rank == 0:
                print("Rank 0: flood_event_folders is None after bcast.")
            return

        for flood_event in tqdm(
            flood_event_folders[start_index:end_index],
            desc=f"Rank {rank} plotting events",
            unit="event",
        ):
            # Pass the filter list down
            self.process_single_flood_event(
                flood_event, stats_to_plot=stats_to_plot
            )

    def _list_flood_event_folders(self):
        """
        Lists all subdirectories in the main directory, assuming each represents a flood event.

        Returns:
        --------
        list of str
            A list of folder names corresponding to flood events within the main directory.
        """

        flood_event_folders = [
            flood_event
            for flood_event in os.listdir(self.main_dir)
            if os.path.isdir(os.path.join(self.main_dir, flood_event))
        ]

        return flood_event_folders

    def _plot_georeferenced_map(
        self,
        depth_array_path,
        output_filename,
        plot_type,
        output_folder="figures",
    ):
        """
        Generates and saves a georeferenced plot for a flood map (WSE or depth).
        
        This method coordinates the loading, plotting, and saving steps.
        """
        # --- 1. Load and Prepare Data ---
        # This new helper handles all the zarr, numpy, and xarray logic
        geodata = self._load_and_prepare_geodata(
            depth_array_path
        )

        # --- 2. Setup Plot Canvas ---
        fig, ax = plt.subplots(figsize=(10, 10))
        minx, miny, maxx, maxy = geodata["mercator_array"].rio.bounds()
        data_to_plot = geodata["mercator_array"].to_numpy()

        # --- 3. Get Plot-Specific Styling ---
        # This new static method returns cmap, vmin, vmax, etc.
        style = self._get_plot_style(plot_type, data_to_plot)

        # --- 4. Plot Main Data ---
        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),
            cmap=style["cmap"],
            vmin=style["vmin"],
            vmax=style["vmax"],
            alpha=0.7,
            interpolation="none",
            zorder=10,
        )

        # --- 5. Plot Sensor Locations ---
        marker_path = self._create_sensor_marker()
        self._plot_sensor_locations(
            ax=ax,
            marker_path=marker_path,
            original_data_shape=geodata["shape"],
            data_affine_transform=geodata["transform"],
        )

        # --- 6. Add Basemap, Text, Colorbar, and Save ---
        # This new helper handles all the finalization and saving boilerplate
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
        Loads the Zarr array, prepares it, and georeferences it.
        
        Returns a dictionary containing the key data products.
        """
        array_store = zarr.open(str(depth_array_path), mode="r")
        depth_array = array_store[:]  # This is the original array

        print(
            f"    -> Number of valid (non-NaN) data points: {np.count_nonzero(~np.isnan(depth_array))}"
        )

        data_for_xarray = np.flipud(depth_array).astype(float)
        H, W = depth_array.shape

        data_affine_transform = Affine(
            self.resolution_m, 0.0, self.min_x_extent, 0.0, -self.resolution_m, self.max_y_extent
        )

        da_hmax = xr.DataArray(
            data=data_for_xarray, dims=["y", "x"], name="flood_data"
        )
        da_hmax.rio.write_crs(self.bbox_crs, inplace=True)
        da_hmax.rio.write_transform(data_affine_transform, inplace=True)
        da_hmax.rio.write_nodata(np.nan, inplace=True)

        da_hmax_mercator = da_hmax.rio.reproject(3857) # Reproject to Web Mercator

        return {
            "mercator_array": da_hmax_mercator, # The reprojected xarray DataArray
            "original_array": depth_array,     # The raw numpy array
            "shape": (H, W),                   # Shape of the original array
            "transform": data_affine_transform # Affine transform for CRS
        }

    @staticmethod
    def _get_plot_style(plot_type, data_to_plot):
        """
        Returns a dictionary of plotting parameters based on the plot type.
        """
        if plot_type == "depth":
            return {
                "vmin": 0,
                "vmax": 0.25,
                "cmap": cmocean.cm.dense,
                "cbar_label": "Depth (m)"
            }
        elif plot_type == "wse":
            return {
                "vmin": np.nanmin(data_to_plot),
                "vmax": np.nanmax(data_to_plot),
                "cmap": "Blues",
                "cbar_label": "Water Surface Elevation (m)"
            }
        else:
            raise ValueError(
                f"Unknown plot_type: '{plot_type}'. Must be 'depth' or 'wse'."
            )

    def _finalize_and_save_plot(
        self,
        fig,
        ax,
        im,
        geodata,
        cbar_label,
        output_folder,
        output_filename
    ):
        """
        Adds final touches (basemap, text, cbar) and saves the figure.
        """
        # Get bounds and crs from the geodata dictionary
        minx, miny, maxx, maxy = geodata["mercator_array"].rio.bounds()
        crs = geodata["mercator_array"].rio.crs

        # Set limits *before* adding basemap
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        ctx.add_basemap(
            ax,
            crs=crs,
            source=ctx.providers.Esri.WorldImagery,
            zorder=1,  # Basemap is at the bottom
        )

        # Calculate spatial extent from the original array
        spatial_extent = round(
            (np.sum(~np.isnan(geodata["original_array"]))) * (self.resolution_m**2), 2
        )
        
        ax.text(
            0.05,
            0.95,
            f"Spatial Extent ($m^2$): {spatial_extent}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            zorder=30,  # Text is on top
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

    # -----------------------------------------------------------------
    # NEW HELPER METHODS
    # -----------------------------------------------------------------

    @staticmethod
    def _create_sensor_marker():
        """
        Creates a custom matplotlib Path object for the sensor marker.

        This is a static method because it doesn't rely on any 'self' attributes.
        """
        # Load SVG path string
        svg_path_str = "M0 0 C0.825 -0.00128906 1.65 -0.00257812 2.5 -0.00390625 C8.08645564 1.2617751 11.16527062 5.79806233 14.4375 10.1875 C15.31557349 11.35479356 16.19453012 12.52142315 17.07421875 13.6875 C17.77877197 14.62529297 17.77877197 14.62529297 18.49755859 15.58203125 C20.63003679 18.39299373 22.81066988 21.16524823 24.98876953 23.94091797 C32.87088989 33.99515578 40.48478691 44.23203019 48 54.5625 C48.55574707 55.32594727 49.11149414 56.08939453 49.68408203 56.87597656 C56.98821361 66.92210529 64.20967289 77.02547875 71.33984375 87.19580078 C72.50493339 88.85673886 73.67175131 90.51646531 74.83984375 92.17529297 C84.23635549 105.52561863 93.36395435 119.01987438 102.19287109 132.75244141 C103.95234767 135.48840134 105.72007622 138.21889483 107.48828125 140.94921875 C150.05895804 206.96147365 200.12438567 285.23012917 182.93359375 366.8671875 C175.39631755 399.94249141 159.3412289 429.9889702 136 454.5625 C135.51080078 455.08424805 135.02160156 455.60599609 134.51757812 456.14355469 C120.27329692 471.2508615 104.32723065 482.85206456 86 492.5625 C85.29407715 492.93745605 84.5881543 493.31241211 83.86083984 493.69873047 C42.55213774 515.24558359 -9.65633974 519.33070778 -54.06469727 505.54663086 C-56.04851041 504.90027389 -58.02462909 504.23416574 -60 503.5625 C-60.63744141 503.35431641 -61.27488281 503.14613281 -61.93164062 502.93164062 C-87.35443777 494.5604055 -109.59679551 479.78392291 -129 461.5625 C-129.53141602 461.06508301 -130.06283203 460.56766602 -130.61035156 460.05517578 C-165.73760723 426.90395362 -185.00181714 380.48876006 -187.28515625 332.51953125 C-188.96966452 274.30244773 -152.74099454 216.2166584 -123 168.5625 C-122.47180664 167.71042969 -121.94361328 166.85835937 -121.39941406 165.98046875 C-102.42600323 135.42588507 -81.73079704 105.87110301 -60.60449219 76.77587891 C-58.99495681 74.55554297 -57.39107213 72.33123958 -55.7890625 70.10546875 C-50.23587409 62.3920993 -44.64681757 54.70763842 -39 47.0625 C-24.93834436 28.1566023 -24.93834436 28.1566023 -12 8.48046875 C-6.07678149 -0.16141451 -6.07678149 -0.16141451 0 0 Z"

        marker_path = parse_path(svg_path_str)
        marker_path.vertices -= marker_path.vertices.mean(axis=0)  # center
        marker_path.vertices[:, 1] *= -1  # Flip vertically
        return marker_path

    def _plot_sensor_locations(
        self, ax, marker_path, original_data_shape, data_affine_transform
    ):
        """
        Calculates sensor coordinates and plots them on the given axes.

        This method handles:
        - Checking if sensors should be plotted.
        - Transforming sensor indices to CRS coordinates.
        - Transforming CRS coordinates to map (Mercator) coordinates.
        - Plotting the scatter points.
        """
        if not self.plot_sensors or self.virtual_sensor_loc is None:
            return  # Do nothing if plotting is disabled or no sensors exist

        print("    -> Plotting sensor locations from array indices...")
        try:
            H, W = original_data_shape

            # Transformer from data CRS (e.g., EPSG:32119) to map CRS (EPSG:3857)
            transformer = Transformer.from_crs(
                self.bbox_crs, "EPSG:3857", always_xy=True
            )

            mercator_x = []
            mercator_y = []

            # Assumes self.virtual_sensor_loc is (N, 2) array of [Y, X]
            # from the *original* (non-flipped) array
            for y_orig, x_orig in self.virtual_sensor_loc:

                # Convert original Y index to the flipped Y index
                y_flipped = (H - 1) - y_orig
                x_flipped = x_orig  # X index stays the same

                if not (0 <= y_flipped < H and 0 <= x_flipped < W):
                    print(
                        f"   -> WARNING: Sensor index (Y={y_orig}, X={x_orig}) is out of bounds for array shape {H, W}."
                    )
                    continue

                # Get CRS coords from *flipped* indices using the affine transform
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
                print("   -> No valid sensor locations found to plot.")
                return

            colors = plt.cm.viridis(np.linspace(0, 1, num_sensors))

            ax.scatter(
                mercator_x,
                mercator_y,
                marker=marker_path,
                c=colors,
                edgecolors="white",
                s=300,
                linewidths=1,
                label="Virtual Sensors",
                zorder=20,  # Sensors are on top of data
            )
        except Exception as e:
            print(f"   -> WARNING: Failed to plot sensor locations: {e}")
