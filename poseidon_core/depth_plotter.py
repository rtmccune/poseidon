import os
import gc
import numpy as np
import pandas as pd
import pytz
import zarr
from tqdm import tqdm
from datetime import datetime
import cmocean
import matplotlib.pyplot as plt
from mpi4py import MPI
import poseidon_core
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import rioxarray
import xarray as xr
from affine import Affine

# Imports needed for the manual colorbar
from matplotlib import cm
from matplotlib.colors import Normalize


class DepthPlotter:

    def __init__(self, main_dir, min_x_extent, max_x_extent,
                 min_y_extent, max_y_extent, virtual_sensor_locations,
                 bbox_crs='EPSG:32119'):
        """
        Initializes the visualization class with directory paths and virtual sensor locations.

        Parameters:
        -----------
        main_dir : str or Path
            The main directory where visualization data or outputs are stored.

        virtual_sensor_locations : dict or pandas.DataFrame
            A mapping or dataset containing coordinates or metadata for virtual sensor locations.

        Attributes:
        -----------
        water_level_color : matplotlib.colors.Colormap
            Color used for water level visualizations.

        max_depth_color : matplotlib.colors.Colormap
            Color used to represent maximum water depth.

        avg_depth_color : matplotlib.colors.Colormap
            Color used to represent average water depth.

        sensor_1_color : matplotlib.colors.Colormap
            Color assigned to sensor 1.

        sensor_2_color : matplotlib.colors.Colormap
            Color assigned to sensor 2.

        sensor_3_color : matplotlib.colors.Colormap
            Color assigned to sensor 3.
        """
        self.main_dir = main_dir
        self.virtual_sensor_loc = virtual_sensor_locations

        self.water_level_color = cmocean.cm.balance(0.2)
        self.max_depth_color = cmocean.cm.balance(0.9)
        self.avg_depth_color = cmocean.cm.balance(0.6)
        self.sensor_1_color = cmocean.cm.phase(0.1)
        self.sensor_2_color = cmocean.cm.phase(0.3)
        self.sensor_3_color = cmocean.cm.phase(0.5)
        
        self.min_x_extent = min_x_extent
        self.max_x_extent = max_x_extent
        self.min_y_extent = min_y_extent
        self.max_y_extent = max_y_extent
        self.bbox_crs = bbox_crs

    def list_flood_event_folders(self):
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

    def match_measurements_to_images(self, flood_event, flood_event_path):
        """
        Matches water level measurements to the closest image timestamps for a given flood event.

        For each image in the "orig_images" folder, this method extracts the timestamp and sensor ID
        from the filename, finds the closest water level observation for that sensor in the CSV file,
        and stores the matched data (image filename, matched timestamp, and converted water level)
        in a new CSV file named "wtr_lvl_obs_to_image_matches.csv".

        Parameters:
        -----------
        flood_event : str
            The name of the flood event (used to find the CSV file with measurements).
        flood_event_path : str
            The path to the flood event directory containing the CSV file and image folder.

        Returns:
        --------
        None
            Writes the matched data to a CSV file in the flood event directory.
        """

        sunnyd_data = pd.read_csv(os.path.join(flood_event_path, flood_event + ".csv"))
        sunnyd_data["time_UTC"] = pd.to_datetime(sunnyd_data["time_UTC"])

        orig_images_path = os.path.join(flood_event_path, "orig_images")
        image_list = sorted(os.listdir(orig_images_path))
        match = []

        # Iterate over image filenames
        for filename in image_list:
            # Extract the sensor id and timestamp
            sensor_id = image_processing.image_utils.extract_camera_name(filename)[4:]
            timestamp = image_processing.image_utils.extract_timestamp(filename)

            timestamp = pytz.utc.localize(datetime.strptime(timestamp, "%Y%m%d%H%M%S"))

            # Filter the dataframe by sensor id
            filtered_df = sunnyd_data[sunnyd_data["sensor_ID"] == sensor_id]

            # Find the closest timestamp
            closest_row = filtered_df.iloc[
                (filtered_df["time_UTC"] - timestamp).abs().argsort()[:1]
            ]

            # Append the result
            if not closest_row.empty:
                result = {
                    "image_filename": filename,
                    "closest_utc_time": closest_row["time_UTC"].values[0],
                    "water_level": closest_row["water_level"].values[0] * 0.3048,
                    # 'sensor_water_level': (closest_row['sensor_water_level_adj'].values[0] - 3.05) * 0.3048
                }
                match.append(result)

        # Convert the results to a dataframe
        obs_to_image_matches = pd.DataFrame(match)
        obs_to_image_matches.to_csv(
            os.path.join(flood_event_path, "wtr_lvl_obs_to_image_matches.csv")
        )

    def gen_virtual_sensor_depths(self, flood_event_path):
        """
        Generates depth measurements for virtual sensors based on depth maps and stores them in a Zarr format.

        This method processes depth maps located in the "zarr/depth_maps_95th_ponding" directory for a specific
        flood event, extracts depth information for predefined virtual sensor locations, and computes the
        maximum and average depths for each map. It then stores the results in a Zarr store for later use.

        Parameters:
        -----------
        flood_event_path : str
            The path to the flood event directory containing the Zarr depth maps.

        Returns:
        --------
        None
            Writes the generated virtual sensor depth data, maximum depth, and average depth to a Zarr store.
        """

        depth_maps_zarr_dir = os.path.join(
            flood_event_path, "zarr", "depth_maps"
        )
        output_zarr_store = os.path.join(
            flood_event_path, "zarr", "virtual_sensor_wses"
        )

        timestamp_list = []

        if os.path.exists(depth_maps_zarr_dir):
            file_names = [
                f for f in os.listdir(depth_maps_zarr_dir) if f.endswith("wse_map_95_perc")
            ]
            num_files = len(file_names)

            # Preallocate NumPy arrays for better performance
            max_depth_array = np.empty(num_files, dtype=np.float32)
            avg_depth_array = np.empty(num_files, dtype=np.float32)
            vs_depth_array = np.empty(
                (num_files, len(self.virtual_sensor_loc)), dtype=np.float32
            )

            for idx, file_name in enumerate(file_names):
                timestamp = image_processing.image_utils.extract_timestamp(file_name)
                timestamp_list.append(timestamp)

                file_zarr_store = os.path.join(depth_maps_zarr_dir, file_name)
                img_store = zarr.open(file_zarr_store, mode="r")
                depth_map = img_store[:]

                max_depth_array[idx] = np.nanmax(depth_map)
                avg_depth_array[idx] = np.nanmean(depth_map)

                for i, (x, y) in enumerate(self.virtual_sensor_loc):
                    vs_depth_array[idx, i] = depth_map[y, x]

            # # Convert timestamps to a NumPy array of strings
            # datetimes = np.array(
            #     pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30"
            # )

            # # Save to a Zarr store
            # root = zarr.open_group(
            #     output_zarr_store, mode="w"
            # )  # Overwrite existing store

            # root.create_array("timestamps", shape=datetimes.shape, dtype=datetimes.dtype)
            # root["timestamps"][:] = datetimes  # Assign data

            # root.create_array(
            #     "max_depths", shape=max_depth_array.shape, dtype=np.float32
            # )
            # root["max_depths"][:] = max_depth_array

            # root.create_array(
            #     "avg_depths", shape=avg_depth_array.shape, dtype=np.float32
            # )
            # root["avg_depths"][:] = avg_depth_array

            # root.create_array("vs_depths", shape=vs_depth_array.shape, dtype=np.float32)
            # root["vs_depths"][:] = vs_depth_array
            
            # 1. Create the NumPy array with a specific, hardcoded string dtype.
            #    A length like 'U30' is safe for standard ISO format datetimes.
            datetimes_np = np.array(
                pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U30"
            )

            # 2. Open the Zarr store for writing.
            root = zarr.open_group(
                output_zarr_store, mode="w"
            )

            # 3. Create the Zarr arrays using the 'data' argument.
            #    This is the simplest and safest way. Zarr will correctly infer
            #    the shape and dtype from your NumPy arrays.
            root.create_array("timestamps", data=datetimes_np)
            root.create_array("max_depths", data=max_depth_array)
            root.create_array("avg_depths", data=avg_depth_array)
            root.create_array("vs_depths", data=vs_depth_array)

    def preprocess_flood_events(self):
        """
        Preprocesses flood events by matching measurements to images and generating virtual sensor depths.

        This method iterates through all flood event folders, processes each flood event by matching
        water level measurements to corresponding images, and generates depth measurements for virtual
        sensors based on the depth maps. The results are stored for further analysis and plotting.

        The following methods are called for each flood event:
        - `match_measurements_to_images`: Matches water level measurements to images based on timestamps.
        - `gen_virtual_sensor_depths`: Generates depth measurements for virtual sensor locations based on depth maps.

        Returns:
        --------
        None
            This method modifies the state of the object by generating files, but does not return any values.
        """

        flood_event_folders = self.list_flood_event_folders()

        for flood_event in tqdm(
            flood_event_folders,
            desc="Preprocessing flood events for plotting...",
            unit="event",
        ):

            flood_event_path = os.path.join(self.main_dir, flood_event)
            self.match_measurements_to_images(flood_event, flood_event_path)
            self.gen_virtual_sensor_depths(flood_event_path)

    def load_virtual_sensor_depths(self, flood_event_path):
        """
        Loads virtual sensor depth data from a Zarr store for a given flood event.

        This method attempts to load depth-related data from the Zarr store located at the specified
        flood event path. It retrieves the timestamps, maximum depths, average depths, and depths for
        each virtual sensor location, and returns them for further processing or analysis.

        Parameters:
        -----------
        flood_event_path : str
            The path to the folder containing the flood event data, including the "zarr" subdirectory
            where the virtual sensor depths are stored.

        Returns:
        --------
        tuple
            A tuple containing four elements:
            - `datetimes` (pd.DatetimeIndex): A pandas datetime object representing the timestamps for the depth data.
            - `max_depths` (np.ndarray): An array of maximum depth values.
            - `avg_depths` (np.ndarray): An array of average depth values.
            - `vs_depths` (np.ndarray): A 2D array of depth values for each virtual sensor location.

        Raises:
        -------
        FileNotFoundError
            If the Zarr store at the specified path does not exist, a FileNotFoundError is raised.

        Notes:
        ------
        The data is expected to be stored in a Zarr format, and this method assumes that the depth maps
        have been previously generated and stored using `gen_virtual_sensor_depths`.
        """
        zarr_store_path = os.path.join(
            flood_event_path, "zarr", "virtual_sensor_wses"
        )

        if os.path.exists(zarr_store_path):
            root = zarr.open(zarr_store_path, mode="r")

            timestamps = root["timestamps"][:]  # Load as an array of strings
            max_depths = root["max_depths"][:]
            avg_depths = root["avg_depths"][:]
            vs_depths = root["vs_depths"][:]

            # Convert timestamps back to pandas datetime
            datetimes = pd.to_datetime(timestamps, utc=True)

            return datetimes, max_depths, avg_depths, vs_depths
        else:
            raise FileNotFoundError(f"Zarr store not found: {zarr_store_path}")

    def plot_depth_map_and_wtr_levels(
        self,
        depth_map_zarr_dir,
        orig_image_rects_zarr_dir,
        datetimes,
        obs_to_img_matches,
        vs_depths,
        plotting_folder,
        depth_min=0,
        depth_max=0.25,
    ):
        """
        Plots depth maps and water level data over time for a given flood event.

        This method generates plots for each depth map and associated water level data in the provided
        directories. It overlays depth maps on original images and plots the observed and virtual sensor
        water levels over time. The generated figures are saved to the specified plotting folder.

        Parameters:
        -----------
        depth_map_zarr_dir : str
            Path to the directory containing depth map Zarr files, including the "_ponding" files to be processed.

        orig_image_rects_zarr_dir : str
            Path to the directory containing the original rectified images in Zarr format, used for overlaying on the depth map.

        datetimes : pd.DatetimeIndex
            A pandas datetime index corresponding to the timestamps for the virtual sensor depths.

        obs_to_img_matches : pd.DataFrame
            Dataframe containing the observed water levels and corresponding image filenames with their timestamps.

        vs_depths : np.ndarray
            2D array containing the virtual sensor depths over time, with each column corresponding to a virtual sensor.

        plotting_folder : str
            Path to the directory where the resulting plots will be saved.

        depth_min : float, optional
            Minimum depth value for the depth map color scale (default is 0).

        depth_max : float, optional
            Maximum depth value for the depth map color scale (default is 0.25).

        Returns:
        --------
        None

        Notes:
        ------
        This method processes depth maps and images in Zarr format, and plots depth maps on top of original
        images, with overlaid scatter points representing the locations of virtual sensors. The water level
        data is plotted over time for comparison. The generated plots are saved as high-resolution PNG images.

        The method assumes that the depth maps and original images have timestamps that can be matched between
        the two directories and that the virtual sensor locations and water levels are correctly provided.
        """

        if os.path.exists(depth_map_zarr_dir):
            for file_name in sorted(os.listdir(depth_map_zarr_dir)):
                if file_name.endswith("_ponding"):
                    timestamp = image_processing.image_utils.extract_timestamp(
                        file_name
                    )
                    date = pd.to_datetime(timestamp, utc=True)

                    orig_file_name = next(
                        (
                            f
                            for f in os.listdir(orig_image_rects_zarr_dir)
                            if image_processing.image_utils.extract_timestamp(f)
                            == timestamp
                        ),
                        None,
                    )

                    if orig_file_name is None:
                        print(
                            f"Warning: No matching original image found for {file_name}"
                        )
                        continue

                    orig_zarr_store_path = os.path.join(
                        orig_image_rects_zarr_dir, orig_file_name
                    )
                    orig_img_store = zarr.open(orig_zarr_store_path, mode="r")
                    orig_image = orig_img_store[:]  # Consider downsampling if necessary

                    zarr_store_path = os.path.join(depth_map_zarr_dir, file_name)
                    img_store = zarr.open(zarr_store_path, mode="r")
                    depth_map = img_store[
                        :
                    ]  # Again, consider loading only necessary slices

                    # print(f"Processing depth map: {file_name}")

                    fig, (ax1, ax3) = plt.subplots(
                        2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [1.5, 1]}
                    )

                    # Plot depth map
                    im = ax1.imshow(
                        orig_image, cmap="gray"
                    )  # Assuming orig_image is grayscale
                    im = ax1.imshow(
                        depth_map, cmap=cmocean.cm.deep, vmin=depth_min, vmax=depth_max
                    )
                    ax1.scatter(
                        self.virtual_sensor_loc[0][0],
                        self.virtual_sensor_loc[0][1],
                        color=self.sensor_1_color,
                        s=15,
                        marker="v",
                    )
                    ax1.scatter(
                        self.virtual_sensor_loc[1][0],
                        self.virtual_sensor_loc[1][1],
                        color=self.sensor_2_color,
                        s=15,
                        marker="s",
                    )
                    ax1.scatter(
                        self.virtual_sensor_loc[2][0],
                        self.virtual_sensor_loc[2][1],
                        color=self.sensor_3_color,
                        s=15,
                        marker="d",
                    )
                    cbar = plt.colorbar(im, ax=ax1, label="Depth")
                    cbar.set_label("Depth (m)")
                    ax1.invert_yaxis()
                    ax1.set_xlabel("X (cm)")
                    ax1.set_ylabel("Y (cm)")
                    ax1.text(
                        0.05,
                        0.95,
                        f"Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_map))) * 0.0025, 2)}",
                        transform=ax1.transAxes,
                        fontsize=12,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
                    )

                    # Plot water levels
                    ax3.plot(
                        obs_to_img_matches["closest_utc_time"],
                        obs_to_img_matches["water_level"],
                        label="Observed Water Level",
                        color=self.water_level_color,
                    )
                    ax3.scatter(
                        datetimes,
                        vs_depths[:, 2],
                        label="Sensor 1 Depth",
                        marker="d",
                        color=self.sensor_3_color,
                        s=10,
                    )
                    ax3.scatter(
                        datetimes,
                        vs_depths[:, 0],
                        label="Sensor 2 Depth",
                        marker="v",
                        color=self.sensor_1_color,
                        s=10,
                    )
                    ax3.scatter(
                        datetimes,
                        vs_depths[:, 1],
                        label="Sensor 3 Depth",
                        marker="s",
                        color=self.sensor_2_color,
                        s=10,
                    )
                    ax3.axvline(x=date, color="k", linestyle="-", zorder=1)
                    ax3.set_ylim(-0.25, 0.75)
                    ax3.tick_params(axis="x", rotation=45)
                    ax3.set_title("Water Level From Virtual Sensor Locations Over Time")
                    ax3.set_ylabel("Water Depth (m)")
                    ax3.grid(True)
                    ax3.legend(loc="upper right")

                    plt.tight_layout()

                    # Save the figure
                    plt.savefig(
                        os.path.join(plotting_folder, file_name),
                        bbox_inches="tight",
                        pad_inches=0.1,
                        dpi=300,
                    )

                    plt.close(fig)  # Close the figure to free memory
                    del orig_image, depth_map  # Delete large variables
                    gc.collect()  # Force garbage collection

    
    def plot_flood_from_numpy(depth_array, min_x, max_x, min_y, max_y,
                                    resolution_m=0.05, bbox_crs='EPSG:32119', output_folder='figures'):
        """
        Plots a flood numpy array, correcting for a "bottom-left" array origin by
        flipping the array vertically before georeferencing.

        Args:
            numpy_array (np.ndarray): The raw numpy array, assumed to have a (0,0) origin
                                    at the bottom-left.
            ... (other args are the same) ...
        """

        # 1. Build and georeferenced the DataArray.
        # --- THE CRITICAL FIX BASED ON YOUR INSIGHT ---
        # Vertically flip the array to convert from a "bottom-left" origin to the
        # "top-left" origin expected by geospatial raster standards.
        data_for_xarray = np.flipud(depth_array).astype(float)

        transform = Affine(resolution_m, 0.0, min_x, 0.0, -resolution_m, max_y)
        da_hmax = xr.DataArray(
            data=data_for_xarray,
            dims=["y", "x"],
            name='flood_depth'
        )
        da_hmax.rio.write_crs(bbox_crs, inplace=True)
        da_hmax.rio.write_transform(transform, inplace=True)
        da_hmax.rio.write_nodata(np.nan, inplace=True)

        # 2. Reproject the raster to Web Mercator.
        da_hmax_mercator = da_hmax.rio.reproject(3857)

        # 3. Create the plot axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # 4. Get the spatial bounds and data from the reprojected raster.
        minx, miny, maxx, maxy = da_hmax_mercator.rio.bounds()
        data_to_plot = da_hmax_mercator.to_numpy()

        # 5. Set the axis limits.
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # 6. Add the basemap FIRST.
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=1)

        # 7. Use ax.imshow() to plot the data ON TOP.
        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),
            cmap=cmocean.cm.dense,
            vmin=0,
            vmax=0.25,
            alpha=0.7,
            interpolation='none',
            zorder=10
        )

        ax.text(
                0.05,
                0.95,
                f"Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_array))) * 0.0025, 2)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            )
        
        # 8. Manually create a colorbar.
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
        cbar.set_label('Depth (m)')
        
        # 9. Clean up and save
        png_path = Path(output_folder) / "flood_map_final.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)

        ax.set_title('')
        ax.set_axis_off()
        plt.tight_layout()

        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Plot created successfully and saved to {png_path}.")

    def process_flood_events(self, plotting_dir):
        """
        Processes and generates plots for flood events.

        This method processes each flood event by loading virtual sensor depths and water level data,
        and then generates depth map and water level plots for each event. The plots are saved in a specified
        directory within each flood event's folder.

        Parameters:
        -----------
        plotting_dir : str
            Directory where the generated plots will be saved within each flood event's folder.

        Returns:
        --------
        None

        Notes:
        ------
        The method iterates over all available flood event folders, loads relevant depth and water level data,
        and calls the `plot_depth_map_and_wtr_levels` method to generate the plots. The resulting figures are
        saved to a subdirectory within the flood event folder.

        The method assumes the following directory structure:
        - Each flood event has a directory containing depth map and image data in Zarr format.
        - A CSV file (`wtr_lvl_obs_to_image_matches.csv`) containing observed water levels is present in each flood event's folder.
        - A pre-defined plotting directory structure will be created for saving the plots.

        This method handles large datasets by deleting unnecessary variables after each event and performing garbage collection.
        """

        flood_event_folders = self.list_flood_event_folders()
        
        self.preprocess_flood_events()

        for flood_event in tqdm(
            flood_event_folders, desc="Plotting flood events...", unit="event"
        ):

            flood_event_path = os.path.join(self.main_dir, flood_event)

            datetimes, max_depths, avg_depths, vs_depths = (
                self.load_virtual_sensor_depths(flood_event_path)
            )

            plotting_folder = os.path.join(flood_event_path, "plots", plotting_dir)
            os.makedirs(plotting_folder, exist_ok=True)

            depth_maps_zarr_dir = os.path.join(
                flood_event_path, "zarr", "wse_maps_95th_ponding"
            )
            orig_image_rects_zarr_dir = os.path.join(
                flood_event_path, "zarr", "orig_image_rects"
            )

            obs_to_img_matches = pd.read_csv(
                os.path.join(flood_event_path, "wtr_lvl_obs_to_image_matches.csv")
            )
            obs_to_img_matches["closest_utc_time"] = pd.to_datetime(
                obs_to_img_matches["closest_utc_time"], utc=True
            )

            self.plot_depth_map_and_wtr_levels(
                depth_maps_zarr_dir,
                orig_image_rects_zarr_dir,
                datetimes,
                obs_to_img_matches,
                vs_depths,
                plotting_folder,
            )

            del (
                datetimes,
                max_depths,
                avg_depths,
                vs_depths,
                obs_to_img_matches,
            )  # Delete large variables
            gc.collect()  # Force garbage collection

    def process_flood_events_HPC(self):
        """
        Processes and generates plots for flood events in parallel using MPI.

        This method utilizes MPI for parallel processing to efficiently process and generate plots for flood events.
        The work is divided among multiple processes, where each process handles a subset of flood event folders.
        The plots are saved in a specified directory within each flood event's folder.

        Parameters:
        -----------
        plotting_dir : str
            Directory where the generated plots will be saved within each flood event's folder.

        Returns:
        --------
        None

        Notes:
        ------
        - The method initializes MPI communication and splits the processing of flood events among available processes.
        - Only the master process (rank 0) will list the flood event folders and preprocess the events.
        - The flood event folders are then broadcast to all processes, and each process handles a specific chunk of the data.
        - Each process calls `process_single_flood_event` to process a subset of flood events and generate corresponding plots.
        - This method is designed to run in a distributed parallel environment using MPI to speed up the processing.

        Assumptions:
        -------------
        - MPI is correctly initialized in the environment where this method is executed.
        - The `list_flood_event_folders` and `preprocess_flood_events` methods are available for use in the master process.
        - The `process_single_flood_event` method handles the processing and plotting for individual flood events.

        This method is intended for high-performance computing (HPC) environments where multiple cores or nodes are available.
        """
        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only the master process will list flood event folders
        if rank == 0:
            flood_event_folders = self.list_flood_event_folders()
            self.preprocess_flood_events()
        else:
            flood_event_folders = None

        # Broadcast the flood event folders to all processes
        flood_event_folders = comm.bcast(flood_event_folders, root=0)

        # Split the work among processes
        n_folders = len(flood_event_folders)
        chunk_size = n_folders // size
        start_index = rank * chunk_size
        end_index = start_index + chunk_size if rank != size - 1 else n_folders

        # Process only the assigned folders
        for flood_event in tqdm(
            flood_event_folders[start_index:end_index],
            desc="Plotting flood events...",
            unit="event",
        ):
            self.process_single_flood_event(flood_event)


    def plot_flood_wse_map(self, depth_array_path, min_x, max_x, min_y, max_y,
                                    output_filename, resolution_m=0.05, 
                                    bbox_crs='EPSG:32119', output_folder='figures'):
        """
        Plots a flood numpy array, correcting for a "bottom-left" array origin by
        flipping the array vertically before georeferencing.

        Args:
            depth_array_path (str): The path to the input Zarr array.
            min_x, max_x, min_y, max_y (float): Bounding box coordinates.
            output_filename (str): The desired name for the output PNG file.
            ... (other args are the same) ...
        """
        array_store = zarr.open(depth_array_path, mode="r")
        depth_array = array_store[:]
        print(f"    -> DEBUG: Number of valid (non-NaN) data points: {np.count_nonzero(~np.isnan(depth_array))}")


        # 1. Build and georeferenced the DataArray.
        data_for_xarray = np.flipud(depth_array).astype(float)

        transform = Affine(resolution_m, 0.0, min_x, 0.0, -resolution_m, max_y)
        da_hmax = xr.DataArray(
            data=data_for_xarray,
            dims=["y", "x"],
            name='flood_depth'
        )
        da_hmax.rio.write_crs(bbox_crs, inplace=True)
        da_hmax.rio.write_transform(transform, inplace=True)
        da_hmax.rio.write_nodata(np.nan, inplace=True)

        # 2. Reproject the raster to Web Mercator.
        da_hmax_mercator = da_hmax.rio.reproject(3857)

        # 3. Create the plot axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # 4. Get the spatial bounds and data from the reprojected raster.
        minx, miny, maxx, maxy = da_hmax_mercator.rio.bounds()
        data_to_plot = da_hmax_mercator.to_numpy()

        # 5. Set the axis limits.
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # 6. Add the basemap FIRST.
        #ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=1)
        
        vmin=np.nanmin(data_to_plot)
        vmax=np.nanmax(data_to_plot)
        
        # 7. Use ax.imshow() to plot the data ON TOP.
        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),
            cmap='Blues',
            alpha=0.7,
            interpolation='none',
            zorder=10
        )

        ax.text(
                0.05,
                0.95,
                f"Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_array))) * 0.0025, 2)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            )
        
        # 8. Manually create a colorbar.
        norm = Normalize(vmin=np.nanmin(data_to_plot), vmax=np.nanmax(data_to_plot))
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='Blues'), ax=ax, shrink=0.6, aspect=30)
        cbar.set_label('Water Surface Elevation (m)')
        
        # 9. Clean up and save
        # --- FIX: Use the dynamic output_filename parameter ---
        png_path = Path(output_folder) / output_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)

        ax.set_title('')
        ax.set_axis_off()
        plt.tight_layout()

        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Plot created successfully and saved to {png_path}.")
    
    def plot_flood_depth_map(self, depth_array_path, min_x, max_x, min_y, max_y,
                                output_filename, resolution_m=0.05, 
                                bbox_crs='EPSG:32119', output_folder='figures'):
        """
        Plots a flood numpy array, correcting for a "bottom-left" array origin by
        flipping the array vertically before georeferencing.

        Args:
            depth_array_path (str): The path to the input Zarr array.
            min_x, max_x, min_y, max_y (float): Bounding box coordinates.
            output_filename (str): The desired name for the output PNG file.
            ... (other args are the same) ...
        """
        array_store = zarr.open(depth_array_path, mode="r")
        depth_array = array_store[:]

        print(f"    -> DEBUG: Number of valid (non-NaN) data points: {np.count_nonzero(~np.isnan(depth_array))}")

        # 1. Build and georeferenced the DataArray.
        data_for_xarray = np.flipud(depth_array).astype(float)

        transform = Affine(resolution_m, 0.0, min_x, 0.0, -resolution_m, max_y)
        da_hmax = xr.DataArray(
            data=data_for_xarray,
            dims=["y", "x"],
            name='flood_depth'
        )
        da_hmax.rio.write_crs(bbox_crs, inplace=True)
        da_hmax.rio.write_transform(transform, inplace=True)
        da_hmax.rio.write_nodata(np.nan, inplace=True)
        
        # Downsample the array by a factor of 2 in each dimension before reprojecting.
        # This reduces the total number of pixels by 75%, making reprojection MUCH faster.
        # Adjust the factor as needed; 4 would be even faster.
        coarsen_factor = 2 
        da_hmax_coarse = da_hmax.coarsen(x=coarsen_factor, y=coarsen_factor, boundary="trim").mean()
        print(f"    -> Coarsened array shape: {da_hmax_coarse.shape}", flush=True)

        # 2. Reproject the SMALLER, coarsened array. (THIS WILL BE FAST)
        da_hmax_mercator = da_hmax_coarse.rio.reproject(3857)

        # 2. Reproject the raster to Web Mercator.
        #da_hmax_mercator = da_hmax.rio.reproject(3857)

        # 3. Create the plot axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # 4. Get the spatial bounds and data from the reprojected raster.
        minx, miny, maxx, maxy = da_hmax_mercator.rio.bounds()
        data_to_plot = da_hmax_mercator.to_numpy()

        # 5. Set the axis limits.
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # 6. Add the basemap FIRST.
        #ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=1)

        vmin = 0
        vmax = 0.25
        
        # 7. Use ax.imshow() to plot the data ON TOP.
        im = ax.imshow(
            data_to_plot,
            extent=(minx, maxx, miny, maxy),
            cmap=cmocean.cm.dense,
            vmin=vmin,
            vmax=vmax,
            alpha=0.7,
            interpolation='none',
            zorder=10
        )

        ax.text(
                0.05,
                0.95,
                f"Spatial Extent ($m^2$): {round((np.sum(~np.isnan(depth_array))) * 0.0025, 2)}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            )
        
        # 8. Manually create a colorbar.
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=30)
        cbar.set_label('Depth (m)')

        
        # 9. Clean up and save
        # --- FIX: Use the dynamic output_filename parameter ---
        png_path = Path(output_folder) / output_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)

        ax.set_title('')
        ax.set_axis_off()
        plt.tight_layout()

        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Plot created successfully and saved to {png_path}.")

    def process_single_flood_event(self, flood_event):
        """
        Processes a single flood event and generates corresponding plots.
        """
        flood_event_path = os.path.join(self.main_dir, flood_event)

        try:
            datetimes, _, _, _ = self.load_virtual_sensor_depths(flood_event_path)
        except FileNotFoundError:
            print(f"WARNING: Could not load virtual sensor depths for {flood_event}. Skipping.", flush=True)
            return

        depth_maps_zarr_dir = os.path.join(
            flood_event_path, "zarr", "depth_maps"
        )

        if not os.path.isdir(depth_maps_zarr_dir):
            print(f"ERROR: Directory not found at '{depth_maps_zarr_dir}'", flush=True)
            return

        # List the contents of the directory
        try:
            contents = os.listdir(depth_maps_zarr_dir)
        except OSError as e:
            print(f"ERROR: Could not list contents of '{depth_maps_zarr_dir}': {e}", flush=True)
            return

        # Loop through every item found in the directory
        for zarr_name in contents:
            full_zarr_path = os.path.join(depth_maps_zarr_dir, zarr_name)

            # --- FIX 1: Explicitly check if the item is a directory (Zarr stores are directories) ---
            if not os.path.isdir(full_zarr_path):
                # This will correctly skip files like 'zarr.json'
                continue

            try:
                output_png_filename = f"{zarr_name}.png"

                if 'wse' in zarr_name:
                    print(f" -> Processing WSE file: {zarr_name}", flush=True)
                    plotting_dir = 'WSE_maps'
                    plotting_folder = os.path.join(flood_event_path, "plots", plotting_dir)

                    self.plot_flood_wse_map(
                        full_zarr_path,
                        self.min_x_extent, self.max_x_extent,
                        self.min_y_extent, self.max_y_extent,
                        output_png_filename,
                        bbox_crs=self.bbox_crs,
                        output_folder=plotting_folder
                    )

                elif 'depth' in zarr_name:
                    print(f" -> Processing DEPTH file: {zarr_name}", flush=True)
                    plotting_dir = 'depth_maps'
                    plotting_folder = os.path.join(flood_event_path, "plots", plotting_dir)

                    self.plot_flood_depth_map(
                        full_zarr_path,
                        self.min_x_extent, self.max_x_extent,
                        self.min_y_extent, self.max_y_extent,
                        output_png_filename,
                        bbox_crs=self.bbox_crs,
                        output_folder=plotting_folder
                    )

            except Exception as e:
                # --- FIX 2: Use flush=True to guarantee the error message is printed immediately ---
                print(f"   -> An ERROR occurred while processing {zarr_name}: {e}", flush=True)
