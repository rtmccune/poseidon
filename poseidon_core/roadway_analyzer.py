import os
import gc
import zarr
import numpy as np
import pandas as pd
import poseidon_core
from tqdm import tqdm
from mpi4py import MPI

class RoadwayAnalyzer:
    
    def __init__(self, main_dir, virtual_sensor_locations):

        self.main_dir = main_dir
        self.virtual_sensor_loc = virtual_sensor_locations

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
    
    def gen_transect_depths(self, flood_event_path):

        depth_maps_zarr_dir = os.path.join(
            flood_event_path, "zarr", "depth_maps"
        )
        output_zarr_store = os.path.join(
            flood_event_path, "zarr", "roadway_transect_depths"
        )

        timestamp_list = []

        if os.path.exists(depth_maps_zarr_dir):
            file_names = [
                f for f in os.listdir(depth_maps_zarr_dir) if f.endswith("_95_perc")
            ]
            num_files = len(file_names)
            print(num_files)

            # Preallocate NumPy arrays for better performance

            transect_depth_array = np.empty(
                (num_files, len(self.virtual_sensor_loc)), dtype=np.float32
            )

            for idx, file_name in enumerate(file_names):
                timestamp = image_processing.image_utils.extract_timestamp(file_name)
                timestamp_list.append(timestamp)

                file_zarr_store = os.path.join(depth_maps_zarr_dir, file_name)
                img_store = zarr.open(file_zarr_store, mode="r")
                depth_map = img_store[:]

                for i, (x, y) in enumerate(self.virtual_sensor_loc):
                    transect_depth_array[idx, i] = depth_map[y, x]

            # Convert timestamps to a NumPy array of strings
            datetimes = np.array(
                pd.to_datetime(timestamp_list, utc=True).astype(str), dtype="U"
            )

            # Save to a Zarr store
            root = zarr.open_group(
                output_zarr_store, mode="w"
            )  # Overwrite existing store

            # root.create_array("timestamps", shape=datetimes.shape, dtype="U")
            # root["timestamps"][:] = datetimes  # Assign data
            root.create_array("timestamps", data=datetimes)
            
            root.create_array("roadway_transect_depths", shape=transect_depth_array.shape, dtype=np.float32)
            root["roadway_transect_depths"][:] = transect_depth_array
            
    def preprocess_flood_events(self):

        flood_event_folders = self.list_flood_event_folders()

        for flood_event in tqdm(
            flood_event_folders,
            desc="Preprocessing flood events for plotting...",
            unit="event",
        ):

            flood_event_path = os.path.join(self.main_dir, flood_event)
            self.gen_transect_depths(flood_event_path)
    
    
    def load_transect_depths(self, flood_event_path):

        zarr_store_path = os.path.join(
            flood_event_path, "zarr", "roadway_transect_depths"
        )

        if os.path.exists(zarr_store_path):
            root = zarr.open(zarr_store_path, mode="r")

            timestamps = root["timestamps"][:]  # Load as an array of strings
            transect_depths = root["roadway_transect_depths"][:]

            # Convert timestamps back to pandas datetime
            datetimes = pd.to_datetime(timestamps, utc=True)

            return datetimes, transect_depths
        else:
            return None, None
            #raise FileNotFoundError(f"Zarr store not found: {zarr_store_path}")
        
    def process_roadway_accessibility(self):
        
        flood_event_folders = self.list_flood_event_folders()
        
        
        for flood_event in tqdm(
            flood_event_folders, desc="Plotting flood events...", unit="event"
            ):
            
            flood_event_path = os.path.join(self.main_dir, flood_event)
            datetimes, transect_depths = self.load_transect_depths(flood_event_path)
            
            # impassable = np.any(transect_depths > 0, axis=1).astype(int)  # convert True/False to 1/0
            if transect_depths is None:
                print(f"Warning: Transect data not found for '{flood_event}'. Skipping.")
                return  # Exit the method for this event
            
            # Calculate impassable status (1 if any depth > 0, else 0)
            impassable = np.any(transect_depths > 0, axis=1).astype(int)

            # NEW: Calculate depth statistics for each timestep
            mean_depths = np.nanmean(transect_depths, axis=1)
            median_depths = np.nanmedian(transect_depths, axis=1)
            max_depths = np.nanmax(transect_depths, axis=1)
            min_depths = np.nanmin(transect_depths, axis=1)

            # Create DataFrame with the original and new columns
            roadway_stats_df = pd.DataFrame({
                'Time': datetimes,
                'Impassable': impassable,
                'MeanDepth': mean_depths,
                'MedianDepth': median_depths,
                'MaxDepth': max_depths,
                'MinDepth': min_depths
            })

            # Ensure 'Time' is a datetime object and sort the DataFrame
            roadway_stats_df['Time'] = pd.to_datetime(roadway_stats_df['Time'])
            roadway_stats_df = roadway_stats_df.sort_values(by='Time')
            
            # Save the DataFrame to a CSV file
            output_path = os.path.join(self.main_dir, flood_event, 'roadway_accessibility_time_series.csv')
            roadway_stats_df.to_csv(output_path, index=False)
            # # Create DataFrame
            # impassable_time_series = pd.DataFrame({
            #     'Time': datetimes,
            #     'Impassable': impassable
            # })
            
            # impassable_time_series['Time'] = pd.to_datetime(impassable_time_series['Time'])
            # impassable_time_series = impassable_time_series.sort_values(by='Time')
            
            # impassable_time_series.to_csv(os.path.join(self.main_dir, flood_event, 'roadway_accessibility_time_series.csv'))
        
    def process_roadway_access_HPC(self):
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
            desc="Calculating roadway access...",
            unit="event",
        ):
            self.process_roadway_accessibility()
