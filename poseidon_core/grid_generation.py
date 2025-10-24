import os
import time
import zarr
import laspy
import numpy as np
from scipy.interpolate import griddata


class GridGenerator:
    """A class for generating and processing LiDAR data grids.

    This class allows for loading LiDAR data from a specified file, creating point 
    arrays based on classification values, and generating grids of points in specified 
    spatial extents and resolutions. The class also provides methods for saving the 
    generated grids as compressed Zarr files.

    Attributes:
        filepath (str): The path to the LiDAR data file.
        filename (str): The name of the LiDAR data file.
        lidar (laspy.LasData): The loaded LiDAR data.
        min_x_extent (float): The minimum x-coordinate extent for grid generation.
        max_x_extent (float): The maximum x-coordinate extent for grid generation.
        min_y_extent (float): The minimum y-coordinate extent for grid generation.
        max_y_extent (float): The maximum y-coordinate extent for grid generation.
        point_mask_val (int): The classification value used to filter points.

    Methods:
        load_lidar(): Load LiDAR data from the specified file.
        create_point_array(point_mask_value=2): Create an array of filtered LiDAR points
            based on classification.
        gen_grid(...): Generates a grid using an efficient PDAL pipeline.
    """

    def __init__(
        self,
        file_path,
        min_x_extent=None,
        max_x_extent=None,
        min_y_extent=None,
        max_y_extent=None,
        extent_units='meters',
        lidar_units='feet'
    ):
        """Initialize the LiDAR processing object.

        This constructor initializes the LiDAR processing object by loading the LiDAR 
        data from the specified file path. It sets the spatial extents for the LiDAR 
        data, which can be specified or automatically determined from the loaded data.

        Args:
            file_path (str): The path to the LiDAR data file to be loaded.
            min_x_extent (float, optional): The minimum x-coordinate grid extent.
                                            Defaults to None, which will set it to the 
                                            minimum x-coordinate of the LiDAR data 
                                            converted to meters.
            max_x_extent (float, optional): The maximum x-coordinate grid extent.
                                            Defaults to None, which will set it to the 
                                            maximum x-coordinate of the LiDAR data 
                                            converted to meters.
            min_y_extent (float, optional): The minimum y-coordinate grid extent.
                                            Defaults to None, which will set it to the 
                                            minimum y-coordinate of the LiDAR data 
                                            converted to meters.
            max_y_extent (float, optional): The maximum y-coordinate grid extent.
                                            Defaults to None, which will set it to the 
                                            maximum y-coordinate of the LiDAR data 
                                            converted to meters.
        """
        self.filepath = file_path
        self.filename = os.path.basename(file_path)
        self.extent_units = extent_units
        self.lidar_units = lidar_units
        self.lidar = self.load_lidar()

        # Determine extents, converting from feet to meters if provided extent units
        # are in feet. Otherwise, they are assumed to be in meters.
        if extent_units == 'feet':
            # If extent provided in feet, convert matching create_point_array logic
            scale_factor = 0.3048
            self.min_x_extent = np.min(self.lidar.x) * scale_factor
            self.max_x_extent = np.max(self.lidar.x) * scale_factor
            self.min_y_extent = np.min(self.lidar.y) * scale_factor
            self.max_y_extent = np.max(self.lidar.y) * scale_factor
        else:
            # If extent units are not feet, assume they are already in meters
            self.min_x_extent = min_x_extent
            self.max_x_extent = max_x_extent
            self.min_y_extent = min_y_extent
            self.max_y_extent = max_y_extent

    def load_lidar(self):
        """Load LiDAR data from the specified file.

        This function reads the LiDAR data from the file located at `self.filepath`
        and returns the LiDAR object for further processing.

        Returns:
            laspy.LasData: The LiDAR data loaded from the file.
        """
        lidar = laspy.read(self.filepath)

        return lidar

    def create_point_array(self, point_mask_value=2):
        """Create an array of LiDAR points based on classification.

        This function filters the LiDAR data to select points that match the
        specified classification value, converts their coordinates from feet to meters,
        and applies extent filtering based on predefined minimum and maximum extents.

        Args:
            orig_units (str, optional): The original units of the LAS file ('feet' or 
                'meters'). Defaults to 'feet'.
            point_mask_value (int, optional): The classification value to filter points.
                Defaults to 2.

        Returns:
            np.ndarray: An array of filtered LiDAR points in meters, within the 
                specified extents.
        """
        self.point_mask_val = point_mask_value

        # Mask points based on classification
        point_mask = (
            self.lidar.classification == point_mask_value
        )  

        if self.lidar_units == "feet":
            # Convert feet to meters and stack into an array
            xyz_m = np.vstack(
                [
                    self.lidar.x[point_mask] * 0.3048,
                    self.lidar.y[point_mask] * 0.3048,
                    self.lidar.z[point_mask] * 0.3048,
                ]
            )
        else:
            # Data is already in meters
            xyz_m = np.vstack(
                [
                    self.lidar.x[point_mask],
                    self.lidar.y[point_mask],
                    self.lidar.z[point_mask],
                ]
            )

        # Apply extent filtering
        # Assumes self.extents are in meters
        extent_mask = (
            (xyz_m[0] >= self.min_x_extent)
            & (xyz_m[0] <= self.max_x_extent)
            & (xyz_m[1] >= self.min_y_extent)
            & (xyz_m[1] <= self.max_y_extent)
        )

        return xyz_m[:, extent_mask]  # Keep only the points within the extents

    def gen_grid(self, resolution, z=0, dir="data/generated_grids", grid_descriptor=""):
        """Generate a grid of points in the specified extent and resolution.

        This function creates a grid based on the specified resolution and z-value. It 
        can generate a uniform grid at a specified elevation (z) or interpolate a set of
        points provided in a 2D array. The generated grid arrays are saved as compressed
        Zarr files in the specified directory.

        Args:
            resolution (float): The spacing between grid points in x and y dimensions.
            z (int, float, np.ndarray, optional): The elevation value to use for the z 
                dimension. Defaults to 0.
            dir (str, optional): The directory where the generated grid files will be 
                saved. Defaults to 'generated_grids'.
            grid_descriptor (str, optional): A unique name to prepend to the
                output filenames (e.g., 'ground_points'). Defaults to "".

        Returns:
            tuple: A tuple containing three numpy arrays:
                - grid_x (np.ndarray): The x-coordinates of the grid points.
                - grid_y (np.ndarray): The y-coordinates of the grid points.
                - grid_z (np.ndarray): The z-coordinates of the grid points.
        """
        # --- Start Logging ---
        print(f"\n--- Starting grid generation for '{grid_descriptor}' at {resolution}m ---")
        start_time = time.time()

        # Check that the path exists
        if not os.path.exists(dir):
            os.makedirs(dir)  # Create the directory if not
            print(f"  [IO] Directory created: {dir}")
        else:
            print(f"  [IO] Using existing directory: {dir}")
        
        # Generate structured grid for extents and resolution
        print(f"  [GRID] Generating grid coordinates...")
        # Using :.2f to format floats to 2 decimal places for cleaner logs
        print(f"    X Extent: {self.min_x_extent:.2f} to {self.max_x_extent:.2f}")
        print(f"    Y Extent: {self.min_y_extent:.2f} to {self.max_y_extent:.2f}")
        grid_x, grid_y = np.mgrid[
            self.min_x_extent : self.max_x_extent : resolution,
            self.min_y_extent : self.max_y_extent : resolution,
        ]
        print(f"    Initial grid shape (before transpose): {grid_x.shape}")

        # --- Z-Grid Generation Logic ---
        
        # If int or float provided for z, create set elevation grid
        if isinstance(z, int) or isinstance(z, float):
            print(f"  [GRID] Creating flat Z grid with constant value: {z}")
            grid_z = np.full_like(grid_x, z)

        # If points array is provided, fit points to structured grid
        elif isinstance(z, np.ndarray):
            print(f"  [GRID] Interpolating Z grid from {z.shape[1]} input points...")
            x = z[0]
            y = z[1]
            z_vals = z[2] # Use a different variable name to avoid confusion
            grid_z = griddata((x, y), z_vals, (grid_x, grid_y), method="linear")
            
            # Check for NaNs, which can happen if interpolation fails
            nan_count = np.count_nonzero(np.isnan(grid_z))
            if nan_count > 0:
                print(f"    [WARN] {nan_count} grid points were outside the "
                      f"interpolation area (set to NaN).")
                # You might want to fill these NaNs, e.g.:
                # grid_z = np.nan_to_num(grid_z, nan=-9999.0)
                # print("    [INFO] NaN values filled with -9999.0")
        
        # Add a check for bad input
        else:
            print(f"  [ERROR] Z parameter is of an unsupported type: {type(z)}")
            raise TypeError(f"Z parameter must be int, float, or np.ndarray, not {type(z)}")

        # --- Transpose ---
        print(f"  [GRID] Transposing grids to (Y, X) convention.")
        grid_x = grid_x.T
        grid_y = grid_y.T
        grid_z = grid_z.T
        print(f"    Final grid shape: {grid_x.shape}")

        # --- Save to Zarr ---
        
        # Handle the descriptor string to make sure filename is clean
        if grid_descriptor and not grid_descriptor.endswith('_'):
            grid_descriptor_str = f"{grid_descriptor}_"
        elif not grid_descriptor:
             grid_descriptor_str = "" # Empty string if none provided
        else:
            grid_descriptor_str = grid_descriptor # Use as-is if it ends in '_'

        # Define file paths
        path_x = os.path.join(dir, f"{grid_descriptor_str}grid_x_{resolution}m.zarr")
        path_y = os.path.join(dir, f"{grid_descriptor_str}grid_y_{resolution}m.zarr")
        path_z = os.path.join(dir, f"{grid_descriptor_str}grid_z_{resolution}m.zarr")
        
        print(f"  [IO] Saving compressed Zarr arrays (mode='w', overwriting)...")
        
        print(f"    X -> {path_x}")
        zarr.open(path_x, mode='w', shape=grid_x.shape, dtype=grid_x.dtype, 
                  chunks=True)[:] = grid_x
        
        print(f"    Y -> {path_y}")
        zarr.open(path_y, mode='w', shape=grid_y.shape, dtype=grid_y.dtype, 
                  chunks=True)[:] = grid_y
        
        print(f"    Z -> {path_z}")
        zarr.open(path_z, mode='w', shape=grid_z.shape, dtype=grid_z.dtype, 
                  chunks=True)[:] = grid_z

        # --- Finish Logging ---
        end_time = time.time()
        print(f"--- Grid generation successful in {end_time - start_time:.2f} seconds. " 
              f"---")

        return grid_x, grid_y, grid_z
