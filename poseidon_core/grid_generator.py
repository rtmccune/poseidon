import os
import time
import zarr
import laspy
import cmocean
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap


class GridGenerator:
    """
    Generates and processes 2D/3D grids from LiDAR point cloud data.

    This class handles loading a LiDAR (.las or .laz) file, filtering
    points by classification, and generating structured 2D or 3D grids
    (X, Y, Z) based on specified extents and resolution. Grids are
    saved to disk as compressed Zarr arrays.

    Parameters
    ----------
    file_path : str
        The path to the LiDAR data file (.las or .laz).
    min_x_extent : float, optional
        The minimum x-coordinate grid extent (in `extent_units`).
        If None, defaults to the min X of the LiDAR data.
    max_x_extent : float, optional
        The maximum x-coordinate grid extent (in `extent_units`).
        If None, defaults to the max X of the LiDAR data.
    min_y_extent : float, optional
        The minimum y-coordinate grid extent (in `extent_units`).
        If None, defaults to the min Y of the LiDAR data.
    max_y_extent : float, optional
        The maximum y-coordinate grid extent (in `extent_units`).
        If None, defaults to the max Y of the LiDAR data.
    extent_units : {'meters', 'feet'}, optional
        The units of the provided extent parameters.
        Default is 'meters'.
    lidar_units : {'feet', 'meters'}, optional
        The source units of the LiDAR file's coordinates.
        Default is 'feet'.
        This is used for conversion by `create_point_array`.

    Attributes
    ----------
    filepath : str
        The path to the LiDAR data file.
    filename : str
        The name of the LiDAR data file.
    lidar : laspy.LasData
        The loaded LiDAR data object.
    min_x_extent : float
        The minimum x-coordinate extent for grid generation (in meters).
    max_x_extent : float
        The maximum x-coordinate extent for grid generation (in meters).
    min_y_extent : float
        The minimum y-coordinate extent for grid generation (in meters).
    max_y_extent : float
        The maximum y-coordinate extent for grid generation (in meters).
    point_mask_val : int
        The classification value used to filter points (set by
        `create_point_array`).

    Notes
    -----
    The typical workflow is:
    1. Initialize the class with a `file_path`.
    2. Call `create_point_array()` for a filtered set of ground points.
    3. Pass the result of (2) to `gen_grid()` to interpolate a Z grid.
    Alternatively, call `gen_grid(z=0)` to create a flat grid.
    """

    def __init__(
        self,
        file_path,
        min_x_extent=None,
        max_x_extent=None,
        min_y_extent=None,
        max_y_extent=None,
        extent_units="meters",
        lidar_units="feet",
    ):
        """Initialize the GridGenerator.

        Parameters
        ----------
        file_path : str
            The path to the LiDAR data file (.las or .laz).
        min_x_extent : float, optional
            Minimum x-coordinate grid extent (in `extent_units`).
            Defaults to None (autodetect from LiDAR file).
        max_x_extent : float, optional
            Maximum x-coordinate grid extent (in `extent_units`).
            Defaults to None (autodetect from LiDAR file).
        min_y_extent : float, optional
            Minimum y-coordinate grid extent (in `extent_units`).
            Defaults to None (autodetect from LiDAR file).
        max_y_extent : float, optional
            Maximum y-coordinate grid extent (in `extent_units`).
            Defaults to None (autodetect from LiDAR file).
        extent_units : {'meters', 'feet'}, optional
            Units of the provided extent parameters.
            Default is 'meters'.
        lidar_units : {'feet', 'meters'}, optional
            Source units of the LiDAR file's coordinates.
            Default is 'feet'.
        """
        self.filepath = file_path
        self.filename = os.path.basename(file_path)
        self.extent_units = extent_units
        self.lidar_units = lidar_units
        self.lidar = self._load_lidar()

        # Determine extents, converting from feet to meters if provided
        # extent units are in feet. Otherwise, they are assumed to be in
        # meters.
        if extent_units == "feet":
            # If extent provided in feet, convert matching
            # the create_point_array logic
            scale_factor = 0.3048
            self.min_x_extent = np.min(self.lidar.x) * scale_factor
            self.max_x_extent = np.max(self.lidar.x) * scale_factor
            self.min_y_extent = np.min(self.lidar.y) * scale_factor
            self.max_y_extent = np.max(self.lidar.y) * scale_factor
        else:
            # If extent units are not feet, assume they are in meters
            self.min_x_extent = min_x_extent
            self.max_x_extent = max_x_extent
            self.min_y_extent = min_y_extent
            self.max_y_extent = max_y_extent

    def create_point_array(self, point_mask_value=2):
        """Create a filtered array of LiDAR points.

        Filters the LiDAR data by classification, applies extent
        filtering (based on `self.min/max_x/y_extent`), and converts
        coordinates to meters if `self.lidar_units` is 'feet'.

        Parameters
        ----------
        point_mask_value : int, optional
            The classification value to filter points. Default is 2
            (typically "ground").

        Returns
        -------
        np.ndarray
            A (3, N) array of filtered [X, Y, Z] LiDAR points in meters,
            within the specified extents.

        Notes
        -----
        This method also sets the `self.point_mask_val` attribute.
        """
        self.point_mask_val = point_mask_value

        # Mask points based on classification
        point_mask = self.lidar.classification == point_mask_value

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

        # Keep only the points within the extents
        return xyz_m[:, extent_mask]

    def gen_grid(
        self, resolution, z=0, dir="data/generated_grids", grid_descriptor=""
    ):
        """Generate and save a grid of points.

        Creates X, Y, and Z grids based on the class extents and a given
        resolution. The Z grid is either a flat plane
        (if `z` is a number) or an interpolated surface
        (if `z` is a point array).

        The generated grids are saved as compressed Zarr files.

        Parameters
        ----------
        resolution : float
            The spacing (in meters) between grid points in x and y
            dimensions.
        z : int, float, or np.ndarray, optional
            The elevation data.
            - If int or float: Creates a flat grid at that constant Z
            value.
            - If np.ndarray (shape 3,N): Interpolates Z values from
            these [X, Y, Z] points.
            Default is 0.
        dir : str, optional
            Directory to save the generated Zarr grid files.
            Default is 'data/generated_grids'.
        grid_descriptor : str, optional
            A prefix for the output filenames (e.g., 'ground_points').
            Default is "".

        Returns
        -------
        grid_x : np.ndarray
            The 2D array of grid x-coordinates
            (transposed to Y, X shape).
        grid_y : np.ndarray
            The 2D array of grid y-coordinates
            (transposed to Y, X shape).
        grid_z : np.ndarray
            The 2D array of grid z-coordinates
            (transposed to Y, X shape).

        Notes
        -----
        The grids are generated using `np.mgrid` and then transposed to
        have a (Y, X) shape. If `z` is an array,
        `scipy.interpolate.griddata` is used with a 'linear' method.
        """
        # --- Start Logging ---
        print(
            f"\n--- Starting grid generation for '{grid_descriptor}' at "
            f"{resolution}m ---"
        )
        start_time = time.time()

        self._prepare_output_dir(dir)
        grid_x, grid_y = self._generate_grid_coordinates(resolution)
        grid_z = self._generate_z_grid(z, grid_x, grid_y)
        grid_x, grid_y, grid_z = self._transpose_grids(grid_x, grid_y, grid_z)

        self._save_grids_to_zarr(
            grid_x, grid_y, grid_z, resolution, dir, grid_descriptor
        )

        # --- Finish Logging ---
        end_time = time.time()
        print(
            f"--- Grid generation successful in {end_time - start_time:.2f} "
            f"seconds. ---"
        )

        return grid_x, grid_y, grid_z

    def _load_lidar(self):
        """Load LiDAR data from the `self.filepath` using laspy.

        Returns
        -------
        laspy.LasData
            The LiDAR data loaded from the file.
        """
        lidar = laspy.read(self.filepath)
        return lidar

    def _prepare_output_dir(self, dir):
        """
        Ensure the output directory exists, creating it if necessary.

        Parameters
        ----------
        dir : str
            The directory path to check and/or create.
        """
        # Check that the path exists
        if not os.path.exists(dir):
            os.makedirs(dir)  # Create the directory if not
            print(f"  [IO] Directory created: {dir}")
        else:
            print(f"  [IO] Using existing directory: {dir}")

    def _generate_grid_coordinates(self, resolution):
        """
        Generate 2D X and Y coordinate grids based on class extents.

        Parameters
        ----------
        resolution : float
            The spacing (in meters) for the grid.

        Returns
        -------
        grid_x : np.ndarray
            2D array of x-coordinates, with shape (X_steps, Y_steps).
        grid_y : np.ndarray
            2D array of y-coordinates, with shape (X_steps, Y_steps).

        Notes
        -----
        Uses `np.mgrid`, which generates coordinates from `start` up to
        (but not including) `stop`, with `step` = `resolution`.
        The initial shape is (X, Y) and must be transposed later to
        match a (Row, Col) or (Y, X) convention.
        """
        # Generate structured grid for extents and resolution
        print("  [GRID] Generating grid coordinates...")
        # Using :.2f to format floats to 2 decimal places for clean logs
        print(
            f"    X Extent: {self.min_x_extent:.2f} to {self.max_x_extent:.2f}"
        )
        print(
            f"    Y Extent: {self.min_y_extent:.2f} to {self.max_y_extent:.2f}"
        )
        grid_x, grid_y = np.mgrid[
            self.min_x_extent : self.max_x_extent : resolution,
            self.min_y_extent : self.max_y_extent : resolution,
        ]
        print(f"    Initial grid shape (before transpose): {grid_x.shape}")

        return grid_x, grid_y

    def _generate_z_grid(self, z, grid_x, grid_y):
        """
        Generate the 2D Z grid, either as a flat plane or interpolated.

        If `z` is a scalar, a flat grid is created.
        If `z` is a (3, N) array, `scipy.interpolate.griddata` is used
        to interpolate Z values onto the grid.

        Parameters
        ----------
        z : int, float, or np.ndarray
            Source elevation data.
            - If int or float: Creates a flat grid at that constant Z
              value.
            - If np.ndarray (shape 3,N): Interpolates Z values from
              these [X, Y, Z] points.
        grid_x : np.ndarray
            The 2D target x-coordinate grid (X, Y shape).
        grid_y : np.ndarray
            The 2D target y-coordinate grid (X, Y shape).

        Returns
        -------
        grid_z : np.ndarray
            The 2D array of z-coordinates (or NaNs where interpolation
            failed), with the same shape as `grid_x` and `grid_y`.

        Raises
        ------
        TypeError
            If `z` is not an int, float, or np.ndarray.
        """

        # If int or float provided for z, create set elevation grid
        if isinstance(z, int) or isinstance(z, float):
            print(f"  [GRID] Creating flat Z grid with constant value: {z}")
            grid_z = np.full_like(grid_x, z)

        # If points array is provided, fit points to structured grid
        elif isinstance(z, np.ndarray):
            print(
                f"  [GRID] Interpolating Z grid from {z.shape[1]} input points"
                f"..."
            )
            x = z[0]
            y = z[1]
            # Use a different variable name to avoid confusion with
            # param 'z'
            z_vals = z[2]

            # Interpolate Z values onto the X, Y grid
            grid_z = griddata((x, y), z_vals, (grid_x, grid_y), method="linear")

            # Check for NaNs, which can happen if interpolation fails
            # (e.g., grid points are outside the convex hull of input
            # points)
            nan_count = np.count_nonzero(np.isnan(grid_z))
            if nan_count > 0:
                print(
                    f"    [WARN] {nan_count} grid points were outside the "
                    f"interpolation area (set to NaN)."
                )
                # Optionally, fill NaNs with a specific value
                # grid_z = np.nan_to_num(grid_z, nan=-9999.0)
                # print("    [INFO] NaN values filled with -9999.0")

        # Add a check for bad input
        else:
            print(f"  [ERROR] Z parameter is of an unsupported type: {type(z)}")
            raise TypeError(
                f"Z parameter must be int, float, or np.ndarray, not {type(z)}"
            )

        return grid_z

    def _transpose_grids(self, grid_x, grid_y, grid_z):
        """
        Transpose X, Y, and Z grids from (X, Y) to (Y, X) shape.

        Parameters
        ----------
        grid_x : np.ndarray
            Original (X_steps, Y_steps) shaped x-grid.
        grid_y : np.ndarray
            Original (X_steps, Y_steps) shaped y-grid.
        grid_z : np.ndarray
            Original (X_steps, Y_steps) shaped z-grid.

        Returns
        -------
        grid_x : np.ndarray
            Transposed (Y_steps, X_steps) shaped x-grid.
        grid_y : np.ndarray
            Transposed (Y_steps, X_steps) shaped y-grid.
        grid_z : np.ndarray
            Transposed (Y_steps, X_steps) shaped z-grid.

        Notes
        -----
        This aligns the grid shape with a (Row, Col) or (Y, X)
        convention, which is standard for image and raster processing.
        `np.mgrid` produces (X, Y) shaped arrays, so this step is necessary.
        """
        # --- Transpose ---
        print("  [GRID] Transposing grids to (Y, X) convention.")
        grid_x = grid_x.T
        grid_y = grid_y.T
        grid_z = grid_z.T
        print(f"    Final grid shape: {grid_x.shape}")

        return grid_x, grid_y, grid_z

    def _save_grids_to_zarr(
        self, grid_x, grid_y, grid_z, resolution, dir, grid_descriptor
    ):
        """
        Save the final X, Y, and Z grids as compressed Zarr arrays.

        Parameters
        ----------
        grid_x : np.ndarray
            The final 2D x-coordinate grid (Y, X shape).
        grid_y : np.ndarray
            The final 2D y-coordinate grid (Y, X shape).
        grid_z : np.ndarray
            The final 2D z-coordinate grid (Y, X shape).
        resolution : float
            Grid resolution (in meters), used in the filename.
        dir : str
            The directory to save the files.
        grid_descriptor : str
            A prefix for the output filenames (e.g., 'ground_points').

        Notes
        -----
        Uses `zarr.open()` with mode='w' to create or overwrite arrays.
        Enables chunking for efficient compressed storage.
        """
        # Handle the descriptor string to make sure filename is clean
        if grid_descriptor and not grid_descriptor.endswith("_"):
            grid_descriptor_str = f"{grid_descriptor}_"
        elif not grid_descriptor:
            grid_descriptor_str = ""  # Empty string if none provided
        else:
            # Use as-is if it ends in '_'
            grid_descriptor_str = grid_descriptor

        # Define file paths
        path_x = os.path.join(
            dir, f"{grid_descriptor_str}grid_x_{resolution}m.zarr"
        )
        path_y = os.path.join(
            dir, f"{grid_descriptor_str}grid_y_{resolution}m.zarr"
        )
        path_z = os.path.join(
            dir, f"{grid_descriptor_str}grid_z_{resolution}m.zarr"
        )

        print("  [IO] Saving compressed Zarr arrays (mode='w', overwriting)...")

        # Save X grid
        print(f"    X -> {path_x}")
        zarr.open(
            path_x,
            mode="w",
            shape=grid_x.shape,
            dtype=grid_x.dtype,
            chunks=True,  # Use default chunking
        )[:] = grid_x

        # Save Y grid
        print(f"    Y -> {path_y}")
        zarr.open(
            path_y,
            mode="w",
            shape=grid_y.shape,
            dtype=grid_y.dtype,
            chunks=True,
        )[:] = grid_y

        # Save Z grid
        print(f"    Z -> {path_z}")
        zarr.open(
            path_z,
            mode="w",
            shape=grid_z.shape,
            dtype=grid_z.dtype,
            chunks=True,
        )[:] = grid_z

    @staticmethod
    def plot_elev_grid(grid_z, save_fig=False, fig_name="pixel_DEM.png"):
        """Plots a DEM grid using a truncated 'topo' colormap.

        This function visualizes a 2D elevation grid, typically a Digital
        Elevation Model (DEM). It uses a modified version of the cmocean
        'topo' colormap, truncating it to only show the upper half
        (values 0.5 to 1.0), which typically corresponds to above-sea-level
        (land) elevations.

        Parameters
        ----------
        grid_z : array_like
            A 2D array representing elevation values. Must be a format
            plottable by `matplotlib.pyplot.imshow`.
        save_fig : bool, optional
            If True, saves the figure to disk instead of displaying it.
            Default is False.
        fig_name : str, optional
            Filename to use when saving the figure. Only used if `save_fig`
            is True. Default is 'pixel_DEM.png'.

        Returns
        -------
        None
            This function does not return any value. It displays a
            matplotlib plot and/or saves it to a file.

        Notes
        -----
        The colormap truncation `cmap(np.linspace(0.5, 1, 256))` assumes
        that the 'topo' colormap's midpoint (0.5) represents sea level,
        and values above it represent land.

        """

        # Define colormap

        # Get the full 'topo' colormap from the cmocean library
        cmap = cmocean.cm.topo

        # Create a new, truncated colormap.
        # We sample the *upper half* (from 0.5 to 1.0) of the original 'topo'
        # colormap. This is done to emphasize land features, assuming 0.5
        # is the "sea level" transition point in this colormap.
        above_land_cmap = LinearSegmentedColormap.from_list(
            "above_land_cmap", cmap(np.linspace(0.5, 1, 256))
        )

        # Create plot

        # Display the grid data as an image.
        # 'origin="lower"' places the (0,0) index at the bottom-left corner,
        # which is standard for cartesian (Easting, Northing) coordinates.
        plt.imshow(grid_z, origin="lower", cmap=above_land_cmap)

        # Add labels and colorbar

        # Add a colorbar to show the mapping of elevation values to colors.
        plt.colorbar(label="Elevation (meters)")

        # Add descriptive labels and a title.
        plt.title("Pixel DEM")
        plt.xlabel("Easting")
        plt.ylabel("Northing")

        # Save or show figure

        if save_fig:
            # If save_fig is True, save the figure to the specified file.
            # 'bbox_inches="tight"' crops excess whitespace from the edges.
            # 'pad_inches=0.1' adds a small margin.
            # 'dpi=300' sets a high resolution for good quality.
            plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.1, dpi=300)

        # Display the plot on the screen.
        plt.show()