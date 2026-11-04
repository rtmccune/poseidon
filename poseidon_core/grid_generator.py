"""Generate regular X/Y/Z elevation grids from LiDAR point clouds.

The grid is the real-world surface onto which camera images are
rectified. Ground-classified LiDAR returns inside the requested extent
are linearly interpolated onto a regular grid whose rows run south to
north (row 0 at the minimum y) and columns west to east.
"""

import logging
import os
import time

import laspy
import numpy as np
import zarr
from scipy.interpolate import griddata

from poseidon_utils.constants import FEET_TO_METERS

logger = logging.getLogger(__name__)

#: Accepted unit names for extents and LiDAR coordinates.
VALID_UNITS = ("meters", "feet")

#: ASPRS LAS classification code for ground points.
GROUND_CLASS = 2


def _unit_scale(units, name):
    """
    Return the factor converting the given units to meters.

    Parameters
    ----------
    units : {'meters', 'feet'}
        Unit name.
    name : str
        Parameter name, used in the error message.

    Returns
    -------
    float
        1.0 for meters, :data:`FEET_TO_METERS` for feet.

    Raises
    ------
    ValueError
        If `units` is not a recognized unit name.
    """
    if units not in VALID_UNITS:
        raise ValueError(f"{name} must be one of {VALID_UNITS}, "
                         f"got '{units}'")
    return FEET_TO_METERS if units == "feet" else 1.0


def _bound_in_meters(value, lidar_bound, extent_scale, lidar_scale):
    """
    Convert one grid bound to meters, defaulting to the LiDAR bound.

    Parameters
    ----------
    value : float or None
        User-supplied bound in extent units, or None.
    lidar_bound : float
        Matching bound of the LiDAR data in LiDAR units.
    extent_scale : float
        Extent-units-to-meters factor.
    lidar_scale : float
        LiDAR-units-to-meters factor.

    Returns
    -------
    float
        The bound in meters.
    """
    if value is None:
        return float(lidar_bound) * lidar_scale
    return float(value) * extent_scale


class GridGenerator:
    """
    Build rectification grids from a LiDAR (.las/.laz) file.

    Parameters
    ----------
    file_path : str or os.PathLike
        LiDAR file to load.
    min_x_extent, max_x_extent, min_y_extent, max_y_extent : float, optional
        Grid bounds in `extent_units`. Any bound left as None defaults
        to the corresponding bound of the LiDAR data.
    extent_units : {'meters', 'feet'}, optional
        Units of the extent arguments. Default is ``"meters"``.
    lidar_units : {'feet', 'meters'}, optional
        Units of the LiDAR point coordinates. Default is ``"feet"``.

    Attributes
    ----------
    lidar : laspy.LasData
        Loaded point cloud.
    min_x_extent, max_x_extent, min_y_extent, max_y_extent : float
        Grid bounds converted to meters.
    point_mask_val : int
        Classification used by the last :meth:`create_point_array`.

    Notes
    -----
    Typical use: ``points = gen.create_point_array()`` then
    ``gen.gen_grid(resolution, points)``. ``gen.gen_grid(res, z=0)``
    builds a flat grid instead.
    """

    def __init__(self, file_path, min_x_extent=None, max_x_extent=None,
                 min_y_extent=None, max_y_extent=None,
                 extent_units="meters", lidar_units="feet"):
        """
        Load the point cloud and resolve the grid extent in meters.

        Parameters
        ----------
        file_path : str or os.PathLike
            LiDAR file to load.
        min_x_extent, max_x_extent : float, optional
            Grid x bounds in `extent_units`; None uses the LiDAR bounds.
        min_y_extent, max_y_extent : float, optional
            Grid y bounds in `extent_units`; None uses the LiDAR bounds.
        extent_units : {'meters', 'feet'}, optional
            Units of the extent arguments. Default is ``"meters"``.
        lidar_units : {'feet', 'meters'}, optional
            Units of the LiDAR coordinates. Default is ``"feet"``.

        Raises
        ------
        ValueError
            If a unit name is not recognized.
        """
        self.filepath = os.fspath(file_path)
        self.filename = os.path.basename(self.filepath)
        self.extent_units = extent_units
        self.lidar_units = lidar_units
        extent_scale = _unit_scale(extent_units, "extent_units")
        self._lidar_scale = _unit_scale(lidar_units, "lidar_units")
        self.lidar = laspy.read(self.filepath)

        x, y = self.lidar.x, self.lidar.y
        scales = (extent_scale, self._lidar_scale)
        self.min_x_extent = _bound_in_meters(min_x_extent, np.min(x), *scales)
        self.max_x_extent = _bound_in_meters(max_x_extent, np.max(x), *scales)
        self.min_y_extent = _bound_in_meters(min_y_extent, np.min(y), *scales)
        self.max_y_extent = _bound_in_meters(max_y_extent, np.max(y), *scales)

    def create_point_array(self, point_mask_value=GROUND_CLASS):
        """
        Return classified LiDAR points inside the extent, in meters.

        Parameters
        ----------
        point_mask_value : int, optional
            LAS classification to keep. Default is 2 (ground).

        Returns
        -------
        numpy.ndarray
            ``(3, N)`` array of ``[X, Y, Z]`` points in meters.
        """
        self.point_mask_val = point_mask_value
        keep = self.lidar.classification == point_mask_value
        xyz = np.vstack([self.lidar.x[keep], self.lidar.y[keep],
                         self.lidar.z[keep]]) * self._lidar_scale

        inside = ((xyz[0] >= self.min_x_extent)
                  & (xyz[0] <= self.max_x_extent)
                  & (xyz[1] >= self.min_y_extent)
                  & (xyz[1] <= self.max_y_extent))
        return xyz[:, inside]

    def gen_grid(self, resolution, z=0, dir="data/generated_grids",
                 grid_descriptor=""):
        """
        Generate X, Y, Z grids and save them as Zarr arrays.

        Parameters
        ----------
        resolution : float
            Grid spacing in meters.
        z : float or numpy.ndarray, optional
            Constant elevation for a flat grid, or a ``(3, N)`` point
            array to interpolate linearly. Default is 0.
        dir : str, optional
            Output folder for the Zarr files. Default is
            ``"data/generated_grids"``.
        grid_descriptor : str, optional
            Filename prefix, e.g. ``"carolina_beach"``. Default ``""``.

        Returns
        -------
        grid_x, grid_y, grid_z : numpy.ndarray
            ``(rows, cols)`` grids with rows increasing in y. Cells
            outside the convex hull of the points are NaN.

        Raises
        ------
        TypeError
            If `z` is neither a number nor a NumPy array.
        """
        logger.info("Generating '%s' grid at %s m resolution.",
                    grid_descriptor, resolution)
        start = time.time()
        os.makedirs(dir, exist_ok=True)

        grid_x, grid_y = self._generate_grid_coordinates(resolution)
        grid_z = self._generate_z_grid(z, grid_x, grid_y)

        # np.mgrid yields (x, y) ordering; transpose to (row=y, col=x)
        grid_x, grid_y, grid_z = grid_x.T, grid_y.T, grid_z.T
        self._save_grids_to_zarr(grid_x, grid_y, grid_z, resolution, dir,
                                 grid_descriptor)

        logger.info("Grid %s generated in %.2f s.", grid_x.shape,
                    time.time() - start)
        return grid_x, grid_y, grid_z

    def _generate_grid_coordinates(self, resolution):
        """
        Build X and Y coordinate grids spanning the extent.

        Parameters
        ----------
        resolution : float
            Grid spacing in meters.

        Returns
        -------
        grid_x, grid_y : numpy.ndarray
            ``(n_x, n_y)`` coordinate grids from ``np.mgrid``; the
            maximum bound is excluded.
        """
        logger.info("  Extent x %.2f-%.2f, y %.2f-%.2f (m)",
                    self.min_x_extent, self.max_x_extent,
                    self.min_y_extent, self.max_y_extent)
        return np.mgrid[
            self.min_x_extent:self.max_x_extent:resolution,
            self.min_y_extent:self.max_y_extent:resolution,
        ]

    def _generate_z_grid(self, z, grid_x, grid_y):
        """
        Fill the Z grid with a constant or interpolated elevations.

        Parameters
        ----------
        z : float or numpy.ndarray
            Constant elevation or ``(3, N)`` points to interpolate.
        grid_x, grid_y : numpy.ndarray
            Target coordinate grids.

        Returns
        -------
        numpy.ndarray
            Elevation grid shaped like `grid_x`.

        Raises
        ------
        TypeError
            If `z` is neither a number nor a NumPy array.
        """
        if isinstance(z, (int, float, np.number)):
            return np.full_like(grid_x, z, dtype=float)
        if not isinstance(z, np.ndarray):
            raise TypeError(f"z must be a number or numpy.ndarray, not "
                            f"{type(z).__name__}")

        logger.info("  Interpolating %d points onto the grid.", z.shape[1])
        grid_z = griddata((z[0], z[1]), z[2], (grid_x, grid_y),
                          method="linear")
        n_missing = int(np.count_nonzero(np.isnan(grid_z)))
        if n_missing:
            logger.warning("  %d grid cells lie outside the LiDAR coverage "
                           "and are NaN.", n_missing)
        return grid_z

    @staticmethod
    def _save_grids_to_zarr(grid_x, grid_y, grid_z, resolution, dir,
                            grid_descriptor):
        """
        Save the grids as ``<prefix>grid_<axis>_<resolution>m.zarr``.

        Parameters
        ----------
        grid_x, grid_y, grid_z : numpy.ndarray
            Grids to save.
        resolution : float
            Grid spacing, used in the filenames.
        dir : str
            Output folder.
        grid_descriptor : str
            Filename prefix; an underscore separator is added if needed.
        """
        prefix = grid_descriptor
        if prefix and not prefix.endswith("_"):
            prefix += "_"
        for axis, grid in (("x", grid_x), ("y", grid_y), ("z", grid_z)):
            path = os.path.join(dir, f"{prefix}grid_{axis}_{resolution}m.zarr")
            # chunks="auto": zarr >= 3.2 no longer accepts chunks=True
            zarr.create_array(store=path, data=grid, chunks="auto",
                              overwrite=True)
            logger.info("  Saved %s", path)

    @staticmethod
    def plot_elev_grid(grid_z, save_fig=False, fig_name="pixel_DEM.png"):
        """
        Display an elevation grid with the land half of cmocean 'topo'.

        Parameters
        ----------
        grid_z : array_like
            2-D elevation grid (meters).
        save_fig : bool, optional
            Also save the figure to `fig_name`. Default is False.
        fig_name : str, optional
            Output filename. Default is ``"pixel_DEM.png"``.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        # Plotting libraries are only needed here, not for grid building
        import cmocean
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Upper half of 'topo' (0.5-1.0) holds the above-sea-level colors
        land_cmap = LinearSegmentedColormap.from_list(
            "above_land_cmap", cmocean.cm.topo(np.linspace(0.5, 1, 256)))

        fig, ax = plt.subplots()
        image = ax.imshow(grid_z, origin="lower", cmap=land_cmap)
        fig.colorbar(image, ax=ax, label="Elevation (meters)")
        ax.set_title("Pixel DEM")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        if save_fig:
            fig.savefig(fig_name, bbox_inches="tight", pad_inches=0.1,
                        dpi=300)
        plt.show()
        return fig
