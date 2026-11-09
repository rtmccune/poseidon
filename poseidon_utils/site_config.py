"""Load per-site camera and grid configuration from YAML files.

Each monitored site (e.g. Carolina Beach, Down East) is described once
in ``poseidon_deploy/configs/sites/<site>.yaml``: camera calibration,
rectification grid extents, coordinate reference system, virtual
sensor pixels, and default data paths. Driver scripts load a site by
name instead of repeating these values on every command line.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

#: Directory holding the version-controlled site YAML files.
DEFAULT_CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "poseidon_deploy" / "configs"
    / "sites"
)

#: Repository root, used to resolve relative data paths in configs.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Number of values in a CIRN-style intrinsics vector.
N_INTRINSICS = 11

#: Number of values in an extrinsics vector (x, y, z, azimuth, tilt,
#: swing).
N_EXTRINSICS = 6

#: Allowed unit names for LiDAR coordinates.
VALID_UNITS = ("meters", "feet")


@dataclass(frozen=True)
class GridExtent:
    """
    Bounding box and spacing of a rectification grid.

    Attributes
    ----------
    min_x, max_x, min_y, max_y : float
        Grid bounds in the site CRS (meters).
    resolution : float
        Grid spacing in meters.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution: float

    def __post_init__(self):
        """
        Validate that the extent is non-empty and spacing positive.

        Raises
        ------
        ValueError
            If a minimum is not below its maximum or the resolution is
            not positive.
        """
        if not (self.min_x < self.max_x and self.min_y < self.max_y):
            raise ValueError(f"Grid extent minima must be below maxima: "
                             f"{self}")
        if self.resolution <= 0:
            raise ValueError(f"Grid resolution must be positive: {self}")


@dataclass(frozen=True)
class SiteConfig:
    """
    Camera, grid, and data settings for one monitoring site.

    Attributes
    ----------
    name : str
        Site name; also used as the grid descriptor (e.g.
        ``"carolina_beach"``).
    sensor_id : str
        SDFP sensor/camera ID without the ``CAM_`` prefix (``"CB_03"``).
    crs : str
        Coordinate reference system of the grid (e.g. ``"EPSG:32119"``).
    grid : GridExtent
        Rectification grid bounds and spacing.
    lidar_units : {'meters', 'feet'}
        Units of the site's LiDAR point coordinates.
    intrinsics : numpy.ndarray
        11-element CIRN intrinsics vector.
    extrinsics : numpy.ndarray
        6-element extrinsics vector (x, y, z, azimuth, tilt, swing).
    virtual_sensors : numpy.ndarray or None
        ``(N, 2)`` integer ``(col, row)`` grid pixels to sample as
        virtual water level sensors.
    latitude, longitude : float or None
        Camera location in decimal degrees, used for sun-position
        filtering.
    lidar_file : str or None
        LiDAR path, relative to the repository root unless absolute.
    basemap : str or None
        Basemap GeoTIFF path, relative to the repository root unless
        absolute.
    roi_json : str or None
        LabelMe JSON of the roadway region of interest, relative to the
        repository root unless absolute.
    camera_roi_json : str or None
        LabelMe JSON of a roadway polygon drawn on an original camera
        image, relative to the repository root unless absolute.
    sensor_wse_floor : float or None
        Lower clip (m) applied to in-situ sensor WSE in comparison
        plots.
    """

    name: str
    sensor_id: str
    crs: str
    grid: GridExtent
    lidar_units: str
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    virtual_sensors: np.ndarray | None = None
    latitude: float | None = None
    longitude: float | None = None
    lidar_file: str | None = None
    basemap: str | None = None
    roi_json: str | None = None
    camera_roi_json: str | None = None
    sensor_wse_floor: float | None = None

    @property
    def camera_name(self):
        """
        Camera name used in image filenames.

        Returns
        -------
        str
            ``"CAM_"`` followed by :attr:`sensor_id`.
        """
        return f"CAM_{self.sensor_id}"

    def resolve(self, relative_path, root=REPO_ROOT):
        """
        Resolve a config path against the repository root.

        Parameters
        ----------
        relative_path : str or None
            Path from the config. Absolute paths are returned as-is.
        root : str or os.PathLike, optional
            Base directory for relative paths. Default is the repo root.

        Returns
        -------
        pathlib.Path or None
            Resolved path, or None when `relative_path` is None.
        """
        if relative_path is None:
            return None
        path = Path(relative_path)
        return path if path.is_absolute() else Path(root) / path


def _as_vector(values, length, field):
    """
    Convert a YAML list to a float vector of the expected length.

    Parameters
    ----------
    values : sequence of float
        Raw values from the YAML file.
    length : int
        Required number of elements.
    field : str
        Field name, used in the error message.

    Returns
    -------
    numpy.ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If the number of elements is wrong.
    """
    vector = np.asarray(values, dtype=float)
    if vector.shape != (length,):
        raise ValueError(f"'{field}' must have {length} values, "
                         f"got {vector.size}")
    return vector


def site_from_dict(data):
    """
    Build a validated :class:`SiteConfig` from a parsed YAML mapping.

    Parameters
    ----------
    data : dict
        Mapping with the keys documented in the site YAML files.

    Returns
    -------
    SiteConfig
        The validated configuration.

    Raises
    ------
    KeyError
        If a required key is missing.
    ValueError
        If a value has the wrong shape or an invalid option.
    """
    units = data.get("lidar_units", "meters")
    if units not in VALID_UNITS:
        raise ValueError(f"lidar_units must be one of {VALID_UNITS}, "
                         f"got '{units}'")

    sensors = data.get("virtual_sensors")
    if sensors is not None:
        sensors = np.asarray(sensors, dtype=int).reshape(-1, 2)

    return SiteConfig(
        name=data["name"],
        sensor_id=data["sensor_id"],
        crs=data["crs"],
        grid=GridExtent(**data["grid"]),
        lidar_units=units,
        intrinsics=_as_vector(data["intrinsics"], N_INTRINSICS,
                              "intrinsics"),
        extrinsics=_as_vector(data["extrinsics"], N_EXTRINSICS,
                              "extrinsics"),
        virtual_sensors=sensors,
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        lidar_file=data.get("lidar_file"),
        basemap=data.get("basemap"),
        roi_json=data.get("roi_json"),
        camera_roi_json=data.get("camera_roi_json"),
        sensor_wse_floor=data.get("sensor_wse_floor"),
    )


def load_site_config(site, config_dir=DEFAULT_CONFIG_DIR):
    """
    Load a site configuration by name or file path.

    Parameters
    ----------
    site : str or os.PathLike
        Site name (``"carolina_beach"`` loads
        ``<config_dir>/carolina_beach.yaml``) or a path to a YAML file.
    config_dir : str or os.PathLike, optional
        Directory searched for named sites. Default is
        :data:`DEFAULT_CONFIG_DIR`.

    Returns
    -------
    SiteConfig
        The parsed and validated site configuration.

    Raises
    ------
    FileNotFoundError
        If no matching YAML file exists.
    """
    path = Path(site)
    if path.suffix not in (".yaml", ".yml"):
        path = Path(config_dir) / f"{site}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in Path(config_dir).glob("*.yaml"))
        raise FileNotFoundError(f"Site config not found: {path}. "
                                f"Available sites: {available}")
    with open(path) as handle:
        return site_from_dict(yaml.safe_load(handle))
