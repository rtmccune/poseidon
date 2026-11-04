"""Shared fixtures and helpers for the POSEIDON test suite.

The tests build small synthetic inputs with known answers (LiDAR
planes, nadir cameras, ponds of known depth) instead of mocking the
code under test. Only the SDFP web API is replaced with a fake.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # never open plot windows during tests

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make the driver and analysis scripts importable as modules
for folder in ("poseidon_deploy/naiads", "poseidon_deploy/analysis",
               "poseidon_deploy/environment"):
    path = str(REPO_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

from poseidon_core.backend import gpu_available  # noqa: E402


def requires_gpu(test):
    """
    Mark a test as needing a GPU and skip it when none is usable.

    Parameters
    ----------
    test : callable
        Test function to decorate.

    Returns
    -------
    callable
        The test with the ``gpu`` marker and a skip condition, so
        ``pytest -m gpu`` selects exactly the GPU tests.
    """
    skip = pytest.mark.skipif(not gpu_available(),
                              reason="no usable CuPy GPU")
    return pytest.mark.gpu(skip(test))


#: Camera intrinsics with no lens distortion: 400 x 300 px image,
#: principal point at the center, 200 px focal length.
NADIR_INTRINSICS = np.array([400, 300, 200.0, 150.0, 200.0, 200.0,
                             0, 0, 0, 0, 0], dtype=float)

#: Camera 10 m above the grid origin looking straight down (tilt 0).
NADIR_EXTRINSICS = np.array([5.0, 5.0, 10.0, 0.0, 0.0, 0.0])


def make_grid(n_rows=40, n_cols=50, spacing=0.25, z=0.0):
    """
    Build X/Y/Z grids with rows increasing in y (row 0 south).

    Parameters
    ----------
    n_rows, n_cols : int, optional
        Grid shape. Defaults are 40 and 50.
    spacing : float, optional
        Cell size in meters. Default is 0.25.
    z : float or numpy.ndarray, optional
        Elevation (constant or an ``(n_rows, n_cols)`` array).

    Returns
    -------
    grid_x, grid_y, grid_z : numpy.ndarray
        Coordinate grids.
    """
    xs = np.arange(n_cols) * spacing
    ys = np.arange(n_rows) * spacing
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_z = np.broadcast_to(np.asarray(z, dtype=float), grid_x.shape).copy()
    return grid_x, grid_y, grid_z


@pytest.fixture
def nadir_grid():
    """
    Grid of 40 x 50 cells at 0.25 m spacing on flat ground (z = 0).

    Returns
    -------
    tuple of numpy.ndarray
        ``(grid_x, grid_y, grid_z)``.
    """
    return make_grid()


def write_las(path, x, y, z, classification):
    """
    Write a minimal LAS file with the given points.

    Parameters
    ----------
    path : str or os.PathLike
        Output ``.las`` path.
    x, y, z : array_like
        Point coordinates.
    classification : array_like
        LAS classification code of each point.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([np.min(x), np.min(y), np.min(z)])
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.asarray(classification, dtype=np.uint8)
    las.write(str(path))
    return Path(path)


@pytest.fixture
def plane_las(tmp_path):
    """
    LAS file of ground points on the plane z = 1 + 0.1 x + 0.2 y.

    Points cover x, y in [0, 10] meters on a 0.5 m lattice, plus
    non-ground (class 1) points 5 m above the plane that must be
    ignored.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    pathlib.Path
        Path of the LAS file (coordinates in meters).
    """
    xs, ys = np.meshgrid(np.arange(0, 10.01, 0.5), np.arange(0, 10.01, 0.5))
    x, y = xs.ravel(), ys.ravel()
    z = 1 + 0.1 * x + 0.2 * y
    noise_x, noise_y = x[::7], y[::7]
    return write_las(
        tmp_path / "plane.las",
        np.concatenate([x, noise_x]), np.concatenate([y, noise_y]),
        np.concatenate([z, 1 + 0.1 * noise_x + 0.2 * noise_y + 5]),
        np.concatenate([np.full(x.size, 2), np.full(noise_x.size, 1)]))


def square_mask(shape, rows, cols, value=1):
    """
    Return a label array with one rectangular region set to `value`.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    rows, cols : slice
        Region to fill.
    value : int, optional
        Fill value. Default is 1 (water).

    Returns
    -------
    numpy.ndarray
        ``uint8`` label array.
    """
    labels = np.zeros(shape, dtype=np.uint8)
    labels[rows, cols] = value
    return labels
