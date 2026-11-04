"""Tests for GridGenerator and ImageRectifier with known geometry."""

import cv2
import numpy as np
import pytest
import zarr
from conftest import (
    NADIR_EXTRINSICS,
    NADIR_INTRINSICS,
    make_grid,
    requires_gpu,
    write_las,
)

from poseidon_core.grid_generator import GridGenerator
from poseidon_core.image_rectifier import ImageRectifier
from poseidon_utils.constants import FEET_TO_METERS
from poseidon_utils.zarr_io import read_zip_arrays

# ----- GridGenerator ---------------------------------------------------------


def test_ground_points_interpolate_the_plane(plane_las, tmp_path):
    """Linear interpolation of points on a plane reproduces the plane."""
    gen = GridGenerator(plane_las, 2, 8, 1, 6, lidar_units="meters")
    points = gen.create_point_array()
    gx, gy, gz = gen.gen_grid(0.5, points, dir=str(tmp_path),
                              grid_descriptor="plane")

    np.testing.assert_allclose(gz, 1 + 0.1 * gx + 0.2 * gy, atol=1e-3)
    # Rows run south to north, columns west to east
    assert gy[1, 0] > gy[0, 0] and gx[0, 1] > gx[0, 0]
    assert gx.shape == (10, 12)  # (6 - 1) / 0.5 rows, (8 - 2) / 0.5 cols


def test_non_ground_and_out_of_extent_points_are_dropped(plane_las):
    """Only ground points inside the extent are kept, in meters."""
    gen = GridGenerator(plane_las, 2, 8, 1, 6, lidar_units="meters")
    x, y, z = gen.create_point_array()
    assert ((x >= 2) & (x <= 8) & (y >= 1) & (y <= 6)).all()
    np.testing.assert_allclose(z, 1 + 0.1 * x + 0.2 * y, atol=1e-3)


def test_grids_are_saved_as_named_zarr_arrays(plane_las, tmp_path):
    """Each axis is saved as <descriptor>_grid_<axis>_<res>m.zarr."""
    gen = GridGenerator(plane_las, 2, 8, 1, 6, lidar_units="meters")
    _, _, gz = gen.gen_grid(0.5, z=1.5, dir=str(tmp_path),
                            grid_descriptor="flat")
    saved = zarr.open(str(tmp_path / "flat_grid_z_0.5m.zarr"), mode="r")[:]
    np.testing.assert_array_equal(saved, gz)
    assert (gz == 1.5).all()


def test_missing_extents_default_to_lidar_bounds_in_meters(tmp_path):
    """None extents use the LiDAR bounds converted from feet."""
    las = write_las(tmp_path / "ft.las", [0.0, 100.0], [0.0, 50.0],
                    [1.0, 2.0], [2, 2])
    gen = GridGenerator(las, lidar_units="feet")
    assert gen.max_x_extent == pytest.approx(100 * FEET_TO_METERS)
    assert gen.max_y_extent == pytest.approx(50 * FEET_TO_METERS)


def test_extents_given_in_feet_are_converted_not_ignored(tmp_path):
    """User extents in feet are honored after conversion to meters."""
    las = write_las(tmp_path / "ft.las", [0.0, 100.0], [0.0, 50.0],
                    [1.0, 2.0], [2, 2])
    gen = GridGenerator(las, 10, 20, 5, 15, extent_units="feet",
                        lidar_units="feet")
    assert (gen.min_x_extent, gen.max_y_extent) == pytest.approx(
        (10 * FEET_TO_METERS, 15 * FEET_TO_METERS))


def test_bad_units_and_z_types_are_rejected(plane_las):
    """Unknown units or unsupported z inputs raise clear errors."""
    with pytest.raises(ValueError, match="lidar_units"):
        GridGenerator(plane_las, lidar_units="yards")
    gen = GridGenerator(plane_las, 2, 8, 1, 6, lidar_units="meters")
    with pytest.raises(TypeError):
        gen.gen_grid(0.5, z="flat")


# ----- ImageRectifier --------------------------------------------------------


def expected_nadir_pixels(grid_x, grid_y):
    """
    Pixel each grid cell projects to for the nadir test camera.

    With no distortion, a camera at (Xc, Yc, H) looking straight down
    maps (X, Y, 0) to u = c0U + fx (X - Xc) / H and
    v = c0V - fy (Y - Yc) / H.

    Parameters
    ----------
    grid_x, grid_y : numpy.ndarray
        Grid coordinates.

    Returns
    -------
    u, v : numpy.ndarray
        Integer image column and row.
    """
    c0u, c0v, fx, fy = NADIR_INTRINSICS[2:6]
    xc, yc, height = NADIR_EXTRINSICS[:3]
    u = np.round(c0u + fx * (grid_x - xc) / height).astype(int)
    v = np.round(c0v - fy * (grid_y - yc) / height).astype(int)
    return u, v


@pytest.fixture
def rectifier(nadir_grid):
    """
    CPU rectifier for the nadir camera over the flat test grid.

    Parameters
    ----------
    nadir_grid : tuple of numpy.ndarray
        Flat test grid from the ``nadir_grid`` fixture.

    Returns
    -------
    ImageRectifier
        Rectifier instance.
    """
    return ImageRectifier(NADIR_INTRINSICS, NADIR_EXTRINSICS, *nadir_grid,
                          use_gpu=False)


def test_nadir_projection_matches_pinhole_formula(rectifier, nadir_grid):
    """Every grid cell maps to the analytically expected pixel."""
    u, v = expected_nadir_pixels(*nadir_grid[:2])
    np.testing.assert_array_equal(rectifier.u_distorted, u)
    np.testing.assert_array_equal(rectifier.v_distorted, v)


def test_rotation_matrix_is_orthonormal(rectifier):
    """The CIRN rotation is a proper rotation for arbitrary angles."""
    rectifier.azimuth, rectifier.tilt, rectifier.swing = 1.3, 1.03, -0.16
    rotation = rectifier._cirn_angles_to_rotation()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rotation)) == pytest.approx(1.0)


def test_zero_distortion_leaves_coordinates_unchanged(rectifier):
    """With zero distortion coefficients, UV coordinates pass through."""
    uv = np.array([[10.0, 200.5, 399.0], [5.0, 150.0, 299.0]])
    u, v, valid = rectifier._distort_uv(uv.copy())
    np.testing.assert_allclose(u, uv[0])
    np.testing.assert_allclose(v, uv[1])
    assert (valid == 1).all()


def test_radial_distortion_moves_points_outward(nadir_grid):
    """Positive d1 pushes off-center points away from the center."""
    intrinsics = NADIR_INTRINSICS.copy()
    intrinsics[6] = 0.1
    rect = ImageRectifier(intrinsics, NADIR_EXTRINSICS, *nadir_grid,
                          use_gpu=False)
    u, _, _ = rect._distort_uv(np.array([[300.0], [150.0]]))
    assert u[0] > 300.0


def test_cells_behind_the_camera_are_blank(tmp_path):
    """Ground above the camera projects nowhere and rectifies to 0."""
    gx, gy, gz = make_grid(z=20.0)  # grid 10 m above the camera
    rect = ImageRectifier(NADIR_INTRINSICS, NADIR_EXTRINSICS, gx, gy, gz,
                          use_gpu=False)
    assert (rect.u_distorted == 0).all()
    image = np.full((300, 400, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "img.png"), image)
    assert (rect.merge_rectify(tmp_path / "img.png") == 0).all()


def test_rectified_pixels_come_from_projected_locations(rectifier, tmp_path,
                                                        nadir_grid):
    """Rectified values equal the image values at the projected pixels."""
    rows, cols = np.mgrid[0:300, 0:400]
    gray = ((rows * 7 + cols * 3) % 251).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "labels.png"), gray)

    rectified = rectifier.merge_rectify(tmp_path / "labels.png", labels=True)
    u, v = expected_nadir_pixels(*nadir_grid[:2])
    assert rectified.shape == (*nadir_grid[0].shape, 1)
    np.testing.assert_array_equal(rectified[..., 0], gray[v, u])


def test_rgb_images_keep_rgb_channel_order(rectifier, tmp_path):
    """Images are returned as RGB, not OpenCV's BGR."""
    bgr = np.zeros((300, 400, 3), np.uint8)
    bgr[..., 2] = 255  # pure red in OpenCV's BGR layout
    cv2.imwrite(str(tmp_path / "red.png"), bgr)
    rectified = rectifier.merge_rectify(tmp_path / "red.png")
    assert (rectified[..., 0] == 255).all()
    assert (rectified[..., 1:] == 0).all()


def test_unreadable_image_raises(rectifier, tmp_path):
    """A missing image raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        rectifier.merge_rectify(tmp_path / "nope.jpg")


def test_folder_rectification_writes_and_replaces_one_store(rectifier,
                                                            tmp_path):
    """All images go into one ZIP store that reruns replace."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("CAM_TS_01_20240101120000.png",
                 "CAM_TS_01_20240101120600.png"):
        cv2.imwrite(str(folder / name), np.full((300, 400, 3), 9, np.uint8))
    (folder / "notes.txt").write_text("not an image")
    (folder / "broken.jpg").write_bytes(b"not a jpeg")

    store = tmp_path / "zarr" / "orig_image_rects"
    assert rectifier.merge_rectify_folder(folder, store) == 2
    names = sorted(read_zip_arrays(f"{store}.zip"))
    assert names == ["CAM_TS_01_20240101120000_rectified",
                     "CAM_TS_01_20240101120600_rectified"]

    # Rerun after removing one image: the store reflects the new run
    (folder / "CAM_TS_01_20240101120600.png").unlink()
    assert rectifier.merge_rectify_folder(folder, store) == 1
    assert list(read_zip_arrays(f"{store}.zip")) == [
        "CAM_TS_01_20240101120000_rectified"]


def test_missing_folder_writes_nothing(rectifier, tmp_path):
    """A missing source folder returns 0 and creates no store."""
    assert rectifier.merge_rectify_folder(tmp_path / "none",
                                          tmp_path / "out") == 0
    assert not (tmp_path / "out.zip").exists()


@requires_gpu
def test_gpu_and_cpu_rectification_agree(nadir_grid, tmp_path):
    """CuPy and NumPy backends produce identical pixels."""
    intrinsics = NADIR_INTRINSICS.copy()
    intrinsics[6:9] = (-0.2, 0.05, -0.01)  # exercise distortion too
    rng = np.random.default_rng(0)
    cv2.imwrite(str(tmp_path / "noise.png"),
                rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    results = []
    for use_gpu in (False, True):
        rect = ImageRectifier(intrinsics, NADIR_EXTRINSICS, *nadir_grid,
                              use_gpu=use_gpu)
        image = rect.merge_rectify(tmp_path / "noise.png")
        results.append(image.get() if use_gpu else image)
    np.testing.assert_array_equal(*results)
