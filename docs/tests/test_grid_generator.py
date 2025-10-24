import pytest
import numpy as np
import laspy
import zarr
import os
from unittest.mock import MagicMock, call
from poseidon_core import GridGenerator  # Assumes your class is in grid_generator.py

# --- Fixtures: Reusable Mock Objects ---


@pytest.fixture
def mock_lidar_data():
    """
    Creates a mock laspy.LasData object with predictable fake data.
    We'll pretend the units are in feet.
    """
    mock_data = MagicMock(spec=laspy.LasData)

    # Create simple, predictable point data
    # (x, y) coords in feet, (z) in feet
    mock_data.x = np.array([100.0, 200.0, 300.0, 400.0])
    mock_data.y = np.array([100.0, 200.0, 300.0, 400.0])
    mock_data.z = np.array([10.0, 11.0, 12.0, 13.0])

    # Create classifications to test masking
    # Two points are 'ground' (2), two are 'other' (5)
    mock_data.classification = np.array([2, 5, 2, 5])

    return mock_data


@pytest.fixture
def mock_laspy_read(mocker, mock_lidar_data):
    """
    Mocks the `laspy.read` function to return our fake lidar data
    instead of reading from a file.
    """
    # 'mocker' is a fixture from pytest-mock
    # This replaces laspy.read with a mock
    mock = mocker.patch("laspy.read", return_value=mock_lidar_data)
    return mock


@pytest.fixture
def mock_os_makedirs(mocker):
    """Mocks `os.makedirs` to prevent it from creating real directories."""
    return mocker.patch("os.makedirs")


@pytest.fixture
def mock_zarr_open(mocker):
    """Mocks `zarr.open` to prevent it from writing real files."""
    # We mock zarr.open and make it return a mock that can be written to
    mock = mocker.patch("zarr.open", return_value=MagicMock())
    # We need to mock the array slicing part [:]
    mock.return_value.__setitem__ = MagicMock()
    return mock


# --- Test Suite ---


# Use mocks for all tests in this class
@pytest.mark.usefixtures("mock_laspy_read", "mock_os_makedirs", "mock_zarr_open")
class TestGridGenerator:

    def test_init_lidar_units_feet(self, mock_laspy_read, mock_lidar_data):
        """
        Tests initialization when lidar_units='feet'.
        It should NOT use provided extents, but calculate them from the data.
        """
        gen = GridGenerator(
            "fake.las",
            lidar_units="feet",
            extent_units="feet",  # This tells it to calculate extents
        )

        # Check that laspy.read was called
        mock_laspy_read.assert_called_once_with("fake.las")

        # Check that the lidar object is our mock
        assert gen.lidar is mock_lidar_data

        # Check that extents were calculated and converted (100ft * 0.3048 = 30.48)
        assert gen.min_x_extent == pytest.approx(100.0 * 0.3048)
        assert gen.max_x_extent == pytest.approx(400.0 * 0.3048)
        assert gen.min_y_extent == pytest.approx(100.0 * 0.3048)
        assert gen.max_y_extent == pytest.approx(400.0 * 0.3048)

    def test_init_lidar_units_meters_provided_extents(self):
        """
        Tests initialization when lidar_units='meters'.
        It SHOULD use the provided extents.
        """
        gen = GridGenerator(
            "fake.las",
            min_x_extent=10.0,
            max_x_extent=50.0,
            min_y_extent=20.0,
            max_y_extent=60.0,
            extent_units="meters",  # This tells it to use provided extents
            lidar_units="meters",
        )

        # Check that extents are exactly what we provided
        assert gen.min_x_extent == 10.0
        assert gen.max_x_extent == 50.0
        assert gen.min_y_extent == 20.0
        assert gen.max_y_extent == 60.0
        assert gen.lidar_units == "meters"

    def test_create_point_array_feet_conversion(self):
        """
        Tests point array creation with 'feet' units, checking for:
        1. Correct classification masking.
        2. Correct conversion from feet to meters.
        """
        gen = GridGenerator("fake.las", lidar_units="feet", extent_units="feet")

        # Call with mask value 2 (ground)
        points = gen.create_point_array(point_mask_value=2)

        # Mock data classifications: [2, 5, 2, 5]
        # We should get 2 points (index 0 and 2)
        assert points.shape == (3, 2)

        # Expected X values: [100.0, 300.0] * 0.3048
        expected_x = np.array([100.0, 300.0]) * 0.3048
        # Expected Z values: [10.0, 12.0] * 0.3048
        expected_z = np.array([10.0, 12.0]) * 0.3048

        np.testing.assert_allclose(points[0], expected_x)
        np.testing.assert_allclose(points[2], expected_z)

    def test_create_point_array_meters_no_conversion(self):
        """
        Tests point array creation with 'meters' units, checking for:
        1. Correct classification masking.
        2. NO conversion.
        """
        # Set up extents manually to ensure we keep all points
        gen = GridGenerator(
            "fake.las",
            min_x_extent=0,
            max_x_extent=500,
            min_y_extent=0,
            max_y_extent=500,
            extent_units="meters",
            lidar_units="meters",
        )

        # Call with mask value 5
        points = gen.create_point_array(point_mask_value=5)

        # Mock data classifications: [2, 5, 2, 5]
        # We should get 2 points (index 1 and 3)
        assert points.shape == (3, 2)

        # Expected X values: [200.0, 400.0] (no conversion)
        expected_x = np.array([200.0, 400.0])
        # Expected Z values: [11.0, 13.0] (no conversion)
        expected_z = np.array([11.0, 13.0])

        np.testing.assert_allclose(points[0], expected_x)
        np.testing.assert_allclose(points[2], expected_z)

    def test_create_point_array_extent_filtering(self):
        """
        Tests that the point array is correctly filtered by the extents.
        """
        # Use feet, so extents are calculated (min_x=30.48, max_x=121.92)
        gen = GridGenerator("fake.las", lidar_units="feet", extent_units="feet")

        # NOW, manually override the extents to filter points
        # The converted 'ground' points are at x=30.48 and x=91.44
        # This new extent should filter out the first point (30.48)
        gen.min_x_extent = 50.0

        points = gen.create_point_array(point_mask_value=2)

        # We should only get one point (index 2 from original data)
        assert points.shape == (3, 1)

        # Expected X: [300.0] * 0.3048 = 91.44
        expected_x = np.array([300.0]) * 0.3048
        np.testing.assert_allclose(points[0], expected_x)

    def test_gen_grid_flat_z(self, mock_os_makedirs, mock_zarr_open):
        """Tests gen_grid when z is a flat integer (e.g., z=5)."""
        gen = GridGenerator(
            "fake.las",
            min_x_extent=10,
            max_x_extent=30,
            min_y_extent=10,
            max_y_extent=30,
            extent_units="meters",
            lidar_units="meters",
        )

        grid_x, grid_y, grid_z = gen.gen_grid(
            resolution=10, z=5, dir="test/output", grid_descriptor="flat"
        )

        # 1. Check directory creation
        mock_os_makedirs.assert_called_once_with("test/output")

        # 2. Check grid generation
        # np.mgrid[10:30:10] gives [10, 20]. Shape (2, 2) before transpose
        # After transpose, shape is (2, 2)
        assert grid_x.shape == (2, 2)

        # 3. Check Z grid
        assert grid_z.shape == (2, 2)
        # Check that all values are 5
        expected_z = np.full_like(grid_x, 5.0)
        np.testing.assert_allclose(grid_z, expected_z)

        # 4. Check Zarr saving
        assert mock_zarr_open.call_count == 3

        # Check that filenames are correct
        expected_calls = [
            call(
                os.path.join("test/output", "flat_grid_x_10m.zarr"),
                mode="w",
                shape=grid_x.shape,
                dtype=grid_x.dtype,
                chunks=True,
            ),
            call(
                os.path.join("test/output", "flat_grid_y_10m.zarr"),
                mode="w",
                shape=grid_y.shape,
                dtype=grid_y.dtype,
                chunks=True,
            ),
            call(
                os.path.join("test/output", "flat_grid_z_10m.zarr"),
                mode="w",
                shape=grid_z.shape,
                dtype=grid_z.dtype,
                chunks=True,
            ),
        ]
        mock_zarr_open.assert_has_calls(expected_calls)

        # Check that data was "written"
        assert mock_zarr_open.return_value.__setitem__.call_count == 3

    def test_gen_grid_interpolated_z(self):
        """Tests gen_grid when z is a numpy array for interpolation."""
        gen = GridGenerator(
            "fake.las",
            min_x_extent=10,
            max_x_extent=20,
            min_y_extent=10,
            max_y_extent=20,
            extent_units="meters",
            lidar_units="meters",
        )

        # Create a simple 4-point cloud for interpolation
        # (10,10) -> z=1
        # (20,10) -> z=2
        # (10,20) -> z=3
        # (20,20) -> z=4
        z_points = np.array(
            [[10, 20, 10, 20], [10, 10, 20, 20], [1, 2, 3, 4]]  # x  # y  # z
        )

        # Use resolution of 5. Grid will be (10, 15) in x and y
        grid_x, grid_y, grid_z = gen.gen_grid(resolution=5, z=z_points)

        # Shape should be (2, 2) after transpose, as mgrid[10:20:5] -> [10, 15]
        assert grid_z.shape == (2, 2)

        # Check interpolated values for the (2, 2) grid
        # grid_z[y_idx, x_idx]

        # Point (x=10, y=10)
        assert grid_z[0, 0] == pytest.approx(1.0)

        # Point (x=15, y=10) -> halfway between (10,10,1) and (20,10,2)
        assert grid_z[0, 1] == pytest.approx(1.5)

        # Point (x=10, y=15) -> halfway between (10,10,1) and (10,20,3)
        assert grid_z[1, 0] == pytest.approx(2.0)

        # Point (x=15, y=15) -> average of all four points
        assert grid_z[1, 1] == pytest.approx(2.5)

    def test_gen_grid_bad_z_type(self):
        """Tests that gen_grid raises a TypeError for an invalid z type."""
        # FIX: Added dummy extents to prevent the print() function from
        # failing on NoneType before the z-check is reached.
        gen = GridGenerator(
            "fake.las",
            min_x_extent=0,
            max_x_extent=10,
            min_y_extent=0,
            max_y_extent=10,
            extent_units="meters",
            lidar_units="meters",
        )

        # Check that a string raises a TypeError
        with pytest.raises(TypeError, match="must be int, float, or np.ndarray"):
            gen.gen_grid(resolution=10, z="this is not valid")

    def test_gen_grid_filename_descriptors(self, mock_zarr_open):
        """Tests that the grid_descriptor is handled correctly."""
        gen = GridGenerator(
            "fake.las",
            min_x_extent=10,
            max_x_extent=20,
            min_y_extent=10,
            max_y_extent=20,
            extent_units="meters",
            lidar_units="meters",
        )

        # --- Test 1: No descriptor ---
        gen.gen_grid(resolution=5, z=0, dir="test/out", grid_descriptor="")
        # Check the first call to zarr.open
        first_call_args = mock_zarr_open.call_args_list[0].args
        assert first_call_args[0] == os.path.join("test/out", "grid_x_5m.zarr")

        # Reset the mock for the next test
        mock_zarr_open.reset_mock()

        # --- Test 2: Descriptor without underscore ---
        gen.gen_grid(resolution=5, z=0, dir="test/out", grid_descriptor="ground")
        first_call_args = mock_zarr_open.call_args_list[0].args
        assert first_call_args[0] == os.path.join("test/out", "ground_grid_x_5m.zarr")

        # --- Test 3: Descriptor with underscore ---
        mock_zarr_open.reset_mock()
        gen.gen_grid(resolution=5, z=0, dir="test/out", grid_descriptor="ground_")
        first_call_args = mock_zarr_open.call_args_list[0].args
        assert first_call_args[0] == os.path.join("test/out", "ground_grid_x_5m.zarr")
