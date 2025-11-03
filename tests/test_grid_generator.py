import pytest
import numpy as np
import laspy
import zarr
import os
from unittest.mock import MagicMock, call

from poseidon_core import GridGenerator

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


@pytest.fixture
def gen_instance():
    """
    Provides a standard GridGenerator instance for testing private methods.
    Uses meter units and standard 0-100 extents.
    Relies on class-level mock_laspy_read fixture.
    """
    gen = GridGenerator(
        "fake.las",
        min_x_extent=0,
        max_x_extent=100,
        min_y_extent=0,
        max_y_extent=100,
        extent_units="meters",
        lidar_units="meters",
    )
    return gen


# --- Test Suite ---


# Use mocks for all tests in this class
@pytest.mark.usefixtures(
    "mock_laspy_read", "mock_os_makedirs", "mock_zarr_open"
)
class TestGridGenerator:

    # --- Tests for __init__ ---

    def test_init_lidar_units_feet(self, mock_laspy_read, mock_lidar_data):
        """
        Tests initialization when lidar_units='feet'.
        It should NOT use provided extents, but calculate them from the
        data.
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

        # Check that extents were calculated and converted
        # (100ft * 0.3048 = 30.48)
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
            extent_units="meters",  # Use provided extents
            lidar_units="meters",
        )

        # Check that extents are exactly what we provided
        assert gen.min_x_extent == 10.0
        assert gen.max_x_extent == 50.0
        assert gen.min_y_extent == 20.0
        assert gen.max_y_extent == 60.0
        assert gen.lidar_units == "meters"

    # --- Tests for create_point_array ---

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
        # Use feet, extents are calculated (min_x=30.48, max_x=121.92)
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

    def test_create_point_array_no_points_match_mask(self):
        """
        Tests create_point_array when the classification mask finds 0
        points.
        """
        # Use meters, extents 0-500 so no extent filtering
        gen = GridGenerator(
            "fake.las",
            min_x_extent=0,
            max_x_extent=500,
            min_y_extent=0,
            max_y_extent=500,
            extent_units="meters",
            lidar_units="meters",
        )

        # Classifications are [2, 5, 2, 5].
        # Mask value 99 will find nothing.
        points = gen.create_point_array(point_mask_value=99)

        # Should return an empty array with the correct shape
        assert points.shape == (3, 0)

    def test_create_point_array_no_points_in_extent(self):
        """
        Tests create_point_array when points are found but are all
        outside the extents.
        """
        # Use meters. Points are at x=[100, 200, 300, 400]
        # Set extents to be far away
        gen = GridGenerator(
            "fake.las",
            min_x_extent=1000,
            max_x_extent=2000,
            min_y_extent=1000,
            max_y_extent=2000,
            extent_units="meters",
            lidar_units="meters",
        )

        # This will find 2 points (class 2), but both will be
        # filtered out by the extents.
        points = gen.create_point_array(point_mask_value=2)

        # Should return an empty array
        assert points.shape == (3, 0)

    # --- Tests for gen_grid (Public Method) ---

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
        # np.mgrid[10:30:10] gives [10, 20]. Shape (2, 2) before
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
            [[10, 20, 10, 20], [10, 10, 20, 20], [1, 2, 3, 4]]
            # x  # y  # z
        )

        # Use resolution of 5. Grid will be (10, 15) in x and y
        grid_x, grid_y, grid_z = gen.gen_grid(resolution=5, z=z_points)

        # Shape should be (2, 2) after transpose,
        # as mgrid[10:20:5] -> [10, 15]
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
        with pytest.raises(
            TypeError, match="must be int, float, or np.ndarray"
        ):
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
        gen.gen_grid(
            resolution=5, z=0, dir="test/out", grid_descriptor="ground"
        )
        first_call_args = mock_zarr_open.call_args_list[0].args
        assert first_call_args[0] == os.path.join(
            "test/out", "ground_grid_x_5m.zarr"
        )

        # --- Test 3: Descriptor with underscore ---
        mock_zarr_open.reset_mock()
        gen.gen_grid(
            resolution=5, z=0, dir="test/out", grid_descriptor="ground_"
        )
        first_call_args = mock_zarr_open.call_args_list[0].args
        assert first_call_args[0] == os.path.join(
            "test/out", "ground_grid_x_5m.zarr"
        )

    # --- New Tests for Private Methods ---

    def test_prepare_output_dir_creates_new(
        self, gen_instance, mock_os_makedirs, mocker
    ):
        """
        Tests that _prepare_output_dir calls os.makedirs if the dir
        does not exist.
        """
        # Mock os.path.exists to return False
        mock_exists = mocker.patch("os.path.exists", return_value=False)

        gen_instance._prepare_output_dir("new/dir")

        # Check that we first checked existence
        mock_exists.assert_called_once_with("new/dir")
        # Check that we then created the dir
        mock_os_makedirs.assert_called_once_with("new/dir")

    def test_prepare_output_dir_uses_existing(
        self, gen_instance, mock_os_makedirs, mocker
    ):
        """
        Tests that _prepare_output_dir does NOT call os.makedirs if
        the dir already exists.
        """
        # Mock os.path.exists to return True
        mock_exists = mocker.patch("os.path.exists", return_value=True)

        gen_instance._prepare_output_dir("existing/dir")

        # Check that we first checked existence
        mock_exists.assert_called_once_with("existing/dir")
        # Check that we did NOT create the dir
        mock_os_makedirs.assert_not_called()

    def test_generate_grid_coordinates(self, gen_instance):
        """
        Tests the _generate_grid_coordinates method for correct shape
        and values.
        """
        # gen_instance extents are 0-100. Resolution 25.
        # mgrid[0:100:25] -> [0, 25, 50, 75] (4 steps)
        # Shape should be (4, 4)
        grid_x, grid_y = gen_instance._generate_grid_coordinates(resolution=25)

        assert grid_x.shape == (4, 4)
        assert grid_y.shape == (4, 4)

        # Check the values (before transposition)
        # grid_x should be constant along columns (Y-axis)
        expected_x = np.array(
            [[0, 0, 0, 0], [25, 25, 25, 25], [50, 50, 50, 50], [75, 75, 75, 75]]
        )
        # grid_y should be constant along rows (X-axis)
        expected_y = np.array(
            [[0, 25, 50, 75], [0, 25, 50, 75], [0, 25, 50, 75], [0, 25, 50, 75]]
        )

        np.testing.assert_allclose(grid_x, expected_x)
        np.testing.assert_allclose(grid_y, expected_y)

    def test_transpose_grids(self, gen_instance):
        """
        Tests that _transpose_grids correctly transposes X, Y, and Z
        grids.
        """
        # Create non-square grids (e.g., 3 X-steps, 2 Y-steps)
        x = np.array([[10, 10], [20, 20], [30, 30]])
        y = np.array([[100, 200], [100, 200], [100, 200]])
        z = np.array([[1, 2], [3, 4], [5, 6]])

        assert x.shape == (3, 2)  # (X, Y)

        t_x, t_y, t_z = gen_instance._transpose_grids(x, y, z)

        # Shape should be (2, 3) (Y, X)
        assert t_x.shape == (2, 3)
        assert t_y.shape == (2, 3)
        assert t_z.shape == (2, 3)

        # Check values
        assert t_x[0, 1] == x[1, 0]  # (y=0, x=1) -> (x=1, y=0)
        assert t_y[1, 0] == y[0, 1]  # (y=1, x=0) -> (x=0, y=1)
        assert t_z[1, 2] == z[2, 1]  # (y=1, x=2) -> (x=2, y=1)

        expected_t_z = np.array([[1, 3, 5], [2, 4, 6]])
        np.testing.assert_allclose(t_z, expected_t_z)

    def test_generate_z_grid_scalar(self, gen_instance):
        """
        Tests _generate_z_grid with int and float scalar values.
        """
        # Dummy grid_x to define the shape
        grid_x = np.ones((5, 5))
        # grid_y is only used for interpolation, so can be None here

        # Test with int
        grid_z_int = gen_instance._generate_z_grid(
            z=10, grid_x=grid_x, grid_y=None
        )
        assert grid_z_int.shape == (5, 5)
        np.testing.assert_allclose(grid_z_int, np.full((5, 5), 10))

        # Test with float
        grid_z_float = gen_instance._generate_z_grid(
            z=-1.5, grid_x=grid_x, grid_y=None
        )
        assert grid_z_float.shape == (5, 5)
        np.testing.assert_allclose(grid_z_float, np.full((5, 5), -1.5))

    def test_generate_z_grid_interpolation_nan(self, gen_instance):
        """
        Tests _generate_z_grid interpolation, ensuring points outside
        the convex hull of input data become NaN.
        """
        # Input points form a triangle: (0,0), (10,0), (0,10)
        z_points = np.array(
            [[0, 10, 0], [0, 0, 10], [100, 200, 300]]  # x  # y  # z
        )

        # Target grid: (0,0), (10,0), (0,10), and (10,10)
        # (10,10) is outside the triangle
        grid_x, grid_y = np.mgrid[0:11:10, 0:11:10]  # (0, 10) x (0, 10)

        assert grid_x.shape == (2, 2)

        grid_z = gen_instance._generate_z_grid(z_points, grid_x, grid_y)

        # Check the values
        # (0, 0) -> 100
        assert grid_z[0, 0] == pytest.approx(100)
        # (10, 0) -> 200
        assert grid_z[1, 0] == pytest.approx(200)
        # (0, 10) -> 300
        assert grid_z[0, 1] == pytest.approx(300)
        # (10, 10) -> Should be NaN
        assert np.isnan(grid_z[1, 1])

    def test_generate_z_grid_bad_type_explicit(self, gen_instance):
        """
        Explicitly tests that _generate_z_grid raises TypeError
        for invalid z.
        """
        grid_x = np.ones((2, 2))
        grid_y = np.ones((2, 2))

        with pytest.raises(
            TypeError, match="must be int, float, or np.ndarray"
        ):
            gen_instance._generate_z_grid(
                z={"this": "is bad"}, grid_x=grid_x, grid_y=grid_y
            )
