import os
import pytest
import numpy as np
import cv2
import zarr

# --- Test Setup: Handle Optional CuPy Import ---
try:
    import cupy as cp
    import cupyx.scipy.interpolate

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None to avoid linter errors

# Import the class to be tested
from poseidon_core import ImageRectifier

# --- Pytest Fixtures ---


@pytest.fixture(
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not CUPY_AVAILABLE,
                reason="CuPy or cupyx.scipy.interpolate is not installed.",
            ),
        ),
    ]
)
def use_gpu(request):
    """Parametrized fixture to test both CPU (False) and GPU (True) paths."""
    return request.param


@pytest.fixture
def xp(use_gpu):
    """Fixture to provide the correct array module (np or cp)."""
    return cp if use_gpu else np


@pytest.fixture
def sample_intrinsics():
    """Provides a sample 11-element intrinsics array."""
    # [NU, NV, c0U, c0V, fx, fy, d1, d2, d3, t1, t2]
    return np.array(
        [
            1920,
            1080,
            960,
            540,  # Image size and principal point
            1000,
            1000,  # Focal lengths
            0.1,
            0.01,
            0,
            0,
            0,  # Distortion coefficients (small)
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_extrinsics():
    """Provides a sample 6-element extrinsics array."""
    # [X, Y, Z, Azimuth, Tilt, Swing]
    # 100m high, 45-degree tilt
    return np.array([0, 0, 100, 0, np.pi / 4, 0], dtype=np.float64)


@pytest.fixture
def sample_grids():
    """Provides sample 10x10 X, Y, Z grids."""
    shape = (10, 10)
    x = np.linspace(-50, 50, shape[1])
    y = np.linspace(-50, 50, shape[0])
    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = np.zeros(shape, dtype=np.float64)
    return grid_x, grid_y, grid_z


@pytest.fixture
def rectifier_instance(
    sample_intrinsics, sample_extrinsics, sample_grids, use_gpu
):
    """
    Creates a full ImageRectifier instance for testing.
    This fixture implicitly tests the entire __init__ chain.
    """
    grid_x, grid_y, grid_z = sample_grids
    return ImageRectifier(
        sample_intrinsics,
        sample_extrinsics,
        grid_x,
        grid_y,
        grid_z,
        use_gpu=use_gpu,
    )


@pytest.fixture
def dummy_image_file(tmp_path):
    """Creates a dummy 100x100 RGB image file and returns its path."""
    img_path = tmp_path / "test_image.png"
    # Create a simple gradient image for predictable interpolation
    y, x = np.indices((100, 100))
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[..., 0] = x * 2.5  # Red gradient
    img[..., 1] = y * 2.5  # Green gradient
    img[..., 2] = 128  # Blue constant
    cv2.imwrite(str(img_path), img)
    return img_path, img


@pytest.fixture
def dummy_image_folder(tmp_path):
    """Creates a folder with 3 dummy images."""
    folder_path = tmp_path / "image_folder"
    folder_path.mkdir()
    num_images = 3
    for i in range(num_images):
        img_path = folder_path / f"img_{i}.png"
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
    return folder_path, num_images


# --- Test Cases ---


def test_initialization(
    rectifier_instance, sample_extrinsics, sample_grids, use_gpu, xp
):
    """
    Test that the __init__ method correctly sets all attributes
    and that their types (np vs cp) are correct.
    """
    rect = rectifier_instance
    grid_x, _, _ = sample_grids
    grid_shape = grid_x.shape

    assert rect.use_gpu is use_gpu
    assert isinstance(rect.intrinsics, xp.ndarray)
    assert isinstance(rect.extrinsics, xp.ndarray)
    assert isinstance(rect.grid_x, xp.ndarray)
    assert isinstance(rect.grid_y, xp.ndarray)
    assert isinstance(rect.grid_z, xp.ndarray)

    assert rect.azimuth == sample_extrinsics[3]
    assert rect.tilt == sample_extrinsics[4]
    assert rect.swing == sample_extrinsics[5]
    assert rect.grid_shape == grid_shape

    # Test that the computed UV maps were created
    assert rect.Ud is not None
    assert rect.Vd is not None
    assert rect.Ud.shape == grid_shape
    assert rect.Vd.shape == grid_shape
    assert rect.Ud.dtype == int
    assert rect.Vd.dtype == int


def test_load_image(rectifier_instance, dummy_image_file, use_gpu, xp):
    """Test loading both color and grayscale images."""
    img_path, _ = dummy_image_file

    # Test loading in color (default)
    img_color = rectifier_instance._load_image(str(img_path), labels=False)
    assert isinstance(img_color, xp.ndarray)
    assert img_color.ndim == 3
    assert img_color.shape == (100, 100, 3)

    # Test loading in grayscale (labels=True)
    img_gray = rectifier_instance._load_image(str(img_path), labels=True)
    assert isinstance(img_gray, xp.ndarray)
    assert img_gray.ndim == 2
    assert img_gray.shape == (100, 100)


def test_reshape_grids(rectifier_instance, xp):
    """Test the grid reshaping logic."""
    xyz = rectifier_instance._reshape_grids()
    assert isinstance(xyz, xp.ndarray)
    # 10x10 grid = 100 points. Shape should be (100, 3)
    assert xyz.shape == (100, 3)


def test_cirn_angles_to_r(rectifier_instance, xp):
    """Test rotation matrix calculation."""
    R = rectifier_instance._CIRN_angles_to_R()
    assert isinstance(R, xp.ndarray)
    assert R.shape == (3, 3)


# FIX 1: Add `use_gpu` to the function signature
def test_build_k(rectifier_instance, sample_intrinsics, xp, use_gpu):
    """Test intrinsic matrix (K) construction."""
    K = rectifier_instance._build_K()
    assert isinstance(K, xp.ndarray)
    assert K.shape == (3, 3)

    # Check that values were assigned correctly
    if use_gpu:
        K = K.get()  # Move to CPU for numpy comparison

    assert K[0, 0] == -sample_intrinsics[4]  # -fx
    assert K[1, 1] == -sample_intrinsics[5]  # -fy
    assert K[0, 2] == sample_intrinsics[2]  # c0U
    assert K[1, 2] == sample_intrinsics[3]  # c0V
    assert K[2, 2] == 1


def test_intrinsics_extrinsics_to_p(rectifier_instance, xp):
    """Test projection matrix (P) calculation."""
    P, K, R, IC = rectifier_instance._intrinsics_extrinsics_to_P()

    assert isinstance(P, xp.ndarray)
    assert P.shape == (3, 4)
    assert isinstance(K, xp.ndarray)
    assert K.shape == (3, 3)
    assert isinstance(R, xp.ndarray)
    assert R.shape == (3, 3)
    assert isinstance(IC, xp.ndarray)
    assert IC.shape == (3, 4)


def test_get_pixels_logic(rectifier_instance, xp, use_gpu):
    """
    Test the core interpolation logic of _get_pixels.
    We "monkey-patch" the Ud and Vd attributes to force known
    sampling points for a predictable result.
    """
    rect = rectifier_instance

    # Create a simple 10x10 gradient image
    # Values from 0 to 99
    img_data = np.arange(100).reshape(10, 10).astype(np.float32)
    img = xp.asarray(img_data)

    # --- Test 1: Identity mapping (sample known pixels) ---

    # FIX 2: Change coordinates to be inside the valid mask
    # (Vd > 1 and Ud > 1) and (Vd < shape[0] and Ud < shape[1])
    # Let's sample points:
    # (Vd=2, Ud=2) -> value 22
    # (Vd=2, Ud=3) -> value 23
    # (Vd=5, Ud=2) -> value 52
    # (Vd=8, Ud=8) -> value 88
    rect.Vd = xp.array([[2, 2], [5, 8]])  # Rows (all valid)
    rect.Ud = xp.array([[2, 3], [2, 8]])  # Cols (all valid)

    pixels = rect._get_pixels(img)

    # FIX 3: Update expected values to match new coordinates
    expected_values = xp.array([[[22.0], [23.0]], [[52.0], [88.0]]])

    xp.testing.assert_allclose(pixels, expected_values)

    # --- Test 2: Out-of-bounds mapping ---
    # Sample points: (row, col)
    # (0, 5)   -> out (Vd=0 is masked)
    # (5, 100) -> out (Ud=100 is > shape)
    # (5, 5)   -> in (value 55)
    # (1, 1)   -> out (Vd=1 is masked)
    rect.Vd = xp.array([[0, 5], [5, 1]])
    rect.Ud = xp.array([[5, 100], [5, 1]])

    pixels = rect._get_pixels(img)

    # GPU uses NaN, CPU uses 0 for fill
    fill_val = xp.nan if use_gpu else 0.0

    expected_values = xp.array([[[fill_val], [fill_val]], [[55.0], [fill_val]]])

    # FIX: Handle np vs cp assertion
    if use_gpu:
        # cupy.testing.assert_allclose doesn't support equal_nan
        # 1. Test that NaN locations are identical
        xp.testing.assert_array_equal(
            xp.isnan(pixels), xp.isnan(expected_values)
        )

        # 2. Test that non-NaN values are close (fill NaNs with 0)
        pixels_nonan = xp.nan_to_num(pixels, nan=0.0)
        expected_nonan = xp.nan_to_num(expected_values, nan=0.0)
        xp.testing.assert_allclose(pixels_nonan, expected_nonan)
    else:
        # numpy.testing.assert_allclose *does* support equal_nan
        xp.testing.assert_allclose(pixels, expected_values, equal_nan=True)

    # --- Test 3: 3-Channel image ---
    img_3c_data = np.stack([img_data, img_data * 2, img_data + 10], axis=-1)
    img_3c = xp.asarray(img_3c_data)

    # Sample point (Vd=2, Ud=4) -> val=24
    rect.Vd = xp.array([[2]])
    rect.Ud = xp.array([[4]])

    pixels = rect._get_pixels(img_3c)

    # Expected: [val, val*2, val+10] where val = 24
    expected_values = xp.array([[[24.0, 48.0, 34.0]]])
    xp.testing.assert_allclose(pixels, expected_values)


def test_merge_rectify(rectifier_instance, dummy_image_file, xp):
    """End-to-end test for rectifying a single image."""
    img_path, _ = dummy_image_file
    grid_shape = rectifier_instance.grid_shape

    # Test color rectification
    rect_img = rectifier_instance.merge_rectify(str(img_path), labels=False)

    assert isinstance(rect_img, xp.ndarray)
    assert rect_img.dtype == np.uint8
    assert rect_img.shape == (grid_shape[0], grid_shape[1], 3)

    # Test grayscale (label) rectification
    rect_label = rectifier_instance.merge_rectify(str(img_path), labels=True)

    assert isinstance(rect_label, xp.ndarray)
    assert rect_label.dtype == np.uint8
    # 2D images get a channel dim added by _get_pixels
    assert rect_label.shape == (grid_shape[0], grid_shape[1], 1)


def test_merge_rectify_folder(
    rectifier_instance, dummy_image_folder, tmp_path, mocker
):
    """End-to-end test for batch processing a folder to a Zarr store."""

    folder_path, num_images = dummy_image_folder
    zarr_store_path = str(tmp_path / "output.zarr")
    grid_shape = rectifier_instance.grid_shape

    rectifier_instance.merge_rectify_folder(
        str(folder_path), zarr_store_path, labels=False
    )

    # Check that the Zarr store was created and has the correct content
    assert os.path.exists(zarr_store_path)
    store = zarr.open_group(zarr_store_path, mode="r")

    # Check correct number of images
    assert len(list(store.keys())) == num_images

    # Check names and shapes
    assert "img_0_rectified" in store
    assert "img_1_rectified" in store
    assert "img_2_rectified" in store

    dset = store["img_0_rectified"]
    assert dset.shape == (grid_shape[0], grid_shape[1], 3)
