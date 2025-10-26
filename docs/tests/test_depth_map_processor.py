import os
import pytest
import numpy as np
import pandas as pd
import zarr
from unittest.mock import MagicMock, call, patch

# --- Test Setup: Handle Optional CuPy Import ---
try:
    import cupy as cp
    # Use cucim for GPU-accelerated morphology and labeling
    from cucim.skimage.morphology import binary_closing
    from cucim.skimage.measure import label
    # Use standard skimage for find_contours as specified
    from skimage.measure import find_contours 
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None for linter
    binary_closing = None
    label = None
    find_contours = None # This will be None, but the mock patches it anyway

# Import the class to be tested
from poseidon_core.depth_map_processor import DepthMapProcessor

# --- Pytest Markers ---
skip_if_no_cupy = pytest.mark.skipif(
    not CUPY_AVAILABLE, reason="CuPy/cuCIM or scikit-image is not installed."
)


@skip_if_no_cupy
class TestDepthMapProcessor:
    """
    Test suite for the DepthMapProcessor class.
    All tests require CuPy/cuCIM to be installed.
    """

    ## --- Fixtures ---

    @pytest.fixture
    def sample_elev_grid_np(self):
        """
        Provides a 12x12 NumPy elevation grid.
        Creates an 8x8 pond (64 pixels > 55) with a 2-pixel border.
        """
        grid = np.full((12, 12), 10.0)
        # 8x8 pond at [2:10, 2:10] (leaves 2-pixel border)
        grid[2:10, 2:10] = 5.0
        # 2x2 deep spot inside the pond
        grid[4:6, 4:6] = 1.0
        return grid

    @pytest.fixture
    def sample_label_array_cp(self, sample_elev_grid_np):
        """
        Provides a 12x12 CuPy binary label array.
        Water (1) is where elevation is <= 5.
        """
        grid_cp = cp.array(sample_elev_grid_np)
        label_array = (grid_cp <= 5).astype(cp.uint8)
        # Reshape to 12x12x1
        return label_array.reshape(12, 12, 1)

    @pytest.fixture
    def processor_instance(self, sample_elev_grid_np):
        """
        Provides a standard DepthMapProcessor instance based on the 12x12 grid.
        """
        processor = DepthMapProcessor(
            elevation_grid=sample_elev_grid_np,
            pond_edge_elev_plot_dir='test_data/plots'
        )
        processor.elev_grid_cp = cp.array(sample_elev_grid_np)
        return processor

    @pytest.fixture
    def local_10x10_processor(self):
        """
        Provides a 10x10 processor for tests that are
        hard-coded to that shape, decoupling them from the main fixture.
        """
        grid = np.full((10, 10), 10.0)
        grid[1:9, 1:9] = 5.0  # 8x8 pond
        grid[4:6, 4:6] = 1.0  # 2x2 deep spot
        
        processor = DepthMapProcessor(
            elevation_grid=grid,
            pond_edge_elev_plot_dir='test_data/plots'
        )
        processor.elev_grid_cp = cp.array(grid)
        return processor

    @pytest.fixture
    def mock_zarr(self, mocker):
        """Mocks zarr.open and zarr.open_group."""
        mock_open = mocker.patch("zarr.open", return_value=MagicMock())
        mock_group_store = MagicMock()
        mock_group_store.__setitem__ = MagicMock()
        mock_open_group = mocker.patch(
            "zarr.open_group", return_value=mock_group_store
        )
        return mock_open, mock_open_group, mock_group_store

    @pytest.fixture
    def mock_os(self, mocker):
        """Mocks os file system interactions."""
        mock_listdir = mocker.patch("os.listdir", return_value=[])
        mock_makedirs = mocker.patch("os.makedirs")
        mocker.patch("os.path.join", lambda *args: "/".join(args))
        return mock_listdir, mock_makedirs

    @pytest.fixture
    def mock_plt(self, mocker):
        """Mocks matplotlib plotting functions."""
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
        mock_close = mocker.patch("matplotlib.pyplot.close")
        mocker.patch("matplotlib.pyplot.figure")
        mocker.patch("matplotlib.pyplot.hist")
        mocker.patch("matplotlib.pyplot.axvline")
        mocker.patch("matplotlib.pyplot.xlim")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")
        mocker.patch("matplotlib.pyplot.title")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.grid")
        return mock_savefig, mock_close

    @pytest.fixture
    def mock_find_contours(self, mocker):
        """
        Mocks skimage.measure.find_contours.
        Returns a contour for an 8x8 pond at [1:9, 1:9].
        """
        contour_array = np.array([
            [0.5, 0.5], [0.5, 8.5], [8.5, 8.5], [8.5, 0.5], [0.5, 0.5]
        ])
        # This patch target is correct because the class uses
        # `from skimage.measure import find_contours`
        return mocker.patch(
            "poseidon_core.depth_map_processor.find_contours",
            return_value=[contour_array]
        )

    ## --- Test Cases ---

    def test_init(self, sample_elev_grid_np):
        """Tests that the elevation grid is set correctly (12x12)."""
        processor = DepthMapProcessor(sample_elev_grid_np)
        np.testing.assert_allclose(processor.elev_grid, sample_elev_grid_np)
        cp.testing.assert_allclose(
            processor.elev_grid_cp, cp.array(sample_elev_grid_np)
        )
        assert processor.pond_edge_elev_plot_dir == 'data/edge_historgrams'

    def test_label_ponds_basic(self, processor_instance, sample_label_array_cp):
        """
        Tests that a simple pond with a 2-pixel border survives
        binary closing and is labeled correctly.
        """
        # sample_label_array_cp is 12x12x1 with a pond at [2:10, 2:10]
        labeled_data = processor_instance._label_ponds(sample_label_array_cp)

        # Expected: 12x12 array, 8x8 pond at [2:10, 2:10] is label 1
        expected = cp.zeros((12, 12), dtype=int)
        expected[2:10, 2:10] = 1

        cp.testing.assert_allclose(labeled_data, expected)

    def test_label_ponds_small_pond_removal(self, processor_instance):
        """
        Tests that ponds smaller than min_size (55) are removed.
        (This test is independent of the fixture)
        """
        label_array = cp.zeros((12, 12, 1), dtype=cp.uint8)
        label_array[1:9, 1:9] = 1  # Large pond (8x8 = 64 pixels > 55)
        label_array[10:12, 10:12] = 1  # Small pond (2x2 = 4 pixels < 55)

        # We use the 12x12 processor instance, which is fine
        labeled_data = processor_instance._label_ponds(label_array)

        unique_labels = cp.unique(labeled_data).get()
        assert len(unique_labels) == 2  # [0, 1]
        assert 1 in unique_labels
        assert labeled_data[10, 10] == 0  # Small pond removed
        assert labeled_data[1, 1] == 1    # Large pond kept

    def test_label_ponds_fills_holes(self, processor_instance):
        """
        Tests that binary_closing correctly fills holes in a pond.
        """
        label_array = cp.zeros((12, 12, 1), dtype=cp.uint8)
        
        # --- FIX 1 ---
        # 8x8 pond (64 pixels) with a 2-pixel border
        label_array[2:10, 2:10] = 1 # Was [1:11, 1:11]
        
        # 2x2 hole in the middle (still inside the 8x8 pond)
        label_array[5:7, 5:7] = 0

        # Pass to the 12x12 processor
        labeled_data = processor_instance._label_ponds(label_array)

        # Expected: 12x12 array, 8x8 pond is labeled 1, hole is FILLED
        expected = cp.zeros((12, 12), dtype=int)
        
        # --- FIX 2 ---
        expected[2:10, 2:10] = 1 # Was [1:11, 1:11]

        cp.testing.assert_allclose(labeled_data, expected)

    def test_extract_contours(
        self, local_10x10_processor, mock_find_contours
    ):
        """
        Tests contour extraction. Uses a local 10x10 processor
        to match the hard-coded mock contour.
        """
        # 1. Create 10x10 data
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # One pond with ID 1
        
        # Create the matching 10x10 label array
        label_array_cp = (local_10x10_processor.elev_grid_cp <= 5)\
            .astype(cp.uint8).reshape(10, 10, 1)

        # 2. Call method on the local_10x10_processor
        pixels_dict, values_dict = local_10x10_processor._extract_contours(
            labeled_data, label_array_cp
        )

        # 3. Check contour pixels
        # Mock contour [0.5, 8.5] rounds to [0, 8]
        expected_pixels = np.array([
            [0, 0], [0, 8], [8, 8], [8, 0], [0, 0]
        ])
        np.testing.assert_allclose(pixels_dict[1], expected_pixels)

        # 4. Check contour values
        # Pixel (8,8) is inside the 10x10 grid's pond [1:9, 1:9] -> 5.0
        # Others are outside -> 0.0
        expected_values = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        np.testing.assert_allclose(values_dict[1], expected_values)

    def test_calculate_depths_wse_mean(self, local_10x10_processor):
        """
        Tests 'wse' (water surface) calculation with 'mean'.
        Uses a local 10x10 processor.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # Pond 1

        contour_values = {1: np.array([10.0, 10.0, 5.0, 5.0])}  # Mean = 7.5

        depth_maps = local_10x10_processor._calculate_depths(
            labeled_data, contour_values, "mean", "wse"
        )

        expected = cp.full((10, 10), cp.nan)
        expected[1:9, 1:9] = 7.5

        cp.testing.assert_allclose(depth_maps[1], expected)

    def test_calculate_depths_depth_95perc(self, local_10x10_processor):
        """
        Tests 'depth' calculation with '95_perc' and clipping.
        Uses a local 10x10 processor.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # Pond 1

        contour_values = {1: np.array([5.0, 5.0, 5.0, 10.0])} # WSE = 9.25

        depth_maps = local_10x10_processor._calculate_depths(
            labeled_data, contour_values, "95_perc", "depth"
        )

        # WSE = 9.25
        # elev_grid (most) = 5.0 -> abs(5.0 - 9.25) = 4.25
        # elev_grid (center) = 1.0 -> abs(1.0 - 9.25) = 8.25
        expected = cp.full((10, 10), cp.nan)
        expected[1:9, 1:9] = 4.25
        expected[4:6, 4:6] = 8.25

        cp.testing.assert_allclose(depth_maps[1], expected)

    def test_calculate_depths_depth_clipping(self, local_10x10_processor):
        """
        Tests that depths > 0 (elev > WSE) are clipped to 0.
        Uses a local 10x10 processor.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # Pond 1

        contour_values = {1: np.array([4.0, 4.0, 4.0])}  # WSE (mean) = 4.0

        depth_maps = local_10x10_processor._calculate_depths(
            labeled_data, contour_values, "mean", "depth"
        )

        # WSE = 4.0
        # elev (most) = 5.0 -> 5.0 - 4.0 = 1.0 -> clips to 0.0
        # elev (center) = 1.0 -> 1.0 - 4.0 = -3.0 -> abs() = 3.0
        expected = cp.full((10, 10), cp.nan)
        expected[1:9, 1:9] = 0.0
        expected[4:6, 4:6] = 3.0

        cp.testing.assert_allclose(depth_maps[1], expected)

    def test_calculate_depths_no_ponds(self, local_10x10_processor):
        """
        Tests calculation when no ponds are found.
        Uses a local 10x10 processor.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        contour_values = {}

        depth_maps = local_10x10_processor._calculate_depths(
            labeled_data, contour_values, "mean", "depth"
        )

        assert 1 in depth_maps
        expected = cp.full((10, 10), cp.nan)
        cp.testing.assert_allclose(depth_maps[1], expected)

    def test_plot_pond_edge_elevations(
        self, local_10x10_processor, mock_os, mock_plt
    ):
        """
        Tests that plotting functions are called correctly.
        Uses a local 10x10 processor.
        """
        mock_savefig, mock_close = mock_plt
        _, mock_makedirs = mock_os

        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1, 1] = 1  # Pond 1
        labeled_data[3, 3] = 2  # Pond 2
        contour_values = {
            1: np.array([5.0, 6.0]),
            2: np.array([7.0, 8.0])
        }
        file_name = "test_file"

        local_10x10_processor._plot_pond_edge_elevations(
            labeled_data, contour_values, file_name
        )

        expected_dirs = [
            call('test_data/plots/all_ponds', exist_ok=True),
            call('test_data/plots/ind_ponds', exist_ok=True)
        ]
        mock_makedirs.assert_has_calls(expected_dirs)

        assert mock_savefig.call_count == 3
        expected_saves = [
            call('test_data/plots/ind_ponds/test_file_Pond_1'),
            call('test_data/plots/ind_ponds/test_file_Pond_2'),
            call('test_data/plots/all_ponds/test_file_All_Ponds_Histogram')
        ]
        mock_savefig.assert_has_calls(expected_saves)
        assert mock_close.call_count == 3

    def test_combine_depth_maps(self, local_10x10_processor):
        """
        Tests combining overlapping depth maps.
        Uses a local 10x10 processor.
        """
        map1 = cp.full((10, 10), cp.nan)
        map1[1, 1] = 5.0
        map1[2, 2] = 3.0

        map2 = cp.full((10, 10), cp.nan)
        map2[1, 1] = 10.0
        map2[3, 3] = 1.0

        pond_depths = {1: map1, 2: map2}

        combined = local_10x10_processor.combine_depth_maps(pond_depths)

        expected = cp.full((10, 10), cp.nan)
        expected[1, 1] = 10.0
        expected[2, 2] = 3.0
        expected[3, 3] = 1.0

        cp.testing.assert_allclose(combined, expected)

    def test_combine_depth_maps_empty(self, local_10x10_processor):
        """
        Tests combining an empty dict of maps.
        Uses a local 10x10 processor.
        """
        combined = local_10x10_processor.combine_depth_maps({})
        expected = cp.full((10, 10), cp.nan)
        cp.testing.assert_allclose(combined, expected)

    def test_save_depth_maps(self, local_10x10_processor, mock_zarr):
        """
        Tests saving a dataframe of depth maps to Zarr.
        Uses a local 10x10 processor (though not strictly necessary).
        """
        _, mock_open_group, mock_store = mock_zarr

        depth_map_cp = cp.array([[1.0, 2.0], [3.0, 4.0]])
        depth_map_np = depth_map_cp.get()

        df = pd.DataFrame([
            {"image_name": "map_A", "depth_map": depth_map_cp}
        ])

        local_10x10_processor._save_depth_maps(df, "test_dir/output.zarr")

        mock_open_group.assert_called_once_with(
            "test_dir/output.zarr", mode="a"
        )

        mock_store.__setitem__.assert_called_once()
        args, kwargs = mock_store.__setitem__.call_args
        assert args[0] == "map_A"
        np.testing.assert_allclose(args[1], depth_map_np)

    def test_process_file_integration(
        self, processor_instance, mock_zarr, mock_plt
    ):
        """
        Integration test for process_file, mocking helper methods.
        Uses the main 12x12 fixture.
        """
        # --- Mock Inputs (12x12) ---
        mock_open, _, _ = mock_zarr
        mock_store_content = np.zeros((12, 12, 1), dtype=np.uint8)
        mock_store_content[2:10, 2:10, 0] = 1  # 8x8 pond at [2:10, 2:10]
        mock_open.return_value.__getitem__.return_value = mock_store_content

        # --- Mock Internal Methods ---
        with patch.object(
            processor_instance, '_label_ponds'
        ) as mock_label, \
             patch.object(
            processor_instance, '_extract_contours'
        ) as mock_extract, \
             patch.object(
            processor_instance, '_plot_pond_edge_elevations'
        ) as mock_plot, \
             patch.object(
            processor_instance, '_calculate_depths'
        ) as mock_calc, \
             patch.object(
            processor_instance, 'combine_depth_maps'
        ) as mock_combine:

            # --- Define Mock Outputs (12x12) ---
            mock_labeled_data = cp.ones((12, 12)) # 12x12
            mock_label.return_value = mock_labeled_data
            mock_contours_val = {1: np.array([5.0])}
            mock_calc.return_value = {"pond_1_depths": cp.array([1.0])}
            mock_combine.return_value = cp.array([99.0])
            mock_extract.return_value = (
                {"ignored_pixels": 1}, mock_contours_val
            )

            # --- Call Method ---
            result_list = processor_instance.process_file(
                "test_dir/label.zarr", "label_file_name"
            )

            # --- Assertions ---
            mock_open.assert_called_once_with("test_dir/label.zarr")
            mock_label.assert_called_once()
            
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            cp.testing.assert_allclose(args[0], mock_labeled_data)
            # Check that the 12x12 mock store content was passed
            cp.testing.assert_allclose(args[1], cp.array(mock_store_content))

            mock_plot.assert_called_once_with(
                mock_labeled_data, mock_contours_val, "label_file_name"
            )

            assert mock_calc.call_count == 8
            assert mock_combine.call_count == 8

            assert len(result_list) == 8
            assert result_list[0]['image_name'] == \
                "label_file_name_wse_map_mean"
            cp.testing.assert_allclose(
                result_list[0]['depth_map'], cp.array([99.0])
            )

    def test_process_depth_maps_integration(
        self, processor_instance, mock_os
    ):
        """
        Top-level integration test for process_depth_maps.
        Uses the main 12x12 fixture.
        """
        mock_listdir, _ = mock_os

        mock_listdir.return_value = [
            "file_A_rectified", "file_B.txt", "file_C_rectified"
        ]

        with patch.object(
            processor_instance, 'process_file'
        ) as mock_process, \
             patch.object(
            processor_instance, '_save_depth_maps'
        ) as mock_save:

            mock_process.side_effect = [
                [{"image_name": "A_map", "depth_map": cp.array([1])}],
                [{"image_name": "C_map", "depth_map": cp.array([2])}],
            ]

            processor_instance.process_depth_maps(
                "labels_dir", "depth_dir"
            )

            mock_listdir.assert_called_once_with("labels_dir")

            assert mock_process.call_count == 2
            mock_process.assert_has_calls([
                call("labels_dir/file_A_rectified", "file_A_rectified"),
                call("labels_dir/file_C_rectified", "file_C_rectified")
            ])

            assert mock_save.call_count == 2