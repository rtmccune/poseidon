import os
import pytest
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
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

    # This will be None, but the mock patches it anyway
    find_contours = None

# Import the class to be tested
# (Update this import path to match your project structure)
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
        Provides a standard DepthMapProcessor instance.
        *** Plotting is DISABLED by default for tests. ***
        """
        processor = DepthMapProcessor(
            elevation_grid=sample_elev_grid_np,
            plot_edges=False,  # Disable plotting by default
            # pond_edge_elev_plot_dir="test_data/plots", <-- REMOVED
        )
        processor.elev_grid_cp = cp.array(sample_elev_grid_np)
        return processor

    @pytest.fixture
    def processor_with_plotting(self, sample_elev_grid_np):
        """
        Provides a DepthMapProcessor instance.
        *** Plotting is ENABLED for specific tests. ***
        """
        processor = DepthMapProcessor(
            elevation_grid=sample_elev_grid_np,
            plot_edges=True,  # <-- UPDATED: Enable plotting
            # pond_edge_elev_plot_dir="test_data/plots", <-- REMOVED
        )
        processor.elev_grid_cp = cp.array(sample_elev_grid_np)
        return processor

    @pytest.fixture
    def local_10x10_processor(self):
        """
        Provides a 10x10 processor for tests.
        *** Plotting is DISABLED by default for tests. ***
        """
        grid = np.full((10, 10), 10.0)
        grid[1:9, 1:9] = 5.0  # 8x8 pond
        grid[4:6, 4:6] = 1.0  # 2x2 deep spot

        processor = DepthMapProcessor(
            elevation_grid=grid,
            plot_edges=False,  # Disable plotting by default
            # pond_edge_elev_plot_dir="test_data/plots", <-- REMOVED
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
        contour_array = np.array(
            [[0.5, 0.5], [0.5, 8.5], [8.5, 8.5], [8.5, 0.5], [0.5, 0.5]]
        )
        return mocker.patch(
            "poseidon_core.depth_map_processor.find_contours",
            return_value=[contour_array],
        )

    ## --- Test Cases ---

    def test_init(self, sample_elev_grid_np):
        """Tests that init sets defaults correctly."""
        processor = DepthMapProcessor(sample_elev_grid_np)
        np.testing.assert_allclose(processor.elev_grid, sample_elev_grid_np)
        cp.testing.assert_allclose(
            processor.elev_grid_cp, cp.array(sample_elev_grid_np)
        )
        # assert processor.pond_edge_elev_plot_dir == "data/edge_histograms" <-- REMOVED
        assert processor.plot_edges is True  # Check default

    def test_init_plotting_disabled(self, sample_elev_grid_np):
        """Tests that init respects plot_edges=False."""
        processor = DepthMapProcessor(sample_elev_grid_np, plot_edges=False)
        assert processor.plot_edges is False  # Check explicit False

    def test_label_ponds_basic(self, processor_instance, sample_label_array_cp):
        """
        Tests that a simple pond with a 2-pixel border survives
        binary closing and is labeled correctly.
        """
        labeled_data = processor_instance._label_ponds(sample_label_array_cp)
        expected = cp.zeros((12, 12), dtype=int)
        expected[2:10, 2:10] = 1
        cp.testing.assert_allclose(labeled_data, expected)

    def test_label_ponds_small_pond_removal(self, processor_instance):
        """
        Tests that ponds smaller than min_size (55) are removed.
        """
        label_array = cp.zeros((12, 12, 1), dtype=cp.uint8)
        label_array[1:9, 1:9] = 1  # Large pond (8x8 = 64 pixels > 55)
        label_array[10:12, 10:12] = 1  # Small pond (2x2 = 4 pixels < 55)

        labeled_data = processor_instance._label_ponds(label_array)

        unique_labels = cp.unique(labeled_data).get()
        assert len(unique_labels) == 2  # [0, 1]
        assert labeled_data[10, 10] == 0  # Small pond removed
        assert labeled_data[1, 1] == 1  # Large pond kept

    def test_label_ponds_fills_holes(self, processor_instance):
        """
        Tests that binary_closing correctly fills holes in a pond.
        """
        label_array = cp.zeros((12, 12, 1), dtype=cp.uint8)
        label_array[2:10, 2:10] = 1  # 8x8 pond
        label_array[5:7, 5:7] = 0  # 2x2 hole in the middle

        labeled_data = processor_instance._label_ponds(label_array)

        expected = cp.zeros((12, 12), dtype=int)
        expected[2:10, 2:10] = 1  # Hole should be filled
        cp.testing.assert_allclose(labeled_data, expected)

    def test_extract_contours(self, local_10x10_processor, mock_find_contours):
        """
        Tests contour extraction. Uses a local 10x10 processor
        to match the hard-coded mock contour.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # One pond with ID 1

        label_array_cp = (
            (local_10x10_processor.elev_grid_cp <= 5)
            .astype(cp.uint8)
            .reshape(10, 10, 1)
        )

        pixels_dict, values_dict = local_10x10_processor._extract_contours(
            labeled_data, label_array_cp
        )

        # Mock contour [0.5, 8.5] rounds to [0, 8]
        expected_pixels = np.array([[0, 0], [0, 8], [8, 8], [8, 0], [0, 0]])
        np.testing.assert_allclose(pixels_dict[1], expected_pixels)

        # Pixel (8,8) is inside the 10x10 grid's pond [1:9, 1:9] -> 5.0
        # Others are outside -> 0.0
        expected_values = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
        np.testing.assert_allclose(values_dict[1], expected_values)

    def test_calculate_all_depths(self, local_10x10_processor):
        """
        Tests the new `_calculate_all_depths` method.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1:9, 1:9] = 1  # Pond 1

        # Use two different contour value sets to check different methods
        contour_values = {
            1: np.array([10.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 10.0])
        }
        # Mean = (3*10 + 7*5) / 10 = 6.5
        # 95th percentile = 10.0

        # --- Call the new method ---
        all_maps = local_10x10_processor._calculate_all_depths(
            labeled_data, contour_values
        )

        # --- Check 1: 'wse' with 'mean' (WSE = 6.5) ---
        expected_wse_mean = cp.full((10, 10), cp.nan)
        expected_wse_mean[1:9, 1:9] = 6.5
        cp.testing.assert_allclose(all_maps["wse_map_mean"], expected_wse_mean)

        # --- Check 2: 'depth' with '95_perc' (WSE = 10.0) ---
        # WSE = 10.0
        # elev_grid (most) = 5.0 -> abs(5.0 - 10.0) = 5.0
        # elev_grid (center) = 1.0 -> abs(1.0 - 10.0) = 9.0
        expected_depth_95 = cp.full((10, 10), cp.nan)
        expected_depth_95[1:9, 1:9] = 5.0
        expected_depth_95[4:6, 4:6] = 9.0
        cp.testing.assert_allclose(
            all_maps["depth_map_95_perc"], expected_depth_95
        )

        # --- Check 3: 'depth' with clipping (WSE = 6.5) ---
        # WSE = 6.5
        # elev (most) = 5.0 -> 5.0 - 6.5 = -1.5 -> abs() = 1.5
        # elev (center) = 1.0 -> 1.0 - 6.5 = -5.5 -> abs() = 5.5
        # local_10x10_processor has elev 10.0 outside pond, but that's NaN
        expected_depth_mean = cp.full((10, 10), cp.nan)
        expected_depth_mean[1:9, 1:9] = 1.5
        expected_depth_mean[4:6, 4:6] = 5.5
        cp.testing.assert_allclose(
            all_maps["depth_map_mean"], expected_depth_mean
        )

        assert len(all_maps) == 8  # Ensure all 8 maps were generated

    def test_calculate_all_depths_no_ponds(self, local_10x10_processor):
        """
        Tests `_calculate_all_depths` when no ponds are found.
        """
        labeled_data = cp.zeros((10, 10), dtype=int)
        contour_values = {}

        all_maps = local_10x10_processor._calculate_all_depths(
            labeled_data, contour_values
        )

        assert len(all_maps) == 8
        expected_empty = cp.full((10, 10), cp.nan)
        # Check one map, they should all be the same
        cp.testing.assert_allclose(all_maps["wse_map_mean"], expected_empty)

    def test_plot_histogram_helper_logic(self, local_10x10_processor, mock_plt):
        """
        Tests the core logic of the generic _plot_histogram_helper.
        Ensures it calculates stats and calls matplotlib functions
        correctly.
        """
        mock_savefig, mock_close = mock_plt
        # Get mocks for the functions we want to check
        mock_hist = patch.object(plt, "hist").start()
        mock_axvline = patch.object(plt, "axvline").start()
        mock_title = patch.object(plt, "title").start()

        # Simple data for easy-to-check stats
        # Mean = 2.0, Median = 1.5, 95th %ile = 3.7
        test_data = np.array([1.0, 1.0, 2.0, 4.0])
        test_bins = np.linspace(0, 5, 11)
        test_title = "My Test Title"
        test_path = "save/here/plot.png"
        test_color = "test_color"

        local_10x10_processor._plot_histogram_helper(
            test_data, test_bins, test_title, test_path, test_color
        )

        # Check that hist was called with the right data
        mock_hist.assert_called_once()
        np.testing.assert_array_equal(mock_hist.call_args[0][0], test_data)
        assert mock_hist.call_args[1]["bins"] is test_bins
        assert mock_hist.call_args[1]["color"] == test_color

        # --- Check axvline calls robustly ---
        assert mock_axvline.call_count == 3

        # Get all calls made to axvline
        all_calls = mock_axvline.call_args_list

        # Convert calls to a dict keyed by label for easier checking
        calls_by_label = {}
        for c in all_calls:
            args, kwargs = c
            # We assume 'label' is in kwargs, which it should be
            calls_by_label[kwargs["label"]] = {
                "value": args[0],
                "kwargs": kwargs,
            }

        # Check Mean
        mean_label = "Mean: 2.00"
        assert mean_label in calls_by_label
        # Mean is a precise float, so direct comparison is fine
        assert calls_by_label[mean_label]["value"] == 2.0
        assert calls_by_label[mean_label]["kwargs"]["color"] == "red"
        assert calls_by_label[mean_label]["kwargs"]["linestyle"] == "dashed"

        # Check Median
        median_label = "Median: 1.50"
        assert median_label in calls_by_label
        # Median is also a precise float here
        assert calls_by_label[median_label]["value"] == 1.5
        assert calls_by_label[median_label]["kwargs"]["color"] == "green"
        assert calls_by_label[median_label]["kwargs"]["linestyle"] == "solid"

        # Check 95th Percentile (the failing one)
        perc_label = "95th %ile: 3.70"
        assert perc_label in calls_by_label
        # Use assert_allclose for robust floating point comparison
        np.testing.assert_allclose(calls_by_label[perc_label]["value"], 3.7)
        assert calls_by_label[perc_label]["kwargs"]["color"] == "purple"
        assert calls_by_label[perc_label]["kwargs"]["linestyle"] == "dashdot"

        # Check title, save, and close
        mock_title.assert_called_once_with(test_title)
        mock_savefig.assert_called_once_with(test_path)
        mock_close.assert_called_once()

        # Stop the patches started in this test
        patch.stopall()

    def test_plot_histogram_helper_empty_data(
        self, local_10x10_processor, mock_plt
    ):
        """
        Tests that the histogram helper correctly handles empty data.
        """
        mock_savefig, mock_close = mock_plt
        mock_hist = patch.object(plt, "hist").start()

        local_10x10_processor._plot_histogram_helper(
            np.array([]), np.linspace(0, 1, 2), "Empty", "path", "color"
        )

        # Should not attempt to plot or save
        mock_hist.assert_not_called()
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()  # Doesn't even create a figure

        # Stop the patches started in this test
        patch.stopall()

    def test_plot_individual_histograms_delegation(
        self, local_10x10_processor, mocker
    ):
        """
        Tests that _plot_individual_histograms loops and calls the
        helper method with the correct arguments for each pond.
        """
        mock_helper = mocker.patch.object(
            local_10x10_processor, "_plot_histogram_helper"
        )

        test_bins = np.linspace(0, 10, 5)
        test_dir = "test/ind_dir"
        test_base = "file_base"
        test_data = {
            1: np.array([1, 2]),
            5: np.array([5, 6, 7]),
        }

        local_10x10_processor._plot_individual_histograms(
            test_data, test_bins, test_dir, test_base
        )

        assert mock_helper.call_count == 2
        expected_calls = [
            call(
                test_data[1],
                test_bins,
                "Elevation Histogram for Pond 1",
                "test/ind_dir/file_base_Pond_1",
                color="lightblue",
            ),
            call(
                test_data[5],
                test_bins,
                "Elevation Histogram for Pond 5",
                "test/ind_dir/file_base_Pond_5",
                color="lightblue",
            ),
        ]

        # Check calls manually to handle numpy array comparison
        call_1_args, call_1_kwargs = mock_helper.call_args_list[0]
        call_2_args, call_2_kwargs = mock_helper.call_args_list[1]

        expected_call_1_args = expected_calls[0].args
        expected_call_2_args = expected_calls[1].args

        np.testing.assert_array_equal(call_1_args[0], expected_call_1_args[0])
        assert call_1_args[1:] == expected_call_1_args[1:]
        assert call_1_kwargs == expected_calls[0].kwargs

        np.testing.assert_array_equal(call_2_args[0], expected_call_2_args[0])
        assert call_2_args[1:] == expected_call_2_args[1:]
        assert call_2_kwargs == expected_calls[1].kwargs

    def test_plot_combined_histogram_delegation(
        self, local_10x10_processor, mocker
    ):
        """
        Tests that _plot_combined_histogram calls the helper method
        once with the correct arguments.
        """
        mock_helper = mocker.patch.object(
            local_10x10_processor, "_plot_histogram_helper"
        )

        test_bins = np.linspace(0, 10, 5)
        test_dir = "test/all_dir"
        test_base = "file_base"
        test_data = np.array([1, 2, 5, 6, 7])

        local_10x10_processor._plot_combined_histogram(
            test_data, test_bins, test_dir, test_base
        )

        mock_helper.assert_called_once()

        # Check call arguments, handling numpy array separately
        args, kwargs = mock_helper.call_args
        expected_args = (
            test_data,
            test_bins,
            "Elevation Histogram - All Ponds Combined",
            "test/all_dir/file_base_All_Ponds_Histogram",
        )

        np.testing.assert_array_equal(args[0], expected_args[0])
        assert args[1:] == expected_args[1:]
        assert kwargs == {"color": "lightcoral"}

    def test_plot_pond_edge_elevations_orchestrator(
        self, local_10x10_processor, mock_os, mocker
    ):
        """
        Tests that the refactored _plot_pond_edge_elevations
        orchestrates directory creation and calls the new helper
        methods correctly.
        """
        _, mock_makedirs = mock_os
        mock_ind_plot = mocker.patch.object(
            local_10x10_processor, "_plot_individual_histograms"
        )
        mock_comb_plot = mocker.patch.object(
            local_10x10_processor, "_plot_combined_histogram"
        )

        # Pond 1: Has data
        # Pond 2: No data in contour_values
        # Pond 3: Has data
        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1, 1] = 1
        labeled_data[2, 2] = 2
        labeled_data[3, 3] = 3

        # Note: pond 2 is missing
        contour_values = {1: np.array([1.0, 2.0]), 3: np.array([3.0, 4.0])}
        file_name = "test_file"

        # --- ADDED ---
        # Set the attribute manually, as process_depth_maps() would
        local_10x10_processor.pond_edge_elev_plot_dir = "test_data/plots"
        # --- END ADDED ---

        local_10x10_processor._plot_pond_edge_elevations(
            labeled_data, contour_values, file_name
        )

        # 1. Check directory creation
        expected_dirs = [
            call("test_data/plots/all_ponds", exist_ok=True),
            call("test_data/plots/ind_ponds", exist_ok=True),
        ]
        mock_makedirs.assert_has_calls(expected_dirs)

        # 2. Check call to individual plotter (with filtered data)
        mock_ind_plot.assert_called_once()
        args_ind, _ = mock_ind_plot.call_args

        expected_filtered_dict = {1: contour_values[1], 3: contour_values[3]}
        assert args_ind[0].keys() == expected_filtered_dict.keys()
        np.testing.assert_array_equal(args_ind[0][1], expected_filtered_dict[1])
        np.testing.assert_array_equal(args_ind[0][3], expected_filtered_dict[3])
        assert args_ind[2] == "test_data/plots/ind_ponds"  # output dir
        assert args_ind[3] == file_name  # file_name_base

        # 3. Check call to combined plotter (with concatenated data)
        mock_comb_plot.assert_called_once()
        args_comb, _ = mock_comb_plot.call_args

        expected_concat_array = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(args_comb[0], expected_concat_array)
        assert args_comb[2] == "test_data/plots/all_ponds"  # output dir
        assert args_comb[3] == file_name  # file_name_base

    def test_plot_pond_edge_elevations_no_data(
        self, local_10x10_processor, mock_os, mocker
    ):
        """
        Tests that the orchestrator exits cleanly if no valid
        pond data is found.
        """
        _, mock_makedirs = mock_os
        mock_ind_plot = mocker.patch.object(
            local_10x10_processor, "_plot_individual_histograms"
        )
        mock_comb_plot = mocker.patch.object(
            local_10x10_processor, "_plot_combined_histogram"
        )

        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1, 1] = 1  # Pond 1 exists
        contour_values = {}  # But has no contour data

        # --- ADDED ---
        # Set the attribute manually, as process_depth_maps() would
        local_10x10_processor.pond_edge_elev_plot_dir = "test_data/plots"
        # --- END ADDED ---

        local_10x10_processor._plot_pond_edge_elevations(
            labeled_data, contour_values, "test_file"
        )

        # It should still make the directories
        assert mock_makedirs.call_count == 2

        # It should NOT call the plotters
        mock_ind_plot.assert_not_called()
        mock_comb_plot.assert_not_called()

    def test_plot_pond_edge_elevations(
        self, local_10x10_processor, mock_os, mock_plt
    ):
        """
        Tests that plotting functions are called correctly.
        (No change needed for this test)
        """
        mock_savefig, mock_close = mock_plt
        _, mock_makedirs = mock_os

        labeled_data = cp.zeros((10, 10), dtype=int)
        labeled_data[1, 1] = 1  # Pond 1
        labeled_data[3, 3] = 2  # Pond 2
        contour_values = {1: np.array([5.0, 6.0]), 2: np.array([7.0, 8.0])}
        file_name = "test_file"

        # --- ADDED ---
        # Set the attribute manually, as process_depth_maps() would
        local_10x10_processor.pond_edge_elev_plot_dir = "test_data/plots"
        # --- END ADDED ---

        local_10x10_processor._plot_pond_edge_elevations(
            labeled_data, contour_values, file_name
        )

        expected_dirs = [
            call("test_data/plots/all_ponds", exist_ok=True),
            call("test_data/plots/ind_ponds", exist_ok=True),
        ]
        mock_makedirs.assert_has_calls(expected_dirs)

        assert mock_savefig.call_count == 3
        expected_saves = [
            call("test_data/plots/ind_ponds/test_file_Pond_1"),
            call("test_data/plots/ind_ponds/test_file_Pond_2"),
            call("test_data/plots/all_ponds/test_file_All_Ponds_Histogram"),
        ]
        mock_savefig.assert_has_calls(expected_saves)
        assert mock_close.call_count == 3

    def test_combine_depth_maps(self, local_10x10_processor):
        """
        Tests combining overlapping depth maps.
        (No change needed for this test)
        """
        map1 = cp.full((10, 10), cp.nan)
        map1[1, 1] = 5.0
        map1[2, 2] = 3.0

        map2 = cp.full((10, 10), cp.nan)
        map2[1, 1] = 10.0
        map2[3, 3] = 1.0

        pond_depths = {1: map1, 2: map2}

        combined = local_10x10_processor._combine_depth_maps(pond_depths)

        expected = cp.full((10, 10), cp.nan)
        expected[1, 1] = 10.0
        expected[2, 2] = 3.0
        expected[3, 3] = 1.0

        cp.testing.assert_allclose(combined, expected)

    def test_combine_depth_maps_empty(self, local_10x10_processor):
        """
        Tests combining an empty dict of maps.
        (No change needed for this test)
        """
        combined = local_10x10_processor._combine_depth_maps({})
        expected = cp.full((10, 10), cp.nan)
        cp.testing.assert_allclose(combined, expected)

    def test_save_depth_maps(self, local_10x10_processor, mock_zarr):
        """
        Tests saving a dataframe of depth maps to Zarr.
        (No change needed for this test)
        """
        _, mock_open_group, mock_store = mock_zarr

        depth_map_cp = cp.array([[1.0, 2.0], [3.0, 4.0]])
        depth_map_np = depth_map_cp.get()

        df = pd.DataFrame([{"image_name": "map_A", "depth_map": depth_map_cp}])

        local_10x10_processor._save_depth_maps(df, "test_dir/output.zarr")

        mock_open_group.assert_called_once_with(
            "test_dir/output.zarr", mode="a"
        )

        mock_store.__setitem__.assert_called_once()
        args, kwargs = mock_store.__setitem__.call_args
        assert args[0] == "map_A"
        np.testing.assert_allclose(args[1], depth_map_np)

    def test_process_single_depth_map_integration(
        self, processor_instance, mock_zarr, mock_plt
    ):
        """
        Integration test for process_single_depth_map (formerly process_file).
        Uses the main 12x12 fixture (plotting=False).
        """
        # --- Mock Inputs (12x12) ---
        mock_open, _, _ = mock_zarr
        mock_store_content = np.zeros((12, 12, 1), dtype=np.uint8)
        mock_store_content[2:10, 2:10, 0] = 1  # 8x8 pond
        mock_open.return_value.__getitem__.return_value = mock_store_content

        # --- Mock Internal Methods ---
        with patch.object(
            processor_instance, "_label_ponds"
        ) as mock_label, patch.object(
            processor_instance, "_extract_contours"
        ) as mock_extract, patch.object(
            processor_instance, "_plot_pond_edge_elevations"
        ) as mock_plot, patch.object(
            processor_instance, "_calculate_all_depths"
        ) as mock_calc_all:

            # --- Define Mock Outputs (12x12) ---
            mock_labeled_data = cp.ones((12, 12))  # 12x12
            mock_label.return_value = mock_labeled_data

            mock_contours_val = {1: np.array([5.0])}
            mock_extract.return_value = (
                {"ignored_pixels": 1},
                mock_contours_val,
            )

            # Mock the new _calculate_all_depths return value
            mock_calc_all.return_value = {
                "wse_map_mean": cp.array([99.0]),
                "depth_map_mean": cp.array([1.0]),
                # ... (only need to mock the ones we check)
                "wse_map_95_perc": cp.array([100.0]),
                "depth_map_95_perc": cp.array([2.0]),
                "wse_map_90_perc": cp.array([101.0]),
                "depth_map_90_perc": cp.array([3.0]),
                "wse_map_median": cp.array([102.0]),
                "depth_map_median": cp.array([4.0]),
            }

            # --- Call Method ---
            result_list = processor_instance.process_single_depth_map(
                "test_dir/label.zarr", "label_file_name"
            )

            # --- Assertions ---
            mock_open.assert_called_once_with("test_dir/label.zarr")
            mock_label.assert_called_once()

            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            cp.testing.assert_allclose(args[0], mock_labeled_data)
            cp.testing.assert_allclose(args[1], cp.array(mock_store_content))

            # Plotting is OFF by default in this fixture
            mock_plot.assert_not_called()

            # Check that the new method was called once
            mock_calc_all.assert_called_once_with(
                mock_labeled_data, mock_contours_val
            )

            # Check final output list
            assert len(result_list) == 8
            assert (
                result_list[0]["image_name"] == "label_file_name_wse_map_mean"
            )
            cp.testing.assert_allclose(
                result_list[0]["depth_map"], cp.array([99.0])
            )
            assert (
                result_list[7]["image_name"]
                == "label_file_name_depth_map_median"
            )
            cp.testing.assert_allclose(
                result_list[7]["depth_map"], cp.array([4.0])
            )

    def test_process_single_depth_map_integration_with_plotting(
        self, processor_with_plotting, mock_zarr, mock_plt
    ):
        """
        NEW TEST: Integration test for process_single_depth_map
        that explicitly checks that plotting IS called.
        """
        # --- Mock Inputs (12x12) ---
        mock_open, _, _ = mock_zarr
        mock_store_content = np.zeros((12, 12, 1), dtype=np.uint8)
        mock_store_content[2:10, 2:10, 0] = 1  # 8x8 pond
        mock_open.return_value.__getitem__.return_value = mock_store_content

        # --- Mock Internal Methods ---
        with patch.object(
            processor_with_plotting, "_label_ponds"
        ) as mock_label, patch.object(
            processor_with_plotting, "_extract_contours"
        ) as mock_extract, patch.object(
            processor_with_plotting, "_plot_pond_edge_elevations"
        ) as mock_plot, patch.object(
            processor_with_plotting, "_calculate_all_depths"
        ) as mock_calc_all:

            # --- Define Mock Outputs (12x12) ---
            mock_labeled_data = cp.ones((12, 12))
            mock_label.return_value = mock_labeled_data
            mock_contours_val = {1: np.array([5.0])}
            mock_extract.return_value = (
                {"ignored_pixels": 1},
                mock_contours_val,
            )

            # Return value doesn't matter here
            mock_calc_all.return_value = {}

            # --- Call Method ---
            # Use the processor_with_plotting fixture
            processor_with_plotting.process_single_depth_map(
                "test_dir/label.zarr", "label_file_name"
            )

            # --- Assertions ---
            # Plotting is ON in this fixture
            mock_plot.assert_called_once_with(
                mock_labeled_data, mock_contours_val, "label_file_name"
            )

    def test_process_depth_maps_integration(self, processor_instance, mock_os):
        """
        Top-level integration test for process_depth_maps.
        Uses the main 12x12 fixture.
        """
        mock_listdir, _ = mock_os

        mock_listdir.return_value = [
            "file_A_rectified",
            "file_B.txt",
            "file_C_rectified",
        ]

        with patch.object(
            processor_instance, "process_single_depth_map"
        ) as mock_process, patch.object(
            processor_instance, "_save_depth_maps"
        ) as mock_save:

            mock_process.side_effect = [
                [{"image_name": "A_map", "depth_map": cp.array([1])}],
                [{"image_name": "C_map", "depth_map": cp.array([2])}],
            ]

            # --- UPDATED CALL ---
            # Pass the plot dir arg explicitly to match old fixture's intent
            processor_instance.process_depth_maps(
                "labels_dir", 
                "depth_dir",
                pond_edge_elev_plot_dir="test_data/plots"
            )
            # --- END UPDATED CALL ---

            mock_listdir.assert_called_once_with("labels_dir")

            assert mock_process.call_count == 2
            mock_process.assert_has_calls(
                [
                    call("labels_dir/file_A_rectified", "file_A_rectified"),
                    call("labels_dir/file_C_rectified", "file_C_rectified"),
                ]
            )

            assert mock_save.call_count == 2