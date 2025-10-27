# test_depth_map_plotter.py

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pathlib
import pytest
import numpy as np
import xarray as xr
import dask.array as da
import zarr
from affine import Affine
import matplotlib.pyplot as plt
import contextily as ctx
import matplotlib.path

# --- IMPORTANT ---
from poseidon_core.depth_map_plotter import DepthMapPlotter, _log

# Import the module itself to patch tqdm inside it for HPC tests
import poseidon_core.depth_map_plotter


class TestDepthMapPlotter:
    """
    Test suite for the DepthMapPlotter class.
    Mocks all I/O, plotting, and HPC dependencies.
    """

    ## --- Fixtures ---
    # (Fixtures from default_config down to mock_mpi remain the same)
    @pytest.fixture
    def default_config(self, tmp_path):
        """Provides a dictionary of default config values for the plotter."""
        return {
            "main_dir": tmp_path,
            "min_x_extent": 1000.0,
            "max_x_extent": 2000.0,
            "min_y_extent": 5000.0,
            "max_y_extent": 6000.0,
            "resolution_m": 1.0,
            "bbox_crs": "EPSG:32119",
        }

    @pytest.fixture
    def base_plotter(self, default_config):
        """Provides a standard DepthMapPlotter instance with plotting
        disabled."""
        return DepthMapPlotter(
            **default_config,
            virtual_sensor_locations=None,
            plot_sensors=False,
        )

    @pytest.fixture
    def plotter_with_sensors(self, default_config, mock_svg):
        """
        Provides a plotter instance with sensor plotting enabled.
        Requires mock_svg to mock the marker creation in __init__.
        """
        sensor_locs = np.array(
            [
                [10, 10],  # Sensor 1 [Y, X]
                [20, 20],  # Sensor 2 [Y, X]
            ]
        )
        return DepthMapPlotter(
            **default_config,
            virtual_sensor_locations=sensor_locs,
            plot_sensors=True,
        )

    @pytest.fixture
    def mock_log(self, mocker):
        """Mocks the internal _log function."""
        return mocker.patch("poseidon_core.depth_map_plotter._log")

    @pytest.fixture
    def mock_fs(self, mocker, tmp_path):
        """Mocks all file system interactions (os, pathlib)."""
        mock_listdir = mocker.patch(
            "poseidon_core.depth_map_plotter.os.listdir"
        )
        mock_isdir = mocker.patch(
            "poseidon_core.depth_map_plotter.os.path.isdir", return_value=True
        )
        mocker.patch(
            "poseidon_core.depth_map_plotter.os.path.join",
            lambda *args: "/".join(map(str, args)),
        )

        # Mock pathlib methods needed globally
        mock_path_mkdir = mocker.patch("pathlib.Path.mkdir")
        # Keep a general patch for is_dir for other tests, but
        # override in specific test
        mock_general_path_isdir = mocker.patch(
            "pathlib.Path.is_dir", return_value=True
        )

        return {
            "listdir": mock_listdir,
            "isdir": mock_isdir,
            "path_mkdir": mock_path_mkdir,
            "general_path_isdir": mock_general_path_isdir,
            # Keep track if needed
        }

    @pytest.fixture
    def mock_zarr(self, mocker):
        """Mocks zarr.open."""
        mock_store = MagicMock(spec=zarr.core.array)
        mock_store.shape = (1000, 1000)
        mock_store.chunks = (100, 100)
        mock_open = mocker.patch(
            "poseidon_core.depth_map_plotter.zarr.open", return_value=mock_store
        )
        return mock_open, mock_store

    @pytest.fixture
    def mock_dask(self, mocker):
        """Mocks dask.array functions."""
        mock_lazy_array = MagicMock(spec=da.core.Array)
        mock_from_array = mocker.patch(
            "poseidon_core.depth_map_plotter.da.from_array",
            return_value=mock_lazy_array,
        )

        mock_sum_obj = MagicMock()
        mock_sum_obj.compute.return_value = 50000
        mock_sum = mocker.patch(
            "poseidon_core.depth_map_plotter.da.sum", return_value=mock_sum_obj
        )

        return {
            "from_array": mock_from_array,
            "sum": mock_sum,
            "lazy_array": mock_lazy_array,
        }

    @pytest.fixture
    def mock_xarray(self, mocker):
        """Mocks the entire xarray/rioxarray processing chain."""
        mock_mercator_array = MagicMock(spec=xr.DataArray)
        mock_mercator_array.rio.bounds.return_value = (-78.5, 35.5, -78.0, 36.0)
        mock_mercator_array.rio.crs = "EPSG:3857"
        mock_mercator_array.to_numpy.return_value = np.full((50, 50), 1.5)

        mock_min_obj = MagicMock()
        mock_min_obj.compute.return_value = 5.0
        mock_mercator_array.min.return_value = mock_min_obj

        mock_max_obj = MagicMock()
        mock_max_obj.compute.return_value = 15.0
        mock_mercator_array.max.return_value = mock_max_obj

        mock_initial_da = MagicMock(spec=xr.DataArray)

        mock_chain = mock_initial_da.isel.return_value
        mock_chain = mock_chain.rio.write_crs.return_value
        mock_chain = mock_chain.rio.write_transform.return_value
        mock_chain = mock_chain.rio.write_nodata.return_value
        mock_chain.rio.reproject.return_value = mock_mercator_array

        mock_da_class = mocker.patch(
            "poseidon_core.depth_map_plotter.xr.DataArray",
            return_value=mock_initial_da,
        )

        return {
            "class": mock_da_class,
            "initial_da": mock_initial_da,
            "mercator_array": mock_mercator_array,
        }

    @pytest.fixture
    def mock_plt(self, mocker):
        """Mocks all matplotlib.pyplot functions."""
        mock_fig = MagicMock(spec=plt.Figure)
        mock_ax = MagicMock(spec=plt.Axes)
        mock_ax.transAxes = "mock_transform_object"

        mock_im = MagicMock()
        mock_cbar = MagicMock()

        mock_ax.imshow.return_value = mock_im
        mock_fig.colorbar.return_value = mock_cbar

        mock_subplots = mocker.patch(
            "poseidon_core.depth_map_plotter.plt.subplots",
            return_value=(mock_fig, mock_ax),
        )
        mock_savefig = mocker.patch(
            "poseidon_core.depth_map_plotter.plt.savefig"
        )
        mock_close = mocker.patch("poseidon_core.depth_map_plotter.plt.close")

        return {
            "subplots": mock_subplots,
            "savefig": mock_savefig,
            "close": mock_close,
            "fig": mock_fig,
            "ax": mock_ax,
            "im": mock_im,
            "cbar": mock_cbar,
        }

    @pytest.fixture
    def mock_ctx(self, mocker):
        """Mocks contextily.add_basemap."""
        return mocker.patch("poseidon_core.depth_map_plotter.ctx.add_basemap")

    @pytest.fixture
    def mock_svg(self, mocker):
        """Mocks svgpath2mpl.parse_path."""
        # Just return the patch object itself, or a basic mock
        return mocker.patch(
            "poseidon_core.depth_map_plotter.parse_path",
            return_value=MagicMock(),
        )

    @pytest.fixture
    def mock_pyproj(self, mocker):
        """Mocks pyproj.Transformer."""
        mock_transformer = MagicMock()
        mock_transformer.transform.side_effect = [
            (1111.0, 5555.0),
            (2222.0, 6666.0),
        ]
        mock_from_crs = mocker.patch(
            "poseidon_core.depth_map_plotter.Transformer.from_crs",
            return_value=mock_transformer,
        )
        return mock_from_crs, mock_transformer

    @pytest.fixture
    def mock_mpi(self, mocker):
        """Mocks mpi4py.MPI."""
        mock_comm = MagicMock()
        mock_comm.Get_rank.return_value = 0
        mock_comm.Get_size.return_value = 1
        mock_comm.bcast.side_effect = lambda data, root: data

        mock_mpi_comm = mocker.patch(
            "poseidon_core.depth_map_plotter.MPI.COMM_WORLD", mock_comm
        )
        return mock_comm

    ## --- Test Cases ---

    def test_init_defaults(self, default_config, tmp_path):
        """Tests that init sets defaults correctly."""
        plotter = DepthMapPlotter(**default_config)
        assert plotter.main_dir == tmp_path
        assert plotter.min_x_extent == 1000.0
        assert plotter.resolution_m == 1.0
        assert plotter.plot_sensors is False
        assert not hasattr(plotter, "sensor_marker_path")

    def test_init_with_sensors(self, default_config, mock_svg):
        """Tests that init creates sensor marker when plot_sensors=True."""
        # --- FIX 1: Simplified assertion ---
        with patch(
            "poseidon_core.depth_map_plotter.parse_path"
        ) as mock_parse_path:
            # We need the mock object returned by parse_path for checks
            mock_path_obj = MagicMock()
            mock_vertices_obj = MagicMock()
            mock_vertices_obj.mean.return_value = np.array([0.0, 0.0])
            mock_path_obj.vertices = mock_vertices_obj
            mock_parse_path.return_value = (
                mock_path_obj  # Configure the patch return
            )

            plotter = DepthMapPlotter(
                **default_config,
                plot_sensors=True,
                virtual_sensor_locations=None,
            )
            assert plotter.plot_sensors is True
            mock_parse_path.assert_called_once()
            # Ensure _create_sensor_marker ran
            assert hasattr(plotter, "sensor_marker_path")
            assert (
                plotter.sensor_marker_path == mock_path_obj
            )  # Check assignment

    def test_log_helper(self, capsys):
        """Tests the _log helper function's output."""
        _log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
        assert "[" in captured.out
        assert "]" in captured.out

    def test_list_flood_event_folders(self, base_plotter, mock_fs):
        """Tests that only directories are returned, sorted."""
        mock_fs["listdir"].return_value = ["event_C", "event_A", "file.txt"]
        mock_fs["isdir"].side_effect = [True, True, False]

        result = base_plotter._list_flood_event_folders()

        assert result == ["event_A", "event_C"]
        mock_fs["isdir"].assert_has_calls(
            [
                call(f"{base_plotter.main_dir}/event_C"),
                call(f"{base_plotter.main_dir}/event_A"),
                call(f"{base_plotter.main_dir}/file.txt"),
            ]
        )

    def test_get_plot_style(self, base_plotter, mock_xarray, mocker):
        """Tests the logic for returning plot styles."""

        mock_cmap = mocker.patch(
            "poseidon_core.depth_map_plotter.cmocean.cm.dense"
        )

        # Test 'depth'
        depth_style = base_plotter._get_plot_style("depth", None)
        assert depth_style["vmin"] == 0
        assert depth_style["vmax"] == 0.25
        assert depth_style["cbar_label"] == "Depth (m)"
        assert depth_style["cmap"] == mock_cmap
        # No assert_called_once() needed

        # Test 'wse'
        mock_data = mock_xarray["mercator_array"]
        wse_style = base_plotter._get_plot_style("wse", mock_data)

        assert wse_style["vmin"] == 5.0
        assert wse_style["vmax"] == 15.0
        assert wse_style["cbar_label"] == "Water Surface Elevation (m)"
        assert wse_style["cmap"] == "Blues"

        # Test 'invalid'
        with pytest.raises(ValueError, match="Unknown plot_type"):
            base_plotter._get_plot_style("invalid", None)

    def test_create_sensor_marker(
        self, mocker
    ):  # No longer need mock_svg fixture
        """Tests that the static marker method calls parse_path and
        performs ops."""

        # 1. Patch 'parse_path' where it's used
        with patch(
            "poseidon_core.depth_map_plotter.parse_path"
        ) as mock_parse_path:

            # 2. Create the mock 'Path' object that parse_path will
            # return
            mock_path_obj = MagicMock(spec=matplotlib.path.Path)

            # 3. Create the mock 'vertices' array for the mock Path
            # object
            mock_vertices_array = MagicMock()
            # Use a non-zero center to verify centering logic
            mock_vertices_array.mean.return_value = np.array([10.0, 20.0])
            mock_path_obj.vertices = mock_vertices_array

            # 4. Configure the patch to return our fully configured
            # mock object
            mock_parse_path.return_value = mock_path_obj

            # 5. Call the static method
            marker = DepthMapPlotter._create_sensor_marker()

            # 6. Assertions
            mock_parse_path.assert_called_once()
            # Check if parse_path was called
            assert marker == mock_path_obj
            # Check if the mock was returned

            # Check centering op: marker.vertices -=
            # marker.vertices.mean(axis=0)
            mock_vertices_array.mean.assert_called_with(axis=0)
            # This is the correct way to check calls with numpy arrays
            mock_vertices_array.__isub__.assert_called_once()
            actual_arg = mock_vertices_array.__isub__.call_args[0][
                0
            ]  # Gets the first positional arg
            expected_arg = np.array([10.0, 20.0])
            np.testing.assert_array_equal(actual_arg, expected_arg)

    def test_load_and_prepare_geodata(
        self, base_plotter, mock_zarr, mock_dask, mock_xarray
    ):
        """Tests the core lazy-loading and reprojection data pipeline."""

        zarr_path = "path/to/mock.zarr"
        result = base_plotter._load_and_prepare_geodata(zarr_path)

        mock_zarr[0].assert_called_with(str(zarr_path), mode="r")
        mock_dask["from_array"].assert_called_with(
            mock_zarr[1], chunks=mock_zarr[1].chunks
        )

        mock_dask["sum"].assert_called_once()
        assert result["spatial_extent"] == 50000.0

        mock_xarray["class"].assert_called_once()
        assert (
            mock_xarray["class"].call_args[1]["data"]
            == mock_dask["lazy_array"].astype.return_value
        )

        mock_xarray["initial_da"].isel.assert_called_with(
            y=slice(None, None, -1)
        )
        reprojected_da = mock_xarray[
            "initial_da"
        ].isel.return_value.rio.write_crs.return_value.rio.write_transform.return_value.rio.write_nodata.return_value
        reprojected_da.rio.reproject.assert_called_with(3857)

        assert result["mercator_array"] == mock_xarray["mercator_array"]
        assert result["shape"] == (1000, 1000)

        expected_transform = Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 6000.0)
        assert result["transform"] == expected_transform

    def test_finalize_and_save_plot(
        self, base_plotter, mock_plt, mock_ctx, mock_fs, tmp_path
    ):
        """Tests that all plotting/saving functions are called
        correctly."""

        mock_data = MagicMock()
        mock_data.rio.bounds.return_value = (10, 20, 30, 40)
        mock_data.rio.crs = "EPSG:3857"
        geodata = {"mercator_array": mock_data, "spatial_extent": 12345.67}
        out_folder = tmp_path / "figures"
        out_file = "test_plot.png"

        base_plotter._finalize_and_save_plot(
            mock_plt["fig"],
            mock_plt["ax"],
            mock_plt["im"],
            geodata,
            "Test Cbar Label",
            out_folder,
            out_file,
        )

        mock_plt["ax"].set_xlim.assert_called_with(10, 30)
        mock_plt["ax"].set_ylim.assert_called_with(20, 40)
        mock_ctx.assert_called_with(
            mock_plt["ax"],
            crs="EPSG:3857",
            source=ctx.providers.Esri.WorldImagery,
            zorder=1,
        )

        mock_plt["ax"].text.assert_called_once_with(
            0.05,
            0.95,
            "Spatial Extent ($m^2$): 12345.67",
            transform=mock_plt["ax"].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            zorder=30,
        )

        mock_plt["fig"].colorbar.assert_called_with(
            mock_plt["im"], ax=mock_plt["ax"], shrink=0.6, aspect=30
        )
        mock_plt["cbar"].set_label.assert_called_with("Test Cbar Label")

        mock_fs["path_mkdir"].assert_called_with(parents=True, exist_ok=True)
        expected_save_path = out_folder / out_file
        mock_plt["savefig"].assert_called_with(
            expected_save_path, dpi=300, bbox_inches="tight", pad_inches=0
        )
        mock_plt["close"].assert_called_with(mock_plt["fig"])

    def test_plot_sensor_locations(
        self, plotter_with_sensors, mock_plt, mock_pyproj
    ):
        """Tests plotting sensor locations when enabled."""
        mock_ax = mock_plt["ax"]
        marker = plotter_with_sensors.sensor_marker_path
        original_shape = (1000, 1000)
        transform = Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 6000.0)

        plotter_with_sensors._plot_sensor_locations(
            mock_ax, marker, original_shape, transform
        )

        mock_pyproj[0].assert_called_with(
            "EPSG:32119", "EPSG:3857", always_xy=True
        )

        mock_pyproj[1].transform.assert_has_calls(
            [
                call(1010.5, 5010.5),
                call(1020.5, 5020.5),
            ]
        )

        mock_ax.scatter.assert_called_once()
        call_args, call_kwargs = mock_ax.scatter.call_args

        np.testing.assert_array_equal(call_args[0], [1111.0, 2222.0])
        np.testing.assert_array_equal(call_args[1], [5555.0, 6666.0])

        assert call_kwargs["marker"] == marker
        assert call_kwargs["s"] == 300
        assert call_kwargs["zorder"] == 20

    def test_plot_sensor_locations_out_of_bounds(
        self, plotter_with_sensors, mock_plt, mock_log
    ):
        """Tests that out-of-bounds sensors are skipped."""
        plotter_with_sensors.virtual_sensor_loc = np.array([[9999, 9999]])
        original_shape = (1000, 1000)
        transform = Affine(1.0, 0.0, 1000.0, 0.0, -1.0, 6000.0)

        plotter_with_sensors._plot_sensor_locations(
            mock_plt["ax"], "marker", original_shape, transform
        )

        mock_log.assert_any_call(
            "    -> WARNING: Sensor index (Y=9999, X=9999) "
            "is out of bounds for array shape (1000, 1000)."
        )
        mock_plt["ax"].scatter.assert_not_called()

    def test_plot_sensor_locations_disabled(self, base_plotter, mock_plt):
        """Tests that no plotting occurs if plot_sensors=False."""
        base_plotter._plot_sensor_locations(
            mock_plt["ax"], "marker", (100, 100), Affine.identity()
        )
        mock_plt["ax"].scatter.assert_not_called()

    def test_plot_georeferenced_map_orchestration(
        self, base_plotter, mocker, mock_plt
    ):
        """
        Tests the orchestration logic of _plot_georeferenced_map
        by mocking its helper methods.
        """
        mock_load = mocker.patch.object(
            base_plotter, "_load_and_prepare_geodata"
        )
        mock_style = mocker.patch.object(base_plotter, "_get_plot_style")
        mock_plot_sensors = mocker.patch.object(
            base_plotter, "_plot_sensor_locations"
        )
        mock_finalize = mocker.patch.object(
            base_plotter, "_finalize_and_save_plot"
        )

        mock_mercator_array = MagicMock()
        mock_mercator_array.rio.bounds.return_value = (1, 2, 3, 4)

        mock_geodata = {
            "mercator_array": mock_mercator_array,
            "shape": (100, 100),
            "transform": Affine.identity(),
            "spatial_extent": 123,
        }
        mock_load.return_value = mock_geodata

        mock_style_dict = {
            "cmap": "test_cmap",
            "vmin": 0,
            "vmax": 1,
            "cbar_label": "Test",
        }
        mock_style.return_value = mock_style_dict

        base_plotter._plot_georeferenced_map(
            "path/to/zarr", "output.png", "depth", "folder"
        )

        mock_load.assert_called_with("path/to/zarr")
        mock_plt["subplots"].assert_called_with(figsize=(10, 10))

        mock_geodata["mercator_array"].to_numpy.assert_called_once()
        mock_style.assert_called_with("depth", mock_geodata["mercator_array"])

        mock_plt["ax"].imshow.assert_called_once()

        mock_plot_sensors.assert_not_called()

        mock_finalize.assert_called_with(
            mock_plt["fig"],
            mock_plt["ax"],
            mock_plt["ax"].imshow.return_value,
            mock_geodata,
            "Test",
            "folder",
            "output.png",
        )

    def test_process_single_flood_event_simple(
        self, base_plotter, mock_fs, mock_log, mocker
    ):
        """Tests the main processing loop for a single event."""
        mock_fs["listdir"].return_value = [
            "depth_map_mean",
            "wse_map_95_perc",
            "not_a_zarr.txt",
            "depth_other",
        ]
        # Ensure the general is_dir patch returns True for this test
        mocker.patch("pathlib.Path.is_dir", return_value=True)

        mock_plotter = mocker.patch.object(
            base_plotter, "_plot_georeferenced_map"
        )

        base_plotter.process_single_flood_event("event_01")

        mock_log.assert_any_call(
            "  Found 4 potential Zarr stores. 4 will be processed."
        )

        assert mock_plotter.call_count == 3

        expected_calls = [
            call(
                depth_array_path=Path(
                    f"{base_plotter.main_dir}/event_01/zarr/depth_maps"
                    f"/depth_map_mean"
                ),
                output_filename="depth_map_mean.png",
                plot_type="depth",
                output_folder=Path(
                    f"{base_plotter.main_dir}/event_01/plots/depth_maps"
                ),
            ),
            call(
                depth_array_path=Path(
                    f"{base_plotter.main_dir}/event_01/zarr/depth_maps"
                    f"/depth_other"
                ),
                output_filename="depth_other.png",
                plot_type="depth",
                output_folder=Path(
                    f"{base_plotter.main_dir}/event_01/plots/depth_maps"
                ),
            ),
            call(
                depth_array_path=Path(
                    f"{base_plotter.main_dir}/event_01/zarr/depth_maps/"
                    f"wse_map_95_perc"
                ),
                output_filename="wse_map_95_perc.png",
                plot_type="wse",
                output_folder=Path(
                    f"{base_plotter.main_dir}/event_01/plots/WSE_maps"
                ),
            ),
        ]
        mock_plotter.assert_has_calls(expected_calls)

        mock_log.assert_any_call(
            "    -> Skipping (unknown plot type): not_a_zarr.txt"
        )
        mock_log.assert_any_call(
            "  Successfully processed and plotted 3 files."
        )
        mock_log.assert_any_call("  Skipped 1 files due to unknown type.")

    def test_process_single_flood_event_with_filter(
        self, base_plotter, mock_fs, mock_log, mocker
    ):
        """Tests that the stats_to_plot filter is applied."""
        mock_fs["listdir"].return_value = [
            "depth_map_mean",
            "wse_map_95_perc",
            "wse_map_mean",
        ]
        # Ensure the general is_dir patch returns True for this test
        mocker.patch("pathlib.Path.is_dir", return_value=True)

        mock_plotter = mocker.patch.object(
            base_plotter, "_plot_georeferenced_map"
        )

        base_plotter.process_single_flood_event(
            "event_01", stats_to_plot=["mean"]
        )

        mock_log.assert_any_call("  Filtering for stats: ['mean']")
        mock_log.assert_any_call(
            "  Found 3 potential Zarr stores. 2 will be processed."
        )

        assert mock_plotter.call_count == 2
        mock_plotter.assert_has_calls(
            [
                call(
                    depth_array_path=Path(
                        f"{base_plotter.main_dir}/event_01/zarr/depth_maps"
                        f"/depth_map_mean"
                    ),
                    output_filename="depth_map_mean.png",
                    plot_type="depth",
                    output_folder=Path(
                        f"{base_plotter.main_dir}/event_01/plots/depth_maps"
                    ),
                ),
                call(
                    depth_array_path=Path(
                        f"{base_plotter.main_dir}/event_01/zarr/depth_maps"
                        f"/wse_map_mean"
                    ),
                    output_filename="wse_map_mean.png",
                    plot_type="wse",
                    output_folder=Path(
                        f"{base_plotter.main_dir}/event_01/plots/WSE_maps"
                    ),
                ),
            ]
        )

        mock_log.assert_any_call(
            "  Successfully processed and plotted 2 files."
        )
        mock_log.assert_any_call("  Skipped 1 files due to name filter.")

    def test_process_single_flood_event_no_dir(
        self, base_plotter, mock_fs, mock_log, mocker
    ):
        """Tests that processing aborts if the zarr dir is missing."""
        zarr_dir_path = (
            base_plotter.main_dir / "event_01" / "zarr" / "depth_maps"
        )

        # --- Correction ---
        # Define the side effect without 'self'
        def is_dir_side_effect(*args, **kwargs):
            # The actual Path instance is implicitly associated with the
            # mock call
            # We assume the first (and only relevant) call is the one we
            # care about a more complex check could inspect call_args if
            # needed
            return False
            # Always return False for this specific test's purpose

        # Patch directly using the correct target string
        path_is_dir_patch = mocker.patch(
            "pathlib.Path.is_dir", side_effect=is_dir_side_effect
        )

        mock_plotter = mocker.patch.object(
            base_plotter, "_plot_georeferenced_map"
        )

        # Call the function
        base_plotter.process_single_flood_event("event_01")

        # Assertions
        mock_log.assert_any_call(
            f"  ERROR: Directory not found at '{zarr_dir_path}'"
        )
        mock_plotter.assert_not_called()
        # Ensure the patch was actually used
        path_is_dir_patch.assert_called()

    # --- ALSO APPLY to test_process_flood_events_hpc_rank_0 ---
    def test_process_flood_events_hpc_rank_0(
        self, base_plotter, mock_mpi, mock_log, mocker
    ):
        """Tests HPC orchestration for Rank 0."""
        mock_mpi.Get_rank.return_value = 0
        mock_mpi.Get_size.return_value = 2

        mock_list_folders = mocker.patch.object(
            base_plotter,
            "_list_flood_event_folders",
            return_value=["event_A", "event_B", "event_C"],
        )
        mock_process_single = mocker.patch.object(
            base_plotter, "process_single_flood_event"
        )

        # --- Correction: Use mocker.patch with the module path ---
        mock_tqdm_instance = mocker.patch(
            "poseidon_core.depth_map_plotter.tqdm",
            # Target name in the module
            side_effect=lambda iterable, **kwargs: iterable,
            # Simpler side effect okay here
        )

        # Call the function
        base_plotter.process_flood_events_HPC(stats_to_plot=["mean"])

        # Assertions
        mock_list_folders.assert_called_once()
        mock_mpi.bcast.assert_called_with(
            ["event_A", "event_B", "event_C"], root=0
        )

        mock_tqdm_instance.assert_called_once()
        call_args, call_kwargs = mock_tqdm_instance.call_args
        assert list(call_args[0]) == ["event_A"]
        # Check iterable for rank 0

        mock_process_single.assert_called_once_with(
            "event_A", stats_to_plot=["mean"]
        )
