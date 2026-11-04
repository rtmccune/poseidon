"""Tests for DepthMapProcessor using ponds of known water level."""

import numpy as np
import pytest
from conftest import requires_gpu, square_mask

from poseidon_core.depth_map_processor import STATISTICS, DepthMapProcessor
from poseidon_utils.zarr_io import (
    read_zip_arrays,
    write_array,
    write_zip_group,
)

SHAPE = (60, 60)


def bowl(level=0.5):
    """
    Return a bowl-shaped elevation grid and its water mask.

    Elevation rises with distance from the center; cells below `level`
    are water, so the true water surface elevation is `level`.

    Parameters
    ----------
    level : float, optional
        Water surface elevation (m). Default is 0.5.

    Returns
    -------
    elevation : numpy.ndarray
        Ground elevation (m).
    labels : numpy.ndarray
        1 where flooded, 0 elsewhere.
    """
    rows, cols = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    radius = np.hypot(rows - 30, cols - 30)
    elevation = 0.1 + 0.02 * radius
    return elevation, (elevation < level).astype(np.uint8)


def processor(elevation, **kwargs):
    """
    Build a CPU processor without histogram output.

    Parameters
    ----------
    elevation : numpy.ndarray
        Ground elevation grid.
    **kwargs
        Extra DepthMapProcessor options.

    Returns
    -------
    DepthMapProcessor
        CPU processor.
    """
    kwargs.setdefault("plot_edges", False)
    return DepthMapProcessor(elevation, use_gpu=False, **kwargs)


def test_flat_pond_has_exact_level_and_zero_depth():
    """On flat ground every statistic recovers the ground elevation."""
    labels = square_mask(SHAPE, slice(10, 30), slice(15, 40))
    maps = processor(np.ones(SHAPE)).process_single_depth_map(labels, "img")

    pond = labels == 1
    for stat in STATISTICS:
        wse, depth = maps[f"img_wse_map_{stat}"], maps[f"img_depth_map_{stat}"]
        np.testing.assert_array_equal(wse[pond], 1.0)
        np.testing.assert_array_equal(depth[pond], 0.0)
        assert np.isnan(wse[~pond]).all() and np.isnan(depth[~pond]).all()


def test_bowl_water_level_is_recovered_by_every_statistic():
    """WSE from the waterline matches the true level for all stats.

    Regression test: sampling outside the pond used to pull the mean
    and median WSE toward zero.
    """
    elevation, labels = bowl(level=0.5)
    maps = processor(elevation).process_single_depth_map(labels, "img")
    pond = labels == 1
    cell_rise = 0.02  # elevation change across one cell
    for stat in STATISTICS:
        wse = maps[f"img_wse_map_{stat}"][pond]
        assert np.unique(wse).size == 1
        assert wse[0] == pytest.approx(0.5, abs=cell_rise)

    # Depth is WSE minus ground, deepest at the center
    depth = maps["img_depth_map_95_perc"]
    wse = maps["img_wse_map_95_perc"]
    np.testing.assert_allclose(depth[pond], np.maximum(
        wse[pond] - elevation[pond], 0))
    assert np.nanargmax(depth) == np.ravel_multi_index((30, 30), SHAPE)


def test_waterline_ignores_the_grid_border():
    """A pond cut off by the grid edge is not "shored" by the edge."""
    cols = np.arange(SHAPE[1], dtype=float)
    elevation = np.tile(cols * 0.01, (SHAPE[0], 1))  # rises eastward
    labels = square_mask(SHAPE, slice(None), slice(0, 20))  # touches 3 edges
    maps = processor(elevation).process_single_depth_map(labels, "img")
    # Only the eastern side (column 19) is a real waterline
    assert maps["img_wse_map_mean"][0, 0] == pytest.approx(0.19)


def test_nan_elevations_on_the_waterline_are_ignored():
    """Missing LiDAR on part of the shore does not void the pond."""
    elevation = np.ones(SHAPE)
    labels = square_mask(SHAPE, slice(10, 30), slice(10, 30))
    elevation[10, 10:30] = np.nan  # northern shore has no data
    maps = processor(elevation).process_single_depth_map(labels, "img")
    for stat in STATISTICS:
        assert np.nanmax(maps[f"img_wse_map_{stat}"]) == 1.0


def test_small_ponds_are_dropped_and_large_ones_kept():
    """Ponds below min_pond_size pixels produce no water surface."""
    labels = square_mask(SHAPE, slice(5, 10), slice(5, 10))  # 25 px
    labels[30:45, 30:45] = 1  # 225 px
    maps = processor(np.ones(SHAPE), min_pond_size=55
                     ).process_single_depth_map(labels, "img")
    wse = maps["img_wse_map_median"]
    assert np.isnan(wse[5:10, 5:10]).all()
    assert (wse[30:45, 30:45] == 1.0).all()


def test_ponds_get_independent_water_levels():
    """Separate ponds keep their own WSE."""
    elevation = np.ones(SHAPE)
    elevation[:, 30:] = 2.0
    labels = square_mask(SHAPE, slice(10, 25), slice(5, 20))
    labels[10:25, 40:55] = 1
    maps = processor(elevation).process_single_depth_map(labels, "img")
    wse = maps["img_wse_map_mean"]
    assert wse[15, 10] == 1.0 and wse[15, 45] == 2.0


def test_one_pixel_gaps_are_closed_into_one_pond():
    """Binary closing joins a pond split by a one-pixel dry line."""
    elevation = np.ones(SHAPE)
    elevation[:, 25:] = 1.2
    labels = square_mask(SHAPE, slice(10, 30), slice(10, 40))
    labels[:, 25] = 0
    maps = processor(elevation).process_single_depth_map(labels, "img")
    wse = maps["img_wse_map_95_perc"]
    assert wse[20, 12] == wse[20, 38]  # one pond, one level


def test_dry_image_gives_all_nan_maps():
    """An image with no water returns NaN maps for every statistic."""
    maps = processor(np.ones(SHAPE)).process_single_depth_map(
        np.zeros(SHAPE, np.uint8), "dry")
    assert len(maps) == 2 * len(STATISTICS)
    assert all(np.isnan(array).all() for array in maps.values())


def test_three_dimensional_labels_are_accepted():
    """Rectified labels with a trailing channel axis are squeezed."""
    labels = square_mask(SHAPE, slice(10, 30), slice(10, 30))[..., None]
    maps = processor(np.ones(SHAPE)).process_single_depth_map(labels, "img")
    assert maps["img_wse_map_mean"].shape == SHAPE


def test_histograms_save_without_calling_process_depth_maps(tmp_path):
    """plot_edges works for a single image (the plot folder is set)."""
    proc = DepthMapProcessor(np.ones(SHAPE), plot_edges=True, use_gpu=False,
                             pond_edge_elev_plot_dir=str(tmp_path))
    labels = square_mask(SHAPE, slice(10, 30), slice(10, 30))
    proc.process_single_depth_map(labels, "img")
    assert (tmp_path / "all_ponds" / "img_All_Ponds_Histogram.png").exists()
    assert (tmp_path / "ind_ponds" / "img_Pond_1.png").exists()


def test_store_processing_writes_every_map_and_replaces_old_output(tmp_path):
    """process_depth_maps writes 8 maps per label array, atomically."""
    labels_zip = tmp_path / "labels_rects.zip"
    with write_zip_group(labels_zip) as group:
        for i in range(2):
            write_array(group, f"CAM_TS_01_2024010112{i}000_rectified",
                        square_mask(SHAPE, slice(10, 30), slice(10, 30)))

    depth_zip = tmp_path / "depth_maps.zip"
    with write_zip_group(depth_zip) as group:
        write_array(group, "stale_map", np.zeros(2))

    proc = processor(np.ones(SHAPE))
    assert proc.process_depth_maps(labels_zip, depth_zip) == 2
    names = list(read_zip_arrays(depth_zip))
    assert len(names) == 2 * 2 * len(STATISTICS)
    assert "stale_map" not in names


def test_missing_label_store_processes_nothing(tmp_path):
    """A missing input store is reported and returns 0."""
    assert processor(np.ones(SHAPE)).process_depth_maps(
        tmp_path / "missing.zip", tmp_path / "out.zip") == 0


@requires_gpu
def test_gpu_and_cpu_depth_maps_agree():
    """CuPy/cuCIM and NumPy/scikit-image give identical maps."""
    elevation, labels = bowl()
    labels[5:9, 5:9] = 1  # small pond to be dropped
    cpu = processor(elevation).process_single_depth_map(labels, "img")
    gpu = DepthMapProcessor(elevation, plot_edges=False, use_gpu=True
                            ).process_single_depth_map(labels, "img")
    for name, array in cpu.items():
        np.testing.assert_array_equal(gpu[name].get(), array)
