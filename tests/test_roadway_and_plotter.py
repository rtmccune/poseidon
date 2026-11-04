"""Tests for RoadwayAnalyzer and DepthMapPlotter on synthetic events."""

import json

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from poseidon_core.depth_map_plotter import DepthMapPlotter
from poseidon_core.roadway_analyzer import (
    RoadwayAnalyzer,
    accessibility_table,
    roi_pixels_from_labelme,
    sample_roi,
)
from poseidon_utils.zarr_io import (
    read_zip_arrays,
    write_array,
    write_zip_group,
)

TIMES = ["20240101120000", "20240101120600", "20240101121200"]


def write_labelme(path, shapes):
    """
    Write a minimal LabelMe JSON file.

    Parameters
    ----------
    path : pathlib.Path
        Output path.
    shapes : list of tuple
        ``(label, shape_type, points)`` for each shape.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    path.write_text(json.dumps({"shapes": [
        {"label": label, "shape_type": kind, "points": points}
        for label, kind, points in shapes]}))
    return path


def write_depth_store(event_dir, depth_by_time, wse_offset=1.0):
    """
    Write a depth_maps.zip with 95th-percentile depth and WSE maps.

    Parameters
    ----------
    event_dir : pathlib.Path
        Event folder; ``zarr/depth_maps.zip`` is created inside.
    depth_by_time : dict of str to numpy.ndarray
        Depth map (grid orientation) for each timestamp.
    wse_offset : float, optional
        WSE is depth plus this offset where depth is finite.

    Returns
    -------
    pathlib.Path
        The store path.
    """
    path = event_dir / "zarr" / "depth_maps.zip"
    with write_zip_group(path) as group:
        for stamp, depth in depth_by_time.items():
            base = f"CAM_TS_01_{stamp}_predseg_labels_rectified"
            write_array(group, f"{base}_depth_map_95_perc", depth)
            write_array(group, f"{base}_wse_map_95_perc", depth + wse_offset)
    return path


# ----- ROI geometry ----------------------------------------------------------


def test_polygon_roi_covers_its_interior_and_edges(tmp_path):
    """A rectangle yields each pixel on or inside its outline once."""
    path = write_labelme(tmp_path / "road.json", [
        ("roadway", "polygon", [[2, 1], [6, 1], [6, 4], [2, 4]])])
    coords, kind = roi_pixels_from_labelme(path, "roadway")
    assert kind == "polygon"
    # Vertices are (x, y); pixel centers on the outline are included
    assert len(coords) == len(np.unique(coords, axis=0)) == 4 * 5
    assert (coords[:, 0].min(), coords[:, 0].max()) == (1, 4)
    assert (coords[:, 1].min(), coords[:, 1].max()) == (2, 6)


def test_line_roi_samples_both_ends(tmp_path):
    """A transect is resampled every step including its last vertex."""
    path = write_labelme(tmp_path / "line.json", [
        ("other", "polygon", [[0, 0], [1, 0], [1, 1]]),
        ("roadway", "linestrip", [[0, 5], [10, 5]])])
    coords, kind = roi_pixels_from_labelme(path, "roadway", step_size=2.0)
    assert kind == "linestrip"
    np.testing.assert_array_equal(coords[:, 1], [0, 2, 4, 6, 8, 10])
    assert (coords[:, 0] == 5).all()


def test_missing_roi_label_lists_the_labels_found(tmp_path):
    """Asking for an absent label reports what the file contains."""
    path = write_labelme(tmp_path / "x.json", [("curb", "line",
                                                [[0, 0], [1, 1]])])
    with pytest.raises(ValueError, match="curb"):
        roi_pixels_from_labelme(path, "roadway")


def test_roi_samples_flip_north_up_rows_to_grid_rows():
    """Image row 0 (north) samples the grid's last (northern) row."""
    grid = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    coords = np.array([[0, 1], [4, 2], [9, 0]])  # last is out of bounds
    np.testing.assert_array_equal(sample_roi(grid, coords),
                                  [4.0, 0.0, np.nan])


def test_accessibility_statistics_per_timestep():
    """Flooded counts, impassability, and NaN-aware depth stats."""
    depths = np.array([[0.1, 0.3, np.nan, 0.0],
                       [np.nan, np.nan, np.nan, np.nan]])
    times = pd.to_datetime(TIMES[:2], format="%Y%m%d%H%M%S", utc=True)
    table = accessibility_table(times, depths)
    assert table["Impassable"].tolist() == [1, 0]
    assert table["FloodedPixels"].tolist() == [2, 0]
    assert (table["TotalPixels"] == 4).all()
    first = table.iloc[0]
    assert first["MaxDepth"] == pytest.approx(0.3)
    assert first["MinDepth"] == 0.0
    assert first["MeanDepth"] == pytest.approx(0.4 / 3)
    assert np.isnan(table.iloc[1]["MeanDepth"])


def test_roadway_event_csvs(tmp_path):
    """An event's ROI depth and WSE time series are written to CSV."""
    event = tmp_path / "TS_01_event"
    depth = np.full((10, 10), np.nan)
    depth[:5, :] = 0.2  # southern half flooded
    write_depth_store(event, {TIMES[1]: depth, TIMES[0]: depth * 0.5})
    # ROI covers image rows 5-9, i.e. the southern grid rows 0-4
    roi = write_labelme(tmp_path / "south_road.json", [
        ("roadway", "polygon", [[0, 5], [9, 5], [9, 9], [0, 9]])])

    analyzer = RoadwayAnalyzer(event, roi)
    assert analyzer.process_single_event()

    access = pd.read_csv(event / "south_road_accessibility_time_series.csv")
    assert access["Time"].is_monotonic_increasing
    assert access["MaxDepth"].tolist() == pytest.approx([0.1, 0.2])
    assert (access["FloodedPixels"] == access["TotalPixels"]).all()
    wse = pd.read_csv(event / "south_road_wse_time_series.csv")
    assert wse["MaxWSE"].tolist() == pytest.approx([1.1, 1.2])
    assert "south_road_depths" in read_zip_arrays(
        event / "zarr" / "south_road_depths.zip")


def test_roi_pixels_outside_the_grid_are_not_counted(tmp_path):
    """Coverage uses only ROI pixels that lie on the grid."""
    event = tmp_path / "TS_01_event"
    write_depth_store(event, {TIMES[0]: np.full((10, 10), 0.1)})
    # 15 x 5 polygon whose eastern 5 columns run past the 10-column grid
    roi = write_labelme(tmp_path / "edge_road.json", [
        ("roadway", "polygon", [[5, 0], [14, 0], [14, 4], [5, 4]])])
    assert RoadwayAnalyzer(event, roi).process_single_event()
    access = pd.read_csv(event / "edge_road_accessibility_time_series.csv")
    assert access.loc[0, "TotalPixels"] == 5 * 5
    assert access.loc[0, "FloodedPixels"] == 5 * 5


def test_roadway_event_without_depth_maps_reports_failure(tmp_path):
    """process_single_event returns False when no store exists."""
    roi = write_labelme(tmp_path / "r.json", [
        ("roadway", "polygon", [[0, 0], [3, 0], [3, 3]])])
    assert RoadwayAnalyzer(tmp_path / "empty_event", roi
                           ).process_single_event() is False


# ----- DepthMapPlotter -------------------------------------------------------


@pytest.fixture
def plot_event(tmp_path):
    """
    Event folder with depth maps, images, and SDFP observations.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    pathlib.Path
        Event folder named like a real event.
    """
    event = tmp_path / "TS_01_20240101114500_20240101123000"
    (event / "orig_images").mkdir(parents=True)
    depth = np.full((40, 50), np.nan)
    depth[10:20, 10:20] = np.linspace(0, 0.2, 10)
    write_depth_store(event, {stamp: depth for stamp in TIMES})
    for stamp in TIMES:
        (event / "orig_images" / f"CAM_TS_01_{stamp}.jpg").write_bytes(b"")
    (event / "orig_images" / "thumbs.db").write_bytes(b"")

    obs_times = pd.date_range("2024-01-01 11:55", periods=6, freq="5min",
                              tz="UTC")
    pd.DataFrame({
        "sensor_ID": "TS_01", "time_UTC": obs_times,
        "sensor_water_level": np.arange(6, dtype=float),
        "road_water_level": np.arange(6, dtype=float) / 10,
    }).to_csv(event / f"{event.name}.csv", index=False)
    return event


def make_plotter(event, sensors=((12, 15), (49, 39)), **kwargs):
    """
    Build a plotter for the synthetic 40 x 50 grid at 0.5 m spacing.

    Parameters
    ----------
    event : pathlib.Path
        Event folder.
    sensors : sequence of tuple, optional
        Virtual sensor ``(col, row)`` pixels.
    **kwargs
        Extra DepthMapPlotter options.

    Returns
    -------
    DepthMapPlotter
        Plotter instance.
    """
    return DepthMapPlotter(event, 712160, 712185, 33100, 33120,
                           resolution_m=0.5, virtual_sensor_locations=sensors,
                           **kwargs)


def test_images_match_nearest_observation_in_meters(plot_event):
    """Each image pairs with its nearest reading, converted to meters."""
    table = make_plotter(plot_event)._match_measurements_to_images()
    assert len(table) == 3  # thumbs.db is skipped
    first = table.iloc[0]  # 12:00:00 image -> 12:00 reading (index 1)
    assert first["closest_utc_time"] == pd.Timestamp("2024-01-01 12:00",
                                                     tz="UTC")
    assert first["obs_wse_meters"] == pytest.approx(1 * 0.3048)
    assert first["obs_depth_meters"] == pytest.approx(0.1 * 0.3048)


def test_virtual_sensors_sample_grid_pixels(plot_event):
    """Sensors read the maps at (col, row) with row 0 at the south."""
    plotter = make_plotter(plot_event)
    assert plotter._gen_virtual_sensor_depths()
    times, wse, depth = plotter._load_virtual_sensor_depths()
    assert len(times) == 3 and depth.shape == (3, 2)
    expected = np.linspace(0, 0.2, 10)[2]  # col 12 -> 3rd ramp value
    assert depth[0, 0] == pytest.approx(expected)
    assert np.isnan(depth[0, 1])  # second sensor is on dry ground
    assert wse[0, 0] == pytest.approx(expected + 1)


def test_no_virtual_sensors_skips_sampling(plot_event):
    """Without sensors nothing is sampled and plotting still works."""
    plotter = make_plotter(plot_event, sensors=None, plot_sensors=True)
    assert plotter.plot_sensors is False
    assert plotter._gen_virtual_sensor_depths() is False


def test_full_event_plotting_writes_every_figure(plot_event):
    """Maps and time series are drawn for every image and statistic."""
    plotter = make_plotter(plot_event, plot_sensors=True)
    plotter.preprocess_single_event()
    assert plotter.process_single_flood_event(["95_perc"]) == 6
    assert plotter.plot_all_time_series() == 3

    plots = plot_event / "plots"
    assert len(list((plots / "depth_maps").glob("*.png"))) == 3
    assert len(list((plots / "WSE_maps").glob("*.png"))) == 3
    assert len(list((plots / "time_series_using_depths").glob("*.png"))) == 6


def test_sensor_markers_are_placed_at_pixel_centers(plot_event):
    """Marker positions match the pixel-center Web Mercator location."""
    plotter = make_plotter(plot_event, sensors=((4, 2),), plot_sensors=True)
    geodata = plotter._load_and_prepare_geodata(np.zeros((40, 50)))

    class Recorder:
        """Axes stand-in that records scatter coordinates."""

        def scatter(self, xs, ys, **kwargs):
            """
            Record the plotted coordinates.

            Parameters
            ----------
            xs, ys : sequence of float
                Marker coordinates.
            **kwargs
                Ignored style options.
            """
            self.points = list(zip(xs, ys))

    axes = Recorder()
    plotter._plot_sensor_locations(axes, plotter.sensor_marker_path,
                                   geodata["shape"], geodata["transform"])
    # Column 4, row 2 from the south: center at x0 + 4.5 res, y0 + 2.5 res
    to_mercator = Transformer.from_crs("EPSG:32119", "EPSG:3857",
                                       always_xy=True)
    expected = to_mercator.transform(712160 + 4.5 * 0.5, 33100 + 2.5 * 0.5)
    np.testing.assert_allclose(axes.points[0], expected, atol=1e-6)
