"""End-to-end test of the HPC stage drivers on a synthetic site.

A synthetic site has LiDAR ground at 1.0 m with a 20 cm dip, a nadir
camera 10 m above datum, and label images whose water patch covers a
known area around the dip.
Stages 06-11 run through their real command-line entry points.
"""

import json

import cv2
import numpy as np
import pandas as pd
import pytest
import run_calc_depths
import run_prep_grid
import run_rectify
import run_road_access_stats
import run_roadway_analyzer
import yaml
from conftest import NADIR_EXTRINSICS, NADIR_INTRINSICS, write_las

from poseidon_utils.zarr_io import read_zip_arrays

TIMES = ["20240101120000", "20240101120600"]

#: Depth (m) of the ground depression under the water patch.
DIP = 0.2


def expected_wet_cells(grid, patch_rows=(100, 200), patch_cols=(150, 250)):
    """
    Grid cells whose nadir projection falls inside the water patch.

    Parameters
    ----------
    grid : numpy.lib.npyio.NpzFile
        Shared grid with ``x``, ``y`` and ``z``.
    patch_rows, patch_cols : tuple of int, optional
        Half-open image row and column range painted as water.

    Returns
    -------
    numpy.ndarray
        Boolean mask in grid orientation.
    """
    c0u, c0v, fx, fy = NADIR_INTRINSICS[2:6]
    xc, yc, zc = NADIR_EXTRINSICS[:3] + np.array([0, 0, 1.0])
    height = zc - grid["z"]
    u = np.round(c0u + fx * (grid["x"] - xc) / height)
    v = np.round(c0v - fy * (grid["y"] - yc) / height)
    return ((u >= patch_cols[0]) & (u < patch_cols[1])
            & (v >= patch_rows[0]) & (v < patch_rows[1]))


@pytest.fixture
def site(tmp_path):
    """
    Write a synthetic site: LiDAR, YAML config, ROI, and one event.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    dict
        Paths: ``config``, ``grid_dir``, ``event``.
    """
    xs, ys = np.meshgrid(np.arange(-1, 14, 0.25), np.arange(-1, 12, 0.25))
    ground = 1.0 - DIP * np.exp(-((xs - 5) ** 2 + (ys - 5) ** 2) / 4)
    las = write_las(tmp_path / "ground.las", xs.ravel(), ys.ravel(),
                    ground.ravel(), np.full(xs.size, 2))

    # Camera 9 m above the 1 m ground: same geometry as conftest's
    # nadir camera over z = 0, raised by 1 m
    extrinsics = NADIR_EXTRINSICS.copy()
    extrinsics[2] += 1.0
    config = tmp_path / "test_site.yaml"
    config.write_text(yaml.safe_dump({
        "name": "test_site", "sensor_id": "TS_01", "crs": "EPSG:32119",
        "grid": {"min_x": 0, "max_x": 12, "min_y": 0, "max_y": 10,
                 "resolution": 0.25},
        "lidar_file": str(las), "lidar_units": "meters",
        "intrinsics": NADIR_INTRINSICS.tolist(),
        "extrinsics": extrinsics.tolist(),
        "virtual_sensors": [[20, 20]],
        "roi_json": str(tmp_path / "road.json"),
    }))

    # Road ROI drawn on the rectified (north-up) image
    (tmp_path / "road.json").write_text(json.dumps({"shapes": [{
        "label": "roadway", "shape_type": "polygon",
        "points": [[0, 0], [47, 0], [47, 39], [0, 39]]}]}))

    event = tmp_path / "flood_events" / "TS_01_20240101110000_20240101130000"
    for sub in ("orig_images", "labels"):
        (event / sub).mkdir(parents=True)
    for i, stamp in enumerate(TIMES):
        name = f"CAM_TS_01_{stamp}"
        cv2.imwrite(str(event / "orig_images" / f"{name}.jpg"),
                    np.full((300, 400, 3), 128, np.uint8))
        labels = np.zeros((300, 400), np.uint8)
        labels[100:200, 150:250 + 20 * i] = 1  # water grows over time
        cv2.imwrite(str(event / "labels" / f"{name}_predseg_labels.png"),
                    labels)
    return {"config": str(config), "grid_dir": str(tmp_path / "grids"),
            "event": str(event)}


def test_stages_06_to_11_produce_consistent_outputs(site, tmp_path):
    """Grid, rectification, depths, roadway CSVs, and summary agree."""
    common = ["--site", site["config"], "--grid_dir", site["grid_dir"]]
    run_prep_grid.main(common)
    grid = np.load(f"{site['grid_dir']}/shared_grid_test_site.npz")
    assert grid["z"].shape == (40, 48)
    assert grid["z"].min() == pytest.approx(1.0 - DIP, abs=0.01)

    event = ["--target_event_dir", site["event"]]
    run_rectify.main(common + event + ["--disable_gpu"])
    run_calc_depths.main(common + event + ["--disable_gpu"])
    run_roadway_analyzer.main(["--site", site["config"]] + event)
    summary_csv = tmp_path / "summary.csv"
    run_road_access_stats.main([
        "--site", site["config"],
        "--event_dir", str(tmp_path / "flood_events"),
        "--resolution", "0.25", "--output_csv", str(summary_csv)])

    # Wet cells: those whose projected pixel lies in the painted patch
    wet = expected_wet_cells(grid, patch_cols=(150, 250))
    depth_maps = read_zip_arrays(f"{site['event']}/zarr/depth_maps.zip")
    first = f"CAM_TS_01_{TIMES[0]}_predseg_labels_rectified"
    wse = depth_maps[f"{first}_wse_map_95_perc"]
    np.testing.assert_array_equal(np.isfinite(wse), wet)

    # Water stands at the shoreline ground level, above the dip floor
    access = pd.read_csv(
        f"{site['event']}/road_accessibility_time_series.csv")
    assert (access["Impassable"] == 1).all()
    assert access["FloodedPixels"].is_monotonic_increasing
    assert 0 < access["FloodedPixels"].iloc[0] <= wet.sum()
    assert 0.5 * DIP < access["MaxDepth"].iloc[0] < DIP

    summary = pd.read_csv(summary_csv)
    assert summary.loc[0, "Total Events Processed"] == 1


def test_rectify_stage_exits_when_the_grid_is_missing(site):
    """A missing shared grid stops the stage with exit status 1."""
    with pytest.raises(SystemExit) as stop:
        run_rectify.main(["--site", site["config"], "--grid_dir",
                          site["grid_dir"], "--target_event_dir",
                          site["event"], "--disable_gpu"])
    assert stop.value.code == 1
