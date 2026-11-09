"""Tests for the analysis scripts and the lock-file converter."""

import itertools

import cv2
import find_stat_anomalies
import image_stitcher
import numpy as np
import pandas as pd
import parse_lock
import pytest
import run_road_access_stats

import access_plotter


def test_depth_from_coverage_inverts_the_logistic_curve():
    """Depth -> coverage -> depth round-trips."""
    depth = np.linspace(0.0, 0.3, 7)
    params = (80.0, 40.0, 0.1)
    coverage = access_plotter.logistic_curve(depth, *params)
    np.testing.assert_allclose(
        access_plotter.depth_from_coverage(coverage, *params), depth,
        atol=1e-9)


def test_fit_recovers_logistic_parameters_in_depth():
    """A noise-free depth-from-coverage curve is recovered exactly."""
    coverage = np.linspace(5, 85, 40)
    depth = access_plotter.depth_from_coverage(coverage, 90, 50, 0.12)
    fit = access_plotter.fit_depth_from_coverage(coverage, depth)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert fit["rmse"] < 1e-6
    assert [fit["params"][k] for k in ("L", "k", "x0")] == pytest.approx(
        [90, 50, 0.12], rel=1e-4)
    assert set(fit["quantiles"]) == {0.1, 0.5, 0.9}


def test_quantile_curves_hold_their_share_of_depths():
    """The median and 80 % band hold 50 % and 80 % of noisy depths."""
    rng = np.random.default_rng(1)
    coverage = rng.uniform(2, 98, 3000)
    depth = (access_plotter.depth_from_coverage(coverage, 100.5, 45, 0.1)
             + rng.normal(0, 0.02, coverage.size))
    events = np.repeat(np.arange(60), 50)
    fit = access_plotter.fit_depth_from_coverage(coverage, depth, events,
                                                 n_boot=50)
    assert fit["below_median_fraction"] == pytest.approx(0.5, abs=0.02)
    assert fit["band_fraction"] == pytest.approx(0.8, abs=0.02)
    calibration = fit["calibration"]
    assert calibration["images"].sum() == 3000
    assert np.all(np.abs(calibration["fraction_inside_80pct_band"] - 0.8)
                  < 0.06)
    low, high = fit["curve_ci"]
    assert np.all(low <= fit["fit"] + 1e-12)
    assert np.all(fit["fit"] <= high + 1e-12)
    assert fit["param_ci"]["x0"][0] < 0.1 < fit["param_ci"]["x0"][1]


def test_zero_coverage_images_are_left_out_of_the_fit():
    """Coverage 0 has no finite depth on the curve and is excluded."""
    coverage = np.r_[0.0, np.linspace(5, 85, 20)]
    depth = np.r_[0.01, access_plotter.depth_from_coverage(
        coverage[1:], 90, 50, 0.12)]
    fit = access_plotter.fit_depth_from_coverage(coverage, depth)
    assert fit["n"] == 20 and fit["n_excluded"] == 1


def test_pinball_loss_is_smallest_at_the_quantile():
    """The 0.5 pinball loss is lowest at the median."""
    y = np.array([1.0, 2.0, 3.0, 10.0])
    loss = [access_plotter.pinball_loss(y - m, 0.5) for m in (1, 2.5, 6)]
    assert loss[1] < loss[0] and loss[1] < loss[2]
    # q = 0.9 penalizes values above the curve 9x more than below
    assert access_plotter.pinball_loss(np.array([1.0]), 0.9) == 0.9
    assert access_plotter.pinball_loss(np.array([-1.0]), 0.9) == (
        pytest.approx(0.1))


def test_r2_and_rmse_reference_values():
    """Metrics match hand calculations."""
    y = np.array([1.0, 2.0, 3.0])
    assert access_plotter.calculate_r2(y, y) == 1.0
    assert access_plotter.calculate_r2(np.ones(3), np.ones(3)) == 0.0
    assert access_plotter.calculate_rmse(y, y + 2) == pytest.approx(2.0)


def write_accessibility(folder, roi, max_depth, coverage, start):
    """
    Write an accessibility CSV for one synthetic event.

    Parameters
    ----------
    folder : pathlib.Path
        Event folder.
    roi : str
        ROI name.
    max_depth, coverage : array_like
        Per-image maximum depth (m) and coverage (%).
    start : str
        Time of the first image.

    Returns
    -------
    pathlib.Path
        CSV path.
    """
    folder.mkdir(parents=True, exist_ok=True)
    n = len(max_depth)
    total = 1000
    path = folder / f"{roi}_accessibility_time_series.csv"
    pd.DataFrame({
        "Time": pd.date_range(start, periods=n, freq="6min", tz="UTC"),
        "Impassable": (np.asarray(coverage) > 0).astype(int),
        "FloodedPixels": (np.asarray(coverage) / 100 * total).astype(int),
        "TotalPixels": total, "MeanDepth": np.asarray(max_depth) / 2,
        "MedianDepth": np.asarray(max_depth) / 2, "MaxDepth": max_depth,
        "MinDepth": 0.0,
    }).to_csv(path, index=False)
    return path


def test_anomaly_detection_flags_an_undercovered_image(tmp_path):
    """An image far below the depth-coverage trend is reported."""
    depth = np.linspace(0.06, 0.3, 60)
    coverage = 100 / (1 + np.exp(-30 * (depth - 0.15)))
    coverage[40] = 1.0  # deep water but almost no coverage
    write_accessibility(tmp_path / "events" / "ev1", "road", depth,
                        coverage, "2024-01-01")
    anomalies = find_stat_anomalies.find_anomalies(
        "road", [str(tmp_path / "events" / "ev1" /
                     "road_accessibility_time_series.csv")],
        z_threshold=-1.0, out_dir=str(tmp_path))
    assert 40 in anomalies.index
    assert (tmp_path / "road_anomaly_report.csv").exists()
    assert (tmp_path / "road_local_anomaly_diagnostic.jpg").exists()


def test_accessibility_plots_are_written_for_each_metric(tmp_path):
    """Every per-image and per-event plot is produced."""
    for i in range(4):
        depth = np.linspace(0.0, 0.2 + 0.02 * i, 12)
        write_accessibility(tmp_path / "events" / f"ev{i}", "road", depth,
                            100 * depth / depth.max(), f"2024-0{i + 1}-01")
    access_plotter.main(["--events_dir", str(tmp_path / "events"),
                         "--out_dir", str(tmp_path / "plots"),
                         "--n_boot", "20"])
    plots = {p.name for p in (tmp_path / "plots").iterdir()}
    for metric in access_plotter.METRICS:
        for suffix in ("1_evolution_plot.jpg", "1b_reversed_fit.jpg",
                       "1b_fit_summary.csv", "1b_band_calibration.csv",
                       "1b_binned_depth.jpg", "1b_depth_by_coverage.csv",
                       "2_summary_scatter.png", "3_depth_cdf.png",
                       "4_depth_pdf.png", "5_depth_exceedance.png"):
            assert f"road_{metric}_{suffix}" in plots
    runs = pd.read_csv(tmp_path / "plots" / "plot_summary.csv")
    peaks = runs[(runs["metric"] == "MaxDepth") & (runs["plot"] == "3")]
    assert peaks["events"].item() == 4
    first = runs[(runs["metric"] == "MaxDepth") & (runs["plot"] == "1")]
    assert first["images"].item() == 4 * 12  # every image has a depth
    assert runs["plot"].iloc[0] == "1"
    assert (tmp_path / "plots" / "plot_summary.md").exists()
    summary = pd.read_csv(tmp_path / "plots" /
                          "road_MaxDepth_1b_fit_summary.csv")
    assert {"r2_depth", "rmse_depth_m", "x0_ci95_low",
            "fraction_inside_80pct_band"} <= set(summary["quantity"])


def test_road_access_event_durations_and_impassable_time(tmp_path):
    """Each image stands for the time until the next image."""
    path = tmp_path / "ev" / "road_accessibility_time_series.csv"
    path.parent.mkdir()
    pd.DataFrame({
        "Time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:10",
                                "2024-01-01 00:20", "2024-01-01 00:50"],
                               utc=True),
        "Impassable": [1, 1, 0, 1], "FloodedPixels": [10, 20, 0, 40],
        "TotalPixels": 100, "MeanDepth": 0.1, "MaxDepth": [0.1, 0.2, 0, 0.3],
        "MinDepth": 0.0,
    }).to_csv(path, index=False)
    stats = run_road_access_stats.event_statistics(path, 0.05**2)
    # Gaps are 10, 10, 30 min; the first image borrows the second's 10
    assert stats["impassable_minutes"] == 10 + 10 + 30
    assert stats["prop_impassable"] == pytest.approx(50 / 60)
    assert stats["max_flooded_area"] == pytest.approx(40 * 0.05**2)

    summary = run_road_access_stats.summarize_roadway_stats(
        str(tmp_path), "road")
    assert summary.loc[0, "Total Events Processed"] == 1


def test_stitched_comparison_has_both_images_side_by_side(tmp_path):
    """The overlay is resized to the original and placed on the right."""
    orig = np.full((20, 30, 3), 50, np.uint8)
    overlay = np.full((10, 15, 3), 200, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), orig)
    cv2.imwrite(str(tmp_path / "b.png"), overlay)
    assert image_stitcher.stitch_image_pair(
        tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png") is None
    stitched = cv2.imread(str(tmp_path / "c.png"))
    assert stitched.shape == (20, 60, 3)
    assert (stitched[:, :30] == 50).all() and (stitched[:, 30:] == 200).all()
    assert "Could not read" in image_stitcher.stitch_image_pair(
        tmp_path / "missing.png", tmp_path / "b.png", tmp_path / "d.png")


def test_lock_file_becomes_an_explicit_spec():
    """Conda URLs are unescaped with md5 hashes; pip pins are listed."""
    lock = {"package": [
        {"platform": "linux-64", "manager": "conda", "name": "a",
         "url": "https://x/a-1%211.0.conda", "hash": {"md5": "abc"}},
        {"platform": "osx-64", "manager": "conda", "name": "b",
         "url": "https://x/b.conda"},
        {"platform": "linux-64", "manager": "pip", "name": "c",
         "version": "2.0"},
    ]}
    lines, pip = parse_lock.explicit_spec(lock)
    assert lines == ["@EXPLICIT", "https://x/a-1!1.0.conda#abc"]
    assert pip == ["c==2.0"]


def test_list_unreviewed_images_skips_scored_names(tmp_path):
    """Scored comparisons (mixed line endings) are excluded by name."""
    import list_unreviewed_images

    images = tmp_path / "daylight"
    images.mkdir()
    for name in ("CAM_DE_01_1.jpg", "CAM_DE_01_2.jpg", "CAM_DE_01_3.jpg",
                 "notes.txt"):
        (images / name).write_text("x")
    labels = tmp_path / "labels.csv"
    labels.write_bytes(b"Filename,Segmentation_Score\n"
                       b"comparison_CAM_DE_01_1.jpg,0\r\n"
                       b"comparison_CAM_DE_01_9.jpg,1\n")
    output = tmp_path / "todo.txt"

    assert list_unreviewed_images.main([
        "--labels", str(labels), "--image_dir", str(images),
        "--output", str(output)]) == 0
    assert output.read_text().split() == ["CAM_DE_01_2.jpg",
                                          "CAM_DE_01_3.jpg"]


def write_access(folder, roi, impassable, flooded, max_depth):
    """Write a minimal accessibility CSV for one ROI of one event."""
    times = pd.date_range("2024-01-01 12:00", periods=len(impassable),
                          freq="6min", tz="UTC")
    pd.DataFrame({
        "Time": times, "Impassable": impassable, "FloodedPixels": flooded,
        "TotalPixels": 10, "MeanDepth": max_depth, "MedianDepth": max_depth,
        "MaxDepth": max_depth, "MinDepth": max_depth,
    }).to_csv(folder / f"{roi}_accessibility_time_series.csv", index=False)


def test_compare_roadway_rois_pairs_images_and_footprints(tmp_path):
    """Two ROIs are paired by time; agreement, bias and IoU are reported."""
    import compare_roadway_rois as compare

    events = tmp_path / "flood_events"
    for name in ("DE_01_a", "DE_01_b"):
        folder = events / name
        folder.mkdir(parents=True)
        write_access(folder, "plan", [0, 1, 1], [0, 5, 10],
                     [np.nan, 0.1, 0.2])
        write_access(folder, "cam", [0, 0, 1], [0, 0, 8],
                     [np.nan, np.nan, 0.3])
    mask_a = np.zeros((10, 10), np.uint8)
    mask_a[:, :6] = 255  # 60 cells
    mask_b = np.zeros((10, 10), np.uint8)
    mask_b[:, 3:] = 255  # 70 cells, 30 shared
    cv2.imwrite(str(events / "DE_01_a" / "plan_roi_mask.png"), mask_a)
    cv2.imwrite(str(events / "DE_01_a" / "cam_roi_mask.png"), mask_b)

    out = tmp_path / "comparison"
    assert compare.main(["--events_dir", str(events), "--roi_a", "plan",
                         "--roi_b", "cam", "--output_dir", str(out)]) == 0

    summary = pd.read_csv(out / "roi_comparison_summary.csv").set_index(
        "metric")["value"]
    assert summary["images"] == 6
    assert summary["impassable_agreement_pct"] == pytest.approx(200 / 3)
    assert summary["impassable_a_only"] == 2
    # Max depth b - a per image: 0, -0.1, +0.1 (dry depths count as 0)
    assert summary["MaxDepth_bias"] == pytest.approx(0.0)
    assert summary["MaxDepth_mae"] == pytest.approx(0.2 / 3)
    assert summary["footprint_iou"] == pytest.approx(30 / 100)
    events_table = pd.read_csv(out / "roi_comparison_events.csv")
    assert events_table["impassable_images_a"].tolist() == [2, 2]
    assert (out / "roi_footprints.png").exists()
    assert (out / "roi_comparison_MaxDepth.png").exists()


def test_sensor_data_accepts_detailed_event_csv(tmp_path):
    """flood_events.csv-style readings load like SDFP downloads."""
    path = tmp_path / "sensor_data_comparison.csv"
    pd.DataFrame({
        "flood_event": [1, 1, 2],
        "sensor_ID": "DE_01",
        "time_UTC": ["2024-01-01 12:00+00:00", "2024-01-01 12:06+00:00",
                     "2024-01-01 12:06+00:00"],  # overlapping windows
        "road_water_level": [-0.1, 0.5, 0.5],
        "sensor_water_level": [1.0, 1.6, 1.6],
    }).to_csv(path, index=False)
    sensor = access_plotter.load_sensor_data(str(path))
    assert len(sensor) == 2
    assert sensor["SensorDepth_m"].tolist() == pytest.approx(
        [0.0, 0.5 * 0.3048])
    assert sensor["SensorWSE_m"].iloc[1] == pytest.approx(1.6 * 0.3048)


def test_empirical_probabilities_handle_ties():
    """CDF is P(X <= x) and exceedance P(X >= x) at each distinct value."""
    x, cdf, exceed = access_plotter.empirical_probabilities(
        [0.3, 0.1, 0.2, 0.2])
    assert x.tolist() == [0.1, 0.2, 0.3]
    assert cdf.tolist() == pytest.approx([0.25, 0.75, 1.0])
    assert exceed.tolist() == pytest.approx([1.0, 0.75, 0.25])
    _, cdf_w, exceed_w = access_plotter.empirical_probabilities(
        [0.3, 0.1, 0.2, 0.2], "weibull")
    assert cdf_w.tolist() == pytest.approx([0.2, 0.6, 0.8])
    assert exceed_w.tolist() == pytest.approx([0.8, 0.6, 0.2])


def test_density_estimates_integrate_to_one():
    """The histogram and the reflected KDE are proper densities."""
    rng = np.random.default_rng(2)
    values = rng.gamma(2.0, 0.04, 80)  # positive, piled up near 0
    edges, rule = access_plotter.density_bins(values)
    density, _ = np.histogram(values, bins=edges, density=True)
    assert rule == "Freedman-Diaconis"
    assert (density * np.diff(edges)).sum() == pytest.approx(1.0)
    x, kde = access_plotter.reflected_kde(values, n_points=4000)
    assert x[0] == 0.0
    assert np.trapezoid(kde, x) == pytest.approx(1.0, abs=0.01)


def test_event_peaks_can_use_well_segmented_images_only():
    """A badly segmented image cannot set the event's peak."""
    event = pd.DataFrame({"Coverage": [20.0, 90.0], "MaxDepth": [0.05, 0.4],
                          "MeanDepth": [0.02, 0.2],
                          "MedianDepth": [0.02, 0.2],
                          "Segmentation": ["good", "other"]})
    assert access_plotter.summarize_event(event)["MaxDepth_max"] == 0.4
    good = access_plotter.summarize_event(event, good_only=True)
    assert good["MaxDepth_max"] == 0.05 and good["images"] == 1


def test_depth_by_coverage_bins_and_full_coverage():
    """Bins are [low, high); full coverage is its own bin with CIs."""
    coverage = np.r_[np.full(12, 5.0), np.full(12, 15.0), np.full(12, 100.0),
                     [9.999, 98.9, 99.0]]
    depth = np.r_[np.linspace(0.01, 0.03, 12), np.linspace(0.04, 0.06, 12),
                  np.linspace(0.10, 0.30, 12), [0.02, 0.09, 0.2]]
    events = np.r_[np.tile(np.arange(4), 9), [0, 1, 2]]
    table = access_plotter.depth_by_coverage(coverage, depth, events,
                                             n_boot=200)
    by_bin = table.set_index("bin")
    assert list(by_bin.index) == ["0-<10%", "10-<20%", "90-<99%",
                                  ">=99% (full)"]
    assert by_bin.loc["0-<10%", "images"] == 13  # 9.999 is below 10
    assert by_bin.loc[">=99% (full)", "images"] == 13  # 99.0 is full
    assert by_bin.loc[">=99% (full)", "full_coverage"]
    full = by_bin.loc[">=99% (full)"]
    values = np.r_[np.linspace(0.10, 0.30, 12), 0.2]
    assert full["median_depth_m"] == pytest.approx(np.median(values))
    assert full["p10_depth_m"] == pytest.approx(np.percentile(values, 10))
    assert (full["median_depth_m_ci95_low"] <= full["median_depth_m"]
            <= full["median_depth_m_ci95_high"])
    assert by_bin.loc["90-<99%", "low_sample"]  # one image
    assert not by_bin.loc["0-<10%", "low_sample"]


def test_feathers_palette_passes_colour_checks():
    """Categorical slots keep contrast and CVD separation on white."""
    import plot_style

    def linear(color):
        c = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def oklab(rgb):
        lms = np.cbrt(np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005]]) @ np.clip(rgb, 0, 1))
        return np.array([
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms

    protan = np.array([[0.152286, 1.052583, -0.204868],
                       [0.114503, 0.786281, 0.099216],
                       [-0.003882, -0.048116, 1.051998]])
    deutan = np.array([[0.367322, 0.860646, -0.227968],
                       [0.280085, 0.672501, 0.047413],
                       [-0.011820, 0.042940, 0.968881]])
    slots = plot_style.CATEGORICAL
    for color in slots:
        lum = np.array([0.2126, 0.7152, 0.0722]) @ linear(color)
        assert (1.05) / (lum + 0.05) >= 3.0
        lightness, a, b = oklab(linear(color))
        assert 0.43 <= lightness <= 0.77 and np.hypot(a, b) >= 0.10
    for x, y in itertools.combinations(slots, 2):
        assert 100 * np.linalg.norm(oklab(linear(x)) - oklab(linear(y))) >= 15
        for sim in (protan, deutan):
            assert 100 * np.linalg.norm(oklab(sim @ linear(x))
                                        - oklab(sim @ linear(y))) >= 8


def test_sensor_depth_plot_is_made_once_per_roi(tmp_path):
    """Plot 8 bins the sensor depth of well-segmented images by coverage."""
    labels, sensor = [], []
    for i in range(4):
        depth = np.linspace(0.02, 0.2 + 0.02 * i, 12)
        start = f"2024-0{i + 1}-01"
        write_accessibility(tmp_path / "events" / f"ev{i}", "road", depth,
                            100 * depth / depth.max(), start)
        times = pd.date_range(start, periods=12, freq="6min", tz="UTC")
        labels += [(f"comparison_CAM_DE_01_{t:%Y%m%d%H%M%S}.jpg", 1)
                   for t in times]
        sensor += [(t, d / 0.3048) for t, d in zip(times, depth)]  # feet
    pd.DataFrame(labels, columns=["Filename", "Segmentation_Score"]).to_csv(
        tmp_path / "labels.csv", index=False)
    pd.DataFrame(sensor, columns=["time_UTC", "road_water_level"]).to_csv(
        tmp_path / "sensor.csv", index=False)

    access_plotter.main(["--events_dir", str(tmp_path / "events"),
                         "--out_dir", str(tmp_path / "plots"),
                         "--labels", str(tmp_path / "labels.csv"),
                         "--sensor", str(tmp_path / "sensor.csv"),
                         "--n_boot", "20"])
    plots = {p.name for p in (tmp_path / "plots").iterdir()}
    for suffix in ("8_binned_depth_good_only.jpg", "8_depth_by_coverage.csv",
                   "8_reversed_fit_good_only.jpg", "8_fit_summary.csv"):
        assert f"road_SensorDepth_{suffix}" in plots
    for metric in access_plotter.METRICS:
        assert f"road_{metric}_9_image_vs_sensor_binned_good_only.jpg" in plots
    paired = pd.read_csv(tmp_path / "plots" /
                         "road_MaxDepth_9_depth_by_coverage.csv")
    assert set(paired["depth_source"]) == {"Image maximum depth",
                                           "In-situ sensor depth"}
    # Same images for both sources, bin by bin
    counts = paired.pivot_table(index="bin", columns="depth_source",
                                values="images")
    assert (counts.iloc[:, 0] == counts.iloc[:, 1]).all()
    table = pd.read_csv(tmp_path / "plots" /
                        "road_SensorDepth_8_depth_by_coverage.csv")
    assert table["images"].sum() == 48
    runs = pd.read_csv(tmp_path / "plots" / "plot_summary.csv")
    sensor_rows = runs[runs["metric"] == "SensorDepth"]
    assert set(sensor_rows["plot"]) == {"8 binned", "8 logistic"}
    assert (sensor_rows["events"] == 4).all()


def test_shared_depth_limits_cover_image_and_sensor_depths():
    """1c and plot 8 share 0 to just above the deepest good depth."""
    data = pd.DataFrame({
        "Segmentation": ["good", "good", "other"], "Coverage": [10, 50, 90],
        "MaxDepth": [0.1, 0.2, 0.9], "MeanDepth": [0.05, 0.1, 0.5],
        "MedianDepth": [0.05, 0.1, 0.5], "SensorDepth_m": [0.3, np.nan, 0.8]})
    low, high = access_plotter.shared_depth_limits(data)
    assert low == 0 and high == pytest.approx(1.05 * 0.3)  # 'other' ignored


def test_sensor_camera_comparison_timing_and_states(tmp_path):
    """Camera onset/end are compared with the sensor inside the image span."""
    import sensor_camera_comparison as sc
    from poseidon_utils.zarr_io import write_array, write_zip_group

    times = pd.date_range("2024-06-01 12:00", periods=11, freq="6min",
                          tz="UTC")
    wet = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)  # 12:06-12:48
    depth = np.where(wet, 0.1, np.nan)
    for name, has_sensor in (("DE_01_20240601110000_20240601140000", True),
                             ("DE_01_20240602110000_20240602140000", False)):
        folder = tmp_path / "events" / name
        (folder / "zarr").mkdir(parents=True)
        stamps = times + pd.Timedelta(days=0 if has_sensor else 1)
        pd.DataFrame({"Time": stamps, "FloodedPixels": wet * 10,
                      "TotalPixels": 20, "MaxDepth": depth,
                      "MeanDepth": depth / 2, "MedianDepth": depth / 2,
                      "MinDepth": np.where(wet, 0.0, np.nan)}).to_csv(
            folder / "road_accessibility_time_series.csv", index=False)
        with write_zip_group(folder / "zarr" / "virtual_sensor_data.zip") as g:
            write_array(g, "timestamps",
                        np.array(stamps.astype(str), dtype="U30"))
            write_array(g, "vs_depth", depth[:, None])
            write_array(g, "vs_wse", depth[:, None])
    # Sensor wet 12:18-12:36 only: camera is 12 min earlier and later
    sensor_wet = (times >= "2024-06-01 12:18") & (times <= "2024-06-01 12:36")
    pd.DataFrame({"time_UTC": times,
                  "road_water_level": np.where(sensor_wet, 0.2, -0.1)}
                 ).to_csv(tmp_path / "sensor.csv", index=False)

    out = tmp_path / "out"
    assert sc.main(["--events_dir", str(tmp_path / "events"), "--roi", "road",
                    "--sensor", str(tmp_path / "sensor.csv"),
                    "--out_dir", str(out)]) == 0
    timing = pd.read_csv(out / "road_sensor_camera_all_images_timing.csv")
    first = timing[(timing["source"] == "ROI")
                   & timing["event"].str.contains("0601")].iloc[0]
    assert first["onset_lead_min"] == 12 and first["end_lag_min"] == 12
    assert not first["camera_onset_censored"]
    obs = pd.read_csv(
        out / "road_sensor_camera_all_images_observations.csv")
    roi_max = obs[obs["source"] == "ROI max"]
    assert set(roi_max["sensor_state"]) == {"sensor wet", "sensor dry",
                                            "no sensor reading"}
    assert (roi_max.loc[~roi_max["has_sensor"], "sensor_state"]
            == "no sensor reading").all()
    peaks = pd.read_csv(
        out / "road_sensor_camera_all_images_event_peaks.csv")
    assert peaks["has_sensor"].tolist() == [True, False]
    for suffix in ("depths.jpg", "timing.jpg", "peaks.jpg", "summary.md"):
        assert (out / f"road_sensor_camera_all_images_{suffix}").exists()
    assert not list(out.glob("*good_only*"))  # no labels, no second set

    labels = tmp_path / "labels.csv"
    pd.DataFrame({"Filename": [f"comparison_CAM_DE_01_{t:%Y%m%d%H%M%S}.jpg"
                               for t in times[:5]],
                  "Segmentation_Score": 1}).to_csv(labels, index=False)
    assert sc.main(["--events_dir", str(tmp_path / "events"), "--roi", "road",
                    "--sensor", str(tmp_path / "sensor.csv"), "--labels",
                    str(labels), "--out_dir", str(out)]) == 0
    good = pd.read_csv(out / "road_sensor_camera_good_only_observations.csv")
    assert good["time"].nunique() < obs["time"].nunique()
    assert (out / "road_sensor_camera_good_only_depths.jpg").exists()


def test_sensor_matching_accepts_mixed_time_resolutions(tmp_path):
    """Camera and sensor times parsed at different resolutions still match."""
    import sensor_camera_comparison as sc

    camera = pd.Series(pd.to_datetime(["2024-01-01 12:00:00"], utc=True)
                       .astype("datetime64[s, UTC]"))
    sensor = pd.DataFrame({
        "datetime_utc": pd.to_datetime(["2024-01-01 12:02:00"], utc=True)
        .astype("datetime64[us, UTC]"),
        "SensorDepth_m": [0.1]})
    state = sc.sensor_state(camera, sensor, 5)
    assert state["sensor_state"].tolist() == ["sensor wet"]
    assert state["sensor_depth_m"].tolist() == [0.1]

    event = pd.DataFrame({"Time": ["2024-01-01 12:00:00+00:00"],
                          "FloodedPixels": [1], "TotalPixels": [2],
                          "MaxDepth": [0.2]})
    path = tmp_path / "ev" / "road_accessibility_time_series.csv"
    path.parent.mkdir()
    event.to_csv(path, index=False)
    loaded = access_plotter.load_event(str(path), {"20240101120000"},
                                       sensor)
    assert loaded["SensorDepth_m"].tolist() == [0.1]
