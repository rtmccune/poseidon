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
    for suffix in ("depths.jpg", "timing.jpg", "peaks.jpg", "summary.md",
                   "aligned_start.jpg", "aligned_peak.jpg"):
        assert (out / f"road_sensor_camera_all_images_{suffix}").exists()

    # Aligned series: the event folder 11:00-14:00 is shorter than its
    # 3 h padding on each side, so its flood period collapses to the
    # midpoint, 12:30; the sensor is at its (constant) maximum there
    _, _, series = sc.collect(str(tmp_path / "events"), "road",
                              sc.load_sensor_data(str(tmp_path /
                                                      "sensor.csv")),
                              set(), 5)
    first = series[series["event"].str.contains("0601")]
    assert (first["flood_start"] == pd.Timestamp("2024-06-01 12:30",
                                                 tz="UTC")).all()
    assert (first["buffer_start"] == pd.Timestamp("2024-06-01 11:00",
                                                  tz="UTC")).all()
    from_buffer = sc.aligned_series(first, "start")
    assert from_buffer["hours"].min() == pytest.approx(1.0)  # 12:00 - 11:00
    from_flood = sc.aligned_series(first, "start", "flood")
    assert from_flood["hours"].min() == pytest.approx(-0.5)  # 12:00 - 12:30
    assert (first["sensor_peak"] == pd.Timestamp("2024-06-01 12:30",
                                                 tz="UTC")).all()
    vs = first[first["source"] == "VS1"]
    assert len(vs) == 11 and vs["depth_m"].isna().sum() == 3  # dry = NaN
    # Dry times skipped by default; counted as 0 m on request
    aligned = sc.aligned_series(first, "start")
    wet_only = sc.composites(aligned, min_events=1)
    with_zero = sc.composites(aligned.fillna({"depth_m": 0.0}),
                              min_events=1)
    vs_bins = wet_only[wet_only["source"] == "VS1"]
    assert vs_bins["center_m"].to_numpy() == pytest.approx(0.1)  # wet mean
    assert (with_zero[with_zero["source"] == "VS1"]["center_m"]
            < 0.1).any()
    peak = sc.aligned_series(first, "peak")
    sensor_rows = peak[peak["source"] == sc.SENSOR]
    at_zero = sensor_rows.loc[sensor_rows["hours"] == 0, "depth_m"]
    assert at_zero.item() == sensor_rows["depth_m"].max()
    assert sc.aligned_series(series, "peak")["event"].nunique() == 1
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


def test_nearest_event_mask_keeps_overlap_images_once():
    """An image in two padded windows stays in the nearer flood period."""
    # Event A floods 12:00-13:00, B floods 17:00-18:00; with 3 h padding
    # their windows (09-16 and 14-21) overlap from 14:00 to 16:00
    a = "DE_01_20240101090000_20240101160000"
    b = "DE_01_20240101140000_20240101210000"
    times = pd.to_datetime(["2024-01-01 14:30", "2024-01-01 15:45",
                            "2024-01-01 14:30", "2024-01-01 15:45",
                            "2024-01-01 12:30"], utc=True)
    keep = access_plotter.nearest_event_mask([a, a, b, b, a], times, 3)
    # 14:30 is 1.5 h after A and 2.5 h before B: kept in A
    # 15:45 is 2.75 h after A and 1.25 h before B: kept in B
    assert keep.tolist() == [True, False, False, True, True]


def test_overlapping_events_count_each_image_once(tmp_path):
    """Plots and the sensor comparison count shared images once."""
    import sensor_camera_comparison as sc

    times = pd.date_range("2024-01-01 14:00", periods=11, freq="12min",
                          tz="UTC")
    depth = np.linspace(0.02, 0.2, 11)
    for name in ("DE_01_20240101090000_20240101160000",
                 "DE_01_20240101140000_20240101210000"):
        write_accessibility(tmp_path / "events" / name, "road", depth,
                            100 * depth / depth.max(), "2024-01-01 14:00")
    access_plotter.main(["--events_dir", str(tmp_path / "events"),
                         "--out_dir", str(tmp_path / "plots"),
                         "--n_boot", "0"])
    runs = pd.read_csv(tmp_path / "plots" / "plot_summary.csv")
    first = runs[(runs["metric"] == "MaxDepth") & (runs["plot"] == "1")]
    assert first["images"].item() == 11  # not 22
    summary = (tmp_path / "plots" / "plot_summary.md").read_text()
    assert "Duplicate copies removed (images in two overlapping event " \
           "windows, kept in the nearer event): 11" in summary

    out = tmp_path / "out"
    assert sc.main(["--events_dir", str(tmp_path / "events"), "--roi",
                    "road", "--out_dir", str(out)]) == 0
    obs = pd.read_csv(out / "road_sensor_camera_all_images_observations.csv")
    roi_max = obs[obs["source"] == "ROI max"]
    assert len(roi_max) == roi_max["time"].nunique() == 11


def test_sensor_that_never_saw_water_has_no_peak():
    """Peak alignment skips events whose sensor stayed dry."""
    import sensor_camera_comparison as sc

    times = pd.date_range("2024-06-01 12:00", periods=5, freq="6min",
                          tz="UTC")
    dry = pd.DataFrame({"datetime_utc": times, "SensorDepth_m": 0.0})
    roi = pd.DataFrame({"time": times, "wet": True, "MeanDepth": 0.05,
                        "MinDepth": 0.0, "MedianDepth": 0.05,
                        "MaxDepth": 0.1})
    series = sc.event_series(
        [("DE_01_20240601090000_20240601160000", roi, None, dry)], 3)
    assert series["sensor_peak"].isna().all()
    assert sc.aligned_series(series, "peak").empty
    assert not sc.aligned_series(series, "start").empty


def test_sensor_wet_threshold_matches_event_detection():
    """Sensor levels at or below 0.02 ft count as dry."""
    import sensor_camera_comparison as sc

    assert sc.SENSOR_WET_M == pytest.approx(0.02 * 0.3048)
    times = pd.Series(pd.date_range("2024-01-01 12:00", periods=3,
                                    freq="6min", tz="UTC"))
    sensor = pd.DataFrame({"datetime_utc": times,
                           "SensorDepth_m": [0.0, 0.01 * 0.3048,
                                             0.05 * 0.3048]})
    state = sc.sensor_state(times, sensor, 1)
    assert state["sensor_state"].tolist() == ["sensor dry", "sensor dry",
                                              "sensor wet"]


def _write_publication_site(site_dir, scale=1.0):
    """Write four synthetic events, labels and sensor CSVs for a site."""
    from poseidon_utils.zarr_io import write_array, write_zip_group

    labels, sensor = [], []
    for e in range(4):
        start = pd.Timestamp("2024-06-01 12:00", tz="UTC") + pd.Timedelta(
            days=e)
        name = (f"DE_01_{start - pd.Timedelta(hours=3):%Y%m%d%H%M%S}_"
                f"{start + pd.Timedelta(hours=4):%Y%m%d%H%M%S}")
        folder = site_dir / "flood_events" / name
        (folder / "zarr").mkdir(parents=True)
        times = pd.date_range(start - pd.Timedelta(minutes=30), periods=16,
                              freq="6min", tz="UTC")
        depth = np.clip(np.sin(np.linspace(0, np.pi, 16)) * 0.1 * scale, 0,
                        None)
        wet = depth > 0
        pd.DataFrame({
            "Time": times, "Impassable": wet.astype(int),
            "FloodedPixels": (np.linspace(5, 100, 16) * 10).astype(int),
            "TotalPixels": 1000, "MaxDepth": depth * 1.5,
            "MeanDepth": depth, "MedianDepth": depth,
            "MinDepth": 0.0}).to_csv(
            folder / "road_accessibility_time_series.csv", index=False)
        with write_zip_group(folder / "zarr" / "virtual_sensor_data.zip") as g:
            write_array(g, "timestamps",
                        np.array(times.astype(str), dtype="U30"))
            vs = np.where(wet, depth, np.nan)[:, None]
            write_array(g, "vs_depth", vs)
            write_array(g, "vs_wse", vs)
        labels += [(f"comparison_CAM_DE_01_{t:%Y%m%d%H%M%S}.jpg", 1)
                   for t in times]
        sensor += [(t, d * 0.9 / 0.3048) for t, d in zip(times, depth)]
    pd.DataFrame(labels, columns=["Filename", "Segmentation_Score"]).to_csv(
        site_dir / "segmentation_labels.csv", index=False)
    pd.DataFrame(sensor, columns=["time_UTC", "road_water_level"]).to_csv(
        site_dir / "sensor_data_comparison.csv", index=False)


def test_publication_figures_from_events(tmp_path):
    """Both figures and their CSVs are written from good images only."""
    import publication_figures as pub

    _write_publication_site(tmp_path)
    out = tmp_path / "pub"
    assert pub.main(["--events_dir", str(tmp_path / "flood_events"),
                     "--roi", "road", "--sensor",
                     str(tmp_path / "sensor_data_comparison.csv"),
                     "--labels", str(tmp_path / "segmentation_labels.csv"),
                     "--out_dir", str(out), "--n_boot", "20"]) == 0
    for name in ("road_pub_fig1_aligned_timing.png",
                 "road_pub_fig1_aligned_timing.pdf",
                 "road_pub_fig2_binned_depth.png",
                 "road_pub_fig2_binned_depth.pdf"):
        assert (out / name).exists()
    vs1 = pd.read_csv(out / "road_pub_fig1_aligned_timing_a_vs1_agreement.csv")
    # VS1 reads 10 % deeper than the sensor wherever both show water
    assert vs1["bias_m"].item() > 0 and vs1["sensor_only_wet"].item() == 0
    agreement = pd.read_csv(out / "road_pub_fig2_binned_depth_agreement.csv")
    assert agreement["image_metric"].tolist() == [
        "Mean Roadway Depth in Field of View",
        "Maximum Roadway Depth in Field of View"]
    means = pd.read_csv(out / "road_pub_fig1_aligned_timing_a_means.csv")
    sensor_hours = means.loc[means["source"] == "In-situ sensor", "hours"]
    assert sensor_hours.min() >= 3  # never before the event start (+3 h)
    metrics = (out / "road_pub_metrics.md").read_text()
    assert "VS1 against the in-situ sensor" in metrics
    assert "Anywhere on Roadway" in metrics and "Bias (m)" in metrics


def test_combined_publication_figures_side_by_side(tmp_path):
    """Two sites make both combined figures, panel (a) peak-aligned."""
    import combined_publication_figures as combined

    _write_publication_site(tmp_path / "left_site")
    _write_publication_site(tmp_path / "right_site", scale=1.5)
    out = tmp_path / "combined"
    assert combined.main(["--sites", "left_site", "right_site",
                          "--rois", "road", "road", "--data_root",
                          str(tmp_path), "--out_dir", str(out),
                          "--n_boot", "20"]) == 0
    for name in ("combined_fig1_aligned_timing",
                 "combined_fig2_depth_by_coverage"):
        assert (out / f"{name}.png").exists()
        assert (out / f"{name}.pdf").exists()
    means = pd.read_csv(
        out / "left_site_combined_fig1_aligned_timing_a_peak_means.csv")
    # Hour 0 is the sensor peak, so the composite runs on both sides of it
    assert means["hours"].min() < 0 < means["hours"].max()
    # Dry times count as 0 m, so the sensor mean falls to 0 m away from
    # its peak instead of leaving those times out
    sensor_means = means.loc[means["source"] == "In-situ sensor", "center_m"]
    assert sensor_means.min() == 0 and sensor_means.max() > 0
    # Each line's event count is recorded for the legend and caption
    events = means.groupby("source")["source_events"].first()
    assert events["VS1"] == 4 and events["In-situ sensor"] == 4
    membership = pd.read_csv(
        out / "left_site_combined_fig1_aligned_timing_event_membership.csv")
    assert set(membership["source"]) == {"ROI", "VS1"}
    assert (membership["reason"] == "in (a) and (b)-(c)").all()
    funnel = (out / "event_funnel.md").read_text()
    assert "| Event folders (stage 05) | 4 | 4 |" in funnel
    assert "| Figure 1 (a): sensor peak and a well-segmented image | 4 | 4 |" \
        in funnel
    assert (out / "right_site_combined_fig2_depth_by_coverage_1c_mean_depth"
                  "_fit_summary.csv").exists()


def test_event_membership_reasons():
    """Each event is placed in (a), (b)-(c), both or neither, with why."""
    import publication_figures as pub
    import sensor_camera_comparison as sc

    peak = pd.Timestamp("2024-06-01 12:00", tz="UTC")
    rows = []
    for event, has_peak, depth in (("both", True, 0.1), ("dry_cam", True, 0.0),
                                   ("night", True, 0.1),
                                   ("no_peak", False, 0.1),
                                   ("no_images", True, None)):
        rows.append({"event": event, "source": sc.SENSOR, "time": peak,
                     "depth_m": 0.1,
                     "sensor_peak": peak if has_peak else pd.NaT})
        if depth is not None:
            rows.append({"event": event, "source": "VS1", "time": peak,
                         "depth_m": depth or np.nan,
                         "sensor_peak": peak if has_peak else pd.NaT})
    series = pd.DataFrame(rows)
    timing = pd.DataFrame([
        {"event": e, "source": "VS1", "camera_images": 5,
         "camera_first_wet": peak if cam else pd.NaT,
         "sensor_first_wet": peak if sens else pd.NaT,
         "onset_lead_min": 0.0 if cam and sens else np.nan}
        for e, cam, sens in (("both", True, True), ("dry_cam", False, True),
                             ("night", True, False),
                             ("no_peak", True, True))])
    table = pub.event_membership(series, timing, "peak", "zero").set_index(
        "event")["reason"]
    assert table["both"] == "in (a) and (b)-(c)"
    assert table["dry_cam"] == "in (a) only: camera location never wet"
    assert table["night"] == ("in (a) only: sensor not wet during the "
                              "camera's image hours")
    assert table["no_peak"] == ("in (b)-(c) only: no sensor peak in the "
                                "event period")
    assert table["no_images"] == "in neither: no well-segmented images"


def test_methods_figure_assets_and_diagram(tmp_path):
    """Every panel is exported for one image and the SVG links them."""
    import json

    import methods_figure_assets as assets
    import methods_process_diagram as diagram
    from poseidon_utils.zarr_io import write_array, write_zip_group

    stamp = "20240601120000"
    event = tmp_path / "events" / "CB_03_20240601090000_20240601160000"
    (event / "orig_images").mkdir(parents=True)
    (event / "labels").mkdir()
    (event / "zarr").mkdir()
    image = np.full((80, 60, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(event / "orig_images" / f"CAM_CB_03_{stamp}.jpg"), image)
    labels = np.zeros((80, 60), dtype=np.uint8)
    labels[50:, :] = 1  # water in the lower part
    cv2.imwrite(str(event / "labels" / f"CAM_CB_03_{stamp}_predseg.png"),
                labels)
    plan = np.zeros((40, 30), dtype=np.uint8)
    plan[:15] = 1  # water along the southern rows
    with write_zip_group(event / "zarr" / "orig_image_rects.zip") as g:
        write_array(g, f"CAM_CB_03_{stamp}_rectified",
                    np.full((40, 30, 3), 90, dtype=np.uint8))
    with write_zip_group(event / "zarr" / "labels_rects.zip") as g:
        write_array(g, f"CAM_CB_03_{stamp}_predseg_rectified", plan)
    with write_zip_group(event / "zarr" / "depth_maps.zip") as g:
        write_array(g, f"CAM_CB_03_{stamp}_predseg_rectified_depth_map_"
                       "95_perc", np.where(plan == 1, 0.1, 0.0))
    z = np.linspace(0, 1, 40 * 30).reshape(40, 30)
    np.savez(tmp_path / "grid.npz", x=z, y=z, z=z)

    out = tmp_path / "assets"
    assert assets.main([
        "--site", "carolina_beach", "--events_dir", str(tmp_path / "events"),
        "--images_dir", str(tmp_path / "images"), "--labels", "unused",
        "--sensor", "unused", "--grid", str(tmp_path / "grid.npz"),
        "--out_dir", str(out), "--event", event.name, "--timestamp", stamp,
        "--plan_scale", "1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    for name in ("01_original.jpg", "02_overlay.jpg", "03_classes.png",
                 "04_water_mask.png", "06_dem.png", "07_rectified.png",
                 "08_rectified_water.png", "09_depth.png",
                 "09_depth_colorbar.svg", "10_roi_sensors.png"):
        assert (out / name).exists() and name in manifest["files"]
    # Plan-view panels share the grid's size, north up
    assert manifest["pixel_sizes"]["09_depth.png"] == [30, 40]
    rectified_water = cv2.imread(str(out / "08_rectified_water.png"))
    assert rectified_water[-1, 0, 0] > rectified_water[0, 0, 0]  # blue south

    svg = tmp_path / "diagram.svg"
    assert diagram.main(["--site", "carolina_beach", "--assets", str(out),
                         "--out", str(svg)]) == 0
    text = svg.read_text()
    assert 'xlink:href="assets/09_depth.png"' in text
    assert 'inkscape:label="Georect zoom"' in text
    assert "(not found)" not in text  # every panel was exported


def test_site_description_figure_renders_offline(tmp_path):
    """The figure renders from local layers, grids and photos only."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    import site_description_figure as sdf
    from poseidon_utils.site_config import load_site_config

    layers = tmp_path / "layers"
    layers.mkdir()
    x0, x1, y0, y1 = sdf.web_box(sdf.NC_EXTENT)
    gpd.GeoDataFrame({"STUSPS": ["NC"], "NAME": ["North Carolina"]},
                     geometry=[box(x0, y0, (x0 + x1) / 2, y1)],
                     crs=sdf.WEB).to_file(layers / "states.gpkg")

    def raster(path, bounds):
        with rasterio.open(path, "w", driver="GTiff", width=20, height=20,
                           count=3, dtype="uint8", crs=sdf.WEB,
                           transform=from_bounds(bounds[0], bounds[2],
                                                 bounds[1], bounds[3], 20,
                                                 20)) as dst:
            dst.write(np.full((3, 20, 20), 180, dtype=np.uint8))

    raster(layers / "context_nc.tif", (x0, x1, y0, y1))
    grids = tmp_path / "grids"
    grids.mkdir()
    for name in ("carolina_beach", "down_east"):
        site = load_site_config(name)
        raster(layers / f"community_{name}.tif",
               sdf.web_box(sdf.community_extent(site, 2.0)))
        g = site.grid
        x, y = np.meshgrid(np.arange(g.min_x, g.max_x, 0.5),
                           np.arange(g.min_y, g.max_y, 0.5))
        np.savez(grids / f"shared_grid_{name}.npz", x=x, y=y,
                 z=0.5 + 0.01 * (x - g.min_x))
    photo = tmp_path / "photo.jpg"
    cv2.imwrite(str(photo), np.full((40, 30, 3), 128, dtype=np.uint8))

    out = tmp_path / "out"
    assert sdf.main(["render", "--sites", "carolina_beach", "down_east",
                     "--layers_dir", str(layers), "--grid_dir", str(grids),
                     "--photos", str(photo), str(photo),
                     "--out_dir", str(out)]) == 0
    for ext in ("png", "pdf", "svg"):
        assert (out / f"site_description.{ext}").exists()
    assert "Esri" in (out / "attribution.txt").read_text()


def test_site_figure_download_falls_back_to_coarser_zoom(tmp_path,
                                                          monkeypatch):
    """A failing zoom level is retried coarser; no partial file is left."""
    import argparse

    import contextily
    import requests

    import site_description_figure as sdf

    tried = []

    def fake(w, s, e, n, path, ll, zoom, source, wait, max_retries):
        tried.append(zoom)
        if zoom == 16:
            open(path, "w").close()  # a partial write before the error
            raise requests.HTTPError("502 Bad Gateway")
        with open(path, "w") as handle:
            handle.write("tiles")

    monkeypatch.setattr(contextily, "bounds2raster", fake)
    out = tmp_path / "layer.tif"
    sdf.download_tiles((-77, -76, 34, 35), str(out), None, 16,
                       argparse.Namespace(wait=0, retries=0), "test")
    assert tried == [16, 15] and out.read_text() == "tiles"
    assert not (tmp_path / "layer.tif.partial.tif").exists()


def test_site_summary_table_from_events(tmp_path):
    """Sensor and image rows, with capped intervals and every output."""
    import site_summary_table as table

    for name in ("carolina_beach", "down_east"):
        _write_publication_site(tmp_path / name)
        # Two wet readings 1 min apart, then one after a 3 h gap
        times = pd.to_datetime(["2024-06-01 12:00", "2024-06-01 12:01",
                                "2024-06-01 15:01"], utc=True)
        pd.DataFrame({"flood_event": 1, "time_UTC": times.astype(str),
                      "road_water_level": [0.1, 0.5, 0.3]}).to_csv(
            tmp_path / name / "flood_events.csv", index=False)
    out = tmp_path / "out"
    assert table.main(["--sites", "carolina_beach", "down_east",
                       "--data_root", str(tmp_path), "--out_dir", str(out),
                       "--rois", "road", "road"]) == 0
    for stem in ("site_summary_table", "site_summary_table_good_only"):
        for ext in ("png", "pdf", "svg", "csv", "tex", "md"):
            assert (out / f"{stem}.{ext}").exists()
    values = pd.read_csv(out / "site_summary_table.csv",
                         index_col="key")["carolina_beach"]
    assert values["sensor_max_depth_m"] == pytest.approx(0.5 * 0.3048)
    # 1 + 1 + 12 (capped) minutes of wet readings
    assert values["sensor_total_hours"] == pytest.approx(14 / 60)
    # Four events with water; of 16 images 6 min apart only the first
    # is dry (sin(pi) rounds to a tiny positive depth)
    assert values["image_events"] == 4
    # No event folder has a sensor CSV, so every event is image-only
    assert values["image_events_no_sensor"] == 4
    assert values["image_total_hours"] == pytest.approx(4 * 15 * 0.1)
    assert values["image_max_depth_m"] == pytest.approx(0.15, abs=0.002)
    assert "\\toprule" in (out / "site_summary_table.tex").read_text()
