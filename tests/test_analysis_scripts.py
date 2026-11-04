"""Tests for the analysis scripts and the lock-file converter."""

import cv2
import find_stat_anomalies
import image_stitcher
import numpy as np
import pandas as pd
import parse_lock
import pytest
import run_road_access_stats

import access_plotter


def test_logit_inverts_the_logistic_curve():
    """Depth -> coverage -> depth round-trips."""
    depth = np.linspace(0.0, 0.3, 7)
    params = (80.0, 40.0, 0.1)
    coverage = access_plotter.logistic_curve(depth, *params)
    np.testing.assert_allclose(
        access_plotter.logit_curve(coverage, *params), depth, atol=1e-6)


def test_fit_recovers_logistic_parameters():
    """The logistic fit reproduces noise-free synthetic data."""
    depth = pd.Series(np.linspace(0.0, 0.3, 40))
    coverage = pd.Series(access_plotter.logistic_curve(depth, 90, 50, 0.12))
    fit = access_plotter.fit_logistic_with_quantiles(depth, coverage)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)
    assert fit["rmse"] < 1e-3
    assert set(fit["quantiles"]) <= {0.1, 0.5, 0.9}


def test_pinball_loss_is_smallest_at_the_quantile():
    """Shifting a perfect fit up or down increases the loss."""
    x = np.linspace(0, 0.3, 50)
    y = access_plotter.logistic_curve(x, 90, 50, 0.12)
    best = access_plotter.pinball_loss((90, 50, 0.12), x, y, 0.5)
    assert best == pytest.approx(0.0)
    assert access_plotter.pinball_loss((95, 50, 0.12), x, y, 0.5) > best


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
                         "--out_dir", str(tmp_path / "plots")])
    plots = {p.name for p in (tmp_path / "plots").iterdir()}
    for metric in access_plotter.METRICS:
        for suffix in ("1_evolution_plot.jpg", "1b_reversed_fit.jpg",
                       "2_summary_scatter.png", "3_depth_cdf.png",
                       "4_depth_pdf.png", "5_depth_exceedance.png"):
            assert f"road_{metric}_{suffix}" in plots


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
