"""Plot roadway accessibility statistics across all flood events.

Reads every ``<roi>_accessibility_time_series.csv`` (and the matching
``<roi>_wse_time_series.csv``) under ``<events_dir>/*/`` and, for each
ROI and depth metric, saves:

1.  Coverage vs depth scatter (``_1_evolution_plot.jpg``).
1b. Depth by coverage bin (all images): ``_1b_binned_depth.jpg`` and
    the lookup table ``_1b_depth_by_coverage.csv``; and a logistic fit
    with median and 80 % band (``_1b_reversed_fit.jpg``).
1c. As 1b using only well-segmented images (needs ``--labels``):
    ``_1c_binned_depth_good_only.jpg``, ``_1c_depth_by_coverage.csv``
    and ``_1c_reversed_fit_good_only.jpg``.
2.  Peak coverage vs peak depth per event.
3.  CDF, 4. PDF, 5. exceedance probability of peak depth per event.
6.  Image vs in-situ sensor depth, 1:1 (needs ``--labels`` and
    ``--sensor``).
7.  Image vs in-situ sensor WSE, 1:1 (same requirements).
9.  1c and 8 on one figure: per coverage bin, the image-derived depth
    (blue) beside the sensor depth (orange) for the same images
    (``_9_image_vs_sensor_binned_good_only.jpg`` and
    ``_9_depth_by_coverage.csv``; needs ``--labels`` and ``--sensor``).
8.  As 1c, but with the in-situ sensor's depth for each well-segmented
    image instead of the image-derived depth (``<roi>_SensorDepth_8_*``;
    once per ROI, as it does not depend on the image depth metric; same
    requirements as 6).

Depth by coverage (1b, 1c; recommended for prediction)
------------------------------------------------------
Images are grouped by roadway coverage into bins of
``--coverage_bin_width`` percent below ``--full_coverage`` (left
closed, e.g. [0, 10), ..., [90, 99)), plus one bin for a fully covered
road (coverage >= ``--full_coverage``). Once the road is fully covered,
coverage no longer limits depth, so that bin is reported and drawn on
its own rather than as part of a curve. For each bin the table gives
the image and event counts; mean and median depth with 95 % confidence
intervals from resampling whole events (``--n_boot`` times; images of
one event are not independent); the 10th and 90th percentiles (the
range of 80 % of observed depths: what to expect at that coverage);
and the minimum and maximum. Percentiles use linear interpolation
between order statistics (numpy's default). Bins with fewer than
:data:`MIN_BIN_IMAGES` images or :data:`MIN_BIN_EVENTS` events are
flagged ``low_sample`` and drawn hollow. These statistics assume no
curve shape, so they hold by construction.

Logistic fits (1b, 1c; a smooth summary)
----------------------------------------
Depth is modeled as the inverse of a logistic coverage curve,
``depth = x0 + ln(c / (L - c)) / k`` (coverage ``c`` in %, ``L`` the
coverage asymptote, ``k`` the steepness, ``x0`` the depth at half of
``L``), and fitted **in depth**:

* The fit line minimizes squared depth errors, so it estimates the mean
  depth at a given coverage. Its R^2, RMSE, and MAE are in depth (m).
* The 0.1, 0.5 and 0.9 quantile curves minimize the pinball loss of the
  depth errors, so the dashed curve is the median depth at a given
  coverage and the band holds the middle 80 % of depths. Curves that
  cross are made monotone by sorting at each coverage (Chernozhukov,
  Fernandez-Val and Galichon, 2010, Econometrica 78:1093).
* The share of images inside the band and below the median, and the
  mean depth error of the fit line, are checked overall and in coverage
  bins (``*_band_calibration.csv``). Departures there mean the logistic
  shape does not describe the data; use the binned statistics instead.
* Images of one event are not independent, so uncertainty comes from
  resampling whole events with replacement (``--n_boot`` times): the
  95 % confidence band of the fit line and of each parameter.

Images with zero coverage are left out of the fits (depth is undefined
there). Every fitted number is written to ``*_fit_summary.csv``.

Event plots (2-5)
-----------------
Each event contributes its peak coverage and peak depth. With
``--labels`` the peaks come from well-segmented images only, and events
without one are left out; otherwise all images are used. Plot 3 is the
empirical CDF ``P(X <= x)`` and plot 5 the exceedance probability
``P(X >= x)``, with empirical (i/n, default) or Weibull (i/(n+1))
plotting positions (``--plotting_position``). Plot 4 is a density
histogram with Freedman-Diaconis bins and a Gaussian kernel density
estimate reflected at 0 m (depths cannot be negative).

Example
-------
``python access_plotter.py --events_dir flood_events --labels
segmentation_labels.csv --sensor sensor_data.csv --sensor_wse_floor
0.92``
"""

import argparse
import glob
import logging
import os
import warnings
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.optimize import least_squares, minimize  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from poseidon_utils.constants import FEET_TO_METERS  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.naming import extract_timestamp  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402

from plot_style import (  # noqa: E402
    BLUE,
    CONTEXT_GREY,
    INK,
    ORANGE,
    SEQUENTIAL_BLUE,
    apply_style,
)

logger = logging.getLogger(__name__)

#: Depth metrics plotted from the accessibility CSVs.
METRICS = ["MaxDepth", "MeanDepth", "MedianDepth"]

#: Human-readable metric names for axis labels.
PRETTY_METRICS = {"MaxDepth": "Maximum Depth", "MeanDepth": "Mean Depth",
                  "MedianDepth": "Median Depth"}

#: Quantiles fitted for the prediction band and median line.
QUANTILES = (0.1, 0.5, 0.9)

#: Default coverage (%) at or above which the road counts as fully
#: covered.
FULL_COVERAGE = 99.0

#: Default coverage bin width (%) below full coverage.
COVERAGE_BIN_WIDTH = 10.0

#: Bins with fewer images or events than these are flagged low_sample.
MIN_BIN_IMAGES = 10
MIN_BIN_EVENTS = 3

#: Order of plots in the run summary.
PLOT_ORDER = ("1", "1b binned", "1b logistic", "1c binned", "1c logistic",
              "1c", "2", "3", "4", "5", "6", "7", "8 binned", "8 logistic",
              "8", "9")

#: Name used for the sensor depth in plot 8 filenames and the summary.
SENSOR_DEPTH = "SensorDepth"

#: Display colours of the segmentation classes (see plot_style).
SEGMENTATION_COLORS = {"good": BLUE, "other": CONTEXT_GREY}

#: Legend proxies for good and other segmentations.
SEGMENTATION_HANDLES = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color,
           markeredgecolor="none", markersize=7)
    for color in (SEGMENTATION_COLORS["good"], SEGMENTATION_COLORS["other"])
]


# ----- Models and metrics ----------------------------------------------------

def logistic_curve(x, L, k, x0):  # noqa: N803 - conventional symbols
    """
    Evaluate a logistic curve (coverage as a function of depth).

    Parameters
    ----------
    x : array_like
        Depth values.
    L : float
        Upper asymptote (maximum coverage).
    k : float
        Steepness.
    x0 : float
        Midpoint depth.

    Returns
    -------
    numpy.ndarray
        Predicted coverage.
    """
    exponent = np.clip(-k * (x - x0), -100, 100)  # avoid overflow
    return L / (1 + np.exp(exponent))


def depth_from_coverage(c, L, k, x0):  # noqa: N803 - conventional symbols
    """
    Invert :func:`logistic_curve`: the depth at which coverage is `c`.

    This is the curve fitted in plots 1b and 1c.

    Parameters
    ----------
    c : array_like
        Coverage values, ``0 < c < L``.
    L, k, x0 : float
        Logistic parameters.

    Returns
    -------
    numpy.ndarray
        Depth.
    """
    c = np.asarray(c, dtype=float)
    return x0 + np.log(c / (L - c)) / k


def pinball_loss(residuals, q):
    """
    Summed quantile (pinball) loss.

    Minimizing it over a curve gives the curve below which a fraction
    `q` of the observations lie.

    Parameters
    ----------
    residuals : array_like
        Observed minus predicted values.
    q : float
        Quantile in (0, 1).

    Returns
    -------
    float
        Summed loss.
    """
    r = np.asarray(residuals, dtype=float)
    return float(np.sum(np.maximum(q * r, (q - 1) * r)))


def calculate_r2(y_true, y_pred):
    """
    Coefficient of determination.

    Parameters
    ----------
    y_true, y_pred : array_like
        Observed and predicted values.

    Returns
    -------
    float
        R-squared, or 0.0 if `y_true` is constant.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


def calculate_rmse(y_true, y_pred):
    """
    Root mean square error.

    Parameters
    ----------
    y_true, y_pred : array_like
        Observed and predicted values.

    Returns
    -------
    float
        RMSE.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def empirical_probabilities(values, plotting_position="empirical"):
    """
    CDF and exceedance probability at each distinct value.

    Parameters
    ----------
    values : array_like
        Sample (NaNs not allowed).
    plotting_position : {'empirical', 'weibull'}, optional
        ``'empirical'`` divides counts by ``n`` (the empirical
        distribution); ``'weibull'`` divides by ``n + 1``, as is common
        for flood frequency, so no value gets probability 0 or 1.

    Returns
    -------
    x : numpy.ndarray
        Sorted distinct values.
    cdf : numpy.ndarray
        ``P(X <= x)``: the share of the sample at or below each value.
    exceedance : numpy.ndarray
        ``P(X >= x)``: the share at or above each value.

    Raises
    ------
    ValueError
        If `plotting_position` is unknown.
    """
    x, counts = np.unique(np.asarray(values, dtype=float),
                          return_counts=True)
    n = counts.sum()
    if plotting_position == "empirical":
        denominator = n
    elif plotting_position == "weibull":
        denominator = n + 1
    else:
        raise ValueError(f"Unknown plotting_position '{plotting_position}'")
    at_or_below = np.cumsum(counts)
    at_or_above = n - at_or_below + counts
    return x, at_or_below / denominator, at_or_above / denominator


# ----- Inputs ----------------------------------------------------------------

def as_utc_ns(times):
    """
    Convert times to UTC at nanosecond resolution.

    pandas keeps the resolution it parses (s, ms, us or ns) and
    ``merge_asof`` refuses keys of different resolutions, so every time
    key is converted to one resolution before matching.

    Parameters
    ----------
    times : array_like
        Times (strings, datetimes, or a datetime Series).

    Returns
    -------
    pandas.Series or pandas.DatetimeIndex
        ``datetime64[ns, UTC]`` times.
    """
    return pd.to_datetime(times, utc=True).astype("datetime64[ns, UTC]")


def load_good_timestamps(labels_csv):
    """
    Read timestamps of images scored as well segmented.

    Parameters
    ----------
    labels_csv : str or None
        CSV with ``Filename`` and ``Segmentation_Score`` (1 = good).

    Returns
    -------
    set of str
        ``YYYYMMDDHHMMSS`` timestamps; empty if no file is given.
    """
    if not labels_csv or not os.path.exists(labels_csv):
        logger.info("No segmentation labels%s; using default colors and "
                    "skipping the good-segmentation plots.",
                    f" at {labels_csv}" if labels_csv else "")
        return set()
    labels = pd.read_csv(labels_csv)
    good = labels.loc[labels["Segmentation_Score"] == 1, "Filename"]
    stamps = {extract_timestamp(name) for name in good.dropna()}
    stamps.discard(None)
    logger.info("Found %d well-segmented timestamps.", len(stamps))
    return stamps


def load_sensor_data(sensor_csv):
    """
    Read in-situ sensor levels for comparison, converted to meters.

    Parameters
    ----------
    sensor_csv : str or None
        SDFP CSV with ``date``, ``road_water_level_adj`` and optionally
        ``sensor_water_level_adj`` (feet), or a detailed event CSV from
        ``EventTracker`` / ``run_pull_event_series.py`` with
        ``time_UTC``, ``road_water_level`` and ``sensor_water_level``.

    Returns
    -------
    pandas.DataFrame or None
        ``datetime_utc``, ``SensorDepth_m`` (negative depths set to 0)
        and, if available, ``SensorWSE_m``, sorted by time.
    """
    if not sensor_csv or not os.path.exists(sensor_csv):
        logger.info("No sensor CSV%s; sensor comparison plots skipped.",
                    f" at {sensor_csv}" if sensor_csv else "")
        return None
    sensor = pd.read_csv(sensor_csv).rename(columns={
        "time_UTC": "date",
        "road_water_level": "road_water_level_adj",
        "sensor_water_level": "sensor_water_level_adj",
    })
    if "road_water_level_adj" not in sensor:
        logger.warning("'road_water_level_adj' not in %s", sensor_csv)
        return None

    # Cameras only see water above the road, so clamp depths at zero
    sensor["SensorDepth_m"] = (sensor["road_water_level_adj"]
                               * FEET_TO_METERS).clip(lower=0.0)
    if "sensor_water_level_adj" in sensor:
        sensor["SensorWSE_m"] = (sensor["sensor_water_level_adj"]
                                 * FEET_TO_METERS)
    sensor["datetime_utc"] = pd.to_datetime(sensor["date"], errors="coerce",
                                            utc=True)
    sensor = sensor.dropna(subset=["datetime_utc", "SensorDepth_m"])
    # Padded event windows can overlap and repeat a reading
    sensor = sensor.drop_duplicates(subset="datetime_utc")
    logger.info("Loaded %d sensor readings.", len(sensor))
    return sensor.sort_values("datetime_utc")


def find_roi_csvs(events_dir):
    """
    Group accessibility CSVs under an events folder by ROI name.

    Parameters
    ----------
    events_dir : str
        Folder of event subfolders.

    Returns
    -------
    dict of str to list of str
        ROI name to its accessibility CSV paths.
    """
    suffix = "_accessibility_time_series.csv"
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(events_dir, "*",
                                              f"*{suffix}"))):
        groups[os.path.basename(path).replace(suffix, "")].append(path)
    return dict(groups)


def load_event(path, good_timestamps, sensor):
    """
    Load one event's accessibility table with coverage and colors.

    Parameters
    ----------
    path : str
        Accessibility CSV path.
    good_timestamps : set of str
        Well-segmented image timestamps.
    sensor : pandas.DataFrame or None
        Sensor readings from :func:`load_sensor_data`.

    Returns
    -------
    pandas.DataFrame
        The table with ``Coverage`` (%), WSE columns when available,
        ``Segmentation`` (``"good"`` or ``"other"``) and, when labels
        are given, the nearest sensor reading within 5 minutes.
    """
    event = pd.read_csv(path)
    event["event"] = os.path.basename(os.path.dirname(path))
    wse_path = path.replace("_accessibility_time_series.csv",
                            "_wse_time_series.csv")
    if os.path.exists(wse_path):
        wse = pd.read_csv(wse_path)
        event = event.merge(wse[["Time", "MeanWSE", "MaxWSE", "MedianWSE"]],
                            on="Time", how="left")

    if {"FloodedPixels", "TotalPixels"} <= set(event.columns):
        event["Coverage"] = event["FloodedPixels"] / event["TotalPixels"] * 100
    else:
        event["Coverage"] = np.nan

    if not good_timestamps or "Time" not in event:
        event["Segmentation"] = "other"
        event["SensorDepth_m"] = np.nan
        return event

    event["datetime_utc"] = pd.to_datetime(event["Time"], errors="coerce",
                                           utc=True)
    stamps = event["datetime_utc"].dt.strftime("%Y%m%d%H%M%S")
    event["Segmentation"] = np.where(stamps.isin(good_timestamps), "good",
                                     "other")
    if sensor is not None:
        columns = [c for c in ("datetime_utc", "SensorDepth_m",
                               "SensorWSE_m") if c in sensor]
        event["datetime_utc"] = as_utc_ns(event["datetime_utc"])
        sensor = sensor[columns].assign(
            datetime_utc=as_utc_ns(sensor["datetime_utc"]))
        event = pd.merge_asof(event.sort_values("datetime_utc"),
                              sensor, on="datetime_utc",
                              direction="nearest",
                              tolerance=pd.Timedelta(minutes=5))
    return event


def summarize_event(event, good_only=False):
    """
    Return an event's peak coverage and peak depth metrics.

    Parameters
    ----------
    event : pandas.DataFrame
        Output of :func:`load_event`.
    good_only : bool, optional
        Use only well-segmented images. Default False.

    Returns
    -------
    dict
        ``Coverage_max`` and ``<metric>_max`` for each metric present
        (NaN if no image qualifies), and ``images`` used.
    """
    rows = event[event["Segmentation"] == "good"] if good_only else event
    summary = {"Coverage_max": rows["Coverage"].max(), "images": len(rows)}
    for metric in METRICS:
        if metric in rows:
            summary[f"{metric}_max"] = rows[metric].max()
    return summary


# ----- Plots -----------------------------------------------------------------

def _save(fig, out_dir, filename, tight=True):
    """
    Save and close a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    out_dir : str
        Output folder.
    filename : str
        Output filename.
    tight : bool, optional
        Apply ``tight_layout`` first. Default True; pass False for
        layouts with fixed panel spacing.
    """
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


#: Coverage bins (%) used to check the prediction band.
CALIBRATION_BINS = (0, 25, 50, 75, 100)


def _unpack(theta, c_upper):
    """
    Map unconstrained parameters to ``(L, k, x0)``.

    Parameters
    ----------
    theta : array_like
        ``(ln(L - c_upper), ln k, x0)``.
    c_upper : float
        Largest coverage the curve must cover.

    Returns
    -------
    tuple of float
        ``(L, k, x0)`` with ``L > c_upper`` and ``k > 0``.
    """
    return c_upper + np.exp(theta[0]), np.exp(theta[1]), theta[2]


def _initial_thetas(coverage, depth, c_upper):
    """
    Starting parameters from linear fits of depth on ``ln(c/(L-c))``.

    For a fixed ``L`` the model is linear in ``ln(c / (L - c))``, so
    several trial asymptotes give closed-form starting points.

    Parameters
    ----------
    coverage, depth : numpy.ndarray
        Paired observations (coverage > 0).
    c_upper : float
        Largest coverage.

    Returns
    -------
    list of numpy.ndarray
        Unconstrained starting parameters.

    Raises
    ------
    ValueError
        If depth does not increase with coverage for any trial ``L``.
    """
    starts = []
    for gap in (0.5, 2.0, 10.0, 50.0):
        asymptote = c_upper + gap
        z = np.log(coverage / (asymptote - coverage))
        slope, intercept = np.polyfit(z, depth, 1)
        if slope > 0:
            starts.append(np.array([np.log(gap), -np.log(slope), intercept]))
    if not starts:
        raise ValueError("Depth does not increase with coverage; no "
                         "logistic fit.")
    return starts


def _fit_central(coverage, depth, c_upper, starts):
    """
    Least-squares fit of depth on coverage.

    Parameters
    ----------
    coverage, depth : numpy.ndarray
        Paired observations.
    c_upper : float
        Largest coverage the curve must cover.
    starts : list of numpy.ndarray
        Unconstrained starting parameters.

    Returns
    -------
    numpy.ndarray
        Best unconstrained parameters.

    Raises
    ------
    RuntimeError
        If no start converges.
    """
    def residuals(theta):
        return depth - depth_from_coverage(coverage,
                                           *_unpack(theta, c_upper))

    best = None
    for theta0 in starts:
        res = least_squares(residuals, theta0)
        if res.success and (best is None or res.cost < best.cost):
            best = res
    if best is None:
        raise RuntimeError("Least-squares fit did not converge.")
    return best.x


def _fit_quantile(coverage, depth, c_upper, q, theta0):
    """
    Fit one quantile curve of depth given coverage.

    The pinball loss is not smooth, so two derivative-free methods
    are run and the lower loss is kept.

    Parameters
    ----------
    coverage, depth : numpy.ndarray
        Paired observations.
    c_upper : float
        Largest coverage the curve must cover.
    q : float
        Quantile.
    theta0 : numpy.ndarray
        Starting parameters (the least-squares fit).

    Returns
    -------
    numpy.ndarray or None
        Unconstrained parameters, or None if both methods fail.
    """
    def loss(theta):
        return pinball_loss(
            depth - depth_from_coverage(coverage, *_unpack(theta, c_upper)),
            q)

    runs = [minimize(loss, theta0, method="Nelder-Mead",
                     options={"maxiter": 20000, "xatol": 1e-9,
                              "fatol": 1e-12}),
            minimize(loss, theta0, method="Powell",
                     options={"maxiter": 20000, "xtol": 1e-9,
                              "ftol": 1e-12})]
    converged = [run for run in runs if run.success]
    if not converged:
        return None
    return min(converged, key=lambda run: run.fun).x


def _bootstrap_fit(coverage, depth, events, c_upper, theta, grid, n_boot,
                   seed):
    """
    Resample whole events to get confidence bands of the fit.

    Parameters
    ----------
    coverage, depth : numpy.ndarray
        Paired observations.
    events : numpy.ndarray
        Event of each observation.
    c_upper : float
        Largest coverage of the full data; every resampled curve must
        cover it.
    theta : numpy.ndarray
        Full-data parameters, used as the starting point.
    grid : numpy.ndarray
        Coverages at which curves are evaluated.
    n_boot : int
        Number of resamples.
    seed : int
        Random seed (results are reproducible).

    Returns
    -------
    dict
        ``curve_ci`` (2.5 and 97.5 percentile curves), ``param_ci``
        (``{name: (low, high)}``), ``n_boot`` and ``failures``.
    """
    rng = np.random.default_rng(seed)
    index = {event: np.flatnonzero(events == event)
             for event in np.unique(events)}
    names = list(index)
    curves, params, failures = [], [], 0
    for _ in range(n_boot):
        drawn = rng.choice(len(names), size=len(names), replace=True)
        rows = np.concatenate([index[names[i]] for i in drawn])
        try:
            theta_b = _fit_central(coverage[rows], depth[rows], c_upper,
                                   [theta])
        except (RuntimeError, ValueError):
            failures += 1
            continue
        fitted = _unpack(theta_b, c_upper)
        params.append(fitted)
        curves.append(depth_from_coverage(grid, *fitted))
    if not curves:
        return {"curve_ci": None, "param_ci": None, "n_boot": n_boot,
                "failures": failures}
    curves, params = np.array(curves), np.array(params)
    return {
        "curve_ci": np.percentile(curves, [2.5, 97.5], axis=0),
        "param_ci": {name: tuple(np.percentile(params[:, i], [2.5, 97.5]))
                     for i, name in enumerate(("L", "k", "x0"))},
        "n_boot": n_boot,
        "failures": failures,
    }


def fit_depth_from_coverage(coverage, depth, events=None, n_boot=0, seed=0):
    """
    Fit depth as a logistic function of coverage, with quantile curves.

    See the module docstring for the model and the meaning of each
    curve.

    Parameters
    ----------
    coverage, depth : array_like
        Paired per-image coverage (%) and depth (m).
    events : array_like, optional
        Event of each image, for the event bootstrap.
    n_boot : int, optional
        Number of event resamples; 0 skips the bootstrap. Default 0.
    seed : int, optional
        Bootstrap random seed. Default 0.

    Returns
    -------
    dict
        ``c`` (200 coverages), ``fit`` (fitted depth), ``params``
        (``L``, ``k``, ``x0``), ``r2``, ``rmse`` and ``mae`` (depth, m),
        ``n``, ``n_events``, ``n_excluded`` (zero-coverage images),
        ``quantiles`` (``{q: curve}``), ``quantile_params``,
        ``failed_quantiles``, ``rearranged`` (curves crossed and were
        sorted), ``band_fraction`` and ``below_median_fraction`` (share
        of images inside the 0.1-0.9 band and at or below the median),
        ``calibration`` (the same by coverage bin), and the bootstrap
        results ``curve_ci``, ``param_ci``, ``n_boot`` and
        ``boot_failures``.

    Raises
    ------
    ValueError
        If fewer than four usable points remain or depth does not
        increase with coverage.
    RuntimeError
        If the least-squares fit does not converge.
    """
    c = np.asarray(coverage, dtype=float)
    d = np.asarray(depth, dtype=float)
    keep = np.isfinite(c) & np.isfinite(d) & (c > 0)
    c, d = c[keep], d[keep]
    ev = None if events is None else np.asarray(events)[keep]
    if len(c) < 4:
        raise ValueError(f"Only {len(c)} usable points; need at least 4.")

    c_upper = float(c.max())
    theta = _fit_central(c, d, c_upper, _initial_thetas(c, d, c_upper))
    params = _unpack(theta, c_upper)
    grid = np.linspace(c.min(), c_upper, 200)
    predicted = depth_from_coverage(c, *params)

    result = {
        "c": grid,
        "fit": depth_from_coverage(grid, *params),
        "params": dict(zip(("L", "k", "x0"), params)),
        "r2": calculate_r2(d, predicted),
        "rmse": calculate_rmse(d, predicted),
        "mae": float(np.mean(np.abs(d - predicted))),
        "n": len(c),
        "n_events": None if ev is None else len(np.unique(ev)),
        "n_excluded": int((~keep).sum()),
        "quantiles": {}, "quantile_params": {}, "failed_quantiles": [],
        "rearranged": False, "band_fraction": np.nan,
        "below_median_fraction": np.nan, "calibration": None,
        "curve_ci": None, "param_ci": None, "n_boot": 0,
        "boot_failures": 0,
    }

    at_points = {}
    for q in QUANTILES:
        theta_q = _fit_quantile(c, d, c_upper, q, theta)
        if theta_q is None:
            logger.warning("Quantile %.1f fit did not converge; curve "
                           "omitted.", q)
            result["failed_quantiles"].append(q)
            continue
        params_q = _unpack(theta_q, c_upper)
        result["quantile_params"][q] = dict(zip(("L", "k", "x0"), params_q))
        result["quantiles"][q] = depth_from_coverage(grid, *params_q)
        at_points[q] = depth_from_coverage(c, *params_q)

    fitted = [q for q in QUANTILES if q in at_points]
    if len(fitted) > 1:
        on_grid = np.array([result["quantiles"][q] for q in fitted])
        if np.any(np.diff(on_grid, axis=0) < 0):
            logger.warning("Quantile curves cross; sorting them at each "
                           "coverage (monotone rearrangement).")
            result["rearranged"] = True
            on_grid = np.sort(on_grid, axis=0)
            points = np.sort([at_points[q] for q in fitted], axis=0)
            for i, q in enumerate(fitted):
                result["quantiles"][q] = on_grid[i]
                at_points[q] = points[i]

    if {0.1, 0.9} <= set(at_points) and 0.5 in at_points:
        inside = (d >= at_points[0.1]) & (d <= at_points[0.9])
        below = d <= at_points[0.5]
        result["band_fraction"] = float(inside.mean())
        result["below_median_fraction"] = float(below.mean())
        bins = pd.cut(c, CALIBRATION_BINS, include_lowest=True)
        table = pd.DataFrame({"bin": bins, "inside": inside,
                              "below": below, "error": d - predicted})
        grouped = table.groupby("bin", observed=False)
        result["calibration"] = pd.DataFrame({
            "coverage_bin_pct": [str(b) for b in grouped.size().index],
            "images": grouped.size().to_numpy(),
            "fraction_inside_80pct_band": grouped["inside"].mean().to_numpy(),
            "fraction_at_or_below_median": grouped["below"].mean().to_numpy(),
            "mean_error_of_fit_m": grouped["error"].mean().to_numpy(),
        })

    if ev is not None and n_boot > 0 and len(np.unique(ev)) > 1:
        boot = _bootstrap_fit(c, d, ev, c_upper, theta, grid, n_boot, seed)
        result.update(curve_ci=boot["curve_ci"], param_ci=boot["param_ci"],
                      n_boot=boot["n_boot"], boot_failures=boot["failures"])
        if boot["failures"]:
            logger.warning("%d of %d bootstrap fits failed.",
                           boot["failures"], n_boot)
    return result


def write_fit_summary(fit, out_dir, stem):
    """
    Save a fit's numbers and band calibration as CSVs.

    Parameters
    ----------
    fit : dict
        Output of :func:`fit_depth_from_coverage`.
    out_dir : str
        Output folder.
    stem : str
        Filename prefix; writes ``<stem>_fit_summary.csv`` and
        ``<stem>_band_calibration.csv``.
    """
    rows = [("images_fitted", fit["n"]),
            ("events", fit["n_events"]),
            ("images_excluded_zero_coverage", fit["n_excluded"])]
    for name, value in fit["params"].items():
        low, high = (fit["param_ci"] or {}).get(name, (np.nan, np.nan))
        rows += [(name, value), (f"{name}_ci95_low", low),
                 (f"{name}_ci95_high", high)]
    rows += [("r2_depth", fit["r2"]), ("rmse_depth_m", fit["rmse"]),
             ("mae_depth_m", fit["mae"]),
             ("fraction_inside_80pct_band", fit["band_fraction"]),
             ("fraction_at_or_below_median", fit["below_median_fraction"]),
             ("quantile_curves_rearranged", fit["rearranged"]),
             ("failed_quantiles", ";".join(map(str, fit["failed_quantiles"]))
              or "none"),
             ("bootstrap_resamples", fit["n_boot"]),
             ("bootstrap_failures", fit["boot_failures"])]
    for q, params in fit["quantile_params"].items():
        rows += [(f"q{q}_{name}", value) for name, value in params.items()]
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(
        os.path.join(out_dir, f"{stem}_fit_summary.csv"), index=False)
    if fit["calibration"] is not None:
        fit["calibration"].to_csv(
            os.path.join(out_dir, f"{stem}_band_calibration.csv"),
            index=False)


def coverage_bins(bin_width=COVERAGE_BIN_WIDTH, full_coverage=FULL_COVERAGE):
    """
    Edges of the partial-coverage bins.

    Parameters
    ----------
    bin_width : float, optional
        Bin width (%). Default :data:`COVERAGE_BIN_WIDTH`.
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered; the
        last partial bin ends there. Default :data:`FULL_COVERAGE`.

    Returns
    -------
    numpy.ndarray
        Edges ``0, w, 2w, ..., full_coverage``; bins are ``[low, high)``.
    """
    edges = np.arange(0.0, full_coverage, bin_width)
    return np.append(edges, full_coverage)


def depth_by_coverage(coverage, depth, events, bin_width=COVERAGE_BIN_WIDTH,
                      full_coverage=FULL_COVERAGE, n_boot=1000, seed=0):
    """
    Summarize depth in coverage bins, with a separate full-coverage bin.

    See the module docstring for the definitions.

    Parameters
    ----------
    coverage, depth : array_like
        Paired per-image coverage (%) and depth (m).
    events : array_like
        Event of each image, for the event bootstrap.
    bin_width : float, optional
        Partial-coverage bin width (%).
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered.
    n_boot : int, optional
        Event resamples for the confidence intervals; 0 skips them.
    seed : int, optional
        Bootstrap random seed.

    Returns
    -------
    pandas.DataFrame
        One row per bin with at least one image: ``bin``,
        ``coverage_low``, ``coverage_high``, ``full_coverage``,
        ``images``, ``events``, ``mean_depth_m`` and
        ``median_depth_m`` with ``_ci95_low`` / ``_ci95_high``,
        ``p10_depth_m``, ``p90_depth_m``, ``min_depth_m``,
        ``max_depth_m``, ``low_sample``.
    """
    c = np.asarray(coverage, dtype=float)
    d = np.asarray(depth, dtype=float)
    ev = np.asarray(events)
    keep = np.isfinite(c) & np.isfinite(d) & (c > 0)
    c, d, ev = c[keep], d[keep], ev[keep]

    edges = coverage_bins(bin_width, full_coverage)
    n_partial = len(edges) - 1
    # Bin index: 0..n_partial-1 for [low, high), n_partial for full
    bin_of = np.where(c >= full_coverage, n_partial,
                      np.clip(np.searchsorted(edges, c, side="right") - 1,
                              0, n_partial - 1))
    labels = [f"{lo:g}-<{hi:g}%" for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f">={full_coverage:g}% (full)")
    lows = list(edges[:-1]) + [full_coverage]
    highs = list(edges[1:]) + [100.0]

    def bin_stats(bins, values):
        means = np.full(n_partial + 1, np.nan)
        medians = np.full(n_partial + 1, np.nan)
        for k in range(n_partial + 1):
            v = values[bins == k]
            if v.size:
                means[k], medians[k] = v.mean(), np.median(v)
        return means, medians

    means, medians = bin_stats(bin_of, d)
    ci = None
    names = np.unique(ev)
    if n_boot > 0 and len(names) > 1:
        rng = np.random.default_rng(seed)
        index = [np.flatnonzero(ev == name) for name in names]
        boot_means = np.empty((n_boot, n_partial + 1))
        boot_medians = np.empty((n_boot, n_partial + 1))
        for b in range(n_boot):
            drawn = rng.choice(len(names), size=len(names), replace=True)
            rows = np.concatenate([index[i] for i in drawn])
            boot_means[b], boot_medians[b] = bin_stats(bin_of[rows], d[rows])
        # A bin missing from a resample is skipped for that resample;
        # bins with no images at all are all-NaN and never reported
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ci = {"mean": np.nanpercentile(boot_means, [2.5, 97.5], axis=0),
                  "median": np.nanpercentile(boot_medians, [2.5, 97.5],
                                             axis=0)}

    rows = []
    for k in range(n_partial + 1):
        in_bin = bin_of == k
        if not in_bin.any():
            continue
        v = d[in_bin]
        n_events = len(np.unique(ev[in_bin]))
        row = {
            "bin": labels[k], "coverage_low": lows[k],
            "coverage_high": highs[k], "full_coverage": k == n_partial,
            "images": int(in_bin.sum()), "events": n_events,
            "mean_depth_m": means[k], "mean_depth_m_ci95_low": np.nan,
            "mean_depth_m_ci95_high": np.nan,
            "median_depth_m": medians[k], "median_depth_m_ci95_low": np.nan,
            "median_depth_m_ci95_high": np.nan,
            "p10_depth_m": np.percentile(v, 10),
            "p90_depth_m": np.percentile(v, 90),
            "min_depth_m": v.min(), "max_depth_m": v.max(),
            "low_sample": bool(in_bin.sum() < MIN_BIN_IMAGES
                               or n_events < MIN_BIN_EVENTS),
        }
        if ci is not None:
            row["mean_depth_m_ci95_low"], row["mean_depth_m_ci95_high"] = (
                ci["mean"][:, k])
            row["median_depth_m_ci95_low"], row[
                "median_depth_m_ci95_high"] = ci["median"][:, k]
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_bin_summary(ax, x, row, width, offset):
    """
    Draw one bin's 10-90 % range, median and mean with their CIs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : float
        Bin center on the x-axis.
    row : pandas.Series
        One row of :func:`depth_by_coverage`.
    width : float
        Width of the 10-90 % box.
    offset : float
        Horizontal offset between the median and mean markers.
    """
    ax.add_patch(plt.Rectangle(
        (x - width / 2, row["p10_depth_m"]), width,
        row["p90_depth_m"] - row["p10_depth_m"], facecolor=BLUE,
        alpha=0.18, edgecolor="none", zorder=1))
    hollow = row["low_sample"]
    for value, low, high, color, marker, dx in (
            (row["median_depth_m"], row["median_depth_m_ci95_low"],
             row["median_depth_m_ci95_high"], BLUE, "o", -offset),
            (row["mean_depth_m"], row["mean_depth_m_ci95_low"],
             row["mean_depth_m_ci95_high"], ORANGE, "D", offset)):
        if np.isfinite(low) and np.isfinite(high):
            ax.errorbar(x + dx, value, yerr=[[value - low], [high - value]],
                        color=color, linewidth=1.5, capsize=3, zorder=3)
        ax.plot(x + dx, value, marker=marker, markersize=7,
                markerfacecolor="white" if hollow else color,
                markeredgecolor=color, markeredgewidth=1.5, linestyle="none",
                zorder=4)


def plot_depth_by_coverage(data, table, metric, pretty, out_dir, filename,
                           title=None, full_coverage=FULL_COVERAGE, seed=0,
                           ylabel=None, ylim=None):
    """
    Draw depth by coverage bin, with full coverage in its own panel.

    Left panel: images (grey), each bin's 10-90 % depth range (shaded),
    median (blue circles, solid line) and mean (orange diamonds, dashed
    line) with 95 % event-bootstrap confidence intervals. Right panel:
    the same for a fully covered road, where coverage no longer limits
    depth. Hollow markers flag low-sample bins.

    Parameters
    ----------
    data : pandas.DataFrame
        Images with ``Coverage`` and `metric`.
    table : pandas.DataFrame
        Output of :func:`depth_by_coverage`.
    metric : str
        Depth metric column.
    pretty : str
        Human-readable metric name for the y-axis.
    out_dir : str
        Output folder.
    filename : str
        Output image filename.
    title : str, optional
        Figure title. Default None (no title).
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered.
    seed : int, optional
        Seed of the horizontal jitter in the full-coverage panel.
    ylabel : str, optional
        Y-axis label. Default ``"<pretty> on Roadway (m)"``.
    ylim : tuple of float, optional
        Depth-axis limits, to share an axis between figures. Default:
        0 to just above the deepest image or bar.
    """
    images = data.dropna(subset=["Coverage", metric])
    images = images[images["Coverage"] > 0]
    partial_images = images[images["Coverage"] < full_coverage]
    full_images = images[images["Coverage"] >= full_coverage]
    partial = table[~table["full_coverage"]]
    full = table[table["full_coverage"]]

    fig, (ax, ax_full) = plt.subplots(
        1, 2, figsize=(10, 5.5), sharey=True,
        gridspec_kw={"width_ratios": [5, 1.2], "wspace": 0.06})

    ax.scatter(partial_images["Coverage"], partial_images[metric], s=9,
               color=CONTEXT_GREY, alpha=0.6, edgecolors="none", zorder=0)
    if not partial.empty:
        widths = partial["coverage_high"] - partial["coverage_low"]
        centers = partial["coverage_low"] + widths / 2
        for x, width, (_, row) in zip(centers, widths, partial.iterrows()):
            _draw_bin_summary(ax, x, row, width * 0.9, width * 0.12)
        ax.plot(centers - widths * 0.12, partial["median_depth_m"],
                color=BLUE, linewidth=1.8, zorder=2)
        ax.plot(centers + widths * 0.12, partial["mean_depth_m"],
                color=ORANGE, linewidth=1.8, linestyle="--", zorder=2)
    ax.set_xlim(0, full_coverage)
    ax.set_xlabel("Roadway Coverage (%)")
    ax.set_ylabel(ylabel or f"{pretty} on Roadway (m)")

    rng = np.random.default_rng(seed)
    ax_full.scatter(rng.uniform(-0.35, 0.35, len(full_images)),
                    full_images[metric], s=9, color=CONTEXT_GREY, alpha=0.6,
                    edgecolors="none", zorder=0)
    if not full.empty:
        row = full.iloc[0]
        _draw_bin_summary(ax_full, 0.0, row, 0.8, 0.12)
        ax_full.text(0.5, 0.02, f"n = {row['images']} images,\n"
                                f"{row['events']} events",
                     transform=ax_full.transAxes, ha="center", va="bottom",
                     fontsize=8, color=INK)
    # Shared depth axis from 0 to the deepest image or bar in either panel
    tops = [images[metric].max()] + [
        table[column].max() for column in
        ("p90_depth_m", "mean_depth_m_ci95_high", "median_depth_m_ci95_high")]
    ax.set_ylim(ylim or (0, 1.05 * np.nanmax(tops)))
    ax_full.set_xlim(-0.6, 0.6)
    ax_full.set_xticks([0.0])
    ax_full.set_xticklabels([f"Full\n(>={full_coverage:g}%)"])
    ax_full.tick_params(axis="y", left=False)
    ax_full.spines["left"].set_visible(False)
    ax_full.grid(axis="x", visible=False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=CONTEXT_GREY,
               markersize=5, label="Images"),
        Patch(facecolor=BLUE, alpha=0.18, label="10th-90th percentile"),
        Line2D([0], [0], color=BLUE, marker="o", markersize=7,
               label="Median depth (95% CI)"),
        Line2D([0], [0], color=ORANGE, marker="D", markersize=6,
               linestyle="--", label="Mean depth (95% CI)"),
    ]
    if table["low_sample"].any():
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="white", markeredgecolor=INK,
            label=f"Hollow: <{MIN_BIN_IMAGES} images or "
                  f"<{MIN_BIN_EVENTS} events"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    if title:
        fig.suptitle(title)
    _save(fig, out_dir, filename, tight=False)


def plot_one_to_one(sens, img, cov, x_label, y_label, trend_color,
                    out_dir, filename):
    """
    Plot image-derived vs sensor values with 1:1 and linear fit lines.

    Parameters
    ----------
    sens, img, cov : numpy.ndarray
        Sensor values, image values, and roadway coverage (%).
    x_label, y_label : str
        Axis labels.
    trend_color : str
        Color of the linear fit line.
    out_dir : str
        Output folder.
    filename : str
        Output filename.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    sizes = 40 + (cov / 100) * 100  # gentle size scaling with coverage
    scatter = ax.scatter(sens, img, c=cov, s=sizes, cmap=SEQUENTIAL_BLUE,
                         vmin=0, vmax=100, alpha=0.85, edgecolors="white",
                         linewidth=0.5)

    lo = min(img.min(), sens.min()) - 0.05
    hi = max(img.max(), sens.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1.5,
            label="1:1 Line")
    if len(img) > 1:
        try:
            m, b = np.polyfit(sens, img, 1)
            sign = "+" if b > 0 else ""
            ax.plot(np.array([lo, hi]), m * np.array([lo, hi]) + b,
                    color=trend_color, linewidth=2,
                    label=f"Linear Fit (y={m:.2f}x{sign}{b:.2f})")
        except (np.linalg.LinAlgError, ValueError) as err:
            logger.warning("Linear fit failed for %s: %s", filename, err)

    ax.set_xlim([lo, hi])
    ax.set_ylim([lo, hi])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.7)
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Roadway Percentage Covered (%)")
    ax.legend(loc="upper left")
    _save(fig, out_dir, filename)


def plot_point_metrics(data, roi, metric, has_labels, sensor,
                       sensor_wse_floor, out_dir, n_boot=0, seed=0,
                       bin_width=COVERAGE_BIN_WIDTH,
                       full_coverage=FULL_COVERAGE, shared_ylim=None):
    """
    Draw the per-image plots (1, 1b, 1c, 6, 7) for one metric.

    Parameters
    ----------
    data : pandas.DataFrame
        All events' rows for one ROI.
    roi : str
        ROI name for titles and filenames.
    metric : str
        Depth metric column.
    has_labels : bool
        Whether segmentation labels were supplied.
    sensor : pandas.DataFrame or None
        Sensor readings.
    sensor_wse_floor : float or None
        Lower clip applied to sensor WSE in plot 7.
    out_dir : str
        Output folder.
    n_boot : int, optional
        Event bootstrap resamples for the fits. Default 0 (none).
    seed : int, optional
        Bootstrap random seed. Default 0.
    bin_width : float, optional
        Coverage bin width (%) of the binned plots.
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered.
    shared_ylim : tuple of float, optional
        Depth-axis limits of the 1c plots, shared with plot 8 (see
        :func:`shared_depth_limits`). Default: automatic.

    Returns
    -------
    list of dict
        One record per plot (see :func:`_record`) for the run summary.
    """
    pretty = PRETTY_METRICS[metric]
    point_colors = data["Segmentation"].map(SEGMENTATION_COLORS)
    stem = f"{roi}_{metric}"
    records = []
    plotted = data.dropna(subset=["Coverage", metric])
    n_good = int((plotted["Segmentation"] == "good").sum())

    # 1: coverage vs depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(data[metric], data["Coverage"], alpha=0.6, s=12,
               c=point_colors, edgecolors="none")
    ax.set_xlabel(f"{metric} on Roadway (m)")
    ax.set_ylabel("Roadway Percentage Covered (%)")
    ax.set_title(f"Roadway Coverage vs {metric}\n({roi})")
    ax.grid(True, linestyle="--", alpha=0.7)
    if has_labels:
        ax.legend(SEGMENTATION_HANDLES,
                  ["Good Segmentation", "Standard/Poor Segmentation"])
    _save(fig, out_dir, f"{stem}_1_evolution_plot.jpg")
    _record(records, "1", f"{stem}_1_evolution_plot.jpg", plotted,
            f"all images with water on the ROI ({n_good} well segmented, "
            f"{len(plotted) - n_good} other)")

    # 1b: depth vs coverage with logistic fit (all data)
    fit_data = plotted
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(data["Coverage"], data[metric], alpha=0.6, s=12,
               c=point_colors, edgecolors="none")
    if len(fit_data) > 3:
        try:
            fit = fit_depth_from_coverage(fit_data["Coverage"],
                                          fit_data[metric],
                                          fit_data["event"], n_boot, seed)
            _draw_depth_fit(ax, fit)
            write_fit_summary(fit, out_dir, f"{stem}_1b")
            fit_note = _fit_note(fit)
        except (RuntimeError, ValueError) as err:
            logger.warning("Logistic fit failed for %s (all): %s", stem, err)
            fit_note = f"fit failed ({err})"
    else:
        fit_note = "too few images to fit"
    ax.set_xlabel("Roadway Percentage Covered (%)")
    ax.set_ylabel(f"{metric} (m)")
    ax.set_title(f"{metric} vs Roadway Coverage (Logistic Fit - All "
                 f"Data)\n({roi})")
    handles, labels = ax.get_legend_handles_labels()
    if has_labels:
        handles = SEGMENTATION_HANDLES + handles
        labels = ["Good Seg.", "Std/Poor Seg."] + labels
    ax.legend(handles, labels)
    _save(fig, out_dir, f"{stem}_1b_reversed_fit.jpg")
    _record(records, "1b logistic", f"{stem}_1b_reversed_fit.jpg", fit_data,
            f"all images; {fit_note}")

    # 1b: depth by coverage bin (all images)
    binned = depth_by_coverage(fit_data["Coverage"], fit_data[metric],
                               fit_data["event"], bin_width, full_coverage,
                               n_boot, seed)
    if not binned.empty:
        binned.to_csv(os.path.join(out_dir,
                                   f"{stem}_1b_depth_by_coverage.csv"),
                      index=False)
        plot_depth_by_coverage(fit_data, binned, metric, pretty, out_dir,
                               f"{stem}_1b_binned_depth.jpg",
                               title=f"{pretty} by Roadway Coverage (all "
                                     f"images; {roi})",
                               full_coverage=full_coverage, seed=seed)
        _record(records, "1b binned", f"{stem}_1b_binned_depth.jpg",
                fit_data[fit_data["Coverage"] > 0],
                "all images with coverage > 0; " + _binned_note(
                    binned, fit_data, full_coverage))

    # 1c: as 1b, well-segmented images only
    good = fit_data[fit_data["Segmentation"] == "good"]
    if len(good) > 3:
        fit = _plot_good_only_fit(good, metric, pretty, out_dir,
                                  f"{stem}_1c_reversed_fit_good_only.jpg",
                                  f"{stem}_1c", n_boot, seed,
                                  ylim=shared_ylim)
        _record(records, "1c logistic",
                f"{stem}_1c_reversed_fit_good_only.jpg", good,
                "well-segmented images; " + (_fit_note(fit) if fit
                                             else "fit failed"))
        binned = depth_by_coverage(good["Coverage"], good[metric],
                                   good["event"], bin_width, full_coverage,
                                   n_boot, seed)
        if not binned.empty:
            binned.to_csv(os.path.join(out_dir,
                                       f"{stem}_1c_depth_by_coverage.csv"),
                          index=False)
            plot_depth_by_coverage(good, binned, metric, pretty, out_dir,
                                   f"{stem}_1c_binned_depth_good_only.jpg",
                                   full_coverage=full_coverage, seed=seed,
                                   ylim=shared_ylim)
            _record(records, "1c binned",
                    f"{stem}_1c_binned_depth_good_only.jpg",
                    good[good["Coverage"] > 0],
                    "well-segmented images with coverage > 0; "
                    + _binned_note(binned, good, full_coverage))
    else:
        reason = ("no segmentation labels given" if not has_labels
                  else f"only {len(good)} well-segmented images")
        if has_labels:
            logger.info("Too few well-segmented points for plot 1c (%s).",
                        stem)
        _record(records, "1c", "(not made)", good.iloc[0:0], reason)

    if sensor is None:
        for plot in ("6", "7"):
            _record(records, plot, "(not made)", data.iloc[0:0],
                    "no sensor CSV, or no segmentation labels")
        return records
    green = data[data["Segmentation"] == "good"]

    # 6: image vs sensor depth
    if "SensorDepth_m" in data:
        rows = green.dropna(subset=[metric, "SensorDepth_m", "Coverage"])
        if not rows.empty:
            plot_one_to_one(rows["SensorDepth_m"].values, rows[metric].values,
                            rows["Coverage"].values,
                            "In-Situ Sensor Depth (m)",
                            f"Image Derived {pretty} (m)", ORANGE,
                            out_dir, f"{stem}_6_sensor_1to1.jpg")
        _record(records, "6", f"{stem}_6_sensor_1to1.jpg" if len(rows)
                else "(not made)", rows,
                "well-segmented images with a sensor reading within 5 min")

    # 7: image vs sensor WSE
    wse_metric = metric.replace("Depth", "WSE")
    if "SensorWSE_m" in data and wse_metric in data:
        rows = green.dropna(subset=[wse_metric, "SensorWSE_m", "Coverage"])
        if not rows.empty:
            sens_wse = rows["SensorWSE_m"]
            if sensor_wse_floor is not None:
                sens_wse = sens_wse.clip(lower=sensor_wse_floor)
            plot_one_to_one(sens_wse.values, rows[wse_metric].values,
                            rows["Coverage"].values,
                            "In-Situ Sensor WSE (m)",
                            f"Image Derived {pretty.replace('Depth', 'WSE')}"
                            f" (m)", ORANGE, out_dir,
                            f"{roi}_{wse_metric}_7_sensor_1to1.jpg")
        _record(records, "7", f"{roi}_{wse_metric}_7_sensor_1to1.jpg"
                if len(rows) else "(not made)", rows,
                f"well-segmented images with {wse_metric} and a sensor WSE "
                f"reading within 5 min")
    return records


def _draw_depth_fit(ax, fit):
    """
    Draw a depth-from-coverage fit with its bands.

    Blue: fit line and its 95 % confidence band. Orange (dashed):
    median. Light blue: the 80 % prediction band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with coverage on x and depth on y.
    fit : dict
        Output of :func:`fit_depth_from_coverage`.
    """
    c, curves = fit["c"], fit["quantiles"]
    if 0.1 in curves and 0.9 in curves:
        inside = fit["band_fraction"]
        note = "" if np.isnan(inside) else f", {100 * inside:.0f}% of images"
        ax.fill_between(c, curves[0.1], curves[0.9], color=BLUE, alpha=0.12,
                        linewidth=0,
                        label=f"80% Prediction Band (0.1-0.9 quantiles{note})")
    if fit["curve_ci"] is not None:
        ax.fill_between(c, *fit["curve_ci"], color=BLUE, alpha=0.35,
                        linewidth=0,
                        label=f"95% CI of Fit ({fit['n_boot']} event "
                              f"resamples)")
    ax.plot(c, fit["fit"], color=BLUE, linewidth=2,
            label=f"Logistic Fit ($R^2$={fit['r2']:.2f}, "
                  f"RMSE={fit['rmse']:.3f} m)")
    if 0.5 in curves:
        ax.plot(c, curves[0.5], linestyle="--", color=ORANGE, linewidth=2,
                label="Median Depth (0.5 Quantile)")


def _plot_good_only_fit(good, metric, pretty, out_dir, filename, stem,
                        n_boot=0, seed=0, ylabel=None, ylim=None):
    """
    Draw plot 1c: logistic fit of well-segmented images only.

    Parameters
    ----------
    good : pandas.DataFrame
        Well-segmented rows with `metric`, ``Coverage`` and ``event``.
    metric : str
        Depth metric column.
    pretty : str
        Human-readable metric name for the y-axis.
    out_dir : str
        Output folder.
    filename : str
        Output image filename.
    stem : str
        Prefix of the fit summary CSVs.
    n_boot : int, optional
        Event bootstrap resamples. Default 0 (none).
    seed : int, optional
        Bootstrap random seed. Default 0.
    ylabel : str, optional
        Y-axis label. Default ``"<pretty> on Roadway (m)"``.
    ylim : tuple of float, optional
        Depth-axis limits, to share an axis between figures. Curves
        outside them (e.g. below 0 m near 0 % coverage) are cut off.
        Default: automatic.

    Returns
    -------
    dict or None
        The fit, or None if it failed.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(good["Coverage"], good[metric], alpha=0.6, s=12,
               c=CONTEXT_GREY, edgecolors="none", label="Images")
    try:
        fit = fit_depth_from_coverage(good["Coverage"], good[metric],
                                      good["event"], n_boot, seed)
        _draw_depth_fit(ax, fit)
        write_fit_summary(fit, out_dir, stem)
    except (RuntimeError, ValueError) as err:
        logger.warning("Logistic fit failed for %s: %s", filename, err)
        fit = None

    ax.set_xlabel("Roadway Percentage Covered (%)")
    ax.set_ylabel(ylabel or f"{pretty} on Roadway (m)")
    if ylim is not None:
        ax.set_ylim(ylim)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # drop duplicate legend entries
    ax.legend(unique.values(), unique.keys())
    _save(fig, out_dir, filename)
    return fit


def plot_event_peaks(peaks, roi, metric, basis, out_dir,
                     plotting_position="empirical"):
    """
    Draw the per-event peak plots (2-5) for one metric.

    Parameters
    ----------
    peaks : pandas.DataFrame
        One row per event from :func:`summarize_event`.
    roi : str
        ROI name for titles and filenames.
    metric : str
        Depth metric column.
    basis : str
        Which images the peaks come from, for titles (e.g.
        ``"well-segmented images"``).
    out_dir : str
        Output folder.
    plotting_position : {'empirical', 'weibull'}, optional
        Probability assigned to sorted values. Default 'empirical'.

    Returns
    -------
    list of dict
        One record per plot (see :func:`_record`) for the run summary.
    """
    column = f"{metric}_max"
    if column not in peaks:
        return []
    valid = peaks.dropna(subset=[column, "Coverage_max"])
    values = valid[column].to_numpy()
    if values.size == 0:
        return []
    stem = f"{roi}_{metric}"
    label = f"Peak Event {metric} (m)"
    subtitle = f"({roi}; {len(values)} events, {basis})"

    # 2: peak coverage vs peak depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(values, valid["Coverage_max"], color=BLUE, alpha=0.8,
               edgecolors="white", linewidths=0.8, s=45)
    ax.set_xlabel(label)
    ax.set_ylabel("Peak Roadway Percentage Covered (%)")
    ax.set_title(f"Peak Coverage vs Peak {metric} per Event\n{subtitle}")
    ax.grid(True, linestyle="--", alpha=0.7)
    _save(fig, out_dir, f"{stem}_2_summary_scatter.png")

    # 3: empirical CDF and 5: exceedance probability. P(X <= x) is
    # right-continuous, so each value holds until the next sample
    # (step "post"); P(X >= x) is left-continuous, so each value holds
    # back to the previous sample (step "pre").
    ordered, cdf, exceedance = empirical_probabilities(values,
                                                       plotting_position)
    position = ("i/n" if plotting_position == "empirical"
                else "Weibull i/(n+1)")
    for probs, where, color, marker, y_label, title, suffix in (
            (cdf, "post", BLUE, "o",
             "Cumulative Probability P(X <= x)", "CDF of Peak",
             "3_depth_cdf"),
            (exceedance, "pre", BLUE, "o",
             "Exceedance Probability P(X >= x)",
             "Exceedance Probability of Peak", "5_depth_exceedance")):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.step(ordered, probs, where=where, color=color, linewidth=1.8)
        ax.plot(ordered, probs, marker, color=color, markersize=6,
                linestyle="none")
        ax.set_xlabel(label)
        ax.set_ylabel(f"{y_label} [{position}]")
        ax.set_title(f"{title} {metric}\n{subtitle}")
        ax.grid(True, linestyle="--", alpha=0.7)
        _save(fig, out_dir, f"{stem}_{suffix}.png")

    # 4: PDF as a density histogram and a boundary-corrected KDE
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    edges, rule = density_bins(values)
    ax.hist(values, bins=edges, density=True, color=BLUE, alpha=0.35,
            edgecolor="white", linewidth=2,
            label=f"Density Histogram ({len(edges) - 1} {rule} bins)")
    kde = reflected_kde(values)
    if kde is not None:
        ax.plot(*kde, color=ORANGE, linewidth=2,
                label="Gaussian KDE (Scott bandwidth, "
                                   "reflected at 0 m)")
    ax.set_xlabel(label)
    ax.set_ylabel("Probability Density (1/m)")
    ax.set_title(f"PDF of Peak {metric}\n{subtitle}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    _save(fig, out_dir, f"{stem}_4_depth_pdf.png")

    dropped = len(peaks) - len(valid)
    note = (f"one peak per event from {basis}; {dropped} of {len(peaks)} "
            f"events left out (no qualifying image with water on the ROI)")
    position_note = f"; plotting position {position}"
    records = []
    for plot, suffix, extra in (
            ("2", "2_summary_scatter.png", ""),
            ("3", "3_depth_cdf.png", position_note),
            ("4", "4_depth_pdf.png",
             f"; {len(edges) - 1} {rule} bins and reflected KDE"),
            ("5", "5_depth_exceedance.png", position_note)):
        records.append({"plot": plot, "file": f"{stem}_{suffix}",
                        "images": int(valid["images"].sum()),
                        "events": len(valid), "notes": note + extra})
    return records


def density_bins(values):
    """
    Histogram bin edges by the Freedman-Diaconis rule.

    Falls back to Sturges' rule when the interquartile range is zero
    or the rule gives a single bin.

    Parameters
    ----------
    values : numpy.ndarray
        Sample.

    Returns
    -------
    edges : numpy.ndarray
        Bin edges.
    rule : str
        Rule used (``"Freedman-Diaconis"`` or ``"Sturges"``).
    """
    edges = np.histogram_bin_edges(values, bins="fd")
    if len(edges) >= 3:
        return edges, "Freedman-Diaconis"
    return np.histogram_bin_edges(values, bins="sturges"), "Sturges"


def reflected_kde(values, n_points=400):
    """
    Gaussian kernel density estimate reflected at zero.

    A plain KDE puts probability below 0 m, where depths cannot be;
    reflecting it about 0 returns that mass to positive depths, so the
    curve integrates to 1 over depth >= 0.

    Parameters
    ----------
    values : numpy.ndarray
        Non-negative sample.
    n_points : int, optional
        Evaluation points. Default 400.

    Returns
    -------
    tuple of numpy.ndarray or None
        ``(x, density)``, or None for fewer than 3 values, a constant
        sample, or negative values.
    """
    if values.size < 3 or np.ptp(values) == 0 or values.min() < 0:
        return None
    kde = gaussian_kde(values)
    x = np.linspace(0.0, values.max() + 3 * values.std(), n_points)
    return x, kde(x) + kde(-x)


def plot_sensor_depth_by_coverage(data, roi, out_dir, n_boot=0, seed=0,
                                  bin_width=COVERAGE_BIN_WIDTH,
                                  full_coverage=FULL_COVERAGE,
                                  shared_ylim=None):
    """
    Draw plot 8: in-situ sensor depth against image coverage.

    The same binned and logistic plots as 1c, but each well-segmented
    image's depth is the sensor's road water level at the nearest
    reading within 5 minutes (negative levels set to 0 m), not a depth
    derived from the image. Coverage still comes from the image. It
    does not depend on the image depth metric, so it is drawn once per
    ROI. Images without a sensor reading (e.g. image-only events) are
    left out.

    Parameters
    ----------
    data : pandas.DataFrame
        All images of one ROI, with ``SensorDepth_m``.
    roi : str
        ROI name for filenames.
    out_dir : str
        Output folder.
    n_boot : int, optional
        Event bootstrap resamples. Default 0 (none).
    seed : int, optional
        Bootstrap random seed. Default 0.
    bin_width : float, optional
        Coverage bin width (%).
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered.
    shared_ylim : tuple of float, optional
        Depth-axis limits shared with the 1c plots. Default: automatic.

    Returns
    -------
    list of dict
        Run-summary records.
    """
    records = []
    stem = f"{roi}_{SENSOR_DEPTH}"
    if "SensorDepth_m" not in data:
        _record(records, "8", "(not made)", data.iloc[0:0],
                "no sensor CSV, or no segmentation labels")
        return records
    good = data[(data["Segmentation"] == "good")
                & (data["Coverage"] > 0)].dropna(
        subset=["Coverage", "SensorDepth_m"])
    if len(good) <= 3:
        _record(records, "8", "(not made)", good,
                f"only {len(good)} well-segmented images with a sensor "
                f"reading")
        return records

    ylabel = "In-Situ Sensor Depth on Roadway (m)"
    pretty = "Sensor Depth"
    included = ("well-segmented images with coverage > 0 and a sensor "
                "reading within 5 min; depth from the sensor")
    binned = depth_by_coverage(good["Coverage"], good["SensorDepth_m"],
                               good["event"], bin_width, full_coverage,
                               n_boot, seed)
    binned.to_csv(os.path.join(out_dir, f"{stem}_8_depth_by_coverage.csv"),
                  index=False)
    plot_depth_by_coverage(good, binned, "SensorDepth_m", pretty, out_dir,
                           f"{stem}_8_binned_depth_good_only.jpg",
                           full_coverage=full_coverage, seed=seed,
                           ylabel=ylabel, ylim=shared_ylim)
    _record(records, "8 binned", f"{stem}_8_binned_depth_good_only.jpg", good,
            f"{included}; " + _binned_note(binned, good, full_coverage))

    fit = _plot_good_only_fit(good, "SensorDepth_m", pretty, out_dir,
                              f"{stem}_8_reversed_fit_good_only.jpg",
                              f"{stem}_8", n_boot, seed, ylabel=ylabel,
                              ylim=shared_ylim)
    _record(records, "8 logistic", f"{stem}_8_reversed_fit_good_only.jpg",
            good, f"{included}; " + (_fit_note(fit) if fit
                                     else "fit failed"))
    return records


def shared_depth_limits(data):
    """
    Depth-axis limits shared by one ROI's 1c plots and plot 8.

    Runs from 0 to 5 % above the deepest well-segmented image depth
    (any of :data:`METRICS`) or matched sensor depth, so every 1c
    figure and plot 8 of the ROI can be compared side by side.

    Parameters
    ----------
    data : pandas.DataFrame
        All images of one ROI.

    Returns
    -------
    tuple of float or None
        ``(0, top)``, or None if there is nothing to plot.
    """
    good = data[(data["Segmentation"] == "good") & (data["Coverage"] > 0)]
    columns = [c for c in (*METRICS, "SensorDepth_m") if c in good]
    if good.empty or not columns:
        return None
    top = np.nanmax(good[columns].to_numpy(dtype=float))
    return (0.0, 1.05 * float(top)) if np.isfinite(top) and top > 0 else None


def _draw_paired_bin(ax, x, width, rows):
    """
    Draw two depth sources side by side in one coverage bin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : float
        Bin center.
    width : float
        Bin width on the x-axis.
    rows : list of tuple
        ``(row, color, marker)`` per source, left to right; ``row`` is
        one row of :func:`depth_by_coverage`.
    """
    box = width * 0.36
    for k, (row, color, marker) in enumerate(rows):
        center = x + (k - 0.5) * width * 0.42
        ax.add_patch(plt.Rectangle(
            (center - box / 2, row["p10_depth_m"]), box,
            row["p90_depth_m"] - row["p10_depth_m"], facecolor=color,
            alpha=0.18, edgecolor="none", zorder=1))
        low, high = (row["median_depth_m_ci95_low"],
                     row["median_depth_m_ci95_high"])
        value = row["median_depth_m"]
        if np.isfinite(low) and np.isfinite(high):
            ax.errorbar(center, value, yerr=[[value - low], [high - value]],
                        color=color, linewidth=1.5, capsize=3, zorder=3)
        ax.plot(center, value, marker=marker, markersize=7, linestyle="none",
                markerfacecolor="white" if row["low_sample"] else color,
                markeredgecolor=color, markeredgewidth=1.5, zorder=4)


def plot_image_vs_sensor_by_coverage(data, roi, metric, out_dir, n_boot=0,
                                     seed=0, bin_width=COVERAGE_BIN_WIDTH,
                                     full_coverage=FULL_COVERAGE,
                                     shared_ylim=None):
    """
    Draw plot 9: image-derived and sensor depth by coverage, together.

    Uses the same images for both sources: well-segmented images with
    coverage > 0, the image metric, and a sensor reading within 5 min.
    Each coverage bin shows the image-derived median (blue circles,
    solid line) beside the sensor median (orange diamonds, dashed),
    each with its 95 % event-bootstrap confidence interval and its
    10th-90th percentile box; full coverage has its own panel.

    Parameters
    ----------
    data : pandas.DataFrame
        All images of one ROI.
    roi : str
        ROI name.
    metric : str
        Image depth metric.
    out_dir : str
        Output folder.
    n_boot : int, optional
        Event bootstrap resamples.
    seed : int, optional
        Bootstrap random seed.
    bin_width : float, optional
        Coverage bin width (%).
    full_coverage : float, optional
        Coverage (%) from which the road counts as fully covered.
    shared_ylim : tuple of float, optional
        Depth-axis limits shared with 1c and 8.

    Returns
    -------
    list of dict
        Run-summary records.
    """
    records = []
    stem = f"{roi}_{metric}"
    filename = f"{stem}_9_image_vs_sensor_binned_good_only.jpg"
    if "SensorDepth_m" not in data:
        _record(records, "9", "(not made)", data.iloc[0:0],
                "no sensor CSV, or no segmentation labels")
        return records
    paired = data[(data["Segmentation"] == "good")
                  & (data["Coverage"] > 0)].dropna(
        subset=["Coverage", metric, "SensorDepth_m"])
    if len(paired) <= 3:
        _record(records, "9", "(not made)", paired,
                f"only {len(paired)} well-segmented images with a sensor "
                f"reading")
        return records

    pretty = PRETTY_METRICS[metric]
    sources = (
        (f"Image {pretty.lower()}", metric, BLUE, "o", "-"),
        ("In-situ sensor depth", "SensorDepth_m", ORANGE, "D", "--"))
    tables = [depth_by_coverage(paired["Coverage"], paired[column],
                                paired["event"], bin_width, full_coverage,
                                n_boot, seed)
              for _, column, *_ in sources]
    pd.concat([table.assign(depth_source=name)
               for (name, *_), table in zip(sources, tables)],
              ignore_index=True).to_csv(
        os.path.join(out_dir, f"{stem}_9_depth_by_coverage.csv"),
        index=False)

    fig, (ax, ax_full) = plt.subplots(
        1, 2, figsize=(10, 5.5), sharey=True,
        gridspec_kw={"width_ratios": [5, 1.2], "wspace": 0.06})
    partial = [t[~t["full_coverage"]].reset_index(drop=True) for t in tables]
    full = [t[t["full_coverage"]] for t in tables]
    widths = partial[0]["coverage_high"] - partial[0]["coverage_low"]
    centers = partial[0]["coverage_low"] + widths / 2
    for i, (x, width) in enumerate(zip(centers, widths)):
        _draw_paired_bin(ax, x, width, [
            (table.iloc[i], color, marker)
            for table, (_, _, color, marker, _) in zip(partial, sources)])
    for k, (table, (_, _, color, _, style)) in enumerate(
            zip(partial, sources)):
        ax.plot(centers + (k - 0.5) * widths * 0.42, table["median_depth_m"],
                color=color, linestyle=style, linewidth=1.8, zorder=2)
    if all(len(t) for t in full):
        _draw_paired_bin(ax_full, 0.0, 1.0, [
            (table.iloc[0], color, marker)
            for table, (_, _, color, marker, _) in zip(full, sources)])
        row = full[0].iloc[0]
        ax_full.text(0.5, 0.02, f"n = {row['images']} images,\n"
                                f"{row['events']} events",
                     transform=ax_full.transAxes, ha="center", va="bottom",
                     fontsize=8, color=INK)

    ax.set_xlim(0, full_coverage)
    ax.set_xlabel("Roadway Coverage (%)")
    ax.set_ylabel("Depth on Roadway (m)")
    tops = [t[c].max() for t in tables
            for c in ("p90_depth_m", "median_depth_m_ci95_high")]
    ax.set_ylim(shared_ylim or (0, 1.05 * np.nanmax(tops)))
    ax_full.set_xlim(-0.6, 0.6)
    ax_full.set_xticks([0.0])
    ax_full.set_xticklabels([f"Full\n(>={full_coverage:g}%)"])
    ax_full.tick_params(axis="y", left=False)
    ax_full.spines["left"].set_visible(False)
    ax_full.grid(axis="x", visible=False)

    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=style,
                      markersize=7, label=f"{name}: median (95% CI)")
               for name, _, color, marker, style in sources]
    handles += [Patch(facecolor=color, alpha=0.18,
                      label=f"{name}: 10th-90th percentile")
                for name, _, color, _, _ in sources]
    if any(t["low_sample"].any() for t in tables):
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=7,
            markerfacecolor="white", markeredgecolor=INK,
            label=f"Hollow: <{MIN_BIN_IMAGES} images or "
                  f"<{MIN_BIN_EVENTS} events"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    _save(fig, out_dir, filename, tight=False)

    _record(records, "9", filename, paired,
            "well-segmented images with coverage > 0 and a sensor reading "
            "within 5 min; image metric and sensor depth from the same "
            "images; " + _binned_note(tables[0], paired, full_coverage))
    return records


# ----- Run summary -----------------------------------------------------------

def _record(records, plot, filename, frame, notes):
    """
    Append one plot's image and event counts to `records`.

    Parameters
    ----------
    records : list of dict
        Records to extend.
    plot : str
        Plot number (e.g. ``"1c binned"``).
    filename : str
        Output file, or ``"(not made)"``.
    frame : pandas.DataFrame
        The images the plot used, with an ``event`` column.
    notes : str
        What the images are and any filters applied.
    """
    records.append({
        "plot": plot, "file": filename, "images": len(frame),
        "events": int(frame["event"].nunique()) if len(frame) else 0,
        "notes": notes})


def _fit_note(fit):
    """
    Summarize a logistic fit for the run summary.

    Parameters
    ----------
    fit : dict
        Output of :func:`fit_depth_from_coverage`.

    Returns
    -------
    str
        Images and events fitted, exclusions, and calibration.
    """
    return (f"fitted to {fit['n']} images from {fit['n_events']} events "
            f"({fit['n_excluded']} zero-coverage images left out); "
            f"R^2 {fit['r2']:.2f}, RMSE {fit['rmse']:.3f} m; "
            f"{100 * fit['band_fraction']:.0f}% of images inside the 80% "
            f"band (check *_band_calibration.csv)")


def _binned_note(table, frame, full_coverage):
    """
    Summarize a binned depth table for the run summary.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`depth_by_coverage`.
    frame : pandas.DataFrame
        The images binned.
    full_coverage : float
        Full-coverage threshold (%).

    Returns
    -------
    str
        Bin, low-sample, and full-coverage counts.
    """
    full = frame[frame["Coverage"] >= full_coverage]
    return (f"{int((~table['full_coverage']).sum())} partial-coverage bins "
            f"with images, {int(table['low_sample'].sum())} flagged "
            f"low_sample; {len(full)} images from "
            f"{full['event'].nunique()} events at full coverage "
            f"(>={full_coverage:g}%)")


def write_run_summary(out_dir, settings, overviews, records):
    """
    Write ``plot_summary.md`` and ``plot_summary.csv`` for a run.

    Parameters
    ----------
    out_dir : str
        Output folder.
    settings : dict
        Inputs and options of the run, shown at the top.
    overviews : dict of str to dict
        Per-ROI totals: ``event_folders``, ``images`` (all rows),
        ``wet`` (images with water on the ROI), ``good`` (well-segmented
        wet images), ``events_with_good``.
    records : list of dict
        Plot records with ``roi`` and ``metric`` added.
    """
    table = pd.DataFrame(records, columns=["roi", "metric", "plot", "file",
                                           "images", "events", "notes"])
    order = {name: i for i, name in enumerate(PLOT_ORDER)}
    table = table.sort_values(
        ["roi", "metric", "plot"], kind="stable",
        key=lambda col: (col.map(order) if col.name == "plot" else
                         col.map({m: i for i, m in
                                  enumerate([*METRICS, SENSOR_DEPTH])})
                         if col.name == "metric" else col))
    table.to_csv(os.path.join(out_dir, "plot_summary.csv"), index=False)

    lines = ["# Roadway accessibility plot summary", "",
             f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by "
             "`access_plotter.py`. What each plot shows is described in "
             "the README section *Roadway coverage and depth plots*.", "",
             "## Run settings", ""]
    lines += [f"- **{key}:** {value}" for key, value in settings.items()]
    lines += ["", "Counts below are the images and events each plot "
              "actually used. *Images* are those with water on the ROI "
              "(a depth value); *events* are flood event folders.", ""]
    for roi, overview in overviews.items():
        lines += [f"## ROI `{roi}`", "",
                  f"- Event folders with results: {overview['event_folders']}",
                  f"- Images in those folders: {overview['images']}, of "
                  f"which {overview['wet']} have water on the ROI "
                  f"({overview['good']} well segmented, "
                  f"{overview['wet'] - overview['good']} poor or not "
                  f"reviewed)",
                  f"- Events with at least one well-segmented image: "
                  f"{overview['events_with_good']}", ""]
        for metric, group in table[table["roi"] == roi].groupby(
                "metric", sort=False):
            lines += [f"### {metric}", "",
                      "| Plot | File | Images | Events | Included |",
                      "|---|---|---:|---:|---|"]
            lines += [f"| {r.plot} | `{r.file}` | {r.images} | {r.events} "
                      f"| {r.notes} |" for r in group.itertuples()]
            lines.append("")
    with open(os.path.join(out_dir, "plot_summary.md"), "w") as handle:
        handle.write("\n".join(lines))
    logger.info("Wrote plot_summary.md and plot_summary.csv to %s", out_dir)


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse. Default reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot POSEIDON roadway accessibility statistics.")
    parser.add_argument("--events_dir", default="flood_events",
                        help="Folder of event subfolders "
                             "(default: flood_events).")
    parser.add_argument("--out_dir", default="plots",
                        help="Output folder (default: plots).")
    parser.add_argument("--labels", default=None,
                        help="segmentation_labels.csv marking good images.")
    parser.add_argument("--sensor", default=None,
                        help="Sensor CSV for 1:1 depth/WSE comparisons.")
    parser.add_argument("--sensor_wse_floor", type=float, default=None,
                        help="Clip sensor WSE below this value (m) in plot "
                             "7; defaults to the --site config value.")
    parser.add_argument("--site", default=None,
                        help="Site config supplying sensor_wse_floor.")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Event bootstrap resamples for the fit "
                             "confidence bands (0 to skip).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Bootstrap random seed.")
    parser.add_argument("--coverage_bin_width", type=float,
                        default=COVERAGE_BIN_WIDTH,
                        help="Coverage bin width (%%) of the binned depth "
                             "plots and tables.")
    parser.add_argument("--full_coverage", type=float, default=FULL_COVERAGE,
                        help="Coverage (%%) from which the road counts as "
                             "fully covered.")
    parser.add_argument("--plotting_position",
                        choices=("empirical", "weibull"),
                        default="empirical",
                        help="Probabilities of sorted event peaks: i/n "
                             "(empirical) or i/(n+1) (Weibull).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Generate all accessibility plots.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    if args.sensor_wse_floor is None and args.site:
        args.sensor_wse_floor = load_site_config(args.site).sensor_wse_floor
    os.makedirs(args.out_dir, exist_ok=True)

    good_timestamps = load_good_timestamps(args.labels)
    sensor = load_sensor_data(args.sensor)
    has_labels = bool(good_timestamps)

    basis = "well-segmented images" if has_labels else "all images"
    records, overviews = [], {}
    for roi, paths in find_roi_csvs(args.events_dir).items():
        events = [load_event(path, good_timestamps, sensor)
                  for path in paths]
        data = pd.concat(events, ignore_index=True)
        peaks = pd.DataFrame([summarize_event(event, has_labels)
                              for event in events])
        if has_labels:
            skipped = int((peaks["images"] == 0).sum())
            logger.info("%s: event peaks use well-segmented images; %d of "
                        "%d events have none and are left out.", roi,
                        skipped, len(peaks))
        wet = data[data[[m for m in METRICS if m in data]].notna().any(
            axis=1)]
        good_rows = wet[wet["Segmentation"] == "good"]
        shared_ylim = shared_depth_limits(data)
        overviews[roi] = {
            "event_folders": len(paths), "images": len(data),
            "wet": len(wet), "good": len(good_rows),
            "events_with_good": int(good_rows["event"].nunique())}
        for metric in METRICS:
            if metric not in data:
                logger.warning("No '%s' column for %s; skipped.", metric,
                               roi)
                continue
            logger.info("Plotting %s - %s", roi, metric)
            made = plot_point_metrics(
                data, roi, metric, has_labels, sensor,
                args.sensor_wse_floor, args.out_dir, args.n_boot,
                args.seed, args.coverage_bin_width, args.full_coverage,
                shared_ylim)
            made += plot_event_peaks(peaks, roi, metric, basis,
                                     args.out_dir, args.plotting_position)
            made += plot_image_vs_sensor_by_coverage(
                data, roi, metric, args.out_dir, args.n_boot, args.seed,
                args.coverage_bin_width, args.full_coverage, shared_ylim)
            records += [dict(record, roi=roi, metric=metric)
                        for record in made]
        records += [dict(record, roi=roi, metric=SENSOR_DEPTH) for record in
                    plot_sensor_depth_by_coverage(
                        data, roi, args.out_dir, args.n_boot, args.seed,
                        args.coverage_bin_width, args.full_coverage,
                        shared_ylim)]
    write_run_summary(args.out_dir, {
        "Events folder": args.events_dir,
        "Segmentation labels": args.labels or "none (all images 'other')",
        "Sensor CSV": args.sensor or "none",
        "Coverage bins": f"{args.coverage_bin_width:g}% wide below "
                         f"{args.full_coverage:g}%; full coverage "
                         f">={args.full_coverage:g}%",
        "Event bootstrap": f"{args.n_boot} resamples, seed {args.seed}",
        "Plotting position (plots 3, 5)": args.plotting_position,
    }, overviews, records)
    logger.info("Plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
