"""Plot roadway accessibility statistics across all flood events.

Reads every ``<roi>_accessibility_time_series.csv`` (and the matching
``<roi>_wse_time_series.csv``) under ``<events_dir>/*/`` and, for each
ROI and depth metric, saves:

1.  Coverage vs depth scatter (``_1_evolution_plot.jpg``).
1b. Depth vs coverage with a logistic fit, median, and 80 % prediction
    band (all images).
1c. As 1b using only well-segmented images (needs ``--labels``).
2.  Peak coverage vs peak depth per event.
3.  CDF, 4. PDF, 5. exceedance probability of peak depth per event.
6.  Image vs in-situ sensor depth, 1:1 (needs ``--labels`` and
    ``--sensor``).
7.  Image vs in-situ sensor WSE, 1:1 (same requirements).

Fits (1b, 1c)
-------------
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
* The share of images inside the band is checked overall and in
  coverage bins (``*_band_calibration.csv``).
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
from collections import defaultdict

import cmocean
import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.optimize import least_squares, minimize  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from poseidon_utils.constants import FEET_TO_METERS  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.naming import extract_timestamp  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402

logger = logging.getLogger(__name__)

#: Depth metrics plotted from the accessibility CSVs.
METRICS = ["MaxDepth", "MeanDepth", "MedianDepth"]

#: Human-readable metric names for axis labels.
PRETTY_METRICS = {"MaxDepth": "Maximum Depth", "MeanDepth": "Mean Depth",
                  "MedianDepth": "Median Depth"}

#: Quantiles fitted for the prediction band and median line.
QUANTILES = (0.1, 0.5, 0.9)

#: Okabe-Ito colors used for reference lines.
OI_BLACK = "#000000"
OI_ORANGE = "#E69F00"
OI_VERMILION = "#D55E00"

#: "Coastal Convergence" palette (light/base/dark shades).
COASTAL_COLORS = {
    "oceanic_blue": {"light": "#7DA6C6", "base": "#1E6091",
                     "dark": "#0F3048"},
    "estuary_teal": {"light": "#81CFC6", "base": "#2A9D8F",
                     "dark": "#154F48"},
    "erosion_rust": {"light": "#F3B6A7", "base": "#E76F51",
                     "dark": "#8E3621"},
    "dune_gold": {"light": "#F4E1B3", "base": "#E9C46A", "dark": "#9A7822"},
    "concrete_slate": {"light": "#A3AEB4", "base": "#5C6B73",
                       "dark": "#2D3539"},
}

#: Legend proxies for good (green) and other (blue) segmentations.
SEGMENTATION_HANDLES = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
           markersize=8) for color in ("green", "blue")
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
        ``PointColor`` (green = good segmentation) and, when labels
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
        event["PointColor"] = "blue"
        event["SensorDepth_m"] = np.nan
        return event

    event["datetime_utc"] = pd.to_datetime(event["Time"], errors="coerce",
                                           utc=True)
    stamps = event["datetime_utc"].dt.strftime("%Y%m%d%H%M%S")
    event["PointColor"] = np.where(stamps.isin(good_timestamps), "green",
                                   "blue")
    if sensor is not None:
        columns = [c for c in ("datetime_utc", "SensorDepth_m",
                               "SensorWSE_m") if c in sensor]
        event = pd.merge_asof(event.sort_values("datetime_utc"),
                              sensor[columns], on="datetime_utc",
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
        Use only well-segmented (green) images. Default False.

    Returns
    -------
    dict
        ``Coverage_max`` and ``<metric>_max`` for each metric present
        (NaN if no image qualifies), and ``images`` used.
    """
    rows = event[event["PointColor"] == "green"] if good_only else event
    summary = {"Coverage_max": rows["Coverage"].max(), "images": len(rows)}
    for metric in METRICS:
        if metric in rows:
            summary[f"{metric}_max"] = rows[metric].max()
    return summary


# ----- Plots -----------------------------------------------------------------

def _save(fig, out_dir, filename):
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
    """
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
                              "below": below})
        grouped = table.groupby("bin", observed=False)
        result["calibration"] = pd.DataFrame({
            "coverage_bin_pct": [str(b) for b in grouped.size().index],
            "images": grouped.size().to_numpy(),
            "fraction_inside_80pct_band": grouped["inside"].mean().to_numpy(),
            "fraction_at_or_below_median": grouped["below"].mean().to_numpy(),
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
    scatter = ax.scatter(sens, img, c=cov, s=sizes, cmap=cmocean.cm.deep,
                         alpha=0.8, edgecolors="k", linewidth=0.5)

    lo = min(img.min(), sens.min()) - 0.05
    hi = max(img.max(), sens.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=OI_BLACK, linestyle="--", linewidth=2,
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
                       sensor_wse_floor, out_dir, n_boot=0, seed=0):
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
    """
    pretty = PRETTY_METRICS[metric]
    stem = f"{roi}_{metric}"

    # 1: coverage vs depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(data[metric], data["Coverage"], alpha=0.5,
               c=data["PointColor"])
    ax.set_xlabel(f"{metric} on Roadway (m)")
    ax.set_ylabel("Roadway Percentage Covered (%)")
    ax.set_title(f"Roadway Coverage vs {metric}\n({roi})")
    ax.grid(True, linestyle="--", alpha=0.7)
    if has_labels:
        ax.legend(SEGMENTATION_HANDLES,
                  ["Good Segmentation", "Standard/Poor Segmentation"])
    _save(fig, out_dir, f"{stem}_1_evolution_plot.jpg")

    # 1b: depth vs coverage with logistic fit (all data)
    fit_data = data.dropna(subset=["Coverage", metric])
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(data["Coverage"], data[metric], alpha=0.5,
               c=data["PointColor"])
    if len(fit_data) > 3:
        try:
            fit = fit_depth_from_coverage(fit_data["Coverage"],
                                          fit_data[metric],
                                          fit_data["event"], n_boot, seed)
            _draw_depth_fit(ax, fit, line_color="black", band_color="red",
                            band_alpha=0.15, median_color="purple",
                            ci_color="gray")
            write_fit_summary(fit, out_dir, f"{stem}_1b")
        except (RuntimeError, ValueError) as err:
            logger.warning("Logistic fit failed for %s (all): %s", stem, err)
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

    # 1c: as 1b, well-segmented images only
    good = fit_data[fit_data["PointColor"] == "green"]
    if len(good) > 3:
        _plot_good_only_fit(good, metric, pretty, out_dir,
                            f"{stem}_1c_reversed_fit_good_only.jpg",
                            f"{stem}_1c", n_boot, seed)
    elif has_labels:
        logger.info("Too few well-segmented points for plot 1c (%s).", stem)

    if sensor is None:
        return
    green = data[data["PointColor"] == "green"]

    # 6: image vs sensor depth
    if "SensorDepth_m" in data:
        rows = green.dropna(subset=[metric, "SensorDepth_m", "Coverage"])
        if not rows.empty:
            plot_one_to_one(rows["SensorDepth_m"].values, rows[metric].values,
                            rows["Coverage"].values,
                            "In-Situ Sensor Depth (m)",
                            f"Image Derived {pretty} (m)", OI_VERMILION,
                            out_dir, f"{stem}_6_sensor_1to1.jpg")

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
                            f" (m)", OI_ORANGE, out_dir,
                            f"{roi}_{wse_metric}_7_sensor_1to1.jpg")


def _draw_depth_fit(ax, fit, line_color, band_color, band_alpha,
                    median_color, ci_color):
    """
    Draw a depth-from-coverage fit with its bands.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes with coverage on x and depth on y.
    fit : dict
        Output of :func:`fit_depth_from_coverage`.
    line_color, band_color, median_color, ci_color : str
        Colors of the fit line, 80 % band, median, and confidence band.
    band_alpha : float
        Opacity of the 80 % band.
    """
    c, curves = fit["c"], fit["quantiles"]
    if 0.1 in curves and 0.9 in curves:
        inside = fit["band_fraction"]
        note = "" if np.isnan(inside) else f", {100 * inside:.0f}% of images"
        ax.fill_between(c, curves[0.1], curves[0.9], color=band_color,
                        alpha=band_alpha,
                        label=f"80% Prediction Band (0.1-0.9 quantiles{note})")
    if fit["curve_ci"] is not None:
        ax.fill_between(c, *fit["curve_ci"], color=ci_color, alpha=0.45,
                        label=f"95% CI of Fit ({fit['n_boot']} event "
                              f"resamples)")
    ax.plot(c, fit["fit"], color=line_color, linewidth=2.5,
            label=f"Logistic Fit ($R^2$={fit['r2']:.2f}, "
                  f"RMSE={fit['rmse']:.3f} m)")
    if 0.5 in curves:
        ax.plot(c, curves[0.5], linestyle="--", color=median_color,
                linewidth=2, label="Median Depth (0.5 Quantile)")


def _plot_good_only_fit(good, metric, pretty, out_dir, filename, stem,
                        n_boot=0, seed=0):
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
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(good["Coverage"], good[metric], alpha=0.6,
               c=COASTAL_COLORS["concrete_slate"]["light"],
               edgecolors="none", label="Segmentation Data")
    try:
        fit = fit_depth_from_coverage(good["Coverage"], good[metric],
                                      good["event"], n_boot, seed)
        _draw_depth_fit(ax, fit,
                        line_color=COASTAL_COLORS["oceanic_blue"]["dark"],
                        band_color=COASTAL_COLORS["estuary_teal"]["light"],
                        band_alpha=0.35,
                        median_color=COASTAL_COLORS["estuary_teal"]["base"],
                        ci_color=COASTAL_COLORS["oceanic_blue"]["light"])
        write_fit_summary(fit, out_dir, stem)
    except (RuntimeError, ValueError) as err:
        logger.warning("Logistic fit failed for %s: %s", filename, err)

    ax.set_xlabel("Roadway Percentage Covered (%)")
    ax.set_ylabel(f"{pretty} on Roadway (m)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # drop duplicate legend entries
    ax.legend(unique.values(), unique.keys())
    _save(fig, out_dir, filename)


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
    """
    column = f"{metric}_max"
    if column not in peaks:
        return
    valid = peaks.dropna(subset=[column, "Coverage_max"])
    values = valid[column].to_numpy()
    if values.size == 0:
        return
    stem = f"{roi}_{metric}"
    label = f"Peak Event {metric} (m)"
    subtitle = f"({roi}; {len(values)} events, {basis})"

    # 2: peak coverage vs peak depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(values, valid["Coverage_max"],
               c=COASTAL_COLORS["oceanic_blue"]["base"], alpha=0.7,
               edgecolors="black", s=40)
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
            (cdf, "post", "blue", "bo",
             "Cumulative Probability P(X <= x)", "CDF of Peak",
             "3_depth_cdf"),
            (exceedance, "pre", "red", "ro",
             "Exceedance Probability P(X >= x)",
             "Exceedance Probability of Peak", "5_depth_exceedance")):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.step(ordered, probs, where=where, color=color, linewidth=2)
        ax.plot(ordered, probs, marker, alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel(f"{y_label} [{position}]")
        ax.set_title(f"{title} {metric}\n{subtitle}")
        ax.grid(True, linestyle="--", alpha=0.7)
        _save(fig, out_dir, f"{stem}_{suffix}.png")

    # 4: PDF as a density histogram and a boundary-corrected KDE
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    edges, rule = density_bins(values)
    ax.hist(values, bins=edges, density=True, color="skyblue",
            edgecolor="black", alpha=0.9,
            label=f"Density Histogram ({len(edges) - 1} {rule} bins)")
    kde = reflected_kde(values)
    if kde is not None:
        ax.plot(*kde, color=COASTAL_COLORS["oceanic_blue"]["dark"],
                linewidth=2, label="Gaussian KDE (Scott bandwidth, "
                                   "reflected at 0 m)")
    ax.set_xlabel(label)
    ax.set_ylabel("Probability Density (1/m)")
    ax.set_title(f"PDF of Peak {metric}\n{subtitle}")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    _save(fig, out_dir, f"{stem}_4_depth_pdf.png")


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
    args = parse_args(argv)
    if args.sensor_wse_floor is None and args.site:
        args.sensor_wse_floor = load_site_config(args.site).sensor_wse_floor
    os.makedirs(args.out_dir, exist_ok=True)

    good_timestamps = load_good_timestamps(args.labels)
    sensor = load_sensor_data(args.sensor)
    has_labels = bool(good_timestamps)

    basis = "well-segmented images" if has_labels else "all images"
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
        for metric in METRICS:
            if metric not in data:
                logger.warning("No '%s' column for %s; skipped.", metric,
                               roi)
                continue
            logger.info("Plotting %s - %s", roi, metric)
            plot_point_metrics(data, roi, metric, has_labels, sensor,
                               args.sensor_wse_floor, args.out_dir,
                               args.n_boot, args.seed)
            plot_event_peaks(peaks, roi, metric, basis, args.out_dir,
                             args.plotting_position)
    logger.info("Plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
