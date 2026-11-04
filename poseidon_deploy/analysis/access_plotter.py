"""Plot roadway accessibility statistics across all flood events.

Reads every ``<roi>_accessibility_time_series.csv`` (and the matching
``<roi>_wse_time_series.csv``) under ``<events_dir>/*/`` and, for each
ROI and depth metric, saves:

1.  Coverage vs depth scatter (``_1_evolution_plot.jpg``).
1b. Depth vs coverage with a logistic fit and quantile band.
1c. As 1b using only well-segmented images (needs ``--labels``).
2.  Peak coverage vs peak depth per event.
3.  CDF, 4. PDF, 5. exceedance probability of peak depth per event.
6.  Image vs in-situ sensor depth, 1:1 (needs ``--labels`` and
    ``--sensor``).
7.  Image vs in-situ sensor WSE, 1:1 (same requirements).

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
from scipy.optimize import curve_fit, minimize  # noqa: E402

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


def logit_curve(c, L, k, x0):  # noqa: N803 - conventional symbols
    """
    Invert :func:`logistic_curve` to predict depth from coverage.

    Parameters
    ----------
    c : array_like
        Coverage values.
    L, k, x0 : float
        Logistic parameters.

    Returns
    -------
    numpy.ndarray
        Predicted depth.
    """
    epsilon = 1e-5
    l_eff = np.maximum(L, np.max(c) + epsilon)  # keep the log finite
    c_clipped = np.clip(c, epsilon, l_eff - epsilon)
    return x0 - (1 / k) * np.log((l_eff / c_clipped) - 1)


def pinball_loss(params, x, y, q):
    """
    Quantile (pinball) loss of a logistic fit.

    Parameters
    ----------
    params : sequence of float
        Logistic parameters ``(L, k, x0)``.
    x, y : numpy.ndarray
        Depths and observed coverages.
    q : float
        Quantile in (0, 1).

    Returns
    -------
    float
        Summed pinball loss.
    """
    err = y - logistic_curve(x, *params)
    return np.sum(np.maximum(q * err, (q - 1) * err))


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
        logger.info("No segmentation labels; using default colors.")
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
        ``sensor_water_level_adj`` (feet).

    Returns
    -------
    pandas.DataFrame or None
        ``datetime_utc``, ``SensorDepth_m`` (negative depths set to 0)
        and, if available, ``SensorWSE_m``, sorted by time.
    """
    if not sensor_csv or not os.path.exists(sensor_csv):
        return None
    sensor = pd.read_csv(sensor_csv)
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


def summarize_event(event):
    """
    Return an event's peak coverage, peak metrics, and color.

    Parameters
    ----------
    event : pandas.DataFrame
        Output of :func:`load_event`.

    Returns
    -------
    dict
        ``Coverage_max``, ``<metric>_max`` for each metric present, and
        ``EventColor`` (green if any image was well segmented).
    """
    summary = {
        "Coverage_max": event["Coverage"].max(),
        "EventColor": ("green" if (event["PointColor"] == "green").any()
                       else "blue"),
    }
    for metric in METRICS:
        if metric in event:
            summary[f"{metric}_max"] = event[metric].max()
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


def fit_logistic_with_quantiles(depth, coverage):
    """
    Fit coverage vs depth with a logistic curve and quantile curves.

    Parameters
    ----------
    depth, coverage : pandas.Series
        Paired observations.

    Returns
    -------
    dict
        ``x`` (200 depths), ``fit`` (fitted coverage), ``r2``
        (coverage space), ``rmse`` (depth space, meters) and
        ``quantiles`` mapping each quantile to its coverage curve.

    Raises
    ------
    RuntimeError
        If the least-squares fit does not converge.
    """
    x = np.linspace(depth.min(), depth.max(), 200)
    p0 = [coverage.max(), 10, depth.median()]
    params, _ = curve_fit(logistic_curve, depth, coverage, p0=p0,
                          maxfev=5000)

    result = {
        "x": x,
        "fit": logistic_curve(x, *params),
        "r2": calculate_r2(coverage, logistic_curve(depth, *params)),
        "rmse": calculate_rmse(depth, logit_curve(coverage, *params)),
        "quantiles": {},
    }
    for q in QUANTILES:
        res = minimize(pinball_loss, params,
                       args=(depth.values, coverage.values, q),
                       method="Nelder-Mead")
        if res.success:
            result["quantiles"][q] = logistic_curve(x, *res.x)
    return result


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
                       sensor_wse_floor, out_dir):
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
            fit = fit_logistic_with_quantiles(fit_data[metric],
                                              fit_data["Coverage"])
            ax.plot(fit["fit"], fit["x"], color="black", linewidth=2,
                    label=f"Logistic Fit ($R^2$={fit['r2']:.2f}, "
                          f"RMSE={fit['rmse']:.2f}m)")
            _draw_quantiles(ax, fit, band_color="red", band_alpha=0.15,
                            median_color="purple")
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
                            f"{stem}_1c_reversed_fit_good_only.jpg")
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


def _draw_quantiles(ax, fit, band_color, band_alpha, median_color):
    """
    Draw the 80 % quantile band and median curve of a fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    fit : dict
        Output of :func:`fit_logistic_with_quantiles`.
    band_color : str
        Fill color of the band.
    band_alpha : float
        Fill opacity.
    median_color : str
        Color of the median curve.
    """
    curves = fit["quantiles"]
    if 0.1 in curves and 0.9 in curves:
        ax.fill_betweenx(fit["x"], curves[0.1], curves[0.9],
                         color=band_color, alpha=band_alpha,
                         label="80% Prediction Interval")
    if 0.5 in curves:
        ax.plot(curves[0.5], fit["x"], linestyle="--", color=median_color,
                linewidth=2, label="Median (0.5 Quantile)")


def _plot_good_only_fit(good, metric, pretty, out_dir, filename):
    """
    Draw plot 1c: logistic fit of well-segmented images only.

    Parameters
    ----------
    good : pandas.DataFrame
        Well-segmented rows with `metric` and ``Coverage``.
    metric : str
        Depth metric column.
    pretty : str
        Human-readable metric name for the y-axis.
    out_dir : str
        Output folder.
    filename : str
        Output filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(good["Coverage"], good[metric], alpha=0.6,
               c=COASTAL_COLORS["concrete_slate"]["light"],
               edgecolors="none", label="Segmentation Data")
    try:
        fit = fit_logistic_with_quantiles(good[metric], good["Coverage"])
        ax.plot(fit["fit"], fit["x"],
                color=COASTAL_COLORS["oceanic_blue"]["dark"], linewidth=2.5,
                label=f"Logistic Fit ($R^2$={fit['r2']:.2f}, "
                      f"RMSE={fit['rmse']:.2f}m)")
        _draw_quantiles(ax, fit,
                        band_color=COASTAL_COLORS["estuary_teal"]["light"],
                        band_alpha=0.35,
                        median_color=COASTAL_COLORS["estuary_teal"]["base"])
    except (RuntimeError, ValueError) as err:
        logger.warning("Logistic fit failed for %s: %s", filename, err)

    ax.set_xlabel("Roadway Percentage Covered (%)")
    ax.set_ylabel(f"{pretty} on Roadway (m)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # drop duplicate legend entries
    ax.legend(unique.values(), unique.keys())
    _save(fig, out_dir, filename)


def plot_event_peaks(peaks, roi, metric, has_labels, out_dir):
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
    has_labels : bool
        Whether segmentation labels were supplied.
    out_dir : str
        Output folder.
    """
    column = f"{metric}_max"
    if column not in peaks:
        return
    valid = peaks.dropna(subset=[column, "Coverage_max"])
    values = valid[column].to_numpy()
    if values.size == 0:
        return
    ordered = np.sort(values)
    n = len(ordered)
    stem = f"{roi}_{metric}"
    label = f"Peak Event {metric} (m)"

    # 2: peak coverage vs peak depth
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.scatter(values, valid["Coverage_max"], c=valid["EventColor"],
               alpha=0.6, edgecolors="black", s=40)
    ax.set_xlabel(label)
    ax.set_ylabel("Peak Roadway Percentage Covered (%)")
    ax.set_title(f"Peak Coverage vs Peak {metric} per Event\n({roi})")
    ax.grid(True, linestyle="--", alpha=0.7)
    if has_labels:
        ax.legend(SEGMENTATION_HANDLES, ["Event Contains Good Segmentation",
                                         "Standard/Poor Event"])
    _save(fig, out_dir, f"{stem}_2_summary_scatter.png")

    # 3: CDF and 5: exceedance probability
    for probs, color, marker, y_label, title, suffix in (
            (np.arange(1, n + 1) / n, "blue", "bo",
             "Cumulative Probability P(X <= x)", "CDF of Peak",
             "3_depth_cdf"),
            (np.arange(n, 0, -1) / n, "red", "ro",
             "Exceedance Probability P(X >= x)",
             "Exceedance Probability of Peak", "5_depth_exceedance")):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.step(ordered, probs, where="post", color=color, linewidth=2)
        ax.plot(ordered, probs, marker, alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} {metric}\n({roi})")
        ax.grid(True, linestyle="--", alpha=0.7)
        _save(fig, out_dir, f"{stem}_{suffix}.png")

    # 4: PDF
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.hist(values, bins=5, density=True, color="skyblue",
            edgecolor="black", alpha=0.9)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(f"PDF of Peak {metric}\n({roi})")
    ax.grid(True, linestyle="--", alpha=0.7)
    _save(fig, out_dir, f"{stem}_4_depth_pdf.png")


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

    for roi, paths in find_roi_csvs(args.events_dir).items():
        events = [load_event(path, good_timestamps, sensor)
                  for path in paths]
        data = pd.concat(events, ignore_index=True)
        peaks = pd.DataFrame([summarize_event(event) for event in events])
        for metric in METRICS:
            if metric not in data:
                logger.warning("No '%s' column for %s; skipped.", metric,
                               roi)
                continue
            logger.info("Plotting %s - %s", roi, metric)
            plot_point_metrics(data, roi, metric, has_labels, sensor,
                               args.sensor_wse_floor, args.out_dir)
            plot_event_peaks(peaks, roi, metric, has_labels, args.out_dir)
    logger.info("Plots saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
