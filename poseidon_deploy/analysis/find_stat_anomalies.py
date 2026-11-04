"""Flag images whose roadway coverage is unusually low for their depth.

For each ROI, every image's (MaxDepth, Coverage) pair from all events
is compared with a robust depth-coverage "core" curve: the 75th
percentile of coverage in 1 cm depth bins, smoothed and forced to be
non-decreasing. Images whose coverage falls more than
``--z_threshold`` local standard deviations below the curve are
written to ``<roi>_anomaly_report.csv`` with a diagnostic plot
``<roi>_local_anomaly_diagnostic.jpg``.

Example
-------
``python find_stat_anomalies.py --events_dir flood_events``
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from poseidon_utils.logging_utils import setup_logging  # noqa: E402

from access_plotter import find_roi_csvs  # noqa: E402

logger = logging.getLogger(__name__)

#: Only images deeper than this (m) shape the core curve.
MIN_FIT_DEPTH = 0.05

#: Rolling window (in depth bins) used to smooth the core curve.
SMOOTHING_WINDOW = 10

#: Floor on the local coverage standard deviation (percentage points).
MIN_LOCAL_STD = 5.0


def load_roi_points(paths):
    """
    Combine (MaxDepth, Coverage) points from an ROI's event CSVs.

    Parameters
    ----------
    paths : list of str
        Accessibility CSV paths for one ROI.

    Returns
    -------
    pandas.DataFrame
        Columns ``Event_Folder``, ``Time``, ``MaxDepth``, ``Coverage``.
    """
    frames = []
    for path in paths:
        event = pd.read_csv(path)
        event["Coverage"] = event["FloodedPixels"] / event["TotalPixels"] * 100
        event["Event_Folder"] = os.path.basename(os.path.dirname(path))
        frames.append(event[["Event_Folder", "Time", "MaxDepth", "Coverage"]])
    return pd.concat(frames, ignore_index=True)


def core_curve(points):
    """
    Build the smoothed, non-decreasing depth-coverage core curve.

    Parameters
    ----------
    points : pandas.DataFrame
        ``MaxDepth`` and ``Coverage`` of images deeper than
        :data:`MIN_FIT_DEPTH`.

    Returns
    -------
    pandas.DataFrame
        Per 1 cm bin: ``Depth_Bin``, ``Smoothed_Core`` (coverage %) and
        ``Smoothed_Std`` (local standard deviation).
    """
    binned = points.assign(
        Depth_Bin=(points["MaxDepth"] * 100).astype(int) / 100)
    curve = binned.groupby("Depth_Bin").agg(
        Core_Coverage=("Coverage", lambda c: c.quantile(0.75)),
        Local_Std=("Coverage", "std")).reset_index()

    rolling = {"window": SMOOTHING_WINDOW, "center": True, "min_periods": 1}
    # Coverage should never fall as depth rises: enforce a plateau
    curve["Smoothed_Core"] = (curve["Core_Coverage"].rolling(**rolling)
                              .mean().cummax())
    curve["Local_Std"] = (curve["Local_Std"].bfill().ffill()
                          .clip(lower=MIN_LOCAL_STD))
    curve["Smoothed_Std"] = curve["Local_Std"].rolling(**rolling).mean()
    return curve


def score_points(points, curve):
    """
    Add expected coverage, residual, and local z-score to each point.

    Parameters
    ----------
    points : pandas.DataFrame
        All ROI points.
    curve : pandas.DataFrame
        Output of :func:`core_curve`.

    Returns
    -------
    pandas.DataFrame
        Copy with ``Expected_Coverage``, ``Local_Std``, ``Residual``
        and ``Local_Z`` columns.
    """
    scored = points.copy()
    scored["Expected_Coverage"] = np.interp(
        scored["MaxDepth"], curve["Depth_Bin"], curve["Smoothed_Core"])
    scored["Local_Std"] = np.interp(
        scored["MaxDepth"], curve["Depth_Bin"], curve["Smoothed_Std"])
    scored["Residual"] = scored["Coverage"] - scored["Expected_Coverage"]
    scored["Local_Z"] = scored["Residual"] / scored["Local_Std"]
    return scored


def plot_diagnostic(scored, anomalies, curve, roi, z_threshold, path):
    """
    Save a scatter of normal and anomalous points with the core curve.

    Parameters
    ----------
    scored : pandas.DataFrame
        All scored points.
    anomalies : pandas.DataFrame
        Flagged points (a subset of `scored`).
    curve : pandas.DataFrame
        Core curve from :func:`core_curve`.
    roi : str
        ROI name for the title.
    z_threshold : float
        Threshold shown in the legend.
    path : str
        Output image path.
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    normal = scored[~scored.index.isin(anomalies.index)]
    ax.scatter(normal["MaxDepth"], normal["Coverage"], color="gray",
               alpha=0.3, label="Normal Data")
    if not anomalies.empty:
        ax.scatter(anomalies["MaxDepth"], anomalies["Coverage"], color="red",
                   alpha=0.8, edgecolor="black",
                   label=f"Anomalies (Local Z < {z_threshold})")
    ax.plot(curve["Depth_Bin"], curve["Smoothed_Core"], color="blue",
            linewidth=3, label="Visual Core Tracker (75th Pct + CumMax)")
    ax.set_xlabel("Maximum Event Depth on Roadway (m)")
    ax.set_ylabel("Roadway Percentage Covered (%)")
    ax.set_title(f"Local Density Anomaly Detection: {roi}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def find_anomalies(roi, paths, z_threshold, out_dir):
    """
    Score one ROI's images and write its report and diagnostic plot.

    Parameters
    ----------
    roi : str
        ROI name.
    paths : list of str
        Accessibility CSV paths for the ROI.
    z_threshold : float
        Points with ``Local_Z`` below this are anomalies.
    out_dir : str
        Output folder.

    Returns
    -------
    pandas.DataFrame or None
        The anomalies, or None if no points exceed
        :data:`MIN_FIT_DEPTH`.
    """
    points = load_roi_points(paths)
    deep = points[points["MaxDepth"] > MIN_FIT_DEPTH]
    if deep.empty:
        logger.info("%s: no points deeper than %.2f m; skipped.", roi,
                    MIN_FIT_DEPTH)
        return None

    curve = core_curve(deep)
    scored = score_points(points, curve)
    anomalies = scored[scored["Local_Z"] < z_threshold].sort_values(
        "Local_Z")

    if anomalies.empty:
        logger.info("%s: no anomalies found.", roi)
    else:
        report = os.path.join(out_dir, f"{roi}_anomaly_report.csv")
        anomalies[["Event_Folder", "Time", "MaxDepth", "Coverage",
                   "Expected_Coverage", "Local_Z"]].round(3).to_csv(
            report, index=False)
        logger.info("%s: %d anomalies -> %s", roi, len(anomalies), report)

    plot_diagnostic(scored, anomalies, curve, roi, z_threshold,
                    os.path.join(out_dir,
                                 f"{roi}_local_anomaly_diagnostic.jpg"))
    return anomalies


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
        description="Flag low-coverage outliers in accessibility data.")
    parser.add_argument("--events_dir", default="flood_events",
                        help="Folder of event subfolders "
                             "(default: flood_events).")
    parser.add_argument("--out_dir", default=".",
                        help="Folder for reports and plots (default: .).")
    parser.add_argument("--z_threshold", type=float, default=-1.0,
                        help="Local z-score below which a point is "
                             "flagged (default: -1.0).")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Run anomaly detection for every ROI.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    setup_logging()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    groups = find_roi_csvs(args.events_dir)
    logger.info("Found %d ROIs in %s", len(groups), args.events_dir)
    for roi, paths in groups.items():
        find_anomalies(roi, paths, args.z_threshold, args.out_dir)


if __name__ == "__main__":
    main()
