"""Compare roadway statistics from two ROIs over the same flood events.

For example, compare a polygon drawn on a rectified (plan view) image
with one drawn on the camera image and rectified (stage 10 with
``--roi_view camera``). Reads ``<roi>_accessibility_time_series.csv``
for both ROIs from every event folder that has both, pairs them by
image time, and writes to ``--output_dir``:

* ``roi_comparison_timesteps.csv`` - one row per image: impassable
  flag, coverage (% of ROI cells flooded), and mean/max depth for each
  ROI, and the differences (``roi_b - roi_a``).
* ``roi_comparison_events.csv`` - one row per event: impassable image
  counts and agreement, peak coverage, and peak depth for each ROI.
* ``roi_comparison_summary.csv`` - agreement of the impassable flag and
  bias, mean absolute difference, RMSE, and correlation of each
  quantity, plus the overlap of the two ROI footprints.
* ``roi_footprints.png`` - both footprints over a rectified image
  (needs the ``<roi>_roi_mask.png`` files written by stage 10).
* ``roi_comparison_<quantity>.png`` - roi_a vs roi_b scatter plots.

Depths are NaN in the CSVs when no ROI cell is wet; they are compared
as 0 m here.

Example
-------
``python compare_roadway_rois.py --events_dir flood_events --roi_a
shell_hill_rd_polygon --roi_b shell_hill_rd_camera_polygon
--output_dir roi_comparison``

or, taking ``roi_json`` (a) and ``camera_roi_json`` (b) from a site
config: ``python compare_roadway_rois.py --site down_east --events_dir
flood_events --output_dir roi_comparison``
"""

import argparse
import glob
import logging
import os

import cv2
import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402
from poseidon_utils.zarr_io import array_names, open_zip_group  # noqa: E402

from plot_style import BLUE, INK, MAGENTA, ORANGE, apply_style  # noqa: E402

logger = logging.getLogger(__name__)

#: Accessibility CSV suffix written by stage 10.
ACCESS_SUFFIX = "_accessibility_time_series.csv"

#: Per-image quantities compared between the ROIs.
QUANTITIES = ("Coverage", "MeanDepth", "MaxDepth")


def load_access(path):
    """
    Read one accessibility CSV with coverage and dry depths as 0.

    Parameters
    ----------
    path : str
        Accessibility CSV.

    Returns
    -------
    pandas.DataFrame
        ``Time``, ``Impassable`` and the :data:`QUANTITIES` columns.
    """
    table = pd.read_csv(path, parse_dates=["Time"])
    table["Coverage"] = 100.0 * table["FloodedPixels"] / table["TotalPixels"]
    for column in ("MeanDepth", "MaxDepth"):
        table[column] = table[column].fillna(0.0)
    return table[["Time", "Impassable", *QUANTITIES]]


def pair_event(event_dir, roi_a, roi_b):
    """
    Pair both ROIs' accessibility tables for one event by image time.

    Parameters
    ----------
    event_dir : str
        Event folder.
    roi_a, roi_b : str
        ROI names.

    Returns
    -------
    pandas.DataFrame or None
        One row per image in both tables, columns suffixed ``_a`` and
        ``_b``, or None if either CSV is missing.
    """
    paths = [os.path.join(event_dir, f"{roi}{ACCESS_SUFFIX}")
             for roi in (roi_a, roi_b)]
    if not all(os.path.exists(p) for p in paths):
        return None
    paired = load_access(paths[0]).merge(load_access(paths[1]), on="Time",
                                         suffixes=("_a", "_b"))
    paired.insert(0, "event", os.path.basename(event_dir))
    for column in QUANTITIES:
        paired[f"{column}_diff"] = paired[f"{column}_b"] - paired[
            f"{column}_a"]
    return paired


def event_table(timesteps):
    """
    Summarize the paired images of each event.

    Parameters
    ----------
    timesteps : pandas.DataFrame
        Output of :func:`pair_event` for every event.

    Returns
    -------
    pandas.DataFrame
        One row per event.
    """
    agree = 100.0 * (timesteps["Impassable_a"] == timesteps["Impassable_b"])
    grouped = timesteps.assign(agree=agree).groupby("event")
    return pd.DataFrame({
        "images": grouped.size(),
        "impassable_images_a": grouped["Impassable_a"].sum(),
        "impassable_images_b": grouped["Impassable_b"].sum(),
        "impassable_agreement_pct": grouped["agree"].mean(),
        "peak_coverage_a": grouped["Coverage_a"].max(),
        "peak_coverage_b": grouped["Coverage_b"].max(),
        "peak_max_depth_a": grouped["MaxDepth_a"].max(),
        "peak_max_depth_b": grouped["MaxDepth_b"].max(),
    }).reset_index()


def difference_metrics(a, b):
    """
    Compare two paired series.

    Parameters
    ----------
    a, b : pandas.Series
        Values for ROI a and ROI b.

    Returns
    -------
    dict
        ``bias`` (mean of b - a), ``mae``, ``rmse`` and Pearson ``r``
        (NaN if either series is constant).
    """
    diff = b - a
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a.corr(b) if len(a) > 1 else np.nan
    return {"bias": diff.mean(), "mae": diff.abs().mean(),
            "rmse": float(np.sqrt((diff ** 2).mean())), "r": r}


def summary_table(timesteps, footprint):
    """
    Build the overall agreement table.

    Parameters
    ----------
    timesteps : pandas.DataFrame
        Paired images of every event.
    footprint : dict or None
        Output of :func:`footprint_overlap`.

    Returns
    -------
    pandas.DataFrame
        ``metric`` and ``value`` columns.
    """
    a, b = timesteps["Impassable_a"], timesteps["Impassable_b"]
    rows = {
        "events": timesteps["event"].nunique(),
        "images": len(timesteps),
        "impassable_agreement_pct": 100.0 * (a == b).mean(),
        "impassable_both": int(((a == 1) & (b == 1)).sum()),
        "impassable_a_only": int(((a == 1) & (b == 0)).sum()),
        "impassable_b_only": int(((a == 0) & (b == 1)).sum()),
    }
    for column in QUANTITIES:
        metrics = difference_metrics(timesteps[f"{column}_a"],
                                     timesteps[f"{column}_b"])
        rows.update({f"{column}_{k}": v for k, v in metrics.items()})
    if footprint:
        rows.update({f"footprint_{k}": v for k, v in footprint.items()})
    return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})


def find_masks(events_dir, roi_a, roi_b):
    """
    Find an event folder with both ROI footprint masks.

    Parameters
    ----------
    events_dir : str
        Folder of event subfolders.
    roi_a, roi_b : str
        ROI names.

    Returns
    -------
    tuple of str or None
        ``(event_dir, mask_a_path, mask_b_path)``, or None.
    """
    for event_dir in sorted(glob.glob(os.path.join(events_dir, "*", ""))):
        paths = [os.path.join(event_dir, f"{roi}_roi_mask.png")
                 for roi in (roi_a, roi_b)]
        if all(os.path.exists(p) for p in paths):
            return event_dir, *paths
    return None


def footprint_overlap(mask_a, mask_b):
    """
    Measure how much two ROI footprints overlap.

    Parameters
    ----------
    mask_a, mask_b : numpy.ndarray
        Boolean masks of the same shape.

    Returns
    -------
    dict
        Cell counts in each ROI, in both, in one only, and the
        intersection over union (IoU).
    """
    both = int((mask_a & mask_b).sum())
    union = int((mask_a | mask_b).sum())
    return {"cells_a": int(mask_a.sum()), "cells_b": int(mask_b.sum()),
            "cells_both": both,
            "cells_a_only": int((mask_a & ~mask_b).sum()),
            "cells_b_only": int((mask_b & ~mask_a).sum()),
            "iou": both / union if union else np.nan}


def rectified_background(event_dir):
    """
    Return the first rectified image of an event, north up.

    Parameters
    ----------
    event_dir : str
        Event folder with ``zarr/orig_image_rects.zip``.

    Returns
    -------
    numpy.ndarray or None
        RGB image, or None if the store is missing or empty.
    """
    path = os.path.join(event_dir, "zarr", "orig_image_rects.zip")
    if not os.path.exists(path):
        return None
    with open_zip_group(path) as group:
        names = array_names(group)
        if not names:
            return None
        return group[names[0]][:][::-1]  # grid row 0 is south


def plot_footprints(mask_a, mask_b, background, roi_a, roi_b, path):
    """
    Draw both ROI footprints, optionally over a rectified image.

    Parameters
    ----------
    mask_a, mask_b : numpy.ndarray
        Boolean north-up masks.
    background : numpy.ndarray or None
        North-up RGB image of the same shape.
    roi_a, roi_b : str
        ROI names for the legend.
    path : str
        Output PNG.
    """
    colors = (BLUE, ORANGE, MAGENTA)  # a only, b only, both
    overlay = np.zeros((*mask_a.shape, 4))
    for mask, color in zip((mask_a & ~mask_b, mask_b & ~mask_a,
                            mask_a & mask_b), colors):
        overlay[mask] = (*matplotlib.colors.to_rgb(color), 0.6)

    fig, ax = plt.subplots(figsize=(8, 8))
    if background is not None:
        ax.imshow(background)
    ax.imshow(overlay)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.75)
               for color in colors]
    ax.legend(handles, [f"{roi_a} only", f"{roi_b} only", "both"],
              loc="lower right")
    ax.set_axis_off()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scatter(timesteps, column, roi_a, roi_b, path):
    """
    Plot one quantity for ROI a against ROI b with a 1:1 line.

    Parameters
    ----------
    timesteps : pandas.DataFrame
        Paired images.
    column : str
        Quantity in :data:`QUANTITIES`.
    roi_a, roi_b : str
        ROI names for the axis labels.
    path : str
        Output PNG.
    """
    a, b = timesteps[f"{column}_a"], timesteps[f"{column}_b"]
    unit = "%" if column == "Coverage" else "m"
    top = float(np.nanmax([a.max(), b.max(), 0.0])) or 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(a, b, s=12, alpha=0.6, color=BLUE, edgecolors="none")
    ax.plot([0, top], [0, top], color=INK, linestyle="--", linewidth=1.5,
            label="1:1")
    ax.set_xlabel(f"{column} ({unit}), {roi_a}")
    ax.set_ylabel(f"{column} ({unit}), {roi_b}")
    ax.set_xlim(0, top * 1.05)
    ax.set_ylim(0, top * 1.05)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
        description="Compare roadway statistics from two ROIs.")
    parser.add_argument("--events_dir", required=True,
                        help="Folder of event subfolders.")
    parser.add_argument("--roi_a", default=None,
                        help="First ROI name (e.g. the plan-view JSON "
                             "stem). Default with --site: its roi_json.")
    parser.add_argument("--roi_b", default=None,
                        help="Second ROI name (e.g. the camera-view JSON "
                             "stem). Differences are roi_b - roi_a. "
                             "Default with --site: its camera_roi_json.")
    parser.add_argument("--site", default=None,
                        help="Site config supplying the default ROI names.")
    parser.add_argument("--output_dir", required=True,
                        help="Folder for the comparison outputs.")
    args = parser.parse_args(argv)
    if args.site:
        site = load_site_config(args.site)
        stem = lambda path: (os.path.splitext(os.path.basename(path))[0]
                             if path else None)  # noqa: E731
        args.roi_a = args.roi_a or stem(site.roi_json)
        args.roi_b = args.roi_b or stem(site.camera_roi_json)
    if not (args.roi_a and args.roi_b):
        parser.error("give --roi_a and --roi_b, or a --site with roi_json "
                     "and camera_roi_json")
    return args


def main(argv=None):
    """
    Compare two ROIs over every event that has results for both.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if no event has both ROIs, else 0.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    event_dirs = sorted(glob.glob(os.path.join(args.events_dir, "*", "")))
    paired = [pair_event(os.path.normpath(d), args.roi_a, args.roi_b)
              for d in event_dirs]
    paired = [table for table in paired if table is not None]
    if not paired:
        logger.error("No event in %s has accessibility CSVs for both '%s' "
                     "and '%s'.", args.events_dir, args.roi_a, args.roi_b)
        return 1
    timesteps = pd.concat(paired, ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)

    footprint = None
    masks = find_masks(args.events_dir, args.roi_a, args.roi_b)
    if masks:
        event_dir, path_a, path_b = masks
        mask_a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE) > 0
        mask_b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE) > 0
        footprint = footprint_overlap(mask_a, mask_b)
        plot_footprints(mask_a, mask_b, rectified_background(event_dir),
                        args.roi_a, args.roi_b,
                        os.path.join(args.output_dir, "roi_footprints.png"))
    else:
        logger.warning("No event has both ROI masks; skipping the footprint "
                       "comparison (rerun stage 10 to write them).")

    outputs = {
        "roi_comparison_timesteps.csv": timesteps,
        "roi_comparison_events.csv": event_table(timesteps),
        "roi_comparison_summary.csv": summary_table(timesteps, footprint),
    }
    for name, table in outputs.items():
        table.to_csv(os.path.join(args.output_dir, name), index=False)
    for column in QUANTITIES:
        plot_scatter(timesteps, column, args.roi_a, args.roi_b,
                     os.path.join(args.output_dir,
                                  f"roi_comparison_{column}.png"))

    summary = outputs["roi_comparison_summary.csv"]
    for metric, value in zip(summary["metric"], summary["value"]):
        logger.info("  %-32s %s", metric, value)
    logger.info("Compared %d images from %d events; outputs in %s",
                len(timesteps), timesteps["event"].nunique(),
                args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
