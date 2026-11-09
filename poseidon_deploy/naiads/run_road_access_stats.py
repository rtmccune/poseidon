"""Stage 11: summarize roadway accessibility across all flood events.

Reads every ``<event>/<roi>_accessibility_time_series.csv`` under
``--event_dir`` and reports impassable time, depth statistics, and
(for polygon ROIs) flooded area across events. Each image is assumed to
represent the time until the next image; the first image takes the
spacing of the second.

The ROI is ``--transect_name`` or, if omitted, the stem of the site
config's ``roi_json`` (``camera_roi_json`` with ``--roi_view camera``).

Example
-------
``python run_road_access_stats.py --event_dir flood_events
--site carolina_beach``
"""

import glob
import os

import numpy as np
import pandas as pd

import cli_common

logger = cli_common.logger


def event_statistics(csv_path, pixel_area_m2):
    """
    Compute one event's accessibility statistics.

    Parameters
    ----------
    csv_path : str
        Accessibility CSV from the roadway analyzer.
    pixel_area_m2 : float
        Ground area of one grid pixel in square meters.

    Returns
    -------
    dict
        Impassable minutes and proportion, depth statistics, and
        flooded-area statistics (NaN for line ROIs).
    """
    event = pd.read_csv(csv_path)
    event["Time"] = pd.to_datetime(event["Time"], utc=True)
    event = event.sort_values("Time")

    # Minutes each image represents: gap to the previous image, with
    # the first image borrowing the second image's gap
    minutes = event["Time"].diff().dt.total_seconds().div(60)
    minutes.iloc[0] = minutes.iloc[1] if len(minutes) > 1 else 0
    total = minutes.sum()
    impassable = minutes[event["Impassable"] == 1].sum()

    stats = {
        "impassable_minutes": impassable,
        "prop_impassable": impassable / total if total > 0 else 0,
        "mean_depth": event["MeanDepth"].mean(),
        "max_depth": event["MaxDepth"].max(),
        "min_depth": event["MinDepth"].min(),
        "max_flooded_area": np.nan,
        "mean_flooded_area": np.nan,
        "mean_percent_covered": np.nan,
    }
    if {"FloodedPixels", "TotalPixels"} <= set(event.columns):
        area = event["FloodedPixels"] * pixel_area_m2
        stats["max_flooded_area"] = area.max()
        stats["mean_flooded_area"] = area.mean()
        stats["mean_percent_covered"] = (event["FloodedPixels"]
                                         / event["TotalPixels"]).mean()
    return stats


def summarize_roadway_stats(main_dir, transect_name, resolution=0.05):
    """
    Aggregate accessibility statistics over every event.

    Parameters
    ----------
    main_dir : str
        Folder of event subfolders.
    transect_name : str
        ROI name (the LabelMe JSON filename without extension).
    resolution : float, optional
        Grid spacing in meters. Default is 0.05.

    Returns
    -------
    pandas.DataFrame or None
        One-row summary, or None if no event CSVs were found.
    """
    pattern = os.path.join(main_dir, "*",
                           f"{transect_name}_accessibility_time_series.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        logger.error("No accessibility CSVs for '%s' in %s", transect_name,
                     main_dir)
        return None

    per_event = []
    for path in paths:
        try:
            per_event.append(event_statistics(path, resolution**2))
        except (KeyError, ValueError) as err:
            logger.error("Skipping %s: %s", path, err)
    stats = pd.DataFrame(per_event)
    if stats.empty:
        logger.error("No readable event CSVs.")
        return None

    minutes = stats["impassable_minutes"]
    summary = {
        "Total Events Processed": len(stats),
        "Total Impassable Time (Minutes)": minutes.sum(),
        "Total Impassable Time (Hours)": minutes.sum() / 60,
        "Mean Impassable Time per Event (Minutes)": minutes.mean(),
        "Mean Impassable Time per Event (Hours)": minutes.mean() / 60,
        "Median Impassable Time (Minutes)": minutes.median(),
        "Median Impassable Time (Hours)": minutes.median() / 60,
        "Avg Proportion of Time Impassable": stats["prop_impassable"].mean(),
        "Absolute Maximum Depth on Road (m)": stats["max_depth"].max(),
        "Mean of Mean Depths (m)": stats["mean_depth"].mean(),
        "Mean of Max Depths (m)": stats["max_depth"].mean(),
        "Mean of Min Depths (m)": stats["min_depth"].mean(),
    }
    if stats["max_flooded_area"].notna().any():
        summary["Peak Spatial Extent (m^2)"] = stats["max_flooded_area"].max()
        summary["Average Spatial Extent (m^2)"] = (
            stats["mean_flooded_area"].mean())
        summary["Average Roadway Coverage (%)"] = (
            stats["mean_percent_covered"].mean())
    return pd.DataFrame([summary])


def format_summary(summary, main_dir, transect_name):
    """
    Format a summary table as human-readable report lines.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`summarize_roadway_stats`.
    main_dir : str
        Event folder, shown in the header.
    transect_name : str
        ROI name, shown in the header.

    Returns
    -------
    list of str
        Report lines.
    """
    lines = ["=== Roadway Accessibility Summary ===",
             f"Location: {main_dir}", f"Transect/ROI: {transect_name}"]
    for key, value in summary.iloc[0].items():
        if key == "Total Events Processed":
            lines.append(f"{key}: {int(value)}")
        elif "Proportion" in key or "Coverage (%)" in key:
            lines.append(f"{key}: {value:.2%}")
        else:
            lines.append(f"{key}: {value:.4f}")
    return lines


def main(argv=None):
    """
    Summarize roadway accessibility and log the report.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--event_dir", required=True,
                        help="Folder of event subfolders.")
    parser.add_argument("--transect_name", default=None,
                        help="ROI name (LabelMe JSON filename stem).")
    parser.add_argument("--site", default=None,
                        help="Site whose roi_json names the ROI when "
                             "--transect_name is omitted.")
    parser.add_argument("--roi_view", choices=("rectified", "camera"),
                        default="rectified",
                        help="With --site: use the site's roi_json "
                             "(rectified) or camera_roi_json (camera).")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Grid spacing in meters.")
    parser.add_argument("--output_csv", default=None,
                        help="Also save the summary to this CSV.")
    args = cli_common.start(parser, argv)
    name = args.transect_name
    key = "camera_roi_json" if args.roi_view == "camera" else "roi_json"
    if name is None and args.site:
        roi_json = getattr(cli_common.load_site(args), key)
        if roi_json:
            name = os.path.splitext(os.path.basename(roi_json))[0]
    if name is None:
        cli_common.fail(f"Pass --transect_name, or --site with a {key}.")

    summary = summarize_roadway_stats(args.event_dir, name, args.resolution)
    if summary is None:
        cli_common.fail("Nothing to summarize.")
    for line in format_summary(summary, args.event_dir, name):
        logger.info(line)
    if args.output_csv:
        summary.to_csv(args.output_csv, index=False)
        logger.info("Saved summary to %s", args.output_csv)


if __name__ == "__main__":
    main()
