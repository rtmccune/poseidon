"""Make the paper table summarizing flooding at each site.

Two sites side by side (``--sites``, left first), in two groups of rows:

In-situ sensor (``<site>/flood_events.csv``, the readings of every
event on the QC'd sheet; depths are the road water level, i.e. above
the edge of road, EOR)
    Flood events (sensor above ``--sensor_wet_ft``); maximum depth;
    mean of each event's maximum depth; total time inundated; mean
    inundation time per event.

Image-derived (``<event>/<roi>_accessibility_time_series.csv`` from
stage 10, for the publication ROI: ``camera_roi_json``, else
``roi_json``)
    Flood events (any image with water on the roadway); maximum depth on
    the roadway; mean of each event's maximum depth; total and mean per
    event time the roadway was inundated (any ROI cell wet); peak
    flooded area; mean flooded area of the inundated images.

Differences from stage 11 (``run_road_access_stats.py``), on purpose:

* Each image or reading stands for the time to the previous one, but at
  most ``--max_gap_minutes``. Stage 11 has no cap, so the first image
  after a night (or a camera gap) is credited with the whole gap, which
  inflates the inundated time.
* Images in two overlapping event windows count once, in the nearer
  event (as in the figures); stage 11 counts them in both.
* Event means are over events with water, and the mean area is over
  inundated images, so they describe a typical flood rather than being
  diluted by dry events and images.
* Every event folder counts, including events without sensor data,
  since the extra events are part of what the imagery adds; a row
  gives how many of the flood events have no sensor data.

Writes ``site_summary_table.{png,pdf,svg}`` (a rendered table in the
figure style), ``.csv`` (unrounded values), ``.tex`` (booktabs) and
``.md`` (the definitions above, with the settings used) to
``--out_dir``, from every image. ``site_summary_table_good_only.*`` is
the same table from well-segmented images only, for comparison (their
time is still measured against every image, so dropping poor images
does not stretch the gaps).

Example
-------
``python site_summary_table.py --sites carolina_beach down_east
--data_root data --out_dir data/publication_combined``
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from poseidon_utils.constants import FEET_TO_METERS  # noqa: E402
from poseidon_utils.event_tracker import FLOOD_START_THRESHOLD_FT  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402

import access_plotter as ap  # noqa: E402
import publication_figures as pub  # noqa: E402
from plot_style import INK, INK_STRONG, MAGENTA, ORANGE, apply_style  # noqa: E402

logger = logging.getLogger(__name__)

#: Header colour of each site, in ``--sites`` order (as in the site
#: description figure).
SITE_COLORS = (ORANGE, MAGENTA)

#: Light fill of alternate rows.
ROW_FILL = "#F4F5F3"

#: Table rows: (group, label, key, decimals).
ROWS = (
    ("In-situ sensor", "Flood events", "sensor_events", 0),
    ("In-situ sensor", "Maximum depth (m, EOR)", "sensor_max_depth_m", 3),
    ("In-situ sensor", "Mean of event maximum depths (m, EOR)",
     "sensor_mean_event_max_m", 3),
    ("In-situ sensor", "Total time inundated (h)", "sensor_total_hours", 1),
    ("In-situ sensor", "Mean time inundated per event (h)",
     "sensor_mean_event_hours", 1),
    ("Image-derived", "Flood events", "image_events", 0),
    ("Image-derived", "Flood events without sensor data",
     "image_events_no_sensor", 0),
    ("Image-derived", "Maximum depth on roadway (m)", "image_max_depth_m",
     3),
    ("Image-derived", "Mean of event maximum depths (m)",
     "image_mean_event_max_m", 3),
    ("Image-derived", "Total time roadway inundated (h)",
     "image_total_hours", 1),
    ("Image-derived", "Mean time inundated per event (h)",
     "image_mean_event_hours", 1),
    ("Image-derived", "Peak flooded area (m²)", "image_peak_area_m2", 1),
    ("Image-derived", "Mean flooded area (m²)", "image_mean_area_m2", 1),
)


def site_title(site):
    """Display name of a site, e.g. ``carolina_beach`` -> Carolina Beach."""
    return site.replace("_", " ").title()


def interval_hours(times, max_gap_minutes):
    """
    Hours each observation stands for: the gap to the previous one,
    capped.

    Parameters
    ----------
    times : pandas.Series
        Sorted observation times.
    max_gap_minutes : float
        Longest interval one observation may stand for.

    Returns
    -------
    numpy.ndarray
        Hours per observation; the first takes the second's gap.
    """
    minutes = times.diff().dt.total_seconds().div(60).to_numpy(copy=True)
    if len(minutes) > 1:
        minutes[0] = minutes[1]
    elif len(minutes):
        minutes[0] = 0.0
    return np.minimum(minutes, max_gap_minutes) / 60


def sensor_statistics(flood_events_csv, wet_ft, max_gap_minutes):
    """
    In-situ sensor rows of the table.

    Parameters
    ----------
    flood_events_csv : str
        ``flood_events.csv`` (``flood_event``, ``time_UTC``,
        ``road_water_level`` in feet).
    wet_ft : float
        Road water level above which the road counts as inundated.
    max_gap_minutes : float
        Longest interval one reading may stand for.

    Returns
    -------
    dict
        ``sensor_*`` values (NaN without the CSV).
    """
    keys = [key for _, _, key, _ in ROWS if key.startswith("sensor")]
    if not os.path.exists(flood_events_csv):
        logger.warning("No %s; sensor rows left empty.", flood_events_csv)
        return dict.fromkeys(keys, np.nan)
    readings = pd.read_csv(flood_events_csv)
    readings["time"] = pd.to_datetime(readings["time_UTC"], utc=True)
    events = []
    for event, group in readings.groupby("flood_event"):
        group = group.sort_values("time")
        hours = interval_hours(group["time"], max_gap_minutes)
        wet = group["road_water_level"].to_numpy() > wet_ft
        if wet.any():
            events.append({
                "max_m": group["road_water_level"].max() * FEET_TO_METERS,
                "hours": hours[wet].sum()})
    events = pd.DataFrame(events, columns=["max_m", "hours"])
    return {"sensor_events": len(events),
            "sensor_max_depth_m": events["max_m"].max(),
            "sensor_mean_event_max_m": events["max_m"].mean(),
            "sensor_total_hours": events["hours"].sum(),
            "sensor_mean_event_hours": events["hours"].mean()}


def image_statistics(events_dir, roi, labels_csv, resolution, args,
                     good_only=False):
    """
    Image-derived rows of the table.

    Parameters
    ----------
    events_dir : str
        Folder of event subfolders.
    roi : str
        Roadway ROI name.
    labels_csv : str or None
        ``segmentation_labels.csv`` (for the good-only table).
    resolution : float
        Grid spacing (m).
    args : argparse.Namespace
        Parsed arguments.
    good_only : bool, optional
        Use well-segmented images only. Default False (every image).

    Returns
    -------
    dict
        ``image_*`` values (NaN without accessibility CSVs). Every event
        folder counts, including events without sensor records
        (``<event>/<event>.csv`` missing; stage 05 writes it only for
        events with readings).
    """
    keys = [key for _, _, key, _ in ROWS if key.startswith("image")]
    good = (ap.load_good_timestamps(labels_csv)
            if labels_csv and os.path.exists(labels_csv) else set())
    if good_only and not good:
        logger.warning("Good-only table without segmentation labels (%s); "
                       "using every image.", labels_csv)
    images = pub.load_roi_images(events_dir, roi, good, None,
                                 args.padding_hours)
    if images is None:
        logger.warning("No accessibility CSVs for ROI %s in %s; image rows "
                       "left empty.", roi, events_dir)
        return dict.fromkeys(keys, np.nan)
    images["Time"] = pd.to_datetime(images["Time"], utc=True)
    pixel_area = resolution ** 2
    events = []
    for event, group in images.groupby("event"):
        group = group.sort_values("Time")
        # Time is measured against every image, then poor ones dropped
        hours = interval_hours(group["Time"], args.max_gap_minutes)
        keep = ((group["Segmentation"] == "good").to_numpy()
                if good_only and good else np.ones(len(group), bool))
        wet = (group["Impassable"] == 1).to_numpy() & keep
        if not wet.any():
            continue
        flooded = group.loc[wet]
        has_sensor = os.path.exists(os.path.join(events_dir, event,
                                                 f"{event}.csv"))
        events.append({
            "has_sensor": has_sensor,
            "max_m": flooded["MaxDepth"].max(),
            "hours": hours[wet].sum(),
            "peak_area": flooded["FloodedPixels"].max() * pixel_area,
            "areas": (flooded["FloodedPixels"] * pixel_area).tolist()})
    if not events:
        return dict.fromkeys(keys, np.nan) | {"image_events": 0}
    events = pd.DataFrame(events)
    areas = np.concatenate(events["areas"].to_list())
    return {"image_events": len(events),
            "image_events_no_sensor": int((~events["has_sensor"]).sum()),
            "image_max_depth_m": events["max_m"].max(),
            "image_mean_event_max_m": events["max_m"].mean(),
            "image_total_hours": events["hours"].sum(),
            "image_mean_event_hours": events["hours"].mean(),
            "image_peak_area_m2": events["peak_area"].max(),
            "image_mean_area_m2": float(np.mean(areas))}


def site_statistics(site, args, roi=None, good_only=False):
    """
    Every table value for one site.

    Parameters
    ----------
    site : str
        Site config name.
    args : argparse.Namespace
        Parsed arguments.
    roi : str, optional
        Roadway ROI name. Default: the publication ROI of the site.
    good_only : bool, optional
        Image rows from well-segmented images only.

    Returns
    -------
    dict
        Values by row key.
    """
    config = load_site_config(site)
    site_dir = os.path.join(args.data_root, site)
    roi = roi or pub.site_roi(site)
    values = sensor_statistics(os.path.join(site_dir, "flood_events.csv"),
                               args.sensor_wet_ft, args.max_gap_minutes)
    values.update(image_statistics(
        os.path.join(site_dir, "flood_events"), roi,
        os.path.join(site_dir, "segmentation_labels.csv"),
        config.grid.resolution, args, good_only))
    logger.info("%s (ROI %s, %s): %s", site, roi,
                "good images" if good_only else "all images",
                {k: round(v, 3) for k, v in values.items()})
    return values


def _format(value, decimals):
    """Format a table value ('–' for missing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    return f"{value:,.{decimals}f}"


def render_table(table, sites, path_stem):
    """
    Draw the table in the figure style and save PNG, PDF and SVG.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows of :data:`ROWS` with one column per site.
    sites : list of str
        Site names, in column order.
    path_stem : str
        Output path without extension.
    """
    apply_style()
    n_rows = len(ROWS) + 2  # plus a header line per group
    row_h, top = 0.27, 0.62
    fig = plt.figure(figsize=(6.4, top + row_h * n_rows + 0.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 6.4)
    ax.set_ylim(top + row_h * n_rows + 0.1, 0)
    ax.axis("off")
    label_x, cols = 0.1, [4.35, 5.6]

    for x, site, color in zip(cols, sites, SITE_COLORS):
        ax.text(x, 0.3, site_title(site), ha="center", va="center",
                fontsize=10, weight="bold", color=INK_STRONG)
        ax.plot([x - 0.55, x + 0.55], [0.47, 0.47], color=color,
                linewidth=2.5, solid_capstyle="butt")

    y, shade, group = top, False, None
    for (row_group, label, key, decimals) in ROWS:
        if row_group != group:
            group, shade = row_group, False
            ax.text((cols[0] + cols[-1]) / 2, y + row_h * 0.55, group,
                    ha="center", va="center", fontsize=9, style="italic",
                    color=INK)
            ax.plot([cols[0] - 0.6, cols[-1] + 0.6],
                    [y + row_h * 0.95] * 2, color=INK, linewidth=0.8)
            y += row_h
        if shade:
            ax.add_patch(plt.Rectangle((0.02, y), 6.36, row_h,
                                       facecolor=ROW_FILL,
                                       edgecolor="none", zorder=0))
        ax.text(label_x, y + row_h / 2, label, va="center", fontsize=9,
                color=INK_STRONG)
        for x, site in zip(cols, sites):
            ax.text(x, y + row_h / 2, _format(table.loc[key, site], decimals),
                    ha="center", va="center", fontsize=9, color=INK_STRONG)
        y += row_h
        shade = not shade
    ax.plot([0.02, 6.38], [y + 0.02] * 2, color=INK, linewidth=0.8)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.05)
    plt.close(fig)


def write_latex(table, sites, path):
    """Write the table as a booktabs LaTeX tabular."""
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             " & " + " & ".join(rf"\textbf{{{site_title(s)}}}"
                                for s in sites) + r" \\"]
    group = None
    for row_group, label, key, decimals in ROWS:
        if row_group != group:
            group = row_group
            lines += [r"\midrule",
                      rf" & \multicolumn{{{len(sites)}}}{{c}}"
                      rf"{{\textit{{{group}}}}} \\",
                      rf"\cmidrule(lr){{2-{len(sites) + 1}}}"]
        cells = [_format(table.loc[key, s], decimals).replace("–", "--")
                 for s in sites]
        lines.append(label.replace("²", r"$^2$") + " & "
                     + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_notes(args, path, good_only=False):
    """Write the row definitions and settings as Markdown."""
    image_set = ("well-segmented images only" if good_only
                 else "every image")
    text = f"""# Site summary table: definitions

Generated by `site_summary_table.py` for {', '.join(args.sites)}.

**In-situ sensor** (`flood_events.csv`: the readings of every event on
the QC'd event sheet). Depths are the road water level (above the edge
of road, EOR). An event counts if a reading exceeds
{args.sensor_wet_ft:g} ft. Each reading stands for the time since the
previous reading, at most {args.max_gap_minutes:g} min.

- Maximum depth: the highest reading.
- Mean of event maximum depths: over the flood events.
- Total time inundated: the time of every reading above the threshold.
- Mean time inundated per event: over the flood events.

**Image-derived** ({image_set}; the roadway ROI of the publication
figures). An image is inundated if any ROI cell has water (stage 10's
`Impassable`). An image in two overlapping event windows counts once, in
the event with the nearest flood period. Each image stands for the time
since the previous image of its event, at most {args.max_gap_minutes:g}
min.

- Flood events: events with at least one inundated image, including
  events without sensor data (confirmed from images, e.g. during sensor
  outages). These are what the imagery adds to the record.
- Flood events without sensor data: of those, events whose folder has
  no sensor readings (stage 05 wrote no `<event>.csv`).
- Maximum depth on roadway: the highest `MaxDepth`.
- Mean of event maximum depths: over the flood events.
- Total / mean time roadway inundated: the time of inundated images.
- Peak flooded area: the most ROI cells flooded at once, times the
  cell area.
- Mean flooded area: over inundated images.

These differ from stage 11 (`run_road_access_stats.py`):
- **Capped intervals:** stage 11 credits the first image after a night
  or a camera gap with the whole gap.
- **Overlapping windows:** stage 11 counts images in overlapping windows
  twice.
- **Averages:** stage 11 averages over all events and images, wet or
  dry.
"""
    with open(path, "w") as handle:
        handle.write(text)


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
        description="Make the site flooding summary table.")
    parser.add_argument("--sites", nargs="+", required=True,
                        help="Site config names, in column order.")
    parser.add_argument("--data_root", default="data",
                        help="Folder holding one data folder per site.")
    parser.add_argument("--out_dir", default="data/publication_combined",
                        help="Output folder.")
    parser.add_argument("--rois", nargs="+", default=None,
                        help="Roadway ROI names, one per site. Default: "
                             "each site's camera_roi_json, else roi_json.")
    parser.add_argument("--max_gap_minutes", type=float, default=12.0,
                        help="Longest interval one image or reading may "
                             "stand for (images are every 6 min).")
    parser.add_argument("--sensor_wet_ft", type=float,
                        default=FLOOD_START_THRESHOLD_FT,
                        help="Road water level (ft) above which the sensor "
                             "counts as inundated.")
    parser.add_argument("--padding_hours", type=float,
                        default=ap.EVENT_PADDING_HOURS,
                        help="Padding stage 05 added around each event.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Compute and write the table.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    setup_logging()
    args = parse_args(argv)
    if len(args.sites) > len(SITE_COLORS):
        raise SystemExit(f"At most {len(SITE_COLORS)} sites.")
    rois = args.rois or [None] * len(args.sites)
    os.makedirs(args.out_dir, exist_ok=True)
    # The paper table uses every image; the good-only copy is for checking
    for good_only, suffix in ((False, ""), (True, "_good_only")):
        table = pd.DataFrame({site: site_statistics(site, args, roi,
                                                    good_only)
                              for site, roi in zip(args.sites, rois)})
        table = table.reindex([key for _, _, key, _ in ROWS])
        stem = os.path.join(args.out_dir, f"site_summary_table{suffix}")
        out = table.copy()
        out.insert(0, "row", [label for _, label, _, _ in ROWS])
        out.insert(0, "group", [group for group, _, _, _ in ROWS])
        out.to_csv(f"{stem}.csv", index_label="key")
        render_table(table, args.sites, stem)
        write_latex(table, args.sites, f"{stem}.tex")
        write_notes(args, f"{stem}.md", good_only)
        logger.info("Table written to %s.{png,pdf,svg,csv,tex,md}", stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
