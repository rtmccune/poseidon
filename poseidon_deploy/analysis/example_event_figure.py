"""Make the example event figure for each site.

For one flood event per site: three panes at the rising leg, the peak
and the falling leg of the flooding, above the event's water depth time
series. Each pane is framed in a colour that matches a dashed vertical
line on the time series at the moment it shows.

Panes (``--pane``)
    ``depth`` (default): the depth map (stage 08, ``--statistic``) on
    the rectified image, north up, with the roadway outline, on one
    shared colour scale. ``image``: the camera image with the predicted
    water shaded.

Time series
    The in-situ sensor's depth (dashed; readings at or below the road
    surface shown as 0 m) and the mean depth on the roadway from each
    well-segmented image (points), in local time.

Choosing the moments
    Only well-segmented images with water on the roadway (and, for depth
    panes, a depth map) are candidates. The peak is the candidate nearest
    the in-situ sensor's peak (or, without sensor water, the candidate
    with the deepest roadway mean); rising and falling are the
    candidates nearest halfway between the first wet image and the peak,
    and between the peak and the last wet image. ``--times`` overrides
    them.

Choosing the event (``--events`` overrides)
    Among events with candidates both before and after the peak, the
    one with the most candidates on its sparser side, ties broken by
    the deeper sensor peak.

Writes ``example_event_<site>.{png,pdf,svg}`` and
``example_event_<site>.csv`` (the moments shown and their depths) to
``--out_dir``.

Example
-------
``python example_event_figure.py --sites carolina_beach down_east
--data_root data --out_dir data/publication_combined``
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import cv2  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize, to_rgb  # noqa: E402

from poseidon_utils.constants import EASTERN_TZ  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402

import access_plotter as ap  # noqa: E402
import publication_figures as pub  # noqa: E402
from methods_figure_assets import _north_up, _stamp_files, _zip_array  # noqa: E402
from plot_style import (BLUE, INK, INK_STRONG, MAGENTA, ORANGE,  # noqa: E402
                        SEQUENTIAL_BLUE, apply_style)

logger = logging.getLogger(__name__)

#: Phases shown, with their frame and line colours. Distinct from the
#: series colours (black sensor, blue roadway mean).
PHASES = (("Rising", ORANGE), ("Peak", MAGENTA), ("Falling", "#6D8600"))


def site_title(site):
    """Display name of a site, e.g. ``carolina_beach`` -> Carolina Beach."""
    return site.replace("_", " ").title()


def stamp_of(time):
    """14-digit UTC filename timestamp of a time."""
    return pd.Timestamp(time).tz_convert("UTC").strftime("%Y%m%d%H%M%S")


def has_depth_map(event_dir, stamp, statistic):
    """Whether an event's depth store holds a map for an image."""
    return _zip_array(os.path.join(event_dir, "zarr", "depth_maps.zip"),
                      stamp, f"depth_map_{statistic}") is not None


def event_candidates(images, event, sensor, args):
    """
    Candidate images and the sensor peak of one event.

    Parameters
    ----------
    images : pandas.DataFrame
        Images of the ROI (:func:`publication_figures.load_roi_images`).
    event : str
        Event folder name.
    sensor : pandas.DataFrame
        Sensor readings.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    tuple
        ``(candidates, peak_time, peak_depth)``: well-segmented images
        with water on the roadway, sorted by time; the sensor's peak
        time (NaT if it saw no water in the event) and depth.
    """
    rows = images[(images["event"] == event)
                  & (images["Segmentation"] == "good")
                  & (images["Impassable"] == 1)].copy()
    rows["time"] = pd.to_datetime(rows["Time"], utc=True)
    rows = rows.dropna(subset=["MeanDepth"]).sort_values("time")
    start, end = ap.event_window(event)
    pad = pd.Timedelta(hours=args.padding_hours)
    window = sensor[(sensor["datetime_utc"] >= start + pad)
                    & (sensor["datetime_utc"] <= end - pad)]
    wet = window[window["SensorDepth_m"] > args.sensor_wet_ft * 0.3048]
    if wet.empty:
        return rows, pd.NaT, np.nan
    best = wet.loc[wet["SensorDepth_m"].idxmax()]
    return rows, best["datetime_utc"], best["SensorDepth_m"]


def pick_moments(rows, peak_time):
    """
    Rising, peak and falling images of an event.

    Returns
    -------
    list of pandas.Series or None
        Three rows of `rows`, or None without candidates on both sides
        of the peak.
    """
    if rows.empty:
        return None
    if pd.isna(peak_time):
        peak_time = rows.loc[rows["MeanDepth"].idxmax(), "time"]
    peak = rows.iloc[(rows["time"] - peak_time).abs().argmin()]
    before = rows[rows["time"] < peak["time"]]
    after = rows[rows["time"] > peak["time"]]
    if before.empty or after.empty:
        return None
    first, last = rows["time"].iloc[0], rows["time"].iloc[-1]
    rise_target = first + (peak["time"] - first) / 2
    fall_target = peak["time"] + (last - peak["time"]) / 2
    rise = before.iloc[(before["time"] - rise_target).abs().argmin()]
    fall = after.iloc[(after["time"] - fall_target).abs().argmin()]
    return [rise, peak, fall]


def choose_event(images, sensor, site_dir, args, event=None, times=None):
    """
    Pick the event and the three moments for one site.

    Returns
    -------
    tuple
        ``(event, moments)``: the event folder name and the three image
        rows (rising, peak, falling).

    Raises
    ------
    SystemExit
        If no event has candidates on both sides of its peak.
    """
    events_dir = os.path.join(site_dir, "flood_events")
    names = [event] if event else sorted(images["event"].unique())
    scored = []
    for name in names:
        rows, peak_time, peak_depth = event_candidates(images, name, sensor,
                                                       args)
        if args.pane == "depth" and not rows.empty:
            event_dir = os.path.join(events_dir, name)
            rows = rows[[has_depth_map(event_dir, stamp_of(t),
                                       args.statistic)
                         for t in rows["time"]]]
        if times:
            wanted = [pd.Timestamp(t, tz="UTC") for t in times]
            chosen = [rows.iloc[(rows["time"] - t).abs().argmin()]
                      for t in wanted] if not rows.empty else None
            return name, chosen
        moments = pick_moments(rows, peak_time)
        if moments is None:
            continue
        sides = min((rows["time"] < moments[1]["time"]).sum(),
                    (rows["time"] > moments[1]["time"]).sum())
        scored.append((sides, np.nan_to_num(peak_depth), name, moments))
    if not scored:
        raise SystemExit(f"No event in {events_dir} has well-segmented "
                         f"images with water before and after its peak; "
                         f"pass --events and --times.")
    sides, depth, name, moments = max(scored, key=lambda s: (s[0], s[1]))
    logger.info("Example event %s (%d candidates on its sparser side, "
                "sensor peak %.3f m)", name, sides, depth)
    return name, moments


def depth_pane(event_dir, stamp, args):
    """
    Depth map on the greyed rectified image, north up.

    Returns
    -------
    tuple
        ``(background_rgb, depth)`` arrays (depth NaN where dry), or
        ``(None, None)`` if missing.
    """
    zarr = os.path.join(event_dir, "zarr")
    depth = _zip_array(os.path.join(zarr, "depth_maps.zip"), stamp,
                       f"depth_map_{args.statistic}")
    rect = _zip_array(os.path.join(zarr, "orig_image_rects.zip"), stamp)
    if depth is None:
        return None, None
    depth = _north_up(np.squeeze(depth)).astype(float)
    if rect is None:
        grey = np.full((*depth.shape, 3), 0.85)
    else:
        rect = _north_up(rect).astype(float)[..., :3] / 255
        grey = np.repeat(rect.mean(axis=2, keepdims=True), 3, axis=2)
        grey = 0.35 + 0.55 * grey  # light, so the depths stand out
    depth[depth <= 0] = np.nan
    return grey, depth


def image_pane(event_dir, stamp, args):
    """
    Camera image with the predicted water shaded.

    Returns
    -------
    numpy.ndarray or None
        RGB image (floats 0-1), or None if missing.
    """
    originals = _stamp_files(os.path.join(event_dir, "orig_images"), stamp)
    if not originals:
        return None
    image = cv2.cvtColor(cv2.imread(originals[0]), cv2.COLOR_BGR2RGB)
    image = image.astype(float) / 255
    labels = _stamp_files(os.path.join(event_dir, "labels"), stamp)
    if labels:
        mask = cv2.imread(labels[0], cv2.IMREAD_UNCHANGED)
        mask = mask[..., 0] if mask.ndim == 3 else mask
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, image.shape[1::-1],
                              interpolation=cv2.INTER_NEAREST)
        water = mask == args.water_class
        image[water] = 0.45 * image[water] + 0.55 * np.array(to_rgb(BLUE))
    return image


def draw_figure(site, event, moments, images, sensor, args):
    """
    Draw one site's figure and write its CSV.

    Returns
    -------
    str
        Output path stem.
    """
    site_dir = os.path.join(args.data_root, site)
    event_dir = os.path.join(site_dir, "flood_events", event)
    roi = args.roi or pub.site_roi(site)
    stamps = [stamp_of(m["time"]) for m in moments]
    local = [m["time"].tz_convert(EASTERN_TZ) for m in moments]

    fig = plt.figure(figsize=(7.5, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.22)
    panes = grid[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.07],
                                wspace=0.08)

    loaded = [depth_pane(event_dir, s, args) if args.pane == "depth"
              else (image_pane(event_dir, s, args), None) for s in stamps]
    depths = [d for _, d in loaded if d is not None]
    top = args.depth_max or (float(np.nanpercentile(
        np.concatenate([d[np.isfinite(d)] for d in depths]), 99))
        if depths and any(np.isfinite(d).any() for d in depths) else 0.1)
    norm = Normalize(0, top)
    mask_path = os.path.join(event_dir, f"{roi}_roi_mask.png")
    roi_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
                if os.path.exists(mask_path) else None)

    for k, ((name, color), (background, depth), when) in enumerate(
            zip(PHASES, loaded, local)):
        ax = fig.add_subplot(panes[k])
        if background is None:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color=INK)
        else:
            ax.imshow(background, interpolation="none")
            if depth is not None:
                ax.imshow(depth, cmap=SEQUENTIAL_BLUE, norm=norm,
                          interpolation="none")
                if roi_mask is not None and roi_mask.shape == depth.shape:
                    ax.contour(roi_mask.astype(float), levels=[0.5],
                               colors=INK_STRONG, linewidths=0.8,
                               linestyles="--")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)
        ax.set_title(f"({'abc'[k]}) {name}, {when:%H:%M}", fontsize=9,
                     color=INK_STRONG, loc="left")
    if args.pane == "depth":
        cax = fig.add_subplot(panes[3])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_BLUE),
                     cax=cax, label="Depth (m)")
        cax.tick_params(labelsize=7)

    # Time series over the event's window, local time
    ax = fig.add_subplot(grid[1])
    start, end = ap.event_window(event)
    window = sensor[(sensor["datetime_utc"] >= start)
                    & (sensor["datetime_utc"] <= end)]
    ax.plot(window["datetime_utc"].dt.tz_convert(EASTERN_TZ),
            window["SensorDepth_m"], color=INK_STRONG, linestyle="--",
            linewidth=1.6, label="In-situ sensor")
    good = images[(images["event"] == event)
                  & (images["Segmentation"] == "good")].copy()
    good["time"] = pd.to_datetime(good["Time"], utc=True)
    good["depth"] = good["MeanDepth"].where(good["Impassable"] == 1, 0.0)
    ax.plot(good["time"].dt.tz_convert(EASTERN_TZ), good["depth"],
            marker="o", markersize=3.5, linestyle="none", color=BLUE,
            label="Mean depth on roadway (image)")
    for (name, color), when in zip(PHASES, local):
        ax.axvline(when, color=color, linestyle="--", linewidth=1.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Water depth (m)")
    first = local[0]
    ax.set_xlabel(f"Time (Eastern), {first:%-d %B %Y}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",
                                                      tz=EASTERN_TZ))
    ax.legend(loc="best", fontsize=8)
    ax.text(-0.07, 1.02, "(d)", transform=ax.transAxes, fontsize=10,
            weight="bold", va="bottom")

    stem = os.path.join(args.out_dir, f"example_event_{site}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame([{
        "site": site, "event": event, "phase": name,
        "time_utc": m["time"], "time_local": when, "image_timestamp": s,
        "roadway_mean_depth_m": m["MeanDepth"],
        "roadway_coverage_pct": m.get("Coverage", np.nan)}
        for (name, _), m, when, s in zip(PHASES, moments, local, stamps)
    ]).to_csv(f"{stem}.csv", index=False)
    return stem


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
        description="Example flood event figure per site.")
    parser.add_argument("--sites", nargs="+", required=True,
                        help="Site config names (one figure each).")
    parser.add_argument("--data_root", default="data",
                        help="Folder holding one data folder per site.")
    parser.add_argument("--out_dir", default="data/publication_combined",
                        help="Output folder.")
    parser.add_argument("--events", nargs="+", default=None,
                        help="Event folder per site, in --sites order "
                             "('auto' to choose). Default: chosen.")
    parser.add_argument("--times", nargs=4, action="append", default=None,
                        metavar=("SITE", "RISING", "PEAK", "FALLING"),
                        help="Image timestamps (14-digit UTC) to show for "
                             "a site, with --events. Repeat per site.")
    parser.add_argument("--pane", choices=("depth", "image"),
                        default="depth",
                        help="Pane content: depth map (plan view) or "
                             "camera image with water shaded.")
    parser.add_argument("--roi", default=None,
                        help="Roadway ROI. Default: the publication ROI.")
    parser.add_argument("--statistic", default="95_perc",
                        help="Depth map statistic (stage 10 uses 95_perc).")
    parser.add_argument("--depth_max", type=float, default=None,
                        help="Top of the depth colour scale (m). Default: "
                             "99th percentile of the three maps.")
    parser.add_argument("--water_class", type=int, default=1,
                        help="Label value of water (image panes).")
    parser.add_argument("--sensor_wet_ft", type=float, default=0.02,
                        help="Sensor level (ft) above which it is wet.")
    parser.add_argument("--padding_hours", type=float,
                        default=ap.EVENT_PADDING_HOURS,
                        help="Padding stage 05 added around each event.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Draw one figure per site.

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
    apply_style()
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    times = {t[0]: t[1:] for t in (args.times or [])}
    for i, site in enumerate(args.sites):
        site_dir = os.path.join(args.data_root, site)
        good = ap.load_good_timestamps(
            os.path.join(site_dir, "segmentation_labels.csv"))
        sensor = ap.load_sensor_data(
            os.path.join(site_dir, "sensor_data_comparison.csv"))
        roi = args.roi or pub.site_roi(site)
        images = pub.load_roi_images(os.path.join(site_dir, "flood_events"),
                                     roi, good, sensor, args.padding_hours)
        if images is None or sensor is None or not good:
            logger.error("%s: needs stage 10 CSVs, sensor_data_comparison.csv "
                         "and segmentation_labels.csv; skipped.", site)
            continue
        event = (args.events[i] if args.events and args.events[i] != "auto"
                 else None)
        event, moments = choose_event(images, sensor, site_dir, args, event,
                                      times.get(site))
        if not moments:
            logger.error("%s: no images to show for %s.", site, event)
            continue
        stem = draw_figure(site, event, moments, images, sensor, args)
        logger.info("%s: %s, %s -> %s.{png,pdf,svg,csv}", site_title(site),
                    event, ", ".join(stamp_of(m["time"]) for m in moments),
                    stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
