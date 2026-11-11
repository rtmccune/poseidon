"""Make the example event figure for each site.

For one flood event per site: three panes at the rising leg, the peak
and the falling leg of the flooding, above the event's water depth time
series. Each pane is framed in a colour that matches a dashed vertical
line on the time series at the moment it shows.

Panes (``--pane``)
    ``depth`` (default): the depth map (stage 08, ``--statistic``)
    drawn exactly as stage 09 draws it (basemap, depth colour scale,
    virtual sensor markers), one colour bar for all three. ``image``:
    the camera image with the predicted water shaded.

Time series
    The in-situ sensor's depth (dashed; readings at or below the road
    surface shown as 0 m), and from every image of the event the mean
    roadway depth in the field of view and each virtual sensor's depth
    (points, in the map marker colours; 0 m where dry), in local time.
    Points from well-segmented images are opaque, the rest faded.

Choosing the moments
    Only well-segmented images with water on the roadway (and, for depth
    panes, a depth map) are candidates. The peak is the candidate nearest
    the in-situ sensor's peak (or, without sensor water, the candidate
    with the deepest roadway mean); rising and falling are the
    candidates nearest halfway between the first wet image and the peak,
    and between the peak and the last wet image. ``--times`` (with
    ``--events``) names the images instead; they are used as given,
    whatever their segmentation score.

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
from matplotlib.colors import to_rgb  # noqa: E402

from poseidon_utils.constants import EASTERN_TZ  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402
from poseidon_utils.zarr_io import read_zip_arrays  # noqa: E402

import access_plotter as ap  # noqa: E402
import publication_figures as pub  # noqa: E402
from methods_figure_assets import _stamp_files, _zip_array  # noqa: E402
from plot_style import (BLUE, INK, INK_STRONG, apply_style,  # noqa: E402
                        virtual_sensor_color)

logger = logging.getLogger(__name__)

#: Phases shown, with their frame and line colours: a light-to-dark
#: red, in time order, distinct from the depth colour scale (blue to
#: purple) and the series (black sensor, blue roadway mean, the virtual
#: sensors' green, amber and pink).
#: Opacity of time series points from images not scored well segmented.
FADED_ALPHA = 0.25

PHASES = (("Rising", "#F28E7F"), ("Peak", "#D7301F"),
          ("Falling", "#7F0000"))


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


def named_moments(images, event, times, event_dir, args):
    """
    The images named with ``--times``, used as given.

    Any image of the event counts, whatever its segmentation score or
    wetness (the automatic choice is limited to well-segmented, wet
    images; a named image is the user's choice). A time without an image
    falls back to the event's nearest image, with a warning.

    Parameters
    ----------
    images : pandas.DataFrame
        Images of the ROI.
    event : str
        Event folder name.
    times : sequence of str
        Rising, peak and falling timestamps (14-digit UTC).
    event_dir : str
        Event folder.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    list of pandas.Series or None
        The three image rows, or None if the event has no images.
    """
    rows = images[images["event"] == event].copy()
    if rows.empty:
        return None
    rows["time"] = pd.to_datetime(rows["Time"], utc=True)
    chosen = []
    for (phase, _), stamp in zip(PHASES, times):
        wanted = pd.Timestamp(stamp, tz="UTC")
        row = rows.iloc[(rows["time"] - wanted).abs().argmin()]
        if row["time"] != wanted:
            logger.warning("%s: no image at %s in %s; using the nearest, "
                           "%s.", phase, stamp, event, stamp_of(row["time"]))
        if row.get("Segmentation") != "good":
            logger.info("%s image %s is not scored well segmented.", phase,
                        stamp_of(row["time"]))
        if (args.pane == "depth"
                and not has_depth_map(event_dir, stamp_of(row["time"]),
                                      args.statistic)):
            logger.warning("%s image %s has no depth map; its pane will be "
                           "empty.", phase, stamp_of(row["time"]))
        chosen.append(row)
    if len({r["time"] for r in chosen}) < len(chosen):
        logger.warning("Two of the named times map to the same image.")
    return chosen


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
    if times and event:
        return event, named_moments(images, event, times,
                                    os.path.join(events_dir, event), args)
    if times:
        logger.warning("--times needs --events for the site; choosing the "
                       "moments automatically.")
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


def map_plotter(site, event_dir):
    """
    Stage 09's map plotter for an event, with the site's virtual sensors.

    Parameters
    ----------
    site : str
        Site config name.
    event_dir : str
        Event folder.

    Returns
    -------
    poseidon_core.depth_map_plotter.DepthMapPlotter
        Plotter drawing depth maps as stage 09 does (basemap, colour
        scale, virtual sensor markers).
    """
    from poseidon_core.depth_map_plotter import DepthMapPlotter

    config = load_site_config(site)
    g = config.grid
    basemap = config.resolve(config.basemap)
    return DepthMapPlotter(
        event_dir, g.min_x, g.max_x, g.min_y, g.max_y,
        resolution_m=g.resolution, bbox_crs=config.crs,
        virtual_sensor_locations=config.virtual_sensors, plot_sensors=True,
        basemap_path=str(basemap) if basemap else None)


def depth_map(event_dir, stamp, args):
    """
    An image's depth map in grid orientation (row 0 south), or None.
    """
    depth = _zip_array(os.path.join(event_dir, "zarr", "depth_maps.zip"),
                       stamp, f"depth_map_{args.statistic}")
    return None if depth is None else np.squeeze(depth).astype(float)


def virtual_sensor_series(event_dir, keep=None):
    """
    Each virtual sensor's depth at the event's images.

    Parameters
    ----------
    event_dir : str
        Event folder (``zarr/virtual_sensor_data.zip`` from stage 09).
    keep : set of str, optional
        Image timestamps to keep. Default: every image.

    Returns
    -------
    pandas.DataFrame or None
        ``time`` and ``VS1``, ``VS2``... (0 m where the cell is dry), or
        None without the store.
    """
    path = os.path.join(event_dir, "zarr", "virtual_sensor_data.zip")
    if not os.path.exists(path):
        logger.warning("No %s; virtual sensors left off the time series.",
                       path)
        return None
    arrays = read_zip_arrays(path, ["timestamps", "vs_depth"])
    stamps = [str(t) for t in arrays["timestamps"]]
    depth = np.atleast_2d(arrays["vs_depth"]).astype(float)
    table = pd.DataFrame(np.nan_to_num(depth, nan=0.0),
                         columns=[f"VS{i + 1}" for i in
                                  range(depth.shape[1])])
    table["stamp"] = [pd.Timestamp(t).strftime("%Y%m%d%H%M%S")
                      if not t.isdigit() else t for t in stamps]
    table["time"] = pd.to_datetime(table["stamp"], format="%Y%m%d%H%M%S",
                                   utc=True)
    if keep is not None:
        table = table[table["stamp"].isin(keep)]
    return table.sort_values("time")


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
    stamps = [stamp_of(m["time"]) for m in moments]
    local = [m["time"].tz_convert(EASTERN_TZ) for m in moments]

    fig = plt.figure(figsize=(7.5, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.1)
    panes = grid[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.07],
                                wspace=0.08)

    plotter = map_plotter(site, event_dir) if args.pane == "depth" else None
    colour_layer = None
    for k, ((name, color), stamp, when) in enumerate(
            zip(PHASES, stamps, local)):
        ax = fig.add_subplot(panes[k])
        if args.pane == "depth":
            depth = depth_map(event_dir, stamp, args)
            if depth is not None:
                # Stage 09's map: basemap, depth scale, virtual sensors
                colour_layer, geodata, style = plotter.draw_map(
                    ax, depth, "depth", basemap=True, marker_size=60,
                    label_size=6.5)
                ax.set_aspect("equal")
                # Wetted area, as stage 09 labels its maps
                ax.text(0.04, 0.96, f"Spatial Extent (m$^2$): "
                                    f"{geodata['spatial_extent']:g}",
                        transform=ax.transAxes, fontsize=6.5, va="top",
                        ha="left", zorder=30,
                        bbox={"facecolor": "white", "alpha": 0.8,
                              "edgecolor": INK, "linewidth": 0.5,
                              "boxstyle": "square,pad=0.25"})
        else:
            image = image_pane(event_dir, stamp, args)
            if image is not None:
                ax.imshow(image)
        if not ax.images:
            ax.text(0.5, 0.5, "image not found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)
        # Bold letter, as in (d), then the phase and time
        ax.text(0, 1.02, f"({'abc'[k]})", transform=ax.transAxes,
                fontsize=10, weight="bold", va="bottom", color=INK_STRONG)
        ax.annotate(f"{name}, {when:%H:%M}", xy=(0, 1.02),
                    xycoords="axes fraction", xytext=(22, 0),
                    textcoords="offset points", fontsize=9, va="bottom",
                    color=INK_STRONG)
    if colour_layer is not None:
        cax = fig.add_subplot(panes[3])
        fig.colorbar(colour_layer, cax=cax, label=style["cbar_label"])
        cax.tick_params(labelsize=7)

    # Time series over the event's window, local time
    ax = fig.add_subplot(grid[1])
    start, end = ap.event_window(event)
    window = sensor[(sensor["datetime_utc"] >= start)
                    & (sensor["datetime_utc"] <= end)]
    ax.plot(window["datetime_utc"].dt.tz_convert(EASTERN_TZ),
            window["SensorDepth_m"], color=INK_STRONG, linestyle="--",
            linewidth=1.6, label="In-Situ Sensor")
    # Every image of the event (the moments were chosen from
    # well-segmented images only)
    shown = images[images["event"] == event].copy()
    shown["time"] = pd.to_datetime(shown["Time"], utc=True)
    shown["depth"] = shown["MeanDepth"].where(shown["Impassable"] == 1, 0.0)
    good = set(shown.loc[shown["Segmentation"] == "good", "time"]
               .dt.strftime("%Y%m%d%H%M%S"))

    def points(times, values, stamps, color, size, label):
        # Opaque for well-segmented images, faded for the rest
        is_good = np.isin(stamps, list(good))
        local_times = times.dt.tz_convert(EASTERN_TZ)
        ax.plot(local_times[~is_good], values[~is_good], marker="o",
                markersize=size, linestyle="none", color=color,
                alpha=FADED_ALPHA)
        ax.plot(local_times[is_good], values[is_good], marker="o",
                markersize=size, linestyle="none", color=color,
                label=label)

    points(shown["time"], shown["depth"],
           shown["time"].dt.strftime("%Y%m%d%H%M%S").to_numpy(), BLUE, 3.5,
           "Mean Roadway Depth in Field of View")
    sensors = virtual_sensor_series(event_dir)
    if sensors is not None:
        for i, column in enumerate(c for c in sensors if
                                   c.startswith("VS")):
            points(sensors["time"], sensors[column],
                   sensors["stamp"].to_numpy(), virtual_sensor_color(i),
                   2.5, column)
    # Legend key for the faded points
    ax.plot([], [], marker="o", markersize=3.5, linestyle="none",
            color=INK, alpha=FADED_ALPHA,
            label="Faded: not well segmented")
    for (name, color), when in zip(PHASES, local):
        ax.axvline(when, color=color, linestyle="--", linewidth=1.6)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Water Depth (m)")
    first = local[0]
    ax.set_xlabel(f"Time (Eastern), {first:%-d %B %Y}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",
                                                      tz=EASTERN_TZ))
    # Headroom above the data so the legend does not cover the peak
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2,
              handlelength=1.6, columnspacing=1.0, labelspacing=0.3,
              borderaxespad=0.3, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9)
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
