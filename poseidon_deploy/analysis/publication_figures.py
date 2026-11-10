"""Make the publication figures for one site and roadway ROI.

Builds on ``sensor_camera_comparison.py`` and ``access_plotter.py`` and
uses only well-segmented images (``--labels`` is required). Figures
have no titles (captions describe them) and are saved as PNG (300 dpi)
and PDF in ``--out_dir`` with the CSVs behind every number shown.

Figure 1, ``<roi>_pub_fig1_aligned_timing``
    (a) Mean depth across events against hours since the start of each
    event's window (the event folder's buffered window; the event
    sheet's start is dotted at +``--padding_hours``) for the in-situ
    sensor, each virtual sensor and the ROI mean. The in-situ sensor is
    drawn only between the event sheet's start and end: it defines the
    event, and its readings in the buffer belong to neighbouring events
    whose padded windows overlap. Dry times are left
    out, so each line is the mean depth among events with water at
    that time (15-min bins with >= 3 events).
    (b), (c) Onset lead and end lag per event: minutes water appeared
    at each camera location before it reached the in-situ sensor, and
    stayed after it left, with the mean across events marked and
    labelled. Events whose first or last image was already wet are
    included without distinction (their true lead or lag may be
    larger); ``_timing.csv`` flags them.

Figure 2, ``<roi>_pub_fig2_binned_depth``
    Mean image depth, maximum image depth and in-situ sensor depth by
    roadway coverage bin, from the same images (well segmented, coverage
    > 0, a sensor reading within ``--match_minutes``). Markers are bin
    means with event-resampled 95 % CIs; shading spans the 10th-90th
    percentiles; a fully covered road has its own panel.

Statistics, ``<roi>_pub_metrics.md``
    The numbers to quote, kept off the figures: VS1 against the in-situ
    sensor image by image (depth error where both show water, and how
    often they agree on whether water is present); the mean and median
    onset lead and end lag per location; and each image depth metric
    against the sensor for figure 2 (bias, MAE, RMSE, r; image -
    sensor), all with event-resampled 95 % CIs.

The in-situ sensor counts as wet above ``--sensor_wet_ft`` (default the
event detection start threshold, 0.02 ft).

Example
-------
``python publication_figures.py --site down_east --events_dir
data/down_east/flood_events --sensor
data/down_east/sensor_data_comparison.csv --labels
data/down_east/segmentation_labels.csv --out_dir
data/down_east/publication``
"""

import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from poseidon_utils.constants import FEET_TO_METERS  # noqa: E402
from poseidon_utils.event_tracker import FLOOD_START_THRESHOLD_FT  # noqa: E402
from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402

import access_plotter as ap  # noqa: E402
import sensor_camera_comparison as sc  # noqa: E402
from plot_style import (  # noqa: E402
    BLUE,
    CONTEXT_GREY,
    INK,
    MAGENTA,
    ORANGE,
    apply_style,
)

logger = logging.getLogger(__name__)


def _save(fig, out_dir, stem):
    """
    Save a figure as PNG and PDF, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure.
    out_dir : str
        Output folder.
    stem : str
        Filename without extension.
    """
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"))
    plt.close(fig)


def _interval(stats, key, digits=3):
    """
    Format a statistic with its 95 % confidence interval.

    Parameters
    ----------
    stats : dict
        Output of :func:`access_plotter.paired_agreement`.
    key : str
        Statistic name.
    digits : int, optional
        Decimal places.

    Returns
    -------
    str
        ``"value [low, high]"`` (or the value alone without a CI).
    """
    value = f"{stats[key]:.{digits}f}"
    low, high = stats[f"{key}_ci95_low"], stats[f"{key}_ci95_high"]
    return value if np.isnan(low) else (
        f"{value} [{low:.{digits}f}, {high:.{digits}f}]")


#: Plain-language names used in the figures.
DISPLAY_NAMES = {sc.SENSOR: "In-Situ Sensor",
                 sc.ROI_STATS["MeanDepth"]: "Mean on Roadway",
                 "ROI": "Anywhere on Roadway"}


def _display(source):
    """
    Plain-language name of a source for figure labels.

    Parameters
    ----------
    source : str
        Internal source name.

    Returns
    -------
    str
        Display name (virtual sensors keep ``VS1``...).
    """
    return DISPLAY_NAMES.get(source, source)


# ----- Figure 1 --------------------------------------------------------------

def vs1_sensor_pairs(series, sensor, match_minutes, wet_m, padding_hours):
    """
    Pair each VS1 image with the in-situ sensor reading at that time.

    Images in two overlapping event windows are kept once, in the
    nearer event.

    Parameters
    ----------
    series : pandas.DataFrame
        Output of :func:`sensor_camera_comparison.event_series`
        (well-segmented images; VS1 NaN where dry).
    sensor : pandas.DataFrame
        All sensor readings.
    match_minutes : float
        Largest gap to the matched reading.
    wet_m : float
        Sensor depth above which it counts as wet.
    padding_hours : float
        Padding stage 05 added around each event.

    Returns
    -------
    pandas.DataFrame
        ``event``, ``time``, ``vs1_m`` (NaN where dry),
        ``sensor_depth_m``, ``sensor_state`` per VS1 image with a
        sensor reading.
    """
    vs1 = series[series["source"] == "VS1"].reset_index(drop=True)
    if vs1.empty:
        return pd.DataFrame(columns=["event", "time", "vs1_m",
                                     "sensor_depth_m", "sensor_state"])
    vs1 = vs1[ap.nearest_event_mask(vs1["event"], vs1["time"],
                                    padding_hours)].reset_index(drop=True)
    state = sc.sensor_state(vs1["time"], sensor, match_minutes, wet_m)
    pairs = pd.DataFrame({"event": vs1["event"], "time": vs1["time"],
                          "vs1_m": vs1["depth_m"],
                          "sensor_depth_m": state["sensor_depth_m"],
                          "sensor_state": state["sensor_state"]})
    return pairs[pairs["sensor_state"] != "no sensor reading"]


def vs1_agreement(pairs, n_boot, seed):
    """
    Depth error and wet/dry agreement of VS1 against the sensor.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`vs1_sensor_pairs`.
    n_boot : int
        Event resamples for the confidence intervals.
    seed : int
        Bootstrap random seed.

    Returns
    -------
    dict
        :func:`access_plotter.paired_agreement` of the images where
        both show water, plus ``paired_images``, ``both_wet``,
        ``vs1_only_wet``, ``sensor_only_wet`` and ``both_dry``.
    """
    vs1_wet = pairs["vs1_m"].notna()
    sensor_wet = pairs["sensor_state"] == "sensor wet"
    both = pairs[vs1_wet & sensor_wet]
    stats = (ap.paired_agreement(both["vs1_m"], both["sensor_depth_m"],
                                 both["event"], n_boot, seed)
             if len(both) > 1 else {})
    stats.update({
        "paired_images": len(pairs), "both_wet": int(len(both)),
        "vs1_only_wet": int((vs1_wet & ~sensor_wet).sum()),
        "sensor_only_wet": int((~vs1_wet & sensor_wet).sum()),
        "both_dry": int((~vs1_wet & ~sensor_wet).sum())})
    return stats


def timing_event_sets(timing):
    """
    Events behind each column of figure 1 (b) and (c), per panel (a) line.

    Parameters
    ----------
    timing : pandas.DataFrame
        Timing table of :func:`sensor_camera_comparison.collect`.

    Returns
    -------
    dict of str to set
        Panel (a) source to its events: each virtual sensor's own, the
        roadway mean the ROI's ("Anywhere on Roadway"), and the in-situ
        sensor the union of them all.
    """
    usable = timing.dropna(subset=["onset_lead_min"])
    sets = {(sc.ROI_STATS["MeanDepth"] if source == "ROI" else source):
            set(group["event"]) for source, group in usable.groupby("source")}
    sets[sc.SENSOR] = set().union(*sets.values()) if sets else set()
    return sets


def _draw_means_panel(ax, series, padding_hours, align="start",
                      dry="nan", event_sets=None, show_counts=False):
    """
    Draw panel (a): mean depth across events per source.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes.
    series : pandas.DataFrame
        Output of :func:`sensor_camera_comparison.event_series`.
    padding_hours : float
        Padding before the event sheet's start.
    align : {'start', 'peak'}, optional
        Hour 0 at the start of each event's window (every event), or at
        the in-situ sensor's peak (events where the sensor saw water).
    dry : {'nan', 'zero'}, optional
        ``'nan'``: dry times are left out, so a line is the mean among
        events wet at that time. ``'zero'``: dry times count as 0 m, so
        every event counts at every time in its window and a line is
        the expected depth then; the sensor reads 0 m outside the event
        sheet's start to end.
    event_sets : dict of str to set, optional
        Events to use per source (:func:`timing_event_sets`), so every
        line draws on the events of its column in panels (b) and (c);
        sources not listed are left out. Default: every event.
    show_counts : bool, optional
        Give each line's event count in the legend (always on with
        `event_sets`). Default False.

    Returns
    -------
    pandas.DataFrame
        The composites drawn.
    """
    # The sensor defines each event, so its line covers only the event
    # sheet's start to end; readings in the buffer belong to neighbouring
    # events whose padded windows overlap. Camera lines keep the whole
    # window: what they see before and after is the point of the panel.
    outside = (series["source"] == sc.SENSOR) & (
        (series["time"] < series["flood_start"])
        | (series["time"] > series["flood_end"]))
    if dry == "zero":
        # Outside its own flood period the event's sensor is dry; any
        # water there belongs to a neighbouring event
        series = series.assign(depth_m=series["depth_m"].fillna(0.0).mask(
            outside, 0.0))
    else:
        series = series[~outside]
    if event_sets is not None:
        series = series[[event in event_sets.get(source, ())
                         for source, event in zip(series["source"],
                                                  series["event"])]]
    aligned = sc.aligned_series(series, align, "buffer")
    table = sc.composites(aligned, stat="mean")
    counts = aligned.groupby("source")["event"].nunique()
    table["source_events"] = table["source"].map(counts)
    present = set(table["source"])
    sources = ([sc.SENSOR] if sc.SENSOR in present else []) + sorted(
        (s for s in present if s.startswith("VS")),
        key=lambda s: int(s[2:])) + [
        s for s in (sc.ROI_STATS["MeanDepth"],) if s in present]
    for source in sources:
        comp = table[table["source"] == source]
        ax.plot(comp["hours"], comp["center_m"],
                color=sc._series_style(source), linewidth=2.2,
                linestyle="--" if source == sc.SENSOR else "-",
                label=f"{_display(source)} ({counts[source]} events)"
                if show_counts or event_sets is not None
                else _display(source))
    events = aligned.drop_duplicates("event")
    if align == "start":
        ax.axvline(padding_hours, color=INK, linewidth=1, linestyle=":",
                   label="Event Start")
        durations = (aligned["flood_end"] - aligned["flood_start"]
                     ).dt.total_seconds() / 3600
        ax.set_xlim(0, float(np.nanpercentile(durations, 90))
                    + 2 * padding_hours)
        ax.set_xlabel("Hours Since Start of Window")
    else:
        ax.axvline(0, color=INK, linewidth=1, linestyle=":",
                   label="In-Situ Sensor Peak")
        before = (events["sensor_peak"] - events["buffer_start"]
                  ).dt.total_seconds() / 3600
        after = (events["flood_end"] - events["sensor_peak"]
                 ).dt.total_seconds() / 3600 + padding_hours
        ax.set_xlim(-float(np.nanpercentile(before, 90)),
                    float(np.nanpercentile(after, 90)))
        ax.set_xlabel("Hours From In-Situ Sensor Peak")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Water Depth (m)")
    ax.legend(loc="upper right", fontsize=8)
    return table


def _draw_timing_panel(ax, timing, column, ylabel, seed):
    """
    Draw a timing panel: events per source with a labelled mean bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes.
    timing : pandas.DataFrame
        Timing table of :func:`sensor_camera_comparison.collect`.
    column : {'onset_lead_min', 'end_lag_min'}
        Value to plot.
    ylabel : str
        Y-axis label.
    seed : int
        Seed of the horizontal jitter.

    Returns
    -------
    list of dict
        ``source``, ``events``, ``mean_min``, ``median_min`` per source.
    """
    usable = timing.dropna(subset=[column])
    sources = ["ROI"] + sorted(
        (s for s in usable["source"].unique() if s != "ROI"),
        key=lambda s: int(s[2:]))
    sources = [s for s in sources if (usable["source"] == s).any()]
    rng = np.random.default_rng(seed)
    rows = []
    for i, source in enumerate(sources):
        values = usable.loc[usable["source"] == source, column]
        color = BLUE if source == "ROI" else sc._series_style(source)
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(values)), values, s=26,
                   color=color, edgecolors="white", linewidths=0.6,
                   zorder=2)
        mean = values.mean()
        ax.plot([i - 0.3, i + 0.3], [mean, mean], color=INK, linewidth=2.2,
                zorder=3)
        # Label the mean above the plotting area so it covers no points
        ax.text(i, 1.01, f"Mean\n{mean:.0f} min", ha="center",
                va="bottom", fontsize=8, color=INK,
                transform=ax.get_xaxis_transform())
        rows.append({"source": source, "events": len(values),
                     "mean_min": mean, "median_min": values.median()})
    ax.axhline(0, color=INK, linewidth=1, linestyle="--", zorder=1)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([f"{_display(s).replace('Anywhere on ', 'Anywhere on' + chr(10))}\n"
                        f"({int((usable['source'] == s).sum())} events)"
                        for s in sources], fontsize=8)
    ax.set_xlim(-0.6, len(sources) - 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10]))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(ylabel)
    return rows


def panel_letter(ax, letter):
    """
    Label a panel ``(letter)`` above the top-left corner of its axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes.
    letter : str
        Panel letter.
    """
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(-42, 4), textcoords="offset points", fontsize=11,
                fontweight="bold", va="bottom")


def aligned_timing_axes(fig):
    """
    Lay out figure 1's panels on a figure or subfigure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Where to draw.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        Panels (a), (b), (c); (b) and (c) share their minutes axis.
    """
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.45,
                            wspace=0.25)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1], sharey=ax_b)  # one minutes scale
    return ax_a, ax_b, ax_c


def draw_aligned_timing(axes, series, timing, sensor, args,
                        letters=("a", "b", "c")):
    """
    Draw figure 1's three panels.

    Parameters
    ----------
    axes : tuple of matplotlib.axes.Axes
        Panels (a), (b), (c), e.g. from :func:`aligned_timing_axes`.
    series, timing : pandas.DataFrame
        From :func:`sensor_camera_comparison.collect` (good images).
    sensor : pandas.DataFrame
        All sensor readings.
    args : argparse.Namespace
        Parsed arguments (``align`` selects panel (a)'s time origin).
    letters : tuple of str, optional
        Panel letters.

    Returns
    -------
    dict
        ``composites`` (panel (a)), ``vs1`` (:func:`vs1_agreement`), and
        ``onset`` and ``end`` (per source event counts, means and
        medians).
    """
    ax_a, ax_b, ax_c = axes
    composites = _draw_means_panel(
        ax_a, series, args.padding_hours, getattr(args, "align", "start"),
        getattr(args, "dry", "nan"),
        timing_event_sets(timing) if getattr(args, "same_events", False)
        else None, getattr(args, "event_counts", False))
    pairs = vs1_sensor_pairs(series, sensor, args.match_minutes,
                             args.sensor_wet_ft * FEET_TO_METERS,
                             args.padding_hours)
    agreement = vs1_agreement(pairs, args.n_boot, args.seed)
    rows = _draw_timing_panel(ax_b, timing, "onset_lead_min",
                              "Onset Lead Time (minutes)", args.seed)
    rows_c = _draw_timing_panel(ax_c, timing, "end_lag_min",
                                "End Lag Time (minutes)", args.seed + 1)
    ax_c.tick_params(axis="y", labelleft=True)
    for ax, letter in zip(axes, letters):
        panel_letter(ax, letter)
    return {"composites": composites, "vs1": agreement, "onset": rows,
            "end": rows_c}


def write_aligned_timing_csvs(result, timing, out_dir, stem):
    """
    Write the CSVs behind figure 1.

    Parameters
    ----------
    result : dict
        Output of :func:`draw_aligned_timing`.
    timing : pandas.DataFrame
        Timing table drawn in panels (b) and (c).
    out_dir : str
        Output folder.
    stem : str
        Filename stem of the figure.
    """
    result["composites"].to_csv(os.path.join(out_dir, f"{stem}_a_means.csv"),
                                index=False)
    pd.DataFrame([result["vs1"]]).to_csv(
        os.path.join(out_dir, f"{stem}_a_vs1_agreement.csv"), index=False)
    pd.concat([pd.DataFrame(result["onset"]).assign(panel="b_onset_lead"),
               pd.DataFrame(result["end"]).assign(panel="c_end_lag")]).to_csv(
        os.path.join(out_dir, f"{stem}_bc_timing_means.csv"), index=False)
    timing.to_csv(os.path.join(out_dir, f"{stem}_timing.csv"), index=False)


def figure_aligned_timing(series, timing, sensor, args, stem):
    """
    Make figure 1 and its CSVs.

    Parameters
    ----------
    series, timing : pandas.DataFrame
        From :func:`sensor_camera_comparison.collect` (good images).
    sensor : pandas.DataFrame
        All sensor readings.
    args : argparse.Namespace
        Parsed arguments.
    stem : str
        Output filename stem.

    Returns
    -------
    dict
        Output of :func:`draw_aligned_timing`, for the metrics file.
    """
    fig = plt.figure(figsize=(11, 9))
    result = draw_aligned_timing(aligned_timing_axes(fig), series, timing,
                                 sensor, args)
    _save(fig, args.out_dir, stem)
    write_aligned_timing_csvs(result, timing, args.out_dir, stem)
    return result


# ----- Figure 2 --------------------------------------------------------------

#: Figure 2 series: label, column, colour, marker, line style.
DEPTH_SOURCES = (("Mean Roadway Depth in Field of View", "MeanDepth", BLUE,
                  "o", "-"),
                 ("Maximum Roadway Depth in Field of View", "MaxDepth",
                  MAGENTA, "s", "-."),
                 ("In-Situ Sensor Depth", "SensorDepth_m", ORANGE, "D",
                  "--"))


def _draw_bin(ax, x, width, rows):
    """
    Draw several sources side by side in one coverage bin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes.
    x : float
        Bin center.
    width : float
        Bin width.
    rows : list of tuple
        ``(row, color, marker)`` per source; ``row`` is one row of
        :func:`access_plotter.depth_by_coverage`.
    """
    n = len(rows)
    step = width * 0.28
    box = width * 0.24
    for k, (row, color, marker) in enumerate(rows):
        center = x + (k - (n - 1) / 2) * step
        ax.add_patch(plt.Rectangle(
            (center - box / 2, row["p10_depth_m"]), box,
            row["p90_depth_m"] - row["p10_depth_m"], facecolor=color,
            alpha=0.16, edgecolor="none", zorder=1))
        value = row["mean_depth_m"]
        low, high = row["mean_depth_m_ci95_low"], row["mean_depth_m_ci95_high"]
        if np.isfinite(low) and np.isfinite(high):
            ax.errorbar(center, value, yerr=[[value - low], [high - value]],
                        color=color, linewidth=1.4, capsize=2.5, zorder=3)
        ax.plot(center, value, marker=marker, markersize=6, linestyle="none",
                markerfacecolor="white" if row["low_sample"] else color,
                markeredgecolor=color, markeredgewidth=1.4, zorder=4)


def paired_images(data):
    """
    Images used by figure 2.

    Parameters
    ----------
    data : pandas.DataFrame
        Images of the ROI (one row each), from
        :func:`access_plotter.load_event`, with a sensor match.

    Returns
    -------
    pandas.DataFrame
        Well-segmented images with water on the road and every depth
        (mean, maximum, sensor).
    """
    return data[(data["Segmentation"] == "good")
                & (data["Coverage"] > 0)].dropna(
        subset=["Coverage", "MeanDepth", "MaxDepth", "SensorDepth_m"])


def binned_depth_tables(paired, args):
    """
    Depth by coverage bin of each figure 2 series.

    Parameters
    ----------
    paired : pandas.DataFrame
        Output of :func:`paired_images`.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    list of pandas.DataFrame
        :func:`access_plotter.depth_by_coverage` per
        :data:`DEPTH_SOURCES` entry.
    """
    return [ap.depth_by_coverage(paired["Coverage"], paired[column],
                                 paired["event"], args.coverage_bin_width,
                                 args.full_coverage, args.n_boot, args.seed)
            for _, column, *_ in DEPTH_SOURCES]


def binned_depth_top(tables):
    """
    Top of figure 2's depth axis: 5 % above the highest bar or band.

    Parameters
    ----------
    tables : list of pandas.DataFrame
        Output of :func:`binned_depth_tables`.

    Returns
    -------
    float
        Upper depth limit (m).
    """
    tops = [t[c].max() for t in tables
            for c in ("p90_depth_m", "mean_depth_m_ci95_high")]
    return 1.05 * float(np.nanmax(tops))


def binned_depth_axes(fig, spec=None):
    """
    Lay out figure 2's partial and full coverage panels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or matplotlib.figure.SubFigure
        Where to draw.
    spec : matplotlib.gridspec.SubplotSpec, optional
        Area to fill. Default: the whole figure.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        Partial coverage panel and the fully covered panel (shared
        depth axis).
    """
    widths = {"width_ratios": [5, 1.2], "wspace": 0.06}
    grid = (spec.subgridspec(1, 2, **widths) if spec is not None
            else fig.add_gridspec(1, 2, **widths))
    ax = fig.add_subplot(grid[0, 0])
    return ax, fig.add_subplot(grid[0, 1], sharey=ax)


def draw_binned_depth(ax, ax_full, tables, args, legend_fontsize=8,
                      ylim=None):
    """
    Draw figure 2 on its two panels.

    Parameters
    ----------
    ax, ax_full : matplotlib.axes.Axes
        Partial coverage panel and fully covered panel (shared y).
    tables : list of pandas.DataFrame
        Output of :func:`binned_depth_tables`.
    args : argparse.Namespace
        Parsed arguments.
    legend_fontsize : float, optional
        Legend font size.
    ylim : tuple of float, optional
        Depth-axis limits (m), e.g. shared between figures; a slightly
        negative bottom keeps markers at 0 m clear of the x-axis.
        Default: 0 to :func:`binned_depth_top`.
    """
    bottom, top = ylim or (0.0, binned_depth_top(tables))
    partial = [t[~t["full_coverage"]].reset_index(drop=True) for t in tables]
    full = [t[t["full_coverage"]] for t in tables]
    widths = partial[0]["coverage_high"] - partial[0]["coverage_low"]
    centers = partial[0]["coverage_low"] + widths / 2
    n = len(DEPTH_SOURCES)
    for i, (x, width) in enumerate(zip(centers, widths)):
        _draw_bin(ax, x, width, [(t.iloc[i], color, marker) for t, (
            _, _, color, marker, _) in zip(partial, DEPTH_SOURCES)])
    for k, (t, (_, _, color, _, style)) in enumerate(
            zip(partial, DEPTH_SOURCES)):
        ax.plot(centers + (k - (n - 1) / 2) * widths * 0.28,
                t["mean_depth_m"], color=color, linestyle=style,
                linewidth=1.6, zorder=2)
    if all(len(t) for t in full):
        _draw_bin(ax_full, 0.0, 1.0, [(t.iloc[0], color, marker) for t, (
            _, _, color, marker, _) in zip(full, DEPTH_SOURCES)])
        row = full[0].iloc[0]
        # Just above 0 m, so the note clears markers sitting there
        ax_full.text(0.5, (0.07 * top - bottom) / (top - bottom),
                     f"n = {row['images']} images,\n"
                     f"{row['events']} events",
                     transform=ax_full.transAxes, ha="center", va="bottom",
                     fontsize=8, color=INK)

    ax.set_xlim(0, args.full_coverage)
    ax.set_xlabel("Roadway Coverage (%)")
    ax.set_ylabel("Depth on Roadway (m)")
    ax.set_ylim(bottom, top)
    ax_full.set_xlim(-0.6, 0.6)
    ax_full.set_xticks([0.0])
    ax_full.set_xticklabels([f"Full\n(>={args.full_coverage:g}%)"])
    ax_full.tick_params(axis="y", left=False, labelleft=False)
    ax_full.spines["left"].set_visible(False)
    ax_full.grid(axis="x", visible=False)

    handles = [Line2D([0], [0], color=color, marker=marker, linestyle=style,
                      markersize=6, label=name)
               for name, _, color, marker, style in DEPTH_SOURCES]
    handles.append(Patch(facecolor=CONTEXT_GREY, alpha=0.5,
                         label="Mean with 95% CI; shading 10th-90th "
                               "percentile"))
    if any(t["low_sample"].any() for t in tables):
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=6,
            markerfacecolor="white", markeredgecolor=INK,
            label=f"Hollow: <{ap.MIN_BIN_IMAGES} images or "
                  f"<{ap.MIN_BIN_EVENTS} events"))
    ax.legend(handles=handles, loc="upper left", fontsize=legend_fontsize)


def binned_depth_agreements(paired, args):
    """
    Each image depth metric against the sensor for figure 2.

    Parameters
    ----------
    paired : pandas.DataFrame
        Output of :func:`paired_images`.
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict
        :func:`access_plotter.paired_agreement` per image metric name.
    """
    return {name: ap.paired_agreement(
        paired[column], paired["SensorDepth_m"], paired["event"],
        args.n_boot, args.seed) for name, column, *_ in DEPTH_SOURCES[:2]}


def write_binned_depth_csvs(tables, agreements, out_dir, stem):
    """
    Write the CSVs behind figure 2.

    Parameters
    ----------
    tables : list of pandas.DataFrame
        Output of :func:`binned_depth_tables`.
    agreements : dict
        Output of :func:`binned_depth_agreements`.
    out_dir : str
        Output folder.
    stem : str
        Filename stem of the figure.
    """
    pd.concat([t.assign(depth_source=name)
               for (name, *_), t in zip(DEPTH_SOURCES, tables)],
              ignore_index=True).to_csv(
        os.path.join(out_dir, f"{stem}_by_coverage.csv"), index=False)
    pd.DataFrame([dict(stats, image_metric=name)
                  for name, stats in agreements.items()]).to_csv(
        os.path.join(out_dir, f"{stem}_agreement.csv"), index=False)


def figure_binned_depth(data, args, stem):
    """
    Make figure 2 and its CSVs.

    Parameters
    ----------
    data : pandas.DataFrame
        Images of the ROI (one row each), from
        :func:`access_plotter.load_event`, with a sensor match.
    args : argparse.Namespace
        Parsed arguments.
    stem : str
        Output filename stem.

    Returns
    -------
    dict or None
        ``images``, ``events`` and ``agreements`` (per image metric,
        :func:`access_plotter.paired_agreement`) for the metrics file,
        or None if the figure was not made.
    """
    paired = paired_images(data)
    if len(paired) <= 3:
        logger.warning("Only %d paired images; figure 2 skipped.",
                       len(paired))
        return None
    tables = binned_depth_tables(paired, args)
    fig = plt.figure(figsize=(10, 5.5))
    draw_binned_depth(*binned_depth_axes(fig), tables, args)
    _save(fig, args.out_dir, stem)

    agreements = binned_depth_agreements(paired, args)
    write_binned_depth_csvs(tables, agreements, args.out_dir, stem)
    return {"images": len(paired), "events": paired["event"].nunique(),
            "agreements": agreements}


# ----- Metrics file ----------------------------------------------------------

def write_metrics(path, args, fig1, fig2):
    """
    Write the statistics behind both figures as Markdown.

    Parameters
    ----------
    path : str
        Output ``.md`` file.
    args : argparse.Namespace
        Parsed arguments.
    fig1, fig2 : dict or None
        Return values of :func:`figure_aligned_timing` and
        :func:`figure_binned_depth`.
    """
    def stats_rows(stats):
        return [f"| Bias (m) | {_interval(stats, 'bias_m')} |",
                f"| MAE (m) | {_interval(stats, 'mae_m')} |",
                f"| RMSE (m) | {_interval(stats, 'rmse_m')} |",
                f"| Pearson r | {_interval(stats, 'pearson_r', 2)} |"]

    lines = ["# Publication figure statistics", "",
             f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by "
             "`publication_figures.py` for ROI "
             f"`{args.roi}`. Well-segmented images only. Errors are "
             "camera minus in-situ sensor; values in brackets are 95% "
             f"confidence intervals from {args.n_boot} event resamples. The "
             f"in-situ sensor counts as wet above {args.sensor_wet_ft:g} ft.",
             ""]
    if fig1:
        vs1 = fig1["vs1"]
        lines += ["## Figure 1: event-aligned depths and timing", "",
                  "### VS1 against the in-situ sensor", ""]
        if "bias_m" in vs1:
            lines += [f"Depth error where both show water "
                      f"({vs1['both_wet']} images, {vs1['events']} events):",
                      "", "| Statistic | VS1 - sensor |", "|---|---|"]
            lines += stats_rows(vs1)
        lines += ["", f"Whether water is present, over "
                  f"{vs1['paired_images']} images with a sensor reading "
                  f"within {args.match_minutes:g} min:", "",
                  "| Both | VS1 only | Sensor only | Neither |",
                  "|---:|---:|---:|---:|",
                  f"| {vs1['both_wet']} | {vs1['vs1_only_wet']} | "
                  f"{vs1['sensor_only_wet']} | {vs1['both_dry']} |", "",
                  "### Panels (b) and (c): minutes relative to the in-situ "
                  "sensor", "",
                  "| Location | Events | Onset lead, mean | Onset lead, "
                  "median | End lag, mean | End lag, median |",
                  "|---|---:|---:|---:|---:|---:|"]
        ends = {row["source"]: row for row in fig1["end"]}
        for row in fig1["onset"]:
            end = ends.get(row["source"], {})
            lines.append(
                f"| {_display(row['source'])} | {row['events']} | "
                f"{row['mean_min']:.1f} | {row['median_min']:.1f} | "
                f"{end.get('mean_min', np.nan):.1f} | "
                f"{end.get('median_min', np.nan):.1f} |")
        lines += ["", "Events already wet at the edge of their image window "
                  "are included; their true lead or lag may be longer "
                  "(flags in `_fig1_aligned_timing_timing.csv`).", ""]
    if fig2:
        lines += ["## Figure 2: depth by roadway coverage", "",
                  f"Same {fig2['images']} images ({fig2['events']} events) "
                  f"for every series.", "",
                  "| Statistic | " + " | ".join(
                      f"{name} - sensor" for name in fig2["agreements"])
                  + " |", "|---|" + "---|" * len(fig2["agreements"])]
        for key, label, digits in (("bias_m", "Bias (m)", 3),
                                   ("mae_m", "MAE (m)", 3),
                                   ("rmse_m", "RMSE (m)", 3),
                                   ("pearson_r", "Pearson r", 2)):
            lines.append(f"| {label} | " + " | ".join(
                _interval(stats, key, digits)
                for stats in fig2["agreements"].values()) + " |")
        lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


# ----- CLI -------------------------------------------------------------------

def add_figure_arguments(parser):
    """
    Add the options shared by every publication figure script.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    """
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="Event bootstrap resamples for the CIs.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (bootstrap and jitter).")
    parser.add_argument("--padding_hours", type=float,
                        default=ap.EVENT_PADDING_HOURS,
                        help="Padding stage 05 added around each event.")
    parser.add_argument("--match_minutes", type=float, default=5.0,
                        help="Largest gap to a matched sensor reading.")
    parser.add_argument("--sensor_wet_ft", type=float,
                        default=FLOOD_START_THRESHOLD_FT,
                        help="Road water level (ft) above which the in-situ "
                             "sensor counts as wet.")
    parser.add_argument("--coverage_bin_width", type=float,
                        default=ap.COVERAGE_BIN_WIDTH,
                        help="Coverage bin width (%%) of figure 2.")
    parser.add_argument("--full_coverage", type=float,
                        default=ap.FULL_COVERAGE,
                        help="Coverage (%%) from which the road is fully "
                             "covered.")


def site_roi(site):
    """
    Default ROI of a site: the stem of camera_roi_json, else roi_json.

    Parameters
    ----------
    site : str
        Site config name or path.

    Returns
    -------
    str or None
        ROI name, or None if the config names no ROI JSON.
    """
    config = load_site_config(site)
    path = config.camera_roi_json or config.roi_json
    return os.path.splitext(os.path.basename(path))[0] if path else None


def load_roi_images(events_dir, roi, good, sensor, padding_hours):
    """
    Every image of one ROI across events, each kept once.

    Parameters
    ----------
    events_dir : str
        Folder of event subfolders.
    roi : str
        ROI name.
    good : set
        Well-segmented timestamps, from
        :func:`access_plotter.load_good_timestamps`.
    sensor : pandas.DataFrame
        All sensor readings.
    padding_hours : float
        Padding stage 05 added around each event.

    Returns
    -------
    pandas.DataFrame or None
        One row per image (:func:`access_plotter.load_event`), images in
        overlapping windows kept in the nearer event; None if the ROI has
        no accessibility CSVs.
    """
    paths = ap.find_roi_csvs(events_dir).get(roi, [])
    if not paths:
        return None
    loaded = pd.concat([ap.load_event(path, good, sensor)
                        for path in paths], ignore_index=True)
    keep = ap.nearest_event_mask(
        loaded["event"], pd.to_datetime(loaded["Time"], utc=True),
        padding_hours)
    return loaded[keep].reset_index(drop=True)


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
        Parsed arguments, with the ROI filled in from ``--site``.
    """
    parser = argparse.ArgumentParser(
        description="Make the publication figures.")
    parser.add_argument("--events_dir", required=True,
                        help="Folder of event subfolders.")
    parser.add_argument("--sensor", required=True,
                        help="Sensor readings CSV.")
    parser.add_argument("--labels", required=True,
                        help="segmentation_labels.csv (well-segmented "
                             "images only are used).")
    parser.add_argument("--out_dir", required=True, help="Output folder.")
    parser.add_argument("--site", default=None,
                        help="Site config supplying the default ROI.")
    parser.add_argument("--roi", default=None,
                        help="ROI name. Default with --site: the stem of "
                             "camera_roi_json, else roi_json.")
    parser.add_argument("--align", choices=("start", "peak"),
                        default="start",
                        help="Hour 0 of figure 1 (a): the start of each "
                             "event's window, or the in-situ sensor's "
                             "peak (events where it saw water).")
    parser.add_argument("--dry", choices=("nan", "zero"), default="nan",
                        help="Figure 1 (a): leave dry times out (mean "
                             "among wet events) or count them as 0 m "
                             "(expected depth over every event).")
    parser.add_argument("--same_events", action="store_true",
                        help="Figure 1 (a): draw each line from the events "
                             "of its column in (b) and (c) (the in-situ "
                             "sensor from all of them), with event counts "
                             "in the legend.")
    parser.add_argument("--event_counts", action="store_true",
                        help="Figure 1 (a): give each line's event count "
                             "in the legend.")
    add_figure_arguments(parser)
    args = parser.parse_args(argv)
    if args.roi is None and args.site:
        args.roi = site_roi(args.site)
    if args.roi is None:
        parser.error("give --roi, or a --site with an ROI JSON")
    return args


def main(argv=None):
    """
    Make every publication figure.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if there are no well-segmented images.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    good = ap.load_good_timestamps(args.labels)
    sensor = ap.load_sensor_data(args.sensor)
    if not good or sensor is None:
        logger.error("Publication figures need segmentation labels and "
                     "sensor readings.")
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = f"{args.roi}_pub"

    _, timing, series = sc.collect(
        args.events_dir, args.roi, sensor, good, args.match_minutes,
        args.padding_hours, wet_m=args.sensor_wet_ft * FEET_TO_METERS)
    fig1 = fig2 = None
    if series.empty:
        logger.warning("No well-segmented camera series for ROI '%s' "
                       "(event folders named <sensor>_<start>_<end>); "
                       "figure 1 skipped.", args.roi)
    else:
        fig1 = figure_aligned_timing(series, timing, sensor, args,
                                     f"{prefix}_fig1_aligned_timing")

    images = load_roi_images(args.events_dir, args.roi, good, sensor,
                             args.padding_hours)
    if images is not None:
        fig2 = figure_binned_depth(images, args,
                                   f"{prefix}_fig2_binned_depth")
    else:
        logger.warning("No accessibility CSVs for ROI '%s'; figure 2 "
                       "skipped.", args.roi)
    if fig1 or fig2:
        write_metrics(os.path.join(args.out_dir, f"{prefix}_metrics.md"),
                      args, fig1, fig2)
    made = sum(f is not None for f in (fig1, fig2))
    logger.info("%d publication figures written to %s", made, args.out_dir)
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
