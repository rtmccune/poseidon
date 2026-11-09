"""Compare the in-situ sensor with camera-derived depths across events.

For every flood event folder, the in-situ sensor (a point on the road)
is compared with two camera-derived sources: the virtual sensors (grid
pixels sampled from each image's depth map, stage 09) and the roadway
ROI (stage 10; by default the site's camera-view polygon). The aim is
to show whether the camera sees water on the road earlier and later
than the point sensor, and deeper water than it records.

Outputs in ``--out_dir``, once with every image (prefix
``<roi>_sensor_camera_all_images``) and, with ``--labels``, once with
well-segmented images only (``<roi>_sensor_camera_good_only``):

* ``_depths.jpg`` - every observation showing water, by source: the
  sensor, each virtual sensor, and the ROI's minimum, median, mean and
  maximum depth. Camera observations are coloured by the sensor's state
  at that moment: wet, dry (the camera saw water while the sensor did
  not), or no sensor reading within ``--match_minutes`` (including
  events the sensor missed entirely).
* ``_timing.jpg`` - per event and camera source, how many minutes
  earlier the camera first saw water than the sensor did (onset lead)
  and how much later it last saw water (end lag).
* ``_peaks.jpg`` - per event, the peak sensor depth against the peak
  ROI maximum, ROI mean and each virtual sensor's depth, with a 1:1
  line; events without sensor readings are shown in a side panel.

Virtual sensors keep the colours of the depth maps
(:data:`poseidon_utils.palette.VIRTUAL_SENSOR_COLORS`, by their order in
the site config) in every figure.
* ``_aligned_start.jpg`` and ``_aligned_peak.jpg`` - every event's
  depth series on a shared clock: hour 0 at the flood start (from the
  event folder's window without its padding; every event) or at the
  in-situ sensor's peak (events with sensor readings). One panel per
  source (sensor, virtual sensors, ROI mean) with each event as a thin
  line and the mean (``--aligned_stat``; or median) across events
  (15-min bins, >= 3 events) in bold with its interquartile range; a
  last panel overlays them (``_aligned_composites.csv``). By default
  dry moments are left out (``--aligned_dry nan``), so the bold line
  is the typical depth among events wet at that time; ``zero`` counts
  them as 0 m. Camera series exist only where images do (daylight).
* ``_observations.csv``, ``_timing.csv``, ``_summary.csv`` and
  ``_summary.md`` - the data behind the figures, and counts.

Definitions
-----------
* **Water.** Sensor: road water level above 0 (depth = level, clipped
  at 0). Virtual sensor: a depth above 0. ROI: at least one ROI cell
  with depth above 0 (``FloodedPixels > 0``); its minimum can then be
  0 m. Depths are in meters.
* **Which images.** The ``all_images`` set uses every image; the
  ``good_only`` set (with ``--labels``) uses well-segmented images only,
  for every camera source (virtual sensors and ROI alike).
* **Fair timing.** The camera only sees in daylight within each event's
  image window, so onset and end are compared within the span of that
  event's (well-segmented) images: the sensor's first and last wet
  reading inside the span against the camera's. Onset lead = sensor
  onset - camera onset (> 0: camera first). End lag = camera end -
  sensor end (> 0: camera later). If the first (last) observation in
  the span is already wet, the true onset (end) lies outside it; that
  value is *censored* (drawn hollow) and the real lead or lag may be
  larger or smaller. Events where the camera saw water but the sensor
  never did within the span are counted as "camera only".
* **Event window.** Sensor readings are taken from the event folder's
  padded window (its name: ``<sensor>_<start>_<end>``).

Example
-------
``python sensor_camera_comparison.py --site down_east --events_dir
data/down_east/flood_events --sensor
data/down_east/sensor_data_comparison.csv --labels
data/down_east/segmentation_labels.csv --out_dir data/down_east/plots``
"""

import argparse
import glob
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless HPC nodes

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from poseidon_utils.logging_utils import setup_logging  # noqa: E402
from poseidon_utils.naming import TIMESTAMP_FORMAT  # noqa: E402
from poseidon_utils.site_config import load_site_config  # noqa: E402
from poseidon_utils.zarr_io import read_zip_arrays  # noqa: E402

from access_plotter import (  # noqa: E402
    EVENT_PADDING_HOURS,
    as_utc_ns,
    event_window,
    load_good_timestamps,
    load_sensor_data,
    nearest_event_mask,
)
from plot_style import (  # noqa: E402
    BLUE,
    CONTEXT_GREY,
    INK,
    INK_STRONG,
    MAGENTA,
    ORANGE,
    apply_style,
    virtual_sensor_color,
)

logger = logging.getLogger(__name__)

#: ROI depth statistics compared, with display names.
ROI_STATS = {"MinDepth": "ROI min", "MedianDepth": "ROI median",
             "MeanDepth": "ROI mean", "MaxDepth": "ROI max"}

#: Name of the in-situ sensor source.
SENSOR = "In-situ sensor"

#: Sensor state of a camera observation, with its colour.
STATE_COLORS = {"sensor wet": BLUE, "sensor dry": ORANGE,
                "no sensor reading": MAGENTA}


# ----- Loading ---------------------------------------------------------------

def _keep_good(times, good_timestamps):
    """
    Mask of well-segmented times (all True without labels).

    Parameters
    ----------
    times : pandas.Series or pandas.DatetimeIndex
        UTC image times.
    good_timestamps : set of str
        Well-segmented ``YYYYMMDDHHMMSS`` stamps; empty keeps all.

    Returns
    -------
    numpy.ndarray
        Boolean mask.
    """
    if not good_timestamps:
        return np.ones(len(times), dtype=bool)
    stamps = pd.DatetimeIndex(times).strftime(TIMESTAMP_FORMAT)
    return np.asarray(stamps.isin(good_timestamps))


def load_roi(event_dir, roi, good_timestamps):
    """
    Read one event's ROI depth statistics per image.

    Parameters
    ----------
    event_dir : str
        Event folder.
    roi : str
        ROI name (accessibility CSV prefix).
    good_timestamps : set of str
        Well-segmented stamps; empty keeps every image.

    Returns
    -------
    pandas.DataFrame or None
        ``time``, ``wet`` and the :data:`ROI_STATS` columns, or None if
        the CSV is missing.
    """
    path = os.path.join(event_dir, f"{roi}_accessibility_time_series.csv")
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    table["time"] = pd.to_datetime(table["Time"], utc=True)
    table["wet"] = table["FloodedPixels"] > 0
    table = table[_keep_good(table["time"], good_timestamps)]
    return table[["time", "wet", *ROI_STATS]].sort_values("time")


def load_virtual_sensors(event_dir, good_timestamps):
    """
    Read one event's virtual sensor depths per image.

    Parameters
    ----------
    event_dir : str
        Event folder with ``zarr/virtual_sensor_data.zip``.
    good_timestamps : set of str
        Well-segmented stamps; empty keeps every image.

    Returns
    -------
    pandas.DataFrame or None
        Long table ``time``, ``source`` (``VS1``...), ``depth_m``
        (NaN where dry) and ``wet``, or None if the store is missing.
    """
    path = os.path.join(event_dir, "zarr", "virtual_sensor_data.zip")
    if not os.path.exists(path):
        return None
    arrays = read_zip_arrays(path)
    times = pd.to_datetime(pd.Series(arrays["timestamps"]), utc=True)
    depths = np.asarray(arrays["vs_depth"], dtype=float)
    keep = _keep_good(times, good_timestamps)
    frames = []
    for i in range(depths.shape[1]):
        frames.append(pd.DataFrame({
            "time": times[keep].to_numpy(), "source": f"VS{i + 1}",
            "depth_m": depths[keep, i]}))
    table = pd.concat(frames, ignore_index=True)
    table["time"] = pd.to_datetime(table["time"], utc=True)
    table["wet"] = table["depth_m"] > 0
    return table.sort_values(["source", "time"])


# ----- Comparison ------------------------------------------------------------

def sensor_state(times, sensor, match_minutes):
    """
    The sensor's depth and state at each camera time.

    Parameters
    ----------
    times : pandas.Series
        UTC camera times.
    sensor : pandas.DataFrame or None
        Readings with ``datetime_utc`` and ``SensorDepth_m``.
    match_minutes : float
        Largest gap to the nearest reading.

    Returns
    -------
    pandas.DataFrame
        ``sensor_depth_m`` and ``sensor_state`` (a key of
        :data:`STATE_COLORS`), aligned with `times`.
    """
    frame = pd.DataFrame({"time": as_utc_ns(pd.Series(times).to_numpy()),
                          "order": np.arange(len(times))})
    if sensor is None or sensor.empty:
        frame["sensor_depth_m"] = np.nan
    else:
        frame = pd.merge_asof(
            frame.sort_values("time"),
            pd.DataFrame({
                "time": as_utc_ns(sensor["datetime_utc"]),
                "sensor_depth_m": sensor["SensorDepth_m"].to_numpy()}),
            on="time", direction="nearest",
            tolerance=pd.Timedelta(minutes=match_minutes)).sort_values(
                "order")
    depth = frame["sensor_depth_m"].to_numpy()
    state = np.where(np.isnan(depth), "no sensor reading",
                     np.where(depth > 0, "sensor wet", "sensor dry"))
    return pd.DataFrame({"sensor_depth_m": depth, "sensor_state": state})


def wet_span(times, wet):
    """
    First and last wet time, and whether each is censored.

    Parameters
    ----------
    times : array_like
        Sorted UTC times.
    wet : array_like of bool
        Whether each time shows water.

    Returns
    -------
    dict
        ``first``, ``last`` (NaT if never wet), ``onset_censored``
        (the first observation is already wet) and ``end_censored``
        (the last observation is still wet).
    """
    times = pd.DatetimeIndex(times)
    wet = np.asarray(wet, dtype=bool)
    if not wet.any():
        return {"first": pd.NaT, "last": pd.NaT, "onset_censored": False,
                "end_censored": False}
    return {"first": times[wet][0], "last": times[wet][-1],
            "onset_censored": bool(wet[0]), "end_censored": bool(wet[-1])}


def event_timing(event, camera_sources, sensor_window):
    """
    Onset lead and end lag of each camera source over the sensor.

    Parameters
    ----------
    event : str
        Event name.
    camera_sources : dict of str to pandas.DataFrame
        Source name to its ``time`` / ``wet`` observations.
    sensor_window : pandas.DataFrame or None
        Sensor readings; only those within the event's image span are used.

    Returns
    -------
    list of dict
        One row per camera source.
    """
    rows = []
    for source, obs in camera_sources.items():
        if obs.empty:
            continue
        start, end = obs["time"].min(), obs["time"].max()
        camera = wet_span(obs["time"], obs["wet"])
        row = {"event": event, "source": source,
               "camera_images": len(obs), "span_start": start,
               "span_end": end, "camera_first_wet": camera["first"],
               "camera_last_wet": camera["last"],
               "camera_onset_censored": camera["onset_censored"],
               "camera_end_censored": camera["end_censored"],
               "sensor_readings_in_span": 0,
               "sensor_first_wet": pd.NaT, "sensor_last_wet": pd.NaT,
               "sensor_onset_censored": False, "sensor_end_censored": False,
               "onset_lead_min": np.nan, "end_lag_min": np.nan,
               "camera_only": False}
        if sensor_window is not None:
            inside = sensor_window[(sensor_window["datetime_utc"] >= start)
                                   & (sensor_window["datetime_utc"] <= end)]
            row["sensor_readings_in_span"] = len(inside)
            if len(inside):
                sensor = wet_span(inside["datetime_utc"],
                                  inside["SensorDepth_m"] > 0)
                row.update(sensor_first_wet=sensor["first"],
                           sensor_last_wet=sensor["last"],
                           sensor_onset_censored=sensor["onset_censored"],
                           sensor_end_censored=sensor["end_censored"])
                if pd.notna(sensor["first"]) and pd.notna(camera["first"]):
                    row["onset_lead_min"] = (
                        sensor["first"] - camera["first"]).total_seconds() / 60
                    row["end_lag_min"] = (
                        camera["last"] - sensor["last"]).total_seconds() / 60
                row["camera_only"] = bool(pd.notna(camera["first"])
                                          and pd.isna(sensor["first"]))
        rows.append(row)
    return rows


def _owned(frame, keep, event):
    """
    Rows of one event's table that it owns after de-duplication.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table with a ``time`` column.
    keep : set of tuple
        ``(event, time in ns)`` pairs kept by
        :func:`access_plotter.nearest_event_mask`.
    event : str
        Event name.

    Returns
    -------
    pandas.DataFrame
        The owned rows.
    """
    stamps = as_utc_ns(frame["time"]).astype("int64")
    return frame[[(event, t) in keep for t in stamps]]


def collect(events_dir, roi, sensor, good_timestamps, match_minutes,
            padding_hours=EVENT_PADDING_HOURS):
    """
    Gather observations and timing for every event.

    Where padded event windows overlap, an image (or sensor reading)
    sits in two event folders; it is kept once, in the event whose
    flood period is nearest (:func:`access_plotter.nearest_event_mask`),
    so pooled counts and depths count it once. The sensor's state at a
    camera time is matched against every sensor reading, and timing uses
    every reading within the event's image span.

    Parameters
    ----------
    events_dir : str
        Folder of event subfolders.
    roi : str
        ROI name.
    sensor : pandas.DataFrame or None
        All sensor readings (:func:`access_plotter.load_sensor_data`).
    good_timestamps : set of str
        Well-segmented stamps; empty keeps every image.
    match_minutes : float
        Largest gap for matching a camera time to a sensor reading.
    padding_hours : float, optional
        Padding stage 05 added around each event.

    Returns
    -------
    observations : pandas.DataFrame
        Long table ``event``, ``has_sensor``, ``time``, ``source``,
        ``depth_m``, ``sensor_depth_m``, ``sensor_state``: every
        observation that shows water, each image and reading once.
    timing : pandas.DataFrame
        One row per event and camera source (:func:`event_timing`).
    """
    loaded = []
    for event_dir in sorted(glob.glob(os.path.join(events_dir, "*", ""))):
        event_dir = os.path.normpath(event_dir)
        window = event_window(event_dir)
        roi_obs = load_roi(event_dir, roi, good_timestamps)
        vs_obs = load_virtual_sensors(event_dir, good_timestamps)
        if window is None or (roi_obs is None and vs_obs is None):
            continue
        sensor_window = None
        if sensor is not None:
            sensor_window = sensor[(sensor["datetime_utc"] >= window[0])
                                   & (sensor["datetime_utc"] <= window[1])]
            sensor_window = sensor_window.assign(
                time=sensor_window["datetime_utc"])
        loaded.append((os.path.basename(event_dir), roi_obs, vs_obs,
                       sensor_window))

    def ownership(tables):
        frames = [pd.DataFrame({"event": event, "time": as_utc_ns(
                      table["time"]).to_numpy()}).drop_duplicates()
                  for event, table in tables
                  if table is not None and len(table)]
        if not frames:
            return set()
        pairs = pd.concat(frames, ignore_index=True)
        mask = nearest_event_mask(pairs["event"], pairs["time"],
                                  padding_hours)
        kept = pairs[mask]
        return set(zip(kept["event"],
                       as_utc_ns(kept["time"]).astype("int64")))

    image_keep = ownership(
        [(event, table) for event, roi_obs, vs_obs, _ in loaded
         for table in (roi_obs, vs_obs)])
    sensor_keep = ownership([(event, window)
                             for event, _, _, window in loaded])

    observations, timing = [], []
    for event, roi_obs, vs_obs, sensor_window in loaded:
        has_sensor = sensor_window is not None and not sensor_window.empty
        if roi_obs is not None:
            roi_obs = _owned(roi_obs, image_keep, event)
        if vs_obs is not None:
            vs_obs = _owned(vs_obs, image_keep, event)

        camera_sources = {}
        if roi_obs is not None and len(roi_obs):
            camera_sources["ROI"] = roi_obs[["time", "wet"]]
            wet = roi_obs[roi_obs["wet"]]
            state = sensor_state(wet["time"], sensor, match_minutes)
            for column, name in ROI_STATS.items():
                observations.append(pd.DataFrame({
                    "event": event, "has_sensor": has_sensor,
                    "time": wet["time"].to_numpy(), "source": name,
                    "depth_m": wet[column].to_numpy(),
                    "sensor_depth_m": state["sensor_depth_m"].to_numpy(),
                    "sensor_state": state["sensor_state"].to_numpy()}))
        if vs_obs is not None and len(vs_obs):
            for source, group in vs_obs.groupby("source", sort=True):
                camera_sources[source] = group[["time", "wet"]]
                wet = group[group["wet"]]
                state = sensor_state(wet["time"], sensor, match_minutes)
                observations.append(pd.DataFrame({
                    "event": event, "has_sensor": has_sensor,
                    "time": wet["time"].to_numpy(), "source": source,
                    "depth_m": wet["depth_m"].to_numpy(),
                    "sensor_depth_m": state["sensor_depth_m"].to_numpy(),
                    "sensor_state": state["sensor_state"].to_numpy()}))
        if has_sensor:
            owned = _owned(sensor_window, sensor_keep, event)
            wet = owned[owned["SensorDepth_m"] > 0]
            observations.append(pd.DataFrame({
                "event": event, "has_sensor": True,
                "time": wet["datetime_utc"].to_numpy(), "source": SENSOR,
                "depth_m": wet["SensorDepth_m"].to_numpy(),
                "sensor_depth_m": wet["SensorDepth_m"].to_numpy(),
                "sensor_state": "sensor wet"}))
        timing += event_timing(event, camera_sources,
                               sensor if has_sensor else None)

    columns = ["event", "has_sensor", "time", "source", "depth_m",
               "sensor_depth_m", "sensor_state"]
    observations = (pd.concat(observations, ignore_index=True)
                    if observations else pd.DataFrame(columns=columns))
    return observations, pd.DataFrame(timing), event_series(
        loaded, padding_hours)


def event_series(loaded, padding_hours=EVENT_PADDING_HOURS):
    """
    Full depth time series of each event and source, for aligned plots.

    Unlike the pooled observations, every event keeps its whole padded
    window (an image in two overlapping windows appears in both
    events' series, at different times relative to each). Dry moments
    are NaN (the sensor's 0 m included); :func:`plot_aligned` decides
    whether to treat them as missing or as 0 m.

    Parameters
    ----------
    loaded : list of tuple
        ``(event, roi_obs, vs_obs, sensor_window)`` as gathered by
        :func:`collect`.
    padding_hours : float, optional
        Padding stage 05 added; removed to get the flood period.

    Returns
    -------
    pandas.DataFrame
        ``event``, ``time``, ``source`` (sensor, ``VS1``..., ROI mean),
        ``depth_m`` (NaN when dry), plus the event's ``buffer_start``
        (the event folder's window start, i.e. flood start minus the
        padding), ``flood_start`` and ``flood_end`` (the window without
        its padding: the event sheet's start and end) and
        ``sensor_peak`` (time of the sensor's
        maximum in the flood period; NaT if the sensor has no readings
        or never saw water there).
    """
    pad = pd.Timedelta(hours=padding_hours)
    frames = []
    for event, roi_obs, vs_obs, sensor_window in loaded:
        window = event_window(event)
        start, end = window[0] + pad, window[1] - pad
        if start > end:
            start = end = window[0] + (window[1] - window[0]) / 2
        parts = []
        peak = pd.NaT
        if sensor_window is not None and len(sensor_window):
            depth = sensor_window["SensorDepth_m"].to_numpy(dtype=float)
            parts.append(pd.DataFrame({
                "time": sensor_window["datetime_utc"].to_numpy(),
                "source": SENSOR,
                "depth_m": np.where(depth > 0, depth, np.nan)}))
            flood = sensor_window[(sensor_window["datetime_utc"] >= start)
                                  & (sensor_window["datetime_utc"] <= end)]
            if flood.empty:
                flood = sensor_window
            # Only a sensor that saw water has a peak to align on
            if (flood["SensorDepth_m"] > 0).any():
                peak = flood.loc[flood["SensorDepth_m"].idxmax(),
                                 "datetime_utc"]
        if vs_obs is not None and len(vs_obs):
            parts.append(vs_obs.assign(
                depth_m=vs_obs["depth_m"].where(vs_obs["depth_m"] > 0))[
                ["time", "source", "depth_m"]])
        if roi_obs is not None and len(roi_obs):
            parts.append(pd.DataFrame({
                "time": roi_obs["time"].to_numpy(),
                "source": ROI_STATS["MeanDepth"],
                "depth_m": roi_obs["MeanDepth"].where(
                    roi_obs["wet"]).to_numpy()}))
        for part in parts:
            frames.append(part.assign(event=event, buffer_start=window[0],
                                      flood_start=start, flood_end=end,
                                      sensor_peak=peak))
    if not frames:
        return pd.DataFrame(columns=["event", "time", "source", "depth_m",
                                     "buffer_start", "flood_start",
                                     "flood_end", "sensor_peak"])
    series = pd.concat(frames, ignore_index=True)
    for column in ("time", "buffer_start", "flood_start", "flood_end",
                   "sensor_peak"):
        series[column] = as_utc_ns(series[column])
    return series


def source_order(observations):
    """
    Plotting order of the sources present.

    Parameters
    ----------
    observations : pandas.DataFrame
        Output of :func:`collect`.

    Returns
    -------
    list of str
        Sensor, virtual sensors, then ROI statistics.
    """
    present = set(observations["source"])
    vs = sorted((s for s in present if s.startswith("VS")),
                key=lambda s: int(s[2:]))
    return ([SENSOR] if SENSOR in present else []) + vs + [
        name for name in ROI_STATS.values() if name in present]


# ----- Figures ---------------------------------------------------------------

def plot_depths(observations, path, seed=0, title=None):
    """
    Draw every water observation by source, coloured by sensor state.

    Boxes span the quartiles, whiskers the 10th-90th percentiles, and
    the line marks the median.

    Parameters
    ----------
    observations : pandas.DataFrame
        Output of :func:`collect`.
    path : str
        Output image.
    seed : int, optional
        Seed of the horizontal jitter.
    title : str, optional
        Figure title (e.g. the image set). Default None.
    """
    order = source_order(observations)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(order) + 2), 6))
    for i, source in enumerate(order):
        group = observations[observations["source"] == source]
        values = group["depth_m"].dropna().to_numpy()
        if values.size == 0:
            continue
        for state, color in STATE_COLORS.items():
            subset = group[group["sensor_state"] == state]
            ax.scatter(i + rng.uniform(-0.3, 0.3, len(subset)),
                       subset["depth_m"], s=7, color=color, alpha=0.45,
                       edgecolors="none", zorder=1)
        p10, q1, median, q3, p90 = np.percentile(values, [10, 25, 50, 75, 90])
        ax.add_patch(plt.Rectangle((i - 0.35, q1), 0.7, q3 - q1,
                                   facecolor="none", edgecolor=INK,
                                   linewidth=1.4, zorder=3))
        ax.plot([i - 0.35, i + 0.35], [median, median], color=INK,
                linewidth=2.2, zorder=3)
        ax.vlines([i, i], [p10, q3], [q1, p90], color=INK, linewidth=1.2,
                  zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([
        f"{s}\n(n={int((observations['source'] == s).sum())})"
        for s in order], fontsize=8)
    ax.tick_params(axis="x", pad=4)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    # Map-colour swatch under each virtual sensor's label
    for i, source in enumerate(order):
        if source.startswith("VS"):
            ax.plot(i, -0.13, marker="s", markersize=9, clip_on=False,
                    color=virtual_sensor_color(int(source[2:]) - 1),
                    transform=ax.get_xaxis_transform())
    ax.set_ylabel("Depth When Water Is Present (m)")
    ax.grid(axis="x", visible=False)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=color,
                      markersize=6, label=label)
               for label, color in (
                   ("Sensor wet at that time", BLUE),
                   ("Sensor dry: camera saw water first or longer", ORANGE),
                   ("No sensor reading (incl. events the sensor missed)",
                    MAGENTA))]
    handles.append(Line2D([0], [0], color=INK, linewidth=2,
                          label="Box: quartiles and median; whiskers: "
                                "10th-90th pct."))
    if any(source.startswith("VS") for source in order):
        handles.append(Line2D([0], [0], marker="s", linestyle="none",
                              color=CONTEXT_GREY, markersize=7,
                              label="Square under VS label: its colour on "
                                    "the depth maps"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, fontsize=8)
    if title:
        ax.text(0, 1.16, title, transform=ax.transAxes, fontsize=11,
                va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timing(timing, path, seed=0, title=None):
    """
    Draw each event's onset lead and end lag per camera source.

    Parameters
    ----------
    timing : pandas.DataFrame
        Output of :func:`collect`.
    path : str
        Output image.
    seed : int, optional
        Seed of the horizontal jitter.
    title : str, optional
        Figure title (e.g. the image set). Default None.
    """
    usable = timing.dropna(subset=["onset_lead_min"])
    sources = ["ROI"] + sorted(
        (s for s in usable["source"].unique() if s != "ROI"),
        key=lambda s: int(s[2:]) if s[2:].isdigit() else 0)
    sources = [s for s in sources if (usable["source"] == s).any()]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, column, censor_cols, label in (
            (axes[0], "onset_lead_min",
             ("camera_onset_censored", "sensor_onset_censored"),
             "Onset lead (min): water at the camera location\n"
             "before it reached the in-situ sensor"),
            (axes[1], "end_lag_min",
             ("camera_end_censored", "sensor_end_censored"),
             "End lag (min): water stayed at the camera location\n"
             "after it left the in-situ sensor")):
        for i, source in enumerate(sources):
            group = usable[usable["source"] == source]
            censored = group[list(censor_cols)].any(axis=1).to_numpy()
            x = i + rng.uniform(-0.25, 0.25, len(group))
            color = (BLUE if source == "ROI"
                     else virtual_sensor_color(int(source[2:]) - 1))
            ax.scatter(x[~censored], group[column][~censored], s=30,
                       color=color, edgecolors="white", linewidths=0.6,
                       zorder=2)
            ax.scatter(x[censored], group[column][censored], s=30,
                       facecolors="white", edgecolors=color, linewidths=1.2,
                       zorder=2)
            if len(group):
                median = group[column].median()
                ax.plot([i - 0.35, i + 0.35], [median, median], color=INK,
                        linewidth=2, zorder=3)
        ax.axhline(0, color=INK, linewidth=1, linestyle="--", zorder=1)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([
            f"{'Anywhere on ROI' if s == 'ROI' else s}\n"
            f"({int((usable['source'] == s).sum())} events)"
            for s in sources], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=CONTEXT_GREY,
               markersize=6, label="Event"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="white", markeredgecolor=INK,
               label="Censored: wet at the image window's edge"),
        Line2D([0], [0], color=INK, linewidth=2, label="Median")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.04))
    if title:
        fig.suptitle(title, y=1.1, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def event_peaks(observations):
    """
    Peak depth per event and source.

    Parameters
    ----------
    observations : pandas.DataFrame
        Output of :func:`collect`.

    Returns
    -------
    pandas.DataFrame
        One row per event: ``has_sensor``, ``sensor`` (peak sensor
        depth), ``roi_max``, ``roi_mean`` and one column per virtual
        sensor (``VS1``...).
    """
    source = observations["source"]
    obs = observations.assign(
        group=np.select(
            [source == SENSOR, source == ROI_STATS["MaxDepth"],
             source == ROI_STATS["MeanDepth"], source.str.startswith("VS")],
            ["sensor", "roi_max", "roi_mean", source], ""))
    peaks = obs[obs["group"] != ""].pivot_table(
        index="event", columns="group", values="depth_m", aggfunc="max")
    if "sensor" not in peaks:  # no event had sensor readings
        peaks["sensor"] = np.nan
    has_sensor = observations.groupby("event")["has_sensor"].any()
    return peaks.join(has_sensor).reset_index()


def plot_peaks(peaks, path, seed=0, title=None):
    """
    Draw peak camera depths against the peak sensor depth per event.

    Parameters
    ----------
    peaks : pandas.DataFrame
        Output of :func:`event_peaks`.
    path : str
        Output image.
    seed : int, optional
        Seed of the horizontal jitter in the side panel.
    title : str, optional
        Figure title (e.g. the image set). Default None.
    """
    # ROI in blue and dark grey; virtual sensors in their map colours
    vs_columns = sorted((c for c in peaks if str(c).startswith("VS")),
                        key=lambda c: int(c[2:]))
    series = [(c, label, color, marker) for c, label, color, marker in (
        ("roi_max", "ROI max", BLUE, "o"),
        ("roi_mean", "ROI mean", INK, "D"))
        if c in peaks] + [
        (c, c, virtual_sensor_color(int(c[2:]) - 1), "s")
        for c in vs_columns]
    with_sensor = peaks[peaks["has_sensor"] & peaks.get(
        "sensor", pd.Series(np.nan, index=peaks.index)).notna()]
    without = peaks[~peaks.index.isin(with_sensor.index)]
    fig, (ax, side) = plt.subplots(
        1, 2, figsize=(10, 6), sharey=True,
        gridspec_kw={"width_ratios": [4, 1.2], "wspace": 0.06})
    for column, label, color, marker in series:
        ax.scatter(with_sensor["sensor"], with_sensor[column], color=color,
                   marker=marker, s=34, edgecolors="white", linewidths=0.6,
                   label=label, zorder=2)
    columns = [c for c, *_ in series]
    top = np.nanmax(peaks[[*columns, *(["sensor"] if "sensor" in peaks
                                       else [])]].to_numpy(dtype=float))
    top = 1.05 * top if np.isfinite(top) and top > 0 else 1.0
    ax.plot([0, top], [0, top], color=INK, linestyle="--", linewidth=1.2,
            label="1:1", zorder=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Peak In-Situ Sensor Depth (m)")
    ax.set_ylabel("Peak Camera-Derived Depth (m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{len(with_sensor)} event"
                 f"{'s' * (len(with_sensor) != 1)} with sensor readings",
                 fontsize=10)

    rng = np.random.default_rng(seed)
    for column, label, color, marker in series:
        side.scatter(rng.uniform(-0.3, 0.3, len(without)), without[column],
                     color=color, marker=marker, s=34, edgecolors="white",
                     linewidths=0.6, zorder=2)
    side.set_xlim(-0.6, 0.6)
    side.set_xticks([0])
    side.set_xticklabels(["No sensor\nreadings"])
    side.set_title(f"{len(without)} event{'s' * (len(without) != 1)}",
                   fontsize=10)
    side.tick_params(axis="y", left=False)
    side.spines["left"].set_visible(False)
    side.grid(axis="x", visible=False)
    if title:
        fig.suptitle(title, y=1.0, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


#: Composite (median across events) time bin, minutes.
COMPOSITE_MINUTES = 15

#: Fewest events a composite bin needs to be drawn.
MIN_COMPOSITE_EVENTS = 3


def aligned_series(series, align, start_origin="buffer"):
    """
    Add hours relative to each event's start or sensor peak.

    Parameters
    ----------
    series : pandas.DataFrame
        Output of :func:`event_series`.
    align : {'start', 'peak'}
        ``'start'``: hour 0 is the event's start (every event).
        ``'peak'``: hour 0 is the in-situ sensor's peak (events where
        the sensor recorded water).
    start_origin : {'buffer', 'flood'}, optional
        For ``'start'``: the buffered window's start (the event folder
        name's first timestamp; default) or the flood start from the
        event sheet (``padding`` hours later).

    Returns
    -------
    pandas.DataFrame
        `series` with ``hours``.
    """
    if align == "start":
        origin = series["buffer_start" if start_origin == "buffer"
                        else "flood_start"]
    else:
        origin = series["sensor_peak"]
    out = series.assign(hours=(series["time"] - origin).dt.total_seconds()
                        / 3600)
    return out.dropna(subset=["hours"])


def composites(aligned, minutes=COMPOSITE_MINUTES,
               min_events=MIN_COMPOSITE_EVENTS, stat="mean"):
    """
    Typical depth and interquartile range across events, per bin.

    Each event first contributes its mean depth in each time bin, so
    every event counts equally however often it was sampled. Dry
    (NaN) readings are skipped, so a bin's ``events`` are those with
    water then (unless dry times were set to 0 m beforehand).

    Parameters
    ----------
    aligned : pandas.DataFrame
        Output of :func:`aligned_series`.
    minutes : float, optional
        Bin width.
    min_events : int, optional
        Bins with fewer contributing events are dropped.
    stat : {'mean', 'median'}, optional
        Statistic across events drawn as the bold line. Default mean.

    Returns
    -------
    pandas.DataFrame
        ``source``, ``hours`` (bin center), ``center_m`` (the mean or
        median across events), ``mean_m``, ``median_m``, ``q1_m``,
        ``q3_m``, ``events``.
    """
    width = minutes / 60
    binned = aligned.dropna(subset=["depth_m"]).assign(
        hours=lambda d: (np.floor(d["hours"] / width) + 0.5) * width)
    per_event = binned.groupby(["source", "hours", "event"])[
        "depth_m"].mean().reset_index()
    grouped = per_event.groupby(["source", "hours"])["depth_m"]
    table = pd.DataFrame({
        "mean_m": grouped.mean(), "median_m": grouped.median(),
        "q1_m": grouped.quantile(0.25), "q3_m": grouped.quantile(0.75),
        "events": grouped.size()}).reset_index()
    table["center_m"] = table[f"{stat}_m"]
    return table[table["events"] >= min_events]


def _series_style(source):
    """
    Colour of a source in the aligned plots.

    Parameters
    ----------
    source : str
        Source name.

    Returns
    -------
    str
        Hex colour: dark ink for the sensor, blue for the ROI mean, the
        map colours for virtual sensors.
    """
    if source == SENSOR:
        return INK_STRONG
    if source.startswith("VS"):
        return virtual_sensor_color(int(source[2:]) - 1)
    return BLUE


def plot_aligned(series, align, path, padding_hours=EVENT_PADDING_HOURS,
                 title=None, stat="mean", dry="nan", start_origin="buffer"):
    """
    Draw every event's depth series on a shared, event-relative clock.

    One panel per source (sensor, virtual sensors, ROI mean) with each
    event as a thin line and the median across events in bold with its
    interquartile range, and a last panel overlaying all medians.

    Parameters
    ----------
    series : pandas.DataFrame
        Output of :func:`event_series`.
    align : {'start', 'peak'}
        Event start or sensor peak at hour 0.
    path : str
        Output image.
    padding_hours : float, optional
        Padding around each event, for the x-axis range.
    title : str, optional
        Figure title.
    stat : {'mean', 'median'}, optional
        Statistic across events for the bold line. Default mean.
    dry : {'nan', 'zero'}, optional
        ``'nan'`` leaves dry times out, so lines break when a location
        is dry and the bold line is the typical depth among events wet
        at that time; ``'zero'`` counts dry times as 0 m.
    start_origin : {'buffer', 'flood'}, optional
        Hour 0 of the start-aligned plot: the buffered window's start
        (default; the flood start is then marked at +`padding_hours`)
        or the flood start itself.

    Returns
    -------
    pandas.DataFrame or None
        The composites drawn, or None if nothing could be aligned.
    """
    if dry == "zero":
        series = series.assign(depth_m=series["depth_m"].fillna(0.0))
    aligned = aligned_series(series, align, start_origin)
    if aligned.empty:
        return None
    present = set(aligned["source"])
    sources = ([SENSOR] if SENSOR in present else []) + sorted(
        (s for s in present if s.startswith("VS")),
        key=lambda s: int(s[2:])) + [
        s for s in (ROI_STATS["MeanDepth"],) if s in present]
    table = composites(aligned, stat=stat)

    flood_start_at = None
    if align == "start":
        durations = (aligned["flood_end"] - aligned["flood_start"]
                     ).dt.total_seconds() / 3600
        span = float(np.nanpercentile(durations, 90)) + padding_hours
        if start_origin == "buffer":
            xlim = (0.0, span + padding_hours)
            xlabel = "Hours Since Buffer Start (event folder window)"
            flood_start_at = padding_hours
        else:
            xlim = (-padding_hours, span)
            xlabel = "Hours Since Flood Start"
    else:
        reach = float(np.nanpercentile(aligned["hours"].abs(), 95))
        xlim = (-reach, reach)
        xlabel = "Hours From In-Situ Sensor Peak"
    top = float(np.nanpercentile(aligned["depth_m"].dropna(), 99.5)
                * 1.1) if aligned["depth_m"].notna().any() else 0.1

    n_panels = len(sources) + 1
    cols = 3
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for ax, source in zip(flat, sources):
        color = _series_style(source)
        group = aligned[aligned["source"] == source]
        for _, event in group.groupby("event"):
            event = event.sort_values("hours")
            ax.plot(event["hours"], event["depth_m"], color=color,
                    alpha=0.18, linewidth=0.8, zorder=1)
        comp = table[table["source"] == source]
        ax.fill_between(comp["hours"], comp["q1_m"], comp["q3_m"],
                        color=color, alpha=0.25, linewidth=0, zorder=2)
        ax.plot(comp["hours"], comp["center_m"], color=color, linewidth=2.2,
                zorder=3)
        wet_events = group.dropna(subset=["depth_m"])["event"].nunique()
        ax.set_title(f"{source} ({wet_events} events with water)",
                     fontsize=10)
    overlay = flat[len(sources)]
    for source in sources:
        comp = table[table["source"] == source]
        overlay.plot(comp["hours"], comp["center_m"],
                     color=_series_style(source), linewidth=2.2,
                     linestyle="--" if source == SENSOR else "-",
                     label=source)
    overlay.set_title(f"{stat.title()}s across events", fontsize=10)
    overlay.legend(fontsize=8, loc="upper right")
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    for ax in flat[:n_panels]:
        ax.axvline(0, color=INK, linewidth=1, linestyle=":", zorder=0)
        if flood_start_at is not None:
            ax.axvline(flood_start_at, color=INK, linewidth=1,
                       linestyle="--", zorder=0)
        ax.set_xlim(xlim)
        ax.set_ylim(0, top)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    origin_note = (f"Dashed line: flood start from the event sheet "
                   f"({padding_hours:g} h after the buffer start). "
                   if flood_start_at is not None else "")
    dry_note = (origin_note + "Dry times left out: the bold line is the typical depth "
                "among events with water at that time (fewer events near "
                "onset and recession; counts in _aligned_composites.csv)."
                if dry == "nan" else origin_note + "Dry times count as "
                                                   "0 m.")
    note = (f"Thin lines: individual events. Bold: {stat} across events in "
            f"{COMPOSITE_MINUTES}-min bins (bins with >= "
            f"{MIN_COMPOSITE_EVENTS} events); band: interquartile range. "
            f"Camera lines only where (daylight) images exist.\n{dry_note}")
    fig.text(0.01, -0.01, note, fontsize=8, color=INK, va="top")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return table.assign(align=align, stat=stat, dry=dry,
                        start_origin=start_origin if align == "start"
                        else "")


# ----- Summary ---------------------------------------------------------------

def summary_table(observations, timing):
    """
    Counts and medians per source.

    Parameters
    ----------
    observations, timing : pandas.DataFrame
        Outputs of :func:`collect`.

    Returns
    -------
    pandas.DataFrame
        One row per source.
    """
    rows = []
    for source in source_order(observations):
        group = observations[observations["source"] == source]
        values = group["depth_m"].dropna()
        row = {"source": source, "observations_with_water": len(group),
               "events": group["event"].nunique(),
               "events_without_sensor_readings": group.loc[
                   ~group["has_sensor"], "event"].nunique(),
               "median_depth_m": values.median(),
               "p90_depth_m": values.quantile(0.9),
               "max_depth_m": values.max()}
        for state in STATE_COLORS:
            row[f"share_{state.replace(' ', '_')}"] = (
                (group["sensor_state"] == state).mean() if len(group)
                else np.nan)
        timing_source = "ROI" if source.startswith("ROI") else source
        times = (timing[timing["source"] == timing_source]
                 if len(timing) else timing)
        if source != SENSOR and len(times):
            usable = times.dropna(subset=["onset_lead_min"])
            row.update({
                "timing_events": len(usable),
                "median_onset_lead_min": usable["onset_lead_min"].median(),
                "share_camera_first": (usable["onset_lead_min"] > 0).mean()
                if len(usable) else np.nan,
                "median_end_lag_min": usable["end_lag_min"].median(),
                "share_camera_later": (usable["end_lag_min"] > 0).mean()
                if len(usable) else np.nan,
                "camera_only_events": int(times["camera_only"].sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary, settings, path):
    """
    Write a readable run summary.

    Parameters
    ----------
    summary : pandas.DataFrame
        Output of :func:`summary_table`.
    settings : dict
        Inputs and options of the run.
    path : str
        Output Markdown file.
    """
    lines = ["# In-situ sensor vs camera comparison", "",
             f"Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC by "
             "`sensor_camera_comparison.py` (definitions in its "
             "docstring).", "", "## Run settings", ""]
    lines += [f"- **{key}:** {value}" for key, value in settings.items()]
    lines += ["", "## By source", "",
              "| Source | Obs. with water | Events | Events w/o sensor | "
              "Median depth (m) | Share while sensor dry | "
              "Median onset lead (min) | Camera first | "
              "Median end lag (min) | Camera later | Camera-only events |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, pct=False, integer=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if integer:
            return str(int(value))
        return f"{100 * value:.0f}%" if pct else (
            f"{value:.3f}" if isinstance(value, float) else str(value))

    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['source']} | {row['observations_with_water']} | "
            f"{row['events']} | {row['events_without_sensor_readings']} | "
            f"{fmt(row['median_depth_m'])} | "
            f"{fmt(row.get('share_sensor_dry'), True)} | "
            f"{fmt(row.get('median_onset_lead_min'))} | "
            f"{fmt(row.get('share_camera_first'), True)} | "
            f"{fmt(row.get('median_end_lag_min'))} | "
            f"{fmt(row.get('share_camera_later'), True)} | "
            f"{fmt(row.get('camera_only_events'), integer=True)} |")
    lines += ["", "Timing columns compare first and last water within each "
              "event's image window; ROI rows share the ROI timing. "
              "Censored events are included in the medians; see "
              "`_timing.csv` for the censoring flags.", ""]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


# ----- CLI -------------------------------------------------------------------

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
        description="Compare the in-situ sensor with camera-derived depths "
                    "across flood events.")
    parser.add_argument("--events_dir", required=True,
                        help="Folder of event subfolders.")
    parser.add_argument("--out_dir", required=True,
                        help="Output folder.")
    parser.add_argument("--sensor", default=None,
                        help="Sensor readings CSV (e.g. "
                             "sensor_data_comparison.csv).")
    parser.add_argument("--labels", default=None,
                        help="segmentation_labels.csv; adds a second set of "
                             "outputs from well-segmented images only.")
    parser.add_argument("--roi", default=None,
                        help="ROI name. Default with --site: the stem of "
                             "camera_roi_json, else roi_json.")
    parser.add_argument("--site", default=None,
                        help="Site config supplying the default ROI.")
    parser.add_argument("--match_minutes", type=float, default=5.0,
                        help="Largest gap between a camera time and the "
                             "sensor reading matched to it.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed of the plot jitter.")
    parser.add_argument("--padding_hours", type=float,
                        default=EVENT_PADDING_HOURS,
                        help="Padding stage 05 added around each event; "
                             "used to keep each image and reading in its "
                             "nearest event.")
    parser.add_argument("--aligned_stat", choices=("mean", "median"),
                        default="mean",
                        help="Statistic across events in the aligned "
                             "time series plots.")
    parser.add_argument("--start_origin", choices=("buffer", "flood"),
                        default="buffer",
                        help="Hour 0 of the start-aligned plot: the "
                             "buffered window's start (event folder name) "
                             "or the flood start from the event sheet.")
    parser.add_argument("--aligned_dry", choices=("nan", "zero"),
                        default="nan",
                        help="Aligned plots: leave dry times out (nan) or "
                             "count them as 0 m (zero).")
    args = parser.parse_args(argv)
    if args.roi is None and args.site:
        site = load_site_config(args.site)
        path = site.camera_roi_json or site.roi_json
        if path:
            args.roi = os.path.splitext(os.path.basename(path))[0]
    if args.roi is None:
        parser.error("give --roi, or a --site with an ROI JSON")
    return args


def main(argv=None):
    """
    Run the comparison and write every output.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if no event has camera data, else 0.
    """
    setup_logging()
    apply_style()
    args = parse_args(argv)
    good = load_good_timestamps(args.labels)
    sensor = load_sensor_data(args.sensor)
    image_sets = [("all_images", set(), "All images")]
    if good:
        image_sets.append(("good_only", good, "Well-segmented images only"))
    os.makedirs(args.out_dir, exist_ok=True)
    written = 0
    for tag, stamps, label in image_sets:
        written += run_image_set(args, sensor, tag, stamps, label)
    return 0 if written else 1


def run_image_set(args, sensor, tag, good_timestamps, label):
    """
    Run the comparison for one set of images and write its outputs.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    sensor : pandas.DataFrame or None
        All sensor readings.
    tag : str
        Filename tag (``"all_images"`` or ``"good_only"``).
    good_timestamps : set of str
        Images to keep; empty keeps every image.
    label : str
        Figure title and summary heading.

    Returns
    -------
    int
        1 if outputs were written, 0 if there were no observations.
    """
    observations, timing, series = collect(args.events_dir, args.roi,
                                           sensor,
                                   good_timestamps, args.match_minutes,
                                   args.padding_hours)
    if observations.empty:
        logger.error("No camera observations for ROI '%s' under %s (%s).",
                     args.roi, args.events_dir, label)
        return 0

    prefix = os.path.join(args.out_dir, f"{args.roi}_sensor_camera_{tag}")
    title = f"{label} ({args.roi})"
    plot_depths(observations, f"{prefix}_depths.jpg", args.seed, title)
    if len(timing) and timing["onset_lead_min"].notna().any():
        plot_timing(timing, f"{prefix}_timing.jpg", args.seed, title)
    else:
        logger.warning("No event has both camera and sensor water; timing "
                       "plot skipped (%s).", label)
    drawn = [plot_aligned(series, align, f"{prefix}_aligned_{align}.jpg",
                          args.padding_hours, f"{title}: aligned on "
                          f"{('buffer start' if args.start_origin == 'buffer' else 'flood start') if align == 'start' else 'in-situ sensor peak'}",
                          args.aligned_stat, args.aligned_dry,
                          args.start_origin)
             for align in ("start", "peak")]
    drawn = [table for table in drawn if table is not None]
    if drawn:
        pd.concat(drawn, ignore_index=True).to_csv(
            f"{prefix}_aligned_composites.csv", index=False)
    peaks = event_peaks(observations)
    plot_peaks(peaks, f"{prefix}_peaks.jpg", args.seed, title)

    summary = summary_table(observations, timing)
    observations.to_csv(f"{prefix}_observations.csv", index=False)
    timing.to_csv(f"{prefix}_timing.csv", index=False)
    peaks.to_csv(f"{prefix}_event_peaks.csv", index=False)
    summary.to_csv(f"{prefix}_summary.csv", index=False)
    write_markdown(summary, {
        "Images": label,
        "Events folder": args.events_dir, "ROI": args.roi,
        "Sensor CSV": args.sensor or "none",
        "Segmentation labels": args.labels or "none",
        "Sensor match": f"nearest reading within {args.match_minutes:g} min",
        "Events with camera data": observations["event"].nunique(),
    }, f"{prefix}_summary.md")
    logger.info("Wrote sensor vs camera comparison to %s_*", prefix)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
