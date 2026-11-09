"""Organize flood event data and images into per-event folders.

Each flood event gets a folder named
``<sensor>_<padded start>_<padded end>`` (UTC, ``YYYYMMDDHHMMSS``)
holding that event's sensor records CSV and image subfolders. The
padded event window is computed from the abbreviated flood event CSV's
UTC columns by :func:`event_windows`, so CSV folders and image folders
always agree.
"""

import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import EASTERN_TZ
from .logging_utils import progress_interval
from .naming import (
    TIMESTAMP_FORMAT,
    camera_from_sensor_id,
    extract_camera_name,
    extract_timestamp,
    parse_timestamp,
)

logger = logging.getLogger(__name__)

#: Image extensions recognized when sorting images into events.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

__all__ = [
    "create_flood_csvs_and_subfolders",
    "event_windows",
    "extract_camera_name",
    "extract_timestamp",
    "filter_abbr_flood_csv_by_eastern_time",
    "organize_images_into_flood_events",
    "prepare_job_lists",
    "prune_empty_event_folders",
    "summarize_event_daylight_coverage",
]


def event_windows(abbr_events, padding_hours=3):
    """
    Compute padded UTC windows and folder names for flood events.

    Parameters
    ----------
    abbr_events : pandas.DataFrame
        Abbreviated events with ``sensor_ID``, ``start_time_UTC`` and
        ``end_time_UTC``.
    padding_hours : float, optional
        Hours added before each start and after each end. Default 3.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensor_ID``, ``camera_ID``, ``start`` and ``end``
        (padded, aware UTC) and ``folder_name``. Rows with unparsable
        times are dropped with a warning.
    """
    padding = pd.Timedelta(hours=padding_hours)
    windows = pd.DataFrame({
        "sensor_ID": abbr_events["sensor_ID"].astype(str),
        "start": pd.to_datetime(abbr_events["start_time_UTC"], utc=True,
                                errors="coerce") - padding,
        "end": pd.to_datetime(abbr_events["end_time_UTC"], utc=True,
                              errors="coerce") + padding,
    })
    invalid = windows["start"].isna() | windows["end"].isna()
    if invalid.any():
        logger.warning("Dropping %d events with invalid timestamps.",
                       int(invalid.sum()))
        windows = windows[~invalid]

    windows["camera_ID"] = windows["sensor_ID"].map(camera_from_sensor_id)
    windows["folder_name"] = (
        windows["sensor_ID"] + "_"
        + windows["start"].dt.strftime(TIMESTAMP_FORMAT) + "_"
        + windows["end"].dt.strftime(TIMESTAMP_FORMAT))
    return windows.reset_index(drop=True)


def filter_abbr_flood_csv_by_eastern_time(input_path, output_path,
                                          min_hour=6, max_hour=19):
    """
    Keep events that start within an Eastern local-time hour range.

    Parameters
    ----------
    input_path : str or os.PathLike
        Abbreviated CSV with a ``start_time_UTC`` column.
    output_path : str or os.PathLike
        Filtered CSV to write.
    min_hour : int, optional
        First hour kept (inclusive, 0-23). Default is 6.
    max_hour : int, optional
        First hour excluded (exclusive, 1-24). Default is 19, keeping
        starts from 06:00 through 18:59.

    Returns
    -------
    pandas.DataFrame
        The filtered events, with an added ``start_time_ET`` column.

    Raises
    ------
    FileNotFoundError
        If `input_path` does not exist.
    KeyError
        If the CSV lacks ``start_time_UTC``.
    """
    events = pd.read_csv(input_path)
    if "start_time_UTC" not in events:
        raise KeyError(f"{input_path} has no 'start_time_UTC' column")

    events["start_time_UTC"] = pd.to_datetime(events["start_time_UTC"],
                                              utc=True, errors="coerce")
    events = events.dropna(subset=["start_time_UTC"])
    events["start_time_ET"] = events["start_time_UTC"].dt.tz_convert(
        EASTERN_TZ)
    hour = events["start_time_ET"].dt.hour
    kept = events[(hour >= min_hour) & (hour < max_hour)].copy()

    kept.to_csv(output_path, index=False)
    logger.info("Kept %d of %d events starting %02d:00-%02d:00 ET -> %s",
                len(kept), len(events), min_hour, max_hour, output_path)
    return kept


#: Columns added by :func:`summarize_event_daylight_coverage`, in order.
DAYLIGHT_COVERAGE_COLUMNS = [
    "buffer_hours",
    "has_images",
    "has_daylight_images",
    "has_event_daylight_images",
    "event_images_total",
    "event_images_daylight",
    "event_images_night",
    "buffer_images_total",
    "buffer_images_daylight",
    "buffer_images_night",
    "event_daylight_image_fraction",
    "event_daylight_hours",
    "event_daylight_start_UTC",
    "event_daylight_end_UTC",
    "first_daylight_image_UTC",
    "last_daylight_image_UTC",
]


def _image_index(image_dir):
    """
    List the images in a folder by camera, with their UTC timestamps.

    Parameters
    ----------
    image_dir : str or os.PathLike
        Folder of images named with a camera and UTC timestamp.

    Returns
    -------
    dict of str to pandas.DatetimeIndex
        Sorted, de-duplicated image timestamps for each camera. Files
        without a camera name or timestamp are ignored.
    """
    stamps = defaultdict(list)
    for entry in os.scandir(image_dir):
        if not (entry.is_file()
                and entry.name.lower().endswith(IMAGE_EXTENSIONS)):
            continue
        camera, timestamp = (extract_camera_name(entry.name),
                             parse_timestamp(entry.name))
        if camera and timestamp is not None:
            stamps[camera].append(timestamp)
    return {camera: pd.DatetimeIndex(times).sort_values().unique()
            for camera, times in stamps.items()}


def _daylight_hours(start, end, latitude, longitude, tz):
    """
    Hours of a UTC window that fall between local sunrise and sunset.

    Parameters
    ----------
    start, end : pandas.Timestamp
        Aware UTC window.
    latitude, longitude : float
        Site location in decimal degrees.
    tz : datetime.tzinfo
        Site time zone.

    Returns
    -------
    float
        Daylight hours inside the window, from the sun's position
        alone (independent of whether the camera recorded).
    """
    from .image_handler import sun_times

    total = pd.Timedelta(0)
    day = start.astimezone(tz).date()
    while day <= end.astimezone(tz).date():
        sunrise, sunset = sun_times(day, latitude, longitude, tz)
        overlap = (min(end, pd.Timestamp(sunset))
                   - max(start, pd.Timestamp(sunrise)))
        total += max(overlap, pd.Timedelta(0))
        day += pd.Timedelta(days=1)
    return round(total.total_seconds() / 3600, 3)


def summarize_event_daylight_coverage(abbr_events_path, image_dir,
                                      output_path, latitude, longitude,
                                      padding_hours=3, tz=EASTERN_TZ):
    """
    Add per-event daylight image counts to the abbreviated event CSV.

    For each event, the images in `image_dir` from the event's camera
    are split into those inside the event (``start_time_UTC`` to
    ``end_time_UTC``) and those in the buffers before and after it
    (`padding_hours` each side), and each group into daylight (local
    sunrise to sunset) and night. The result is every original row
    and column plus :data:`DAYLIGHT_COVERAGE_COLUMNS`:

    * ``has_images`` / ``has_daylight_images``: any image / any
      daylight image in the buffered window. Events without daylight
      images get no event folder (see
      :func:`prune_empty_event_folders`).
    * ``has_event_daylight_images``: any daylight image inside the
      event itself, not counting the buffers.
    * ``event_images_*`` / ``buffer_images_*``: total, daylight, and
      night image counts inside the event and in the two buffers.
    * ``event_daylight_image_fraction``: daylight share of the event's
      images (empty if it has none).
    * ``event_daylight_hours``: hours of the event between sunrise and
      sunset, from the sun alone. Comparing it with
      ``duration_(hours)`` separates "the event was at night" from "the
      camera did not record".
    * ``event_daylight_start_UTC`` / ``event_daylight_end_UTC``: first
      and last daylight image inside the event, the window to use when
      comparing camera results with the sensor over the same period.
    * ``first_daylight_image_UTC`` / ``last_daylight_image_UTC``: the
      same over the buffered window.

    Parameters
    ----------
    abbr_events_path : str or os.PathLike
        Abbreviated CSV with ``sensor_ID``, ``start_time_UTC`` and
        ``end_time_UTC``. It is not modified.
    image_dir : str or os.PathLike
        Folder of all pulled images (before daylight filtering).
    output_path : str or os.PathLike
        CSV to write. Rewritten in full on every call.
    latitude, longitude : float
        Camera location in decimal degrees.
    padding_hours : float, optional
        Buffer on each side of an event; match the image pull and
        event setup stages. Default is 3.
    tz : datetime.tzinfo, optional
        Site time zone. Default is US Eastern.

    Returns
    -------
    pandas.DataFrame
        The annotated events, in the input's row order.

    Raises
    ------
    FileNotFoundError
        If the CSV or `image_dir` does not exist.
    """
    from .image_handler import is_daylight

    events = pd.read_csv(abbr_events_path)
    # Drop coverage columns from an earlier run so they are recomputed
    events = events.drop(columns=[c for c in DAYLIGHT_COVERAGE_COLUMNS
                                  if c in events.columns])
    images = _image_index(image_dir)
    is_day = {camera: np.array([is_daylight(t, latitude, longitude, tz)
                                for t in times], dtype=bool)
              for camera, times in images.items()}
    padding = pd.Timedelta(hours=padding_hours)
    empty = pd.DatetimeIndex([], tz="UTC")

    def first(times):
        """Return the earliest time, or NaT if there is none."""
        return times[0] if len(times) else pd.NaT

    def last(times):
        """Return the latest time, or NaT if there is none."""
        return times[-1] if len(times) else pd.NaT

    starts = pd.to_datetime(events["start_time_UTC"], utc=True,
                            errors="coerce")
    ends = pd.to_datetime(events["end_time_UTC"], utc=True, errors="coerce")
    rows = []
    for sensor_id, start, end in zip(events["sensor_ID"].astype(str), starts,
                                     ends):
        if pd.isna(start) or pd.isna(end):
            logger.warning("Event for %s has invalid times; coverage left "
                           "empty.", sensor_id)
            rows.append({"buffer_hours": padding_hours})
            continue
        camera = camera_from_sensor_id(sensor_id)
        times = images.get(camera, empty)
        day = is_day.get(camera, np.zeros(0, dtype=bool))

        in_window = (times >= start - padding) & (times <= end + padding)
        in_event = (times >= start) & (times <= end)
        in_buffer = in_window & ~in_event
        window_day = times[in_window & day]
        event_day = times[in_event & day]

        n_event, n_event_day = int(in_event.sum()), int((in_event & day).sum())
        n_buffer = int(in_buffer.sum())
        n_buffer_day = int((in_buffer & day).sum())
        rows.append({
            "buffer_hours": padding_hours,
            "has_images": bool(in_window.any()),
            "has_daylight_images": len(window_day) > 0,
            "has_event_daylight_images": len(event_day) > 0,
            "event_images_total": n_event,
            "event_images_daylight": n_event_day,
            "event_images_night": n_event - n_event_day,
            "buffer_images_total": n_buffer,
            "buffer_images_daylight": n_buffer_day,
            "buffer_images_night": n_buffer - n_buffer_day,
            "event_daylight_image_fraction": (
                round(n_event_day / n_event, 3) if n_event else np.nan),
            "event_daylight_hours": _daylight_hours(start, end, latitude,
                                                    longitude, tz),
            "event_daylight_start_UTC": first(event_day),
            "event_daylight_end_UTC": last(event_day),
            "first_daylight_image_UTC": first(window_day),
            "last_daylight_image_UTC": last(window_day),
        })

    coverage = pd.DataFrame(rows, columns=DAYLIGHT_COVERAGE_COLUMNS,
                            index=events.index)
    annotated = pd.concat([events, coverage], axis=1)
    annotated.to_csv(output_path, index=False)

    kept = annotated["has_daylight_images"].fillna(False).astype(bool)
    logger.info("Daylight coverage for %d events -> %s: %d with daylight "
                "images (%d inside the event itself), %d night or buffer "
                "only, %d with no images.", len(annotated), output_path,
                int(kept.sum()),
                int(annotated["has_event_daylight_images"].fillna(False)
                    .astype(bool).sum()),
                int((annotated["has_images"].fillna(False).astype(bool)
                     & ~kept).sum()),
                int((~annotated["has_images"].fillna(False).astype(bool))
                    .sum()))
    return annotated


def create_flood_csvs_and_subfolders(abbr_events_path, full_events_path,
                                     output_parent_dir, padding_hours=3):
    """
    Write each event's sensor records into its own event folder.

    Events without records (e.g. floods confirmed only from images
    while the sensor was down) get no CSV; their folders are still
    made when their images are sorted, and later stages run on the
    images alone.

    Parameters
    ----------
    abbr_events_path : str or os.PathLike
        Abbreviated CSV (``sensor_ID``, ``start_time_UTC``,
        ``end_time_UTC``).
    full_events_path : str or os.PathLike or None
        Sensor records CSV with ``sensor_ID`` and ``time_UTC``. None
        treats every event as image-only.
    output_parent_dir : str or os.PathLike
        Parent folder for the event folders.
    padding_hours : float, optional
        Hours added around each event. Default is 3.

    Returns
    -------
    int
        Number of event CSVs written.

    Raises
    ------
    FileNotFoundError
        If a given CSV does not exist.
    """
    windows = event_windows(pd.read_csv(abbr_events_path), padding_hours)
    os.makedirs(output_parent_dir, exist_ok=True)
    if full_events_path is None:
        logger.info("No sensor records CSV; all %d events are image-only.",
                    len(windows))
        return 0
    records = pd.read_csv(full_events_path)
    records["time_UTC"] = pd.to_datetime(records["time_UTC"], utc=True)
    records["sensor_ID"] = records["sensor_ID"].astype(str)

    written = 0
    image_only = []
    for window in windows.itertuples():
        rows = records[(records["sensor_ID"] == window.sensor_ID)
                       & (records["time_UTC"] >= window.start)
                       & (records["time_UTC"] <= window.end)]
        if rows.empty:
            image_only.append(window.folder_name)
            continue
        folder = os.path.join(output_parent_dir, window.folder_name)
        os.makedirs(folder, exist_ok=True)
        rows.to_csv(os.path.join(folder, f"{window.folder_name}.csv"),
                    index=False)
        written += 1

    logger.info("Wrote %d of %d event CSVs under %s", written, len(windows),
                output_parent_dir)
    if image_only:
        logger.info("%d events have no sensor records (image-only): %s",
                    len(image_only), ", ".join(image_only))
    return written


def organize_images_into_flood_events(image_folder, csv_file,
                                      destination_folder, subfolder_name,
                                      padding_hours=3):
    """
    Copy images into the event folders whose window contains them.

    An image within several overlapping windows is copied into each.
    Files already present at the destination are not copied again.

    Parameters
    ----------
    image_folder : str or os.PathLike
        Folder of images named with a camera and UTC timestamp.
    csv_file : str or os.PathLike
        Abbreviated CSV defining the events.
    destination_folder : str or os.PathLike
        Parent folder of the event folders.
    subfolder_name : str
        Subfolder within each event folder, e.g. ``"orig_images"``.
    padding_hours : float, optional
        Hours added around each event. Default is 3.

    Returns
    -------
    int
        Number of files copied.

    Raises
    ------
    FileNotFoundError
        If `csv_file` or `image_folder` does not exist.
    """
    windows = event_windows(pd.read_csv(csv_file), padding_hours)
    by_camera = defaultdict(list)
    for window in windows.itertuples():
        by_camera[window.camera_ID].append(
            (window.start, window.end, window.folder_name))

    images = sorted(name for name in os.listdir(image_folder)
                    if name.lower().endswith(IMAGE_EXTENSIONS))
    logger.info("Sorting %d images into %d events.", len(images),
                len(windows))

    copied = unmatched = 0
    interval = progress_interval(len(images))
    for i, name in enumerate(images):
        if i and i % interval == 0:
            logger.info("  %d%% (%d of %d images)",
                        100 * i // len(images), i, len(images))
        camera, taken = extract_camera_name(name), parse_timestamp(name)
        matches = [folder for start, end, folder in by_camera.get(camera, [])
                   if taken is not None and start <= taken <= end]
        if not matches:
            unmatched += 1
        for folder in matches:
            dest_dir = os.path.join(destination_folder, folder,
                                    subfolder_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, name)
            if not os.path.exists(dest):
                shutil.copy2(os.path.join(image_folder, name), dest)
                copied += 1

    logger.info("Copied %d files; %d images matched no event.", copied,
                unmatched)
    return copied


def prepare_job_lists(image_folder, num_jobs, output_dir=None):
    """
    Split a folder's files into text lists for an HPC job array.

    Parameters
    ----------
    image_folder : str or os.PathLike
        Folder whose (non-hidden) files are listed.
    num_jobs : int
        Number of lists, one per job array task.
    output_dir : str or os.PathLike, optional
        Folder for ``file_list_<n>.txt`` (1-based). Default is
        ``job_file_lists`` beside `image_folder`.

    Returns
    -------
    list of pathlib.Path
        The list files written.

    Raises
    ------
    NotADirectoryError
        If `image_folder` is not a directory.
    ValueError
        If `num_jobs` is not positive.
    """
    image_path = Path(image_folder).resolve()
    if not image_path.is_dir():
        raise NotADirectoryError(f"Image folder not found: {image_path}")
    if num_jobs < 1:
        raise ValueError("num_jobs must be at least 1")

    out_path = (image_path.parent / "job_file_lists" if output_dir is None
                else Path(output_dir).resolve())
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in image_path.iterdir()
                   if p.is_file() and not p.name.startswith("."))
    if not files:
        logger.warning("No files in %s; no lists created.", image_path)
        return []

    list_paths = []
    for i, chunk in enumerate(np.array_split(files, num_jobs), start=1):
        list_path = out_path / f"file_list_{i}.txt"
        list_path.write_text("".join(f"{name}\n" for name in chunk))
        list_paths.append(list_path)
    logger.info("Wrote %d file lists for %d files to %s", len(list_paths),
                len(files), out_path)
    return list_paths


def prune_empty_event_folders(output_parent_dir, image_subfolder_name):
    """
    Delete event folders that never received an image subfolder.

    Parameters
    ----------
    output_parent_dir : str or os.PathLike
        Parent folder of the event folders.
    image_subfolder_name : str
        Subfolder an event must contain to be kept, e.g.
        ``"orig_images"``.

    Returns
    -------
    list of str
        Names of the deleted event folders.

    Raises
    ------
    FileNotFoundError
        If `output_parent_dir` does not exist.
    """
    removed = []
    for name in sorted(os.listdir(output_parent_dir)):
        event_dir = os.path.join(output_parent_dir, name)
        if not os.path.isdir(event_dir):
            continue
        if not os.path.exists(os.path.join(event_dir, image_subfolder_name)):
            try:
                shutil.rmtree(event_dir)
                removed.append(name)
            except OSError as err:
                logger.error("Could not delete %s: %s", event_dir, err)
    logger.info("Pruned %d event folders without '%s'.", len(removed),
                image_subfolder_name)
    return removed
