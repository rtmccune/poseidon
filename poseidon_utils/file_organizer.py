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


def create_flood_csvs_and_subfolders(abbr_events_path, full_events_path,
                                     output_parent_dir, padding_hours=3):
    """
    Write each event's sensor records into its own event folder.

    Parameters
    ----------
    abbr_events_path : str or os.PathLike
        Abbreviated CSV (``sensor_ID``, ``start_time_UTC``,
        ``end_time_UTC``).
    full_events_path : str or os.PathLike
        Sensor records CSV with ``sensor_ID`` and ``time_UTC``.
    output_parent_dir : str or os.PathLike
        Parent folder for the event folders.
    padding_hours : float, optional
        Hours added around each event. Default is 3.

    Returns
    -------
    int
        Number of event CSVs written. Events without records are
        skipped.

    Raises
    ------
    FileNotFoundError
        If either CSV does not exist.
    """
    windows = event_windows(pd.read_csv(abbr_events_path), padding_hours)
    records = pd.read_csv(full_events_path)
    records["time_UTC"] = pd.to_datetime(records["time_UTC"], utc=True)
    records["sensor_ID"] = records["sensor_ID"].astype(str)
    os.makedirs(output_parent_dir, exist_ok=True)

    written = 0
    for window in windows.itertuples():
        rows = records[(records["sensor_ID"] == window.sensor_ID)
                       & (records["time_UTC"] >= window.start)
                       & (records["time_UTC"] <= window.end)]
        if rows.empty:
            continue
        folder = os.path.join(output_parent_dir, window.folder_name)
        os.makedirs(folder, exist_ok=True)
        rows.to_csv(os.path.join(folder, f"{window.folder_name}.csv"),
                    index=False)
        written += 1

    logger.info("Wrote %d of %d event CSVs under %s", written, len(windows),
                output_parent_dir)
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
