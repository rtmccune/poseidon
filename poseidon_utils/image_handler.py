"""Find, filter, and copy images from the SDFP camera image archive.

The archive is organized as
``<drive>/<YYYY> Archive/<camera>/<YYYY-MM-DD>/<image>``, where dates
and the timestamps in image filenames are UTC. :class:`ImageHandler`
pulls the images that fall within flood events, filters folders of
images by clock hours or by daylight, and builds labeled/unlabeled or
random image sets for segmentation work.
"""

import concurrent.futures
import functools
import logging
import os
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .constants import EASTERN_TZ
from .logging_utils import progress_interval, should_report
from .naming import (
    camera_from_sensor_id,
    extract_camera_name,
    extract_timestamp,
    parse_timestamp,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4096)
def sun_times(local_date, latitude, longitude, tz=EASTERN_TZ):
    """
    Compute sunrise and sunset for a local calendar date.

    Parameters
    ----------
    local_date : datetime.date
        Date in the site's local time zone.
    latitude, longitude : float
        Observer location in decimal degrees.
    tz : datetime.tzinfo, optional
        Site time zone. Default is US Eastern.

    Returns
    -------
    sunrise, sunset : datetime.datetime
        Timezone-aware event times.

    Raises
    ------
    ImportError
        If the ``astral`` package is not installed.
    """
    from astral import Observer
    from astral.sun import sun

    events = sun(Observer(latitude, longitude), date=local_date, tzinfo=tz)
    return events["sunrise"], events["sunset"]


def is_daylight(timestamp, latitude, longitude, tz=EASTERN_TZ):
    """
    Test whether a time falls between local sunrise and sunset.

    Sunrise and sunset are computed for the *local* calendar date of
    `timestamp`. Using the UTC date instead returns the previous
    evening's sunset whenever sunset falls after midnight UTC, which
    for US East Coast sites rejects every image from spring to fall.

    Parameters
    ----------
    timestamp : datetime.datetime
        Timezone-aware time to test.
    latitude, longitude : float
        Observer location in decimal degrees.
    tz : datetime.tzinfo, optional
        Site time zone. Default is US Eastern.

    Returns
    -------
    bool
        True if ``sunrise <= timestamp <= sunset``.
    """
    local_date = timestamp.astimezone(tz).date()
    sunrise, sunset = sun_times(local_date, latitude, longitude, tz)
    return sunrise <= timestamp <= sunset


def _copy_to(destination, file_path):
    """
    Copy one file into a folder, reporting success.

    Parameters
    ----------
    destination : str or os.PathLike
        Existing destination folder.
    file_path : str or os.PathLike
        File to copy.

    Returns
    -------
    bool
        True if the copy succeeded.
    """
    try:
        shutil.copy(file_path, destination)
        return True
    except OSError as err:  # missing file, permissions, disk full, ...
        logger.error("Could not copy %s: %s", file_path, err)
        return False


class ImageHandler:
    """
    Pull and filter images from the SDFP camera archive.

    Parameters
    ----------
    sdfp_image_drive : str or os.PathLike, optional
        Root of the image archive. Only needed by methods that read
        the archive (:meth:`pull_flood_event_images`,
        :meth:`create_test_image_set`).

    Attributes
    ----------
    drive : str or None
        Root of the image archive.
    """

    def __init__(self, sdfp_image_drive=None):
        """
        Store the archive root.

        Parameters
        ----------
        sdfp_image_drive : str or os.PathLike, optional
            Root of the image archive. Default is None.
        """
        self.drive = (None if sdfp_image_drive is None
                      else os.fspath(sdfp_image_drive))
        logger.info("ImageHandler initialized (archive: %s)", self.drive)

    # ----- Public workflows --------------------------------------------------

    def pull_flood_event_images(self, destination_folder,
                                flood_event_csv_path="abbr_flood_events.csv",
                                time_buffer_hours=0, max_workers=None):
        """
        Copy every archived image taken during a flood event.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            Folder receiving all images; created if needed.
        flood_event_csv_path : str or os.PathLike, optional
            Abbreviated flood event CSV (``sensor_ID``,
            ``start_time_UTC``, ``end_time_UTC``). Default
            ``"abbr_flood_events.csv"``.
        time_buffer_hours : float, optional
            Hours added before each start and after each end. Default 0.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.

        Returns
        -------
        int
            Number of images copied.
        """
        events = self._format_abbreviated_events_for_pull(
            flood_event_csv_path)
        buffer = timedelta(hours=time_buffer_hours)
        logger.info("Pulling images for %d events (buffer %s h) into %s",
                    len(events), time_buffer_hours, destination_folder)

        files = set()
        total = len(events)
        interval = progress_interval(total)
        for i, event in enumerate(events.itertuples()):
            if should_report(i, total, interval):
                logger.info("  Event %d/%d: %s on %s", i + 1, total,
                            event.camera_ID, event.start_time_UTC.date())
            start = event.start_time_UTC - buffer
            end = event.end_time_UTC + buffer
            day = start.date()
            while day <= end.date():  # events may span several dates
                day_files = self._list_files_in_archive_date_dir(
                    day, event.camera_ID)
                files.update(self._filter_files_by_start_and_end_time(
                    day_files, start, end))
                day += timedelta(days=1)

        return self._parallel_copy_files(sorted(files), destination_folder,
                                         max_workers, "Image pull")

    def copy_images_using_hour_window(self, image_dir, destination_folder,
                                      start_hour_east=6, end_hour_east=19,
                                      max_workers=None):
        """
        Copy images taken within an Eastern local-time hour window.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of images to filter.
        destination_folder : str or os.PathLike
            Folder receiving matching images; created if needed.
        start_hour_east, end_hour_east : int, optional
            Inclusive hour range in US Eastern local time. Default 6-19.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.

        Returns
        -------
        int
            Number of images copied.
        """
        files = self._filter_files_eastern_time_window(
            self._list_files_in_dir(image_dir), start_hour_east,
            end_hour_east)
        return self._parallel_copy_files(files, destination_folder,
                                         max_workers, "Hour-window copy")

    def copy_images_using_sun_position(self, image_dir, destination_folder,
                                       latitude, longitude, tz=EASTERN_TZ,
                                       max_workers=None):
        """
        Copy images taken between local sunrise and sunset.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of images to filter.
        destination_folder : str or os.PathLike
            Folder receiving daylight images; created if needed.
        latitude, longitude : float
            Camera location in decimal degrees.
        tz : datetime.tzinfo, optional
            Site time zone. Default is US Eastern.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.

        Returns
        -------
        int
            Number of images copied.
        """
        files = self._filter_files_by_sun_position(
            self._list_files_in_dir(image_dir), latitude, longitude, tz)
        return self._parallel_copy_files(files, destination_folder,
                                         max_workers, "Daylight copy")

    def generate_unlabeled_images_folder(self, image_dir, labels_dir,
                                         destination_folder,
                                         max_workers=None):
        """
        Copy images that have no matching label file.

        Images and labels match when they share a camera name and
        timestamp.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of images.
        labels_dir : str or os.PathLike
            Folder of label images.
        destination_folder : str or os.PathLike
            Folder receiving unlabeled images; created if needed.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.

        Returns
        -------
        int
            Number of images copied.
        """
        labeled = {self._get_image_key(os.path.basename(path))
                   for path in self._list_files_in_dir(labels_dir)}
        labeled.discard(None)

        unlabeled, unparsed = [], 0
        for path in self._list_files_in_dir(image_dir):
            key = self._get_image_key(os.path.basename(path))
            if key is None:
                unparsed += 1
            elif key not in labeled:
                unlabeled.append(path)
        if unparsed:
            logger.info("Ignored %d images without a camera name and "
                        "timestamp.", unparsed)
        logger.info("%d labeled keys; %d unlabeled images.", len(labeled),
                    len(unlabeled))
        return self._parallel_copy_files(unlabeled, destination_folder,
                                         max_workers, "Unlabeled copy")

    def create_test_image_set(self, destination_folder, camera_id,
                              num_images, start_year=2023, end_year=2024,
                              hour_window=None, max_workers=None):
        """
        Copy a random sample of archived images for one camera.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            Folder receiving the sample; created if needed.
        camera_id : str
            Camera name, e.g. ``"CAM_CB_03"``.
        num_images : int
            Number of images wanted.
        start_year, end_year : int, optional
            Inclusive year range to sample. Default 2023-2024.
        hour_window : tuple of int, optional
            Inclusive ``(start, end)`` Eastern hours. Default all hours.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.

        Returns
        -------
        int
            Number of images copied.
        """
        files = self._get_random_images(camera_id, start_year, end_year,
                                        num_images, hour_window)
        return self._parallel_copy_files(files, destination_folder,
                                         max_workers, "Test set")

    # ----- Listing and filtering ---------------------------------------------

    @staticmethod
    def _format_abbreviated_events_for_pull(flood_event_csv_path):
        """
        Load event windows with camera names from the abbreviated CSV.

        Parameters
        ----------
        flood_event_csv_path : str or os.PathLike
            Abbreviated flood event CSV.

        Returns
        -------
        pandas.DataFrame
            Columns ``camera_ID``, ``start_time_UTC``, ``end_time_UTC``
            (aware UTC).
        """
        events = pd.read_csv(flood_event_csv_path,
                             usecols=["sensor_ID", "start_time_UTC",
                                      "end_time_UTC"])
        return pd.DataFrame({
            "camera_ID": events["sensor_ID"].astype(str).map(
                camera_from_sensor_id),
            "start_time_UTC": pd.to_datetime(events["start_time_UTC"],
                                             utc=True),
            "end_time_UTC": pd.to_datetime(events["end_time_UTC"], utc=True),
        })

    def _archive_dir(self, day, camera_id):
        """
        Return the archive folder holding one camera's images for a day.

        Parameters
        ----------
        day : datetime.date
            UTC date.
        camera_id : str
            Camera name.

        Returns
        -------
        pathlib.Path
            ``<drive>/<YYYY> Archive/<camera>/<YYYY-MM-DD>``.

        Raises
        ------
        ValueError
            If no archive drive was configured.
        """
        if self.drive is None:
            raise ValueError("This operation needs sdfp_image_drive")
        return (Path(self.drive) / f"{day.year} Archive" / camera_id
                / day.strftime("%Y-%m-%d"))

    def _list_files_in_archive_date_dir(self, day, camera_id):
        """
        List one camera's archived files for a day.

        Parameters
        ----------
        day : datetime.date
            UTC date.
        camera_id : str
            Camera name.

        Returns
        -------
        list of str
            File paths; empty if the folder does not exist.
        """
        return self._list_files_in_dir(self._archive_dir(day, camera_id),
                                       missing_ok=True)

    @staticmethod
    def _list_files_in_dir(directory_path, missing_ok=False):
        """
        List the regular files in a folder.

        Parameters
        ----------
        directory_path : str or os.PathLike
            Folder to scan.
        missing_ok : bool, optional
            Log a missing folder at debug rather than warning level.
            Default is False.

        Returns
        -------
        list of str
            Sorted file paths; empty if the folder is missing or
            unreadable.
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            log = logger.debug if missing_ok else logger.warning
            log("Directory not found: %s", directory)
            return []
        try:
            return sorted(os.fspath(p) for p in directory.iterdir()
                          if p.is_file())
        except OSError as err:
            logger.error("Could not read %s: %s", directory, err)
            return []

    @staticmethod
    def _filter_files_by_condition(file_list, condition_func):
        """
        Keep files whose filename timestamp satisfies a condition.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        condition_func : callable
            Called with each file's aware UTC timestamp; returns True
            to keep the file.

        Returns
        -------
        list of str
            Files that passed. Files without a timestamp are dropped.
        """
        kept = []
        for path in file_list:
            timestamp = parse_timestamp(os.path.basename(path))
            if timestamp is None:
                logger.debug("No timestamp in %s; skipped.", path)
            elif condition_func(timestamp):
                kept.append(path)
        return kept

    def _filter_files_by_start_and_end_time(self, file_list, start_time,
                                            end_time):
        """
        Keep files timestamped within an inclusive UTC window.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        start_time, end_time : datetime.datetime
            Timezone-aware window bounds.

        Returns
        -------
        list of str
            Files inside the window.

        Raises
        ------
        ValueError
            If either bound is timezone-naive.
        """
        if start_time.tzinfo is None or end_time.tzinfo is None:
            raise ValueError("start_time and end_time must be "
                             "timezone-aware")
        return self._filter_files_by_condition(
            file_list, lambda ts: start_time <= ts <= end_time)

    def _filter_files_eastern_time_window(self, file_list, start_hour_east,
                                          end_hour_east):
        """
        Keep files taken within an inclusive Eastern hour range.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        start_hour_east, end_hour_east : int
            Inclusive hour range in US Eastern local time.

        Returns
        -------
        list of str
            Files inside the hour range.
        """
        return self._filter_files_by_condition(
            file_list, lambda ts: (start_hour_east
                                   <= ts.astimezone(EASTERN_TZ).hour
                                   <= end_hour_east))

    def _filter_files_by_sun_position(self, file_list, latitude, longitude,
                                      tz=EASTERN_TZ):
        """
        Keep files taken between local sunrise and sunset.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        latitude, longitude : float
            Camera location in decimal degrees.
        tz : datetime.tzinfo, optional
            Site time zone. Default is US Eastern.

        Returns
        -------
        list of str
            Daylight files.

        Raises
        ------
        ImportError
            If the ``astral`` package is not installed.
        """
        return self._filter_files_by_condition(
            file_list, lambda ts: is_daylight(ts, latitude, longitude, tz))

    # ----- Copying -----------------------------------------------------------

    @staticmethod
    def _parallel_copy_files(tasks, destination_folder, max_workers=None,
                             task_name="File copy"):
        """
        Copy files into a folder using a thread pool.

        Parameters
        ----------
        tasks : list of str
            Files to copy.
        destination_folder : str or os.PathLike
            Destination folder; created if needed.
        max_workers : int, optional
            Copy threads. Default lets the executor choose.
        task_name : str, optional
            Label for log messages. Default ``"File copy"``.

        Returns
        -------
        int
            Number of files copied successfully.
        """
        if not tasks:
            logger.info("%s: no files to copy.", task_name)
            return 0
        os.makedirs(destination_folder, exist_ok=True)

        total = len(tasks)
        interval = progress_interval(total)
        logger.info("%s: copying %d files to %s", task_name, total,
                    destination_folder)
        copy = functools.partial(_copy_to, destination_folder)
        copied = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            for i, ok in enumerate(executor.map(copy, tasks)):
                copied += ok
                if should_report(i, total, interval):
                    logger.info("  %d/%d files processed", i + 1, total)
        logger.info("%s complete: %d copied, %d failed.", task_name, copied,
                    total - copied)
        return copied

    # ----- Sampling ----------------------------------------------------------

    def _get_random_images(self, camera_id, start_year, end_year,
                           num_images=10, hour_window=None):
        """
        Pick random archived images spread across random dates.

        Dates are visited in random order and each contributes a random
        1-15 % of `num_images`, so the sample spans many days.

        Parameters
        ----------
        camera_id : str
            Camera name.
        start_year, end_year : int
            Inclusive year range.
        num_images : int, optional
            Images wanted. Default is 10.
        hour_window : tuple of int, optional
            Inclusive ``(start, end)`` Eastern hours. Default all hours.

        Returns
        -------
        list of str
            Selected files; fewer than `num_images` if the archive runs
            out.
        """
        dates = self._generate_date_list(start_year, end_year)
        random.shuffle(dates)

        selected = []
        while len(selected) < num_images and dates:
            images = self._list_files_in_archive_date_dir(dates.pop(),
                                                          camera_id)
            if hour_window:
                images = self._filter_files_eastern_time_window(
                    images, *hour_window)
            if not images:
                continue
            random.shuffle(images)
            per_day = max(1, int(num_images * random.uniform(0.01, 0.15)))
            selected.extend(images[:min(per_day,
                                        num_images - len(selected))])

        if len(selected) < num_images:
            logger.warning("Found only %d of %d requested images.",
                           len(selected), num_images)
        return selected

    @staticmethod
    def _get_image_key(filename):
        """
        Return the ``(camera, timestamp)`` key identifying an image.

        Parameters
        ----------
        filename : str
            Image filename.

        Returns
        -------
        tuple of str or None
            ``(camera_name, timestamp)``, or None if either is absent.
        """
        camera, stamp = extract_camera_name(filename), extract_timestamp(
            filename)
        return (camera, stamp) if camera and stamp else None

    @staticmethod
    def _generate_date_list(start_year, end_year):
        """
        List every date in an inclusive range of years.

        Parameters
        ----------
        start_year, end_year : int
            Inclusive year range.

        Returns
        -------
        list of datetime.date
            Dates from January 1 of `start_year` to December 31 of
            `end_year`.

        Raises
        ------
        ValueError
            If `end_year` is before `start_year`.
        """
        if end_year < start_year:
            raise ValueError("end_year cannot be before start_year")
        return list(pd.date_range(date(start_year, 1, 1),
                                  date(end_year, 12, 31)).date)
