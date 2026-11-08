"""
Camera image retrieval from the SDFP image archive.

This module defines :class:`ImageHandler`, which finds and copies
camera images for flood events and sensor outages out of the SDFP image
archive. The archive is expected to be laid out as::

    <archive root>/<YYYY> Archive/CAM_<sensor_ID>/<YYYY-MM-DD>/<image>

where the date folder is a UTC date and every image file name contains
a 14-digit UTC timestamp (``YYYYMMDDHHMMSS``).

Classes
-------
ImageHandler
    Copy archive images for flood events, outages, and test sets.

Functions
---------
sun_times
    Sunrise and sunset for a local calendar date.
is_daylight
    Check whether a time falls between local sunrise and sunset.
"""

import concurrent.futures
import functools
import os
import random
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

from .utils import log

# US Eastern local time zone (observes daylight saving time)
EASTERN_TZ = ZoneInfo("America/New_York")

# Format of the UTC timestamp embedded in image file names
IMAGE_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"


@functools.lru_cache(maxsize=4096)
def sun_times(local_date, latitude, longitude, tz=EASTERN_TZ):
    """
    Get sunrise and sunset for a local calendar date.

    Parameters
    ----------
    local_date : datetime.date
        The date in the site's local time zone.
    latitude : float
        Site latitude in decimal degrees.
    longitude : float
        Site longitude in decimal degrees.
    tz : datetime.tzinfo, optional
        The site's time zone. Default is US Eastern.

    Returns
    -------
    sunrise, sunset : datetime.datetime
        Timezone-aware sunrise and sunset times.
    """
    # Describe the site for the sun calculation
    loc = LocationInfo(latitude=latitude, longitude=longitude)
    s = sun(loc.observer, date=local_date, tzinfo=tz)
    return s["sunrise"], s["sunset"]


def is_daylight(timestamp, latitude, longitude, tz=EASTERN_TZ):
    """
    Check whether a time falls between local sunrise and sunset.

    Sunrise and sunset are taken for the *local* calendar date of
    `timestamp`. Using the UTC date instead returns the previous
    evening's sunset whenever sunset is after midnight UTC, which on
    the US East Coast rejects every image from spring to fall.

    Parameters
    ----------
    timestamp : datetime.datetime
        Timezone-aware time to check.
    latitude : float
        Site latitude in decimal degrees.
    longitude : float
        Site longitude in decimal degrees.
    tz : datetime.tzinfo, optional
        The site's time zone. Default is US Eastern.

    Returns
    -------
    bool
        True if sunrise <= `timestamp` <= sunset.
    """
    local_date = timestamp.astimezone(tz).date()
    sunrise, sunset = sun_times(local_date, latitude, longitude, tz)
    return sunrise <= timestamp <= sunset


class ImageHandler:
    """
    Find and copy camera images from the SDFP image archive.

    Attributes
    ----------
    drive : str or os.PathLike or None
        Root folder of the image archive, or None if only local
        folders are filtered.
    copy_destination_dir : str or os.PathLike
        Folder that :meth:`_copy_file` copies into. Set by each public
        method before it copies files.
    images_folder : pathlib.Path
        Source image folder, set by
        :meth:`generate_unlabeled_images_folder`.
    """

    def __init__(self, sdfp_image_drive=None):
        """
        Initialize the handler with the image archive location.

        Parameters
        ----------
        sdfp_image_drive : str or os.PathLike, optional
            Root folder of the image archive. Only needed by the methods
            that read the archive (:meth:`pull_flood_event_images`,
            :meth:`pull_outage_images`, :meth:`create_test_image_set`).
            Default is None.
        """
        # Store the archive root folder
        self.drive = sdfp_image_drive
        log(f"Image Handler initialized. Image Drive: {self.drive}")

    # ===== Public methods used by the QC app =====

    def pull_flood_event_images(
        self,
        destination_folder,
        flood_event_csv_path="abbr_flood_events.csv",
        time_buffer_hours=0,
        max_workers=None,
    ):
        """
        Copy the archive images taken during each flood event.

        For every event in the abbreviated flood event CSV, finds the
        images from the event's camera between its start and end
        times (widened by `time_buffer_hours`) and copies them into
        `destination_folder`. Images already in the destination are
        skipped.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            Folder to copy the images into. Created if needed.
        flood_event_csv_path : str or os.PathLike, optional
            Path of the abbreviated flood event CSV. Default is
            ``"abbr_flood_events.csv"``.
        time_buffer_hours : int or float, optional
            Hours added before each event's start and after its end.
            Default is 0 (no buffer).
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.

        Returns
        -------
        int
            Number of images copied.

        Raises
        ------
        ValueError
            If the handler was created without an image archive.
        """
        # The archive is required to find the images
        self._require_drive()

        log(
            "Starting flood event image pull. "
            f"Destination: {destination_folder}"
        )

        # Report the buffer if one is used
        if time_buffer_hours > 0:
            log(
                f"Applying {time_buffer_hours}-hour buffer to all event "
                "times."
            )

        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        try:
            # Load the events as (camera ID, start, end) rows
            flood_events = self._format_abbreviated_events_for_pull(
                flood_event_csv_path
            )
            log(
                f"Loaded {len(flood_events)} flood events from "
                f"{flood_event_csv_path}"
            )
        # Stop if the CSV is missing or malformed
        except Exception as e:
            log(
                "FATAL: Failed to load/format flood events from "
                f"{flood_event_csv_path}. Reason: {e}",
                level="error",
            )
            return 0

        # Nothing to copy
        if flood_events.empty:
            log("No flood events found to process.")
            return 0

        log("Gathering file list for all events... (This may take a moment)")
        total_events = len(flood_events)
        # Log progress roughly every 10% of events
        event_report_interval = max(1, total_events // 10)
        # Paths of every image to copy
        tasks = []

        for i, row in enumerate(flood_events.itertuples(name="Event")):
            # Log the first, last, and every 10% of events
            if (
                i == 0
                or (i + 1) == total_events
                or (i + 1) % event_report_interval == 0
            ):
                try:
                    log(
                        f"  Processing event {i + 1}/{total_events}: "
                        f"Camera {row.camera_ID} @ "
                        f"{row.start_time_UTC.date()}"
                    )
                # Fall back to a short message if a field is malformed
                except Exception:
                    log(f"  Processing event {i + 1}/{total_events}")

            try:
                # Images are stored by camera ID
                camera_id = row.camera_ID

                # Widen the event window by the buffer
                buffer = timedelta(hours=time_buffer_hours)
                start_time = row.start_time_UTC - buffer
                end_time = row.end_time_UTC + buffer

                # Visit every UTC date folder the window touches (an
                # event can span more than one day)
                current_date = start_time.date()
                end_date = end_time.date()

                while current_date <= end_date:
                    # List the images in this date's folder
                    file_list = self._list_files_in_archive_date_dir(
                        current_date, camera_id
                    )

                    # Keep the images inside the window
                    filtered_files = self._filter_files_by_start_and_end_time(
                        file_list, start_time, end_time
                    )

                    # Queue them for copying
                    tasks.extend(filtered_files)

                    # Move on to the next day
                    current_date += timedelta(days=1)

            # Skip rows missing an expected column
            except AttributeError as e:
                log(
                    "Skipping event row: Missing expected CSV column. "
                    f"Error: {e}",
                    level="error",
                )
            # Skip rows that fail for any other reason
            except Exception as e:
                log(
                    f"Skipping event row due to unexpected error: {e}",
                    level="error",
                )

        log(
            f"Finished event processing. Found {len(tasks)} files "
            "(pre-deduplication)."
        )

        # Remove repeats (overlapping events can find the same image)
        tasks = sorted(list(set(tasks)))

        # Skip images that were already copied in an earlier pull
        log("Checking for existing images to avoid re-pulling...")
        existing_files = set(os.listdir(destination_folder))
        new_tasks = [
            t for t in tasks if os.path.basename(t) not in existing_files
        ]
        skipped_count = len(tasks) - len(new_tasks)

        if skipped_count > 0:
            log(f"Skipped {skipped_count} images that already exist locally.")

        # Copy the remaining images in parallel
        return self._parallel_copy_files(new_tasks, max_workers, "Image pull")

    def pull_outage_images(
        self,
        destination_folder,
        outage_csv_path="sensor_outages.csv",
        max_workers=None,
        only_daylight=True,
    ):
        """
        Copy the archive images taken during each sensor outage.

        Lets reviewers check whether flooding happened while a sensor
        was not recording. Images already in the destination are
        skipped.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            Folder to copy the images into. Created if needed.
        outage_csv_path : str or os.PathLike, optional
            Path of the outage CSV, with ``sensor_ID``,
            ``start_time_UTC``, ``end_time_UTC``, ``latitude``, and
            ``longitude`` columns. Default is ``"sensor_outages.csv"``.
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.
        only_daylight : bool, optional
            If True (the default), only copy images taken between
            sunrise and sunset at the sensor's location.

        Returns
        -------
        int
            Number of images copied.

        Raises
        ------
        ValueError
            If the handler was created without an image archive.
        """
        # The archive is required to find the images
        self._require_drive()

        # Nothing to do if the outage CSV does not exist
        if not os.path.exists(outage_csv_path):
            log(f"Outage CSV not found: {outage_csv_path}")
            return 0

        try:
            # Load the outages and parse their times as UTC
            outages_df = pd.read_csv(outage_csv_path)
            outages_df["start_time_UTC"] = pd.to_datetime(
                outages_df["start_time_UTC"], utc=True
            )
            outages_df["end_time_UTC"] = pd.to_datetime(
                outages_df["end_time_UTC"], utc=True
            )
        # Stop if the CSV is malformed
        except Exception as e:
            log(f"Failed to load outage CSV: {e}", level="error")
            return 0

        # Nothing to copy
        if outages_df.empty:
            log("No outages to process.")
            return 0

        log(
            f"Gathering image tasks for {len(outages_df)} outages "
            f"(Daylight Only: {only_daylight})..."
        )
        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        # Paths of every image to copy
        tasks = []

        for _, row in outages_df.iterrows():
            # Skip rows with no sensor ID
            sensor_id = row.get("sensor_ID")
            if not sensor_id:
                continue

            # Images are stored by camera ID
            camera_id = f"CAM_{sensor_id}"
            start_time = row["start_time_UTC"]
            end_time = row["end_time_UTC"]

            # Daylight only: intersect the outage with each day's
            # sunrise-to-sunset window
            if only_daylight:
                # Read the sensor's location (0.0 if not recorded)
                lat = row.get("latitude", 0.0)
                lon = row.get("longitude", 0.0)

                # Describe the sensor's location for the sun calculation
                loc = LocationInfo(
                    name=sensor_id,
                    region="NC",
                    timezone="America/New_York",
                    latitude=lat,
                    longitude=lon,
                )

                # Step through local calendar days, since sunrise and
                # sunset are defined per local day
                local_start = start_time.astimezone(EASTERN_TZ)
                local_end = end_time.astimezone(EASTERN_TZ)

                current_date = local_start.date()
                end_date = local_end.date()

                while current_date <= end_date:
                    try:
                        # Get this day's sunrise and sunset in UTC
                        s = sun(
                            loc.observer, date=current_date, tzinfo=EASTERN_TZ
                        )
                        day_start = s["sunrise"].astimezone(timezone.utc)
                        day_end = s["sunset"].astimezone(timezone.utc)

                        # Overlap of the outage and the daylight window
                        query_start = max(start_time, day_start)
                        query_end = min(end_time, day_end)

                        # Only search if the overlap is not empty
                        if query_start < query_end:
                            # The daylight window can cross a UTC date
                            # boundary, so search both UTC date folders
                            utc_dates_to_check = set(
                                [query_start.date(), query_end.date()]
                            )

                            for u_date in utc_dates_to_check:
                                # List this date's images
                                file_list = (
                                    self._list_files_in_archive_date_dir(
                                        u_date, camera_id
                                    )
                                )
                                # Keep those inside the daylight overlap
                                filtered_files = (
                                    self._filter_files_by_start_and_end_time(
                                        file_list, query_start, query_end
                                    )
                                )
                                # Queue them for copying
                                tasks.extend(filtered_files)

                    # Log days where sunrise/sunset cannot be computed
                    except Exception as e:
                        log(
                            f"Failed solar calculation for {sensor_id} on "
                            f"{current_date}: {e}",
                            level="error",
                        )

                    # Move on to the next local day
                    current_date += timedelta(days=1)

            # All hours: search every UTC date folder in the outage
            else:
                current_date = start_time.date()
                end_date = end_time.date()

                while current_date <= end_date:
                    # List this date's images
                    file_list = self._list_files_in_archive_date_dir(
                        current_date, camera_id
                    )
                    # Keep those inside the outage window
                    filtered_files = self._filter_files_by_start_and_end_time(
                        file_list, start_time, end_time
                    )
                    # Queue them for copying
                    tasks.extend(filtered_files)
                    # Move on to the next day
                    current_date += timedelta(days=1)

        # Remove repeats
        tasks = sorted(list(set(tasks)))
        log(f"Found {len(tasks)} outage images.")

        # Skip images that were already copied
        log("Checking for existing outage images...")
        existing_files = set(os.listdir(destination_folder))
        new_tasks = [
            t for t in tasks if os.path.basename(t) not in existing_files
        ]
        skipped_count = len(tasks) - len(new_tasks)

        if skipped_count > 0:
            log(
                f"Skipped {skipped_count} outage images that already "
                "exist locally."
            )

        # Copy the remaining images in parallel
        return self._parallel_copy_files(
            new_tasks, max_workers, "Outage image pull"
        )

    # ===== Other public methods (image labeling and test sets) =====

    def copy_images_using_hour_window(
        self,
        image_dir,
        destination_folder,
        start_hour_east=6,
        end_hour_east=19,
        max_workers=None,
    ):
        """
        Copy the images in a folder taken within an Eastern hour window.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of images to filter.
        destination_folder : str or os.PathLike
            Folder to copy the images into. Created if needed.
        start_hour_east : int, optional
            First hour to include, in US Eastern time (0-23). Default
            is 6.
        end_hour_east : int, optional
            Last hour to include, in US Eastern time (0-23). Default is
            19.
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.

        Returns
        -------
        int
            Number of images copied.
        """
        tasks = []

        log(
            "Starting copy of images matching provided hour window "
            f"(Eastern). Destination: {destination_folder} Hours: "
            f"{start_hour_east} to {end_hour_east}"
        )

        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        # List the folder's files and keep those inside the hour window
        file_list = self._list_files_in_dir(image_dir)
        filtered_files = self._filter_files_eastern_time_window(
            file_list, start_hour_east, end_hour_east
        )

        # Queue them for copying
        tasks.extend(filtered_files)

        # Copy them in parallel
        return self._parallel_copy_files(
            tasks, max_workers, "Windowed image pull"
        )

    def copy_images_using_sun_position(
        self,
        image_dir,
        destination_folder,
        latitude,
        longitude,
        tz=EASTERN_TZ,
        max_workers=None,
    ):
        """
        Copy the images in a folder taken between sunrise and sunset.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of images to filter.
        destination_folder : str or os.PathLike
            Folder to copy the images into. Created if needed.
        latitude : float
            Camera latitude in decimal degrees.
        longitude : float
            Camera longitude in decimal degrees.
        tz : datetime.tzinfo, optional
            The camera's local time zone. Default is US Eastern.
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.

        Returns
        -------
        int
            Number of images copied.
        """
        log(
            "Starting copy of daylight images. "
            f"Destination: {destination_folder} Location: "
            f"({latitude}, {longitude})"
        )

        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        # List the folder's files and keep those taken in daylight
        file_list = self._list_files_in_dir(image_dir)
        filtered_files = self._filter_files_by_sun_position(
            file_list, latitude, longitude, tz
        )

        # Copy them in parallel
        return self._parallel_copy_files(
            filtered_files, max_workers, "Daylight image copy"
        )

    def generate_unlabeled_images_folder(
        self, image_dir, labels_dir, destination_folder, max_workers=None
    ):
        """
        Copy the images that do not have a matching label file.

        An image matches a label when both file names contain the same
        camera ID (``CAM_XX_00``) and the same 14-digit timestamp.

        Parameters
        ----------
        image_dir : str or os.PathLike
            Folder of all images to check.
        labels_dir : str or os.PathLike
            Folder of labeled images.
        destination_folder : str or os.PathLike
            Folder to copy the unlabeled images into. Created if needed.
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.

        Returns
        -------
        int
            Number of images copied.
        """
        # Remember the source image folder
        self.images_folder = Path(image_dir)

        log(
            "Starting unlabeled image generation. "
            f"Destination: {destination_folder}"
        )

        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        # List the files in both folders
        log(f"Scanning label directory: {labels_dir}")
        label_files = self._list_files_in_dir(labels_dir)

        log(f"Scanning image directory: {image_dir}")
        image_files = self._list_files_in_dir(image_dir)

        # Nothing to compare
        if not image_files:
            log("No images found in the source directory.")
            return 0

        # Build the set of (camera ID, timestamp) keys that have labels
        log("Building key set from label files...")
        labeled_keys = set()
        for f_path in label_files:
            key = self._get_image_key(os.path.basename(f_path))
            if key:
                labeled_keys.add(key)

        log(f"Found {len(labeled_keys)} unique label keys.")

        # Find the images whose key is not in the labeled set
        log("Finding unlabeled images... (This may take a moment)")
        tasks = []

        total_images = len(image_files)
        # Log progress roughly every 10% of images
        report_interval = max(1, total_images // 10)

        # File names whose key could not be read, reported at the end
        parse_failures = []

        for i, f_path in enumerate(image_files):
            # Read the image's key without logging each failure
            key = self._get_image_key(
                os.path.basename(f_path), log_failure=False
            )

            # Queue the image if it has a key and no label
            if key:
                if key not in labeled_keys:
                    tasks.append(f_path)
            # Otherwise note the file name
            else:
                parse_failures.append(os.path.basename(f_path))

            # Log the first, last, and every 10% of images
            if (
                i == 0
                or (i + 1) == total_images
                or (i + 1) % report_interval == 0
            ):
                log(
                    f"  Scanning progress: {i + 1}/{total_images} images "
                    "checked."
                )

        # Report the files whose names could not be parsed
        if parse_failures:
            log(
                f"Could not parse a key for {len(parse_failures)} image "
                f"files (e.g., '{parse_failures[0]}')."
            )

        # Copy the unlabeled images in parallel
        return self._parallel_copy_files(
            tasks, max_workers, "Unlabeled image copy"
        )

    def create_test_image_set(
        self,
        destination_folder,
        camera_id,
        num_images,
        start_year=2023,
        end_year=2024,
        hour_window=None,
        max_workers=None,
    ):
        """
        Copy a random sample of one camera's archive images.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            Folder to copy the images into. Created if needed.
        camera_id : str
            The camera to sample (e.g. ``"CAM_XX_00"``).
        num_images : int
            Number of images to try to copy.
        start_year : int, optional
            First year to sample from (inclusive). Default is 2023.
        end_year : int, optional
            Last year to sample from (inclusive). Default is 2024.
        hour_window : tuple of (int, int), optional
            (first hour, last hour) in US Eastern time to sample from.
            If None (the default), all hours are used.
        max_workers : int, optional
            Number of threads used to copy files. If None (the
            default), the ThreadPoolExecutor default is used.

        Returns
        -------
        int
            Number of images copied.

        Raises
        ------
        ValueError
            If the handler was created without an image archive.
        """
        # The archive is required to sample images
        self._require_drive()

        log(
            "Starting test image set creation. "
            f"Destination: {destination_folder}"
        )

        # Create the destination folder; stop if that fails
        if not self._setup_destination_dir(destination_folder):
            return 0

        # Pick the random images
        log(f"Selecting {num_images} random images for {camera_id}...")
        tasks = self._get_random_images(
            camera_id,
            start_year,
            end_year,
            num_images=num_images,
            hour_window=hour_window,
        )

        # Copy them in parallel
        return self._parallel_copy_files(tasks, max_workers, "Test set creation")

    # ===== Archive listing and filtering helpers =====

    def _format_abbreviated_events_for_pull(self, flood_event_csv_path):
        """
        Load flood events as (camera ID, start, end) rows.

        Parameters
        ----------
        flood_event_csv_path : str or os.PathLike
            Path of the abbreviated flood event CSV.

        Returns
        -------
        pandas.DataFrame
            Columns ``camera_ID`` (``"CAM_" + sensor_ID``),
            ``start_time_UTC``, and ``end_time_UTC``, with times as
            timezone-aware datetimes.
        """
        # Columns to read from the CSV, and columns to return
        columns_to_read = ["sensor_ID", "start_time_UTC", "end_time_UTC"]
        final_columns = ["camera_ID", "start_time_UTC", "end_time_UTC"]

        flood_events = (
            # Read only the needed columns
            pd.read_csv(flood_event_csv_path, usecols=columns_to_read).assign(
                # Parse the start and end times as UTC; no fixed format,
                # since hand-edited CSVs may drop the seconds or offset
                start_time_UTC=lambda x: pd.to_datetime(
                    x["start_time_UTC"], utc=True
                ),
                end_time_UTC=lambda x: pd.to_datetime(
                    x["end_time_UTC"], utc=True
                ),
                # Archive folders are named by camera, not sensor
                camera_ID=lambda x: "CAM_" + x["sensor_ID"].astype(str),
            )
        )

        # Return only the camera ID and the times
        return flood_events[final_columns]

    def _require_drive(self):
        """
        Check that the handler has an image archive to read.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the handler was created without `sdfp_image_drive`.
        """
        if self.drive is None:
            raise ValueError(
                "This operation reads the image archive; create the "
                "ImageHandler with sdfp_image_drive."
            )

    def _archive_date_dir(self, date, camera_id):
        """
        Build the path of one camera's archive folder for one date.

        Parameters
        ----------
        date : datetime.date or datetime.datetime
            The (UTC) date of the folder.
        camera_id : str
            The camera ID (e.g. ``"CAM_CB_01"``).

        Returns
        -------
        pathlib.Path
            ``<drive>/<YYYY> Archive/<camera_id>/<YYYY-MM-DD>``.

        Raises
        ------
        ValueError
            If the handler was created without an image archive.
        """
        self._require_drive()
        return (
            Path(self.drive)
            / f"{date.year} Archive"
            / camera_id
            / date.strftime("%Y-%m-%d")
        )

    def _list_files_in_archive_date_dir(self, date, camera_id):
        """
        List the files in one camera's archive folder for one date.

        The folder is ``<drive>/<YYYY> Archive/<camera_id>/<YYYY-MM-DD>``.

        Parameters
        ----------
        date : datetime.date or datetime.datetime
            The (UTC) date of the folder to list.
        camera_id : str
            The camera ID (e.g. ``"CAM_CB_01"``).

        Returns
        -------
        list of str
            Paths of the files in the folder, or an empty list if the
            folder does not exist or cannot be read.
        """
        # Build the folder path from the archive layout
        directory = self._archive_date_dir(date, camera_id)

        # A missing folder is normal (e.g. no images that day)
        if not os.path.isdir(directory):
            log(f"Directory not found (this may be expected): {directory}")
            return []

        try:
            # Check again with pathlib before listing
            if not directory.is_dir():
                return []

            # Return the paths of the files (not subfolders)
            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]

        # Log read errors such as missing permissions
        except OSError as e:
            log(
                f"Error reading files from directory '{directory}'. "
                f"Reason: {e}",
                level="error",
            )
            return []

    def _list_files_in_dir(self, directory_path):
        """
        List the files in a folder.

        Parameters
        ----------
        directory_path : str or os.PathLike
            The folder to list.

        Returns
        -------
        list of str
            Paths of the files in the folder, or an empty list if the
            folder does not exist or cannot be read.
        """
        directory = Path(directory_path)
        # Stop if the folder does not exist
        if not directory.is_dir():
            log(f"Directory not found: {directory_path}")
            return []

        try:
            # Return the paths of the files (not subfolders)
            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]
        # Log read errors such as missing permissions
        except OSError as e:
            log(
                f"Error reading files from '{directory_path}'. Reason: {e}",
                level="error",
            )
            return []

    def _filter_files_by_condition(self, file_list, condition_func):
        """
        Keep the files whose file-name timestamp passes a test.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        condition_func : callable
            Takes a timezone-aware UTC datetime and returns True if the
            file should be kept.

        Returns
        -------
        list of str
            The paths that pass the test. Files whose names have no
            readable timestamp are skipped.
        """
        filtered_files = []

        for file in file_list:
            try:
                # Pull the 14-digit timestamp out of the file name
                file_name = os.path.basename(file)
                file_timestamp_str = self._extract_timestamp(file_name)

                # Parse it and mark it as UTC
                file_timestamp_naive = datetime.strptime(
                    file_timestamp_str, IMAGE_TIMESTAMP_FORMAT
                )
                file_timestamp_utc = file_timestamp_naive.replace(
                    tzinfo=timezone.utc
                )

                # Keep the file if it passes the test
                if condition_func(file_timestamp_utc):
                    filtered_files.append(file)
            # Skip files with a missing or malformed timestamp
            except (IndexError, ValueError, TypeError):
                log(
                    "Skipping file: Could not parse timestamp from "
                    f"'{file_name}'."
                )
                continue

        return filtered_files

    def _filter_files_by_start_and_end_time(
        self, file_list, start_time, end_time
    ):
        """
        Keep the files taken between two times (inclusive).

        Parameters
        ----------
        file_list : list of str
            File paths to filter. Each name must contain a 14-digit UTC
            timestamp.
        start_time : datetime.datetime
            Start of the window. Must be timezone-aware.
        end_time : datetime.datetime
            End of the window. Must be timezone-aware.

        Returns
        -------
        list of str
            The paths taken inside the window, or an empty list if
            either time is not timezone-aware.
        """
        # Comparing timezone-aware and naive times raises a TypeError,
        # so check the inputs before filtering
        if start_time.tzinfo is None or end_time.tzinfo is None:
            log(
                "Cannot filter files: start_time and end_time must be "
                "timezone-aware.",
                level="error",
            )
            return []

        def in_window(ft_utc):
            """
            Check whether a timestamp falls inside the window.

            Parameters
            ----------
            ft_utc : datetime.datetime
                The file's UTC timestamp.

            Returns
            -------
            bool
                True if `ft_utc` is between `start_time` and `end_time`.
            """
            return start_time <= ft_utc <= end_time

        # Filter the files with the window test
        return self._filter_files_by_condition(file_list, in_window)

    def _filter_files_eastern_time_window(
        self, file_list, start_hour_east, end_hour_east
    ):
        """
        Keep the files taken within an hour window in Eastern time.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        start_hour_east : int
            First hour to keep, in US Eastern time (inclusive).
        end_hour_east : int
            Last hour to keep, in US Eastern time (inclusive).

        Returns
        -------
        list of str
            The paths taken inside the hour window.
        """

        def in_hour_window(ft_utc):
            """
            Check whether a timestamp's Eastern hour is in the window.

            Parameters
            ----------
            ft_utc : datetime.datetime
                The file's UTC timestamp.

            Returns
            -------
            bool
                True if the Eastern-time hour of `ft_utc` is between
                `start_hour_east` and `end_hour_east`.
            """
            # Convert to Eastern time and compare its hour
            return (
                start_hour_east
                <= ft_utc.astimezone(EASTERN_TZ).hour
                <= end_hour_east
            )

        # Filter the files with the hour test
        return self._filter_files_by_condition(file_list, in_hour_window)

    def _filter_files_by_sun_position(
        self, file_list, latitude, longitude, tz=EASTERN_TZ
    ):
        """
        Keep the files taken between local sunrise and sunset.

        Parameters
        ----------
        file_list : list of str
            File paths to filter.
        latitude : float
            Camera latitude in decimal degrees.
        longitude : float
            Camera longitude in decimal degrees.
        tz : datetime.tzinfo, optional
            The camera's local time zone. Default is US Eastern.

        Returns
        -------
        list of str
            The paths taken in daylight.
        """
        # Filter the files with the daylight test
        return self._filter_files_by_condition(
            file_list, lambda ft_utc: is_daylight(ft_utc, latitude, longitude, tz)
        )

    # ===== Copy helpers =====

    def _setup_destination_dir(self, destination_folder):
        """
        Create the destination folder and set it as the copy target.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The folder to create (with any missing parent folders).

        Returns
        -------
        bool
            True if the folder exists, False if it could not be
            created.
        """
        try:
            # Create the folder if it does not exist
            os.makedirs(destination_folder, exist_ok=True)
            # Tell _copy_file where to copy to
            self.copy_destination_dir = destination_folder
            log(f"Ensured destination folder exists: {destination_folder}")
            return True
        # Log failures such as missing permissions
        except OSError as e:
            log(
                "FATAL: Could not create destination folder "
                f"'{destination_folder}'. Reason: {e}",
                level="error",
            )
            return False

    def _parallel_copy_files(self, tasks, max_workers, task_name="File copy"):
        """
        Copy files into the destination folder using several threads.

        Parameters
        ----------
        tasks : list of str
            Paths of the files to copy into ``copy_destination_dir``.
        max_workers : int or None
            Number of threads to use. If None, the ThreadPoolExecutor
            default is used.
        task_name : str, optional
            Name used in the log messages. Default is ``"File copy"``.

        Returns
        -------
        int
            Number of files copied.
        """
        # Nothing to copy
        total_files = len(tasks)
        if total_files == 0:
            log(f"No files found to copy for task: '{task_name}'.")
            return 0

        # Log progress roughly every 10% of files
        copy_report_interval = max(1, total_files // 10)

        log(
            f"Starting parallel copy of {total_files} files "
            f"(max_workers={max_workers or 'default'})..."
        )
        success_count = 0
        fail_count = 0

        # Run the copies on a pool of worker threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Copy every file; results come back in submission order
            results = executor.map(self._copy_file, tasks)

            for i, result in enumerate(results):
                # _copy_file returns True on success, False on failure
                if result:
                    success_count += 1
                else:
                    fail_count += 1

                # Log the first, last, and every 10% of files
                if (
                    i == 0
                    or (i + 1) == total_files
                    or (i + 1) % copy_report_interval == 0
                ):
                    log(
                        f"  Copy progress: {i + 1}/{total_files} files "
                        "processed."
                    )

        log(
            f"{task_name} complete. Copied: {success_count}, "
            f"Failed: {fail_count}"
        )
        return success_count

    def _copy_file(self, file_path):
        """
        Copy one file into the destination folder.

        Parameters
        ----------
        file_path : str or os.PathLike
            Path of the file to copy into ``copy_destination_dir``.

        Returns
        -------
        bool
            True if the file was copied, False otherwise.
        """
        # The destination must be set by the calling method first
        if not hasattr(self, "copy_destination_dir"):
            log(
                "Cannot copy: self.copy_destination_dir is not set.",
                level="error",
            )
            return False

        try:
            # Copy the file and its permissions
            shutil.copy(file_path, self.copy_destination_dir)
            return True

        # The source file disappeared or the path is wrong
        except FileNotFoundError:
            log(f"File not found, could not copy: {file_path}", "error")
            return False

        # shutil-specific errors (e.g. copying a file onto itself)
        except shutil.Error as e:
            log(f"shutil error copying {file_path}. Reason: {e}", "error")
            return False

        # Lower-level errors (e.g. permission denied, disk full)
        except OSError as e:
            log(f"OS error copying {file_path}. Reason: {e}", "error")
            return False

    def _get_random_images(
        self, camera_id, start_year, end_year, num_images=10, hour_window=None
    ):
        """
        Pick a random sample of one camera's archive images.

        Dates are visited in random order, and a small random number of
        images is taken from each date so the sample is spread out.

        Parameters
        ----------
        camera_id : str
            The camera to sample (e.g. ``"CAM_XX_00"``).
        start_year : int
            First year to sample from (inclusive).
        end_year : int
            Last year to sample from (inclusive).
        num_images : int, optional
            Number of images to pick. Default is 10.
        hour_window : tuple of (int, int), optional
            (first hour, last hour) in US Eastern time to sample from.
            If None (the default), all hours are used.

        Returns
        -------
        list of str
            Paths of the picked images. Can be shorter than
            `num_images` if not enough images were found.
        """
        log(
            f"Starting random image selection for '{camera_id}' "
            f"({start_year}-{end_year})..."
        )

        try:
            # List every date in the year range, in random order
            date_list = self._generate_date_list(start_year, end_year)
            random.shuffle(date_list)
        # Stop if the year range is invalid
        except Exception as e:
            log(
                "FATAL: Failed to generate or shuffle date list. "
                f"Reason: {e}",
                level="error",
            )
            return []

        selected_images = []

        # Keep picking until there are enough images or no dates left
        while len(selected_images) < num_images and date_list:
            date = date_list.pop()

            # Build the camera's folder path for this date
            date_dir = self._archive_date_dir(date, camera_id)

            # List the date's images (logs if the folder is missing)
            images = self._list_files_in_dir(date_dir)

            # Skip dates with no images
            if not images:
                continue

            # Keep only the images inside the hour window, if given
            if hour_window:
                try:
                    start_hour_east, end_hour_east = hour_window
                    filtered_images = self._filter_files_eastern_time_window(
                        images, start_hour_east, end_hour_east
                    )
                # Skip the date if the filter fails
                except Exception as e:
                    log(
                        f"Error applying time filter for {date_dir}: {e}",
                        level="error",
                    )
                    continue
            # No window: use every image
            else:
                filtered_images = images

            # Skip dates with no images left after filtering
            if not filtered_images:
                continue

            # Put the date's images in random order
            random.shuffle(filtered_images)

            # Take 1-15% of the requested total from this date (at
            # least one) to spread the sample across dates
            percentage = random.uniform(0.01, 0.15)
            num_to_select = max(1, int(num_images * percentage))

            # Number of images still needed
            needed = num_images - len(selected_images)

            # Number of images available on this date
            available = len(filtered_images)

            # Take no more than wanted, needed, or available
            num_to_add = min(num_to_select, needed, available)

            selected_images.extend(filtered_images[:num_to_add])

        # Warn if the sample is short
        if len(selected_images) < num_images:
            log(
                "Selection complete. Warning: Only found "
                f"{len(selected_images)} of {num_images} requested images."
            )
        else:
            log(f"Successfully selected {len(selected_images)} random images.")

        return selected_images

    # ===== File name parsing helpers =====

    @staticmethod
    def _get_image_key(filename, log_failure=True):
        """
        Get the (camera ID, timestamp) key from a file name.

        Parameters
        ----------
        filename : str
            The file name (not a full path).
        log_failure : bool, optional
            If True (the default), log a message when no key can be
            read.

        Returns
        -------
        tuple of (str, str) or None
            (camera ID, 14-digit timestamp), or None if either part is
            missing.
        """
        # Read both parts of the key from the name
        sensor_id = ImageHandler._extract_sensor_name(filename)
        timestamp = ImageHandler._extract_timestamp(filename)

        # Return the key only if both parts were found
        if sensor_id and timestamp:
            return (sensor_id, timestamp)

        # Otherwise optionally log the failure
        if log_failure:
            log(f"Skipping file: Could not parse key from '{filename}'.")
        return None

    @staticmethod
    def _extract_timestamp(filename):
        """
        Get the 14-digit UTC timestamp (YYYYMMDDHHMMSS) from a name.

        Parameters
        ----------
        filename : str
            The file name.

        Returns
        -------
        str or None
            The first run of 14 digits in the name, or None if there is
            none.
        """
        # Match the first run of exactly 14 digits
        pattern = r"\d{14}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _extract_sensor_name(filename):
        """
        Get the camera ID (CAM_XX_00) from a file name.

        Parameters
        ----------
        filename : str
            The file name.

        Returns
        -------
        str or None
            The camera ID, or None if the name does not contain one.
        """
        # "CAM_", two capital letters, "_", and two digits
        pattern = r"CAM_[A-Z]{2}_[0-9]{2}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _generate_date_list(start_year, end_year):
        """
        List every date from January 1 to December 31 of a year range.

        Parameters
        ----------
        start_year : int
            First year (inclusive).
        end_year : int
            Last year (inclusive).

        Returns
        -------
        list of datetime.date
            Every date from ``start_year``-01-01 to ``end_year``-12-31.

        Raises
        ------
        ValueError
            If `end_year` is before `start_year`.
        """
        # Reject a backwards range
        if end_year < start_year:
            raise ValueError("end_year cannot be before start_year")

        log(f"Generating date list from {start_year} to {end_year}...")
        date_list = []
        # First and last dates of the range
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()

        # Add one date at a time until the end date
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        log(f"Generated {len(date_list)} dates.")
        return date_list
