import concurrent.futures
import os
import random
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def _log(message, level="info"):
    """
    Prints a timestamped message to stdout or stderr.

    Parameters
    ----------
    message : str
        The message to be logged.
    level : str, optional
        The log level. "info" (default) prints to stdout,
        "error" prints to stderr.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level.upper()}: {message}"

    if level.lower() == "error":
        # Print to standard error
        print(formatted_message, file=sys.stderr)
    else:
        # Print to standard output
        print(formatted_message, file=sys.stdout)


class ImageHandler:
    """
    Handles the retrieval and copying of images from an archive
    based on flood event data.

    Parameters
    ----------
    sdfp_image_drive : str or os.PathLike
        The root file path to the image archive drive.

    Attributes
    ----------
    drive : str
        The root file path to the image archive drive.
    copy_destination_dir : str
        The destination folder path set by a public method call.
        This is used by the internal `_copy_file` method.

    """

    def __init__(self, sdfp_image_drive):
        """
        Initializes the ImageHandler with the path to the image drive.

        Parameters
        ----------
        sdfp_image_drive : str or os.PathLike
            The root file path to the image archive drive.
        """
        self.drive = sdfp_image_drive
        _log(
            f"Image Handler initialized. Image Drive: {self.drive}",
            level="info",
        )

    def pull_flood_event_images(
        self,
        destination_folder,
        flood_event_csv_path="abbr_flood_events.csv",
        time_buffer_hours=0,
        max_workers=None,
    ):
        """
        Pulls all images associated with flood events into a single
        folder.

        This is the main public method. It performs the following steps:
        - Creates the destination folder.
        - Loads and formats flood events from the provided CSV.
        - Iterates through each event, spanning multiple days if
          necessary.
        - Applies an optional time buffer to the start and end of
          each event.
        - Lists all files in the relevant archive directories.
        - Filters files to match the event's (buffered) time range.
        - Compiles a unique list of all files to be copied.
        - Uses a thread pool to copy all files to the destination
          folder.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        flood_event_csv_path : str or os.PathLike, optional
            The file path to the CSV containing flood event data.
            Defaults to 'abbr_flood_events.csv'.
        time_buffer_hours : int or float, optional
            The number of hours to add as a buffer *before* the
            start_time and *after* the end_time of each event.
            For example, a value of 1 will pull images from
            1 hour before the event started until 1 hour after
            it ended. Defaults to 0 (no buffer).
        max_workers : int, optional
            The maximum number of worker threads to use for copying
            files.
            If None, the default for ThreadPoolExecutor is used.

        """
        _log(
            f"Starting flood event image pull. Destination: {destination_folder}",
            level="info",
        )

        if time_buffer_hours > 0:
            _log(
                f"Applying {time_buffer_hours}-hour buffer to all event times.",
                level="info",
            )

        # Create destination folder if it does not already exist
        if not self._setup_destination_dir(destination_folder):
            return

        try:
            # Load flood events from abbreviated csv
            flood_events = self._format_abbreviated_events_for_pull(
                flood_event_csv_path
            )
            _log(
                f"Loaded {len(flood_events)} flood events from {flood_event_csv_path}",
                level="info",
            )
        except Exception as e:
            _log(
                f"FATAL: Failed to load/format flood events from {flood_event_csv_path}. Reason: {e}",
                level="error",
            )
            return

        if flood_events.empty:
            _log("No flood events found to process.", level="info")
            return

        _log(
            "Gathering file list for all events... (This may take a moment)",
            level="info",
        )
        total_events = len(flood_events)
        event_report_interval = max(1, total_events // 10)
        tasks = []

        for i, row in enumerate(flood_events.itertuples(name="Event")):
            if (
                i == 0
                or (i + 1) == total_events
                or (i + 1) % event_report_interval == 0
            ):
                try:
                    # Try to log with details
                    _log(
                        f"  Processing event {i + 1}/{total_events}: Camera {row.camera_ID} @ {row.start_time_UTC.date()}",
                        level="info",
                    )
                except Exception:
                    # Failsafe log if row attributes are weird
                    _log(
                        f"  Processing event {i + 1}/{total_events}",
                        level="info",
                    )

            try:
                # Set the camera ID
                camera_id = row.camera_ID

                # Apply the time buffer, if any
                buffer = timedelta(hours=time_buffer_hours)
                start_time = row.start_time_UTC - buffer
                end_time = row.end_time_UTC + buffer

                # set the current time to begin iterating through folders
                current_date = start_time.date()
                # set the end date (possible that the flood event
                # occurred over more than one date)
                end_date = end_time.date()

                while current_date <= end_date:
                    # list the files in the current date iteration's
                    # corresponding folder
                    file_list = self._list_files_in_archive_date_dir(
                        current_date, camera_id
                    )

                    # filter the file list to between the start and end
                    # time (using end time rather than end date accounts
                    # for multiday events)
                    filtered_files = self._filter_files_by_start_and_end_time(
                        file_list, start_time, end_time
                    )

                    # add the file paths to the task list to be copied
                    tasks.extend(filtered_files)

                    # continue to the next day
                    current_date += timedelta(days=1)

            except AttributeError as e:
                _log(
                    f"Skipping event row: Missing expected CSV column. Error: {e}",
                    level="error",
                )
            except Exception as e:
                _log(
                    f"Skipping event row due to unexpected error: {e}",
                    level="error",
                )

        _log(
            f"Finished event processing. Found {len(tasks)} files (pre-deduplication).",
            level="info",
        )

        # De-duplicate the list in case events overlapped and found the
        # same file
        tasks = sorted(list(set(tasks)))

        # The helper will log the unique file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Image pull")

    def copy_images_using_hour_window(
        self,
        image_dir,
        destination_folder,
        start_hour_east=6,
        end_hour_east=19,
        max_workers=None,
    ):
        """
        Copies images from a directory that fall within an Eastern
        Time hour window.

        Parameters
        ----------
        image_dir : str or os.PathLike
            The source directory containing images to filter.
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        start_hour_east : int, optional
            The start hour in Eastern Time (inclusive, 0-23).
            Defaults to 6.
        end_hour_east : int, optional
            The end hour in Eastern Time (inclusive, 0-23).
            Defaults to 19.
        max_workers : int, optional
            The maximum number of worker threads to use for copying
            files.
            If None, the default for ThreadPoolExecutor is used.

        """
        tasks = []

        _log(
            f"Starting copy of images matching provided hour window (Eastern). Destination: {destination_folder} Hours: {start_hour_east} to {end_hour_east}",
            level="info",
        )

        # Create destination folder if it does not already exist
        if not self._setup_destination_dir(destination_folder):
            return

        file_list = self._list_files_in_dir(image_dir)
        filtered_files = self._filter_files_eastern_time_window(
            file_list, start_hour_east, end_hour_east
        )

        # add the file paths to the task list to be copied
        tasks.extend(filtered_files)

        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Windowed image pull")

    def generate_unlabeled_images_folder(
        self, image_dir, labels_dir, destination_folder, max_workers=None
    ):
        """
        Finds images in 'image_dir' that do NOT have a corresponding
        label in 'labels_dir' and copies them to 'destination_folder'.

        A match is determined by comparing the sensor ID
        (e.g., CAM_XX_00) and the timestamp (YYYYMMDDHHMMSS) extracted
        from the filenames.

        Parameters
        ----------
        image_dir : str or os.PathLike
            The directory containing all images to check.
        labels_dir : str or os.PathLike
            The directory containing labeled images.
        destination_folder : str or os.PathLike
            The path to the folder where all *unlabeled* images
            will be copied. It will be created if it does not exist.
        max_workers : int, optional
            The maximum number of worker threads to use for copying.
            If None, the default for ThreadPoolExecutor is used.
        """
        self.images_folder = Path(image_dir)

        _log(
            f"Starting unlabeled image generation. Destination: {destination_folder}",
            level="info",
        )

        # Setup destination folder
        if not self._setup_destination_dir(destination_folder):
            return

        # Get all files from both directories
        _log(f"Scanning label directory: {labels_dir}", level="info")
        label_files = self._list_files_in_dir(labels_dir)

        _log(f"Scanning image directory: {image_dir}", level="info")
        image_files = self._list_files_in_dir(image_dir)

        if not image_files:
            _log("No images found in the source directory.", level="info")
            return

        # Build the set of labeled keys (sensor_id, timestamp)
        _log("Building key set from label files...", level="info")
        labeled_keys = set()
        for f_path in label_files:
            key = self._get_image_key(os.path.basename(f_path))
            if key:
                labeled_keys.add(key)

        _log(f"Found {len(labeled_keys)} unique label keys.", level="info")

        # Find images that are NOT in the labeled set
        _log(
            "Finding unlabeled images... (This may take a moment)", level="info"
        )
        tasks = []  # This will be our list of files to copy

        total_images = len(image_files)
        report_interval = max(1, total_images // 10)

        # Use a list to store files we couldn't parse, to log at the end
        parse_failures = []

        for i, f_path in enumerate(image_files):
            key = self._get_image_key(
                os.path.basename(f_path), log_failure=False
            )

            if key:
                if key not in labeled_keys:
                    tasks.append(f_path)
            else:
                parse_failures.append(os.path.basename(f_path))

            if (
                i == 0
                or (i + 1) == total_images
                or (i + 1) % report_interval == 0
            ):
                _log(
                    f"  Scanning progress: {i + 1}/{total_images} images checked.",
                    level="info",
                )

        if parse_failures:
            _log(
                f"Could not parse a key for {len(parse_failures)} image files (e.g., '{parse_failures[0]}').",
                level="info",
            )

        # Copy the files
        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Unlabeled image copy")

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
        Creates a test set of random images for a camera.

        This method combines `_get_random_images` to find a list
        of files and then uses parallel copy logic to move
        them into the specified destination folder.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder where all images will be copied.
            It will be created if it does not exist.
        camera_id : str
            The ID of the camera (e.g., "CAM_XX_00").
        num_images : int
            The total number of images to try and retrieve.
        start_year : int, optional
            The starting year (inclusive) to search. Defaults to 2023.
        end_year : int, optional
            The ending year (inclusive) to search. Defaults to 2024.
        hour_window : tuple(int, int), optional
            A (start_hour, end_hour) tuple in Eastern Time.
            Defaults to None (all hours).
        max_workers : int, optional
            The maximum number of worker threads to use for copying.
            If None, the default for ThreadPoolExecutor is used.
        """

        _log(
            f"Starting test image set creation. Destination: {destination_folder}",
            level="info",
        )

        # Setup destination folder
        if not self._setup_destination_dir(destination_folder):
            return

        # Get the list of random images
        _log(
            f"Selecting {num_images} random images for {camera_id}...",
            level="info",
        )
        tasks = self._get_random_images(
            camera_id,
            start_year,
            end_year,
            num_images=num_images,
            hour_window=hour_window,
        )

        # Copy files in parallel
        # The helper will log the file count, handle the
        # "no tasks" case, and execute the copy.
        self._parallel_copy_files(tasks, max_workers, "Test set creation")

    def _format_abbreviated_events_for_pull(self, flood_event_csv_path):
        """
        Loads and formats flood event data from a CSV file.

        Reads the specified CSV, keeps only necessary columns, converts
        time strings to timezone-aware datetime objects, and creates
        a 'camera_ID' column from the 'sensor_ID'.

        Parameters
        ----------
        flood_event_csv_path : str or os.PathLike
            The file path to the abbreviated flood events CSV.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ['camera_ID', 'start_time_UTC',
            'end_time_UTC'].

        """
        columns_to_read = ["sensor_ID", "start_time_UTC", "end_time_UTC"]
        final_columns = ["camera_ID", "start_time_UTC", "end_time_UTC"]
        date_format = "%Y-%m-%d %H:%M:%S%z"

        flood_events = (
            # Read in necessary columns from abbr flood events csv
            pd.read_csv(flood_event_csv_path, usecols=columns_to_read).assign(
                # Convert times from strings to datetimes
                start_time_UTC=lambda x: pd.to_datetime(
                    x["start_time_UTC"], format=date_format
                ),
                end_time_UTC=lambda x: pd.to_datetime(
                    x["end_time_UTC"], format=date_format
                ),
                # Create camera ID column because sensor images are
                # stored with camera name
                camera_ID=lambda x: "CAM_" + x["sensor_ID"],
            )
        )

        # Drop unnecessary sensor_id column from dataframe
        return flood_events[final_columns]

    def _list_files_in_archive_date_dir(self, date, camera_id):
        """
        Lists all files in the archive directory for a specific date and
        camera.

        Constructs a directory path based on the archive structure
        (e.g., .../<YYYY> Archive/<camera_id>/<YYYY-MM-DD>/)
        and returns a list of all files within it.

        Parameters
        ----------
        date : datetime.date or datetime.datetime
            The date for which to list files.
        camera_id : str
            The ID of the camera.

        Returns
        -------
        list[str]
            A list of absolute file paths. Returns an empty list if the
            directory does not exist or an error occurs.

        """
        # Set directory to match provided date
        directory = (
            Path(self.drive)
            / f"{date.year} Archive"
            / camera_id
            / date.strftime("%Y-%m-%d")
        )

        if not os.path.isdir(directory):
            # This is an informational message, not a critical failure
            _log(
                f"Directory not found (this may be expected): {directory}",
                level="info",
            )
            return []

        try:
            # Re-check with Path object method
            if not directory.is_dir():
                return []

            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]

        except OSError as e:
            # e.g., if you lack read permissions on 'directory'
            _log(
                f"Error reading files from directory '{directory}'. Reason: {e}",
                level="error",
            )
            return []  # Return an empty list on failure

    def _list_files_in_dir(self, directory_path):
        """
        Lists all files in a given directory.

        Parameters
        ----------
        directory_path : str or os.PathLike
            The directory to scan.

        Returns
        -------
        list[str]
            A list of absolute file paths. Returns an empty list
            if the directory is invalid or an error occurs.
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            _log(f"Directory not found: {directory_path}", level="info")
            return []

        try:
            return [os.fspath(p) for p in directory.iterdir() if p.is_file()]
        except OSError as e:
            _log(
                f"Error reading files from '{directory_path}'. Reason: {e}",
                level="error",
            )
            return []

    def _filter_files_by_condition(self, file_list, condition_func):
        """
        Filters files based on a timestamp-checking function.

        Iterates a file list, parses the UTC timestamp from each
        filename, and applies the `condition_func` to it.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        condition_func : callable
            A function that accepts one argument (a timezone-aware
            UTC datetime object) and returns True if the file
            should be included in the list, False otherwise.

        Returns
        -------
        list[str]
            The sub-list of file paths that satisfy the condition.
        """
        filtered_files = []

        # For each file in provided list
        for file in file_list:
            try:
                # strip timestamp from file name
                file_name = os.path.basename(file)
                file_timestamp_str = self._extract_timestamp(file_name)

                # and localize to time aware UTC
                file_timestamp_naive = datetime.strptime(
                    file_timestamp_str, "%Y%m%d%H%M%S"
                )
                file_timestamp_utc = file_timestamp_naive.replace(
                    tzinfo=timezone.utc
                )

                # Apply the provided condition function
                if condition_func(file_timestamp_utc):
                    filtered_files.append(file)
            except (IndexError, ValueError, TypeError):
                # Log the specific file that failed parsing, continue
                _log(
                    f"Skipping file: Could not parse timestamp from '{file_name}'.",
                    level="info",
                )
                continue

        return filtered_files

    def _filter_files_by_start_and_end_time(
        self, file_list, start_time, end_time
    ):
        """
        Filters a file list to those within a specified time range.

        Assumes filenames contain a UTC timestamp in the format
        '..._YYYYMMDDHHMMSS.ext'. Both start_time and end_time must be
        timezone-aware.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        start_time : datetime.datetime
            The timezone-aware start of the time range (inclusive).
        end_time : datetime.datetime
            The timezone-aware end of the time range (inclusive).

        Returns
        -------
        list[str]
            The sub-list of file paths that fall within the time range.

        """
        # Comparing aware and naive datetimes raises a TypeError.
        # This check ensures the inputs are valid before starting the
        # loop.
        if start_time.tzinfo is None or end_time.tzinfo is None:
            _log(
                "Cannot filter files: start_time and end_time must be timezone-aware.",
                level="error",
            )
            return []

        # Define the condition and pass it to the helper
        condition = lambda ft_utc: start_time <= ft_utc <= end_time

        return self._filter_files_by_condition(file_list, condition)

    def _filter_files_eastern_time_window(
        self, file_list, start_hour_east, end_hour_east
    ):
        """
        Filters files to those between specific hours in Eastern Time.

        Parameters
        ----------
        file_list : list[str]
            A list of file paths to filter.
        start_hour_east : int
            The start hour in Eastern Time (inclusive), e.g., 6.
        end_hour_east : int
            The end hour in Eastern Time (inclusive), e.g., 18.

        Returns
        -------
        list[str]
            The sub-list of file paths that fall within the time range.
        """
        try:
            # Define the timezone
            eastern_tz = ZoneInfo("America/New_York")
        except Exception as e:
            _log(
                f"Could not load timezone 'America/New_York'. Error: {e}",
                level="error",
            )
            return []

        condition = lambda ft_utc: (
            start_hour_east
            <= ft_utc.astimezone(eastern_tz).hour
            <= end_hour_east
        )

        return self._filter_files_by_condition(file_list, condition)

    def _setup_destination_dir(self, destination_folder):
        """
        Ensures the destination folder exists and sets it as the
        copy target.

        This method creates the folder (including parent directories)
        if it does not exist. It also sets the
        `self.copy_destination_dir` attribute, which is required
        by the `_copy_file` method.

        Parameters
        ----------
        destination_folder : str or os.PathLike
            The path to the folder to create.

        Returns
        -------
        bool
            True if the directory was created or already exists,
            False if an OSError occurred.
        """
        try:
            os.makedirs(destination_folder, exist_ok=True)
            self.copy_destination_dir = destination_folder
            _log(
                f"Ensured destination folder exists: {destination_folder}",
                level="info",
            )
            return True
        except OSError as e:
            _log(
                f"FATAL: Could not create destination folder '{destination_folder}'. Reason: {e}",
                level="error",
            )
            return False

    def _parallel_copy_files(self, tasks, max_workers, task_name="File copy"):
        """
        Copies a list of files in parallel to the destination
        directory.

        Uses a ThreadPoolExecutor to copy all files from the
        `tasks` list to the `self.copy_destination_dir`. It logs
        progress and a final summary.

        Parameters
        ----------
        tasks : list[str]
            A list of absolute file paths to be copied.
        max_workers : int or None
            The maximum number of worker threads to use.
        task_name : str, optional
            A human-readable name for the task, used in log
            messages. Defaults to "File copy".

        """
        total_files = len(tasks)
        if total_files == 0:
            _log(
                f"No files found to copy for task: '{task_name}'.", level="info"
            )
            return

        # Determine report interval for copying
        copy_report_interval = max(1, total_files // 10)

        _log(
            f"Starting parallel copy of {total_files} files (max_workers={max_workers or 'default'})...",
            level="info",
        )
        success_count = 0
        fail_count = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # map() submits all tasks and returns an iterator of results
            # in the order the tasks were submitted.
            results = executor.map(self._copy_file, tasks)

            # Iterate over results to count success/failure
            for i, result in enumerate(results):
                # _copy_file returns a truthy value on success
                # and a falsy (False) on failure.
                if result:
                    # result is truthy, so it's a success
                    success_count += 1
                else:
                    # result is falsy (False), so it's a failure
                    fail_count += 1

                if (
                    i == 0
                    or (i + 1) == total_files
                    or (i + 1) % copy_report_interval == 0
                ):
                    _log(
                        f"  Copy progress: {i + 1}/{total_files} files processed.",
                        level="info",
                    )

        _log(
            f"{task_name} complete. Copied: {success_count}, Failed: {fail_count}",
            level="info",
        )

    def _copy_file(self, file_path):
        """
        Copies a single file to the class's destination folder.

        Relies on `self.copy_destination_dir` being set by the
        calling method. Logs errors and returns a status.

        Parameters
        ----------
        file_path : str or os.PathLike
            The path to the source file to be copied.

        Returns
        -------
        bool
            True if the copy is successful, False otherwise.

        """
        # Check the destination folder has been set by the calling method
        if not hasattr(self, "copy_destination_dir"):
            _log(
                "Cannot copy: self.copy_destination_dir is not set.",
                level="error",
            )
            return False  # Return falsy on failure

        try:
            # shutil.copy will copy the file and its permissions.
            shutil.copy(file_path, self.copy_destination_dir)

            return True  # Return truthy on success

        except FileNotFoundError:
            # This is a common, expected error
            _log(f"File not found, could not copy: {file_path}", level="error")
            return False  # Return falsy on failure

        except shutil.Error as e:
            # This catches specific shutil errors (e.g., "Disk Full")
            _log(
                f"shutil error copying {file_path}. Reason: {e}", level="error"
            )
            return False  # Return falsy on failure

        except OSError as e:
            # This catches lower-level OS errors (e.g., "Permission Denied")
            _log(f"OS error copying {file_path}. Reason: {e}", level="error")
            return False  # Return falsy on failure

    def _get_random_images(
        self, camera_id, start_year, end_year, num_images=10, hour_window=None
    ):
        """
        Selects a random sample of images for a given camera.

        It searches within a year range, optionally filters for an
        Eastern Time hour window, and attempts to pull a specified
        number of images. The sampling is distributed by picking
        random dates and taking a small, random number of images
        from each date to ensure variety.

        Parameters
        ----------
        camera_id : str
            The ID of the camera (e.g., "CAM_XX_00").
        start_year : int
            The starting year (inclusive) to search for images.
        end_year : int
            The ending year (inclusive) to search for images.
        num_images : int, optional
            The total number of images to try and retrieve.
            Defaults to 10.
        hour_window : tuple(int, int), optional
            A (start_hour, end_hour) tuple in Eastern Time.
            If provided, only images within this window (inclusive)
            will be selected. Defaults to None (all hours).

        Returns
        -------
        list[str]
            A list of absolute file paths to the selected images.
            May contain fewer images than `num_images` if not
            enough were found.
        """
        _log(
            f"Starting random image selection for '{camera_id}' ({start_year}-{end_year})...",
            level="info",
        )

        try:
            date_list = self._generate_date_list(start_year, end_year)
            random.shuffle(date_list)
        except Exception as e:
            _log(
                f"FATAL: Failed to generate or shuffle date list. Reason: {e}",
                level="error",
            )
            return []

        selected_images = []

        while len(selected_images) < num_images and date_list:
            date = date_list.pop()

            # Construct the directory path
            date_dir = (
                Path(self.drive)
                / f"{date.year} Archive"
                / camera_id
                / date.strftime("%Y-%m-%d")
            )

            # _list_files_in_dir handles logging if dir not found
            images = self._list_files_in_dir(date_dir)

            if not images:
                continue  # Skip this date if dir was empty or missing

            # Apply hour window filter if provided
            if hour_window:
                try:
                    start_hour_east, end_hour_east = hour_window
                    filtered_images = self._filter_files_eastern_time_window(
                        images, start_hour_east, end_hour_east
                    )
                except Exception as e:
                    _log(
                        f"Error applying time filter for {date_dir}: {e}",
                        level="error",
                    )
                    continue  # Skip this day
            else:
                filtered_images = images  # Use all images

            if not filtered_images:
                continue  # Skip if filter produced no results

            random.shuffle(filtered_images)

            # This logic takes a random small chunk (1-15% of total
            # requested) from the day to distribute the sample.
            percentage = random.uniform(0.01, 0.15)
            num_to_select = max(1, int(num_images * percentage))

            # How many do we still need?
            needed = num_images - len(selected_images)

            # How many are available on this day?
            available = len(filtered_images)

            # Take the minimum of the number we want, need, and have
            num_to_add = min(num_to_select, needed, available)

            selected_images.extend(filtered_images[:num_to_add])

        if len(selected_images) < num_images:
            _log(
                f"Selection complete. Warning: Only found {len(selected_images)} "
                f"of {num_images} requested images.",
                level="info",
            )
        else:
            _log(
                f"Successfully selected {len(selected_images)} random images.",
                level="info",
            )

        return selected_images

    @staticmethod
    def _get_image_key(filename, log_failure=True):
        """
        Extracts the (sensor, timestamp) key from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.
        log_failure : bool, optional
            If True, logs a message if a key cannot be parsed.
            Defaults to True.

        Returns
        -------
        tuple (str, str) or None
            A tuple of (sensor_id, timestamp) or None if either
            part cannot be parsed.
        """
        sensor_id = ImageHandler._extract_sensor_name(filename)
        timestamp = ImageHandler._extract_timestamp(filename)

        if sensor_id and timestamp:
            return (sensor_id, timestamp)

        if log_failure:
            _log(
                f"Skipping file: Could not parse key from '{filename}'.",
                level="info",
            )
        return None

    @staticmethod
    def _extract_timestamp(filename):
        """
        Extracts UTC timestamp (YYYYMMDDHHMMSS) from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.

        Returns
        -------
        str or None
            The 14-digit timestamp string, or None if not found.
        """
        # Regular expression pattern to match the UTC timestamp
        pattern = r"\d{14}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _extract_sensor_name(filename):
        """
        Extracts sensor ID (CAM_XX_00) from a filename.

        Parameters
        ----------
        filename : str
            The basename of the file.

        Returns
        -------
        str or None
            The sensor ID string, or None if not found.
        """
        # Regular expression pattern to match the sensor name
        pattern = r"CAM_[A-Z]{2}_[0-9]{2}"

        match = re.search(pattern, filename)
        return match.group(0) if match else None

    @staticmethod
    def _generate_date_list(start_year, end_year):
        """
        Generates a list of all dates within a given year range.

        Parameters
        ----------
        start_year : int
            The starting year (inclusive).
        end_year : int
            The ending year (inclusive).

        Returns
        -------
        list[datetime.date]
            A list of all dates from YYYY-01-01 to YYYY-12-31
            for the specified year range.

        Raises
        ------
        ValueError
            If `end_year` is before `start_year`.
        """
        if end_year < start_year:
            raise ValueError("end_year cannot be before start_year")

        _log(
            f"Generating date list from {start_year} to {end_year}...",
            level="info",
        )
        date_list = []
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()

        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)

        _log(f"Generated {len(date_list)} dates.", level="info")
        return date_list
