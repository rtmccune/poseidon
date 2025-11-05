import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
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


def filter_abbr_flood_csv_by_eastern_time(
    input_path, output_path, min_hour=6, max_hour=19
):
    """
    Filter a CSV by hour of day in US/Eastern Time.

    Reads a CSV file containing a 'start_time_UTC' column. It converts
    these timestamps to 'America/New_York' (Eastern Time), correctly
    handling EST/EDT. The DataFrame is then filtered to include only
    rows where the hour of the day falls within the specified range
    [min_hour, max_hour). The result is saved to a new CSV file.

    Parameters
    ----------
    input_path : str
        The file path to the input CSV. Must contain a
        'start_time_UTC' column.
    output_path : str
        The file path where the filtered CSV will be saved.
    min_hour : int, optional
        The starting hour (inclusive, 0-23) for the filter.
        Default is 6 (representing 6:00 AM).
    max_hour : int, optional
        The ending hour (exclusive, 0-24) for the filter.
        Default is 20 (representing 8:00 PM, so it includes
        hours 6 through 19).

    Returns
    -------
    None
        This function does not return a value. It saves the filtered
        DataFrame to the file specified by `output_path` and logs
        status messages.

    Raises
    ------
    FileNotFoundError
        If the file at `input_path` is not found.
    Exception
        Catches other potential errors during file reading,
        processing, or writing, such as permissions errors.
    """
    try:
        _log(f"Reading data from '{input_path}'")
        df = pd.read_csv(input_path)

        # Ensure the required UTC column exists
        required_column = "start_time_UTC"
        if required_column not in df.columns:
            _log(
                f"Input CSV must contain the column '{required_column}'",
                level="error",
            )
            return

        _log(
            f"Converting UTC to Eastern Time and filtering for hours {min_hour}:00 to {max_hour}:00"
        )

        # Convert the 'start_time_UTC' column to datetime objects
        df["start_time_UTC"] = pd.to_datetime(
            df["start_time_UTC"], errors="coerce", utc=True
        )

        # Drop rows where the date conversion failed
        df.dropna(subset=["start_time_UTC"], inplace=True)

        # Localize the UTC times and convert to US/Eastern.
        df["start_time_ET"] = df["start_time_UTC"].dt.tz_convert(
            "America/New_York"
        )

        # Get the hour of the day
        hour_of_day = df["start_time_ET"].dt.hour

        # Filter the DataFrame
        filtered_df = df[
            (hour_of_day >= min_hour) & (hour_of_day < max_hour)
        ].copy()

        _log(f"Saving filtered data to '{output_path}'")
        filtered_df.to_csv(output_path, index=False)

        _log(f"Success! Filtered file saved with {len(filtered_df)} rows.")

    except FileNotFoundError:
        _log(f"The file '{input_path}' was not found.", level="error")
    except Exception as e:
        _log(f"An unexpected error occurred: {e}", level="error")


def create_flood_csvs_and_subfolders(
    abbr_events_path, full_events_path, output_parent_dir, padding_hours=3
):
    """
    Filters full event data based on abbreviated event windows and saves
    each event into its own subfolder.

    This function reads a list of "abbreviated" flood events (defined
    in EST) and a "full" data record (defined in UTC). It applies a
    fixed-offset EST timezone ('Etc/GMT+5'), converts the event windows
    to UTC, applies padding, and then filters the full data for each
    event. Each filtered event's data is saved as a new CSV in its
    own uniquely named subfolder.

    Parameters
    ----------
    abbr_events_path : str
        File path to the abbreviated flood events CSV.
        Requires columns: 'sensor_ID', 'start_time_EST', 'end_time_EST'.
    full_events_path : str
        File path to the full data record (all sensor readings).
        Requires columns: 'sensor_ID', 'time_UTC'.
    output_parent_dir : str
        The parent directory where new subfolders for each event
        will be created.
    padding_hours : int, optional
        The number of hours to add before the start and after the end
        of each event window for padding. Default is 3.

    Returns
    -------
    None
        Files are written to disk. Logs status.

    Raises
    ------
    FileNotFoundError
        If `abbr_events_path` or `full_events_path` is not found.
    Exception
        Catches other potential errors during file loading or time
        conversion.
    """
    try:
        _log(f"Loading abbreviated events from: {abbr_events_path}")
        abbr_df = pd.read_csv(abbr_events_path)

        _log(f"Loading full data record from: {full_events_path}")
        full_df = pd.read_csv(full_events_path)
    except FileNotFoundError as e:
        _log(f"File not found. {e}", level="error")
        return
    except Exception as e:
        _log(f"Error loading files: {e}", level="error")
        return

    _log(f"Processing event time windows with {padding_hours}-hour padding...")

    # --- Timezone Processing ---
    padding = pd.Timedelta(hours=padding_hours)

    # Convert naive EST strings to aware datetime objects
    try:
        # Parse the EST/EDT strings. pd.to_datetime() will automatically
        # make them "aware" because the strings likely contain offsets.
        start_time_aware = pd.to_datetime(
            abbr_df["start_time_EST"], errors="coerce"
        )
        end_time_aware = pd.to_datetime(
            abbr_df["end_time_EST"], errors="coerce"
        )

        # Convert to UTC (from whatever offset they had) and apply
        # padding. This replaces the .tz_localize()...tz_convert() chain.
        abbr_df["start_time_UTC_padded"] = (
            start_time_aware.dt.tz_convert("UTC") - padding
        )
        abbr_df["end_time_UTC_padded"] = (
            end_time_aware.dt.tz_convert("UTC") + padding
        )

    except Exception as e:
        _log(
            f"Error processing time columns ('start_time_EST'): {e}",
            level="error",
        )
        _log(
            "Please ensure 'start_time_EST' and 'end_time_EST' are valid timestamps.",
            level="error",
        )
        return

    # Create string representations for file naming
    abbr_df["start_str"] = abbr_df["start_time_UTC_padded"].dt.strftime(
        "%Y%m%d%H%M%S"
    )
    abbr_df["end_str"] = abbr_df["end_time_UTC_padded"].dt.strftime(
        "%Y%m%d%H%M%S"
    )

    # Ensure the full data's time column is timezone-aware UTC
    full_df["time_UTC"] = pd.to_datetime(full_df["time_UTC"], utc=True)

    _log(f"Generating filtered CSVs in '{output_parent_dir}'...")
    os.makedirs(output_parent_dir, exist_ok=True)

    created_count = 0
    total_events = len(abbr_df)
    _log(f"Starting to process {total_events} events...")

    # --- Main Loop: Filter and Save ---
    for index, row in abbr_df.iterrows():
        sensor_id = row["sensor_ID"]
        start_time = row["start_time_UTC_padded"]
        end_time = row["end_time_UTC_padded"]

        # Filter the full DataFrame
        filtered_df = full_df[
            (full_df["sensor_ID"] == sensor_id)
            & (full_df["time_UTC"] >= start_time)
            & (full_df["time_UTC"] <= end_time)
        ]

        if filtered_df.empty:
            continue

        # --- File and Folder Path Generation ---
        folder_name = f"{sensor_id}_{row['start_str']}_{row['end_str']}"
        subfolder_path = os.path.join(output_parent_dir, folder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        csv_filename = f"{folder_name}.csv"
        output_path = os.path.join(subfolder_path, csv_filename)

        filtered_df.to_csv(output_path, index=False)
        created_count += 1

    _log(
        f"Script complete. Successfully created {created_count} CSV files in subfolders."
    )


def extract_camera_name(filename):
    """
    Extracts sensor ID from filenames based on a specific pattern.

    Parameters
    ----------
    filename : str
        The filename to parse (e.g., "CAM_NC_01_20230101120000.jpg").

    Returns
    -------
    str or None
        The matched camera name (e.g., "CAM_NC_01") or None if
        no match is found.
    """
    # Pattern: CAM_XX_00
    pattern = r"CAM_[A-Z]{2}_[0-9]{2}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def extract_timestamp(filename):
    """
    Extracts a 14-digit UTC timestamp from filenames.

    Parameters
    ----------
    filename : str
        The filename to parse (e.g., "CAM_NC_01_20230101120000.jpg").

    Returns
    -------
    str or None
        The matched timestamp string (e.g., "20230101120000") or
        None if no match is found.
    """
    # Pattern: 20230101120000
    pattern = r"\d{14}"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def organize_images_into_flood_events(
    image_folder, csv_file, destination_folder, subfolder_name, padding_hours=3
):
    """
    Efficiently organizes images into folders based on flood event time
    ranges.

    This "single-pass" method builds a lookup of events first, then
    iterates through all images once, placing them in the correct
    event folder.

    Parameters
    ----------
    image_folder : str
        Path to the directory containing the original images.
    csv_file : str
        Path to the CSV file defining flood events.
        Requires columns: 'sensor_ID', 'start_time_UTC', 'end_time_UTC'.
    destination_folder : str
        Path where the organized event folders should be created.
    subfolder_name : str
        Name of the subfolder inside each event folder to store the
        images (e.g., "orig_images").
    padding_hours : int, optional
        The number of hours to add before the start and after the end
        of each event window. Default is 3.

    Returns
    -------
    None
        Files are copied into new directories. Logs status.

    Raises
    ------
    FileNotFoundError
        If `csv_file` or `image_folder` is not found.
    """

    # --- 1. Build Event Lookup Dictionary ---
    _log(f"Reading event data from '{csv_file}'")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        _log(f"CSV file not found at '{csv_file}'", level="error")
        return

    _log(f"Applying {padding_hours}-hour padding to event windows...")

    padding = pd.Timedelta(hours=padding_hours)

    # Apply padding and create folder names
    df["start_time_UTC"] = (
        pd.to_datetime(df["start_time_UTC"], utc=True) - padding
    )
    df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True) + padding

    df["start_time_str"] = df["start_time_UTC"].dt.strftime("%Y%m%d%H%M%S")
    df["end_time_str"] = df["end_time_UTC"].dt.strftime("%Y%m%d%H%M%S")

    df["camera_ID"] = "CAM_" + df["sensor_ID"]
    df["folder_name"] = (
        df["sensor_ID"] + "_" + df["start_time_str"] + "_" + df["end_time_str"]
    )

    # Create the lookup: { camera_ID: [ (start, end, folder_name), ... ] }
    event_lookup = defaultdict(list)
    for _, row in df.iterrows():
        event_details = (
            row["start_time_UTC"],
            row["end_time_UTC"],
            row["folder_name"],
        )
        event_lookup[row["camera_ID"]].append(event_details)

    _log(
        f"Built lookup for {len(df)} events across {len(event_lookup)} cameras."
    )

    # --- 2. Process All Images in a Single Pass ---
    _log(f"Scanning image folder: '{image_folder}'...")

    try:
        all_files = [
            f
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except FileNotFoundError:
        _log(f"Image folder not found at '{image_folder}'", level="error")
        return

    total_files = len(all_files)  # <-- Total for progress
    if not all_files:
        _log("No image files found in the source directory.")
        return

    # --- Progress Reporting Setup ---
    # Only set a step if there are enough files to warrant updates
    progress_step = 0
    if total_files > 10:
        progress_step = total_files // 10  # 10% increment
    # --- End Progress Setup ---

    copy_count = 0
    skip_count = 0
    copied_files = set()  # Keep track of files already copied

    _log(f"Organizing {total_files} images...")

    # Use enumerate to get the index 'i'
    for i, filename in enumerate(all_files):

        # --- Progress Reporting Logic ---
        # Report at 10%, 20%, ... 90%
        # Check i > 0 to avoid reporting at 0%
        if progress_step > 0 and i % progress_step == 0 and i > 0:
            percent_complete = int((i / total_files) * 100)
            _log(
                f"  ...progress: {percent_complete}% complete ({i} of {total_files} images processed)"
            )
        # --- End Progress Reporting ---

        # Extract info from the image filename
        camera_name = extract_camera_name(filename)
        timestamp_str = extract_timestamp(filename)

        if not camera_name or not timestamp_str:
            skip_count += 1
            continue

        try:
            img_time = pd.to_datetime(
                timestamp_str, format="%Y%m%d%H%M%S", utc=True
            )
        except ValueError:
            skip_count += 1  # Skip files with bad timestamps
            continue

        # Find a matching event for this image
        if camera_name in event_lookup:
            for start_time, end_time, folder_name in event_lookup[camera_name]:

                # Check if the image time is within the event window
                if start_time <= img_time <= end_time:

                    # --- 3. Create Folder and Copy File ---
                    dest_path = os.path.join(
                        destination_folder, folder_name, subfolder_name
                    )
                    os.makedirs(dest_path, exist_ok=True)

                    src_file = os.path.join(image_folder, filename)
                    dest_file = os.path.join(dest_path, filename)

                    # Only copy if it doesn't already exist
                    if dest_file not in copied_files and not os.path.exists(
                        dest_file
                    ):
                        shutil.copy2(
                            src_file, dest_file
                        )  # copy2 preserves metadata
                        copy_count += 1
                        copied_files.add(dest_file)

                    # An image might belong to multiple overlapping events
                    # So we continue checking, not 'break'

    _log("--- Image Organization Complete ---")
    _log(f"Successfully copied: {copy_count} files")
    _log(f"Skipped (no match):  {skip_count} files")


def prepare_job_lists(image_folder: str, num_jobs: int, output_dir: str = None):
    """
    Prepare file lists for an HPC job array.

    Scans a source directory for files, filters hidden files (starting with '.'),
    and splits the sorted file list into a specified number of new text files.
    These output files are designed to be used as inputs for an HPC job array,
    where each job processes one text file listing its assigned images.

    Parameters
    ----------
    image_folder : str
        The absolute or relative path to the directory containing the files
        to be processed.
    num_jobs : int
        The total number of job array tasks, which determines the number
        of output file lists to create.
    output_dir : str, optional
        The directory where the 'file_list_*.txt' files will be saved.
        If None (default), a new directory 'job_file_lists' will be
        created in the parent directory of `image_folder`.

    Returns
    -------
    None
        This function does not return a value. It creates files on disk as
        a side effect and prints status messages to the console.

    """
    # Resolve the input path immediately to get a full, absolute path.
    image_path = Path(image_folder).resolve()

    # Check if the source directory exists early on
    if not image_path.is_dir():
        print(
            f"Error: The specified image folder does not exist: '{image_path}'"
        )
        return

    if output_dir is None:
        # Create 'job_file_lists' in the parent directory of the source images.
        final_output_path = image_path.parent / "job_file_lists"
        print(
            f"Output directory not specified. Defaulting to: {final_output_path}"
        )
    else:
        # If an output directory is specified, resolve it to an absolute path too.
        final_output_path = Path(output_dir).resolve()
        print(f"Using specified output directory: {final_output_path}")

    # Create the final output directory if it doesn't exist.
    final_output_path.mkdir(parents=True, exist_ok=True)

    # Get sorted list of files using modern pathlib
    files = sorted(
        [
            f.name
            for f in image_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
    )

    if not files:
        print(
            f"Warning: No valid files found in '{image_path}'. No lists created."
        )
        return

    total_files = len(files)

    # Split into chunks for number of jobs
    file_chunks = np.array_split(files, num_jobs)

    for i, chunk in enumerate(file_chunks):
        list_filename = final_output_path / f"file_list_{i+1}.txt"

        with open(list_filename, "w") as f:
            for filename in chunk:
                f.write(filename + "\n")

    print(
        f"\nSuccess! Created {len(file_chunks)} file lists in '{final_output_path}' for a total of {total_files} files."
    )
