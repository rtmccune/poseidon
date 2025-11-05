# Standard library imports
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime

# Third-party imports
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


def filter_abbr_flood_csv_by_eastern_time(input_path, output_path, min_hour=6, max_hour=20):
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
        required_column = 'start_time_UTC'
        if required_column not in df.columns:
            _log(f"Input CSV must contain the column '{required_column}'", level="error")
            return

        _log(f"Converting UTC to Eastern Time and filtering for hours {min_hour}:00 to {max_hour}:00")

        # Convert the 'start_time_UTC' column to datetime objects
        df['start_time_UTC'] = pd.to_datetime(df['start_time_UTC'], errors='coerce')

        # Drop rows where the date conversion failed
        df.dropna(subset=['start_time_UTC'], inplace=True)

        # Localize the UTC times and convert to US/Eastern.
        df['start_time_ET'] = df['start_time_UTC'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

        # Get the hour of the day
        hour_of_day = df['start_time_ET'].dt.hour

        # Filter the DataFrame
        filtered_df = df[(hour_of_day >= min_hour) & (hour_of_day < max_hour)].copy()

        _log(f"Saving filtered data to '{output_path}'")
        filtered_df.to_csv(output_path, index=False)

        _log(f"Success! Filtered file saved with {len(filtered_df)} rows.")

    except FileNotFoundError:
        _log(f"The file '{input_path}' was not found.", level="error")
    except Exception as e:
        _log(f"An unexpected error occurred: {e}", level="error")


def create_flood_csvs_and_subfolders(abbr_events_path, full_events_path, output_parent_dir, padding_hours=3):
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
    
    # 'Etc/GMT+5' is the IANA string for a fixed UTC-5 offset.
    fixed_est_tz = 'Etc/GMT+5'

    # Convert naive EST strings to aware datetime objects
    try:
        abbr_df['start_time_EST_aware'] = pd.to_datetime(
            abbr_df['start_time_EST']
        ).dt.tz_localize(fixed_est_tz)
        
        abbr_df['end_time_EST_aware'] = pd.to_datetime(
            abbr_df['end_time_EST']
        ).dt.tz_localize(fixed_est_tz)
        
    except Exception as e:
        _log(f"Error localizing time columns: {e}", level="error")
        _log("Please ensure 'start_time_EST' and 'end_time_EST' are valid timestamps.", level="error")
        return

    # Convert to UTC and apply padding
    abbr_df['start_time_UTC_padded'] = abbr_df['start_time_EST_aware'].dt.tz_convert('UTC') - padding
    abbr_df['end_time_UTC_padded'] = abbr_df['end_time_EST_aware'].dt.tz_convert('UTC') + padding

    # Create string representations for file naming
    abbr_df['start_str'] = abbr_df['start_time_UTC_padded'].dt.strftime('%Y%m%d%H%M%S')
    abbr_df['end_str'] = abbr_df['end_time_UTC_padded'].dt.strftime('%Y%m%d%H%M%S')

    # Ensure the full data's time column is timezone-aware UTC
    full_df['time_UTC'] = pd.to_datetime(full_df['time_UTC'], utc=True)

    _log(f"Generating filtered CSVs in '{output_parent_dir}'...")
    os.makedirs(output_parent_dir, exist_ok=True)
    
    created_count = 0
    total_events = len(abbr_df)
    _log(f"Starting to process {total_events} events...")

    # --- Main Loop: Filter and Save ---
    for index, row in abbr_df.iterrows():
        sensor_id = row['sensor_ID']
        start_time = row['start_time_UTC_padded']
        end_time = row['end_time_UTC_padded']
        
        # Filter the full DataFrame
        filtered_df = full_df[
            (full_df['sensor_ID'] == sensor_id) &
            (full_df['time_UTC'] >= start_time) &
            (full_df['time_UTC'] <= end_time)
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

    _log(f"Script complete. Successfully created {created_count} CSV files in subfolders.")
    

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
    Efficiently organizes images into folders based on flood event time ranges.

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
    df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True) - padding
    df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True) + padding
    
    df["start_time_str"] = df["start_time_UTC"].dt.strftime("%Y%m%d%H%M%S")
    df["end_time_str"] = df["end_time_UTC"].dt.strftime("%Y%m%d%H%M%S")
    
    df["camera_ID"] = "CAM_" + df["sensor_ID"]
    df["folder_name"] = df["sensor_ID"] + "_" + df["start_time_str"] + "_" + df["end_time_str"]

    # Create the lookup: { camera_ID: [ (start, end, folder_name), ... ] }
    event_lookup = defaultdict(list)
    for _, row in df.iterrows():
        event_details = (
            row["start_time_UTC"],
            row["end_time_UTC"],
            row["folder_name"]
        )
        event_lookup[row["camera_ID"]].append(event_details)

    _log(f"Built lookup for {len(df)} events across {len(event_lookup)} cameras.")

    # --- 2. Process All Images in a Single Pass ---
    _log(f"Scanning image folder: '{image_folder}'...")
    
    try:
        all_files = [
            f for f in os.listdir(image_folder) 
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    except FileNotFoundError:
        _log(f"Image folder not found at '{image_folder}'", level="error")
        return

    if not all_files:
        _log("No image files found in the source directory.")
        return

    copy_count = 0
    skip_count = 0
    copied_files = set() # Keep track of files already copied

    _log(f"Organizing {len(all_files)} images...")
    for filename in all_files:
        
        # Extract info from the image filename
        camera_name = extract_camera_name(filename)
        timestamp_str = extract_timestamp(filename)

        if not camera_name or not timestamp_str:
            skip_count += 1
            continue

        try:
            img_time = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S", utc=True)
        except ValueError:
            skip_count += 1 # Skip files with bad timestamps
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
                    if dest_file not in copied_files and not os.path.exists(dest_file):
                        shutil.copy2(src_file, dest_file) # copy2 preserves metadata
                        copy_count += 1
                        copied_files.add(dest_file)
                    
                    # An image might belong to multiple overlapping events
                    # So we continue checking, not 'break'
    
    _log("--- Image Organization Complete ---")
    _log(f"Successfully copied: {copy_count} files")
    _log(f"Skipped (no match):  {skip_count} files")