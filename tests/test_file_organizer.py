# Standard library imports
import os
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

# Import the functions to be tested
# (Assumes your script is saved as flood_processing.py)
from poseidon_utils.file_organizer import (
    _log,  # <-- FIX 1: Added _log to imports
    filter_abbr_flood_csv_by_eastern_time,
    create_flood_csvs_and_subfolders,
    extract_camera_name,
    extract_timestamp,
    organize_images_into_flood_events,
)

# --- Tests for Helper Functions ---


def test_log_info(capsys):
    """Tests that 'info' level logs to stdout."""
    _log("Test info message")
    captured = capsys.readouterr()
    assert "INFO: Test info message" in captured.out
    assert captured.err == ""


def test_log_error(capsys):
    """Tests that 'error' level logs to stderr."""
    _log("Test error message", level="error")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ERROR: Test error message" in captured.err


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("CAM_NC_01_20230101120000.jpg", "CAM_NC_01"),
        ("prefix_CAM_VA_12_suffix.png", "CAM_VA_12"),
        ("CAM_XX_99_20200101000000.jpeg", "CAM_XX_99"),
        ("no_match.jpg", None),
        ("CAM_nc_01_lowercase.jpg", None),  # Pattern requires uppercase letters
        ("CAM_ABC_01.jpg", None),  # Pattern requires 2 letters
        ("CAM_NC_1.jpg", None),  # Pattern requires 2 digits
    ],
)
def test_extract_camera_name(filename, expected):
    """Tests the camera name extraction regex."""
    assert extract_camera_name(filename) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("CAM_NC_01_20230101120000.jpg", "20230101120000"),
        ("prefix_20221231235959_suffix.png", "20221231235959"),
        ("CAM_VA_10_20211110100000.jpeg", "20211110100000"),
        ("only_13_digits_1234567890123.jpg", None),
        ("no_digits.jpg", None),
        ("20230101_120000.jpg", None),  # Not 14 consecutive digits
    ],
)
def test_extract_timestamp(filename, expected):
    """Tests the timestamp extraction regex."""
    assert extract_timestamp(filename) == expected


# --- Tests for filter_abbr_flood_csv_by_eastern_time ---


def test_filter_by_eastern_time_success(tmp_path):
    """
    Tests successful filtering, including handling of both
    EST (winter) and EDT (summer) timezones.
    """
    # 1. Create mock input CSV
    input_csv = tmp_path / "input.csv"
    data = {
        "start_time_UTC": [
            # EDT (UTC-4) - 'America/New_York'
            # FIX 2: Removed 'Z' to make timestamps naive, as script expects
            "2023-08-15T10:00:00",  # 6:00 AM EDT (Keep, min_hour=6)
            "2023-08-15T14:00:00",  # 10:00 AM EDT (Keep)
            "2023-08-15T09:59:00",  # 5:59 AM EDT (Drop)
            "2023-08-16T00:00:00",  # 8:00 PM EDT (Drop, max_hour=20 is exclusive)
            # EST (UTC-5) - 'America/New_York'
            "2023-11-15T11:00:00",  # 6:00 AM EST (Keep)
            "2023-11-15T15:00:00",  # 10:00 AM EST (Keep)
            "2023-11-15T10:59:00",  # 5:59 AM EST (Drop)
            "2023-11-16T01:00:00",  # 8:00 PM EST (Drop)
            "bad_date_string",  # (Drop)
        ],
        "row_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    pd.DataFrame(data).to_csv(input_csv, index=False)

    output_csv = tmp_path / "output.csv"

    # 2. Run the function with default hours (6-20)
    filter_abbr_flood_csv_by_eastern_time(str(input_csv), str(output_csv))

    # 3. Read and check the output
    assert output_csv.exists()  # This assertion should now pass
    result_df = pd.read_csv(output_csv)

    # Check that only the 4 correct rows were kept
    assert len(result_df) == 4
    assert result_df["row_id"].tolist() == [1, 2, 5, 6]

    # Check that the ET column was added and is correct
    assert "start_time_ET" in result_df.columns
    et_hours = (
        pd.to_datetime(result_df["start_time_ET"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.hour
    )
    assert all(h >= 6 for h in et_hours)
    assert all(h < 19 for h in et_hours)


def test_filter_by_eastern_time_file_not_found(tmp_path, capsys):
    """Tests logging when input file is missing."""
    input_path = "non_existent.csv"
    output_path = tmp_path / "output.csv"

    filter_abbr_flood_csv_by_eastern_time(input_path, str(output_path))

    captured = capsys.readouterr()
    assert "ERROR: The file 'non_existent.csv' was not found." in captured.err
    assert not output_path.exists()


def test_filter_by_eastern_time_missing_column(tmp_path, capsys):
    """Tests logging when the required 'start_time_UTC' column is missing."""
    input_csv = tmp_path / "input.csv"
    pd.DataFrame({"wrong_column": [1, 2]}).to_csv(input_csv, index=False)
    output_csv = tmp_path / "output.csv"

    filter_abbr_flood_csv_by_eastern_time(str(input_csv), str(output_csv))

    captured = capsys.readouterr()
    assert (
        "ERROR: Input CSV must contain the column 'start_time_UTC'"
        in captured.err
    )
    assert not output_csv.exists()


# --- Tests for create_flood_csvs_and_subfolders ---


def test_create_flood_csvs_and_subfolders_success(tmp_path):
    """
    Tests successful creation of subfolders and filtered CSVs,
    checking the fixed 'Etc/GMT+5' timezone logic and padding.
    """
    # 1. Create mock input files
    abbr_events_path = tmp_path / "abbr_events.csv"
    full_data_path = tmp_path / "full_data.csv"
    output_parent_dir = tmp_path / "output_events"

    # Abbr events (NOW as aware UTC-5)
    abbr_data = {
        "sensor_ID": ["S1", "S2"],
        "start_time_EST": [
            "2023-10-01 10:00:00-05:00",
            "2023-10-02 12:00:00-05:00",
        ],
        "end_time_EST": [
            "2023-10-01 12:00:00-05:00",
            "2023-10-02 13:00:00-05:00",
        ],
    }
    pd.DataFrame(abbr_data).to_csv(abbr_events_path, index=False)

    # Full data (UTC)
    full_data = {
        "sensor_ID": ["S1", "S1", "S1", "S1", "S1", "S2", "S2", "S2"],
        "time_UTC": [
            # Event 1 (S1): 10-12 EST -> 15-17 UTC. Padded (3hr): 12:00 to 20:00 UTC
            "2023-10-01 11:59:00Z",  # Out (before pad)
            "2023-10-01 12:00:00Z",  # In (on start pad)
            "2023-10-01 16:00:00Z",  # In (middle of event)
            "2023-10-01 20:00:00Z",  # In (on end pad, inclusive)
            "2023-10-01 20:01:00Z",  # Out (after pad)
            # Event 2 (S2): 12-13 EST -> 17-18 UTC. Padded (3hr): 14:00 to 21:00 UTC
            "2023-10-02 13:59:00Z",  # Out (before pad)
            "2023-10-02 14:30:00Z",  # In
            "2023-10-02 21:30:00Z",  # Out (after pad)
        ],
        "value": [100, 101, 102, 103, 104, 200, 201, 202],
    }
    pd.DataFrame(full_data).to_csv(full_data_path, index=False)

    # 2. Run the function (default padding=3)
    create_flood_csvs_and_subfolders(
        str(abbr_events_path), str(full_data_path), str(output_parent_dir)
    )

    # 3. Check results

    # Event 1: S1, Padded UTC: 2023-10-01 12:00:00 to 2023-10-01 20:00:00
    folder_name1 = "S1_20231001120000_20231001200000"
    event1_path = output_parent_dir / folder_name1 / f"{folder_name1}.csv"
    assert event1_path.exists(), "Event 1 CSV not created"

    # Event 2: S2, Padded UTC: 2023-10-02 14:00:00 to 2023-10-02 21:00:00
    folder_name2 = "S2_20231002140000_20231002210000"
    event2_path = output_parent_dir / folder_name2 / f"{folder_name2}.csv"
    assert event2_path.exists(), "Event 2 CSV not created"

    # Check content of event 1 CSV
    df1 = pd.read_csv(event1_path)
    assert len(df1) == 3
    assert df1["value"].tolist() == [101, 102, 103]

    # Check content of event 2 CSV
    df2 = pd.read_csv(event2_path)
    assert len(df2) == 1
    assert df2["value"].tolist() == [201]


def test_create_flood_csvs_file_not_found(tmp_path, capsys):
    """Tests logging when an input file is missing."""
    create_flood_csvs_and_subfolders(
        "non_existent.csv", "real.csv", str(tmp_path)
    )
    captured = capsys.readouterr()
    assert "ERROR: File not found." in captured.err


# --- Tests for organize_images_into_flood_events ---


@pytest.fixture
def image_setup(tmp_path):
    """Fixture to create a mock image directory and event CSV."""

    # 1. Create source image folder and mock images
    image_folder = tmp_path / "source_images"
    image_folder.mkdir()
    (image_folder / "CAM_NC_01_20230101123000.jpg").touch()  # Event 1
    (image_folder / "CAM_NC_01_20230101150000.jpg").touch()  # Event 1 (on edge)
    (image_folder / "CAM_NC_01_20230101085900.jpg").touch()  # Before event 1
    (image_folder / "CAM_VA_10_20230202103000.jpg").touch()  # Event 2
    (
        image_folder / "CAM_VA_10_20230202113000.jpg"
    ).touch()  # Event 2 & 3 (overlap)
    (image_folder / "CAM_XX_99_20230303030000.jpg").touch()  # No event
    (image_folder / "bad_name.jpg").touch()  # Bad name
    (image_folder / "not_an_image.txt").touch()  # Not image

    # 2. Create event CSV (times are UTC *before* padding)
    csv_file = tmp_path / "events.csv"
    event_data = {
        "sensor_ID": ["NC_01", "VA_10", "VA_10"],
        "start_time_UTC": [
            "2023-01-01 12:00:00Z",  # Event 1
            "2023-02-02 10:00:00Z",  # Event 2
            "2023-02-02 11:00:00Z",  # Event 3 (overlaps 2)
        ],
        "end_time_UTC": [
            "2023-01-01 12:00:00Z",  # Event 1 (short)
            "2023-02-02 11:00:00Z",  # Event 2
            "2023-02-02 12:00:00Z",  # Event 3
        ],
    }
    pd.DataFrame(event_data).to_csv(csv_file, index=False)

    # 3. Define output dir and params
    destination_folder = tmp_path / "organized_output"
    subfolder_name = "raw_images"
    padding_hours = 3

    return (
        image_folder,
        csv_file,
        destination_folder,
        subfolder_name,
        padding_hours,
    )


def test_organize_images_success_and_overlap(image_setup, capsys):
    """
    Tests that images are correctly copied into event folders,
    including handling of overlapping events.
    """
    image_folder, csv_file, dest_folder, subfolder, padding = image_setup

    # Run the function
    organize_images_into_flood_events(
        str(image_folder), str(csv_file), str(dest_folder), subfolder, padding
    )

    # --- Check results ---

    # Event 1: NC_01, 12:00-12:00 UTC. Padded (3hr): 09:00 to 15:00 UTC
    # Folder name: NC_01_20230101090000_20230101150000
    folder1 = "NC_01_20230101090000_20230101150000"
    img1_path = (
        dest_folder / folder1 / subfolder / "CAM_NC_01_20230101123000.jpg"
    )
    img2_path = (
        dest_folder / folder1 / subfolder / "CAM_NC_01_20230101150000.jpg"
    )
    assert img1_path.exists()
    assert img2_path.exists()  # 15:00:00 is <= 15:00:00

    # Event 2: VA_10, 10:00-11:00 UTC. Padded: 07:00 to 14:00 UTC
    # Folder name: VA_10_20230202070000_20230202140000
    folder2 = "VA_10_20230202070000_20230202140000"
    img3_path = (
        dest_folder / folder2 / subfolder / "CAM_VA_10_20230202103000.jpg"
    )
    img4_path = (
        dest_folder / folder2 / subfolder / "CAM_VA_10_20230202113000.jpg"
    )
    assert img3_path.exists()
    assert img4_path.exists()

    # Event 3: VA_10, 11:00-12:00 UTC. Padded: 08:00 to 15:00 UTC
    # Folder name: VA_10_20230202080000_20230202150000
    folder3 = "VA_10_20230202080000_20230202150000"
    img5_path = (
        dest_folder / folder3 / subfolder / "CAM_VA_10_20230202103000.jpg"
    )  # Overlap
    img6_path = (
        dest_folder / folder3 / subfolder / "CAM_VA_10_20230202113000.jpg"
    )  # Overlap
    assert img5_path.exists(), "Image 1 not copied to overlapping event"
    assert img6_path.exists(), "Image 2 not copied to overlapping event"

    # Check that files *not* in an event were *not* copied
    assert not (
        dest_folder / folder1 / subfolder / "CAM_NC_01_20230101085900.jpg"
    ).exists()

    # FIX 3: Corrected assertion count
    # 3 event folders + 3 subfolders + 6 images = 12 total items
    assert len(list(dest_folder.glob("**/*"))) == 12

    # Check logs
    captured = capsys.readouterr()

    # 7 images found (.txt is ignored)
    # 6 files copied (2 for event 1, 2 for event 2, 2 for event 3)
    # 1 file skipped (bad_name.jpg)
    # 2 files had no match (CAM_NC_01_...085900, CAM_XX_99_...)
    # Wait, the log logic is different:
    # `skip_count` is only for files with bad names/timestamps.
    # Total files = 7 images.
    # 1 skipped (bad_name.jpg).
    # 6 files processed.
    # Copied:
    # - NC_01_123000 -> folder1 (copy 1)
    # - NC_01_150000 -> folder1 (copy 2)
    # - VA_10_103000 -> folder2 (copy 3)
    # - VA_10_113000 -> folder2 (copy 4)
    # - VA_10_103000 -> folder3 (copy 5)
    # - VA_10_113000 -> folder3 (copy 6)
    # `copy_count` = 6
    # `skip_count` = 1 (for bad_name.jpg)
    assert "Successfully copied: 6 files" in captured.out
    assert "Skipped (no match):  1 files" in captured.out


def test_organize_images_file_not_found(tmp_path, capsys):
    """Tests logging when the event CSV is missing."""
    organize_images_into_flood_events(
        str(tmp_path), "non_existent.csv", "dest", "sub"
    )
    captured = capsys.readouterr()
    assert "ERROR: CSV file not found" in captured.err


def test_organize_images_folder_not_found(tmp_path, capsys):
    """Tests logging when the image folder is missing."""
    csv = tmp_path / "dummy.csv"

    # FIX 4: Create a dummy CSV with the *correct headers* but no rows
    # This ensures the function gets past the CSV-reading step.
    pd.DataFrame(
        columns=["sensor_ID", "start_time_UTC", "end_time_UTC"]
    ).to_csv(csv, index=False)

    organize_images_into_flood_events(
        "non_existent_img_folder", str(csv), "dest", "sub"
    )

    captured = capsys.readouterr()
    # This is the error we *actually* want to test for
    assert "ERROR: Image folder not found" in captured.err
