import concurrent.futures
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from unittest.mock import MagicMock, call, patch

# --- Module Imports ---
# Assumes your class is in a file named `image_handler.py`
# If it's named something else, change the import.
from poseidon_utils.image_handler import ImageHandler, _log

# --- Fixtures ---


@pytest.fixture
def mock_log(mocker):
    """Mocks the _log function to prevent stdout spam and allow assertions."""
    # Patch the _log function in the module where it is DEFINED.
    return mocker.patch("poseidon_utils.image_handler._log")


@pytest.fixture
def handler_setup(tmp_path):
    """
    Provides a core setup for most tests, including a handler instance,
    a temporary drive root, and a temporary destination root.
    """
    drive_root = tmp_path / "image_drive"
    dest_root = tmp_path / "destination"
    drive_root.mkdir()
    dest_root.mkdir()

    handler = ImageHandler(str(drive_root))
    return handler, drive_root, dest_root


def create_fake_archive_file(drive_root, camera_id, date_obj, time_str_utc):
    """Helper to create a realistic-looking fake image file."""
    # time_str_utc should be like "20231027143000"
    year = date_obj.year
    date_str = date_obj.strftime("%Y-%m-%d")

    file_dir = drive_root / f"{year} Archive" / camera_id / date_str
    file_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{camera_id}_{time_str_utc}.jpg"
    file_path = file_dir / file_name
    file_path.touch()  # Create an empty file
    return file_path


# --- Test Classes ---


class TestStaticMethods:
    """Tests all @staticmethod methods that have no 'self' dependency."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("CAM_AB_01_20231027143000.jpg", "20231027143000"),
            ("prefix_CAM_AB_01_20231027143000_suffix.png", "20231027143000"),
            ("no_timestamp.jpg", None),
            ("CAM_AB_01_20231027.jpg", None),  # Incomplete timestamp
            ("img_12345.jpg", None),
        ],
    )
    def test_extract_timestamp(self, filename, expected):
        assert ImageHandler._extract_timestamp(filename) == expected

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("CAM_AB_01_20231027143000.jpg", "CAM_AB_01"),
            ("prefix_CAM_XY_99_20231027143000_suffix.png", "CAM_XY_99"),
            ("no_sensor.jpg", None),
            ("CAM_AB_1.jpg", None),  # Incomplete sensor ID
            ("CAM_12_AB.jpg", None),
        ],
    )
    def test_extract_sensor_name(self, filename, expected):
        assert ImageHandler._extract_sensor_name(filename) == expected

    @pytest.mark.parametrize(
        "filename, expected_key, log_failure",
        [
            (
                "CAM_AB_01_20231027143000.jpg",
                ("CAM_AB_01", "20231027143000"),
                True,
            ),
            ("no_sensor_20231027143000.jpg", None, True),
            ("CAM_AB_01_no_timestamp.jpg", None, True),
            ("bad_file.jpg", None, False),  # Test no-logging path
        ],
    )
    def test_get_image_key(self, mock_log, filename, expected_key, log_failure):
        key = ImageHandler._get_image_key(filename, log_failure=log_failure)
        assert key == expected_key

        # Check if logging occurred (or didn't) as expected
        if expected_key is None and log_failure:
            mock_log.assert_called_with(
                f"Skipping file: Could not parse key from '{filename}'.",
                level="info",
            )
        else:
            # Assert it was NOT called with this message
            for call_item in mock_log.call_args_list:
                assert (
                    call_item[0][0]
                    != f"Skipping file: Could not parse key from '{filename}'."
                )

    def test_generate_date_list(self, mock_log):
        start_year, end_year = 2023, 2024
        dates = ImageHandler._generate_date_list(start_year, end_year)

        assert len(dates) == 366 + 365  # 2024 is a leap year
        assert dates[0] == datetime(2023, 1, 1).date()
        assert dates[-1] == datetime(2024, 12, 31).date()
        mock_log.assert_called_with(
            f"Generated {len(dates)} dates.", level="info"
        )

    def test_generate_date_list_single_year(self):
        dates = ImageHandler._generate_date_list(2023, 2023)
        assert len(dates) == 365
        assert dates[0] == datetime(2023, 1, 1).date()
        assert dates[-1] == datetime(2023, 12, 31).date()

    def test_generate_date_list_invalid_range(self, mock_log):
        with pytest.raises(
            ValueError, match="end_year cannot be before start_year"
        ):
            ImageHandler._generate_date_list(2024, 2023)


class TestFileSystemHelpers:
    """Tests methods that interact directly with the filesystem."""

    def test_init(self, mock_log):
        handler = ImageHandler("/fake/drive")
        assert handler.drive == "/fake/drive"
        mock_log.assert_called_with(
            "Image Handler initialized. Image Drive: /fake/drive", level="info"
        )

    def test_list_files_in_dir(self, handler_setup, mock_log):
        handler, drive_root, _ = handler_setup

        # Create test files and a subdirectory
        (drive_root / "file1.txt").touch()
        (drive_root / "file2.jpg").touch()
        (drive_root / "subdir").mkdir()
        (drive_root / "subdir" / "file3.png").touch()

        files = handler._list_files_in_dir(drive_root)

        assert len(files) == 2
        assert str(drive_root / "file1.txt") in files
        assert str(drive_root / "file2.jpg") in files
        assert (
            str(drive_root / "subdir" / "file3.png") not in files
        )  # Ignored subdir

    def test_list_files_in_dir_not_found(self, handler_setup, mock_log):
        handler, drive_root, _ = handler_setup

        missing_dir = drive_root / "missing"
        files = handler._list_files_in_dir(missing_dir)

        assert files == []
        mock_log.assert_called_with(
            f"Directory not found: {missing_dir}", level="info"
        )

    def test_list_files_in_dir_permission_error(
        self, handler_setup, mocker, mock_log
    ):
        handler, drive_root, _ = handler_setup

        # Mock Path.iterdir() to raise an OSError
        mocker.patch.object(
            Path, "iterdir", side_effect=OSError("Permission denied")
        )

        files = handler._list_files_in_dir(drive_root)
        assert files == []
        mock_log.assert_called_with(
            f"Error reading files from '{drive_root}'. Reason: Permission denied",
            level="error",
        )

    def test_list_files_in_archive_date_dir(self, handler_setup, mock_log):
        handler, drive_root, _ = handler_setup

        cam_id = "CAM_AA_01"
        date = datetime(2023, 10, 27).date()

        # Create the expected file structure
        f1 = create_fake_archive_file(
            drive_root, cam_id, date, "20231027140000"
        )
        f2 = create_fake_archive_file(
            drive_root, cam_id, date, "20231027150000"
        )

        # Create a file in a different camera/date to ensure it's not picked up
        create_fake_archive_file(
            drive_root, "CAM_BB_02", date, "20231027140000"
        )

        files = handler._list_files_in_archive_date_dir(date, cam_id)

        assert len(files) == 2
        assert str(f1) in files
        assert str(f2) in files

    def test_list_files_in_archive_date_dir_not_found(
        self, handler_setup, mock_log
    ):
        handler, _, _ = handler_setup

        date = datetime(2023, 1, 1).date()
        cam_id = "CAM_ZZ_99"

        files = handler._list_files_in_archive_date_dir(date, cam_id)

        assert files == []
        expected_dir = (
            Path(handler.drive) / "2023 Archive" / cam_id / "2023-01-01"
        )
        mock_log.assert_called_with(
            f"Directory not found (this may be expected): {expected_dir}",
            level="info",
        )

    def test_setup_destination_dir(self, handler_setup, mock_log):
        handler, _, dest_root = handler_setup

        new_dest = dest_root / "new_folder" / "sub_folder"

        assert not new_dest.exists()

        success = handler._setup_destination_dir(new_dest)

        assert success is True
        assert new_dest.exists()
        assert new_dest.is_dir()
        assert handler.copy_destination_dir == new_dest
        mock_log.assert_called_with(
            f"Ensured destination folder exists: {new_dest}", level="info"
        )

    def test_setup_destination_dir_already_exists(self, handler_setup):
        handler, _, dest_root = handler_setup

        # It already exists
        assert dest_root.exists()
        success = handler._setup_destination_dir(dest_root)
        assert success is True
        assert handler.copy_destination_dir == dest_root

    def test_setup_destination_dir_os_error(
        self, handler_setup, mocker, mock_log
    ):
        handler, _, dest_root = handler_setup

        # Mock os.makedirs to raise an error
        mocker.patch("os.makedirs", side_effect=OSError("Permission denied"))

        bad_dest = dest_root / "cant_make_this"
        success = handler._setup_destination_dir(bad_dest)

        assert success is False
        assert not hasattr(handler, "copy_destination_dir")
        mock_log.assert_called_with(
            f"FATAL: Could not create destination folder '{bad_dest}'. Reason: Permission denied",
            level="error",
        )


class TestFilterHelpers:
    """Tests the various file filtering methods."""

    # Define some reusable datetimes
    tz_utc = timezone.utc
    tz_est = ZoneInfo("America/New_York")

    # 10:00 AM EST on Oct 27, 2023 -> 14:00 UTC
    dt_1400_utc = datetime(2023, 10, 27, 14, 0, 0, tzinfo=tz_utc)
    # 2:00 PM EST on Oct 27, 2023 -> 18:00 UTC
    dt_1800_utc = datetime(2023, 10, 27, 18, 0, 0, tzinfo=tz_utc)
    # 8:00 PM EST on Oct 27, 2023 -> 00:00 UTC on Oct 28
    dt_0000_utc_next = datetime(2023, 10, 28, 0, 0, 0, tzinfo=tz_utc)

    # Fake file list corresponding to the datetimes
    file_list = [
        "/fake/CAM_AA_01_20231027140000.jpg",  # 14:00 UTC
        "/fake/CAM_AA_01_20231027175959.jpg",  # 17:59 UTC
        "/fake/CAM_AA_01_20231027180000.jpg",  # 18:00 UTC
        "/fake/CAM_AA_01_20231028000000.jpg",  # 00:00 UTC next day
        "/fake/bad_timestamp.jpg",  # Will be skipped
    ]

    def test_filter_files_by_start_and_end_time(self, mock_log):
        handler = ImageHandler(None)  # No drive needed

        start = self.dt_1400_utc
        end = self.dt_1800_utc

        filtered = handler._filter_files_by_start_and_end_time(
            self.file_list, start, end
        )

        assert len(filtered) == 3
        assert self.file_list[0] in filtered  # 14:00 (inclusive start)
        assert self.file_list[1] in filtered  # 17:59
        assert self.file_list[2] in filtered  # 18:00 (inclusive end)
        assert self.file_list[3] not in filtered

        # Check that the bad file was logged
        mock_log.assert_called_with(
            "Skipping file: Could not parse timestamp from 'bad_timestamp.jpg'.",
            level="info",
        )

    def test_filter_files_by_start_and_end_time_naive_datetimes(self, mock_log):
        handler = ImageHandler(None)
        start_naive = datetime(2023, 10, 27, 14, 0, 0)
        end_aware = self.dt_1800_utc

        filtered = handler._filter_files_by_start_and_end_time(
            self.file_list, start_naive, end_aware
        )

        assert filtered == []
        mock_log.assert_called_with(
            "Cannot filter files: start_time and end_time must be timezone-aware.",
            level="error",
        )

    def test_filter_files_eastern_time_window(self, mock_log):
        handler = ImageHandler(None)

        # Window: 10:00 AM EST to 1:59 PM EST (hour 13)
        # This corresponds to 14:00 UTC to 17:59 UTC during EDT
        start_hour_east = 10
        end_hour_east = 13

        filtered = handler._filter_files_eastern_time_window(
            self.file_list, start_hour_east, end_hour_east
        )

        # 20231027140000.jpg -> 10:00 EST (IN)
        # 20231027175959.jpg -> 13:59 EST (IN)
        # 20231027180000.jpg -> 14:00 EST (OUT)
        # 20231028000000.jpg -> 20:00 EST (Oct 27) (OUT)

        assert len(filtered) == 2
        assert self.file_list[0] in filtered
        assert self.file_list[1] in filtered
        assert self.file_list[2] not in filtered
        assert self.file_list[3] not in filtered

    def test_filter_files_eastern_time_window_overnight(self, mock_log):
        handler = ImageHandler(None)

        # Window: 8:00 PM EST (20) to 10:00 AM EST (10)
        # This is an invalid assumption for the current logic,
        # it expects start <= end.
        # Let's test a valid window that crosses midnight UTC
        # e.g., 8:00 PM EST (20) to 11:00 PM EST (23)
        # This is 00:00 UTC to 03:00 UTC *the next day*

        start_hour_east = 20
        end_hour_east = 23

        filtered = handler._filter_files_eastern_time_window(
            self.file_list, start_hour_east, end_hour_east
        )

        # 20231027140000.jpg -> 10:00 EST (OUT)
        # 20231027175959.jpg -> 13:59 EST (OUT)
        # 20231027180000.jpg -> 14:00 EST (OUT)
        # 20231028000000.jpg -> 20:00 EST (Oct 27) (IN)

        assert len(filtered) == 1
        assert self.file_list[3] in filtered

    def test_filter_files_by_condition_parse_error(self, mock_log):
        handler = ImageHandler(None)
        file_list = [
            "/fake/CAM_AA_01_20231027140000.jpg",
            "/fake/CAM_AA_01_not_a_date.jpg",
            "/fake/CAM_AA_01_20231027150000.jpg",
        ]

        # Condition that just returns True
        filtered = handler._filter_files_by_condition(
            file_list, lambda dt: True
        )

        assert len(filtered) == 2
        assert file_list[0] in filtered
        assert file_list[2] in filtered
        mock_log.assert_called_with(
            "Skipping file: Could not parse timestamp from 'CAM_AA_01_not_a_date.jpg'.",
            level="info",
        )


class TestCopyHelpers:
    """Tests _copy_file and _parallel_copy_files."""

    def test_copy_file(self, handler_setup):
        handler, drive_root, dest_root = handler_setup

        # Set the destination dir on the handler
        handler.copy_destination_dir = str(dest_root)

        src_file = drive_root / "test_file.txt"
        src_file.write_text("hello")

        dest_file = dest_root / "test_file.txt"

        assert not dest_file.exists()

        success = handler._copy_file(str(src_file))

        assert success is True
        assert dest_file.exists()
        assert dest_file.read_text() == "hello"

    def test_copy_file_no_dest_dir_set(self, handler_setup, mock_log):
        handler, drive_root, _ = handler_setup
        # Note: We do NOT set handler.copy_destination_dir

        src_file = drive_root / "test_file.txt"
        src_file.touch()

        success = handler._copy_file(str(src_file))

        assert success is False
        mock_log.assert_called_with(
            "Cannot copy: self.copy_destination_dir is not set.", level="error"
        )

    @pytest.mark.parametrize(
        "error_type, error_msg",
        [
            (FileNotFoundError, "File not found"),
            (shutil.Error, "Disk full"),
            (OSError, "Permission denied"),
        ],
    )
    def test_copy_file_shutil_errors(
        self, handler_setup, mocker, mock_log, error_type, error_msg
    ):
        handler, drive_root, dest_root = handler_setup
        handler.copy_destination_dir = str(dest_root)

        src_file = drive_root / "test_file.txt"
        src_file.touch()

        # Mock shutil.copy to raise the error
        mocker.patch("shutil.copy", side_effect=error_type(error_msg))

        success = handler._copy_file(str(src_file))

        assert success is False

        # Check for the correct log message
        if error_type == FileNotFoundError:
            log_msg = f"File not found, could not copy: {src_file}"
        elif error_type == shutil.Error:
            log_msg = f"shutil error copying {src_file}. Reason: {error_msg}"
        else:  # OSError
            log_msg = f"OS error copying {src_file}. Reason: {error_msg}"

        mock_log.assert_called_with(log_msg, level="error")

    def test_parallel_copy_files(self, handler_setup, mocker, mock_log):
        handler, _, _ = handler_setup

        tasks = ["/file/a", "/file/b", "/file/c"]

        # Mock the ThreadPoolExecutor
        mock_executor = MagicMock()
        mock_executor_context = MagicMock()
        mock_executor_context.__enter__.return_value = mock_executor
        mock_executor_context.__exit__.return_value = False
        mocker.patch(
            "concurrent.futures.ThreadPoolExecutor",
            return_value=mock_executor_context,
        )

        # Define the results of _copy_file
        # (True for success, False for failure)
        mock_results = [True, False, True]
        mock_executor.map.return_value = mock_results

        handler._parallel_copy_files(
            tasks, max_workers=8, task_name="Test Copy"
        )

        # Check that ThreadPoolExecutor was called with correct max_workers
        concurrent.futures.ThreadPoolExecutor.assert_called_with(max_workers=8)

        # Check that executor.map was called with the correct function and tasks
        mock_executor.map.assert_called_with(handler._copy_file, tasks)

        # Check the final log message
        mock_log.assert_called_with(
            "Test Copy complete. Copied: 2, Failed: 1", level="info"
        )

        # Check progress logging (first, last)
        mock_log.assert_any_call(
            "  Copy progress: 1/3 files processed.", level="info"
        )
        mock_log.assert_any_call(
            "  Copy progress: 3/3 files processed.", level="info"
        )

    def test_parallel_copy_files_no_tasks(
        self, handler_setup, mocker, mock_log
    ):
        handler, _, _ = handler_setup
        mock_executor = mocker.patch("concurrent.futures.ThreadPoolExecutor")

        handler._parallel_copy_files([], max_workers=4, task_name="Empty Copy")

        # Ensure no executor was created
        mock_executor.assert_not_called()

        # Check log
        mock_log.assert_called_with(
            "No files found to copy for task: 'Empty Copy'.", level="info"
        )


class TestDataHelpers:
    """Tests for _format... and _get_random_images."""

    def test_format_abbreviated_events_for_pull(
        self, tmp_path, mocker, mock_log
    ):
        handler = ImageHandler(None)

        csv_path = tmp_path / "events.csv"
        csv_content = (
            "sensor_ID,start_time_UTC,end_time_UTC,other_col\n"
            "AA_01,2023-10-27 14:00:00+00:00,2023-10-27 18:00:00+00:00,foo\n"
            "BB_02,2023-11-01 05:00:00+00:00,2023-11-01 06:00:00+00:00,bar"
        )
        csv_path.write_text(csv_content)

        # Mock pd.read_csv to read from our string
        mock_df = pd.read_csv(csv_path)
        mocker.patch("pandas.read_csv", return_value=mock_df)

        df = handler._format_abbreviated_events_for_pull(csv_path)

        # Check that read_csv was called correctly
        pd.read_csv.assert_called_with(
            csv_path, usecols=["sensor_ID", "start_time_UTC", "end_time_UTC"]
        )

        # Check DataFrame structure and content
        assert len(df) == 2
        assert list(df.columns) == [
            "camera_ID",
            "start_time_UTC",
            "end_time_UTC",
        ]
        assert df.iloc[0]["camera_ID"] == "CAM_AA_01"
        assert df.iloc[1]["camera_ID"] == "CAM_BB_02"

        # Check datetime conversion
        expected_start = datetime(2023, 10, 27, 14, 0, 0, tzinfo=timezone.utc)
        assert df.iloc[0]["start_time_UTC"] == expected_start
        assert df.iloc[0]["start_time_UTC"].tzinfo is not None

    def test_format_abbreviated_events_file_not_found(self, mocker, mock_log):
        handler = ImageHandler(None)

        mocker.patch(
            "pandas.read_csv", side_effect=FileNotFoundError("No such file")
        )

        with pytest.raises(FileNotFoundError):
            handler._format_abbreviated_events_for_pull("bad.csv")

    def test_get_random_images(self, handler_setup, mocker, mock_log):
        handler, drive_root, _ = handler_setup

        cam_id = "CAM_CC_03"
        start_year, end_year = 2023, 2023
        num_images = 5

        # --- Setup Fake Files ---
        date1 = datetime(2023, 1, 15).date()
        date2 = datetime(2023, 6, 20).date()

        # Create 10 files for date1
        files_date1 = [
            create_fake_archive_file(
                drive_root, cam_id, date1, f"2023011510{i:02d}00"
            )
            for i in range(10)
        ]

        # Create 10 files for date2
        files_date2 = [
            create_fake_archive_file(
                drive_root, cam_id, date2, f"2023062014{i:02d}00"
            )
            for i in range(10)
        ]

        # --- Mock external calls ---

        # Mock _generate_date_list to return only our two dates + a blank one
        blank_date = datetime(2023, 3, 3).date()
        mock_dates = [date1, blank_date, date2]
        mocker.patch.object(
            handler, "_generate_date_list", return_value=mock_dates
        )

        # Mock random.shuffle to control the order
        mocker.patch("random.shuffle", side_effect=lambda x: x.reverse())
        # The list will be processed in order: [date2, blank_date, date1]

        # Mock the internal _list_files_in_dir
        def mock_list_files(dir_path):
            if str(date1) in str(dir_path):
                return [str(p) for p in files_date1]
            if str(date2) in str(dir_path):
                return [str(p) for p in files_date2]
            return []  # For blank_date

        mocker.patch.object(
            handler, "_list_files_in_dir", side_effect=mock_list_files
        )

        # Mock random.uniform to return a fixed percentage (e.g., 0.1 -> 10%)
        # 10% of 5 images = 0.5, which max(1, int(0.5)) = 1.
        # Let's mock 0.5 -> 50% -> int(2.5) = 2.
        mocker.patch("random.uniform", return_value=0.5)
        # num_to_select will be max(1, int(5 * 0.5)) = 2

        # --- Run ---
        selected = handler._get_random_images(
            cam_id, start_year, end_year, num_images
        )

        # --- Assert ---
        # It needs 5 images.
        # 1. Pops date2. _list_files finds 10. num_to_select = 2.
        #    Takes 2. selected_images = 2.
        # 2. Pops blank_date. _list_files finds 0. Skips.
        # 3. Pops date1. _list_files finds 10. num_to_select = 2.
        #    Takes 2. selected_images = 4.
        # 4. Loop ends (date_list is empty).

        # Whoops, my logic was slightly off. The *available* logic will
        # take over. Let's re-trace.

        # 1. Pops date2. Finds 10. num_to_select = 2. needed = 5. available = 10.
        #    num_to_add = min(2, 5, 10) = 2.
        #    Adds 2 files. `selected_images` = 2.
        # 2. Pops blank_date. Finds 0. Skips.
        # 3. Pops date1. Finds 10. num_to_select = 2. needed = 3. available = 10.
        #    num_to_add = min(2, 3, 10) = 2.
        #    Adds 2 files. `selected_images` = 4.
        # 4. Loop ends.

        assert len(selected) == 4

        # Check that it selected from both dates
        assert any(str(date1) in str(f) for f in selected)
        assert any(str(date2) in str(f) for f in selected)

        # Check log for finding fewer images
        mock_log.assert_called_with(
            f"Selection complete. Warning: Only found {len(selected)} "
            f"of {num_images} requested images.",
            level="info",
        )

    def test_get_random_images_with_hour_window(self, handler_setup, mocker):
        handler, drive_root, _ = handler_setup

        cam_id = "CAM_DD_04"
        date1 = datetime(2023, 1, 1).date()

        # 09:00 EST -> 14:00 UTC
        f_in_window = create_fake_archive_file(
            drive_root, cam_id, date1, "20230101140000"
        )
        # 05:00 EST -> 10:00 UTC
        f_out_window = create_fake_archive_file(
            drive_root, cam_id, date1, "20230101100000"
        )

        mocker.patch.object(
            handler, "_generate_date_list", return_value=[date1]
        )
        mocker.patch("random.shuffle", lambda x: x)
        mocker.patch("random.uniform", return_value=1.0)  # Take all

        # --- Run ---
        selected = handler._get_random_images(
            cam_id,
            2023,
            2023,
            num_images=5,
            hour_window=(8, 12),  # 8am-12pm EST
        )

        # --- Assert ---
        # Only the 09:00 EST (14:00 UTC) file should be selected
        assert len(selected) == 1
        assert str(f_in_window) in selected
        assert str(f_out_window) not in selected


class TestPublicMethods:
    """
    Integration tests for the main public methods.
    These tests mock the internal helper methods to test
    the orchestration logic, error handling, and flow.
    """

    @pytest.fixture
    def mock_handler(self, handler_setup, mocker):
        """Fixture to create a handler and mock all its internal methods."""
        handler, _, dest_root = handler_setup

        mocker.patch.object(
            handler, "_setup_destination_dir", return_value=True
        )
        mocker.patch.object(handler, "_format_abbreviated_events_for_pull")
        mocker.patch.object(handler, "_list_files_in_archive_date_dir")
        mocker.patch.object(handler, "_filter_files_by_start_and_end_time")
        mocker.patch.object(handler, "_parallel_copy_files")
        mocker.patch.object(handler, "_list_files_in_dir")
        mocker.patch.object(handler, "_filter_files_eastern_time_window")
        mocker.patch.object(handler, "_get_image_key")
        mocker.patch.object(handler, "_get_random_images")

        return handler, dest_root

    def test_pull_flood_event_images_happy_path(
        self, mock_handler, mock_log, tmp_path
    ):
        handler, dest_root = mock_handler

        # --- Mock Data ---
        csv_path = tmp_path / "events.csv"

        # A single, one-day event
        event_start = datetime(2023, 10, 27, 14, 0, 0, tzinfo=timezone.utc)
        event_end = datetime(2023, 10, 27, 18, 0, 0, tzinfo=timezone.utc)

        mock_df = pd.DataFrame(
            [
                {
                    "camera_ID": "CAM_AA_01",
                    "start_time_UTC": event_start,
                    "end_time_UTC": event_end,
                }
            ]
        )

        # --- Mock Method Behavior ---
        handler._format_abbreviated_events_for_pull.return_value = mock_df

        # Mock file listing
        archive_files = ["/drive/file1.jpg", "/drive/file2.jpg"]
        handler._list_files_in_archive_date_dir.return_value = archive_files

        # Mock filtering
        filtered_files = ["/drive/file1.jpg"]
        handler._filter_files_by_start_and_end_time.return_value = (
            filtered_files
        )

        # --- Run ---
        handler.pull_flood_event_images(dest_root, csv_path)

        # --- Assert ---
        handler._setup_destination_dir.assert_called_with(dest_root)
        handler._format_abbreviated_events_for_pull.assert_called_with(csv_path)

        # Assert list_files was called with correct date and camera
        handler._list_files_in_archive_date_dir.assert_called_with(
            event_start.date(), "CAM_AA_01"
        )

        # Assert filter was called with correct files and times
        handler._filter_files_by_start_and_end_time.assert_called_with(
            archive_files, event_start, event_end
        )

        # Assert parallel_copy was called with the final, filtered list
        handler._parallel_copy_files.assert_called_with(
            filtered_files, None, "Image pull"
        )

    def test_pull_flood_event_images_multi_day_event(
        self, mock_handler, mock_log
    ):
        handler, dest_root = mock_handler

        # A multi-day event
        event_start = datetime(2023, 10, 27, 22, 0, 0, tzinfo=timezone.utc)
        event_end = datetime(2023, 10, 28, 2, 0, 0, tzinfo=timezone.utc)

        mock_df = pd.DataFrame(
            [
                {
                    "camera_ID": "CAM_AA_01",
                    "start_time_UTC": event_start,
                    "end_time_UTC": event_end,
                }
            ]
        )

        handler._format_abbreviated_events_for_pull.return_value = mock_df

        # Mock file listing and filtering for both days
        files_day1 = ["/drive/day1/file1.jpg", "/drive/day1/file2.jpg"]
        files_day2 = ["/drive/day2/file3.jpg", "/drive/day2/file4.jpg"]

        def list_side_effect(date, cam_id):
            if date == event_start.date():
                return files_day1
            if date == event_end.date():
                return files_day2
            return []

        handler._list_files_in_archive_date_dir.side_effect = list_side_effect

        # Mock filtering to return one file from each day
        def filter_side_effect(file_list, start, end):
            if file_list == files_day1:
                return ["/drive/day1/file2.jpg"]
            if file_list == files_day2:
                return ["/drive/day2/file3.jpg"]
            return []

        handler._filter_files_by_start_and_end_time.side_effect = (
            filter_side_effect
        )

        # --- Run ---
        handler.pull_flood_event_images(dest_root, "events.csv")

        # --- Assert ---
        # Check that list_files was called for both dates
        handler._list_files_in_archive_date_dir.assert_has_calls(
            [
                call(event_start.date(), "CAM_AA_01"),
                call(event_end.date(), "CAM_AA_01"),
            ]
        )

        # Check that filter was called for both sets of files
        handler._filter_files_by_start_and_end_time.assert_has_calls(
            [
                call(files_day1, event_start, event_end),
                call(files_day2, event_start, event_end),
            ]
        )

        # Check that final copy list is correct
        expected_tasks = sorted(
            ["/drive/day1/file2.jpg", "/drive/day2/file3.jpg"]
        )
        handler._parallel_copy_files.assert_called_with(
            expected_tasks, None, "Image pull"
        )

    def test_pull_flood_event_images_deduplication(
        self, mock_handler, mock_log
    ):
        handler, dest_root = mock_handler

        # Two events that overlap
        event1 = {
            "camera_ID": "CAM_AA_01",
            "start_time_UTC": datetime(
                2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc
            ),
            "end_time_UTC": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        }
        event2 = {
            "camera_ID": "CAM_AA_01",
            "start_time_UTC": datetime(
                2023, 1, 1, 11, 0, 0, tzinfo=timezone.utc
            ),
            "end_time_UTC": datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
        }
        mock_df = pd.DataFrame([event1, event2])
        handler._format_abbreviated_events_for_pull.return_value = mock_df

        # Mock list_files to return the same list for both
        all_files = ["/file/a", "/file/b", "/file/c"]
        handler._list_files_in_archive_date_dir.return_value = all_files

        # Mock filter to return overlapping files
        def filter_side_effect(file_list, start, end):
            if start.hour == 10:  # Event 1
                return ["/file/a", "/file/b"]
            if start.hour == 11:  # Event 2
                return ["/file/b", "/file/c"]
            return []

        handler._filter_files_by_start_and_end_time.side_effect = (
            filter_side_effect
        )

        # --- Run ---
        handler.pull_flood_event_images(dest_root, "events.csv")

        # --- Assert ---
        # Check that the final list was deduplicated
        expected_tasks = sorted(["/file/a", "/file/b", "/file/c"])
        handler._parallel_copy_files.assert_called_with(
            expected_tasks, None, "Image pull"
        )
        mock_log.assert_any_call(
            "Finished event processing. Found 4 files (pre-deduplication).",
            level="info",
        )

    def test_pull_flood_event_images_setup_dir_fails(
        self, mock_handler, mock_log
    ):
        handler, dest_root = mock_handler
        handler._setup_destination_dir.return_value = False  # Simulate failure

        handler.pull_flood_event_images(dest_root, "events.csv")

        # Should exit early
        handler._format_abbreviated_events_for_pull.assert_not_called()
        handler._parallel_copy_files.assert_not_called()

    def test_pull_flood_event_images_format_events_fails(
        self, mock_handler, mock_log
    ):
        handler, dest_root = mock_handler

        handler._format_abbreviated_events_for_pull.side_effect = Exception(
            "Bad CSV"
        )

        handler.pull_flood_event_images(dest_root, "events.csv")

        mock_log.assert_called_with(
            "FATAL: Failed to load/format flood events from events.csv. Reason: Bad CSV",
            level="error",
        )
        handler._parallel_copy_files.assert_not_called()

    def test_pull_flood_event_images_attribute_error_in_loop(
        self, mock_handler, mock_log
    ):
        handler, dest_root = mock_handler

        # Create a DataFrame-like object that will raise AttributeError
        class BadRow:
            pass

        class BadDataFrame:
            def __init__(self):
                self.rows = [BadRow()]  # A row without 'start_time_UTC' etc.
                self.empty = False

            def itertuples(self, name):
                return iter(self.rows)

            def __len__(self):
                return len(self.rows)

        mock_df = BadDataFrame()
        handler._format_abbreviated_events_for_pull.return_value = mock_df

        handler.pull_flood_event_images(dest_root, "events.csv")

        # Should log the error and continue
        mock_log.assert_any_call(
            "Skipping event row: Missing expected CSV column. Error: 'BadRow' object has no attribute 'camera_ID'",
            level="error",
        )
        # Should still try to copy (with an empty list)
        handler._parallel_copy_files.assert_called_with([], None, "Image pull")

    def test_copy_images_using_hour_window(self, mock_handler, mock_log):
        handler, dest_root = mock_handler

        image_dir = "/src/images"
        all_files = ["/src/images/file1.jpg", "/src/images/file2.jpg"]
        filtered = ["/src/images/file1.jpg"]

        handler._list_files_in_dir.return_value = all_files
        handler._filter_files_eastern_time_window.return_value = filtered

        handler.copy_images_using_hour_window(
            image_dir, dest_root, 6, 19, max_workers=4
        )

        handler._setup_destination_dir.assert_called_with(dest_root)
        handler._list_files_in_dir.assert_called_with(image_dir)
        handler._filter_files_eastern_time_window.assert_called_with(
            all_files, 6, 19
        )
        handler._parallel_copy_files.assert_called_with(
            filtered, 4, "Windowed image pull"
        )

    def test_generate_unlabeled_images_folder(self, mock_handler, mock_log):
        handler, dest_root = mock_handler

        image_dir = "/all/images"
        labels_dir = "/labeled/images"

        # --- Mock Method Behavior ---
        label_files = ["/labels/CAM_AA_01_20230101120000.jpg"]
        image_files = [
            "/images/CAM_AA_01_20230101120000.jpg",  # Labeled
            "/images/CAM_AA_01_20230101130000.jpg",  # Unlabeled
            "/images/bad_file.jpg",  # Parse failure
        ]

        def list_side_effect(dir_path):
            if dir_path == labels_dir:
                return label_files
            if dir_path == image_dir:
                return image_files
            return []

        handler._list_files_in_dir.side_effect = list_side_effect

        # Mock _get_image_key
        def key_side_effect(filename, log_failure=True):
            if "20230101120000" in filename:
                return ("CAM_AA_01", "20230101120000")
            if "20230101130000" in filename:
                return ("CAM_AA_01", "20230101130000")
            return None  # For bad_file.jpg

        handler._get_image_key.side_effect = key_side_effect

        # --- Run ---
        handler.generate_unlabeled_images_folder(
            image_dir, labels_dir, dest_root, max_workers=2
        )

        # --- Assert ---
        handler._setup_destination_dir.assert_called_with(dest_root)
        handler._list_files_in_dir.assert_any_call(labels_dir)
        handler._list_files_in_dir.assert_any_call(image_dir)

        # Check _get_image_key calls
        # 1 for label file, 3 for image files
        assert handler._get_image_key.call_count == 4

        # Check copy list
        expected_tasks = ["/images/CAM_AA_01_20230101130000.jpg"]
        handler._parallel_copy_files.assert_called_with(
            expected_tasks, 2, "Unlabeled image copy"
        )

        # Check log for parse failure
        mock_log.assert_any_call(
            "Could not parse a key for 1 image files (e.g., 'bad_file.jpg').",
            level="info",
        )

    def test_create_test_image_set(self, mock_handler, mock_log):
        handler, dest_root = mock_handler

        random_files = ["/drive/rand1.jpg", "/drive/rand2.jpg"]
        handler._get_random_images.return_value = random_files

        handler.create_test_image_set(
            dest_root,
            "CAM_ZZ_99",
            10,
            2023,
            2024,
            hour_window=(6, 18),
            max_workers=8,
        )

        handler._setup_destination_dir.assert_called_with(dest_root)
        handler._get_random_images.assert_called_with(
            "CAM_ZZ_99", 2023, 2024, num_images=10, hour_window=(6, 18)
        )
        handler._parallel_copy_files.assert_called_with(
            random_files, 8, "Test set creation"
        )
