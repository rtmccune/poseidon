"""Tests for ImageHandler and file_organizer on synthetic folders."""

import random
from datetime import datetime, timezone

import pandas as pd
import pytest

from poseidon_utils import file_organizer as fo
from poseidon_utils.image_handler import ImageHandler, is_daylight

CB_LAT, CB_LON = 34.0435, -77.8894  # Carolina Beach camera


def touch(folder, *names):
    """
    Create empty files in a folder.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to create files in (created if needed).
    *names : str
        Filenames.

    Returns
    -------
    pathlib.Path
        The folder.
    """
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


def names_in(folder):
    """
    Return the sorted filenames in a folder.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to list.

    Returns
    -------
    list of str
        Sorted filenames (empty if the folder is missing).
    """
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def utc(text):
    """
    Parse a UTC time string.

    Parameters
    ----------
    text : str
        ``"YYYY-MM-DD HH:MM"`` in UTC.

    Returns
    -------
    datetime.datetime
        Aware UTC datetime.
    """
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


# ----- Daylight filtering ----------------------------------------------------


@pytest.mark.parametrize("time, expected", [
    ("2024-06-15 16:00", True),   # noon EDT
    ("2024-06-15 23:30", True),   # 7:30 pm EDT, sun still up
    ("2024-06-16 02:00", False),  # 10 pm EDT
    ("2024-01-15 17:00", True),   # noon EST
    ("2024-01-15 23:30", False),  # 6:30 pm EST, after winter sunset
])
def test_daylight_uses_the_local_date(time, expected):
    """Summer evenings are daylight (regression: UTC-date sunsets)."""
    pytest.importorskip("astral")
    assert is_daylight(utc(time), CB_LAT, CB_LON) is expected


def test_sun_position_copy_keeps_only_daylight_images(tmp_path):
    """Folder filtering copies daylight images and skips the rest."""
    pytest.importorskip("astral")
    source = touch(tmp_path / "all", "CAM_CB_03_20240615160000.jpg",
                   "CAM_CB_03_20240615233000.jpg",
                   "CAM_CB_03_20240616020000.jpg", "notes.txt")
    copied = ImageHandler().copy_images_using_sun_position(
        source, tmp_path / "day", CB_LAT, CB_LON)
    assert copied == 2
    assert names_in(tmp_path / "day") == ["CAM_CB_03_20240615160000.jpg",
                                          "CAM_CB_03_20240615233000.jpg"]


def test_hour_window_follows_daylight_saving(tmp_path):
    """10:30 UTC is 6:30 am in July (EDT) but 5:30 am in January."""
    source = touch(tmp_path / "all", "CAM_CB_03_20240715103000.jpg",
                   "CAM_CB_03_20240115103000.jpg")
    ImageHandler().copy_images_using_hour_window(source, tmp_path / "out",
                                                 6, 19)
    assert names_in(tmp_path / "out") == ["CAM_CB_03_20240715103000.jpg"]


# ----- Archive access --------------------------------------------------------


@pytest.fixture
def archive(tmp_path):
    """
    Fake SDFP archive with two days of CAM_CB_03 images.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    pathlib.Path
        Archive root.
    """
    root = tmp_path / "archive"
    day1 = root / "2024 Archive" / "CAM_CB_03" / "2024-03-01"
    day2 = root / "2024 Archive" / "CAM_CB_03" / "2024-03-02"
    touch(day1, *(f"CAM_CB_03_20240301{h:02d}0000.jpg" for h in range(24)))
    touch(day2, *(f"CAM_CB_03_20240302{h:02d}0000.jpg" for h in range(24)))
    return root


def test_event_pull_spans_midnight_applies_buffer_and_dedupes(archive,
                                                              tmp_path):
    """Overlapping, multi-day events copy each image exactly once."""
    csv = tmp_path / "abbr.csv"
    pd.DataFrame({
        "sensor_ID": ["CB_03", "CB_03"],
        "start_time_UTC": ["2024-03-01 22:00:00+00:00",
                           "2024-03-01 23:00:00+00:00"],
        "end_time_UTC": ["2024-03-02 01:00:00+00:00",
                         "2024-03-02 01:30:00+00:00"],
    }).to_csv(csv, index=False)

    copied = ImageHandler(archive).pull_flood_event_images(
        tmp_path / "pulled", csv, time_buffer_hours=1)
    stamps = [name[10:20] for name in names_in(tmp_path / "pulled")]
    assert copied == 6  # buffered union is 21:00 Mar 1 to 02:30 Mar 2
    assert stamps == ["2024030121", "2024030122", "2024030123",
                      "2024030200", "2024030201", "2024030202"]


def test_random_test_set_spreads_across_days(archive, tmp_path):
    """Each day contributes a small share, so two days give two images."""
    random.seed(1)
    copied = ImageHandler(archive).create_test_image_set(
        tmp_path / "sample", "CAM_CB_03", num_images=5, start_year=2024,
        end_year=2024, hour_window=(8, 17))
    sample = names_in(tmp_path / "sample")
    assert copied == 2  # fewer than requested: the archive has 2 days
    assert {name[10:18] for name in sample} == {"20240301", "20240302"}
    # 08:00-17:00 Eastern is 13:00-22:00 UTC in early March (EST)
    assert all(13 <= int(name[18:20]) <= 22 for name in sample)


def test_archive_operations_need_a_drive(tmp_path):
    """Archive methods explain that sdfp_image_drive is required."""
    with pytest.raises(ValueError, match="sdfp_image_drive"):
        ImageHandler()._list_files_in_archive_date_dir(
            datetime(2024, 1, 1).date(), "CAM_CB_03")


def test_unlabeled_images_are_those_without_matching_labels(tmp_path):
    """Images are matched to labels by camera and timestamp."""
    images = touch(tmp_path / "images", "CAM_CB_03_20240101120000.jpg",
                   "CAM_CB_03_20240101120600.jpg", "unnamed.jpg")
    labels = touch(tmp_path / "labels",
                   "CAM_CB_03_20240101120000_predseg_labels.png")
    ImageHandler().generate_unlabeled_images_folder(images, labels,
                                                    tmp_path / "todo")
    assert names_in(tmp_path / "todo") == ["CAM_CB_03_20240101120600.jpg"]


def test_time_filters_reject_naive_bounds(tmp_path):
    """Naive bounds keep nothing instead of raising mid-filter."""
    images = touch(tmp_path / "images", "CAM_CB_03_20240101120000.jpg")
    assert ImageHandler()._filter_files_by_start_and_end_time(
        [str(images / "CAM_CB_03_20240101120000.jpg")],
        datetime(2024, 1, 1), datetime(2024, 1, 2)) == []


# ----- file_organizer --------------------------------------------------------


@pytest.fixture
def abbr_csv(tmp_path):
    """
    Abbreviated CSV with two events (one per sensor).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's per-test temporary directory.

    Returns
    -------
    pathlib.Path
        CSV path.
    """
    path = tmp_path / "abbr.csv"
    pd.DataFrame({
        "sensor_ID": ["CB_03", "DE_01"],
        "start_time_UTC": ["2024-03-01 12:00:00+00:00",
                           "2024-07-15 10:30:00+00:00"],
        "end_time_UTC": ["2024-03-01 14:00:00+00:00",
                         "2024-07-15 11:00:00+00:00"],
    }).to_csv(path, index=False)
    return path


def test_event_windows_are_padded_and_named(abbr_csv):
    """Folder names encode the padded UTC window."""
    windows = fo.event_windows(pd.read_csv(abbr_csv), padding_hours=3)
    assert windows["folder_name"].tolist() == [
        "CB_03_20240301090000_20240301170000",
        "DE_01_20240715073000_20240715140000"]
    assert windows["camera_ID"].tolist() == ["CAM_CB_03", "CAM_DE_01"]


def test_event_windows_handle_empty_and_invalid_rows():
    """No rows or bad timestamps never crash (pandas 3 regression)."""
    empty = pd.DataFrame(columns=["sensor_ID", "start_time_UTC",
                                  "end_time_UTC"])
    assert fo.event_windows(empty).empty
    bad = pd.DataFrame({"sensor_ID": ["CB_03"], "start_time_UTC": ["junk"],
                        "end_time_UTC": ["2024-01-01 00:00+00:00"]})
    assert fo.event_windows(bad).empty


def test_eastern_hour_filter_is_daylight_saving_aware(abbr_csv, tmp_path):
    """Starts are filtered on local clock time."""
    out = tmp_path / "filtered.csv"
    kept = fo.filter_abbr_flood_csv_by_eastern_time(abbr_csv, out, 7, 19)
    # 12:00 UTC in March is 08:00 EDT (kept); 10:30 UTC in July is
    # 06:30 EDT (dropped by min_hour=7)
    assert kept["sensor_ID"].tolist() == ["CB_03"]
    assert pd.read_csv(out)["sensor_ID"].tolist() == ["CB_03"]


def test_event_folders_get_their_records_and_images(abbr_csv, tmp_path):
    """Records and images land in the padded event's folder only."""
    records = tmp_path / "flood_events.csv"
    pd.DataFrame({
        "sensor_ID": ["CB_03", "CB_03", "CB_03", "DE_01"],
        "time_UTC": ["2024-03-01 08:00+00:00", "2024-03-01 10:00+00:00",
                     "2024-03-01 16:30+00:00", "2024-07-15 09:00+00:00"],
        "road_water_level": [0.0, 0.1, 0.2, 0.3],
    }).to_csv(records, index=False)
    out = tmp_path / "events"
    assert fo.create_flood_csvs_and_subfolders(abbr_csv, records, out) == 2

    cb = out / "CB_03_20240301090000_20240301170000"
    assert len(pd.read_csv(cb / f"{cb.name}.csv")) == 2  # 08:00 is outside

    images = touch(tmp_path / "images", "CAM_CB_03_20240301100000.jpg",
                   "CAM_CB_03_20240301180000.jpg",
                   "CAM_DE_01_20240715100000.jpg", "readme.txt")
    assert fo.organize_images_into_flood_events(images, abbr_csv, out,
                                                "orig_images") == 2
    assert names_in(cb / "orig_images") == ["CAM_CB_03_20240301100000.jpg"]
    # Re-running copies nothing new
    assert fo.organize_images_into_flood_events(images, abbr_csv, out,
                                                "orig_images") == 0


def test_folders_without_images_are_pruned(tmp_path):
    """Only event folders lacking the image subfolder are removed."""
    touch(tmp_path / "events" / "keep" / "orig_images", "a.jpg")
    touch(tmp_path / "events" / "drop", "drop.csv")
    removed = fo.prune_empty_event_folders(tmp_path / "events",
                                           "orig_images")
    assert removed == ["drop"]
    assert names_in(tmp_path / "events") == ["keep"]


def test_job_lists_split_files_evenly(tmp_path):
    """Hidden files are skipped; lists are numbered from 1."""
    folder = touch(tmp_path / "imgs", *(f"img_{i:02d}.jpg" for i in range(7)),
                   ".hidden")
    lists = fo.prepare_job_lists(folder, num_jobs=3)
    assert [p.name for p in lists] == ["file_list_1.txt", "file_list_2.txt",
                                       "file_list_3.txt"]
    sizes = [len(p.read_text().split()) for p in lists]
    assert sizes == [3, 2, 2]
    assert lists[0].parent == tmp_path / "job_file_lists"
    with pytest.raises(NotADirectoryError):
        fo.prepare_job_lists(tmp_path / "missing", 2)


def test_daylight_coverage_splits_event_and_buffer_images(tmp_path):
    """An event across sunset is counted by daylight, inside vs. buffer."""
    pytest.importorskip("astral")
    # Hourly images 18:00 Jan 15 to 06:00 Jan 16 UTC, plus two at night
    # on Jan 17. Carolina Beach sunset on Jan 15 is about 22:25 UTC.
    hours = pd.date_range("2024-01-15 18:00", "2024-01-16 06:00", freq="h")
    night = pd.to_datetime(["2024-01-17 02:00", "2024-01-17 03:00"])
    images = touch(tmp_path / "all", *(
        f"CAM_CB_03_{t:%Y%m%d%H%M%S}.jpg" for t in hours.append(night)))
    csv = tmp_path / "abbr.csv"
    pd.DataFrame({
        "flood_event": [1, 2, 3],
        "sensor_ID": ["CB_03", "CB_03", "CB_02"],
        # 6 h event across sunset; night-only event; event with no camera
        "start_time_UTC": ["2024-01-15 21:00:00+00:00",
                           "2024-01-17 02:00:00+00:00",
                           "2024-01-15 21:00:00+00:00"],
        "end_time_UTC": ["2024-01-16 03:00:00+00:00",
                         "2024-01-17 03:00:00+00:00",
                         "2024-01-16 03:00:00+00:00"],
        "duration_(hours)": [6.0, 1.0, 6.0],
    }).to_csv(csv, index=False)
    original = csv.read_text()

    out = tmp_path / "abbr_daylight.csv"
    fo.summarize_event_daylight_coverage(csv, images, out, CB_LAT, CB_LON,
                                         padding_hours=3)
    cov = pd.read_csv(out)

    sunset_event = cov.iloc[0]
    assert sunset_event["has_daylight_images"]
    assert sunset_event["has_event_daylight_images"]
    # Inside the event: 21:00-03:00 is 7 images, daylight only 21 and 22
    assert (sunset_event["event_images_total"],
            sunset_event["event_images_daylight"],
            sunset_event["event_images_night"]) == (7, 2, 5)
    # Buffers: 18-20 UTC (daylight) before, 04-06 UTC (night) after
    assert (sunset_event["buffer_images_total"],
            sunset_event["buffer_images_daylight"],
            sunset_event["buffer_images_night"]) == (6, 3, 3)
    assert sunset_event["event_daylight_image_fraction"] == round(2 / 7, 3)
    assert 1.3 < sunset_event["event_daylight_hours"] < 1.5
    assert sunset_event["event_daylight_start_UTC"].startswith(
        "2024-01-15 21:00")
    assert sunset_event["event_daylight_end_UTC"].startswith(
        "2024-01-15 22:00")
    assert sunset_event["first_daylight_image_UTC"].startswith(
        "2024-01-15 18:00")

    night_event, no_camera = cov.iloc[1], cov.iloc[2]
    assert night_event["has_images"] and not night_event["has_daylight_images"]
    assert night_event["event_images_night"] == 2
    assert night_event["event_daylight_hours"] == 0
    assert not no_camera["has_images"]
    assert no_camera["event_images_total"] == 0

    # The original sheet is untouched; rerunning on the output is stable
    assert csv.read_text() == original
    again = fo.summarize_event_daylight_coverage(out, images, out, CB_LAT,
                                                 CB_LON, padding_hours=3)
    assert list(again.columns) == list(cov.columns)
