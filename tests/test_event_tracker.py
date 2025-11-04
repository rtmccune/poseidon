import pytest
from unittest.mock import MagicMock, patch, call, ANY
import pandas as pd
import numpy as np
import datetime
import pytz
import os
import requests
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Import the class from its file
# Assuming you saved your class in 'event_tracker.py'
from poseidon_utils.event_tracker import EventTracker

# --- Fixtures: Reusable setup code ---


@pytest.fixture(autouse=True)
def mock_matplotlib(mocker):
    """Auto-mock all matplotlib plotting to prevent windows from popping up."""
    mocker.patch("matplotlib.pyplot.figure")
    mocker.patch("matplotlib.pyplot.plot")
    mocker.patch("matplotlib.pyplot.ylim")
    mocker.patch("matplotlib.pyplot.xticks")
    mocker.patch("matplotlib.pyplot.yticks")
    mocker.patch("matplotlib.pyplot.grid")
    mocker.patch("matplotlib.pyplot.xlabel")
    mocker.patch("matplotlib.pyplot.ylabel")
    mocker.patch("matplotlib.pyplot.title")
    mocker.patch("matplotlib.pyplot.legend")
    mocker.patch("matplotlib.pyplot.tight_layout")
    mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")


@pytest.fixture(autouse=True)
def quiet_prints(mocker):
    """Auto-mock the built-in print function to quiet test output."""
    mocker.patch("builtins.print")


@pytest.fixture
def default_dates():
    """Provides a default date range for initializing the tracker."""
    return {
        "min_date": datetime.date(2023, 1, 1),
        "max_date": datetime.date(2023, 1, 5),
    }


@pytest.fixture
def tracker(default_dates):
    """Returns a standard, initialized EventTracker instance."""
    return EventTracker(
        authorization=("user", "pass"),
        min_date=default_dates["min_date"],
        max_date=default_dates["max_date"],
        max_workers=2,
    )


@pytest.fixture
def mock_session(mocker):
    """Mocks the requests.Session object."""
    return mocker.MagicMock(spec=requests.Session)


@pytest.fixture
def sample_api_json():
    """Sample raw JSON response from the API."""
    return [
        {
            "date": "2023-01-01T12:00:00+00:00",
            "date_surveyed": "2023-01-01T12:00:00+00:00",
            "sensor_ID": "DE_01",
            "road_water_level_adj": 0.5,
            "sensor_water_level_adj": 1.0,
        },
        {
            "date": "2023-01-01T12:05:00+00:00",
            "date_surveyed": "2023-01-01T12:05:00+00:00",
            "sensor_ID": "DE_01",
            "road_water_level_adj": -0.1,
            "sensor_water_level_adj": 0.4,
        },
    ]


@pytest.fixture
def sample_api_df(sample_api_json):
    """DataFrame equivalent of the sample_api_json."""
    df = pd.DataFrame(sample_api_json)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
    return df


@pytest.fixture
def sample_flood_data():
    """Sample DataFrame of raw data for testing CSV generation."""
    eastern = pytz.timezone("EST")
    base_time = pd.to_datetime("2023-01-01 12:00:00", utc=True)
    data = []
    # Event 1: DE_01
    data.append(
        {
            "date": base_time - datetime.timedelta(minutes=5),
            "sensor_ID": "DE_01",
            "road_water_level_adj": -0.1,
            "sensor_water_level_adj": 0.4,
        }
    )
    data.append(
        {
            "date": base_time,
            "sensor_ID": "DE_01",
            "road_water_level_adj": 0.1,
            "sensor_water_level_adj": 1.0,
        }
    )  # Start
    data.append(
        {
            "date": base_time + datetime.timedelta(minutes=5),
            "sensor_ID": "DE_01",
            "road_water_level_adj": 0.5,
            "sensor_water_level_adj": 1.5,
        }
    )  # Max
    data.append(
        {
            "date": base_time + datetime.timedelta(minutes=10),
            "sensor_ID": "DE_01",
            "road_water_level_adj": -0.1,
            "sensor_water_level_adj": 0.4,
        }
    )  # End

    # Event 2: DE_02 (overlaps with event 1)
    data.append(
        {
            "date": base_time + datetime.timedelta(minutes=5),
            "sensor_ID": "DE_02",
            "road_water_level_adj": 0.2,
            "sensor_water_level_adj": 2.0,
        }
    )  # Start
    data.append(
        {
            "date": base_time + datetime.timedelta(minutes=10),
            "sensor_ID": "DE_02",
            "road_water_level_adj": 0.3,
            "sensor_water_level_adj": 2.1,
        }
    )  # Max
    data.append(
        {
            "date": base_time + datetime.timedelta(minutes=15),
            "sensor_ID": "DE_02",
            "road_water_level_adj": -0.1,
            "sensor_water_level_adj": 1.9,
        }
    )  # End

    # Outage Data (Gap)
    data.append(
        {
            "date": base_time + datetime.timedelta(hours=5),
            "sensor_ID": "DE_01",
            "road_water_level_adj": -0.1,
            "sensor_water_level_adj": 0.4,
        }
    )

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# --- Test Cases ---


class TestEventTrackerInit:
    def test_initialization_success(self, default_dates):
        """Tests successful instantiation of the class."""
        tracker = EventTracker(
            authorization=("user", "pass"),
            min_date=default_dates["min_date"],
            max_date=default_dates["max_date"],
            max_workers=8,
            timeout=30,
        )
        assert tracker.authorization == ("user", "pass")
        assert tracker.min_date == default_dates["min_date"]
        assert tracker.max_date == default_dates["max_date"]
        assert tracker.max_workers == 8
        assert tracker.timeout == 30

    def test_init_missing_authorization(self, default_dates):
        """Tests ValueError if authorization is missing or incomplete."""
        with pytest.raises(ValueError, match="Authorization"):
            EventTracker(authorization=None, **default_dates)
        with pytest.raises(ValueError, match="Authorization"):
            EventTracker(authorization=("user", None), **default_dates)
        with pytest.raises(ValueError, match="Authorization"):
            EventTracker(authorization=(), **default_dates)

    def test_init_missing_dates(self):
        """Tests ValueError if dates are missing."""
        with pytest.raises(ValueError, match="min_date and max_date"):
            EventTracker(
                authorization=("user", "pass"),
                min_date=None,
                max_date=datetime.date(2023, 1, 1),
            )
        with pytest.raises(ValueError, match="min_date and max_date"):
            EventTracker(
                authorization=("user", "pass"),
                min_date=datetime.date(2023, 1, 1),
                max_date=None,
            )


class TestSensorAndDateLogic:

    @pytest.mark.parametrize(
        "location_name, expected_list",
        [
            ("carolina beach", ["CB_01", "CB_02", "CB_03"]),
            ("beaufort", ["BF_01"]),
            ("down east", ["DE_01", "DE_02", "DE_03", "DE_04"]),
            ("new bern", ["NB_01", "NB_02"]),
            ("north river", ["NR_01"]),
            (
                "CAROLINA BEACH",
                ["CB_01", "CB_02", "CB_03"],
            ),  # Test case-insensitivity
        ],
    )
    def test_sensor_list_generator_valid(
        self, tracker, location_name, expected_list
    ):
        """Tests all valid location names for the sensor list generator."""
        assert tracker._sensor_list_generator(location_name) == expected_list

    def test_sensor_list_generator_invalid(self, tracker):
        """Tests that an invalid location name raises a ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            tracker._sensor_list_generator("Unknown Location")

    def test_resolve_sensor_ids(self, tracker, mocker):
        """Tests resolving sensor IDs from a string or a list."""
        # Test string input (valid)
        mock_gen = mocker.patch.object(
            tracker, "_sensor_list_generator", return_value=["DE_01"]
        )
        assert tracker._resolve_sensor_ids("down east") == ["DE_01"]
        mock_gen.assert_called_with("down east")

        # Test list input
        assert tracker._resolve_sensor_ids(["MY_01", "MY_02"]) == [
            "MY_01",
            "MY_02",
        ]

        # Test string input (invalid)
        mocker.patch.object(
            tracker,
            "_sensor_list_generator",
            side_effect=ValueError("Not recognized"),
        )
        assert tracker._resolve_sensor_ids("Unknown") is None

    def test_create_date_chunks(self, tracker):
        """Tests the date chunking logic."""
        # 5 days, 2-day chunks -> [ (1,2), (3,4), (5,5) ]
        tracker.min_date = datetime.date(2023, 1, 1)
        tracker.max_date = datetime.date(2023, 1, 5)
        chunks = tracker._create_date_chunks(chunk_days=2)
        assert chunks == [
            (datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)),
            (datetime.date(2023, 1, 3), datetime.date(2023, 1, 4)),
            (datetime.date(2023, 1, 5), datetime.date(2023, 1, 5)),
        ]

    def test_create_date_chunks_single_day(self, tracker):
        """Tests chunking for a single-day range."""
        tracker.min_date = datetime.date(2023, 1, 1)
        tracker.max_date = datetime.date(2023, 1, 1)
        chunks = tracker._create_date_chunks(chunk_days=30)
        assert chunks == [
            (datetime.date(2023, 1, 1), datetime.date(2023, 1, 1))
        ]

    def test_create_date_chunks_perfect_fit(self, tracker):
        """Tests chunking where the range is a perfect multiple."""
        tracker.min_date = datetime.date(2023, 1, 1)
        tracker.max_date = datetime.date(2023, 1, 4)  # 4 days
        chunks = tracker._create_date_chunks(chunk_days=2)
        assert chunks == [
            (datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)),
            (datetime.date(2023, 1, 3), datetime.date(2023, 1, 4)),
        ]


class TestFetchData:
    def test_fetch_data_success(self, tracker, mock_session):
        """Tests _fetch_data successful path."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "success"}
        mock_session.get.return_value = mock_response

        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )

        assert result == {"data": "success"}
        expected_params = {
            "min_date": "2023-01-01",
            "max_date": "2023-01-02",
            "sensor_ID": "DE_01",
        }
        mock_session.get.assert_called_with(
            tracker.BASE_URL, params=expected_params, timeout=tracker.timeout
        )
        mock_response.raise_for_status.assert_called_once()

    def test_fetch_data_http_error(self, tracker, mock_session, mocker):
        """
        Tests _fetch_data handling a 404 HTTPError (a client error).
        It should NOT retry and should return None immediately.
        """
        # 1. Mock time.sleep just in case (though it shouldn't be called)
        mock_sleep = mocker.patch("time.sleep")

        # 2. Create the mock response and error
        mock_response = MagicMock(spec=requests.Response)
        mock_error = requests.exceptions.HTTPError("404 Not Found")

        # 3. This is the key: create the nested .response.status_code
        mock_error.response = mock_response
        mock_error.response.status_code = 404

        # 4. Assign the fully-formed error object
        mock_response.raise_for_status.side_effect = mock_error
        mock_session.get.return_value = mock_response

        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )

        # 5. Assert the function returned None and DID NOT retry
        assert result is None
        assert mock_session.get.call_count == 1  # Should fail once and give up
        assert mock_sleep.call_count == 0  # Should not have slept

    def test_fetch_data_request_exception(self, tracker, mock_session, mocker):
        """Tests _fetch_data handling RequestException (e.g., timeout)."""
        mocker.patch("time.sleep")

        mock_session.get.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )
        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )
        assert result is None

    def test_fetch_data_json_decode_error(self, tracker, mock_session):
        """Tests _fetch_data handling bad JSON."""
        mock_response = MagicMock()
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "msg", "doc", 0
        )
        mock_session.get.return_value = mock_response

        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )
        assert result is None

    def test_fetch_chunk_success(
        self, tracker, mock_session, sample_api_json, sample_api_df, mocker
    ):
        """Tests _fetch_chunk success path."""
        mocker.patch.object(
            tracker, "_fetch_data", return_value=sample_api_json
        )

        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )

        # Check that the end date was correctly incremented by 1 day for the API call
        tracker._fetch_data.assert_called_with(
            session=mock_session,
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 1, 3),  # 2 + 1 day
            sensor_id="DE_01",
        )

        pd.testing.assert_frame_equal(result, sample_api_df)

    def test_fetch_chunk_no_data(self, tracker, mock_session, mocker):
        """Tests _fetch_chunk when API returns None or empty list."""
        mocker.patch.object(tracker, "_fetch_data", return_value=None)
        assert (
            tracker._fetch_chunk(
                mock_session,
                "DE_01",
                datetime.date(2023, 1, 1),
                datetime.date(2023, 1, 2),
            )
            is None
        )

        mocker.patch.object(tracker, "_fetch_data", return_value=[])
        assert (
            tracker._fetch_chunk(
                mock_session,
                "DE_01",
                datetime.date(2023, 1, 1),
                datetime.date(2023, 1, 2),
            )
            is None
        )

    def test_fetch_chunk_key_error(self, tracker, mock_session, mocker):
        """Tests _fetch_chunk handling bad data (KeyError)."""
        bad_json = [{"not_a_date_column": "2023-01-01"}]
        mocker.patch.object(tracker, "_fetch_data", return_value=bad_json)

        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None


class TestGetDataOrchestration:

    # This helper function correctly mocks the logic of
    # _process_and_combine_data
    def _mock_combiner(self, all_dataframes):
        if not all_dataframes:
            print("Mock Combiner: No dataframes received.")
            return None
        try:
            print(
                f"Mock Combiner: Concatenating {len(all_dataframes)} dataframes."
            )
            combined_data = pd.concat(all_dataframes, ignore_index=True)
            print(
                f"Mock Combiner: Shape before drop_duplicates: {combined_data.shape}"
            )
            combined_data.drop_duplicates(
                subset=["date", "sensor_ID"], inplace=True
            )
            print(
                f"Mock Combiner: Shape after drop_duplicates: {combined_data.shape}"
            )
            combined_data.sort_values(by=["date"], inplace=True)
            combined_data.reset_index(drop=True, inplace=True)
            return combined_data
        except Exception as e:
            print(f"Mock Combiner Error: {e}")
            return None

    @patch("poseidon_utils.event_tracker.requests.Session")
    @patch("poseidon_utils.event_tracker.ThreadPoolExecutor")
    @patch("poseidon_utils.event_tracker.as_completed")
    @patch(
        "poseidon_utils.event_tracker.EventTracker._process_and_combine_data"
    )
    @patch("poseidon_utils.event_tracker.EventTracker._create_date_chunks")
    @patch("poseidon_utils.event_tracker.EventTracker._resolve_sensor_ids")
    def test_get_data_happy_path(
        self,
        mock_resolve_ids,
        mock_create_chunks,
        mock_process_data,
        mock_as_completed,
        mock_executor_cls,
        mock_session_cls,
        tracker,
        sample_api_df,
    ):
        """Integration test for get_data: success on first try."""

        # --- Configure Mocks ---
        mock_session = mock_session_cls.return_value.__enter__.return_value

        # This is the executor for the *first* `with` block (in get_data)
        mock_executor = mock_executor_cls.return_value.__enter__.return_value

        # Configure the one future that will be submitted
        mock_future = MagicMock()
        mock_future.result.return_value = sample_api_df
        mock_executor.submit.return_value = mock_future

        # as_completed will be called with {mock_future: task_info}
        # and should return an iterator
        mock_as_completed.return_value = [mock_future]

        # Configure helper mocks
        mock_resolve_ids.return_value = ["DE_01"]
        date_chunks = [(datetime.date(2023, 1, 1), datetime.date(2023, 1, 5))]
        mock_create_chunks.return_value = date_chunks

        mock_process_data.side_effect = self._mock_combiner

        # --- Run ---
        result = tracker.get_data(sensor_ids="DE_01", chunk_days=5)

        # --- Asserts ---
        mock_resolve_ids.assert_called_with("DE_01")

        # Assert that the primary executor's submit was called
        mock_executor.submit.assert_called_once_with(
            tracker._fetch_chunk,
            mock_session,
            "DE_01",
            date_chunks[0][0],
            date_chunks[0][1],
        )
        mock_process_data.assert_called_once()
        pd.testing.assert_frame_equal(result, sample_api_df)

    @patch("poseidon_utils.event_tracker.requests.Session")
    @patch("poseidon_utils.event_tracker.ThreadPoolExecutor")
    @patch("poseidon_utils.event_tracker.as_completed")
    @patch(
        "poseidon_utils.event_tracker.EventTracker._process_and_combine_data"
    )
    @patch("poseidon_utils.event_tracker.EventTracker._create_date_chunks")
    @patch("poseidon_utils.event_tracker.EventTracker._resolve_sensor_ids")
    def test_get_data_fallback_logic(
        self,
        mock_resolve_ids,
        mock_create_chunks,
        mock_process_data,
        mock_as_completed,
        mock_executor_cls,
        mock_session_cls,
        tracker,
        sample_api_df,
    ):
        """Integration test for get_data: primary chunk fails, fallback succeeds."""

        # --- Configure Mocks ---
        mock_session = mock_session_cls.return_value.__enter__.return_value

        # ThreadPoolExecutor is called TWICE. We need to mock both.
        mock_executor_primary = MagicMock(name="PrimaryExecutor")
        mock_executor_fallback = MagicMock(name="FallbackExecutor")
        mock_executor_cls.return_value.__enter__.side_effect = [
            mock_executor_primary,
            mock_executor_fallback,
        ]

        # 1. Primary (chunk) execution -> Fails
        mock_fail_future = MagicMock()
        mock_fail_future.result.side_effect = Exception("Chunk Failed!")
        mock_executor_primary.submit.return_value = mock_fail_future

        # 2. Fallback (day) execution -> Succeeds
        #    <-- FIX: Create 5 *distinct* futures
        mock_day_futures = [
            MagicMock(result=lambda: sample_api_df.iloc[[0]].copy())
            for _ in range(5)
        ]
        #    <-- FIX: Make submit return one future at a time from the list
        mock_executor_fallback.submit.side_effect = mock_day_futures

        # Configure as_completed to be called twice
        mock_as_completed.side_effect = [
            [mock_fail_future],  # First call (in _process_chunk_results)
            mock_day_futures,  # <-- FIX: Return the *same list* of futures
        ]

        # Configure helper mocks
        mock_resolve_ids.return_value = ["DE_01"]
        date_chunks = [
            (datetime.date(2023, 1, 1), datetime.date(2023, 1, 5))
        ]  # 5 days total
        mock_create_chunks.return_value = date_chunks

        mock_process_data.side_effect = self._mock_combiner

        # --- Run ---
        result = tracker.get_data(sensor_ids="DE_01", chunk_days=5)

        # --- Asserts ---
        # 1. Primary task was submitted to the primary executor
        mock_executor_primary.submit.assert_called_once_with(
            tracker._fetch_chunk,
            mock_session,
            "DE_01",
            date_chunks[0][0],
            date_chunks[0][1],
        )

        # 2. Fallback daily tasks were submitted to the fallback executor
        # 5 days in the date range (1, 2, 3, 4, 5)
        assert mock_executor_fallback.submit.call_count == 5
        mock_executor_fallback.submit.assert_any_call(
            tracker._fetch_day, mock_session, "DE_01", datetime.date(2023, 1, 1)
        )
        mock_executor_fallback.submit.assert_any_call(
            tracker._fetch_day, mock_session, "DE_01", datetime.date(2023, 1, 5)
        )

        # 3. Final data is correct
        assert not result.empty
        # 5 calls all returned the same 1-row df, _mock_combiner will drop_duplicates
        assert len(result) == 1

    def test_process_and_combine_data(self, tracker, sample_api_df):
        """Tests the final data combination and deduplication step."""
        df1 = sample_api_df.copy()  # 2 rows
        df2 = sample_api_df.copy()  # 2 duplicate rows
        df3 = sample_api_df.copy()
        df3["date"] = df3["date"] + pd.Timedelta(days=1)  # 2 new rows

        all_dfs = [df1, df2, df3]
        result = tracker._process_and_combine_data(all_dfs)

        # Should have original 2 rows + 2 new rows = 4
        # The duplicate df2 should be dropped
        assert len(result) == 4
        assert result.iloc[0]["date"] < result.iloc[-1]["date"]  # Sorted

    def test_process_and_combine_data_empty(self, tracker):
        """Tests combination with no data."""
        assert tracker._process_and_combine_data([]) is None


class TestCSVGeneration:

    def test_gen_abbr_flood_event_csv_new_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Tests generating the abbreviated CSV from scratch."""
        mocker.patch("pandas.read_csv", side_effect=FileNotFoundError)

        saved_df = None

        def capture_df_self(df_self, *args, **kwargs):
            nonlocal saved_df
            saved_df = df_self

        mock_to_csv = mocker.patch(
            "pandas.DataFrame.to_csv",
            side_effect=capture_df_self,
            autospec=True,
        )
        csv_path = tmp_path / "abbr.csv"

        tracker._gen_abbr_flood_event_csv(sample_flood_data, str(csv_path))

        mock_to_csv.assert_called_once_with(ANY, str(csv_path), index=False)

        result_df = saved_df

        assert len(result_df) == 2
        assert (result_df["flood_event"] == 1).all()
        assert result_df.iloc[0]["sensor_ID"] == "DE_01"
        assert result_df.iloc[1]["sensor_ID"] == "DE_02"
        assert result_df.iloc[0]["max_road_water_level_(ft)"] == 0.5
        assert result_df.iloc[1]["max_road_water_level_(ft)"] == 0.3

    def test_gen_flood_tracker_csv_new_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Tests generating the detailed (row-by-row) CSV from scratch."""
        mocker.patch("pandas.read_csv", side_effect=FileNotFoundError)

        saved_df = None

        def capture_df_self(df_self, *args, **kwargs):
            nonlocal saved_df
            saved_df = df_self

        mock_to_csv = mocker.patch(
            "pandas.DataFrame.to_csv",
            side_effect=capture_df_self,
            autospec=True,
        )
        csv_path = tmp_path / "full.csv"

        tracker._gen_flood_tracker(sample_flood_data, str(csv_path))

        mock_to_csv.assert_called_once_with(ANY, str(csv_path), index=False)
        result_df = saved_df

        # 3 rows for DE_01 event + 3 rows for DE_02 event = 6 rows
        assert len(result_df) == 6
        # Both should be part of flood_event 1 due to overlap
        assert (result_df["flood_event"] == 1).all()

        # <-- FIX: The re-numbering function sorts by time, so the
        # end of the first event is at iloc[3], not iloc[2].
        # result_df.iloc[0] # DE_01 12:00
        # result_df.iloc[1] # DE_01 12:05
        # result_df.iloc[2] # DE_02 12:05  <-- This was the row being checked
        # result_df.iloc[3] # DE_01 12:10  <-- This is the correct row

        # Check that the last row of the first event has full data (now at iloc 3)
        assert result_df.iloc[3]["duration_(hours)"] is not None
        assert result_df.iloc[3]["max_road_water_level_(ft)"] == 0.5

        # Check that the last row of the second event has full data (at iloc 5)
        assert result_df.iloc[5]["duration_(hours)"] is not None
        assert result_df.iloc[5]["max_road_water_level_(ft)"] == 0.3


class TestOutageLogic:

    def test_find_outages(self, tracker, sample_flood_data, tmp_path, mocker):
        """Tests finding a time gap > 1 hour."""
        mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
        csv_path = tmp_path / "outages.csv"

        result_df = tracker._find_outages(sample_flood_data, str(csv_path))

        mock_to_csv.assert_called_once_with(str(csv_path), index=False)
        assert result_df is not None
        assert len(result_df) == 1
        assert result_df.iloc[0]["outage_number"] == 1
        assert result_df.iloc[0]["sensor_ID"] == "DE_01"
        assert result_df.iloc[0]["duration_(hours)"] > 4.0  # Check the gap

    def test_find_outages_no_gaps(
        self, tracker, sample_api_df, tmp_path, mocker
    ):
        """Tests that no outages are found when gaps are small."""
        mock_to_csv = mocker.patch("pandas.DataFrame.to_csv")
        csv_path = tmp_path / "outages.csv"

        result_df = tracker._find_outages(sample_api_df, str(csv_path))

        mock_to_csv.assert_not_called()
        assert result_df is None

    def test_check_for_outage_during_flood(self, tracker, tmp_path, mocker):
        """Tests flagging outages that occur during a flood."""
        # Create dummy data
        flood_df = pd.DataFrame(
            {
                "flood_event": [1],
                "sensor_ID": ["DE_01"],
                "start_time_UTC": [
                    pd.to_datetime("2023-01-01 12:00:00", utc=True)
                ],
                "end_time_UTC": [
                    pd.to_datetime("2023-01-01 14:00:00", utc=True)
                ],
                "outage": ["No"],
            }
        )
        outage_df = pd.DataFrame(
            {
                "outage_number": [101],
                "sensor_ID": ["DE_01"],
                "start_time_UTC": [
                    pd.to_datetime("2023-01-01 13:00:00", utc=True)
                ],  # During flood
                "end_time_UTC": [
                    pd.to_datetime("2023-01-01 13:30:00", utc=True)
                ],
                "during_flood_event": ["No"],
                "flood_event_number": [pd.NA],
            }
        )

        # Pass copies to the mock reader
        mocker.patch(
            "pandas.read_csv", side_effect=[flood_df.copy(), outage_df.copy()]
        )

        # <-- FIX: Capture the DataFrames sent to to_csv
        saved_dfs = {}

        def capture_df_self(df_self, path, *args, **kwargs):
            if "floods.csv" in str(path):
                saved_dfs["floods"] = df_self
            if "outages.csv" in str(path):
                saved_dfs["outages"] = df_self

        mock_to_csv = mocker.patch(
            "pandas.DataFrame.to_csv",
            side_effect=capture_df_self,
            autospec=True,
        )

        outage_csv = str(tmp_path / "outages.csv")
        flood_csv = str(tmp_path / "floods.csv")

        tracker._check_for_outage_during_flood(outage_csv, flood_csv)

        # Should have been called twice (once for each file)
        assert mock_to_csv.call_count == 2

        # <-- FIX: Check the *captured* DataFrames, not the originals
        assert "floods" in saved_dfs
        assert "outages" in saved_dfs

        # Check captured flood_df
        assert saved_dfs["floods"].iloc[0]["outage"] == "Yes"

        # Check captured outage_df
        assert saved_dfs["outages"].iloc[0]["during_flood_event"] == "Yes"
        assert saved_dfs["outages"].iloc[0]["flood_event_number"] == 1


class TestPipelineAndPlotting:

    def test_plot_and_save_flood_plots(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Tests the plotting logic."""
        # Create a dummy flood CSV
        flood_df = pd.DataFrame(
            {
                "flood_event": [1],
                "sensor_ID": ["DE_01"],
                "start_time_UTC": [
                    pd.to_datetime("2023-01-01 12:00:00", utc=True)
                ],
                "end_time_UTC": [
                    pd.to_datetime("2023-01-01 12:10:00", utc=True)
                ],
            }
        )

        plot_dir = tmp_path / "plots"
        csv_path = tmp_path / "floods.csv"

        mocker.patch("pandas.read_csv", return_value=flood_df)
        mock_exists = mocker.patch("os.path.exists", return_value=False)
        mock_makedirs = mocker.patch("os.makedirs")

        tracker._plot_and_save_flood_plots(
            sample_flood_data, str(csv_path), str(plot_dir)
        )

        # Check folder creation
        mock_exists.assert_any_call(str(plot_dir))
        mock_makedirs.assert_called_with(str(plot_dir))

        # Check plot saving
        expected_plot_path = os.path.join(plot_dir, "flood_event_1_DE_01.png")
        mock_exists.assert_any_call(expected_plot_path)
        plt.savefig.assert_called_with(expected_plot_path)

    def test_plot_skips_existing(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Tests that plotting is skipped if the file already exists."""
        flood_df = pd.DataFrame(
            {
                "flood_event": [1],
                "sensor_ID": ["DE_01"],
                "start_time_UTC": ["2023-01-01"],
                "end_time_UTC": ["2023-01-02"],
            }
        )
        plot_dir = tmp_path / "plots"
        csv_path = tmp_path / "floods.csv"

        mocker.patch("pandas.read_csv", return_value=flood_df)

        # Mock os.path.exists to return True for the plot file
        def exists_side_effect(path):
            if str(path).endswith(".png"):
                return True  # Plot file exists
            if str(path) == str(plot_dir):
                return True  # Dir exists
            return False

        mocker.patch("os.path.exists", side_effect=exists_side_effect)
        mock_makedirs = mocker.patch("os.makedirs")

        tracker._plot_and_save_flood_plots(
            sample_flood_data, str(csv_path), str(plot_dir)
        )

        mock_makedirs.assert_not_called()  # Dir already exists
        plt.savefig.assert_not_called()

    def test_pull_data_gen_csvs_and_plots_full_run(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Integration test for the main pipeline method."""
        mocker.patch.object(tracker, "get_data", return_value=sample_flood_data)
        mock_gen_flood = mocker.patch.object(tracker, "_gen_flood_tracker")
        mock_gen_abbr = mocker.patch.object(
            tracker, "_gen_abbr_flood_event_csv"
        )
        mock_find_outage = mocker.patch.object(tracker, "_find_outages")
        mock_check_outage = mocker.patch.object(
            tracker, "_check_for_outage_during_flood"
        )
        mock_plot = mocker.patch.object(tracker, "_plot_and_save_flood_plots")

        csv_path = str(tmp_path / "data")

        result = tracker.pull_data_gen_csvs_and_plots(
            location="DE_01",
            full_csv_name=f"{csv_path}/full.csv",
            abbr_csv_name=f"{csv_path}/abbr.csv",
            outage_csv_name=f"{csv_path}/outage.csv",
            plot_folder=f"{csv_path}/plots",
        )

        assert result is sample_flood_data
        mock_gen_flood.assert_called_once_with(
            sample_flood_data, f"{csv_path}/full.csv"
        )
        mock_gen_abbr.assert_called_once_with(
            sample_flood_data, f"{csv_path}/abbr.csv"
        )
        mock_find_outage.assert_called_once_with(
            sample_flood_data, f"{csv_path}/outage.csv"
        )
        mock_check_outage.assert_called_once_with(
            f"{csv_path}/outage.csv", f"{csv_path}/abbr.csv"
        )
        mock_plot.assert_called_once_with(
            sample_flood_data, f"{csv_path}/abbr.csv", f"{csv_path}/plots"
        )

    def test_pull_data_get_data_fails(self, tracker, mocker):
        """Tests that the pipeline stops if get_data returns None."""
        mocker.patch.object(tracker, "get_data", return_value=None)
        mock_gen_flood = mocker.patch.object(tracker, "_gen_flood_tracker")

        result = tracker.pull_data_gen_csvs_and_plots(location="DE_01")

        assert result is None
        mock_gen_flood.assert_not_called()


class TestStaticMethods:

    def test_num_of_flood_days_by_start(self, tmp_path, mocker):
        """Tests static method num_of_flood_days_by_start."""
        csv_path = tmp_path / "floods.csv"
        df = pd.DataFrame(
            {
                "start_time_UTC": [
                    "2023-01-01 10:00:00",  # Day 1
                    "2023-01-01 20:00:00",  # Day 1
                    "2023-01-03 12:00:00",  # Day 3
                ]
            }
        )
        mocker.patch("pandas.read_csv", return_value=df)

        num_days = EventTracker.num_of_flood_days_by_start(str(csv_path))
        assert num_days == 2

    def test_num_of_flood_days_spanning(self, tmp_path, mocker):
        """Tests static method num_of_flood_days with spanning events."""
        csv_path = tmp_path / "floods.csv"
        df = pd.DataFrame(
            {
                "start_time_UTC": ["2023-01-01 23:00:00"],  # Spans 2 days
                "end_time_UTC": ["2023-01-02 01:00:00"],
                "start_time_EST": [
                    "2023-01-01 18:00:00"
                ],  # All on one day in EST
                "end_time_EST": ["2023-01-01 20:00:00"],
            }
        )
        mocker.patch("pandas.read_csv", return_value=df)

        num_days_utc = EventTracker.num_of_flood_days(
            timezone="UTC", csv_name=str(csv_path)
        )
        assert num_days_utc == 2

        num_days_est = EventTracker.num_of_flood_days(
            timezone="EST", csv_name=str(csv_path)
        )
        assert num_days_est == 1

    def test_avg_errors(self, tmp_path, mocker):
        """Tests static method avg_errors."""
        flood_csv_path = tmp_path / "floods.csv"
        error_csv_path = tmp_path / "errors.csv"

        flood_df = pd.DataFrame(
            {"end_time_UTC": [pd.to_datetime("2023-01-10 12:00:00", utc=True)]}
        )
        error_df = pd.DataFrame(
            {
                "Date": [
                    pd.to_datetime(
                        "2023-01-01 12:00:00", utc=True
                    ),  # Out of range
                    pd.to_datetime(
                        "2023-01-05 12:00:00", utc=True
                    ),  # In 7-day range
                    pd.to_datetime(
                        "2023-01-08 12:00:00", utc=True
                    ),  # In 7-day range
                ],
                "Mean": [100, 10, 20],
                "5th Pct": [100, 1, 3],
                "95th Pct": [100, 20, 40],
            }
        )

        mocker.patch(
            "pandas.read_csv", side_effect=[flood_df.copy(), error_df.copy()]
        )

        saved_df = None

        def capture_df_self(df_self, *args, **kwargs):
            nonlocal saved_df
            saved_df = df_self

        mock_to_csv = mocker.patch(
            "pandas.DataFrame.to_csv",
            side_effect=capture_df_self,
            autospec=True,
        )

        result = EventTracker.avg_errors(
            str(error_csv_path), csv_filename=str(flood_csv_path)
        )

        assert result is True
        mock_to_csv.assert_called_once_with(
            ANY, str(flood_csv_path), index=False
        )

        assert "Mean of Mean" in saved_df.columns
        assert saved_df.iloc[0]["Mean of Mean"] == 15.0
        assert saved_df.iloc[0]["Mean of 5th Pct"] == 2.0
        assert saved_df.iloc[0]["Mean of 95th Pct"] == 30.0


class TestErrorHandling:
    """
    Tests specific exception-handling branches not covered by
    happy-path or standard failure tests.
    """

    def test_fetch_day_key_error(self, tracker, mock_session, mocker):
        """Tests _fetch_day handling bad data (KeyError)."""
        bad_json = [{"not_a_date_column": "2023-01-01"}]
        mocker.patch.object(tracker, "_fetch_data", return_value=bad_json)

        result = tracker._fetch_day(
            mock_session, "DE_01", datetime.date(2023, 1, 1)
        )
        assert result is None

    def test_fetch_day_generic_exception(self, tracker, mock_session, mocker):
        """Tests _fetch_day handling a generic exception during pd.DataFrame creation."""
        mocker.patch.object(
            tracker, "_fetch_data", return_value=[{"date": "..."}]
        )
        # Patch pandas.DataFrame inside the poseidon_utils.event_tracker module
        mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame",
            side_effect=Exception("Generic Error"),
        )

        result = tracker._fetch_day(
            mock_session, "DE_01", datetime.date(2023, 1, 1)
        )
        assert result is None

    def test_fetch_chunk_generic_exception(self, tracker, mock_session, mocker):
        """Tests _fetch_chunk handling a generic exception during pd.DataFrame creation."""
        mocker.patch.object(
            tracker, "_fetch_data", return_value=[{"date": "..."}]
        )
        # Patch pandas.DataFrame inside the poseidon_utils.event_tracker module
        mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame",
            side_effect=Exception("Generic Error"),
        )

        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None

    def test_process_and_combine_data_exception(
        self, tracker, sample_api_df, mocker
    ):
        """Tests _process_and_combine_data failing on pd.concat."""
        mocker.patch(
            "poseidon_utils.event_tracker.pd.concat",
            side_effect=Exception("Concat Error"),
        )
        result = tracker._process_and_combine_data(
            [sample_api_df, sample_api_df]
        )
        assert result is None

    def test_check_for_outage_file_not_found(self, tracker, tmp_path, mocker):
        """Tests _check_for_outage_during_flood when a CSV is missing."""

        # <-- FIX: This mock now correctly sets the .filename attribute
        def mock_read_csv(filepath):
            e = FileNotFoundError()
            e.filename = filepath  # Manually set the filename
            raise e

        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=mock_read_csv,
        )

        # Call the function (outage_csv="missing.csv", abbr_flood_csv="also_missing.csv")
        tracker._check_for_outage_during_flood(
            "missing.csv", "also_missing.csv"
        )

        # <-- FIX: Assert against the *first* file it tries to read
        print.assert_any_call(
            "Error: Cannot check for outages, file not found: also_missing.csv"
        )

    def test_plot_and_save_file_not_found(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """Tests _plot_and_save_flood_plots when the flood CSV is missing."""
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=FileNotFoundError("File missing"),
        )
        csv_path = str(tmp_path / "missing_floods.csv")
        plot_folder = str(tmp_path / "plots")

        tracker._plot_and_save_flood_plots(
            sample_flood_data, csv_path, plot_folder
        )

        # Assert that the function printed the error and exited gracefully
        print.assert_any_call(
            f"Error: Cannot create plots, file not found: {csv_path}"
        )
        plt.figure.assert_not_called()  # Plotting should not have started

    def test_pull_data_csv_error(self, tracker, sample_flood_data, mocker):
        """Tests the except block for CSV generation in the main pipeline."""
        mocker.patch.object(tracker, "get_data", return_value=sample_flood_data)
        mocker.patch.object(
            tracker, "_gen_flood_tracker", side_effect=Exception("CSV Error")
        )

        tracker.pull_data_gen_csvs_and_plots(location="DE_01")
        print.assert_any_call(
            "An error occurred during CSV generation: CSV Error"
        )

    def test_pull_data_outage_error(self, tracker, sample_flood_data, mocker):
        """Tests the except block for outage checks in the main pipeline."""
        mocker.patch.object(tracker, "get_data", return_value=sample_flood_data)
        mocker.patch.object(tracker, "_gen_flood_tracker")  # Mock previous step
        mocker.patch.object(
            tracker, "_gen_abbr_flood_event_csv"
        )  # Mock previous step
        mocker.patch.object(
            tracker, "_find_outages", side_effect=Exception("Outage Error")
        )

        tracker.pull_data_gen_csvs_and_plots(location="DE_01")
        print.assert_any_call(
            "An error occurred during outage check: Outage Error"
        )

    def test_pull_data_plot_error(self, tracker, sample_flood_data, mocker):
        """Tests the except block for plotting in the main pipeline."""
        mocker.patch.object(tracker, "get_data", return_value=sample_flood_data)
        mocker.patch.object(tracker, "_gen_flood_tracker")  # Mock previous step
        mocker.patch.object(
            tracker, "_gen_abbr_flood_event_csv"
        )  # Mock previous step
        mocker.patch.object(tracker, "_find_outages")  # Mock previous step
        mocker.patch.object(
            tracker, "_check_for_outage_during_flood"
        )  # Mock previous step
        mocker.patch.object(
            tracker,
            "_plot_and_save_flood_plots",
            side_effect=Exception("Plot Error"),
        )

        tracker.pull_data_gen_csvs_and_plots(location="DE_01")
        print.assert_any_call("An error occurred during plotting: Plot Error")


class TestStaticErrorHandling:
    """Tests exception handling for the static methods."""

    # <-- FIX: This test is rewritten to handle different arg names and print statements
    @pytest.mark.parametrize(
        "method_to_test, args, csv_arg_name, expected_print",
        [
            (
                EventTracker.num_of_flood_days_by_start,
                (),
                "csv_name",
                "Error: File 'missing.csv' not found.",
            ),
            (
                EventTracker.num_of_flood_days,
                (),
                "csv_name",
                "Error: File 'missing.csv' not found.",
            ),
            (
                EventTracker.avg_errors,
                ("errors.csv",),
                "csv_filename",
                "Error: File not found. File missing",
            ),
        ],
    )
    def test_static_file_not_found(
        self, mocker, method_to_test, args, csv_arg_name, expected_print
    ):
        """Tests FileNotFoundError for static methods that read CSVs."""
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=FileNotFoundError("File missing"),
        )

        # Build the keyword argument dict, e.g., {"csv_name": "missing.csv"}
        csv_kwargs = {csv_arg_name: "missing.csv"}

        result = method_to_test(*args, **csv_kwargs)
        assert result is None

        # Check that the specific error message was printed
        print.assert_any_call(expected_print)

    def test_num_of_flood_days_bad_timezone(self, mocker, tmp_path):
        """Tests the error handling for an invalid timezone."""
        # Need a valid read_csv mock so it doesn't fail on that first
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            return_value=pd.DataFrame(),
        )

        result = EventTracker.num_of_flood_days(
            timezone="InvalidTZ", csv_name="dummy.csv"
        )
        assert result is None
        print.assert_any_call(
            "Error: Timezone 'InvalidTZ' not recognized. Use 'EST' or 'UTC'."
        )


class TestGetDataEdgeCases:
    """
    Covers edge cases in the get_data pipeline, specifically
    targeting lines reported as 'missing' by pytest-cov.
    """

    def test_get_data_no_sensor_ids(self, tracker, mocker):
        """
        Covers: 112
        Tests get_data when _resolve_sensor_ids returns None.
        """
        mocker.patch.object(tracker, "_resolve_sensor_ids", return_value=None)
        assert tracker.get_data("some_location") is None

    def test_get_data_no_date_chunks(self, tracker, mocker):
        """
        Covers: 117-118
        Tests get_data when _create_date_chunks returns an empty list.
        """
        mocker.patch.object(
            tracker, "_resolve_sensor_ids", return_value=["DE_01"]
        )
        mocker.patch.object(tracker, "_create_date_chunks", return_value=[])
        assert tracker.get_data("DE_01") is None
        print.assert_any_call(
            "No date chunks to process (check min/max dates)."
        )

    def test_fetch_data_request_exception(self, tracker, mock_session, mocker):
        """
        Covers: 431
        Tests the except block for requests.exceptions.RequestException.
        """
        mocker.patch("time.sleep")

        mock_session.get.side_effect = requests.exceptions.RequestException(
            "Timeout"
        )
        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )
        assert result is None
        print.assert_any_call(
            "Request error for sensor DE_01 (2023-01-01 to 2023-01-02): Timeout. Attempt 1 of 3."
        )

    def test_fetch_chunk_empty_json(self, tracker, mock_session, mocker):
        """
        Covers: 483, 488
        Tests _fetch_chunk when the API returns an empty list [].
        """
        mocker.patch.object(tracker, "_fetch_data", return_value=[])
        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None

    def test_fetch_chunk_key_error(self, tracker, mock_session, mocker):
        """
        Covers: 491-492
        Tests the except KeyError block in _fetch_chunk.
        """
        bad_json = [{"not_the_right_column": "value"}]
        mocker.patch.object(tracker, "_fetch_data", return_value=bad_json)
        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None
        print.assert_any_call(
            "Error processing DataFrame for DE_01: Missing expected column 'date'."
        )

    def test_fetch_day_empty_json(self, tracker, mock_session, mocker):
        """
        Covers: 582-585
        Tests _fetch_day when the API returns an empty list [].
        """
        mocker.patch.object(tracker, "_fetch_data", return_value=[])
        result = tracker._fetch_day(
            mock_session, "DE_01", datetime.date(2023, 1, 1)
        )
        assert result is None

    @patch("poseidon_utils.event_tracker.requests.Session")
    @patch("poseidon_utils.event_tracker.ThreadPoolExecutor")
    @patch("poseidon_utils.event_tracker.as_completed")
    def test_get_data_non_exception_failure(
        self,
        mock_as_completed,
        mock_executor_cls,
        mock_session_cls,
        tracker,
        mocker,
    ):
        """
        Covers: 617
        Tests the 'else' block in _process_chunk_results (task returns None).
        """
        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_executor = mock_executor_cls.return_value.__enter__.return_value

        # Mock a future that returns None instead of raising an error
        mock_future = MagicMock()
        mock_future.result.return_value = None

        mock_executor.submit.return_value = mock_future
        mock_as_completed.return_value = [mock_future]

        # Mock helpers
        mocker.patch.object(
            tracker, "_resolve_sensor_ids", return_value=["DE_01"]
        )
        mocker.patch.object(
            tracker,
            "_create_date_chunks",
            return_value=[
                (datetime.date(2023, 1, 1), datetime.date(2023, 1, 5))
            ],
        )

        # Run
        tracker.get_data("DE_01")

        # Check that the task was registered as a failure
        print.assert_any_call(
            "Task failed, queueing for 1-day fallback: DE_01 (2023-01-01 to 2023-01-05)"
        )

    def test_handle_failed_chunks_empty(self, tracker, mock_session):
        """
        Covers: 635
        Tests _handle_failed_chunks when given an empty set of tasks.
        """
        # This line is only reachable if called directly, as get_data()
        # has a check that prevents this.
        result = tracker._handle_failed_chunks(mock_session, set())
        assert result == []

    @patch("poseidon_utils.event_tracker.requests.Session")
    @patch("poseidon_utils.event_tracker.ThreadPoolExecutor")
    @patch("poseidon_utils.event_tracker.as_completed")
    def test_handle_failed_chunks_exception_in_fallback(
        self,
        mock_as_completed,
        mock_executor_cls,
        mock_session_cls,
        tracker,
        mocker,
    ):
        """
        Covers: 653-654
        Tests the 'except Exception' block in _handle_failed_chunks.
        """
        mock_session = mock_session_cls.return_value.__enter__.return_value
        mock_executor = mock_executor_cls.return_value.__enter__.return_value

        # Mock a future that *fails* during the fallback
        mock_fail_future = MagicMock()
        mock_fail_future.result.side_effect = Exception("Fallback task failed")

        mock_executor.submit.return_value = mock_fail_future
        mock_as_completed.return_value = [mock_fail_future]

        failed_task_set = {
            ("DE_01", datetime.date(2023, 1, 1), datetime.date(2023, 1, 1))
        }

        # Run
        result = tracker._handle_failed_chunks(mock_session, failed_task_set)

        # Assert it returned no data and printed the error
        assert result == []
        print.assert_any_call(
            "Error in fallback task DE_01 on 2023-01-01: Fallback task failed"
        )


class TestReassignLogicEdgeCases:
    """
    Covers edge cases in the _reassign_..._numbers functions.
    """

    def test_reassign_abbr_flood_numbers_empty(self, tracker):
        """
        Covers: 721
        Tests _reassign_abbr_flood_numbers with an empty DataFrame.
        """
        empty_df = pd.DataFrame(columns=["start_time_UTC", "end_time_UTC"])
        result = tracker._reassign_abbr_flood_numbers(empty_df)
        assert result.empty

    def test_reassign_abbr_flood_numbers_no_column(self, tracker):
        """
        Covers: 724
        Tests _reassign_abbr_flood_numbers when 'flood_event' column is missing.
        """
        df = pd.DataFrame(
            {
                "start_time_UTC": [pd.to_datetime("2023-01-01 12:00")],
                "end_time_UTC": [pd.to_datetime("2023-01-01 13:00")],
            }
        )
        result = tracker._reassign_abbr_flood_numbers(df)
        assert "flood_event" in result.columns
        assert result.iloc[0]["flood_event"] == 1

    def test_reassign_flood_numbers_empty_complete(self, tracker):
        """
        Covers: 923-924
        Tests _reassign_flood_numbers when there are no 'complete' rows.
        """
        df = pd.DataFrame(
            {
                "start_time_UTC": [pd.NaT],
                "end_time_UTC": [pd.NaT],
                "time_UTC": [pd.to_datetime("2023-01-01 12:00")],
                "sensor_ID": ["DE_01"],
                "duration_(hours)": [np.nan],
            }
        )
        result = tracker._reassign_flood_numbers(df)
        assert (
            "flood_event" not in result.columns
        )  # Column is only added if events are found
        print.assert_any_call("No complete flood events to re-number.")

    def test_reassign_flood_numbers_multiple_sensor_events(self, tracker):
        """
        Covers: 942-949
        Tests the 'elif row["sensor_ID"] in last_assigned_event' branch.
        This requires two non-overlapping events *on the same sensor*.
        """
        df = pd.DataFrame(
            [
                # Event 1 (Sensor DE_01)
                {
                    "time_UTC": "2023-01-01 12:00",
                    "sensor_ID": "DE_01",
                    "start_time_UTC": "2023-01-01 12:00",
                    "end_time_UTC": "2023-01-01 13:00",
                    "duration_(hours)": 1.0,
                    "flood_event": 0,
                },
                # Event 2 (Sensor DE_01) - does not overlap
                {
                    "time_UTC": "2023-01-02 12:00",
                    "sensor_ID": "DE_01",
                    "start_time_UTC": "2023-01-02 12:00",
                    "end_time_UTC": "2023-01-02 13:00",
                    "duration_(hours)": 1.0,
                    "flood_event": 0,
                },
            ]
        )

        result = tracker._reassign_flood_numbers(df)

        assert result.iloc[0]["flood_event"] == 1
        assert result.iloc[1]["flood_event"] == 2  # This is the second event


class TestPipelineEdgeCases:
    """
    Covers edge cases for pipeline helper methods.
    """

    def test_check_for_outage_empty_dfs(self, tracker, tmp_path, mocker):
        """
        Covers: 1231-1232
        Tests _check_for_outage_during_flood when one of the DFs is empty.
        """
        # Case 1: Flood DF is empty
        flood_df = pd.DataFrame()
        outage_df = pd.DataFrame({"sensor_ID": ["DE_01"]})
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=[flood_df, outage_df],
        )

        tracker._check_for_outage_during_flood("outages.csv", "floods.csv")
        print.assert_any_call("No flood or outage data to compare.")

        # Case 2: Outage DF is empty
        flood_df = pd.DataFrame({"sensor_ID": ["DE_01"]})
        outage_df = pd.DataFrame()
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=[flood_df, outage_df],
        )

        tracker._check_for_outage_during_flood("outages.csv", "floods.csv")
        print.assert_any_call("No flood or outage data to compare.")

    def test_plot_and_save_empty_csv(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 1327-1328
        Tests _plot_and_save_flood_plots when the abbreviated flood CSV is empty.
        """
        empty_flood_df = pd.DataFrame()
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            return_value=empty_flood_df,
        )
        mocker.patch(
            "poseidon_utils.event_tracker.os.path.exists", return_value=True
        )  # Say dir exists

        tracker._plot_and_save_flood_plots(
            sample_flood_data, "empty.csv", "plots"
        )

        print.assert_any_call("No flood events in CSV to plot.")
        plt.figure.assert_not_called()

    def test_plot_and_save_no_matching_raw_data(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 1357-1360
        Tests _plot_and_save_flood_plots when flood event has no matching raw data.
        """
        # Create a flood event that is far in the future
        future_flood_df = pd.DataFrame(
            {
                "flood_event": [99],
                "sensor_ID": ["DE_01"],
                "start_time_UTC": [
                    pd.to_datetime("2099-01-01 12:00", utc=True)
                ],
                "end_time_UTC": [pd.to_datetime("2099-01-01 13:00", utc=True)],
            }
        )
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            return_value=future_flood_df,
        )
        mocker.patch(
            "poseidon_utils.event_tracker.os.path.exists", return_value=False
        )  # Dir and file do not exist
        mocker.patch("poseidon_utils.event_tracker.os.makedirs")

        # sample_flood_data is from 2023, so it won't match the 2099 event
        tracker._plot_and_save_flood_plots(
            sample_flood_data, "floods.csv", "plots"
        )

        print.assert_any_call(
            "No raw data found for event 99 (DE_01). Skipping plot."
        )
        plt.figure.assert_not_called()


class TestCSVGenerationEdgeCases:
    """
    Covers edge cases for generating CSVs, like pre-existing or empty files.
    """

    def test_gen_abbr_csv_no_new_events(self, tracker, tmp_path, mocker):
        """
        Covers: 776
        Tests _gen_abbr_flood_event_csv when the input data has no floods.
        """
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=FileNotFoundError,
        )
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "abbr.csv")

        # Create data with no water levels > 0.02
        no_flood_data = pd.DataFrame(
            {
                "date": [pd.to_datetime("2023-01-01 12:00", utc=True)],
                "sensor_ID": ["DE_01"],
                "road_water_level_adj": [0.01],
                "sensor_water_level_adj": [0.5],
            }
        )

        tracker._gen_abbr_flood_event_csv(no_flood_data, csv_path)

        print.assert_any_call("No new flood events found.")
        mock_to_csv.assert_not_called()

    def test_gen_abbr_csv_with_existing_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 869-880
        Tests _gen_abbr_flood_event_csv when the CSV file already exists.
        """
        existing_df = pd.DataFrame(
            {
                "flood_event": [1],
                "sensor_ID": ["OLD_01"],
                "start_time_UTC": [
                    pd.to_datetime("2022-01-01 12:00", utc=True)
                ],
                "end_time_UTC": [pd.to_datetime("2022-01-01 13:00", utc=True)],
            }
        )
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv", return_value=existing_df
        )

        # <-- FIX: Use autospec=True to capture the 'self' (DataFrame) argument
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "abbr.csv")

        tracker._gen_abbr_flood_event_csv(sample_flood_data, csv_path)

        # <-- FIX: Get the DataFrame (self) from the call args
        saved_df = mock_to_csv.call_args.args[0]

        # Should contain the old event (event 1) and the new merged event (event 2)
        assert len(saved_df) == 3  # 1 old + 2 new
        assert saved_df["flood_event"].nunique() == 2
        assert saved_df.iloc[0]["flood_event"] == 1
        assert saved_df.iloc[1]["flood_event"] == 2
        assert saved_df.iloc[2]["flood_event"] == 2

    def test_gen_abbr_csv_with_empty_existing_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 750-753
        Tests _gen_abbr_flood_event_csv when CSV exists but is empty.
        """
        # File exists, but is empty (no 'flood_event' column)
        existing_df = pd.DataFrame()
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv", return_value=existing_df
        )

        # <-- FIX: Use autospec=True
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "abbr.csv")

        tracker._gen_abbr_flood_event_csv(sample_flood_data, csv_path)

        # <-- FIX: Get the DataFrame (self) from the call args
        saved_df = mock_to_csv.call_args.args[0]

        # Should be numbered starting from 1
        assert saved_df["flood_event"].min() == 1

    def test_gen_flood_tracker_csv_no_new_events(
        self, tracker, tmp_path, mocker
    ):
        """
        Covers: 992
        Tests _gen_flood_tracker when the input data has no floods.
        """
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=FileNotFoundError,
        )
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "full.csv")

        # Create data with no water levels > 0.02
        no_flood_data = pd.DataFrame(
            {
                "date": [pd.to_datetime("2023-01-01 12:00", utc=True)],
                "sensor_ID": ["DE_01"],
                "road_water_level_adj": [0.01],
                "sensor_water_level_adj": [0.5],
            }
        )

        tracker._gen_flood_tracker(no_flood_data, csv_path)

        print.assert_any_call("No new detailed flood data to track.")
        mock_to_csv.assert_not_called()

    def test_gen_flood_tracker_with_existing_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 1114-1115, 1122-1129
        Tests _gen_flood_tracker when the CSV file already exists.
        """
        existing_df = pd.DataFrame(
            {
                "flood_event": [1],
                "sensor_ID": ["OLD_01"],
                "time_UTC": [pd.to_datetime("2022-01-01 12:00", utc=True)],
                "start_time_UTC": [
                    pd.to_datetime("2022-01-01 12:00", utc=True)
                ],  # <-- FIX
                "end_time_UTC": [
                    pd.to_datetime("2022-01-01 12:05", utc=True)
                ],  # <-- FIX
                "duration_(hours)": [
                    0.083
                ],  # <-- FIX: This makes it a "complete" event
            }
        )
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv", return_value=existing_df
        )

        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "full.csv")

        tracker._gen_flood_tracker(sample_flood_data, csv_path)

        saved_df = mock_to_csv.call_args.args[0]

        # Should contain the old row (1) + new rows (6)
        assert len(saved_df) == 7
        assert (
            saved_df["flood_event"].nunique() == 2
        )  # Old event 1, new event 2

    def test_gen_flood_tracker_with_empty_existing_file(
        self, tracker, sample_flood_data, tmp_path, mocker
    ):
        """
        Covers: 954-972 (else branch of last_event_number)
        Tests _gen_flood_tracker when CSV exists but is empty.
        """
        # File exists, but is empty (no 'flood_event' column)
        existing_df = pd.DataFrame()
        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv", return_value=existing_df
        )

        # <-- FIX: Use autospec=True
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )
        csv_path = str(tmp_path / "full.csv")

        tracker._gen_flood_tracker(sample_flood_data, csv_path)

        # <-- FIX: Get the DataFrame (self) from the call args
        saved_df = mock_to_csv.call_args.args[0]

        # Should be numbered starting from 1
        assert saved_df["flood_event"].min() == 1


class TestStaticMethodEdgeCases:
    """
    Covers edge cases for static methods.
    """

    def test_avg_errors_with_save_file_as(self, tmp_path, mocker):
        """
        Covers: 1608
        Tests the 'else' branch of avg_errors by providing 'save_file_as'.
        """
        flood_csv_path = tmp_path / "floods.csv"
        error_csv_path = tmp_path / "errors.csv"
        save_as_path = tmp_path / "new_file.csv"

        flood_df = pd.DataFrame(
            {"end_time_UTC": [pd.to_datetime("2023-01-10 12:00", utc=True)]}
        )
        error_df = pd.DataFrame(
            {
                "Date": [pd.to_datetime("2023-01-05 12:00", utc=True)],
                "Mean": [10],
                "5th Pct": [1],
                "95th Pct": [20],
            }
        )

        mocker.patch(
            "poseidon_utils.event_tracker.pd.read_csv",
            side_effect=[flood_df.copy(), error_df.copy()],
        )

        # <-- FIX: Use autospec=True to correctly capture the call
        mock_to_csv = mocker.patch(
            "poseidon_utils.event_tracker.pd.DataFrame.to_csv", autospec=True
        )

        EventTracker.avg_errors(
            str(error_csv_path),
            save_file_as=str(save_as_path),
            csv_filename=str(flood_csv_path),
        )

        # <-- FIX: The assertion now correctly checks for the 'self' arg (ANY)
        # and the positional filename arg.
        mock_to_csv.assert_called_once_with(ANY, str(save_as_path), index=False)


class TestFinalCoverage:
    """
    Re-adding tests for lines that are still showing as missed.
    This may be redundant, but ensures they are present.
    """

    def test_fetch_data_request_exception(self, tracker, mock_session, mocker):
        """
        Covers: 431
        """
        mocker.patch("time.sleep")

        mock_session.get.side_effect = requests.exceptions.RequestException(
            "Timeout"
        )
        result = tracker._fetch_data(
            mock_session,
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
            "DE_01",
        )
        assert result is None
        print.assert_any_call(
            "Request error for sensor DE_01 (2023-01-01 to 2023-01-02): Timeout. Attempt 1 of 3."
        )

    def test_fetch_chunk_empty_json(self, tracker, mock_session, mocker):
        """
        Covers: 488
        """
        mocker.patch.object(tracker, "_fetch_data", return_value=[])
        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None

    def test_fetch_chunk_key_error(self, tracker, mock_session, mocker):
        """
        Covers: 491-492
        """
        bad_json = [{"not_the_right_column": "value"}]
        mocker.patch.object(tracker, "_fetch_data", return_value=bad_json)
        result = tracker._fetch_chunk(
            mock_session,
            "DE_01",
            datetime.date(2023, 1, 1),
            datetime.date(2023, 1, 2),
        )
        assert result is None
        print.assert_any_call(
            "Error processing DataFrame for DE_01: Missing expected column 'date'."
        )

    def test_handle_failed_chunks_empty(self, tracker, mock_session):
        """
        Covers: 635
        """
        result = tracker._handle_failed_chunks(mock_session, set())
        assert result == []

    def test_reassign_flood_numbers_multiple_sensor_events(self, tracker):
        """
        Covers: 942-949
        Tests the 'elif row["sensor_ID"] in last_assigned_event' branch.
        This requires two non-overlapping events *on the same sensor*.
        """
        df = pd.DataFrame(
            [
                # Event 1 (Sensor DE_01)
                {
                    "time_UTC": "2023-01-01 12:00",
                    "sensor_ID": "DE_01",
                    "start_time_UTC": "2023-01-01 12:00",
                    "end_time_UTC": "2023-01-01 13:00",
                    "duration_(hours)": 1.0,
                    "flood_event": 0,
                },
                # Event 2 (Sensor DE_01) - does not overlap
                {
                    "time_UTC": "2023-01-02 12:00",
                    "sensor_ID": "DE_01",
                    "start_time_UTC": "2023-01-02 12:00",
                    "end_time_UTC": "2023-01-02 13:00",
                    "duration_(hours)": 1.0,
                    "flood_event": 0,
                },
            ]
        )

        result = tracker._reassign_flood_numbers(df)

        assert result.iloc[0]["flood_event"] == 1
        assert result.iloc[1]["flood_event"] == 2  # This is the second event
