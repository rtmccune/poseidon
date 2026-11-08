"""Tests for EventTracker flood detection, numbering, and API access."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import requests

from poseidon_utils import event_tracker as et_module
from poseidon_utils.event_tracker import EventTracker

START = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def records(levels_by_sensor, step_minutes=6, start=START):
    """
    Build raw SDFP records from per-sensor road water levels.

    Parameters
    ----------
    levels_by_sensor : dict of str to list of float
        Road water level (ft) series for each sensor.
    step_minutes : int, optional
        Minutes between readings. Default is 6.
    start : pandas.Timestamp, optional
        Time of the first reading.

    Returns
    -------
    pandas.DataFrame
        Raw records with the API's column names.
    """
    rows = []
    for sensor, levels in levels_by_sensor.items():
        for i, level in enumerate(levels):
            rows.append({
                "sensor_ID": sensor,
                "date": start + pd.Timedelta(minutes=step_minutes * i),
                "road_water_level_adj": level,
                "sensor_water_level_adj": level + 3.0,
                "date_surveyed": start,
            })
    return pd.DataFrame(rows)


def detect_flood_events(data):
    """
    Run event detection and return the events as a table.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw SDFP records.

    Returns
    -------
    pandas.DataFrame
        One row per completed flood event.
    """
    return pd.DataFrame(EventTracker._detect_flood_events(data))


@pytest.fixture
def tracker():
    """
    EventTracker for a one-week window with dummy credentials.

    Returns
    -------
    EventTracker
        Tracker instance.
    """
    return EventTracker(("user", "pass"), dt.date(2024, 1, 1),
                        dt.date(2024, 1, 7), max_workers=2)


# ----- Event detection and numbering -----------------------------------------


def test_event_spans_first_rise_to_first_negative_reading():
    """Events start above 0.02 ft and end at the first reading < 0."""
    events = detect_flood_events(records(
        {"AA_01": [-0.5, 0.01, 0.05, 0.4, 0.0, -0.1, -0.5]}))
    assert len(events) == 1
    event = events.iloc[0]
    assert event["start_time_UTC"] == START + pd.Timedelta(minutes=12)
    assert event["end_time_UTC"] == START + pd.Timedelta(minutes=30)
    assert event["max_road_water_level_(ft)"] == 0.4
    assert event["max_road_water_level_(m)"] == round(0.4 * 0.3048, 3)
    assert event["duration_(hours)"] == 0.3
    assert event["start_time_EST"].utcoffset() == dt.timedelta(hours=-5)


def test_unfinished_flood_does_not_leak_into_the_next_sensor():
    """Flood state resets per sensor (regression test)."""
    events = detect_flood_events(records({
        "AA_01": [-0.5, 0.1, 0.3, -0.1, 0.05, 0.4, 0.3],  # 2nd never ends
        "BB_01": [-0.5, -0.4, 0.03, 0.25, -0.05, -0.2, -0.3],
    }))
    assert events["sensor_ID"].tolist() == ["AA_01", "BB_01"]
    assert (events["end_time_UTC"] > events["start_time_UTC"]).all()
    assert events.iloc[1]["max_road_water_level_(ft)"] == 0.25


def test_missing_levels_neither_start_nor_end_events():
    """NaN readings are ignored by the threshold tests."""
    events = detect_flood_events(records(
        {"AA_01": [np.nan, 0.1, np.nan, 0.2, -0.1]}))
    assert len(events) == 1
    assert events.iloc[0]["max_road_water_level_(ft)"] == 0.2


def test_overlapping_events_across_sensors_share_a_number(tracker):
    """Numbers group events that overlap in time, in start order."""
    t = START
    events = pd.DataFrame({
        "sensor_ID": ["B", "A", "C", "A"],
        "start_time_UTC": [t + pd.Timedelta(hours=h) for h in (1, 0, 5, 3)],
        "end_time_UTC": [t + pd.Timedelta(hours=h) for h in (4, 2, 6, 3.5)],
    })
    numbered = tracker._reassign_abbr_flood_numbers(events)
    assert numbered["sensor_ID"].tolist() == ["A", "B", "A", "C"]
    assert numbered["flood_event"].tolist() == [1, 1, 1, 2]


def test_event_starting_exactly_at_previous_end_is_new(tracker):
    """Touching (not overlapping) events get separate numbers."""
    t = START
    numbered = tracker._reassign_abbr_flood_numbers(pd.DataFrame({
        "sensor_ID": ["A", "B"],
        "start_time_UTC": [t, t + pd.Timedelta(hours=1)],
        "end_time_UTC": [t + pd.Timedelta(hours=1), t + pd.Timedelta(hours=2)],
    }))
    assert numbered["flood_event"].tolist() == [1, 2]


def test_detailed_rows_carry_the_summary_event_numbers(tracker, tmp_path):
    """Detailed rows use the summary numbering; summary on last row."""
    raw = records({"AA_01": [-0.5, 0.1, 0.3, -0.1, -0.5],
                   "BB_01": [-0.5, -0.5, 0.2, 0.1, -0.2]})
    abbr, full = tmp_path / "abbr.csv", tmp_path / "full.csv"
    tracker._gen_abbr_flood_event_csv(raw, abbr)
    tracker._gen_flood_tracker(raw, full, abbr)
    detailed = pd.read_csv(full)

    assert len(detailed) == 3 + 3  # rise through the ending reading
    assert set(detailed["flood_event"]) == {1}
    last_rows = detailed.groupby("sensor_ID").tail(1)
    assert last_rows["duration_(hours)"].notna().all()
    assert detailed["duration_(hours)"].notna().sum() == 2


# ----- CSV generation --------------------------------------------------------


def test_new_events_merge_without_overwriting_manual_edits(tracker, tmp_path):
    """Existing CSV rows win; new events are appended and renumbered."""
    csv = tmp_path / "abbr.csv"
    first = records({"AA_01": [-0.5, 0.1, 0.3, -0.1]})
    tracker._gen_abbr_flood_event_csv(first, csv)

    edited = pd.read_csv(csv)
    edited.loc[0, "max_road_water_level_(ft)"] = 9.9  # manual edit
    edited.to_csv(csv, index=False)

    later = records({"AA_01": [-0.5, 0.1, 0.3, -0.1]},
                    start=START + pd.Timedelta(days=1))
    tracker._gen_abbr_flood_event_csv(pd.concat([first, later]), csv)
    merged = pd.read_csv(csv)
    assert merged["flood_event"].tolist() == [1, 2]
    assert merged.loc[0, "max_road_water_level_(ft)"] == 9.9


def test_outages_are_gaps_longer_than_one_hour(tracker, tmp_path):
    """A 2 h gap is an outage; a 1 h gap is not."""
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00",
                            "2024-01-01 03:00"], utc=True)
    data = pd.DataFrame({"sensor_ID": "AA_01", "date": times})
    outages = tracker._find_outages(data, tmp_path / "outages.csv")
    assert len(outages) == 1
    assert outages.iloc[0]["duration_(hours)"] == 2.0


def test_outages_during_floods_are_flagged_both_ways(tracker, tmp_path):
    """Outage and flood CSVs each mark their overlaps."""
    abbr, outage = tmp_path / "abbr.csv", tmp_path / "outages.csv"
    pd.DataFrame({
        "flood_event": [1, 2], "sensor_ID": ["AA_01", "AA_01"],
        "start_time_UTC": ["2024-01-01 00:00+00:00", "2024-01-02 00:00+00:00"],
        "end_time_UTC": ["2024-01-01 06:00+00:00", "2024-01-02 06:00+00:00"],
    }).to_csv(abbr, index=False)
    pd.DataFrame({
        "outage_number": [1, 2], "sensor_ID": ["AA_01", "BB_01"],
        "start_time_UTC": ["2024-01-01 02:00+00:00", "2024-01-01 02:00+00:00"],
    }).to_csv(outage, index=False)

    tracker._check_for_outage_during_flood(outage, abbr)
    outages, floods = pd.read_csv(outage), pd.read_csv(abbr)
    assert outages["during_flood_event"].tolist() == ["Yes", "No"]
    assert outages.loc[0, "flood_event_number"] == 1
    assert floods["outage"].tolist() == ["Yes", "No"]


def test_full_pipeline_and_regeneration(tracker, tmp_path, monkeypatch):
    """Pull outputs are consistent and survive a manual merge edit."""
    raw = records({"AA_01": [-0.5, 0.1, 0.3, -0.1, 0.2, 0.5, -0.3, -0.5]})
    monkeypatch.setattr(tracker, "get_data", lambda *a, **k: raw)
    out = tmp_path / "out"
    tracker.pull_data_gen_csvs_and_plots("ignored", output_dir=str(out))

    abbr = pd.read_csv(out / "abbr_flood_events.csv")
    assert abbr["flood_event"].tolist() == [1, 2]
    assert sorted(p.name for p in (out / "flood_plots").iterdir()) == [
        "flood_event_1_AA_01.png", "flood_event_2_AA_01.png"]

    # Merge the two events by hand, then regenerate
    abbr.loc[0, "end_time_UTC"] = abbr.loc[1, "end_time_UTC"]
    abbr.iloc[[0]].assign(note="merged").to_csv(
        out / "abbr_flood_events.csv", index=False)
    tracker.regenerate_outputs_from_csv(output_dir=str(out))

    regenerated = pd.read_csv(out / "abbr_flood_events.csv")
    assert len(regenerated) == 1
    assert regenerated.loc[0, "max_road_water_level_(ft)"] == 0.5
    assert regenerated.loc[0, "note"] == "merged"  # extra columns kept
    detailed = pd.read_csv(out / "flood_events.csv")
    assert len(detailed) == 6 and set(detailed["flood_event"]) == {1}


def test_repeat_floods_in_one_combined_event_get_distinct_plots(tracker,
                                                                tmp_path):
    """Two floods of one sensor in one combined event both get plots."""
    raw = records({"AA_01": [-0.5, 0.1, 0.3, -0.1, 0.2, 0.5, -0.3, -0.5],
                   "BB_01": [-0.5, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, -0.5]})
    abbr = tmp_path / "abbr.csv"
    tracker._gen_abbr_flood_event_csv(raw, abbr)
    tracker._plot_and_save_flood_plots(raw, abbr, tmp_path / "plots")
    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == [
        "flood_event_1_AA_01.png", "flood_event_1_AA_01_2.png",
        "flood_event_1_BB_01.png"]


def test_flood_day_counts_depend_on_the_clock(tmp_path):
    """An event across UTC midnight spans two UTC days but one EST day."""
    csv = tmp_path / "abbr.csv"
    pd.DataFrame({
        "start_time_UTC": ["2024-01-01 22:00+00:00"],
        "end_time_UTC": ["2024-01-02 02:00+00:00"],
        "start_time_EST": ["2024-01-01 17:00-05:00"],
        "end_time_EST": ["2024-01-01 21:00-05:00"],
    }).to_csv(csv, index=False)
    assert EventTracker.num_of_flood_days("UTC", csv) == 2
    assert EventTracker.num_of_flood_days("EST", csv) == 1
    assert EventTracker.num_of_flood_days_by_start(csv) == 1
    assert EventTracker.num_of_flood_days("PST", csv) is None


def test_error_averages_use_the_week_before_each_event_end(tmp_path):
    """Sensor error statistics are averaged over the prior 7 days."""
    abbr, errors = tmp_path / "abbr.csv", tmp_path / "errors.csv"
    pd.DataFrame({"end_time_UTC": ["2024-01-10 00:00+00:00"]}).to_csv(
        abbr, index=False)
    pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-05", "2024-01-09"],
        "Mean": [100.0, 1.0, 3.0], "5th Pct": [0, 0, 0],
        "95th Pct": [2.0, 2.0, 4.0],
    }).to_csv(errors, index=False)
    assert EventTracker.avg_errors(str(errors), csv_filename=str(abbr))
    result = pd.read_csv(abbr)
    assert result.loc[0, "Mean of Mean"] == 2.0  # Jan 1 is outside
    assert result.loc[0, "Mean of 95th Pct"] == 3.0


# ----- API access ------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status=200, payload=None, bad_json=False):
        """
        Store the scripted reply.

        Parameters
        ----------
        status : int, optional
            HTTP status code. Default is 200.
        payload : list, optional
            JSON body.
        bad_json : bool, optional
            Make ``json()`` raise a decode error.
        """
        self.status_code = status
        self.payload = payload if payload is not None else []
        self.bad_json = bad_json

    def raise_for_status(self):
        """
        Raise ``HTTPError`` for 4xx/5xx statuses.

        Raises
        ------
        requests.exceptions.HTTPError
            If the status is an error.
        """
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        """
        Return the payload or raise a JSON decode error.

        Returns
        -------
        list
            The payload.

        Raises
        ------
        requests.exceptions.JSONDecodeError
            If constructed with ``bad_json``.
        """
        if self.bad_json:
            raise requests.exceptions.JSONDecodeError("bad", "doc", 0)
        return self.payload


class FakeSession:
    """Session returning scripted responses and logging requests."""

    def __init__(self, replies):
        """
        Store replies to return in order.

        Parameters
        ----------
        replies : list or callable
            Responses (or exceptions) in order, or a function of the
            request params returning one.
        """
        self.replies = replies
        self.calls = []
        self.auth = None

    def __enter__(self):
        """
        Enter the ``with`` block.

        Returns
        -------
        FakeSession
            This session.
        """
        return self

    def __exit__(self, *exc_info):
        """
        Leave the ``with`` block without suppressing errors.

        Parameters
        ----------
        *exc_info
            Exception details, ignored.

        Returns
        -------
        bool
            False, so exceptions propagate.
        """
        return False

    def get(self, url, params=None, timeout=None):
        """
        Return (or raise) the next scripted reply.

        Parameters
        ----------
        url : str
            Ignored.
        params : dict, optional
            Query parameters, recorded in ``calls``.
        timeout : float, optional
            Ignored.

        Returns
        -------
        FakeResponse
            The scripted response.
        """
        self.calls.append(params)
        reply = (self.replies(params) if callable(self.replies)
                 else self.replies[len(self.calls) - 1])
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """
    Make retry backoff instantaneous.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's attribute patcher.
    """
    monkeypatch.setattr(et_module.time, "sleep", lambda seconds: None)


def test_server_errors_are_retried_until_success(tracker):
    """5xx responses and dropped connections are retried."""
    session = FakeSession([FakeResponse(503),
                           requests.exceptions.ConnectionError(),
                           FakeResponse(200, [{"x": 1}])])
    assert tracker._fetch_data(session, dt.date(2024, 1, 1),
                               dt.date(2024, 1, 2), "AA_01") == [{"x": 1}]
    assert len(session.calls) == 3
    assert session.calls[0] == {"min_date": "2024-01-01",
                                "max_date": "2024-01-02",
                                "sensor_ID": "AA_01"}


@pytest.mark.parametrize("reply", [FakeResponse(404),
                                   FakeResponse(200, bad_json=True)])
def test_client_errors_and_bad_json_are_not_retried(tracker, reply):
    """4xx and malformed JSON fail immediately."""
    session = FakeSession([reply] * 3)
    assert tracker._fetch_data(session, dt.date(2024, 1, 1),
                               dt.date(2024, 1, 2), "AA_01") is None
    assert len(session.calls) == 1


def test_date_chunks_cover_the_window_inclusively(tracker):
    """Chunks tile the window and the last one is truncated."""
    chunks = tracker._create_date_chunks(3)
    assert chunks == [(dt.date(2024, 1, 1), dt.date(2024, 1, 3)),
                      (dt.date(2024, 1, 4), dt.date(2024, 1, 6)),
                      (dt.date(2024, 1, 7), dt.date(2024, 1, 7))]


def test_location_names_expand_to_sensor_ids(tracker):
    """Location names are case-insensitive; unknown names give None."""
    assert tracker._resolve_sensor_ids("Down East") == [
        "DE_01", "DE_02", "DE_03", "DE_04"]
    assert tracker._resolve_sensor_ids(["XX_01"]) == ["XX_01"]
    assert tracker._resolve_sensor_ids("Atlantis") is None
    assert tracker.get_data("Atlantis") is None


def test_failed_chunks_fall_back_to_daily_requests(tracker, monkeypatch):
    """A failing multi-day chunk is re-fetched one day at a time."""
    def reply(params):
        """
        Fail multi-day requests; serve one reading per day.

        Parameters
        ----------
        params : dict
            Query parameters.

        Returns
        -------
        FakeResponse
            503 for chunks, one record for single days.
        """
        start = dt.date.fromisoformat(params["min_date"])
        end = dt.date.fromisoformat(params["max_date"])
        if (end - start).days > 1:
            return FakeResponse(503)
        return FakeResponse(200, [{
            "sensor_ID": params["sensor_ID"], "date": f"{start}T12:00:00Z",
            "date_surveyed": "2024-01-01T00:00:00Z",
            "road_water_level_adj": 0.0, "sensor_water_level_adj": 3.0}])

    session = FakeSession(reply)
    monkeypatch.setattr(et_module.requests, "Session", lambda: session)
    data = tracker.get_data(["AA_01"], chunk_days=7)
    assert len(data) == 7  # one reading for each day of the window
    assert data["date"].is_monotonic_increasing
    assert len(session.calls) == 3 + 7  # 3 chunk attempts + 7 days


def test_constructor_validates_inputs():
    """Missing credentials or reversed dates are rejected."""
    with pytest.raises(ValueError):
        EventTracker(("user", ""), dt.date(2024, 1, 1), dt.date(2024, 1, 2))
    with pytest.raises(ValueError):
        EventTracker(("u", "p"), dt.date(2024, 1, 2), dt.date(2024, 1, 1))
