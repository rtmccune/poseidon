"""Download Sunny Day Flooding Project data and summarize flood events.

:class:`EventTracker` pulls water level records from the SDFP API in
parallel date chunks and turns them into:

* ``abbr_flood_events.csv`` - one row per flood event (start, end,
  duration, peak level), numbered so that events overlapping in time
  across sensors share a number.
* ``flood_events.csv`` - every observation inside each event, with the
  event summary repeated on the event's last row.
* ``sensor_outages.csv`` - data gaps longer than one hour.
* One plot per event.

A flood event starts when the road water level rises above
:data:`FLOOD_START_FT` and ends at the first reading below
:data:`FLOOD_END_FT`. Events still in progress when a sensor's record
ends are not reported. Levels are in feet relative to the road
surface; ``_EST`` columns use a fixed UTC-5 offset (see
:mod:`poseidon_utils.constants`).
"""

import datetime
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import IncompleteRead

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests

from .constants import EST, FEET_TO_METERS

logger = logging.getLogger(__name__)

#: Road water level (ft) above which a flood event starts.
FLOOD_START_FT = 0.02

#: Road water level (ft) below which a flood event ends.
FLOOD_END_FT = 0.0

#: Data gaps longer than this are reported as sensor outages.
OUTAGE_GAP = pd.Timedelta(hours=1)

#: Sensor IDs for each named SDFP location.
LOCATION_SENSORS = {
    "carolina beach": ["CB_01", "CB_02", "CB_03"],
    "beaufort": ["BF_01"],
    "down east": ["DE_01", "DE_02", "DE_03", "DE_04"],
    "new bern": ["NB_01", "NB_02"],
    "north river": ["NR_01"],
}

#: Columns of the abbreviated (one row per event) CSV.
ABBR_COLUMNS = [
    "flood_event", "sensor_ID", "start_time_UTC", "end_time_UTC",
    "start_time_EST", "end_time_EST", "duration_(hours)",
    "max_road_water_level_(ft)", "max_road_water_level_(m)",
]

#: Event summary columns repeated on the last row of each detailed event.
SUMMARY_COLUMNS = ABBR_COLUMNS[2:]

#: Columns of the detailed (one row per observation) CSV.
DETAILED_COLUMNS = [
    "flood_event", "sensor_ID", "time_UTC", "time_EST", "road_water_level",
    "sensor_water_level", *SUMMARY_COLUMNS,
]


def _summarize_event(sensor_id, start, end, peak_ft):
    """
    Build one abbreviated-CSV row for a detected flood event.

    Parameters
    ----------
    sensor_id : str
        Sensor ID.
    start, end : pandas.Timestamp
        Aware UTC event start and end.
    peak_ft : float
        Maximum road water level during the event (feet).

    Returns
    -------
    dict
        Row keyed by :data:`ABBR_COLUMNS`; ``flood_event`` is 0 until
        events are numbered.
    """
    return {
        "flood_event": 0,
        "sensor_ID": sensor_id,
        "start_time_UTC": start,
        "end_time_UTC": end,
        "start_time_EST": start.astimezone(EST),
        "end_time_EST": end.astimezone(EST),
        "duration_(hours)": round((end - start).total_seconds() / 3600, 3),
        "max_road_water_level_(ft)": round(peak_ft, 3),
        "max_road_water_level_(m)": round(peak_ft * FEET_TO_METERS, 3),
    }


def detect_flood_events(data):
    """
    Find flood events in raw SDFP water level records.

    Parameters
    ----------
    data : pandas.DataFrame
        Records with ``sensor_ID``, ``date`` (aware UTC) and
        ``road_water_level_adj`` (feet).

    Returns
    -------
    pandas.DataFrame
        One row per completed event with :data:`ABBR_COLUMNS`,
        not yet numbered.
    """
    events = []
    ordered = data.sort_values("date", kind="stable")
    for sensor_id, records in ordered.groupby("sensor_ID", sort=False):
        start, peak = None, 0.0  # flood state is per sensor
        for timestamp, level in zip(records["date"],
                                    records["road_water_level_adj"]):
            if start is None and level > FLOOD_START_FT:
                start, peak = timestamp, level
            if start is None:
                continue
            peak = max(peak, level)
            if level < FLOOD_END_FT:
                events.append(_summarize_event(sensor_id, start, timestamp,
                                               peak))
                start, peak = None, 0.0
    return pd.DataFrame(events, columns=ABBR_COLUMNS)


def number_flood_events(events):
    """
    Sort events by start time and number overlapping ones together.

    Events whose start precedes the latest end of the events before
    them (across all sensors) share that combined event's number.

    Parameters
    ----------
    events : pandas.DataFrame
        Abbreviated events with ``start_time_UTC`` and ``end_time_UTC``.

    Returns
    -------
    pandas.DataFrame
        Copy sorted by start time with a 1-based ``flood_event``.
    """
    events = events.copy()
    for column in ("start_time_UTC", "end_time_UTC"):
        events[column] = pd.to_datetime(events[column], utc=True)
    events = events.sort_values("start_time_UTC", kind="stable")
    events = events.reset_index(drop=True)
    if events.empty:
        events["flood_event"] = pd.Series(dtype=int)
        return events

    latest_prior_end = events["end_time_UTC"].cummax().shift(1)
    starts_new = (events["start_time_UTC"] >= latest_prior_end)
    starts_new.iloc[0] = True
    events["flood_event"] = starts_new.cumsum().astype(int)
    return events


def build_detailed_events(observations, events):
    """
    Expand numbered events into one row per observation.

    Parameters
    ----------
    observations : pandas.DataFrame
        Records with ``sensor_ID``, ``time_UTC`` (aware UTC),
        ``road_water_level`` and ``sensor_water_level``.
    events : pandas.DataFrame
        Numbered abbreviated events.

    Returns
    -------
    pandas.DataFrame
        :data:`DETAILED_COLUMNS` rows sorted by event and time. Summary
        columns are filled only on each event's last row. Events with
        no observations are omitted.
    """
    pieces = []
    for event in events.to_dict("records"):
        inside = ((observations["sensor_ID"] == event["sensor_ID"])
                  & (observations["time_UTC"] >= event["start_time_UTC"])
                  & (observations["time_UTC"] <= event["end_time_UTC"]))
        rows = observations.loc[inside].sort_values("time_UTC").copy()
        if rows.empty:
            logger.warning("No observations for event %s (%s); omitted "
                           "from the detailed CSV.", event["flood_event"],
                           event["sensor_ID"])
            continue
        rows["flood_event"] = event["flood_event"]
        rows["time_EST"] = rows["time_UTC"].dt.tz_convert(EST)
        for column in SUMMARY_COLUMNS:
            rows[column] = None
            rows.iloc[-1, rows.columns.get_loc(column)] = event[column]
        pieces.append(rows[DETAILED_COLUMNS])

    if not pieces:
        return pd.DataFrame(columns=DETAILED_COLUMNS)
    detailed = pd.concat(pieces, ignore_index=True)
    return detailed.sort_values(["flood_event", "time_UTC"],
                                kind="stable").reset_index(drop=True)


def observations_from_raw(data):
    """
    Rename raw API columns to the detailed-CSV observation schema.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw records with ``date``, ``road_water_level_adj`` and
        ``sensor_water_level_adj``.

    Returns
    -------
    pandas.DataFrame
        Columns ``sensor_ID``, ``time_UTC``, ``road_water_level``,
        ``sensor_water_level``.
    """
    observations = data.rename(columns={
        "date": "time_UTC",
        "road_water_level_adj": "road_water_level",
        "sensor_water_level_adj": "sensor_water_level",
    })[["sensor_ID", "time_UTC", "road_water_level", "sensor_water_level"]]
    observations["time_UTC"] = pd.to_datetime(observations["time_UTC"],
                                              utc=True)
    return observations


def _read_csv_if_exists(path):
    """
    Read a CSV, returning None if it does not exist or is empty.

    Parameters
    ----------
    path : str
        CSV path.

    Returns
    -------
    pandas.DataFrame or None
        The table, or None.
    """
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    return None if table.empty else table


class EventTracker:
    """
    Client for the SDFP water level API with flood event summaries.

    Parameters
    ----------
    authorization : tuple of str
        ``(username, password)`` for the API.
    min_date, max_date : datetime.date
        First and last day (inclusive) of every query.
    max_workers : int, optional
        Concurrent request threads. Default is 4.
    timeout : int, optional
        Per-request timeout in seconds. Default is 60.

    Attributes
    ----------
    BASE_URL : str
        Water level API endpoint.
    """

    BASE_URL = (
        "https://api-sunnydayflood.apps.cloudapps.unc.edu/get_water_level"
    )

    def __init__(self, authorization, min_date, max_date, max_workers=4,
                 timeout=60):
        """
        Validate credentials and store the query window.

        Parameters
        ----------
        authorization : tuple of str
            ``(username, password)`` for the API.
        min_date, max_date : datetime.date
            First and last day (inclusive) of every query.
        max_workers : int, optional
            Concurrent request threads. Default is 4.
        timeout : int, optional
            Per-request timeout in seconds. Default is 60.

        Raises
        ------
        ValueError
            If credentials or dates are missing, or the dates are
            reversed.
        """
        if not authorization or not all(authorization):
            raise ValueError("authorization must be a (username, password) "
                             "tuple")
        if not min_date or not max_date:
            raise ValueError("min_date and max_date are required")
        if max_date < min_date:
            raise ValueError("max_date must not be before min_date")

        self.authorization = authorization
        self.min_date = min_date
        self.max_date = max_date
        self.max_workers = max_workers
        self.timeout = timeout
        logger.info("EventTracker for %s to %s (%d workers, %ds timeout)",
                    min_date, max_date, max_workers, timeout)

    # ----- Data retrieval ----------------------------------------------------

    def get_data(self, sensor_ids, chunk_days=30):
        """
        Download records for sensors over the query window.

        Each sensor's window is split into `chunk_days` requests run in
        parallel. Chunks that fail or come back empty are retried one
        day at a time.

        Parameters
        ----------
        sensor_ids : str or list of str
            Location name (e.g. ``"Down East"``) or sensor IDs.
        chunk_days : int, optional
            Days per request. Default is 30.

        Returns
        -------
        pandas.DataFrame or None
            Records sorted by date with duplicates removed, or None if
            nothing was retrieved.

        Raises
        ------
        ValueError
            If a location name is not recognized.
        """
        sensors = self._resolve_sensor_ids(sensor_ids)
        chunks = self._create_date_chunks(chunk_days)
        logger.info("Requesting %d sensors x %d chunks of <=%d days.",
                    len(sensors), len(chunks), chunk_days)

        tasks = [(sensor, start, end) for sensor in sensors
                 for start, end in chunks]
        with requests.Session() as session:
            session.auth = self.authorization
            frames, failed = self._run_fetches(session, tasks)
            if failed:
                daily = [(sensor, day, day) for sensor, start, end in failed
                         for day in pd.date_range(start, end).date]
                logger.info("Retrying %d failed chunks as %d daily "
                            "requests.", len(failed), len(daily))
                daily_frames, _ = self._run_fetches(session, daily)
                frames.extend(daily_frames)
        return self._process_and_combine_data(frames)

    def _run_fetches(self, session, tasks):
        """
        Run fetch tasks in a thread pool.

        Parameters
        ----------
        session : requests.Session
            Authenticated session shared by the threads.
        tasks : list of tuple
            ``(sensor_id, first_day, last_day)`` requests.

        Returns
        -------
        frames : list of pandas.DataFrame
            Non-empty results.
        failed : list of tuple
            Tasks that failed or returned no rows.
        """
        frames, failed = [], []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._fetch_range, session, *task):
                       task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    frame = future.result()
                except Exception:  # noqa: BLE001 - retried by the caller
                    logger.exception("Request %s raised", task)
                    frame = None
                # Empty results are retried too: large chunks have been
                # observed to come back empty from the API
                if frame is None or frame.empty:
                    failed.append(task)
                else:
                    frames.append(frame)
        return frames, failed

    def _resolve_sensor_ids(self, sensor_ids):
        """
        Expand a location name into its sensor IDs.

        Parameters
        ----------
        sensor_ids : str or list of str
            Location name or explicit sensor IDs.

        Returns
        -------
        list of str
            Sensor IDs.

        Raises
        ------
        ValueError
            If a location name is not in :data:`LOCATION_SENSORS`.
        """
        if not isinstance(sensor_ids, str):
            return list(sensor_ids)
        try:
            return list(LOCATION_SENSORS[sensor_ids.lower()])
        except KeyError:
            raise ValueError(f"Location name '{sensor_ids}' not recognized; "
                             f"expected one of {sorted(LOCATION_SENSORS)}"
                             ) from None

    def _create_date_chunks(self, chunk_days):
        """
        Split the query window into consecutive inclusive day ranges.

        Parameters
        ----------
        chunk_days : int
            Maximum days per chunk.

        Returns
        -------
        list of tuple of datetime.date
            ``(first_day, last_day)`` pairs covering the window.
        """
        chunks = []
        start = self.min_date
        while start <= self.max_date:
            end = min(start + datetime.timedelta(days=chunk_days - 1),
                      self.max_date)
            chunks.append((start, end))
            start += datetime.timedelta(days=chunk_days)
        return chunks

    def _fetch_data(self, session, start_date, end_date, sensor_id,
                    max_retries=3, backoff_factor=2.0):
        """
        Make one API request, retrying transient failures.

        Server (5xx) and connection errors are retried with a linear
        backoff; client (4xx) errors and malformed JSON are not.

        Parameters
        ----------
        session : requests.Session
            Authenticated session.
        start_date : datetime.date
            First day requested (inclusive).
        end_date : datetime.date
            Day after the last day requested (exclusive).
        sensor_id : str
            Sensor ID.
        max_retries : int, optional
            Attempts before giving up. Default is 3.
        backoff_factor : float, optional
            Sleep ``backoff_factor * attempt`` seconds between attempts.
            Default is 2.0.

        Returns
        -------
        list or dict or None
            Parsed JSON, or None if the request failed.
        """
        params = {
            "min_date": start_date.strftime("%Y-%m-%d"),
            "max_date": end_date.strftime("%Y-%m-%d"),
            "sensor_ID": sensor_id,
        }
        task = f"{sensor_id} {params['min_date']}..{params['max_date']}"

        for attempt in range(1, max_retries + 1):
            try:
                response = session.get(self.BASE_URL, params=params,
                                       timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as err:
                if not 500 <= err.response.status_code <= 599:
                    logger.error("Client error for %s: %s", task, err)
                    return None
                logger.warning("Server error for %s (attempt %d/%d): %s",
                               task, attempt, max_retries, err)
            except requests.exceptions.JSONDecodeError:
                # Must precede RequestException, its parent class
                logger.error("Malformed JSON for %s", task)
                return None
            except (requests.exceptions.RequestException,
                    IncompleteRead) as err:
                logger.warning("Request error for %s (attempt %d/%d): %s",
                               task, attempt, max_retries, err)
            if attempt < max_retries:
                time.sleep(backoff_factor * attempt)

        logger.error("Giving up on %s after %d attempts.", task, max_retries)
        return None

    def _fetch_range(self, session, sensor_id, first_day, last_day):
        """
        Fetch one sensor's records for an inclusive range of days.

        Parameters
        ----------
        session : requests.Session
            Authenticated session.
        sensor_id : str
            Sensor ID.
        first_day, last_day : datetime.date
            Inclusive day range.

        Returns
        -------
        pandas.DataFrame or None
            Records with parsed ``date`` and ``date_surveyed`` columns
            (empty if the API returned no rows), or None on failure.
        """
        payload = self._fetch_data(session, first_day,
                                   last_day + datetime.timedelta(days=1),
                                   sensor_id)
        if payload is None:
            return None
        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame
        try:
            for column in ("date", "date_surveyed"):
                frame[column] = pd.to_datetime(frame[column], utc=True)
        except KeyError as err:
            logger.error("Response for %s lacks column %s", sensor_id, err)
            return None
        return frame

    @staticmethod
    def _process_and_combine_data(frames):
        """
        Concatenate fetched chunks, drop duplicates, and sort by date.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            Fetched records.

        Returns
        -------
        pandas.DataFrame or None
            Combined records, or None if `frames` is empty.
        """
        if not frames:
            logger.error("No data was retrieved for any sensor.")
            return None
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "sensor_ID"])
        combined = combined.sort_values("date", kind="stable")
        logger.info("Retrieved %d records.", len(combined))
        return combined.reset_index(drop=True)

    # ----- Pipelines ---------------------------------------------------------

    def pull_data_gen_csvs_and_plots(
            self, location, chunk_days=30, output_dir="data",
            full_csv_name="flood_events.csv",
            abbr_csv_name="abbr_flood_events.csv",
            outage_csv_name="sensor_outages.csv", plot_folder="flood_plots",
            raw_data_cache_name="raw_sensor_data.parquet"):
        """
        Download data and write event CSVs, outages, and plots.

        New events are merged into existing CSVs in `output_dir`;
        rows already present (same sensor and start time) are kept as
        they are, preserving manual edits.

        Parameters
        ----------
        location : str or list of str
            Location name or sensor IDs.
        chunk_days : int, optional
            Days per request. Default is 30.
        output_dir : str, optional
            Folder for all outputs. Default is ``"data"``.
        full_csv_name : str, optional
            Detailed CSV name. Default ``"flood_events.csv"``.
        abbr_csv_name : str, optional
            Abbreviated CSV name. Default ``"abbr_flood_events.csv"``.
        outage_csv_name : str, optional
            Outage CSV name. Default ``"sensor_outages.csv"``.
        plot_folder : str, optional
            Plot folder name. Default ``"flood_plots"``.
        raw_data_cache_name : str, optional
            Raw data cache name. Default ``"raw_sensor_data.parquet"``.

        Returns
        -------
        pandas.DataFrame or None
            The downloaded records, or None if the download failed.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {key: os.path.join(output_dir, name) for key, name in (
            ("full", full_csv_name), ("abbr", abbr_csv_name),
            ("outage", outage_csv_name), ("plots", plot_folder),
            ("raw", raw_data_cache_name))}

        data = self.get_data(location, chunk_days)
        if data is None or data.empty:
            logger.error("Download returned no data; nothing generated.")
            return None

        self._cache_raw_data(data, paths["raw"])
        events = self._gen_abbr_flood_event_csv(data, paths["abbr"])
        if events is not None:
            self._gen_flood_tracker(data, events, paths["full"])
        if self._find_outages(data, paths["outage"]) is not None:
            self._check_for_outage_during_flood(paths["outage"],
                                                paths["abbr"])
        self._plot_and_save_flood_plots(data, paths["abbr"], paths["plots"])
        return data

    def regenerate_outputs_from_csv(
            self, output_dir="data",
            raw_data_cache_name="raw_sensor_data.parquet",
            abbr_csv_name="abbr_flood_events.csv",
            full_csv_name="flood_events.csv",
            outage_csv_name="sensor_outages.csv", plot_folder="flood_plots"):
        """
        Rebuild outputs after manually editing the abbreviated CSV.

        Recomputes each event's duration, EST times, and peak level
        from the cached raw data, renumbers events, rewrites the
        detailed CSV, re-flags outages, and redraws every plot.

        Parameters
        ----------
        output_dir : str, optional
            Folder holding the inputs and outputs. Default ``"data"``.
        raw_data_cache_name : str, optional
            Raw data cache name. Default ``"raw_sensor_data.parquet"``.
        abbr_csv_name : str, optional
            Edited abbreviated CSV. Default ``"abbr_flood_events.csv"``.
        full_csv_name : str, optional
            Detailed CSV to overwrite. Default ``"flood_events.csv"``.
        outage_csv_name : str, optional
            Outage CSV to update. Default ``"sensor_outages.csv"``.
        plot_folder : str, optional
            Plot folder. Default ``"flood_plots"``.

        Raises
        ------
        FileNotFoundError
            If the abbreviated CSV or the raw data cache is missing.
        """
        abbr_path = os.path.join(output_dir, abbr_csv_name)
        raw = self._load_raw_cache(os.path.join(output_dir,
                                                raw_data_cache_name))
        events = pd.read_csv(abbr_path)

        events = number_flood_events(self._recalculate_event_summaries(
            number_flood_events(events), raw))
        events.to_csv(abbr_path, index=False)
        logger.info("Recalculated %d events in %s", len(events), abbr_path)

        detailed = build_detailed_events(observations_from_raw(raw), events)
        detailed.to_csv(os.path.join(output_dir, full_csv_name), index=False)

        outage_path = os.path.join(output_dir, outage_csv_name)
        if os.path.exists(outage_path):
            self._check_for_outage_during_flood(outage_path, abbr_path)
        self._plot_and_save_flood_plots(
            raw, abbr_path, os.path.join(output_dir, plot_folder),
            force_overwrite=True)

    @staticmethod
    def _cache_raw_data(data, path):
        """
        Save downloaded records to Parquet, falling back to CSV.

        Parameters
        ----------
        data : pandas.DataFrame
            Records to cache.
        path : str
            Parquet path; ``raw_sensor_data.csv`` is used beside it if
            no Parquet engine is installed.

        Returns
        -------
        str
            Path written.
        """
        try:
            data.to_parquet(path, index=False)
        except ImportError:
            path = os.path.join(os.path.dirname(path), "raw_sensor_data.csv")
            logger.warning("No Parquet engine; caching raw data as CSV.")
            data.to_csv(path, index=False)
        logger.info("Cached raw data to %s", path)
        return path

    @staticmethod
    def _load_raw_cache(path):
        """
        Load the raw data cache, trying the CSV fallback if needed.

        Parameters
        ----------
        path : str
            Parquet (or CSV) cache path.

        Returns
        -------
        pandas.DataFrame
            Raw records with parsed ``date``.

        Raises
        ------
        FileNotFoundError
            If neither the cache nor its CSV fallback exists.
        """
        fallback = os.path.join(os.path.dirname(path), "raw_sensor_data.csv")
        if os.path.exists(path):
            raw = (pd.read_parquet(path) if path.endswith(".parquet")
                   else pd.read_csv(path))
        elif os.path.exists(fallback):
            raw = pd.read_csv(fallback)
        else:
            raise FileNotFoundError(
                f"Raw data cache not found at {path} or {fallback}; run "
                f"pull_data_gen_csvs_and_plots first.")
        raw["date"] = pd.to_datetime(raw["date"], utc=True)
        return raw

    # ----- Event tables ------------------------------------------------------

    @staticmethod
    def _recalculate_event_summaries(events, raw):
        """
        Recompute event durations, EST times, and peaks from raw data.

        Parameters
        ----------
        events : pandas.DataFrame
            Abbreviated events with aware UTC start/end times.
        raw : pandas.DataFrame
            Raw records with parsed ``date``.

        Returns
        -------
        pandas.DataFrame
            Copy with summary columns recomputed and any extra columns
            kept; events without raw data get a peak of 0.
        """
        updated = events.copy().astype({c: object for c in SUMMARY_COLUMNS
                                        if c in events})
        for index, event in events.iterrows():
            inside = ((raw["sensor_ID"] == event["sensor_ID"])
                      & (raw["date"] >= event["start_time_UTC"])
                      & (raw["date"] <= event["end_time_UTC"]))
            peak = raw.loc[inside, "road_water_level_adj"].max()
            summary = _summarize_event(
                event["sensor_ID"], event["start_time_UTC"],
                event["end_time_UTC"], 0.0 if pd.isna(peak) else peak)
            for column in SUMMARY_COLUMNS:
                updated.at[index, column] = summary[column]
        return updated

    def _gen_abbr_flood_event_csv(self, data,
                                  csv_filename="abbr_flood_events.csv"):
        """
        Detect events and merge them into the abbreviated CSV.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw records.
        csv_filename : str, optional
            CSV to create or update. Default ``"abbr_flood_events.csv"``.

        Returns
        -------
        pandas.DataFrame or None
            Numbered events written to the CSV, or None if there are
            none.
        """
        events = detect_flood_events(data)
        existing = _read_csv_if_exists(csv_filename)
        if existing is not None:
            existing["start_time_UTC"] = pd.to_datetime(
                existing["start_time_UTC"], utc=True)
            # Existing rows first so manual edits win the de-duplication
            events = pd.concat([existing, events], ignore_index=True)
            events = events.drop_duplicates(
                subset=["sensor_ID", "start_time_UTC"], keep="first")
        if events.empty:
            logger.info("No flood events found.")
            return None

        events = number_flood_events(events)
        events.to_csv(csv_filename, index=False)
        logger.info("Wrote %d events to %s", len(events), csv_filename)
        return events

    def _gen_flood_tracker(self, data, events,
                           csv_filename="flood_events.csv"):
        """
        Write the detailed CSV for numbered events.

        Observations come from `data` plus any rows already in the CSV,
        so events found in earlier downloads keep their records.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw records.
        events : pandas.DataFrame
            Numbered abbreviated events.
        csv_filename : str, optional
            CSV to create or update. Default ``"flood_events.csv"``.

        Returns
        -------
        pandas.DataFrame
            The detailed table written.
        """
        observations = observations_from_raw(data)
        existing = _read_csv_if_exists(csv_filename)
        if existing is not None:
            previous = existing[observations.columns].copy()
            previous["time_UTC"] = pd.to_datetime(previous["time_UTC"],
                                                  utc=True)
            observations = pd.concat([previous, observations],
                                     ignore_index=True).drop_duplicates(
                subset=["sensor_ID", "time_UTC"], keep="first")

        detailed = build_detailed_events(observations, events)
        detailed.to_csv(csv_filename, index=False)
        logger.info("Wrote %d event observations to %s", len(detailed),
                    csv_filename)
        return detailed

    # ----- Outages -----------------------------------------------------------

    def _find_outages(self, dataframe, csv_filename="sensor_outages.csv"):
        """
        Record gaps longer than :data:`OUTAGE_GAP` in each sensor's data.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Raw records with ``sensor_ID`` and ``date``.
        csv_filename : str, optional
            Output CSV. Default ``"sensor_outages.csv"``.

        Returns
        -------
        pandas.DataFrame or None
            Numbered outages, or None if there are none.
        """
        outages = []
        for sensor_id, records in dataframe.groupby("sensor_ID", sort=False):
            times = records["date"].sort_values().reset_index(drop=True)
            for i in np.flatnonzero(times.diff() > OUTAGE_GAP):
                start, end = times[i - 1], times[i]
                outages.append({
                    "outage_number": None,
                    "sensor_ID": sensor_id,
                    "start_time_UTC": start,
                    "end_time_UTC": end,
                    "start_time_EST": start.astimezone(EST),
                    "end_time_EST": end.astimezone(EST),
                    "duration_(hours)": round(
                        (end - start).total_seconds() / 3600, 3),
                })
        if not outages:
            logger.info("No sensor outages found.")
            return None

        table = pd.DataFrame(outages).sort_values("start_time_UTC",
                                                  kind="stable")
        table["outage_number"] = range(1, len(table) + 1)
        table.to_csv(csv_filename, index=False)
        return table

    def _check_for_outage_during_flood(
            self, outage_csv="sensor_outages.csv",
            abbr_flood_csv="abbr_flood_events.csv"):
        """
        Flag outages that begin during a flood event of the same sensor.

        Adds ``during_flood_event``/``flood_event_number`` to the outage
        CSV and an ``outage`` column to the abbreviated CSV, then saves
        both.

        Parameters
        ----------
        outage_csv : str, optional
            Outage CSV. Default ``"sensor_outages.csv"``.
        abbr_flood_csv : str, optional
            Abbreviated CSV. Default ``"abbr_flood_events.csv"``.
        """
        floods = _read_csv_if_exists(abbr_flood_csv)
        outages = _read_csv_if_exists(outage_csv)
        if floods is None or outages is None:
            logger.info("No flood or outage data to compare.")
            return

        for table, columns in ((floods, ("start_time_UTC", "end_time_UTC")),
                               (outages, ("start_time_UTC",))):
            for column in columns:
                table[column] = pd.to_datetime(table[column], utc=True)

        outage_starts = outages[["outage_number", "sensor_ID",
                                 "start_time_UTC"]].rename(
            columns={"start_time_UTC": "outage_start"})
        flood_windows = floods[["flood_event", "sensor_ID", "start_time_UTC",
                                "end_time_UTC"]]
        merged = outage_starts.merge(flood_windows, on="sensor_ID")
        during = merged[(merged["outage_start"] >= merged["start_time_UTC"])
                        & (merged["outage_start"] <= merged["end_time_UTC"])]
        event_by_outage = (during.drop_duplicates("outage_number")
                           .set_index("outage_number")["flood_event"])

        flood_numbers = outages["outage_number"].map(event_by_outage)
        outages["during_flood_event"] = np.where(flood_numbers.notna(),
                                                 "Yes", "No")
        outages["flood_event_number"] = flood_numbers
        floods["outage"] = np.where(
            floods["flood_event"].isin(during["flood_event"]), "Yes", "No")

        floods.to_csv(abbr_flood_csv, index=False)
        outages.to_csv(outage_csv, index=False)

    # ----- Plots -------------------------------------------------------------

    def _plot_and_save_flood_plots(self, sunnyd_data,
                                   csv_filename="abbr_flood_events.csv",
                                   plot_folder="flood_plots",
                                   force_overwrite=False):
        """
        Plot each event's water level with one day of context per side.

        Parameters
        ----------
        sunnyd_data : pandas.DataFrame
            Raw records with ``date`` and ``road_water_level_adj``.
        csv_filename : str, optional
            Abbreviated CSV listing events. Default
            ``"abbr_flood_events.csv"``.
        plot_folder : str, optional
            Output folder. Default ``"flood_plots"``.
        force_overwrite : bool, optional
            Redraw plots that already exist. Default is False.
        """
        events = _read_csv_if_exists(csv_filename)
        if events is None:
            logger.info("No flood events to plot.")
            return
        os.makedirs(plot_folder, exist_ok=True)

        # A sensor can flood twice within one combined event; number the
        # repeats so their plots do not overwrite each other
        repeat = events.groupby(["flood_event", "sensor_ID"]).cumcount() + 1
        for event, k in zip(events.to_dict("records"), repeat):
            suffix = "" if k == 1 else f"_{k}"
            plot_path = os.path.join(
                plot_folder, f"flood_event_{event['flood_event']}_"
                             f"{event['sensor_ID']}{suffix}.png")
            if os.path.exists(plot_path) and not force_overwrite:
                continue

            start = pd.to_datetime(event["start_time_UTC"], utc=True)
            end = pd.to_datetime(event["end_time_UTC"], utc=True)
            window = sunnyd_data[
                (sunnyd_data["sensor_ID"] == event["sensor_ID"])
                & (sunnyd_data["date"] >= start - pd.Timedelta(days=1))
                & (sunnyd_data["date"] <= end + pd.Timedelta(days=1))]
            if window.empty:
                logger.warning("No raw data to plot event %s (%s).",
                               event["flood_event"], event["sensor_ID"])
                continue
            self._draw_event_plot(window, start, end, event, plot_path)

    @staticmethod
    def _draw_event_plot(window, start, end, event, plot_path):
        """
        Draw and save one flood event plot.

        Parameters
        ----------
        window : pandas.DataFrame
            Raw records spanning the plot.
        start, end : pandas.Timestamp
            Event start and end (aware UTC).
        event : dict
            Abbreviated CSV row of the event.
        plot_path : str
            Output PNG path.
        """
        in_event = (window["date"] >= start) & (window["date"] <= end)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(window["date"], window["road_water_level_adj"],
                label="Water Level", color="#427e93")
        ax.plot(window["date"], np.zeros(len(window)), label="Roadway",
                linewidth=2, color="black")
        ax.plot(window.loc[in_event, "date"],
                window.loc[in_event, "road_water_level_adj"], marker="x",
                markersize=5, linestyle="None", color="#cc0000",
                label="Flood Event Data Points")

        ax.set_ylim(-2.5, 2)
        ax.set_yticks(np.arange(-2.5, 2.25, 0.25))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(which="major", axis="y", linewidth=0.5, color="gray")
        ax.grid(which="minor", axis="y", linewidth=0.5, color="black")
        ax.set_xlabel("Date")
        ax.set_ylabel("Water Level (ft above road)")
        ax.set_title(f"Flood Event {event['flood_event']} for "
                     f"{event['sensor_ID']}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    # ----- Summary statistics ------------------------------------------------

    @staticmethod
    def num_of_flood_days_by_start(csv_name="abbr_flood_events.csv"):
        """
        Count distinct UTC calendar days on which floods started.

        Parameters
        ----------
        csv_name : str, optional
            Abbreviated CSV. Default ``"abbr_flood_events.csv"``.

        Returns
        -------
        int
            Number of distinct start days.

        Raises
        ------
        FileNotFoundError
            If `csv_name` does not exist.
        """
        events = pd.read_csv(csv_name)
        starts = pd.to_datetime(events["start_time_UTC"], utc=True)
        n_days = starts.dt.date.nunique()
        logger.info("Number of days of flooding: %d", n_days)
        return n_days

    @staticmethod
    def num_of_flood_days(timezone="EST", csv_name="abbr_flood_events.csv"):
        """
        Count distinct calendar days touched by any flood event.

        Every day from each event's start date through its end date is
        counted once.

        Parameters
        ----------
        timezone : {'EST', 'UTC'}, optional
            Clock used to assign days. Default is ``"EST"``.
        csv_name : str, optional
            Abbreviated CSV. Default ``"abbr_flood_events.csv"``.

        Returns
        -------
        int
            Number of distinct flood days.

        Raises
        ------
        ValueError
            If `timezone` is not ``"EST"`` or ``"UTC"``.
        FileNotFoundError
            If `csv_name` does not exist.
        """
        zones = {"EST": EST, "UTC": datetime.timezone.utc}
        if timezone not in zones:
            raise ValueError("timezone must be 'EST' or 'UTC'")
        events = pd.read_csv(csv_name)
        starts = pd.to_datetime(events["start_time_UTC"], utc=True)
        ends = pd.to_datetime(events["end_time_UTC"], utc=True)
        zone = zones[timezone]

        days = set()
        for start, end in zip(starts.dt.tz_convert(zone).dt.date,
                              ends.dt.tz_convert(zone).dt.date):
            days.update(pd.date_range(start, end).date)
        logger.info("Number of days of flooding: %d", len(days))
        return len(days)

    @staticmethod
    def avg_errors(error_csv, save_file_as=None,
                   csv_filename="abbr_flood_events.csv"):
        """
        Append 7-day mean sensor errors before each event's end.

        For each event, the ``Mean``, ``5th Pct`` and ``95th Pct``
        columns of `error_csv` are averaged over the week ending at
        the event's end time.

        Parameters
        ----------
        error_csv : str
            CSV with ``Date``, ``Mean``, ``5th Pct``, ``95th Pct``.
        save_file_as : str, optional
            Output path. Default overwrites `csv_filename`.
        csv_filename : str, optional
            Abbreviated CSV. Default ``"abbr_flood_events.csv"``.

        Returns
        -------
        pandas.DataFrame
            Events with ``Mean of Mean``, ``Mean of 5th Pct`` and
            ``Mean of 95th Pct`` columns added.

        Raises
        ------
        FileNotFoundError
            If either CSV does not exist.
        """
        events = pd.read_csv(csv_filename)
        errors = pd.read_csv(error_csv)
        events["end_time_UTC"] = pd.to_datetime(events["end_time_UTC"],
                                                utc=True)
        errors["Date"] = pd.to_datetime(errors["Date"], utc=True)

        columns = ["Mean", "5th Pct", "95th Pct"]
        averages = np.full((len(events), len(columns)), np.nan)
        for i, end in enumerate(events["end_time_UTC"]):
            week = errors[(errors["Date"] >= end - pd.Timedelta(days=7))
                          & (errors["Date"] <= end)]
            averages[i] = week[columns].mean().round(4).to_numpy()

        events[[f"Mean of {c}" for c in columns]] = averages
        save_path = save_file_as or csv_filename
        events.to_csv(save_path, index=False)
        logger.info("Saved event error averages to %s", save_path)
        return events
