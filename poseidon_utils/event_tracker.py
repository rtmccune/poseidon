import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from http.client import IncompleteRead
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytz
import requests


class EventTracker:
    """
    A client for fetching data from the Sunny Day Flooding Project API
    with built-in flood event summary and plotting.

    Attributes
    ----------
    BASE_URL : str
        The base API endpoint for fetching water level data.
    authorization : Tuple[str, str]
        The (username, password) tuple for API authentication.
    min_date : datetime.date | datetime.datetime
        The overall start date for all data queries.
    max_date : datetime.date | datetime.datetime
        The overall end date for all data queries.
    max_workers : int
        The number of concurrent threads to use for fetching.
    timeout : int
        The request timeout in seconds.
    """

    BASE_URL = (
        "https://api-sunnydayflood.apps.cloudapps.unc.edu/get_water_level"
    )

    def __init__(
        self,
        authorization: Tuple[str, str],
        min_date: datetime.date | datetime.datetime,
        max_date: datetime.date | datetime.datetime,
        max_workers: int = 4,
        timeout: int = 60,
    ):
        """
        Initializes the EventTracker.

        Parameters
        ----------
        authorization : Tuple[str, str]
            A (username, password) tuple for API authentication.
        min_date : datetime.date | datetime.datetime
            The overall start date for all data queries (inclusive).
        max_date : datetime.date | datetime.datetime
            The overall end date for all data queries (inclusive).
        max_workers : int, optional
            The number of concurrent threads to use for fetching, by
            default 16.
        timeout : int, optional
            The request timeout in seconds, by default 60.

        Raises
        ------
        ValueError
            If the authorization tuple is not provided or is incomplete.
        """
        if not authorization or not all(authorization):
            raise ValueError(
                "Authorization (username, password) tuple is required."
            )
        if not min_date or not max_date:
            raise ValueError("min_date and max_date are required.")

        self.authorization = authorization
        self.min_date = min_date
        self.max_date = max_date
        self.max_workers = max_workers
        self.timeout = timeout
        print(
            f"EventTracker initialized. Max workers: {self.max_workers}, Timeout: {self.timeout}s"
        )
        print(f"Date Range Set: {self.min_date} to {self.max_date}")

    def get_data(
        self, sensor_ids: str | List[str], chunk_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Retrieves data using a chunking method and a robust fallback.

        Parameters
        ----------
        sensor_ids : str | List[str]
            A single location name (e.g., "Down East") or a list of
            specific sensor ID strings.
        chunk_days : int, optional
            The number of days to fetch in each parallel request, by
            default 30.

        Returns
        -------
        Optional[pd.DataFrame]
            A combined, deduplicated, and sorted pandas DataFrame
            containing all successfully retrieved data, or None if no
            data could be fetched.
        """

        # Input validation and setup
        sensor_ids_list = self._resolve_sensor_ids(sensor_ids)
        if sensor_ids_list is None:
            return None  # Error already printed by helper

        # Create chunks
        date_chunks = self._create_date_chunks(chunk_days)
        if not date_chunks:
            print("No date chunks to process (check min/max dates).")
            return None

        total_tasks = len(sensor_ids_list) * len(date_chunks)
        print(
            f"Total date range split into {len(date_chunks)} chunks of ~{chunk_days} days each."
        )
        print(
            f"Starting chunked download for {len(sensor_ids_list)} sensors... ({total_tasks} total requests)"
        )

        all_dataframes = []

        # Create Session, Executor, and Run Tasks
        with requests.Session() as session:
            session.auth = self.authorization

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

                # Submit Primary (Chunked) Tasks
                futures = self._submit_chunk_tasks(
                    executor, session, sensor_ids_list, date_chunks
                )

                # Process Primary Results
                primary_data, failed_tasks = self._process_chunk_results(
                    futures
                )
                all_dataframes.extend(primary_data)

            # Handle Failed Tasks (The Fallback)
            if failed_tasks:
                fallback_data = self._handle_failed_chunks(
                    session, failed_tasks
                )
                all_dataframes.extend(fallback_data)

        print("Data retrieval complete.")

        # Combine and Return Data
        return self._process_and_combine_data(all_dataframes)

    # === MODIFIED: Added raw_data_cache_name parameter and caching step ===
    def pull_data_gen_csvs_and_plots(
        self,
        location: str | List[str],
        chunk_days: int = 30,
        output_dir: str = "data",
        full_csv_name: str = "flood_events.csv",
        abbr_csv_name: str = "abbr_flood_events.csv",
        outage_csv_name: str = "sensor_outages.csv",
        plot_folder: str = "flood_plots",
        raw_data_cache_name: str = "raw_sensor_data.parquet",
    ) -> Optional[pd.DataFrame]:
        """
        Runs the full data pipeline: fetch, process, and output.

        Parameters
        ----------
        location : str | List[str]
            A single location name (e.g., "Down East") or a list of
            specific sensor ID strings.
        chunk_days : int, optional
            The number of days to fetch in each parallel request,
            by default 30.
        output_dir : str, optional
            Directory to save outputs of method to.
        full_csv_name : str, optional
            Filename for the detailed, row-by-row flood event CSV.
        abbr_csv_name : str, optional
            Filename for the abbreviated, one-row-per-event CSV.
        outage_csv_name : str, optional
            Filename for the sensor outage log CSV.
        plot_folder : str, optional
            Name of the directory to save flood plots to.
        raw_data_cache_name : str, optional
            Filename for caching the raw downloaded data (e.g.,
            'raw_data.parquet'). Default is 'raw_sensor_data.parquet'.

        Returns
        -------
        Optional[pd.DataFrame]
            The downloaded and combined DataFrame, or None if the
            data pull fails.
        """
        os.makedirs(output_dir, exist_ok=True)
        full_csv_path = os.path.join(output_dir, full_csv_name)
        abbr_csv_path = os.path.join(output_dir, abbr_csv_name)
        outage_csv_path = os.path.join(output_dir, outage_csv_name)
        plot_folder_path = os.path.join(output_dir, plot_folder)
        raw_data_path = os.path.join(output_dir, raw_data_cache_name)

        print(f"Starting data pull for location: '{location}'")
        download_data = self.get_data(location, chunk_days)

        if download_data is None or download_data.empty:
            print(
                "Data download failed or returned no data. Stopping processing."
            )
            return None

        print(f"Data pull successful. Shape: {download_data.shape}")

        # === NEW: Cache the raw data ===
        print(f"Caching raw data to {raw_data_path}...")
        try:
            # Use parquet for efficiency and type preservation
            download_data.to_parquet(raw_data_path, index=False)
            print("Raw data cached successfully.")
        except ImportError:
            # Fallback to CSV if pyarrow isn't installed
            print(
                "Warning: 'pyarrow' not installed. Falling back to CSV for cache."
            )
            raw_data_path = os.path.join(
                output_dir, "raw_sensor_data.csv"
            )
            download_data.to_csv(raw_data_path, index=False)
        except Exception as e:
            print(f"Failed to cache raw data: {e}")
            # Don't stop the whole process, just warn
        # === END NEW ===

        print("Generating CSVs...")
        try:
            self._gen_flood_tracker(download_data, full_csv_path)
            self._gen_abbr_flood_event_csv(download_data, abbr_csv_path)
            print("CSV files created successfully.")
        except Exception as e:
            print(f"An error occurred during CSV generation: {e}")

        print("Checking for sensor outages...")
        try:
            self._find_outages(download_data, outage_csv_path)
            self._check_for_outage_during_flood(
                outage_csv_path, abbr_csv_path
            )
            print("Sensor outage logs generated and CSVs appended.")
        except Exception as e:
            print(f"An error occurred during outage check: {e}")

        print("Plotting flood events...")
        try:
            # === MODIFIED: Pass the plot_folder_path ===
            self._plot_and_save_flood_plots(
                download_data, abbr_csv_path, plot_folder_path
            )
            print("Plotting completed.")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")

        return download_data

    # === NEW: Public method for regeneration ===
    def regenerate_outputs_from_csv(
        self,
        output_dir: str = "data",
        raw_data_cache_name: str = "raw_sensor_data.parquet",
        abbr_csv_name: str = "abbr_flood_events.csv",
        full_csv_name: str = "flood_events.csv",
        outage_csv_name: str = "sensor_outages.csv",
        plot_folder: str = "flood_plots",
    ):
        """
        Regenerates all outputs from a manually edited abbreviated
        flood event CSV.

        This method reads the specified `abbr_csv_name`,
        recalculates all summaries (duration, max levels),
        cleans/re-numbers it, and then uses it along with the
        `raw_data_cache_name` to regenerate the detailed
        `full_csv_name` and all `plot_folder` plots. It also
        re-runs the outage check.

        Parameters
        ----------
        output_dir : str, optional
            Directory where all input/output files are located.
        raw_data_cache_name : str, optional
            Filename of the cached raw data (e.g.,
            'raw_sensor_data.parquet').
        abbr_csv_name : str, optional
            Filename of the *manually edited* abbreviated flood event
            CSV.
        full_csv_name : str, optional
            Filename for the detailed, row-by-row flood event CSV
            (this file will be overwritten).
        outage_csv_name : str, optional
            Filename for the sensor outage log CSV (will be read and
            updated).
        plot_folder : str, optional
            Name of the directory to save new plots to (will
            overwrite existing plots).
        """
        print("--- Starting Regeneration Process ---")
        # Define all full paths
        full_csv_path = os.path.join(output_dir, full_csv_name)
        abbr_csv_path = os.path.join(output_dir, abbr_csv_name)
        outage_csv_path = os.path.join(output_dir, outage_csv_name)
        plot_folder_path = os.path.join(output_dir, plot_folder)
        raw_data_path = os.path.join(output_dir, raw_data_cache_name)

        # 1. Load Inputs
        try:
            print(f"Loading manually-edited events from: {abbr_csv_path}")
            abbr_df = pd.read_csv(abbr_csv_path)
        except FileNotFoundError:
            print(f"Error: Abbreviated events file not found at {abbr_csv_path}")
            return

        try:
            print(f"Loading cached raw data from: {raw_data_path}")
            if raw_data_path.endswith(".parquet"):
                raw_data_df = pd.read_parquet(raw_data_path)
            else:
                raw_data_df = pd.read_csv(raw_data_path)
        except FileNotFoundError:
            # Try CSV fallback
            csv_fallback_path = os.path.join(
                output_dir, "raw_sensor_data.csv"
            )
            try:
                print(
                    f"Parquet not found. Trying CSV fallback: {csv_fallback_path}"
                )
                raw_data_df = pd.read_csv(csv_fallback_path)
            except FileNotFoundError:
                print(
                    f"Error: Raw data cache not found at {raw_data_path} or {csv_fallback_path}"
                )
                print(
                    "Please run `pull_data_gen_csvs_and_plots` first to generate the cache."
                )
                return
        except Exception as e:
            print(f"Error loading raw data: {e}")
            return

        # === MODIFIED: Ensure all date columns are converted before processing ===
        try:
            raw_data_df["date"] = pd.to_datetime(raw_data_df["date"], utc=True)
            abbr_df["start_time_UTC"] = pd.to_datetime(
                abbr_df["start_time_UTC"], utc=True
            )
            abbr_df["end_time_UTC"] = pd.to_datetime(
                abbr_df["end_time_UTC"], utc=True
            )
        except Exception as e:
            print(f"Error converting date columns: {e}. Aborting.")
            return

        print("Raw data and event CSVs loaded successfully.")

        # 2. QC and Recalculate the Abbreviated CSV
        print("Recalculating event summaries (duration, max levels)...")
        # === NEW: Call the new helper ===
        try:
            abbr_df = self._recalculate_event_summaries(abbr_df, raw_data_df)
        except Exception as e:
            print(f"Error during recalculation: {e}")
            # Continue, but warn the user
            
        print("Re-numbering and cleaning flood events...")
        # This re-sorts by date and re-assigns sequential flood_event numbers
        abbr_df = self._reassign_abbr_flood_numbers(abbr_df)

        # Save the updated file
        abbr_df.to_csv(abbr_csv_path, index=False)
        print(f"Cleaned and recalculated event file saved to {abbr_csv_path}")

        # 3. Regenerate Detailed CSV
        print(f"Regenerating detailed event file: {full_csv_path}...")
        try:
            self._gen_flood_tracker_from_abbr(
                raw_data_df, abbr_df, full_csv_path
            )
        except Exception as e:
            print(f"Error during detailed CSV regeneration: {e}")
            return

        # 4. Re-check Outages
        print(f"Re-checking outages for {outage_csv_path}...")
        try:
            self._check_for_outage_during_flood(
                outage_csv_path, abbr_csv_path
            )
            print("Outage check complete.")
        except Exception as e:
            print(f"Error during outage check: {e}")
            # Don't stop, just warn

        # 5. Regenerate Plots
        print(f"Regenerating plots in {plot_folder_path} (forcing overwrite)...")
        try:
            self._plot_and_save_flood_plots(
                raw_data_df,
                abbr_csv_path,
                plot_folder_path,
                force_overwrite=True,
            )
            print("Plotting completed.")
        except Exception as e:
            print(f"An error occurred during plotting: {e}")

        print("--- Regeneration Complete ---")
    
    def _sensor_list_generator(self, location_name: str) -> List[str]:
        """
        Provides a sensor identity list for a given location.

        Parameters
        ----------
        location_name : str
            Name of sensor location.

        Returns
        -------
        List[str]
            List of sensor ids as strings.

        Raises
        ------
        ValueError
            If the location_name is not recognized.
        """

        if location_name.lower() == "carolina beach":
            sensor_ids = ["CB_01", "CB_02", "CB_03"]
        elif location_name.lower() == "beaufort":
            sensor_ids = ["BF_01"]
        elif location_name.lower() == "down east":
            sensor_ids = ["DE_01", "DE_02", "DE_03", "DE_04"]
        elif location_name.lower() == "new bern":
            sensor_ids = ["NB_01", "NB_02"]
        elif location_name.lower() == "north river":
            sensor_ids = ["NR_01"]
        else:
            raise ValueError(f"Location name '{location_name}' not recognized.")
        return sensor_ids

    def _resolve_sensor_ids(
        self, sensor_ids_input: str | List[str]
    ) -> Optional[List[str]]:
        """
        Resolves a potential location name string into a list of sensor
        IDs.

        If input is a list, it is returned as-is.
        If input is a string, it's passed to `_sensor_list_generator`.

        Parameters
        ----------
        sensor_ids_input : str | List[str]
            The user-provided sensor input.

        Returns
        -------
        Optional[List[str]]
            A list of sensor ID strings, or None if a string location
            name is not recognized.
        """
        if isinstance(sensor_ids_input, str):
            try:
                print(f"Resolving location name: '{sensor_ids_input}'")
                return self._sensor_list_generator(sensor_ids_input)
            except ValueError as e:
                print(f"Error: {e}")
                print("Stopping data retrieval.")
                return None

        # If not a string, assume it's a List[str] and return it
        return sensor_ids_input

    def _create_date_chunks(
        self, chunk_days: int
    ) -> List[Tuple[datetime.date, datetime.date]]:
        """
        Splits the instance's date range (self.min_date to self.max_date)
        into n-day chunks.

        Parameters
        ----------
        chunk_days : int
            The maximum number of days for each chunk.

        Returns
        -------
        List[Tuple[datetime.date, datetime.date]]
            A list of (start_date, end_date) tuples.
        """
        date_chunks = []
        current_start_date = self.min_date

        # While current time is before end date
        while current_start_date <= self.max_date:
            current_end_date = current_start_date + datetime.timedelta(
                days=chunk_days - 1
            )
            if current_end_date > self.max_date:
                current_end_date = self.max_date
            date_chunks.append((current_start_date, current_end_date))
            current_start_date += datetime.timedelta(days=chunk_days)
        return date_chunks

    def _fetch_data(
        self,
        session: requests.Session,
        start_date: datetime.date | datetime.datetime,
        end_date: datetime.date | datetime.datetime,
        sensor_id: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> Optional[Dict[str, Any] | List[Any]]:
        """
        Performs a single, raw API request with robust retries.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        start_date : datetime.date | datetime.datetime
            The start date for the API query (inclusive).
        end_date : datetime.date | datetime.datetime
            The end date for the API query (exclusive).
        sensor_id : str
            The specific sensor ID to query.
        max_retries : int, optional
            The maximum number of retry attempts, by default 3.
        backoff_factor : float, optional
            The factor to determine sleep time (sleep = backoff * (attempt + 1)),
            by default 2.0.

        Returns
        -------
        Optional[Dict[str, Any] | List[Any]]
            The raw JSON response from the API or None if the request
            fails after all retries.
        """

        query_params = {
            "min_date": start_date.strftime("%Y-%m-%d"),
            "max_date": end_date.strftime("%Y-%m-%d"),
            "sensor_ID": sensor_id,
        }

        task_id = f"sensor {sensor_id} ({query_params['min_date']} to {query_params['max_date']})"

        for attempt in range(max_retries):
            try:
                response = session.get(
                    self.BASE_URL, params=query_params, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as http_err:
                # Only retry on 5xx server errors
                if 500 <= http_err.response.status_code <= 599:
                    print(
                        f"HTTP error for {task_id}: {http_err}. "
                        f"Attempt {attempt + 1} of {max_retries}."
                    )
                else:
                    # Don't retry on 4xx client errors (e.g., 404, 401)
                    print(
                        f"HTTP client error for {task_id}: {http_err}. Not retrying."
                    )
                    return None  # Failed permanently

            except (
                requests.exceptions.RequestException,  # Catches all connection/timeout errors
                IncompleteRead,  # Catches the specific connection drop
            ) as req_err:
                # These cover IncompleteRead and other connection drops
                print(
                    f"Request error for {task_id}: {req_err}. "
                    f"Attempt {attempt + 1} of {max_retries}."
                )

            except requests.exceptions.JSONDecodeError:
                print(
                    f"Error: Server returned OK but failed to decode JSON for {task_id}."
                )
                # This is a server error, but retrying may not help
                # if it's consistently sending bad JSON.
                return None

            # If we are not on the last attempt, sleep before retrying
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (attempt + 1)
                print(f"Waiting {sleep_time}s before next retry...")
                time.sleep(sleep_time)

        print(f"All {max_retries} retries failed for {task_id}.")
        return None

    def _fetch_chunk(
        self,
        session: requests.Session,
        sensor_id: str,
        start_date: datetime.date | datetime.datetime,
        end_date: datetime.date | datetime.datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and processes data for a single sensor over a date range
        chunk.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        sensor_id : str
            The sensor ID for the chunk.
        start_date : datetime.date | datetime.datetime
            The first day of the chunk (inclusive).
        end_date : datetime.date | datetime.datetime
            The last day of the chunk (inclusive).

        Returns
        -------
        Optional[pd.DataFrame]
            A processed DataFrame for the chunk, or None if processing
            fails.
        """

        exclusive_end_date = end_date + datetime.timedelta(days=1)

        json_data = self._fetch_data(
            session=session,
            start_date=start_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        if not json_data:
            return None

        try:
            df = pd.DataFrame(json_data)
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        except KeyError as e:
            print(
                f"Error processing DataFrame for {sensor_id}: Missing expected column {e}."
            )
        except Exception as e:
            print(
                f"An unexpected error occurred processing data for {sensor_id}: {e}"
            )

        return None

    def _fetch_day(
        self,
        session: requests.Session,
        sensor_id: str,
        target_date: datetime.date,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches and processes data for a single sensor for a single day.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        sensor_id : str
            The sensor ID for the request.
        target_date : datetime.date
            The specific day to fetch data for (inclusive).

        Returns
        -------
        Optional[pd.DataFrame]
            A processed DataFrame for the single day, or None if
            processing fails.
        """

        exclusive_end_date = target_date + datetime.timedelta(days=1)

        json_data = self._fetch_data(
            session=session,
            start_date=target_date,
            end_date=exclusive_end_date,
            sensor_id=sensor_id,
        )

        if not json_data:
            return None

        try:
            df = pd.DataFrame(json_data)
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["date_surveyed"] = pd.to_datetime(df["date_surveyed"], utc=True)
            return df

        except KeyError as e:
            print(
                f"Error processing DataFrame for {sensor_id} on {target_date}: Missing column {e}."
            )
        except Exception as e:
            print(
                f"Unexpected error processing data for {sensor_id} on {target_date}: {e}"
            )

        return None

    def _submit_chunk_tasks(
        self,
        executor: ThreadPoolExecutor,
        session: requests.Session,
        sensor_ids: List[str],
        date_chunks: List[Tuple[datetime.date, datetime.date]],
    ) -> Dict[Future, Tuple[str, datetime.date, datetime.date]]:
        """
        Submits all sensor/date_chunk tasks to the executor.

        Parameters
        ----------
        executor : ThreadPoolExecutor
            The executor instance to submit tasks to.
        session : requests.Session
            The authenticated session to pass to the task.
        sensor_ids : List[str]
            The list of sensor IDs to fetch.
        date_chunks : List[Tuple[datetime.date, datetime.date]]
            The list of date range tuples.

        Returns
        -------
        Dict[Future, Tuple[str, datetime.date, datetime.date]]
            A dictionary mapping the submitted Future objects to a
            tuple of (sensor_id, start_chunk, end_chunk) for tracking.
        """
        futures = {}
        for sensor_id in sensor_ids:
            for start_chunk, end_chunk in date_chunks:
                task = executor.submit(
                    self._fetch_chunk,
                    session,
                    sensor_id,
                    start_chunk,
                    end_chunk,
                )
                futures[task] = (sensor_id, start_chunk, end_chunk)
        return futures

    def _process_chunk_results(
        self, futures: Dict[Future, Tuple[str, datetime.date, datetime.date]]
    ) -> Tuple[
        List[pd.DataFrame], Set[Tuple[str, datetime.date, datetime.date]]
    ]:
        """
        Processes completed futures as they finish.

        Sorts results into successfully fetched DataFrames and a set
        of task info tuples for failed tasks.

        Parameters
        ----------
        futures : Dict[Future, Tuple[str, datetime.date, datetime.date]]
            The dictionary of futures returned by `_submit_chunk_tasks`.

        Returns
        -------
        Tuple[List[pd.DataFrame], Set[Tuple[str, ...]]]
            A tuple containing:
            - A list of all successfully fetched DataFrames.
            - A set of (sensor_id, start_date, end_date) tuples for
              all tasks that failed and require fallback.
        """
        all_dataframes = []
        failed_tasks = set()

        for future in as_completed(futures):
            task_info = futures[future]
            task_id = f"{task_info[0]} ({task_info[1]} to {task_info[2]})"
            try:
                df = future.result()
                if df is not None:
                    all_dataframes.append(df)
                else:
                    # Handles non-exception failures
                    # (e.g., HTTP 404, empty JSON)
                    print(
                        f"Task failed, queueing for 1-day fallback: {task_id}"
                    )
                    failed_tasks.add(task_info)
            except Exception as e:
                # Handles exceptions raised during the task execution
                print(
                    f"Task error, queueing for 1-day fallback: {task_id}: {e}"
                )
                failed_tasks.add(task_info)

        return all_dataframes, failed_tasks

    def _handle_failed_chunks(
        self,
        session: requests.Session,
        failed_tasks: Set[Tuple[str, datetime.date, datetime.date]],
    ) -> List[pd.DataFrame]:
        """
        Manages the fallback process for failed multi-day chunks.

        Parameters
        ----------
        session : requests.Session
            The authenticated requests session.
        failed_tasks : Set[Tuple[str, datetime.date, datetime.date]]
            A set of (sensor_id, start_date, end_date) tuples.

        Returns
        -------
        List[pd.DataFrame]
            A list of all DataFrames successfully retrieved during
            fallback.
        """
        if not failed_tasks:
            return []

        print(
            f"Handling {len(failed_tasks)} failed chunks by fetching 1-day increments..."
        )
        all_fallback_data = []

        daily_tasks_to_run = []
        for sensor_id, start_date, end_date in failed_tasks:
            print(
                f"Queueing 1-day fallback for {sensor_id} from {start_date} to {end_date}"
            )
            total_days = (end_date - start_date).days
            for i in range(total_days + 1):
                current_date = start_date + datetime.timedelta(days=i)
                daily_tasks_to_run.append((sensor_id, current_date))

        if not daily_tasks_to_run:
            return []

        print(f"Total fallback 1-day requests: {len(daily_tasks_to_run)}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_day, session, sensor_id, target_date
                ): f"{sensor_id} on {target_date}"
                for (sensor_id, target_date) in daily_tasks_to_run
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_fallback_data.append(df)
                except Exception as e:
                    print(f"Error in fallback task {task_id}: {e}")

        print("Fallback data retrieval complete.")
        return all_fallback_data

    def _process_and_combine_data(
        self, all_dataframes: List[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """
        Combines, de-duplicates, and sorts the final list of DataFrames.

        Parameters
        ----------
        all_dataframes : List[pd.DataFrame]
            A list of all DataFrames retrieved (both primary and
            fallback).

        Returns
        -------
        Optional[pd.DataFrame]
            The final, cleaned DataFrame, or None if no data was found.
        """
        if not all_dataframes:
            print("No data was successfully retrieved for any sensor.")
            return None

        try:
            print(f"Combining {len(all_dataframes)} DataFrame chunks...")
            combined_data = pd.concat(all_dataframes, ignore_index=True)

            print("Dropping duplicates and sorting by date...")
            combined_data.drop_duplicates(
                subset=["date", "sensor_ID"], inplace=True
            )
            combined_data.sort_values(by=["date"], inplace=True)
            combined_data.reset_index(drop=True, inplace=True)

            print(f"Final DataFrame shape: {combined_data.shape}")
            return combined_data
        except Exception as e:
            print(f"Failed to concatenate DataFrames: {e}")
            return None

    def _reassign_abbr_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassigns flood event numbers for the abbreviated CSV.

        Ensures that overlapping flood events are merged and
        numbered sequentially.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of abbreviated flood events.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with reassigned flood event numbers.
        """
        df = df.copy()

        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        df = df.sort_values(by=["start_time_UTC"]).reset_index(drop=True)

        if df.empty:
            return df

        if "flood_event" not in df.columns:
            df["flood_event"] = 0

        # Get the columns as typed Series objects
        start_times = df["start_time_UTC"]
        end_times = df["end_time_UTC"]

        # Initialize variables with the first row (at position 0)
        current_event_number = 1
        current_end_time = end_times.iloc[0]

        # Set the value for the first row
        df.loc[0, "flood_event"] = current_event_number

        # Iterate by int position from the second row (position 1) onward
        for i in range(1, len(df)):
            # Get data directly from the typed Series
            start_time = start_times.iloc[i]
            end_time = end_times.iloc[i]

            # Check if the current event overlaps
            if start_time < current_end_time:
                # Assign the same flood event number
                df.loc[i, "flood_event"] = current_event_number
                current_end_time = max(current_end_time, end_time)
            else:
                # Assign a new flood event number
                current_event_number += 1
                df.loc[i, "flood_event"] = current_event_number
                # Update the current end time
                current_end_time = end_time

        df["flood_event"] = df["flood_event"].astype(int)
        return df
    
    def _recalculate_event_summaries(
        self, abbr_df: pd.DataFrame, raw_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Updates duration, EST times, and max water levels in
        abbr_df based on its start/end times and the raw data.

        Parameters
        ----------
        abbr_df : pd.DataFrame
            The loaded abbreviated events DataFrame.
        raw_data_df : pd.DataFrame
            The cached raw sensor data.

        Returns
        -------
        pd.DataFrame
            The abbr_df with recalculated summary columns.
        """
        # Make sure we are not modifying a copy
        abbr_df = abbr_df.copy()
        
        # Ensure raw data `date` column is datetime
        raw_data_df["date"] = pd.to_datetime(raw_data_df["date"], utc=True)
        
        eastern = pytz.timezone("EST")

        for index, event in abbr_df.iterrows():
            start_utc = event["start_time_UTC"]
            end_utc = event["end_time_UTC"]
            sensor_id = event["sensor_ID"]

            # 1. Recalculate Duration
            duration_seconds = (end_utc - start_utc).total_seconds()
            duration_hours = round(duration_seconds / 3600, 3)
            abbr_df.loc[index, "duration_(hours)"] = duration_hours

            # 2. Recalculate EST Times
            abbr_df.loc[index, "start_time_EST"] = start_utc.astimezone(
                eastern
            )
            abbr_df.loc[index, "end_time_EST"] = end_utc.astimezone(eastern)

            # 3. Recalculate Max Water Levels
            event_mask = (
                (raw_data_df["sensor_ID"] == sensor_id)
                & (raw_data_df["date"] >= start_utc)
                & (raw_data_df["date"] <= end_utc)
            )
            event_raw_data = raw_data_df[event_mask]

            if event_raw_data.empty:
                # No raw data for this event, set maxes to 0
                max_ft = 0.0
            else:
                max_ft = event_raw_data["road_water_level_adj"].max()
                # Handle case where all data in range might be NaN
                if pd.isna(max_ft):
                    max_ft = 0.0

            max_m = max_ft / 3.28

            abbr_df.loc[index, "max_road_water_level_(ft)"] = round(max_ft, 3)
            abbr_df.loc[index, "max_road_water_level_(m)"] = round(max_m, 3)

        return abbr_df

    def _gen_abbr_flood_event_csv(
        self,
        dataframe: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
    ):
        """
        Generates an abbreviated (one row per event) flood event CSV.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The raw water level DataFrame from the API.
        csv_filename : str, optional
            The name of the CSV file to save.
        """
        # Read the existing CSV file if it exists
        try:
            existing_data = pd.read_csv(csv_filename)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        except FileNotFoundError:
            column_names = [
                "flood_event",
                "sensor_ID",
                "start_time_UTC",
                "end_time_UTC",
                "start_time_EST",
                "end_time_EST",
                "duration_(hours)",
                "max_road_water_level_(ft)",
                "max_road_water_level_(m)",
            ]
            existing_data = pd.DataFrame(columns=column_names)
            last_event_number = 0

        # Initialize variables
        flood_start_time = None
        max_water_level = 0
        flood_events = []
        eastern = pytz.timezone("EST")

        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                water_level = row["road_water_level_adj"]
                timestamp = row["date"]

                # Start of a new flood event
                if water_level > 0.02 and flood_start_time is None:
                    flood_start_time = timestamp
                    max_water_level = water_level

                # During a flood event, track max water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # End of a flood event
                if water_level < 0 and flood_start_time is not None:
                    flood_end_time = timestamp
                    start_time_est = flood_start_time.astimezone(eastern)
                    end_time_est = flood_end_time.astimezone(eastern)

                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "start_time_UTC": flood_start_time,
                            "end_time_UTC": flood_end_time,
                            "start_time_EST": start_time_est,
                            "end_time_EST": end_time_est,
                            "duration_(hours)": round(
                                (end_time_est - start_time_est).total_seconds()
                                / 3600,
                                3,
                            ),
                            "max_road_water_level_(ft)": round(
                                max_water_level, 3
                            ),
                            "max_road_water_level_(m)": round(
                                max_water_level / 3.28, 3
                            ),
                        }
                    )

                    # Reset variables
                    last_event_number += 1
                    flood_start_time = None
                    max_water_level = 0

        if not flood_events:
            print("No new flood events found.")
            return

        flood_event_df = pd.DataFrame(flood_events)

        if existing_data.empty:
            updated_data = self._reassign_abbr_flood_numbers(flood_event_df)
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            # Ensure UTC times are strings for deduplication, as loaded
            # CSVs may be strings
            merged_df["start_time_UTC"] = merged_df["start_time_UTC"].astype(
                str
            )
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "start_time_UTC"], keep="first"
            )
            updated_data = self._reassign_abbr_flood_numbers(filtered_df)

        updated_data.to_csv(csv_filename, index=False)

    def _reassign_flood_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassigns flood event numbers for the detailed flood event CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of detailed flood events (row-by-row).

        Returns
        -------
        pd.DataFrame
            The same DataFrame with reassigned flood event numbers.
        """
        df = df.copy()

        # Convert time columns to Timestamp
        df["start_time_UTC"] = pd.to_datetime(df["start_time_UTC"], utc=True)
        df["end_time_UTC"] = pd.to_datetime(df["end_time_UTC"], utc=True)
        df["time_UTC"] = pd.to_datetime(df["time_UTC"], utc=True)

        # Create a dictionary to map alphanumeric sensor IDs to numeric
        # values
        sensor_id_list = df["sensor_ID"].unique()
        sensor_id_mapping = {
            sensor_id: i for i, sensor_id in enumerate(sensor_id_list)
        }

        df["Sensor_ID_numeric"] = df["sensor_ID"].map(sensor_id_mapping)

        # Sort the dataframe by time_UTC and sensor ID
        df = df.sort_values(by=["time_UTC", "Sensor_ID_numeric"])
        df = df.reset_index(drop=True)
        df = df.drop(columns=["Sensor_ID_numeric"])

        # Dataframe of rows indicating the end of flood events
        complete_rows = df[df["duration_(hours)"].notnull()]

        if complete_rows.empty:
            print("No complete flood events to re-number.")
            return df

        # Initialize flood numbers and assigned event dictionary
        flood_event_number = 1
        last_assigned_event = {}
        current_end_time = complete_rows.iloc[0]["end_time_UTC"]

        # Iterate over each complete row (end of flood event)
        for index, row in complete_rows.iterrows():
            if row["start_time_UTC"] < current_end_time:
                # Part of the current combined event
                if row["sensor_ID"] not in last_assigned_event:
                    df.loc[
                        (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index
                elif row["sensor_ID"] in last_assigned_event:
                    df.loc[
                        (df.index > last_assigned_event[row["sensor_ID"]])
                        & (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index

                current_end_time = max(current_end_time, row["end_time_UTC"])
            else:
                # Start of a new combined event
                flood_event_number += 1

                if row["sensor_ID"] not in last_assigned_event:
                    df.loc[
                        (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index
                elif row["sensor_ID"] in last_assigned_event:
                    df.loc[
                        (df.index > last_assigned_event[row["sensor_ID"]])
                        & (df.index <= index)
                        & (df["sensor_ID"] == row["sensor_ID"]),
                        "flood_event",
                    ] = flood_event_number
                    last_assigned_event[row["sensor_ID"]] = index

                current_end_time = row["end_time_UTC"]

        return df

    def _gen_flood_tracker(
        self, dataframe: pd.DataFrame, csv_filename: str = "flood_events.csv"
    ):
        """
        Generates a detailed (row-by-row) flood event CSV file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.
        """
        # Read the existing CSV file if it exists
        try:
            existing_data = pd.read_csv(csv_filename)
            last_event_number = (
                existing_data["flood_event"].max() + 1
                if "flood_event" in existing_data.columns
                and not existing_data.empty
                else 0
            )
        except FileNotFoundError:
            column_names = [
                "flood_event",
                "sensor_ID",
                "time_UTC",
                "time_EST",
                "road_water_level",
                "sensor_water_level",
                "start_time_UTC",
                "end_time_UTC",
                "start_time_EST",
                "end_time_EST",
                "duration_(hours)",
                "max_road_water_level_(ft)",
                "max_road_water_level_(m)",
            ]
            existing_data = pd.DataFrame(columns=column_names)
            last_event_number = 0

        # Initialize variables
        flood_start_time = None
        flood_end_time = None
        max_water_level = 0
        flood_events = []
        eastern = pytz.timezone("EST")

        sorted_dataframe = dataframe.sort_values(by=["date"])
        sensors = sorted_dataframe["sensor_ID"].unique().tolist()

        for sensor in sensors:
            filtered_dataframe = sorted_dataframe[
                sorted_dataframe["sensor_ID"] == sensor
            ]

            for _, row in filtered_dataframe.iterrows():
                water_level = row["road_water_level_adj"]
                sensor_wtr_level = row["sensor_water_level_adj"]
                timestamp = row["date"]

                # Start of a new flood event
                if water_level > 0.02 and flood_start_time is None:
                    flood_start_time = timestamp
                    max_water_level = water_level

                # Track max water level
                if (
                    flood_start_time is not None
                    and water_level > max_water_level
                ):
                    max_water_level = water_level

                # End of a flood event (only set the end time)
                if water_level < 0 and flood_start_time is not None:
                    flood_end_time = timestamp

                if flood_start_time is not None and flood_end_time is None:
                    # During a flood, before it ends
                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(eastern),
                            "road_water_level": water_level,
                            "sensor_water_level": sensor_wtr_level,
                            "start_time_UTC": None,
                            "end_time_UTC": None,
                            "start_time_EST": None,
                            "end_time_EST": None,
                            "duration_(hours)": None,
                            "max_road_water_level_(ft)": None,
                            "max_road_water_level_(m)": None,
                        }
                    )
                elif (
                    flood_start_time is not None and flood_end_time is not None
                ):

                    # Define EST times just before they are used.
                    start_time_est = flood_start_time.astimezone(eastern)
                    end_time_est = flood_end_time.astimezone(eastern)

                    # At the end of a flood
                    flood_events.append(
                        {
                            "flood_event": last_event_number,
                            "sensor_ID": sensor,
                            "time_UTC": timestamp,
                            "time_EST": timestamp.astimezone(eastern),
                            "road_water_level": water_level,
                            "sensor_water_level": sensor_wtr_level,
                            "start_time_UTC": flood_start_time,
                            "end_time_UTC": flood_end_time,
                            "start_time_EST": start_time_est,
                            "end_time_EST": end_time_est,
                            "duration_(hours)": round(
                                (end_time_est - start_time_est).total_seconds()
                                / 3600,
                                3,
                            ),
                            "max_road_water_level_(ft)": round(
                                max_water_level, 3
                            ),
                            "max_road_water_level_(m)": round(
                                max_water_level / 3.28, 3
                            ),
                        }
                    )

                    # Reset variables
                    last_event_number += 1
                    flood_start_time = None
                    flood_end_time = None
                    max_water_level = 0

        if not flood_events:
            print("No new detailed flood data to track.")
            return

        flood_event_df = pd.DataFrame(flood_events)

        if existing_data.empty:
            updated_data = self._reassign_flood_numbers(flood_event_df)
        else:
            merged_df = pd.concat(
                [existing_data, flood_event_df], ignore_index=True
            )
            merged_df["time_UTC"] = merged_df["time_UTC"].astype(str)
            filtered_df = merged_df.drop_duplicates(
                subset=["sensor_ID", "time_UTC"], keep="first"
            )
            updated_data = self._reassign_flood_numbers(filtered_df)

        updated_data.to_csv(csv_filename, index=False)

    # === NEW: Private method for regenerating detailed CSV ===
    def _gen_flood_tracker_from_abbr(
        self,
        raw_data: pd.DataFrame,
        abbr_events_df: pd.DataFrame,
        csv_filename: str,
    ):
        """
        Regenerates the detailed flood tracker CSV from an
        abbreviated event list.

        This method uses the event start/end times from the
        `abbr_events_df` and filters the `raw_data` to build a
        new, detailed `flood_events.csv`.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The cached raw data DataFrame (e.g., from
            'raw_sensor_data.parquet').
        abbr_events_df : pd.DataFrame
            The loaded and cleaned abbreviated flood events
            DataFrame.
        csv_filename : str
            The path to save the new detailed CSV file to.
        """
        all_event_dfs = []
        eastern = pytz.timezone("EST")

        # Ensure raw_data dates are correct
        raw_data["date"] = pd.to_datetime(raw_data["date"], utc=True)

        # Ensure abbr_events_df dates are correct
        abbr_events_df["start_time_UTC"] = pd.to_datetime(
            abbr_events_df["start_time_UTC"], utc=True
        )
        abbr_events_df["end_time_UTC"] = pd.to_datetime(
            abbr_events_df["end_time_UTC"], utc=True
        )
        
        # Handle EST columns, ensuring they are timezone-aware.
        # This handles strings from CSV or existing datetime objects.
        abbr_events_df["start_time_EST"] = pd.to_datetime(
            abbr_events_df["start_time_EST"]
        ).apply(lambda x: x.tz_localize(eastern) if x.tzinfo is None else x.tz_convert(eastern))
        abbr_events_df["end_time_EST"] = pd.to_datetime(
            abbr_events_df["end_time_EST"]
        ).apply(lambda x: x.tz_localize(eastern) if x.tzinfo is None else x.tz_convert(eastern))


        for _, row in abbr_events_df.iterrows():
            event_num = row["flood_event"]
            sensor_id = row["sensor_ID"]
            start_utc = row["start_time_UTC"]
            end_utc = row["end_time_UTC"]

            # Filter raw data for the exact event timespan
            event_data_mask = (
                (raw_data["sensor_ID"] == sensor_id)
                & (raw_data["date"] >= start_utc)
                & (raw_data["date"] <= end_utc)
            )
            event_df = raw_data.loc[event_data_mask].copy()

            if event_df.empty:
                print(f"Warning: No raw data found for event {event_num} ({sensor_id}). Skipping.")
                continue  # Skip if no raw data found for this event

            # Populate columns for the detailed CSV
            event_df["flood_event"] = event_num
            event_df["time_UTC"] = event_df["date"]
            event_df["time_EST"] = event_df["date"].dt.tz_convert(eastern)
            event_df["road_water_level"] = event_df["road_water_level_adj"]
            event_df["sensor_water_level"] = event_df[
                "sensor_water_level_adj"
            ]

            # Initialize summary columns
            event_df["start_time_UTC"] = pd.Series(
                pd.NaT, index=event_df.index, dtype="datetime64[ns, UTC]"
            )
            event_df["end_time_UTC"] = pd.Series(
                pd.NaT, index=event_df.index, dtype="datetime64[ns, UTC]"
            )
            event_df["start_time_EST"] = pd.Series(
                pd.NaT,
                index=event_df.index,
                dtype=f"datetime64[ns, {eastern}]",
            )
            event_df["end_time_EST"] = pd.Series(
                pd.NaT,
                index=event_df.index,
                dtype=f"datetime64[ns, {eastern}]",
            )
            
            event_df["duration_(hours)"] = pd.NA
            event_df["max_road_water_level_(ft)"] = pd.NA
            event_df["max_road_water_level_(m)"] = pd.NA

            # Set the summary values *only* on the last row
            last_index = event_df.index[-1]
            event_df.loc[last_index, "start_time_UTC"] = start_utc
            event_df.loc[last_index, "end_time_UTC"] = end_utc
            event_df.loc[last_index, "start_time_EST"] = row["start_time_EST"]
            event_df.loc[last_index, "end_time_EST"] = row["end_time_EST"]
            event_df.loc[last_index, "duration_(hours)"] = row[
                "duration_(hours)"
            ]
            event_df.loc[last_index, "max_road_water_level_(ft)"] = row[
                "max_road_water_level_(ft)"
            ]
            event_df.loc[last_index, "max_road_water_level_(m)"] = row[
                "max_road_water_level_(m)"
            ]

            all_event_dfs.append(event_df)

        # Define columns even if no events are found
        final_columns = [
            "flood_event",
            "sensor_ID",
            "time_UTC",
            "time_EST",
            "road_water_level",
            "sensor_water_level",
            "start_time_UTC",
            "end_time_UTC",
            "start_time_EST",
            "end_time_EST",
            "duration_(hours)",
            "max_road_water_level_(ft)",
            "max_road_water_level_(m)",
        ]

        if not all_event_dfs:
            print(
                "No flood events from abbr_csv had matching raw data. No output generated."
            )
            # Create empty file to not cause FileNotFoundError later
            final_df = pd.DataFrame(columns=final_columns)
            final_df.to_csv(csv_filename, index=False)
            return

        final_df = pd.concat(all_event_dfs)

        # Reorder columns to match the original spec
        final_df = final_df[final_columns]

        final_df.sort_values(by=["flood_event", "time_UTC"], inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        final_df.to_csv(csv_filename, index=False)
        print(f"Successfully regenerated detailed event file: {csv_filename}")
    # === END NEW ===

    def _find_outages(
        self, dataframe: pd.DataFrame, csv_filename: str = "sensor_outages.csv"
    ) -> Optional[pd.DataFrame]:
        """
        Generates a list of outages and saves them to a CSV file.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            A string representing the name of the created CSV file.

        Returns
        -------
        Optional[pd.DataFrame]
            A DataFrame of outages, or None if no outages are found.
        """
        sensor_id_list = dataframe["sensor_ID"].unique().tolist()
        outages = []
        eastern = pytz.timezone("EST")

        for sensor_id in sensor_id_list:
            filtered_dataframe = dataframe[dataframe["sensor_ID"] == sensor_id]
            sorted_dataframe = filtered_dataframe.sort_values(
                by="date"
            ).reset_index(drop=True)

            # Use diff() to find time gaps
            time_differences = sorted_dataframe["date"].diff()

            # Identify gaps larger than 1 hour
            gaps = time_differences > pd.Timedelta(hours=1)
            indices_of_gaps = gaps[gaps].index

            for i in indices_of_gaps:
                outage_start_time = sorted_dataframe.loc[i - 1, "date"]
                outage_end_time = sorted_dataframe.loc[i, "date"]
                start_time_est = outage_start_time.astimezone(eastern)
                end_time_est = outage_end_time.astimezone(eastern)

                outages.append(
                    {
                        "outage_number": None,
                        "sensor_ID": sensor_id,
                        "start_time_UTC": outage_start_time,
                        "end_time_UTC": outage_end_time,
                        "start_time_EST": start_time_est,
                        "end_time_EST": end_time_est,
                        "duration_(hours)": round(
                            (end_time_est - start_time_est).total_seconds()
                            / 3600,
                            3,
                        ),
                    }
                )

        if not outages:
            print("No sensor outages found.")
            return None

        outage_dataframe = pd.DataFrame(outages).sort_values(
            by="start_time_UTC"
        )

        # Assign outage numbers sequentially
        outage_dataframe["outage_number"] = range(1, len(outage_dataframe) + 1)

        outage_dataframe.to_csv(csv_filename, index=False)
        return outage_dataframe

    def _check_for_outage_during_flood(
        self,
        outage_csv: str = "sensor_outages.csv",
        abbr_flood_csv: str = "abbr_flood_events.csv",
    ):
        """
        Compares flood events to outages and flags overlaps.

        Uses a `pd.merge` for efficient comparison.

        Parameters
        ----------
        outage_csv : str, optional
            Filename of the outages CSV.
        abbr_flood_csv : str, optional
            Filename of the abbreviated flood events CSV.
        """
        try:
            abbr_floods = pd.read_csv(abbr_flood_csv)
            outage_dataframe = pd.read_csv(outage_csv)
        except FileNotFoundError as e:
            print(
                f"Error: Cannot check for outages, file not found: {e.filename}"
            )
            return

        if abbr_floods.empty or outage_dataframe.empty:
            print("No flood or outage data to compare.")
            return

        # Ensure date columns are in datetime format for comparison
        abbr_floods["start_time_UTC"] = pd.to_datetime(
            abbr_floods["start_time_UTC"]
        )
        abbr_floods["end_time_UTC"] = pd.to_datetime(
            abbr_floods["end_time_UTC"]
        )
        outage_dataframe["start_time_UTC"] = pd.to_datetime(
            outage_dataframe["start_time_UTC"]
        )

        # Initialize flag columns
        outage_dataframe["during_flood_event"] = "No"
        outage_dataframe["flood_event_number"] = pd.NA
        abbr_floods["outage"] = "No"

        # Merge on sensor_ID to compare all outages with all floods for
        # that sensor
        merged = pd.merge(
            outage_dataframe,
            abbr_floods,
            on="sensor_ID",
            suffixes=("_outage", "_flood"),
        )
        
        if merged.empty:
            print("No matching sensor_IDs in outage and flood files. Skipping check.")
            # Save the files with the initialized "No" columns
            abbr_floods.to_csv(abbr_flood_csv, index=False)
            outage_dataframe.to_csv(outage_csv, index=False)
            return

        # Create a mask to find where outage start time is within a
        # flood event
        mask = (
            merged["start_time_UTC_outage"] >= merged["start_time_UTC_flood"]
        ) & (merged["start_time_UTC_outage"] <= merged["end_time_UTC_flood"])

        # Get the original indices of the rows that match
        outage_indices_to_flag = merged.loc[mask, "outage_number"].unique()
        flood_event_nums_to_flag = merged.loc[mask, "flood_event"].unique()

        # Update the outage DataFrame
        matching_outages = merged.loc[
            mask, ["outage_number", "flood_event"]
        ].set_index("outage_number")

        # Update the original outage_dataframe
        for outage_num, flood_event_num in matching_outages[
            "flood_event"
        ].items():
            outage_dataframe.loc[
                outage_dataframe["outage_number"] == outage_num,
                "during_flood_event",
            ] = "Yes"
            outage_dataframe.loc[
                outage_dataframe["outage_number"] == outage_num,
                "flood_event_number",
            ] = flood_event_num

        # Update the flood DataFrame
        abbr_floods.loc[
            abbr_floods["flood_event"].isin(flood_event_nums_to_flag), "outage"
        ] = "Yes"

        # Save the updated DataFrames back to CSVs
        abbr_floods.to_csv(abbr_flood_csv, index=False)
        outage_dataframe.to_csv(outage_csv, index=False)

    # === MODIFIED: Added force_overwrite parameter ===
    def _plot_and_save_flood_plots(
        self,
        sunnyd_data: pd.DataFrame,
        csv_filename: str = "abbr_flood_events.csv",
        plot_folder: str = "flood_plots",
        force_overwrite: bool = False,
    ):
        """
        Plots flood events listed in the abbreviated flood event CSV.

        Parameters
        ----------
        sunnyd_data : pd.DataFrame
            Sunny Day Flooding Project dataframe from the water level API.
        csv_filename : str, optional
            Filename of the abbreviated flood events CSV file.
        plot_folder : str, optional
            Name of the directory to save flood plots to.
        force_overwrite : bool, optional
            If True, existing plots will be overwritten. Default is
            False.
        """

        # Create a folder to save the plots if it doesn't exist
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
            print(f"Folder {plot_folder} created successfully.")

        try:
            abbr_dataframe = pd.read_csv(csv_filename)
        except FileNotFoundError:
            print(f"Error: Cannot create plots, file not found: {csv_filename}")
            return

        if abbr_dataframe.empty:
            print("No flood events in CSV to plot.")
            return

        # Iterate over each row (event) in the DataFrame
        for _, row in abbr_dataframe.iterrows():
            sensor_id = row["sensor_ID"]
            event_number = row["flood_event"]

            # Check if file already exists
            plot_filename = os.path.join(
                plot_folder, f"flood_event_{event_number}_{sensor_id}.png"
            )
            # === MODIFIED: Check force_overwrite flag ===
            if os.path.exists(plot_filename) and not force_overwrite:
                print(f"Plot already exists (skipping): {plot_filename}")
                continue
            elif os.path.exists(plot_filename) and force_overwrite:
                print(f"Plot already exists (overwriting): {plot_filename}")
            # === END MODIFIED ===

            flood_start_time = pd.to_datetime(row["start_time_UTC"], utc=True)
            flood_end_time = pd.to_datetime(row["end_time_UTC"], utc=True)

            plot_start_time = flood_start_time - datetime.timedelta(days=1)
            plot_end_time = flood_end_time + datetime.timedelta(days=1)

            # Filter the main DataFrame to get data for this plot
            filtered_df = sunnyd_data[
                (sunnyd_data["date"] >= plot_start_time)
                & (sunnyd_data["date"] <= plot_end_time)
                & (sunnyd_data["sensor_ID"] == sensor_id)
            ]

            if filtered_df.empty:
                print(
                    f"No raw data found for event {event_number} ({sensor_id}). Skipping plot."
                )
                continue

            plt.figure(figsize=(12, 10))

            plt.plot(
                filtered_df["date"],
                filtered_df["road_water_level_adj"],
                label="Water Level",
                linestyle="-",
                color="#427e93",
            )

            plt.plot(
                filtered_df["date"],
                np.zeros_like(filtered_df["date"]),
                label="Roadway",
                linestyle="-",
                linewidth=2,
                color="black",
            )

            # Identify the indices within the flood event
            flood_indices = (filtered_df["date"] >= flood_start_time) & (
                filtered_df["date"] <= flood_end_time
            )

            plt.plot(
                filtered_df.loc[flood_indices, "date"],
                filtered_df.loc[flood_indices, "road_water_level_adj"],
                marker="x",
                markersize=5,
                linestyle="None",
                color="#cc0000",
                label="Flood Event Data Points",
            )

            plt.ylim(-2.5, 2)
            plt.xticks(rotation=45)
            plt.yticks(np.arange(-2.5, 2.25, 0.25))
            plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
            plt.grid(
                which="major",
                axis="y",
                linestyle="-",
                linewidth="0.5",
                color="gray",
            )
            plt.grid(
                which="minor",
                axis="y",
                linestyle="-",
                linewidth="0.5",
                color="black",
            )
            plt.xlabel("Date")
            plt.ylabel("Water Level (ft above road)")
            plt.title(f"Flood Event {event_number} for {sensor_id}")
            plt.legend(loc="upper right")
            plt.tight_layout()

            plt.savefig(plot_filename)
            plt.close()

    # ===== Static Methods ========

    @staticmethod
    def num_of_flood_days_by_start(csv_name="abbr_flood_events.csv"):
        """Counts unique flood days based on the start date in UTC.

        Parameters
        ----------
        csv_name : str, optional
            The name of the CSV file containing flood events.
            Default is "abbr_flood_events.csv".

        Returns
        -------
        int
            The number of unique flooding days. Returns None if the
            file is not found.
        """

        # Read dataframe
        try:
            read_df = pd.read_csv(csv_name)
        except FileNotFoundError:
            print(f"Error: File '{csv_name}' not found.")
            return

        # Convert 'start_time_UTC' to Pandas datetime object
        read_df["start_time_UTC"] = pd.to_datetime(read_df["start_time_UTC"])

        # Extract date part from datetime column
        read_df["date"] = read_df["start_time_UTC"].dt.date

        # Count the number of unique dates
        num_days = read_df["date"].nunique()

        print("Number of days of flooding:", num_days)
        return num_days

    @staticmethod
    def num_of_flood_days(timezone="EST", csv_name="abbr_flood_events.csv"):
        """Counts unique flood days spanning start and end dates.

        Calculates the total number of unique calendar days a flood
        occurred, considering the full range from start to end time.
        The calculation can be performed in either EST or UTC.

        Parameters
        ----------
        timezone : str, optional
            The timezone to use for date calculation ('EST' or 'UTC').
            Default is "EST".
        csv_name : str, optional
            The name of the CSV file containing flood events.
            Default is "abbr_flood_events.csv".

        Returns
        -------
        int
            The total number of unique flooding days. Returns None
            on file not found or invalid timezone.
        """

        # Read dataframe
        try:
            read_df = pd.read_csv(csv_name)
        except FileNotFoundError:
            print(f"Error: File '{csv_name}' not found.")
            return

        if timezone == "UTC":

            # Convert 'start_time_UTC' to Pandas datetime object
            read_df["start_time"] = pd.to_datetime(read_df["start_time_UTC"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_UTC"])

        elif timezone == "EST":

            # Convert 'start_time_EST' to Pandas datetime object
            read_df["start_time"] = pd.to_datetime(read_df["start_time_EST"])
            read_df["end_time"] = pd.to_datetime(read_df["end_time_EST"])

        else:
            print(
                f"Error: Timezone '{timezone}' not recognized. Use 'EST' or 'UTC'."
            )
            return

        # Extract date part from datetime column
        read_df["start_date"] = read_df["start_time"].dt.date
        read_df["end_date"] = read_df["end_time"].dt.date

        # Initialize an empty list to store all dates
        all_dates = []

        # Iterate over each row and append all dates in the range
        for index, row in read_df.iterrows():
            start_date = row["start_date"]
            end_date = row["end_date"]
            date_range = pd.date_range(start=start_date, end=end_date)
            all_dates.extend(date_range)

        # Convert the list of dates to a Pandas Series
        all_dates = pd.Series(all_dates, name="combined_date")

        num_days = all_dates.nunique()

        print("Number of days of flooding:", num_days)
        return num_days

    @staticmethod
    def avg_errors(
        error_csv, save_file_as=None, csv_filename="abbr_flood_events.csv"
    ):
        """Calculates and appends average errors to the flood events CSV.

        Reads a separate CSV with error data, calculates the 7-day
        rolling average of errors ('Mean', '5th Pct', '95th Pct')
        prior to each flood's end time, and appends these
        averages to the main flood events DataFrame. Saves the
        result to a new or overwritten CSV.

        Parameters
        ----------
        error_csv : str
            Name of the CSV file containing error data.
        save_file_as : str, optional
            File path to save the updated DataFrame. If None,
            it overwrites the file specified by `csv_filename`.
            Default is None.
        csv_filename : str, optional
            Name of the abbreviated flood events CSV to read and
            update. Default is "abbr_flood_events.csv".

        Returns
        -------
        bool
            True if the file was saved successfully, None if an
            error (e.g., FileNotFoundError) occurred.
        """
        # Read CSV files
        try:
            flood_df = pd.read_csv(csv_filename)
            other_df = pd.read_csv(error_csv)
        except FileNotFoundError as e:
            print(f"Error: File not found. {e}")
            return

        # Convert 'flood_end_time' to datetime
        flood_df["end_time_UTC"] = pd.to_datetime(
            flood_df["end_time_UTC"], utc=True
        )
        other_df["Date"] = pd.to_datetime(other_df["Date"], utc=True)

        # Loop through each row in flood_df
        averages = []
        for index, row in flood_df.iterrows():
            # Define the time range for a week before the flood end time
            start_time = row["end_time_UTC"] - pd.Timedelta(days=7)
            end_time = row["end_time_UTC"]

            # Select rows from other_df that fall within the defined
            # time range
            selected_rows = other_df[
                (other_df["Date"] >= start_time)
                & (other_df["Date"] <= end_time)
            ]

            # Calculate the average of desired columns
            avg_values = (
                selected_rows[["Mean", "5th Pct", "95th Pct"]].mean().round(4)
            )
            averages.append(avg_values.values)

        # Convert the list of averages to DataFrame
        averages_df = pd.DataFrame(
            averages,
            columns=["Mean of Mean", "Mean of 5th Pct", "Mean of 95th Pct"],
        )

        # Concatenate averages_df with flood_df
        flood_df_with_averages = pd.concat([flood_df, averages_df], axis=1)

        if save_file_as is None:
            save_file = csv_filename
        else:
            save_file = save_file_as

        # Save the result
        flood_df_with_averages.to_csv(save_file, index=False)

        print(f"File with averages saved successfully to '{save_file}'.")
        return True