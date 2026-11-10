"""Download the detailed sensor records for a QC'd flood event sheet.

Builds the ``flood_events.csv`` needed by stage 05 from an abbreviated
event sheet that was edited or merged outside ``EventTracker`` (e.g.
the QC results of the flood_event_QC app). Uses
``EventTracker.pull_qced_time_series_bulk``, which keeps the sheet's
``flood_event`` numbers as they are. Events with no readings in their
window, such as floods confirmed only from images during a sensor
outage, are reported and left out; stage 05 still builds their folders
from the images.

The output has the same columns as ``EventTracker``'s detailed CSV:
one row per reading, with the event's summary columns on its last row.

Credentials come from the ``SDFP_API_USERNAME`` and
``SDFP_API_PASSWORD`` environment variables, else from ``API_USERNAME``
and ``API_PASSWORD`` in a ``mysecrets.py`` at the repository root or in
``notebooks/``, else from a prompt.

Example
-------
``python run_pull_event_series.py --abbr_events_csv
data/down_east/abbr_flood_events.csv``
"""

import getpass
import importlib.util
import logging
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd

from poseidon_utils.event_tracker import (
    DETAILED_COLUMNS,
    SUMMARY_COLUMNS,
    EventTracker,
)

import cli_common

logger = logging.getLogger("poseidon.naiads")

#: Repository root (this file is in poseidon_deploy/naiads/).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Places searched for mysecrets.py.
SECRETS_PATHS = (REPO_ROOT / "mysecrets.py",
                 REPO_ROOT / "notebooks" / "mysecrets.py")


def get_credentials():
    """
    Find the SDFP API username and password.

    Returns
    -------
    tuple of (str, str)
        Username and password.
    """
    user = os.environ.get("SDFP_API_USERNAME")
    password = os.environ.get("SDFP_API_PASSWORD")
    if user and password:
        logger.info("Using SDFP credentials from the environment.")
        return user, password
    for path in SECRETS_PATHS:
        if path.exists():
            spec = importlib.util.spec_from_file_location("mysecrets", path)
            secrets = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(secrets)
            logger.info("Using SDFP credentials from %s", path)
            return secrets.API_USERNAME, secrets.API_PASSWORD
    return input("SDFP API username: "), getpass.getpass("Password: ")


def date_range(events, buffer_hours):
    """
    Return the UTC dates spanning every event window.

    Parameters
    ----------
    events : pandas.DataFrame
        Events with ``start_time_UTC`` and ``end_time_UTC``.
    buffer_hours : float
        Hours added before and after each event.

    Returns
    -------
    tuple of datetime.date
        First and last day to download (one extra day each side).
    """
    buffer = timedelta(hours=buffer_hours, days=1)
    start = pd.to_datetime(events["start_time_UTC"], utc=True).min()
    end = pd.to_datetime(events["end_time_UTC"], utc=True).max()
    return (start - buffer).date(), (end + buffer).date()


def add_event_summaries(readings, events):
    """
    Put each event's summary columns on its last reading.

    Parameters
    ----------
    readings : pandas.DataFrame
        Output of ``pull_qced_time_series_bulk``.
    events : pandas.DataFrame
        The abbreviated event sheet.

    Returns
    -------
    pandas.DataFrame
        `readings` with ``DETAILED_COLUMNS``, sorted by event and time.
    """
    readings = readings.sort_values(["flood_event", "time_UTC"],
                                    kind="stable").reset_index(drop=True)
    for column in SUMMARY_COLUMNS:
        readings[column] = None
    summaries = events.set_index(["flood_event", "sensor_ID"])
    last = readings.groupby(["flood_event", "sensor_ID"]).tail(1)
    for index, row in last.iterrows():
        key = (row["flood_event"], row["sensor_ID"])
        if key in summaries.index:
            for column in SUMMARY_COLUMNS:
                if column in summaries.columns:
                    readings.at[index, column] = summaries.at[key, column]
    return readings[DETAILED_COLUMNS]


def report_gaps(detailed, gap_minutes):
    """
    Log events whose readings have a gap longer than `gap_minutes`.

    Days the API never returned, even after the daily retries, are
    skipped by ``EventTracker`` and show up here as long gaps. Sensor
    outages appear too; check them against the QC sheet's ``outage``.

    Parameters
    ----------
    detailed : pandas.DataFrame
        Detailed readings with ``flood_event`` and ``time_UTC``.
    gap_minutes : float
        Longest gap between readings that is not reported.

    Returns
    -------
    list of tuple
        (event, gap start, gap length in minutes) for every gap found.
    """
    times = pd.to_datetime(detailed["time_UTC"], utc=True, format="mixed")
    frame = pd.DataFrame({"event": detailed["flood_event"], "time": times})
    gaps = []
    for event, group in frame.sort_values("time").groupby("event"):
        steps = group["time"].diff().dt.total_seconds() / 60
        for index in steps.index[steps > gap_minutes]:
            start = group["time"].shift(1)[index]
            gaps.append((event, start, float(steps[index])))
    if gaps:
        logger.warning("%d gaps longer than %g min between readings "
                       "(possible missing data):", len(gaps), gap_minutes)
        for event, start, minutes in gaps:
            logger.warning("  event %s: %.0f min after %s",
                           event, minutes, start)
    else:
        logger.info("No gaps longer than %g min in any event.", gap_minutes)
    return gaps


def main(argv=None):
    """
    Write the detailed sensor records for an abbreviated event sheet.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default reads ``sys.argv``.

    Returns
    -------
    int
        Process exit code: 1 if nothing could be written, else 0.
    """
    parser = cli_common.make_parser(__doc__.splitlines()[0])
    parser.add_argument("--abbr_events_csv", required=True,
                        help="Abbreviated (one row per event) CSV.")
    parser.add_argument("--output", default=None,
                        help="Detailed CSV to write; default is "
                             "flood_events.csv beside --abbr_events_csv.")
    parser.add_argument("--buffer_hours", type=float, default=0.0,
                        help="Hours of readings kept before and after "
                             "each event. 0 matches EventTracker's "
                             "flood_events.csv.")
    parser.add_argument("--chunk_days", type=int, default=30,
                        help="Days requested per API call. Lower it "
                             "(e.g. 7) if the server returns 504 "
                             "Gateway Time-out errors.")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Requests sent to the API at once.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds to wait for each API response.")
    parser.add_argument("--gap_minutes", type=float, default=30.0,
                        help="Report events whose readings have a gap "
                             "longer than this (e.g. a day the API "
                             "never returned).")
    args = cli_common.start(parser, argv)

    abbr_path = Path(args.abbr_events_csv)
    output = Path(args.output) if args.output else (
        abbr_path.parent / "flood_events.csv")
    events = pd.read_csv(abbr_path)
    min_date, max_date = date_range(events, args.buffer_hours)

    tracker = EventTracker(get_credentials(), min_date, max_date,
                           max_workers=args.max_workers,
                           timeout=args.timeout)
    # Write to a temporary file, then add the summaries to the final one
    raw_path = output.with_suffix(".raw.csv")
    readings = tracker.pull_qced_time_series_bulk(
        str(abbr_path), str(raw_path), chunk_days=args.chunk_days,
        time_buffer_hours=args.buffer_hours)
    raw_path.unlink(missing_ok=True)
    if readings is None:
        logger.error("No readings were found for any event.")
        return 1

    detailed = add_event_summaries(readings, events)
    detailed.to_csv(output, index=False)

    covered = set(zip(detailed["flood_event"], detailed["sensor_ID"]))
    missing = events[[(e, s) not in covered for e, s in
                      zip(events["flood_event"], events["sensor_ID"])]]
    logger.info("Wrote %d readings for %d of %d events to %s",
                len(detailed), len(covered), len(events), output)
    if not missing.empty:
        logger.info("%d events have no sensor readings (image-only): %s",
                    len(missing), ", ".join(
                        str(n) for n in missing["flood_event"]))
    report_gaps(detailed, args.gap_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
