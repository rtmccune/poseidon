"""Logging configuration for POSEIDON scripts and notebooks.

Library modules only create loggers with ``logging.getLogger(__name__)``
and never configure handlers. Entry points (HPC driver scripts,
notebooks) call :func:`setup_logging` once to choose the format and
verbosity.
"""

import logging
import sys

#: Default record format: timestamp, level, logger name, message.
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

#: Default timestamp format for log records.
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level=logging.INFO, fmt=LOG_FORMAT, stream=None):
    """
    Configure the root logger to print timestamped records.

    Calling this more than once replaces the previous configuration,
    which keeps notebook re-runs from duplicating output.

    Parameters
    ----------
    level : int or str, optional
        Minimum level to emit. Default is ``logging.INFO``.
    fmt : str, optional
        Record format string. Default is :data:`LOG_FORMAT`.
    stream : file-like, optional
        Output stream. Default is ``sys.stdout`` so HPC job logs keep
        records in order with other printed output.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=DATE_FORMAT,
        stream=stream or sys.stdout,
        force=True,
    )
    return logging.getLogger()


def progress_interval(total, steps=10):
    """
    Return how many items to process between progress log messages.

    Parameters
    ----------
    total : int
        Total number of items in the loop.
    steps : int, optional
        Approximate number of progress messages wanted. Default is 10.

    Returns
    -------
    int
        Positive interval such that roughly ``steps`` messages print.
    """
    return max(1, total // steps)


def should_report(index, total, interval):
    """
    Decide whether to log progress for a zero-based loop index.

    Parameters
    ----------
    index : int
        Zero-based index of the current item.
    total : int
        Total number of items.
    interval : int
        Reporting interval from :func:`progress_interval`.

    Returns
    -------
    bool
        True for the first item, every ``interval`` items, and the last.
    """
    count = index + 1
    return index == 0 or count == total or count % interval == 0
