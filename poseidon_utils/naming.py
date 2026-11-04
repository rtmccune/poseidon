"""Parse SDFP camera names and timestamps from image filenames.

SDFP image filenames follow the pattern
``CAM_<SITE>_<NN>_<YYYYMMDDHHMMSS>[suffix].<ext>``, for example
``CAM_CB_03_20231002131826.jpg`` or
``CAM_CB_03_20231002131826_predseg_labels.png``. The 14-digit
timestamp is in UTC.
"""

import re
from datetime import datetime

from .constants import UTC

#: Regex matching a camera name such as ``CAM_CB_03``.
CAMERA_PATTERN = re.compile(r"CAM_[A-Z]{2}_[0-9]{2}")

#: Regex matching a 14-digit ``YYYYMMDDHHMMSS`` timestamp.
TIMESTAMP_PATTERN = re.compile(r"\d{14}")

#: ``strptime`` format of the filename timestamp.
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"

#: Prefix that turns a sensor ID (``CB_03``) into a camera name.
CAMERA_PREFIX = "CAM_"


def extract_camera_name(filename):
    """
    Extract the camera name from a filename.

    Parameters
    ----------
    filename : str
        Filename or path to parse, e.g. ``"CAM_CB_03_20231002131826.jpg"``.

    Returns
    -------
    str or None
        The camera name (e.g. ``"CAM_CB_03"``), or None if absent.
    """
    match = CAMERA_PATTERN.search(str(filename))
    return match.group(0) if match else None


def extract_timestamp(filename):
    """
    Extract the 14-digit UTC timestamp string from a filename.

    Parameters
    ----------
    filename : str
        Filename or path to parse, e.g. ``"CAM_CB_03_20231002131826.jpg"``.

    Returns
    -------
    str or None
        The timestamp (e.g. ``"20231002131826"``), or None if absent.
    """
    match = TIMESTAMP_PATTERN.search(str(filename))
    return match.group(0) if match else None


def parse_timestamp(filename):
    """
    Parse the filename timestamp into a timezone-aware UTC datetime.

    Parameters
    ----------
    filename : str
        Filename or path containing a ``YYYYMMDDHHMMSS`` timestamp.

    Returns
    -------
    datetime.datetime or None
        The aware UTC datetime, or None if no valid timestamp exists.
    """
    stamp = extract_timestamp(filename)
    if stamp is None:
        return None
    try:
        return datetime.strptime(stamp, TIMESTAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        # 14 digits that are not a real date (e.g. month 13)
        return None


def sensor_id_from_camera(camera_name):
    """
    Convert a camera name to its SDFP sensor ID.

    Parameters
    ----------
    camera_name : str
        Camera name such as ``"CAM_CB_03"``.

    Returns
    -------
    str
        Sensor ID such as ``"CB_03"``.
    """
    return camera_name.removeprefix(CAMERA_PREFIX)


def camera_from_sensor_id(sensor_id):
    """
    Convert an SDFP sensor ID to its camera name.

    Parameters
    ----------
    sensor_id : str
        Sensor ID such as ``"CB_03"``.

    Returns
    -------
    str
        Camera name such as ``"CAM_CB_03"``.
    """
    return f"{CAMERA_PREFIX}{sensor_id}"
