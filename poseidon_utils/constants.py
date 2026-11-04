"""Physical constants and time zones shared across POSEIDON.

Timestamp conventions
---------------------
Sunny Day Flooding Project (SDFP) data and image filenames are in UTC.
Columns suffixed ``_EST`` in the flood event CSVs use a *fixed* UTC-5
offset (Eastern Standard Time, no daylight saving), matching the
convention used by the SDFP API. Clock-hour filters that care about
local daylight (e.g. "6 am to 7 pm") use ``EASTERN_TZ``, which does
observe daylight saving time.
"""

from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

#: Exact conversion factor from international feet to meters.
FEET_TO_METERS = 0.3048

#: Fixed-offset Eastern Standard Time (UTC-5, no daylight saving).
EST = timezone(timedelta(hours=-5), "EST")

#: US Eastern local time, including daylight saving transitions.
EASTERN_TZ = ZoneInfo("America/New_York")

#: Coordinated Universal Time.
UTC = timezone.utc
