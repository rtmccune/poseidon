"""Shared utilities for POSEIDON: data access, file organization, I/O.

Submodules
----------
constants
    Unit conversions and time zones.
naming
    Camera name and timestamp parsing for SDFP image filenames.
logging_utils
    Logging setup for scripts and notebooks.
zarr_io
    Atomic read/write helpers for ZIP Zarr stores.
site_config
    Per-site camera and grid configuration loaded from YAML.
event_tracker
    Sunny Day Flooding Project API client and flood event summaries.
image_handler
    Pull and filter images from the SDFP image archive.
file_organizer
    Organize event CSVs and images into per-event folders.

Heavy dependencies (``requests``, ``matplotlib``, ``astral``) load only
when the classes that need them are first accessed.
"""

import importlib

from .logging_utils import setup_logging
from .site_config import SiteConfig, load_site_config

_LAZY_ATTRIBUTES = {
    "EventTracker": "event_tracker",
    "ImageHandler": "image_handler",
}
_LAZY_MODULES = {
    "constants", "event_tracker", "file_organizer", "image_handler",
    "logging_utils", "naming", "site_config", "zarr_io",
}

__all__ = [
    "EventTracker",
    "ImageHandler",
    "SiteConfig",
    "load_site_config",
    "setup_logging",
    *sorted(_LAZY_MODULES),
]


def __getattr__(name):
    """
    Import public classes and submodules on first access.

    Parameters
    ----------
    name : str
        Attribute requested from the package.

    Returns
    -------
    object
        The requested class or submodule.

    Raises
    ------
    AttributeError
        If `name` is not a public attribute of the package.
    """
    if name in _LAZY_ATTRIBUTES:
        module = importlib.import_module(f".{_LAZY_ATTRIBUTES[name]}",
                                         __name__)
        return getattr(module, name)
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
