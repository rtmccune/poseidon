"""
Shared helpers for the flood_qc package.

Functions
---------
log
    Print a timestamped message to stdout or stderr.
"""

import sys
from datetime import datetime


def log(message, level="info"):
    """
    Print a timestamped log message to stdout or stderr.

    Parameters
    ----------
    message : str
        The message to print.
    level : str, optional
        The log level. ``"error"`` prints to stderr; any other value
        prints to stdout. Default is ``"info"``.

    Returns
    -------
    None
    """
    # Format the current local time for the start of the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Build the message as "[timestamp] LEVEL: message"
    formatted_message = f"[{timestamp}] {level.upper()}: {message}"

    # Send errors to stderr so they can be told apart from normal output
    if level.lower() == "error":
        print(formatted_message, file=sys.stderr)
    # Send every other level to stdout
    else:
        print(formatted_message, file=sys.stdout)
