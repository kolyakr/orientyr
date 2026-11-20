"""One logging setup for every script, and the one line that says which device a run uses (M0-07).

    log = setup_logging(args.log_level)
    log_device(pick_device(args.device), args.device)      # 12:34:56 INFO dronenav: device=mps requested=auto

Logs go to stderr, so stdout stays clean for a script's results (tables, paths) and can be piped.
"""
from __future__ import annotations

import logging
import re
import sys

from dronenav.device import DEVICES

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")       # what --log-level accepts
LOGGER_NAME = "dronenav"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"                                  # a run lasts minutes: the date would only be noise

# A whole log line with the device, for tests and for grepping logs: `grep ' device='`.
DEVICE_LINE = re.compile(
    rf"^\d\d:\d\d:\d\d INFO {LOGGER_NAME}: "
    rf"device=(?P<device>{'|'.join(DEVICES[1:])}) requested=(?P<requested>{'|'.join(DEVICES)})$")


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is when the record is emitted, not when the handler was made.

    The class also marks the handler as ours, so setup_logging can replace it and leave foreign handlers alone;
    and the late lookup keeps the output visible when stderr is swapped after the setup (pytest, notebooks).
    """

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:       # StreamHandler.__init__ assigns it; there is nothing to keep
        pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for a script and return the package logger ("dronenav").

    ``level`` is one of LOG_LEVELS, in any letter case; anything else is a ValueError (and changes nothing).
    Safe to call again: the second call replaces our handler and the level, so no line is ever printed twice.
    """
    name = level.upper() if isinstance(level, str) else None
    if name not in LOG_LEVELS:
        raise ValueError(f"unknown log level {level!r}: choose one of {', '.join(LOG_LEVELS)}")
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
    # the root logger, so modules that log under their own names (dronenav.search, a library) share the format
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if not isinstance(h, _StderrHandler)] + [handler]
    root.setLevel(name)
    return logging.getLogger(LOGGER_NAME)


def log_device(device: str, requested: str | None = None) -> str:
    """Log the chosen device in the one fixed format and return the message: ``device=mps requested=auto``.

    ``requested`` is what the user asked for (None = nothing = "auto"); showing both tells an explicit choice
    from an automatic one when two runs are compared.
    """
    message = f"device={device} requested={requested or 'auto'}"
    logging.getLogger(LOGGER_NAME).info(message)
    return message
