"""M0-07 · One logging setup: the contract of dronenav.logs.

- `setup_logging(level)` gives every script the same line format on stderr (stdout stays for results);
- it is idempotent: a second call replaces the handler and the level, it never doubles the lines;
- a bad level is an error, not a silent default;
- `log_device` writes the chosen device as one fixed line that `grep 'device='` finds in any script's log.
"""
from __future__ import annotations

import logging
import re

import pytest

from dronenav.logs import DEVICE_LINE, LOG_LEVELS, log_device, setup_logging

# "12:34:56 INFO dronenav: message": time, level, logger name, message
LINE = re.compile(r"^\d\d:\d\d:\d\d (DEBUG|INFO|WARNING|ERROR) [\w.]+: .+$")


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """setup_logging configures the root logger: put it back, so no other test sees our handler or level."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _lines(capsys) -> list[str]:
    return capsys.readouterr().err.splitlines()


def test_the_levels_are_the_cli_choices():
    assert LOG_LEVELS == ("DEBUG", "INFO", "WARNING", "ERROR")


def test_one_format_on_stderr_and_nothing_on_stdout(capsys):
    log = setup_logging("INFO")
    log.info("hello")
    logging.getLogger("dronenav.search").warning("from a module")
    out = capsys.readouterr()
    assert out.out == ""
    lines = out.err.splitlines()
    assert len(lines) == 2 and all(LINE.fullmatch(ln) for ln in lines), lines
    assert lines[0].endswith(" INFO dronenav: hello")
    assert lines[1].endswith(" WARNING dronenav.search: from a module")


def test_the_default_level_is_info(capsys):
    log = setup_logging()
    log.debug("hidden")
    log.info("shown")
    assert [ln.split(": ", 1)[1] for ln in _lines(capsys)] == ["shown"]


@pytest.mark.parametrize("level, shown", [("DEBUG", 4), ("INFO", 3), ("WARNING", 2), ("ERROR", 1)])
def test_the_level_filters(capsys, level, shown):
    log = setup_logging(level)
    for emit in (log.debug, log.info, log.warning, log.error):
        emit("x")
    assert len(_lines(capsys)) == shown


def test_the_level_name_is_case_insensitive(capsys):
    setup_logging("debug").debug("x")
    assert len(_lines(capsys)) == 1


def test_calling_setup_twice_does_not_duplicate_lines(capsys):
    before = len(logging.getLogger().handlers)
    setup_logging("INFO")
    log = setup_logging("INFO")
    assert len(logging.getLogger().handlers) == before + 1      # one handler of ours, however many calls
    log.info("once")
    assert len(_lines(capsys)) == 1


def test_a_second_call_changes_the_level(capsys):
    setup_logging("ERROR")
    log = setup_logging("DEBUG")
    log.debug("now visible")
    assert len(_lines(capsys)) == 1


def test_setup_leaves_foreign_handlers_alone():
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    setup_logging("INFO")
    setup_logging("INFO")
    assert foreign in logging.getLogger().handlers


@pytest.mark.parametrize("level", ["LOUD", "", "10", None, 15])
def test_a_bad_level_is_a_value_error_listing_the_choices(level):
    with pytest.raises(ValueError) as e:
        setup_logging(level)
    for name in LOG_LEVELS:
        assert name in str(e.value)


def test_a_bad_level_leaves_the_logging_setup_untouched():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    with pytest.raises(ValueError):
        setup_logging("LOUD")
    assert root.handlers == handlers and root.level == level


@pytest.mark.parametrize("device, requested", [("cpu", "auto"), ("mps", "auto"), ("cuda", "cuda")])
def test_the_device_line_has_one_fixed_format(capsys, device, requested):
    setup_logging("INFO")
    message = log_device(device, requested)
    assert message == f"device={device} requested={requested}"
    (line,) = _lines(capsys)
    m = DEVICE_LINE.fullmatch(line)
    assert m and m.group("device") == device and m.group("requested") == requested, line


def test_the_device_line_says_auto_when_nothing_was_requested(capsys):
    setup_logging("INFO")
    assert log_device("cpu") == "device=cpu requested=auto"


def test_the_device_line_is_info_so_warning_level_hides_it(capsys):
    setup_logging("WARNING")
    log_device("cpu")
    assert _lines(capsys) == []


def test_device_line_pattern_rejects_other_lines():
    assert DEVICE_LINE.fullmatch("12:00:00 INFO dronenav: device=cpu") is None
    assert DEVICE_LINE.fullmatch("12:00:00 INFO dronenav: device=gpu requested=auto") is None
    assert DEVICE_LINE.fullmatch("device=cpu requested=auto") is None
