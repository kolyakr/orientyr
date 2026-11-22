"""Command-line options declared once, as a dataclass (M0-07).

A script writes its options as fields, with the default and the help text next to the name, and gets back a
typed, frozen instance:

    @dataclass(frozen=True, kw_only=True)
    class Args(CommonArgs):                       # CommonArgs brings --device and --log-level
        '''Localise a folder of frames on a map.'''
        map: Path = arg(help="map image")                       # no default: required
        frames: int = arg(50, help="number of frames")          # --frames, int, default 50
        out: Path | None = arg(None, help="output folder")      # optional, None unless given
        dry_run: bool = arg(False, help="write nothing")        # a flag: --dry-run

    def main(argv=None) -> int:
        args = parse_config(Args, argv)
        device = init_cli(args)                   # logging set up, device picked and logged
        ...

``kw_only=True`` lets a required field follow the defaults inherited from CommonArgs. A script that already has a
hand-written argparse parser joins in with two lines instead: ``add_shared_args(parser)`` and ``init_cli(args)``.
"""
from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dronenav.device import DEVICES, DeviceUnavailableError, pick_device
from dronenav.logs import LOG_LEVELS, log_device, setup_logging

T = TypeVar("T")
_SIMPLE = (int, float, str, Path)        # annotations argparse can read with the type itself as the converter


def arg(default: Any = dataclasses.MISSING, *, help: str, **extras: Any) -> Any:
    """A dataclass field that is also a command-line option.

    No ``default`` makes the option required. ``extras`` go to ``parser.add_argument`` as they are
    (``choices=``, ``type=`` for a value argparse cannot read from the annotation, ``metavar=`` ...).
    """
    return dataclasses.field(default=default, metadata={"help": help, "argparse": extras})


def _value_type(hint: Any) -> Any:
    """``int | None`` → ``int``: on the command line an optional value is just a value that may be absent."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        rest = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(rest) == 1:
            return rest[0]
    return hint


def _add_field(parser: argparse.ArgumentParser, owner: type, field: dataclasses.Field, hint: Any) -> None:
    if "help" not in field.metadata:
        raise TypeError(f"{owner.__name__}.{field.name}: declare it with arg(..., help=...), "
                        "so every option explains itself in --help")
    kwargs: dict[str, Any] = dict(field.metadata["argparse"])
    text = field.metadata["help"].replace("%", "%%")        # argparse %-formats help strings
    required = field.default is dataclasses.MISSING
    kind = _value_type(hint)
    if kind is bool and "type" not in kwargs:
        # a flag: --dry-run for a False default; --no-cache as well when the default is True
        kwargs["action"] = argparse.BooleanOptionalAction if field.default is True else "store_true"
    else:
        if "type" not in kwargs:
            if kind not in _SIMPLE:
                raise TypeError(f"{owner.__name__}.{field.name}: argparse cannot read a {hint} from text; "
                                "pass arg(..., type=<function that parses the string>)")
            kwargs["type"] = kind
        if required:
            text += " (required)"
        elif field.default is not None:
            text += f" (default: {field.default})".replace("%", "%%")
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = field.default
    parser.add_argument("--" + field.name.replace("_", "-"), dest=field.name, help=text, **kwargs)


def add_arguments(parser: argparse.ArgumentParser, cls: type, only: tuple[str, ...] | None = None) -> None:
    """Add the dataclass's fields to ``parser`` as ``--options`` (all of them, or just the names in ``only``)."""
    hints = typing.get_type_hints(cls)
    for field in dataclasses.fields(cls):
        if only is None or field.name in only:
            _add_field(parser, cls, field, hints[field.name])


def parse_config(cls: type[T], argv: list[str] | None = None, *, description: str | None = None) -> T:
    """Parse ``argv`` (default: sys.argv) into an instance of the dataclass ``cls``.

    The description in ``--help`` is ``description`` or the first line of the class docstring. Bad arguments end
    the program the argparse way: usage on stderr, exit code 2.
    """
    doc = (cls.__doc__ or "").strip()
    if doc.startswith(f"{cls.__name__}("):       # the signature dataclass() writes when there is no docstring
        doc = ""
    # allow_abbrev=False: "--frame" must not quietly mean "--frames" in a command someone copies into a README
    parser = argparse.ArgumentParser(description=description or doc.split("\n")[0] or None, allow_abbrev=False)
    add_arguments(parser, cls)
    ns = parser.parse_args(argv)
    return cls(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(cls)})


@dataclass(frozen=True, kw_only=True)
class CommonArgs:
    """The options every script shares. Subclass it (see the module docstring) or use add_shared_args."""

    # type=str.lower / str.upper: "--device CPU" and "--log-level debug" work, and code sees one spelling
    device: str = arg("auto", help="compute device for torch models; auto = MPS, then CUDA, then CPU. "
                                   "Asking for one that is missing is an error",
                      choices=DEVICES, type=str.lower)
    log_level: str = arg("INFO", help="how much to log to stderr", choices=LOG_LEVELS, type=str.upper)


def add_shared_args(parser: argparse.ArgumentParser, *, device: bool = True) -> None:
    """Add ``--log-level`` and ``--device`` to a hand-written parser, identical to CommonArgs.

    ``device=False`` is for a script with nothing to run on a device (it only draws or writes files): an option
    that changes nothing would be a lie in its --help.
    """
    add_arguments(parser, CommonArgs, only=("device", "log_level") if device else ("log_level",))


def init_cli(args: Any) -> str | None:
    """The one start-up call of a script: set up logging, pick the device, log it. Returns the device name.

    ``args`` is anything with ``log_level`` and, when the script takes one, ``device`` (a CommonArgs or an
    argparse namespace). Without a ``device`` attribute only logging is set up and None is returned.
    A device that cannot be used ends the program with exit code 2 and the reason on stderr, like any other bad
    argument; code that wants the exception calls dronenav.device.pick_device itself.
    """
    log = setup_logging(args.log_level)
    requested = getattr(args, "device", None)
    if requested is None:
        return None
    try:
        device = pick_device(requested)
    except (ValueError, DeviceUnavailableError) as e:
        log.error("%s", e)
        raise SystemExit(2) from None
    log_device(device, requested)
    return device
