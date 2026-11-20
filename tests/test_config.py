"""M0-07 · Config dataclass pattern and the shared CLI options: the contract of dronenav.config.

- a script declares its options once, as a dataclass whose fields carry default, help and argparse extras
  (`arg`); `parse_config` turns argv into a typed, frozen instance;
- `CommonArgs` / `add_shared_args` give every script the same `--log-level` and `--device`;
- `init_cli` is the one start-up call: logging set up, device picked and logged in the fixed line;
- every script in scripts/ takes `--log-level`; the ones that match frames also take `--device` and write the
  same device line (the issue's "Done when: scripts pick a device and log it the same way").
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronenav.config import CommonArgs, add_arguments, add_shared_args, arg, init_cli, parse_config
from dronenav.logs import DEVICE_LINE

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("accuracy.py", "chaos_sheet.py", "check_data.py", "confidence_check.py", "fake_run.py", "render_image.py",
           "render_video.py", "rotation_check.py", "run.py", "scale_table.py", "search_run.py", "spoof_check.py")
# the scripts that match frames to the map: a torch backend would run there, so they pick and log a device.
# The others only check, draw or write files (numpy / OpenCV / ffmpeg on the CPU): no device to pick.
DEVICE_SCRIPTS = ("scale_table.py", "search_run.py")
# run.py, confidence_check.py, accuracy.py (it localizes with --map/--frames) and rotation_check.py match too and
# take --device, but their required arguments stop them before any work when called bare,
# so only the option itself is checked here; its device line is the same init_cli() call as in the two above.
DEVICE_OPTION_SCRIPTS = (*DEVICE_SCRIPTS, "run.py", "confidence_check.py", "accuracy.py", "rotation_check.py")


def _size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.lower().split("x"))
    return w, h


@dataclass(frozen=True, kw_only=True)
class DemoArgs(CommonArgs):
    """Fly a demo."""
    map: Path = arg(help="map image")                                  # no default: required
    frames: int = arg(120, help="number of frames")
    fps: float = arg(12.0, help="frames per second")
    name: str = arg("run", help="run name")
    jam_start: int | None = arg(None, help="first jammed frame")       # Optional: None unless given
    out: Path | None = arg(None, help="output folder")
    dry_run: bool = arg(False, help="do not write anything")           # bool: a flag
    size: tuple[int, int] = arg((640, 480), help="WxH", type=_size)    # custom parser
    mode: str = arg("fast", help="quality", choices=("fast", "best"))


@pytest.fixture(autouse=True)
def _restore_root_logger():
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def no_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)      # `import torch` raises ImportError: only "cpu" exists


# ---------------------------------------------------------------- the dataclass pattern


def test_defaults_come_from_the_dataclass():
    cfg = parse_config(DemoArgs, ["--map", "m.png"])
    assert cfg == DemoArgs(map=Path("m.png"))
    assert (cfg.frames, cfg.fps, cfg.name, cfg.jam_start, cfg.out, cfg.dry_run, cfg.size, cfg.mode) == \
        (120, 12.0, "run", None, None, False, (640, 480), "fast")


def test_values_are_converted_to_the_annotated_types():
    cfg = parse_config(DemoArgs, ["--map", "m.png", "--frames", "7", "--fps", "30", "--name", "x", "--jam-start", "3",
                                  "--out", "o", "--dry-run", "--size", "32x16", "--mode", "best"])
    assert cfg == DemoArgs(map=Path("m.png"), frames=7, fps=30.0, name="x", jam_start=3, out=Path("o"),
                           dry_run=True, size=(32, 16), mode="best")
    assert isinstance(cfg.map, Path) and isinstance(cfg.fps, float) and isinstance(cfg.frames, int)


def test_the_result_is_a_frozen_instance_of_the_class():
    cfg = parse_config(DemoArgs, ["--map", "m.png"])
    assert type(cfg) is DemoArgs
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.frames = 1


def test_underscores_become_dashes_on_the_command_line():
    assert parse_config(DemoArgs, ["--map", "m", "--jam-start", "5"]).jam_start == 5
    with pytest.raises(SystemExit):
        parse_config(DemoArgs, ["--map", "m", "--jam_start", "5"])


@pytest.mark.parametrize("argv, word", [
    ([], "--map"),                                        # a field without a default is required
    (["--map", "m", "--frames", "many"], "--frames"),     # wrong type
    (["--map", "m", "--mode", "slow"], "--mode"),         # not one of the choices
    (["--map", "m", "--nope", "1"], "--nope"),            # unknown option
])
def test_bad_arguments_exit_2_and_name_the_option(capsys, argv, word):
    with pytest.raises(SystemExit) as e:
        parse_config(DemoArgs, argv)
    assert e.value.code == 2
    assert word in capsys.readouterr().err


def test_help_shows_every_option_its_help_and_its_default(capsys):
    with pytest.raises(SystemExit) as e:
        parse_config(DemoArgs, ["--help"])
    assert e.value.code == 0
    text = " ".join(capsys.readouterr().out.split())      # argparse wraps lines: compare without the wrapping
    assert "Fly a demo." in text                          # the class docstring is the description
    for option in ("--map", "--frames", "--fps", "--name", "--jam-start", "--out", "--dry-run", "--size", "--mode",
                   "--device", "--log-level"):
        assert option in text, option
    assert "number of frames (default: 120)" in text
    assert "map image (required)" in text


def test_the_description_can_be_overridden(capsys):
    with pytest.raises(SystemExit):
        parse_config(DemoArgs, ["--help"], description="Another text")
    assert "Another text" in capsys.readouterr().out


def test_a_config_inherits_the_shared_options():
    cfg = parse_config(DemoArgs, ["--map", "m"])
    assert (cfg.device, cfg.log_level) == ("auto", "INFO")
    cfg = parse_config(DemoArgs, ["--map", "m", "--device", "CPU", "--log-level", "debug"])
    assert (cfg.device, cfg.log_level) == ("cpu", "DEBUG")           # normalised, so code compares one spelling


@pytest.mark.parametrize("argv", [["--device", "gpu"], ["--log-level", "loud"]])
def test_bad_shared_options_exit_2(capsys, argv):
    with pytest.raises(SystemExit) as e:
        parse_config(DemoArgs, ["--map", "m", *argv])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_a_field_declared_without_arg_is_an_error():
    @dataclass(frozen=True, kw_only=True)
    class Plain:
        frames: int = 3                                  # no help text: every option must explain itself

    with pytest.raises(TypeError, match="frames"):
        parse_config(Plain, [])


def test_an_unsupported_annotation_needs_an_explicit_type():
    @dataclass(frozen=True, kw_only=True)
    class Odd:
        size: tuple[int, int] = arg((1, 2), help="WxH")  # no type=: argparse cannot guess how to read a tuple

    with pytest.raises(TypeError, match="size"):
        parse_config(Odd, [])


def test_add_arguments_extends_an_existing_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=int, default=1)
    add_arguments(ap, DemoArgs)
    a = ap.parse_args(["--map", "m", "--frames", "9"])
    assert (a.legacy, a.frames, a.map, a.device, a.log_level) == (1, 9, Path("m"), "auto", "INFO")


# ---------------------------------------------------------------- shared options on a hand-written parser


def test_add_shared_args_adds_both_options_with_defaults():
    ap = argparse.ArgumentParser()
    add_shared_args(ap)
    assert vars(ap.parse_args([])) == {"device": "auto", "log_level": "INFO"}
    assert vars(ap.parse_args(["--device", "mps", "--log-level", "ERROR"])) == {"device": "mps", "log_level": "ERROR"}


def test_add_shared_args_without_device_is_for_scripts_that_only_draw():
    ap = argparse.ArgumentParser()
    add_shared_args(ap, device=False)
    assert vars(ap.parse_args(["--log-level", "warning"])) == {"log_level": "WARNING"}
    with pytest.raises(SystemExit):
        ap.parse_args(["--device", "cpu"])


# ---------------------------------------------------------------- init_cli: logging + device in one call


def test_init_cli_sets_up_logging_picks_the_device_and_logs_the_line(capsys, no_torch):
    device = init_cli(parse_config(DemoArgs, ["--map", "m"]))
    assert device == "cpu"
    (line,) = capsys.readouterr().err.splitlines()
    m = DEVICE_LINE.fullmatch(line)
    assert m and (m.group("device"), m.group("requested")) == ("cpu", "auto"), line


def test_init_cli_honours_the_override(capsys, monkeypatch):
    fake = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
                           cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert init_cli(SimpleNamespace(device="auto", log_level="INFO")) == "mps"
    assert init_cli(SimpleNamespace(device="cuda", log_level="INFO")) == "cuda"
    lines = capsys.readouterr().err.splitlines()
    assert [DEVICE_LINE.fullmatch(ln).group("device", "requested") for ln in lines] == \
        [("mps", "auto"), ("cuda", "cuda")]


def test_init_cli_exits_2_with_a_clear_message_when_the_device_is_unavailable(capsys, no_torch):
    with pytest.raises(SystemExit) as e:
        init_cli(SimpleNamespace(device="mps", log_level="INFO"))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "mps" in err and "not available" in err and "Traceback" not in err
    assert not any(DEVICE_LINE.fullmatch(ln) for ln in err.splitlines())      # nothing was picked: no device line


def test_init_cli_without_a_device_option_only_sets_up_logging(capsys, no_torch):
    assert init_cli(SimpleNamespace(log_level="DEBUG")) is None
    logging.getLogger("dronenav").debug("x")
    (line,) = capsys.readouterr().err.splitlines()
    assert line.endswith(" DEBUG dronenav: x")


def test_init_cli_applies_the_log_level(capsys, no_torch):
    assert init_cli(SimpleNamespace(device="auto", log_level="WARNING")) == "cpu"
    assert capsys.readouterr().err == ""                  # the device line is INFO


# ---------------------------------------------------------------- the scripts


def _script(name: str, *args: str) -> subprocess.CompletedProcess:
    """Run scripts/<name> as a user would, with this checkout's src/ first on the import path."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args],
                          check=False, capture_output=True, text=True, env=env, cwd=ROOT)


def _device_lines(text: str) -> list:
    return [m for m in map(DEVICE_LINE.fullmatch, text.splitlines()) if m]


def test_the_contract_covers_every_script():
    assert {p.name for p in (ROOT / "scripts").glob("*.py")} == set(SCRIPTS)


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_script_exposes_the_shared_log_level(name):
    res = _script(name, "--help")
    assert res.returncode == 0, res.stderr
    assert "--log-level" in res.stdout
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        assert level in res.stdout


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_script_rejects_a_bad_log_level(name):
    res = _script(name, "--log-level", "loud")
    assert res.returncode == 2 and "--log-level" in res.stderr and "Traceback" not in res.stderr, res.stderr


@pytest.mark.parametrize("name", DEVICE_OPTION_SCRIPTS)
def test_matching_scripts_expose_the_device_option(name):
    res = _script(name, "--help")
    assert res.returncode == 0, res.stderr
    assert "--device" in res.stdout
    for device in ("auto", "mps", "cuda", "cpu"):
        assert device in res.stdout


@pytest.mark.parametrize("name", DEVICE_SCRIPTS)
def test_matching_scripts_log_the_same_device_line(name):
    # no dataset given: scale_table has nothing to measure, search_run stops with a usage error. Either way the
    # device is picked and logged first, which is all this test needs (and it needs no data/).
    res = _script(name, "--device", "cpu")
    (m,) = _device_lines(res.stderr)
    assert m.group("device", "requested") == ("cpu", "cpu")
    assert "device=" not in res.stdout                    # logs go to stderr; stdout stays for the results
    assert "Traceback" not in res.stderr


@pytest.mark.parametrize("name", DEVICE_SCRIPTS)
def test_matching_scripts_pick_a_device_by_default(name):
    res = _script(name)                                   # whatever this machine has: one of the three names
    (m,) = _device_lines(res.stderr)
    assert m.group("requested") == "auto" and m.group("device") in ("mps", "cuda", "cpu")


@pytest.mark.parametrize("name", DEVICE_OPTION_SCRIPTS)
def test_matching_scripts_reject_an_unknown_device(name):
    res = _script(name, "--device", "gpu")
    assert res.returncode == 2 and "invalid choice" in res.stderr and not _device_lines(res.stderr)


@pytest.mark.parametrize("name", DEVICE_SCRIPTS)
def test_the_log_level_silences_the_device_line(name):
    res = _script(name, "--device", "cpu", "--log-level", "warning")
    assert not _device_lines(res.stderr)


def test_scale_table_and_search_run_keep_their_behaviour_without_a_dataset():
    res = _script("scale_table.py", "--device", "cpu")
    assert res.returncode == 0 and res.stdout.startswith("Rule: factor = scale / ")
    res = _script("search_run.py", "--device", "cpu")
    assert res.returncode == 2 and "give --task or --irpin" in res.stderr


@pytest.mark.parametrize("name, args, word", [
    ("fake_run.py", ("--map", "nope.png", "--out", "o"), "nope.png"),
    ("render_image.py", ("--run", "no_such_run"), "run.json"),
    ("render_video.py", ("--run", "no_such_run"), "run.json"),
])
def test_drawing_scripts_accept_the_log_level_and_fail_as_before(tmp_path, name, args, word):
    args = tuple(str(tmp_path / a) if not a.startswith("--") else a for a in args)
    res = _script(name, *args, "--log-level", "debug")
    assert res.returncode == 2 and word in res.stderr and "Traceback" not in res.stderr, res.stderr
    assert not _device_lines(res.stderr)                  # no device to pick in a script that only draws
