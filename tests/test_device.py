"""M0-07 · Device helper: the contract of dronenav.device.

- `pick_device()` prefers MPS, then CUDA, then CPU;
- an explicit override wins, and asking for a device that is not there fails loudly (never a silent fallback);
- the module imports without torch (the classic pipeline needs none), and a missing torch means "cpu".

No test here looks at the real hardware: torch is replaced in sys.modules by a stand-in with the two probes
dronenav.device may ask (`torch.backends.mps.is_available`, `torch.cuda.is_available`), or by None, which makes
`import torch` raise ImportError.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dronenav.device import DEVICES, DeviceUnavailableError, available_devices, pick_device

ROOT = Path(__file__).resolve().parents[1]


def _torch(mps: bool = False, cuda: bool = False) -> SimpleNamespace:
    return SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                           cuda=SimpleNamespace(is_available=lambda: cuda))


@pytest.fixture
def torch_with(monkeypatch):
    """Install a stand-in torch: torch_with(mps=True), or torch_with(None) for 'torch is not installed'."""
    def install(*args, **probes):
        monkeypatch.setitem(sys.modules, "torch", None if args == (None,) else _torch(**probes))
    return install


def test_the_device_names_are_the_cli_choices():
    assert DEVICES == ("auto", "mps", "cuda", "cpu")


@pytest.mark.parametrize("probes, expected", [
    ({"mps": True, "cuda": True}, "mps"),      # MPS first
    ({"mps": True, "cuda": False}, "mps"),
    ({"mps": False, "cuda": True}, "cuda"),    # then CUDA
    ({"mps": False, "cuda": False}, "cpu"),    # then CPU
])
@pytest.mark.parametrize("override", [None, "auto"])
def test_auto_prefers_mps_then_cuda_then_cpu(torch_with, probes, expected, override):
    torch_with(**probes)
    assert pick_device(override) == expected


def test_available_devices_are_listed_best_first_and_always_end_with_cpu(torch_with):
    torch_with(mps=True, cuda=True)
    assert available_devices() == ("mps", "cuda", "cpu")
    torch_with(cuda=True)
    assert available_devices() == ("cuda", "cpu")
    torch_with()
    assert available_devices() == ("cpu",)


@pytest.mark.parametrize("override", ["cpu", "cuda", "mps"])
def test_an_explicit_override_wins_over_the_preference_order(torch_with, override):
    torch_with(mps=True, cuda=True)
    assert pick_device(override) == override


def test_the_override_is_case_and_space_insensitive(torch_with):
    torch_with(mps=True, cuda=True)
    assert pick_device(" CUDA ") == "cuda"
    assert pick_device("Auto") == "mps"


@pytest.mark.parametrize("override, probes", [("mps", {"cuda": True}), ("cuda", {"mps": True}), ("mps", {})])
def test_an_unavailable_device_fails_loudly_instead_of_falling_back(torch_with, override, probes):
    torch_with(**probes)
    with pytest.raises(DeviceUnavailableError) as e:
        pick_device(override)
    # the message names what was asked for and what can be used instead
    assert override in str(e.value) and "not available" in str(e.value)
    for name in available_devices():
        assert name in str(e.value)


@pytest.mark.parametrize("override", ["gpu", "cuda:0", "", "tpu"])
def test_an_unknown_device_name_is_a_value_error_listing_the_choices(torch_with, override):
    torch_with(mps=True, cuda=True)
    with pytest.raises(ValueError) as e:
        pick_device(override)
    assert not isinstance(e.value, DeviceUnavailableError)
    for name in DEVICES:
        assert name in str(e.value)


def test_without_torch_auto_is_cpu(torch_with):
    torch_with(None)
    assert pick_device() == "cpu"
    assert pick_device("auto") == "cpu"
    assert pick_device("cpu") == "cpu"
    assert available_devices() == ("cpu",)


@pytest.mark.parametrize("override", ["mps", "cuda"])
def test_without_torch_asking_for_an_accelerator_says_torch_is_missing(torch_with, override):
    torch_with(None)
    with pytest.raises(DeviceUnavailableError, match="torch is not installed"):
        pick_device(override)


def test_a_torch_build_without_the_mps_backend_is_not_an_error(monkeypatch):
    # torch older than 1.12 has no torch.backends.mps: that means "no MPS", not a crash
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(backends=SimpleNamespace(), cuda=SimpleNamespace(is_available=lambda: True)))
    assert pick_device() == "cuda"


def test_importing_the_module_does_not_import_torch():
    # a fresh interpreter: only then does sys.modules tell what the import itself pulled in
    code = "import sys, dronenav.device; sys.exit('torch imported' if 'torch' in sys.modules else 0)"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    res = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
