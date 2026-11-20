"""Which compute device to run on: MPS, then CUDA, then CPU, unless the user says otherwise (M0-07).

    device = pick_device(args.device)        # "mps" | "cuda" | "cpu"

Only the deep-learning matchers need torch, so this module never imports it at import time: the classic pipeline
must run on a machine without torch. No torch simply means the only device is "cpu".
"""
from __future__ import annotations

DEVICES = ("auto", "mps", "cuda", "cpu")     # what --device accepts; "auto" = the best one available
_PREFERENCE = ("mps", "cuda", "cpu")         # Apple Silicon first (the team's laptops), then NVIDIA, then anything


class DeviceUnavailableError(RuntimeError):
    """A device was asked for by name and this machine (or this torch build) does not have it."""


def _torch():
    """The torch module, or None when it is not installed. Imported here, late, so `import dronenav` stays light."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def available_devices() -> tuple[str, ...]:
    """The devices usable right now, best first; always ends with "cpu"."""
    torch = _torch()
    found = []
    if torch is not None:
        # getattr: torch.backends.mps appeared in torch 1.12; an older build has no MPS rather than a crash
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            found.append("mps")
        if torch.cuda.is_available():
            found.append("cuda")
    return (*found, "cpu")


def pick_device(override: str | None = None) -> str:
    """The device name to hand to torch: ``override`` if given, else the first available of MPS, CUDA, CPU.

    ``override`` is one of DEVICES in any letter case; None and "auto" mean "choose for me".
    Raises ValueError for a name that is not in DEVICES, and DeviceUnavailableError when the named device is not
    there. It never falls back silently: someone who asks for the GPU and gets the CPU would only notice from a
    run that is ten times slower.
    """
    name = "auto" if override is None else override.strip().lower()
    if name not in DEVICES:
        raise ValueError(f"unknown device {override!r}: choose one of {', '.join(DEVICES)}")
    available = available_devices()
    if name == "auto":
        return next(d for d in _PREFERENCE if d in available)
    if name not in available:
        why = "torch is not installed" if _torch() is None else "torch does not see it on this machine"
        raise DeviceUnavailableError(
            f"device {name!r} is not available ({why}); available: {', '.join(available)}. "
            "Use --device auto to take the best one.")
    return name
