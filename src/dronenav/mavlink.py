"""MAVLink ``GPS_INPUT`` from our fixes: what an autopilot would receive in place of a jammed GPS (M9-01, a stub).

ArduPilot and PX4 accept an external position as ``GPS_INPUT`` (ArduPilot: ``GPS_TYPE = 14``, "MAV"), as if a GPS
receiver sent it. Here each accepted, measured fix of a run becomes one such message:

- **the fix, not the track.** The autopilot's own EKF fuses the messages with its IMU; sending our already
  filtered position would make it count the same evidence twice and trust it too much. Frames with no accepted
  fix (and interpolated ones) send nothing: the autopilot coasts, as on a GPS dropout;
- **lat/lon** from the map's georeference (:mod:`dronenav.georef`) at the fix's centre;
  **horiz_accuracy** is the fix's σ (:func:`dronenav.confidence.fix_sigma`, M4-08) in metres;
- **no altitude and no velocity**: a homography against a map gives neither honestly, so their ignore flags are
  set and the fix is 2D (``fix_type`` 2); ``satellites_visible`` is 0, since none were;
- **time:** ``time_usec`` is the frame's UNIX time (a start time + the record's ``t``); ``time_week`` and
  ``time_week_ms`` are the same instant in GPS time (UTC + 18 leap seconds, counted from 1980-01-06).

The messages are written as a ``.tlog`` (each MAVLink 2 packet after an 8-byte big-endian µs timestamp), which
pymavlink, MAVExplorer and Mission Planner replay, and can also be sent over UDP to a SITL simulator. The log
opens with a STATUSTEXT that says what this is: a stub, never flown with a real autopilot.
"""
from __future__ import annotations

import datetime as dt
import struct
from collections.abc import Iterable
from pathlib import Path

from pymavlink.dialects.v20 import common as mav

from dronenav.confidence import fix_sigma
from dronenav.georef import GeorefBlock
from dronenav.telemetry import Record

__all__ = ["STUB_TEXT", "gps_input", "gps_time", "run_to_tlog"]

STUB_TEXT = "orientyr GPS_INPUT stub, not flight-tested"     # fits STATUSTEXT's 50 characters
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)
LEAP_SECONDS = 18                                             # GPS - UTC since 2017-01-01; no change announced
_WEEK_S = 7 * 86400
_SYSTEM_ID, _COMPONENT_ID = 1, mav.MAV_COMP_ID_ONBOARD_COMPUTER   # a companion computer next to the autopilot
_IGNORE = (mav.GPS_INPUT_IGNORE_FLAG_ALT | mav.GPS_INPUT_IGNORE_FLAG_HDOP | mav.GPS_INPUT_IGNORE_FLAG_VDOP
           | mav.GPS_INPUT_IGNORE_FLAG_VEL_HORIZ | mav.GPS_INPUT_IGNORE_FLAG_VEL_VERT
           | mav.GPS_INPUT_IGNORE_FLAG_SPEED_ACCURACY | mav.GPS_INPUT_IGNORE_FLAG_VERTICAL_ACCURACY)
_FIX_2D = 2


def gps_time(when: dt.datetime) -> tuple[int, int]:
    """A UTC instant -> (GPS week, milliseconds into that week)."""
    ms = round(((when - GPS_EPOCH).total_seconds() + LEAP_SECONDS) * 1000)
    week, week_ms = divmod(ms, _WEEK_S * 1000)
    return int(week), int(week_ms)


def gps_input(rec: Record, block: GeorefBlock, start: dt.datetime, gps_id: int = 0):
    """The ``GPS_INPUT`` for one frame, or None when it has no accepted, measured fix.

    block: the georeference of the map the run's pixels are in. start: the UTC time of the run's ``t = 0``.
    """
    fix = rec.track.fix
    if fix is None or not fix.accepted or rec.interpolated:
        return None
    when = start + dt.timedelta(seconds=rec.t)
    week, week_ms = gps_time(when)
    lon, lat = block.px_to_lonlat(fix.x, fix.y)
    return mav.MAVLink_gps_input_message(
        time_usec=round(when.timestamp() * 1e6), gps_id=gps_id, ignore_flags=_IGNORE,
        time_week_ms=week_ms, time_week=week, fix_type=_FIX_2D,
        lat=round(lat * 1e7), lon=round(lon * 1e7), alt=0.0, hdop=0.0, vdop=0.0, vn=0.0, ve=0.0, vd=0.0,
        speed_accuracy=0.0, horiz_accuracy=fix_sigma(fix) * block.m_per_px, vert_accuracy=0.0,
        satellites_visible=0, yaw=0)   # yaw 0 = "not available": the camera's heading is not the airframe's


def run_to_tlog(records: Iterable[Record], block: GeorefBlock, out: str | Path, start: dt.datetime,
                udp: str | None = None) -> int:
    """Write the run's ``GPS_INPUT`` messages to a ``.tlog`` (and to ``udp`` "host:port" if given).

    Returns the number of ``GPS_INPUT`` messages (the STATUSTEXT that opens the log is not counted).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    link = mav.MAVLink(None, srcSystem=_SYSTEM_ID, srcComponent=_COMPONENT_ID)
    sock = None
    if udp is not None:
        from pymavlink import mavutil  # only when sending: it opens a socket
        sock = mavutil.mavlink_connection(f"udpout:{udp}", source_system=_SYSTEM_ID,
                                          source_component=_COMPONENT_ID)
    n = 0
    try:
        with out.open("wb") as f:
            def emit(msg, usec: int) -> None:
                packet = msg.pack(link)
                link.seq = (link.seq + 1) % 256       # pack() alone does not advance the sequence
                f.write(struct.pack(">Q", usec) + packet)
                if sock is not None:
                    sock.write(packet)

            emit(mav.MAVLink_statustext_message(mav.MAV_SEVERITY_NOTICE, STUB_TEXT.encode()),
                 round(start.timestamp() * 1e6))
            for rec in records:
                msg = gps_input(rec, block, start)
                if msg is not None:
                    emit(msg, msg.time_usec)
                    n += 1
    finally:
        if sock is not None:
            sock.close()
    return n
