"""Check that outputs/backups/ has every recording the pitch needs as a stand-in for a live step (M8-05).

    python scripts/check_backups.py
    python scripts/check_backups.py --backups /path/to/backups

Prints every missing or truncated recording by name. Run this on both machines before the pitch - that is the
"on two machines" half of "Done when" (see NOTES.md, "Backup recordings"). Exit code 0: nothing wrong.
1: problems found (printed). 2: bad arguments or the backups folder itself is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dronenav.check_backups import check_backups
from dronenav.config import add_shared_args, init_cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backups", default="outputs/backups", help="the backups folder to check (default: outputs/backups)")
    add_shared_args(ap, device=False)   # only reads the backups folder: logging yes, no device to pick
    a = ap.parse_args(argv)
    init_cli(a)

    report = check_backups(a.backups)
    for p in sorted(report.problems, key=lambda p: p.path):
        print(f"{p.kind}: {p.path} ({p.detail})")
    if not Path(a.backups).is_dir():
        return 2
    if not report.ok:
        print(f"check_backups: {len(report.problems)} problem(s) in {report.checked} recording(s) checked",
              file=sys.stderr)
        return 1
    print(f"check_backups: ok ({report.checked} recording(s) checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
