"""Check the optional benchmark data layout described in data/README.md.

    python scripts/check_data.py
    python scripts/check_data.py --data /path/to/data

Prints every missing or truncated file by name. Exit code 0: nothing wrong. 1: problems found (printed). 2: bad
arguments or the data folder itself is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dronenav.check_data import check_data
from dronenav.config import add_shared_args, init_cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data", help="the data folder to check (default: data)")
    add_shared_args(ap, device=False)   # only reads the data folder: logging yes, no device to pick
    a = ap.parse_args(argv)
    init_cli(a)

    report = check_data(a.data)
    for p in sorted(report.problems, key=lambda p: p.path):
        print(f"{p.kind}: {p.path} ({p.detail})")
    if not Path(a.data).is_dir():
        return 2
    if not report.ok:
        print(f"check_data: {len(report.problems)} problem(s) in {report.checked} files checked",
              file=sys.stderr)
        return 1
    print(f"check_data: ok ({report.checked} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
