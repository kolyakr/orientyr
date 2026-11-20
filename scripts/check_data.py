"""Check that data/ has the layout the team's Google Drive mirrors (M0-06); see data/README.md.

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data", help="the data folder to check (default: data)")
    a = ap.parse_args(argv)

    report = check_data(a.data)
    for p in sorted(report.problems, key=lambda p: p.path):
        print(f"{p.kind}: {p.path} ({p.detail})")
    if not Path(a.data).is_dir():
        return 2
    if not report.ok:
        print(f"check_data: {len(report.problems)} problem(s) in {report.checked} files/dirs checked",
              file=sys.stderr)
        return 1
    print(f"check_data: ok ({report.checked} files/dirs checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
