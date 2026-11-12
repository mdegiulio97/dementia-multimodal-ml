"""Refuse to commit files that look like individual-level ADNI data.

Usage: python scripts/check_no_adni.py FILE [FILE ...]
Exit code 1 if any file is under results/adni, figures/adni or models/adni,
is a data file with 'ADNIMERGE' in its name, or is a CSV/TSV/Parquet table with an RID or
PTID column. Only headers/schemas are read, never the data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PRIVATE_DIRS = ("results/adni/", "figures/adni/", "models/adni/")
DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".xlsx",
    ".xls",
    ".rda",
    ".rdata",
    ".zip",
    ".gz",
}
ID_COLUMNS = {"RID", "PTID"}


def _header(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                first = fh.readline()
            delimiter = "\t" if "\t" in first else ","
            return {
                c.strip().strip('"') for c in next(csv.reader([first], delimiter=delimiter), [])
            }
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            return set(pq.read_schema(path).names)
    except (OSError, ValueError):
        return set()
    return set()


def offending_reason(name: str) -> str | None:
    posix = name.replace("\\", "/")
    if any(d in posix for d in PRIVATE_DIRS):
        return "private ADNI output directory"
    if "ADNIMERGE" in Path(posix).name.upper() and Path(posix).suffix.lower() in DATA_SUFFIXES:
        return "ADNIMERGE file"
    path = Path(name)
    if path.is_file() and _header(path) & ID_COLUMNS:
        return "table with RID/PTID column"
    return None


def main(argv: list[str]) -> int:
    bad = [(f, r) for f in argv if (r := offending_reason(f))]
    for f, reason in bad:
        print(f"refused: {f} ({reason})", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
