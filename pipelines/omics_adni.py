"""ADNI omics check of the MR candidates (local; aggregate outputs).

uv run python pipelines/omics_adni.py --raw "$DEMENTIA_ML_DATA/adni/raw_2026-11-13"
"""

import argparse
import json
from pathlib import Path

from dementia_ml.omics import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--window-days", type=int, default=180)
    args = ap.parse_args()
    out = run.run(args.raw, args.window_days)
    out.pop("provenance", None)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
