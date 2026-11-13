"""Build $DEMENTIA_ML_DATA/adni/ADNIMERGE.csv from the ADNIMERGE2 package (local only).

    uv run python pipelines/prepare_adni.py --raw "$DEMENTIA_ML_DATA/adni/raw_2026-11-13"

Prints aggregate counts only; the table never leaves the data folder.
"""

import argparse
import json
from pathlib import Path

from dementia_ml.data import adnimerge2, schema
from dementia_ml.paths import data_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True, help="unpacked ADNI download folder")
    ap.add_argument("--window-days", type=int, default=180)
    args = ap.parse_args()
    df = adnimerge2.build(adnimerge2.load_tables(args.raw), window_days=args.window_days)
    schema.validate(df)
    out = data_root() / "adni" / "ADNIMERGE.csv"
    df.to_csv(out, index=False)
    bl = df[df["VISCODE"] == "bl"]
    mci = bl[bl["DX_bl"].isin(schema.DX_BL_MCI)]
    summary = {
        "rows": len(df),
        "participants": int(df["RID"].nunique()),
        "baseline_dx": bl["DX_bl"].value_counts().to_dict(),
        "mci_baseline": len(mci),
        "mci_with_hippocampus": int(mci["Hippocampus"].notna().sum()),
        "mci_with_csf": int(mci["ABETA"].notna().sum()),
        "mci_with_apoe": int(mci["APOE4"].notna().sum()),
        "source": str(args.raw.name),
        "written": str(out),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
