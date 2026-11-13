"""ADNI multimodal fusion for MCI-to-dementia conversion (local; aggregate outputs).

uv run python pipelines/fusion_adni.py --raw "$DEMENTIA_ML_DATA/adni/raw_2026-11-13"
"""

import argparse
from pathlib import Path

from dementia_ml.fusion import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    out = run.run(ap.parse_args().raw, ap.parse_args().n_boot)
    print("coverage", out["coverage"], "complete-case", out["n_complete_case"])
    for combo, kinds in out["results"].items():
        for kind, r in kinds.items():
            c, d = r["c_index"], r.get("delta_vs_reference")
            delta = (
                f"  dC vs clinical {d['estimate']:+.3f} ({d['ci_low']:+.3f}, {d['ci_high']:+.3f})"
                if d
                else ""
            )
            print(
                f"{combo:26s} {kind:4s} C {c['estimate']:.3f} ({c['ci_low']:.3f}-{c['ci_high']:.3f}){delta}"
            )
    if out["complete_case"]:
        for combo, kinds in out["complete_case"].items():
            c = kinds["cox"]["c_index"]
            print(
                f"[complete-case] {combo:20s} C {c['estimate']:.3f} ({c['ci_low']:.3f}-{c['ci_high']:.3f})"
            )


if __name__ == "__main__":
    main()
