"""Run the MCI-to-dementia conversion analysis.

uv run python pipelines/conversion.py --data synthetic
uv run python pipelines/conversion.py --data adni      # needs local ADNI files
"""

import argparse
from pathlib import Path

import yaml

from dementia_ml.survival import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", choices=run.KINDS, default="synthetic")
    ap.add_argument("--config", type=Path, default=Path("configs/conversion.yaml"))
    args = ap.parse_args()
    summary = run.run(args.data, yaml.safe_load(args.config.read_text()))
    c = summary["c_index"]
    print(
        f"[{args.data}] n={summary['n']} best={summary['best_model']} "
        f"C={c['estimate']:.3f} (95% CI {c['ci_low']:.3f}-{c['ci_high']:.3f})"
    )


if __name__ == "__main__":
    main()
