"""Mixed-effects cognitive trajectories: converters vs stable MCI.

uv run python pipelines/trajectories.py --data synthetic
"""

import argparse
from pathlib import Path

import yaml

from dementia_ml.longitudinal import run
from dementia_ml.survival.run import KINDS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", choices=KINDS, default="synthetic")
    ap.add_argument("--config", type=Path, default=Path("configs/trajectories.yaml"))
    args = ap.parse_args()
    out = run.run(args.data, yaml.safe_load(args.config.read_text()))
    for name, r in out["outcomes"].items():
        d = r["slope_difference"]
        print(
            f"[{args.data}] {name}: converters - stable = {d['estimate']:.2f}/yr "
            f"(95% CI {d['ci_low']:.2f} to {d['ci_high']:.2f}), n={r['n_subjects']}"
        )


if __name__ == "__main__":
    main()
