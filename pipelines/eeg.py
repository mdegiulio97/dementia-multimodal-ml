"""Resting-state EEG (OpenNeuro ds004504): AD / FTD / controls.

aws s3 sync --no-sign-request s3://openneuro.org/ds004504 $DEMENTIA_ML_DATA/eeg/ds004504
uv run python pipelines/eeg.py
"""

import argparse
from pathlib import Path

import yaml

from dementia_ml.eeg import run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/eeg.yaml"))
    args = ap.parse_args()
    out = run.run(yaml.safe_load(args.config.read_text()))
    for task, models in out["results"].items():
        for model, r in models.items():
            a, b = r["auc"], r["balanced_accuracy"]
            print(
                f"{task:10s} {model:6s} n={r['n_subjects']:3d} "
                f"AUC {a['estimate']:.3f} ({a['ci_low']:.3f}-{a['ci_high']:.3f}) "
                f"BA {b['estimate']:.3f} ({b['ci_low']:.3f}-{b['ci_high']:.3f})"
            )


if __name__ == "__main__":
    main()
