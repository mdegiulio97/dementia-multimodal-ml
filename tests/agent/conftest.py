import json

import pandas as pd
import pytest

from dementia_ml.longitudinal import run as traj_run
from dementia_ml.survival import run as conv_run

SMALL = {
    "n_subjects": 600,
    "seed": 0,
    "n_splits": 3,
    "n_boot": 30,
    "horizons": [24, 36, 48],
    "calibration_horizon": 36,
    "uno_tau": 36,
    "models": {"rsf": {"n_estimators": 30}, "gbsa": {"n_estimators": 30}},
    "n_boot_models": 10,
    "outcomes": ["MMSE", "ADAS13"],
}


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Synthetic results/models plus small open-data result files."""
    root = tmp_path_factory.mktemp("ws")
    conv_run.run("synthetic", SMALL, out_root=root)
    traj_run.run("synthetic", SMALL, out_root=root)
    open_dir = root / "results/open"
    open_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trait": ["PROT_A", "MET_B"],
            "kind": ["pqtl", "mqtl"],
            "method": ["ivw", "wald_ratio"],
            "beta": [0.3, -0.1],
            "se": [0.05, 0.03],
            "ci_low": [0.2, -0.16],
            "ci_high": [0.4, -0.04],
            "p": [1e-9, 1e-3],
            "q_fdr": [1e-7, 0.01],
            "n_snps": [4, 1],
            "egger_intercept_p": [0.6, None],
        }
    ).to_csv(open_dir / "mr_candidates.csv", index=False)
    (open_dir / "mr_candidates.json").write_text(
        json.dumps({"selected_for_adni": ["PROT_A", "MET_B"], "provenance": {"dataset": "fixture"}})
    )
    (open_dir / "eeg_metrics.json").write_text(
        json.dumps(
            {
                "results": {
                    "AD_vs_CN": {
                        "logreg": {"auc": {"estimate": 0.81, "ci_low": 0.7, "ci_high": 0.9}}
                    }
                },
                "provenance": {"dataset": "fixture"},
            }
        )
    )
    return root
