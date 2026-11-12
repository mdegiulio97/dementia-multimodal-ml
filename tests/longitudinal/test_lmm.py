import json

import numpy as np
import pandas as pd

from dementia_ml.longitudinal import lmm


def _simulated(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        conv = i % 3 == 0
        slope = (-2.0 if conv else -0.5) + rng.normal(0, 0.4)
        b0 = 27 + rng.normal(0, 1.2)
        for m in (0, 6, 12, 24, 36, 48):
            rows.append(
                {
                    "RID": i,
                    "M": m,
                    "MMSE": b0 + slope * m / 12 + rng.normal(0, 0.8),
                    "converter": int(conv),
                }
            )
    return pd.DataFrame(rows)


def test_recovers_simulated_slope_difference():
    res = lmm.fit_trajectory(_simulated(), outcome="MMSE")
    est, lo, hi = res.slope_difference
    assert abs(est - (-1.5)) < 0.3
    assert lo < -1.5 < hi
    assert res.n_subjects == 300 and res.n_obs == 1800
    assert {"coef", "se", "ci_low", "ci_high", "p"} <= set(res.fixed_effects.columns)


def test_trajectory_run_on_synthetic(tmp_path):
    from dementia_ml.longitudinal import run

    run.run(
        "synthetic",
        {"n_subjects": 600, "seed": 0, "outcomes": ["MMSE", "ADAS13"]},
        out_root=tmp_path,
    )
    res = json.loads((tmp_path / "results/synthetic/trajectories.json").read_text())
    assert res["outcomes"]["MMSE"]["slope_difference"]["estimate"] < 0
    assert res["outcomes"]["ADAS13"]["slope_difference"]["estimate"] > 0
    assert (tmp_path / "figures/synthetic/trajectories_MMSE.png").exists()
