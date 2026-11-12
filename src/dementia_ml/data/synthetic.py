"""Synthetic cohort shaped like ADNIMERGE, with a known conversion hazard.

Used by tests, CI and the public demo. It mimics the structure and the messy
parts of the real table (missing CSF, assay-limit strings, diagnosis
reversions, missing visit diagnoses) but contains no real participant data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

VISITS = (0, 6, 12, 18, 24, 36, 48)
TRUE_LOG_HR: dict[str, float] = {
    "APOE4": 0.5,
    "MMSE": -0.15,
    "ADAS13": 0.06,
    "Hippocampus_z": -0.6,
}
WEIBULL_SHAPE = 1.3
TARGET_CONVERSION_48M = 0.40  # true probability; dropout lowers the observed share
DX_BL_PROBS = {"CN": 0.25, "EMCI": 0.25, "LMCI": 0.30, "AD": 0.20}


def _viscode(m: int) -> str:
    return "bl" if m == 0 else f"m{m:02d}"


def _calibrate_scale(lp: np.ndarray) -> float:
    def gap(log_scale: float) -> float:
        cum = np.exp(log_scale) * 48.0**WEIBULL_SHAPE * np.exp(lp)
        return float(np.mean(1 - np.exp(-cum)) - TARGET_CONVERSION_48M)

    return float(np.exp(brentq(gap, -30, 5)))


def make_cohort(n_subjects: int = 800, seed: int = 0) -> pd.DataFrame:
    """Return a long-format table (one row per visit) with ADNIMERGE columns."""
    rng = np.random.default_rng(seed)
    n = n_subjects
    rid = np.arange(1, n + 1) + 10_000
    dx_bl = rng.choice(list(DX_BL_PROBS), size=n, p=list(DX_BL_PROBS.values()))
    is_mci = np.isin(dx_bl, ["EMCI", "LMCI"])
    severity = np.select([dx_bl == "CN", dx_bl == "EMCI", dx_bl == "LMCI"], [0.0, 1.0, 2.0], 3.0)

    age = np.round(rng.normal(73, 7, n), 1)
    sex = rng.choice(["Male", "Female"], size=n)
    educ = np.clip(np.round(rng.normal(16, 2.8, n)), 6, 20)
    apoe4 = rng.choice([0, 1, 2], size=n, p=[0.5, 0.38, 0.12]).astype(float)
    icv = rng.normal(1.5e6, 1.5e5, n)
    hip_z = rng.normal(-0.5 * severity / 3 - 0.15 * apoe4, 1.0)
    hip_z = (hip_z - hip_z.mean()) / hip_z.std()
    hippocampus = 3000 + 0.0027 * icv + 850 * hip_z
    mmse = np.clip(np.round(rng.normal(29 - 1.2 * severity, 1.3)), 15, 30)
    adas13 = np.clip(np.round(rng.normal(9 + 5 * severity, 4.5), 1), 0, 85)
    cdrsb = np.clip(np.round(rng.normal(0.2 + 1.2 * severity, 0.8) * 2) / 2, 0, 18)
    faq = np.clip(np.round(rng.normal(0.5 + 3 * severity, 2.5)), 0, 30)

    abeta = np.clip(rng.normal(1100 - 120 * severity - 150 * apoe4, 350), 200, 1700).round(1)
    tau = np.clip(rng.normal(230 + 35 * severity, 90), 80, 1300).round(1)
    ptau = np.clip(rng.normal(21 + 4 * severity, 9), 8, 120).round(2)
    csf_missing = rng.random(n) < 0.5

    # Conversion time for MCI subjects (Weibull proportional hazards).
    lp = (
        TRUE_LOG_HR["APOE4"] * apoe4
        + TRUE_LOG_HR["MMSE"] * mmse
        + TRUE_LOG_HR["ADAS13"] * adas13
        + TRUE_LOG_HR["Hippocampus_z"] * hip_z
    )
    lp = lp - lp[is_mci].mean()
    scale = _calibrate_scale(lp[is_mci])
    e = rng.exponential(1.0, n)
    t_conv = (e / (scale * np.exp(lp))) ** (1 / WEIBULL_SHAPE)
    t_conv[~is_mci] = np.inf

    # Follow-up: 3% baseline only, 55% complete, rest drop out at a random visit.
    u = rng.random(n)
    last_idx = np.where(
        u < 0.03, 0, np.where(u < 0.58, len(VISITS) - 1, rng.integers(1, len(VISITS) - 1, n))
    )
    reverter = rng.random(n) < 0.03
    slope = np.where(np.isfinite(t_conv) & (t_conv <= 60), -2.0, -0.5)  # MMSE points / year

    rows = []
    for i in range(n):
        first_dementia_seen = False
        for k, m in enumerate(VISITS[: last_idx[i] + 1]):
            years = m / 12
            if dx_bl[i] == "CN":
                dx = "CN"
            elif dx_bl[i] == "AD":
                dx = "Dementia"
            elif m >= t_conv[i]:
                dx = "Dementia"
                if first_dementia_seen and reverter[i]:
                    dx = "MCI"
                first_dementia_seen = True
            else:
                dx = "MCI"
            if k > 0 and rng.random() < 0.03:
                dx = None
            csf = (None, None, None) if csf_missing[i] or k > 0 else (abeta[i], tau[i], ptau[i])
            rows.append(
                {
                    "RID": int(rid[i]),
                    "VISCODE": _viscode(m),
                    "EXAMDATE": str(
                        np.datetime64("2010-01-01") + np.timedelta64(int(m * 30.4), "D")
                    ),
                    "M": m,
                    "DX_bl": dx_bl[i],
                    "DX": dx,
                    "AGE": age[i],
                    "PTGENDER": sex[i],
                    "PTEDUCAT": educ[i],
                    "APOE4": apoe4[i],
                    "MMSE": float(
                        np.clip(np.round(mmse[i] + slope[i] * years + rng.normal(0, 0.8)), 0, 30)
                    )
                    if k > 0
                    else mmse[i],
                    "ADAS13": float(
                        np.clip(adas13[i] - 1.5 * slope[i] * years + rng.normal(0, 2), 0, 85)
                    )
                    if k > 0
                    else adas13[i],
                    "CDRSB": cdrsb[i],
                    "FAQ": faq[i],
                    "Hippocampus": float(hippocampus[i] * (1 - 0.02 * years)),
                    "ICV": float(icv[i]),
                    "ABETA": csf[0],
                    "TAU": csf[1],
                    "PTAU": csf[2],
                }
            )
    df = pd.DataFrame(rows)
    # Assay-limit strings, as in the real CSF columns.
    bl_csf = df.index[(df["M"] == 0) & df["ABETA"].notna()]
    hit = rng.random(len(bl_csf)) < 0.10
    df["ABETA"] = df["ABETA"].astype(object)
    df.loc[bl_csf[hit], "ABETA"] = ">1700"
    hit_p = rng.random(len(bl_csf)) < 0.03
    df["PTAU"] = df["PTAU"].astype(object)
    df.loc[bl_csf[hit_p], "PTAU"] = "<8"
    return df
