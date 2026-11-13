"""Build an ADNIMERGE-shaped table from the ADNIMERGE2 R package (ADNI 1/GO/2/3/4).

ADNIMERGE2 replaced the single merged ADNIMERGE.csv with CDISC-style tables:
- ``ADSL``: one row per participant, baseline characteristics;
- ``ADRS``: clinical diagnosis per visit (CN, MCI, DEM);
- ``ADQS``: questionnaire scores per visit.

Volumes come from ``ADSP_PHC_T1_FS`` (FreeSurfer, ComBat-harmonised across scanners) and CSF
from the UPENN Roche Elecsys table. Both are matched to the enrolment date (closest record
within ``window_days``) and placed on the baseline row only, because the conversion models use
baseline values only.

Nothing here prints or returns participant rows outside the local pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dementia_ml.data import schema

DAYS_PER_MONTH = 30.4375
DX_VISIT = {"CN": "CN", "MCI": "MCI", "DEM": "Dementia"}
DX_BASELINE = {"CN": "CN", "MCI": "MCI", "DEM": "AD"}
SCORES = {"MMSCORE": "MMSE", "ADASTT13": "ADAS13", "CDRSB": "CDRSB", "FAQTOTAL": "FAQ"}
CSF = {"ABETA42": "ABETA", "TAU": "TAU", "PTAU": "PTAU"}
# Regional FreeSurfer measures (ComBat): volumes summed over hemispheres, thickness averaged.
REGION_VOLUMES = {
    "Amygdala": ("Left_Amygdala_combat", "Right_Amygdala_combat"),
    "InfLatVent": ("Left_Inf_Lat_Vent_combat", "Right_Inf_Lat_Vent_combat"),
    "LatVent": ("Left_Lateral_Ventricle_combat", "Right_Lateral_Ventricle_combat"),
}
REGION_THICKNESS = {
    f"{name}_thk": (f"lh_{region}_thickness_combat", f"rh_{region}_thickness_combat")
    for name, region in [
        ("Entorhinal", "entorhinal"),
        ("Parahippocampal", "parahippocampal"),
        ("Fusiform", "fusiform"),
        ("InferiorTemporal", "inferiortemporal"),
        ("MiddleTemporal", "middletemporal"),
        ("Precuneus", "precuneus"),
        ("PosteriorCingulate", "posteriorcingulate"),
    ]
}
MRI_REGIONAL = list(REGION_VOLUMES) + list(REGION_THICKNESS)


def load_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the tables used here from an unpacked ADNIMERGE2 package plus the Elecsys CSV."""
    import pyreadr

    data = Path(raw_dir) / "ADNIMERGE2" / "data"

    def rda(name: str) -> pd.DataFrame:
        return next(iter(pyreadr.read_r(data / f"{name}.rda").values()))

    elecsys = sorted(Path(raw_dir).glob("UPENNBIOMK_ROCHE_ELECSYS_*.csv"))
    if not elecsys:
        raise FileNotFoundError(f"UPENNBIOMK_ROCHE_ELECSYS_*.csv not found in {raw_dir}")
    return {
        "ADSL": rda("ADSL"),
        "ADRS": rda("ADRS"),
        "ADQS": rda("ADQS"),
        "PHC_T1_FS": rda("ADSP_PHC_T1_FS"),
        "ELECSYS": pd.read_csv(elecsys[-1], dtype={"ABETA42": str, "TAU": str, "PTAU": str}),
    }


def _apoe4(genotype: pd.Series) -> pd.Series:
    g = genotype.astype("string")
    return g.str.count("ε4").astype("Float64").astype(float)


def _nearest(
    records: pd.DataFrame, date_col: str, anchors: pd.Series, window_days: int
) -> pd.DataFrame:
    """For each RID in ``anchors`` (RID -> date), the record closest in time within the window."""
    rec = records.copy()
    rec["RID"] = rec["RID"].astype(int)
    rec["_date"] = pd.to_datetime(rec[date_col], errors="coerce")
    rec = rec.dropna(subset=["_date"])
    rec = rec[rec["RID"].isin(anchors.index)]
    rec["_gap"] = (rec["_date"] - rec["RID"].map(anchors)).abs().dt.days
    rec = rec[rec["_gap"] <= window_days].sort_values(["RID", "_gap"])
    return rec.drop_duplicates("RID").set_index("RID")


def build(tables: dict[str, pd.DataFrame], window_days: int = 180) -> pd.DataFrame:
    adsl = tables["ADSL"]
    adsl = adsl[adsl["ENRLFL"] == "Y"].copy()
    adsl["RID"] = adsl["SUBJID"].astype(int)
    adsl["ENRLDT"] = pd.to_datetime(adsl["ENRLDT"], errors="coerce")
    adsl = adsl.set_index("RID")

    dx = tables["ADRS"]
    dx = dx[(dx["PARAMCD"] == "DX") & (dx["ANL01FL"] == "Y")].copy()
    dx["RID"] = dx["SUBJID"].astype(int)
    dx = dx[dx["RID"].isin(adsl.index)]
    is_bl = dx["ABLFL"] == "Y"
    dx = dx[is_bl | (dx["ADY"] > 0)]
    dx["M"] = np.where(dx["ABLFL"] == "Y", 0, np.round(dx["ADY"] / DAYS_PER_MONTH)).astype(int)
    dx = dx.sort_values(["RID", "M", "ADT"]).drop_duplicates(["RID", "M"])
    dx["VISCODE"] = np.where(dx["M"] == 0, "bl", "m" + dx["M"].astype(str).str.zfill(2))

    scores = tables["ADQS"]
    scores = scores[scores["PARAMCD"].isin(SCORES)].copy()
    scores["RID"] = scores["SUBJID"].astype(int)
    wide = (
        scores.pivot_table(
            index=["RID", "AVISIT"], columns="PARAMCD", values="AVAL", aggfunc="first"
        )
        .rename(columns=SCORES)
        .reset_index()
    )
    out = dx.merge(wide, on=["RID", "AVISIT"], how="left")
    for col in SCORES.values():
        if col not in out:
            out[col] = np.nan

    bl_rows = out["VISCODE"] == "bl"
    for src, col in SCORES.items():
        fallback = out.loc[bl_rows, "RID"].map(adsl[src])
        out.loc[bl_rows, col] = out.loc[bl_rows, col].fillna(fallback)

    out["DX"] = out["AVALC"].map(DX_VISIT)
    out["DX_bl"] = out["RID"].map(adsl["DX"].map(DX_BASELINE))
    out["AGE"] = out["RID"].map(adsl["AGE"]).astype(float)
    out["PTGENDER"] = out["RID"].map(adsl["SEX"])
    out["PTEDUCAT"] = out["RID"].map(adsl["EDUC"]).astype(float)
    out["APOE4"] = out["RID"].map(_apoe4(adsl["APOE"]))
    out["EXAMDATE"] = out["ADT"].astype(str)
    out["ORIGPROT"] = out["RID"].map(adsl.get("ORIGPROT", pd.Series(dtype=str)))

    anchors = adsl["ENRLDT"].dropna()
    mri = _nearest(tables["PHC_T1_FS"], "PHC_SCANDATE", anchors, window_days)
    hip = mri["Left_Hippocampus_combat"] + mri["Right_Hippocampus_combat"]
    icv = mri["EstimatedTotalIntraCranialVol_combat"]
    csf = _nearest(tables["ELECSYS"], "EXAMDATE", anchors, window_days)
    out["Hippocampus"] = np.where(bl_rows, out["RID"].map(hip), np.nan)
    out["ICV"] = np.where(bl_rows, out["RID"].map(icv), np.nan)
    for src, col in CSF.items():
        out[col] = pd.Series(np.where(bl_rows, out["RID"].map(csf[src]), None), dtype=object)

    for name, (left, right) in {**REGION_VOLUMES, **REGION_THICKNESS}.items():
        if left in mri and right in mri:
            pair = mri[[left, right]]
            value = pair.sum(axis=1, min_count=2) if name in REGION_VOLUMES else pair.mean(axis=1)
            out[name] = np.where(bl_rows, out["RID"].map(value), np.nan)
        else:
            out[name] = np.nan

    cols = list(schema.ADNIMERGE_COLUMNS) + ["ORIGPROT"] + MRI_REGIONAL
    return out[cols].sort_values(["RID", "M"]).reset_index(drop=True)
