"""ADNIMERGE2 adapter, tested on tiny made-up tables with the ADNIMERGE2 layout."""

import numpy as np
import pandas as pd

from dementia_ml.data import adnimerge2, conversion, schema


def _tables():
    adsl = pd.DataFrame(
        {
            "SUBJID": ["101", "102", "103"],
            "ENRLFL": ["Y", "Y", None],
            "ENRLDT": ["2010-01-01", "2012-06-01", "2015-01-01"],
            "DX": ["MCI", "CN", None],
            "AGE": [71.0, 68.0, 80.0],
            "SEX": ["Female", "Male", "Male"],
            "EDUC": [16.0, 12.0, 14.0],
            "APOE": ["ε3/ε4", "ε4/ε4", None],
            "MMSCORE": [27.0, 29.0, np.nan],
            "ADASTT13": [15.0, 8.0, np.nan],
            "CDRSB": [1.5, 0.0, np.nan],
            "FAQTOTAL": [3.0, 0.0, np.nan],
            "ORIGPROT": ["ADNI2", "ADNI2", "ADNI3"],
        }
    )
    adrs = pd.DataFrame(
        {
            "SUBJID": ["101"] * 4 + ["102"] * 2,
            "PARAMCD": ["DX"] * 6,
            "ANL01FL": ["Y"] * 6,
            "ABLFL": ["Y", None, None, None, "Y", None],
            "AVISIT": ["BASELINE", "Y1", "Y2", "SCR", "BASELINE", "Y1"],
            "ADY": [1, 366, 731, -30, 1, 360],
            "ADT": [
                "2010-01-01",
                "2011-01-01",
                "2012-01-01",
                "2009-12-01",
                "2012-06-01",
                "2013-06-01",
            ],
            "AVALC": ["MCI", "MCI", "DEM", "MCI", "CN", "CN"],
        }
    )
    adqs = pd.DataFrame(
        {
            "SUBJID": ["101", "101", "101", "101"],
            "AVISIT": ["BASELINE", "Y1", "Y2", "Y2"],
            "PARAMCD": ["MMSCORE", "MMSCORE", "MMSCORE", "ADASTT13"],
            "AVAL": [27.0, 26.0, 22.0, 25.0],
        }
    )
    phc = pd.DataFrame(
        {
            "RID": [101.0, 101.0, 102.0],
            "PHC_SCANDATE": ["2010-03-01", "2011-01-10", "2014-01-01"],
            "Left_Hippocampus_combat": [3000.0, 2900.0, 3500.0],
            "Right_Hippocampus_combat": [3100.0, 3000.0, 3600.0],
            "EstimatedTotalIntraCranialVol_combat": [1.4e6, 1.4e6, 1.6e6],
        }
    )
    elecsys = pd.DataFrame(
        {
            "RID": [101, 102],
            "EXAMDATE": ["2010-01-20", "2012-07-01"],
            "ABETA42": [">1700", "650"],
            "TAU": ["300", "200"],
            "PTAU": ["28", "<8"],
        }
    )
    return {"ADSL": adsl, "ADRS": adrs, "ADQS": adqs, "PHC_T1_FS": phc, "ELECSYS": elecsys}


def test_visits_months_and_diagnosis_codes():
    df = adnimerge2.build(_tables())
    s101 = df[df["RID"] == 101].sort_values("M")
    assert s101["VISCODE"].tolist() == ["bl", "m12", "m24"]  # pre-enrollment screening dropped
    assert s101["DX"].tolist() == ["MCI", "MCI", "Dementia"]
    assert set(df["DX_bl"]) == {"MCI", "CN"}
    assert 103 not in set(df["RID"])  # not enrolled


def test_scores_come_from_adqs_per_visit():
    df = adnimerge2.build(_tables()).set_index(["RID", "VISCODE"])
    assert df.loc[(101, "m24"), "MMSE"] == 22.0 and df.loc[(101, "m24"), "ADAS13"] == 25.0
    assert df.loc[(101, "bl"), "CDRSB"] == 1.5  # baseline falls back to ADSL


def test_apoe4_count_and_demographics():
    bl = adnimerge2.build(_tables()).query("VISCODE == 'bl'").set_index("RID")
    assert bl.loc[101, "APOE4"] == 1 and bl.loc[102, "APOE4"] == 2
    assert bl.loc[101, "PTGENDER"] == "Female" and bl.loc[101, "PTEDUCAT"] == 16.0


def test_imaging_and_csf_nearest_within_window_baseline_only():
    df = adnimerge2.build(_tables(), window_days=180)
    bl = df.query("VISCODE == 'bl'").set_index("RID")
    assert bl.loc[101, "Hippocampus"] == 6100.0 and bl.loc[101, "ICV"] == 1.4e6
    assert np.isnan(bl.loc[102, "Hippocampus"])  # scan 19 months after enrolment
    assert bl.loc[101, "ABETA"] == ">1700" and bl.loc[102, "PTAU"] == "<8"
    later = df.query("VISCODE != 'bl'")
    assert later[["Hippocampus", "ABETA"]].isna().all().all()


def test_output_feeds_the_conversion_dataset():
    df = adnimerge2.build(_tables())
    schema.validate(df)
    data = conversion.build(df, features=conversion.FULL_FEATURES)
    assert list(data.X.index) == [101]
    assert data.time.tolist() == [24.0] and data.event.tolist() == [True]
