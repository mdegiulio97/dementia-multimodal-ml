import numpy as np
import pandas as pd
import pytest

from dementia_ml.data import adnimerge, conversion, schema, synthetic


def _subject(rid, dx_bl, months, dx, mmse=None, **extra):
    n = len(months)
    rows = {
        "RID": [rid] * n,
        "VISCODE": ["bl" if m == 0 else f"m{m:02d}" for m in months],
        "EXAMDATE": ["2010-01-01"] * n,
        "M": months,
        "DX_bl": [dx_bl] * n,
        "DX": dx,
        "AGE": [70.0] * n,
        "PTGENDER": ["Female"] * n,
        "PTEDUCAT": [16.0] * n,
        "APOE4": [1.0] * n,
        "MMSE": mmse if mmse is not None else [28.0] * n,
        "ADAS13": [12.0] * n,
        "CDRSB": [1.0] * n,
        "FAQ": [2.0] * n,
        "Hippocampus": [7000.0] * n,
        "ICV": [1.5e6] * n,
        "ABETA": [900.0] + [None] * (n - 1),
        "TAU": [250.0] + [None] * (n - 1),
        "PTAU": [22.0] + [None] * (n - 1),
    }
    rows.update(extra)
    return pd.DataFrame(rows)


def test_only_mci_at_baseline():
    df = pd.concat(
        [
            _subject(1, "LMCI", [0, 6], ["MCI", "MCI"]),
            _subject(2, "CN", [0, 6], ["CN", "CN"]),
            _subject(3, "AD", [0, 6], ["Dementia", "Dementia"]),
        ]
    )
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert list(data.X.index) == [1]
    assert data.meta["excluded"]["not_mci_at_baseline"] == 2


def test_event_is_first_dementia_visit():
    df = _subject(1, "EMCI", [0, 6, 12, 24], ["MCI", "MCI", "Dementia", "MCI"])
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert data.time.tolist() == [12] and data.event.tolist() == [True]


def test_missing_dx_visit_is_ignored():
    df = _subject(1, "EMCI", [0, 6, 12], ["MCI", None, "MCI"])
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert data.time.tolist() == [12] and data.event.tolist() == [False]


def test_trailing_missing_dx_censors_at_last_known_diagnosis():
    df = _subject(1, "EMCI", [0, 6, 12], ["MCI", "MCI", None])
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert data.time.tolist() == [6] and data.event.tolist() == [False]


def test_no_follow_up_excluded():
    df = pd.concat([_subject(1, "EMCI", [0], ["MCI"]), _subject(2, "EMCI", [0, 6], ["MCI", "MCI"])])
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert list(data.X.index) == [2]
    assert data.meta["excluded"]["no_follow_up"] == 1


def test_features_are_baseline_only():
    df = _subject(1, "EMCI", [0, 6, 12], ["MCI", "MCI", "MCI"], mmse=[28.0, 27.0, 26.0])
    before = conversion.build(df, features=conversion.FULL_FEATURES).X
    df.loc[df["M"] == 12, "MMSE"] = 5.0
    after = conversion.build(df, features=conversion.FULL_FEATURES).X
    pd.testing.assert_frame_equal(before, after)
    assert after.loc[1, "MMSE"] == 28.0


def test_bound_strings_parsed(tmp_path):
    df = _subject(1, "EMCI", [0, 6], ["MCI", "MCI"])
    df["ABETA"] = [">1700", None]
    df["PTAU"] = ["<8", None]
    path = tmp_path / "adnimerge_like.csv"
    df.to_csv(path, index=False)
    loaded = adnimerge.load(path)
    bl = loaded[loaded["M"] == 0].iloc[0]
    assert bl["ABETA"] == 1700.0 and bl["ABETA_bound"] == 1
    assert bl["PTAU"] == 8.0 and bl["PTAU_bound"] == -1
    assert bl["TAU"] == 250.0 and bl["TAU_bound"] == 0


def test_build_parses_bounds_on_synthetic():
    df = synthetic.make_cohort(n_subjects=300, seed=3)
    data = conversion.build(df, features=conversion.FULL_FEATURES)
    assert data.X["ABETA"].dtype == np.float64
    assert data.X["ABETA"].max() <= 1700.0
    assert set(data.X.columns) >= set(conversion.FULL_FEATURES)
    assert data.meta["n_included"] == len(data.X) == len(data.time) == len(data.event)


def test_load_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"RID": [1]}).to_csv(path, index=False)
    with pytest.raises(schema.SchemaError):
        adnimerge.load(path)


def test_adni3_plain_mci_baseline_included():
    df = _subject(7, "MCI", [0, 6], ["MCI", "Dementia"])
    data = conversion.build(df, features=conversion.BASELINE_FEATURES)
    assert list(data.X.index) == [7] and data.event.tolist() == [True]
