"""Column contract for the ADNIMERGE table (subset used in this project).

Names follow the ADNIMERGE data dictionary; confirm them against the file you
download (see docs/ADNI_DOWNLOAD_LIST.md) before running on real data.
"""

from __future__ import annotations

import pandas as pd

ADNIMERGE_COLUMNS: dict[str, str] = {
    "RID": "int64",
    "VISCODE": "string",
    "EXAMDATE": "string",
    "M": "int64",
    "DX_bl": "string",
    "DX": "string",
    "AGE": "float64",
    "PTGENDER": "string",
    "PTEDUCAT": "float64",
    "APOE4": "float64",
    "MMSE": "float64",
    "ADAS13": "float64",
    "CDRSB": "float64",
    "FAQ": "float64",
    "Hippocampus": "float64",
    "ICV": "float64",
    # CSF biomarkers can hold strings at assay limits, e.g. ">1700" or "<8".
    "ABETA": "object",
    "TAU": "object",
    "PTAU": "object",
}

DX_BL_MCI = {"EMCI", "LMCI"}
DX_DEMENTIA = "Dementia"
BASELINE_VISCODE = "bl"


class SchemaError(ValueError):
    """Raised when a table does not match the expected ADNIMERGE columns."""


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ADNIMERGE_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing ADNIMERGE columns: {', '.join(missing)}")
    return df
