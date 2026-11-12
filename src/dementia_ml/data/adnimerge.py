"""Load ADNIMERGE and normalise the values that arrive as assay-limit strings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dementia_ml.data import schema

BOUNDED_COLUMNS = ("ABETA", "TAU", "PTAU")


def _parse_value(value: object) -> tuple[float, int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, 0
    if isinstance(value, int | float | np.number):
        return float(value), 0
    text = str(value).strip()
    if not text:
        return np.nan, 0
    if text.startswith(">"):
        return float(text[1:]), 1
    if text.startswith("<"):
        return float(text[1:]), -1
    return float(text), 0


def parse_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '>x' / '<x' strings to x and add an int8 ``<col>_bound`` flag.

    Idempotent: a column that already has its flag is left untouched.
    """
    out = df.copy()
    for col in BOUNDED_COLUMNS:
        if col not in out.columns or f"{col}_bound" in out.columns:
            continue
        parsed = [_parse_value(v) for v in out[col]]
        out[col] = np.array([p[0] for p in parsed], dtype=float)
        out[f"{col}_bound"] = np.array([p[1] for p in parsed], dtype=np.int8)
    return out


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    schema.validate(df)
    return parse_bounds(df)
