"""Read OpenNeuro ds004504 (resting-state EEG: AD, FTD, healthy controls)."""

from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from dementia_ml.mr.sources import config


def participants(bids_root: Path) -> pd.DataFrame:
    codes = config("eeg_ds004504")["group_codes"]
    raw = pd.read_csv(Path(bids_root) / "participants.tsv", sep="\t")
    return pd.DataFrame(
        {
            "subject": raw["participant_id"].str.removeprefix("sub-"),
            "group": raw["Group"].map(codes),
            "age": raw["Age"],
            "sex": raw["Gender"],
            "mmse": raw["MMSE"],
        }
    )


def subject_path(bids_root: Path, subject: str, use_derivatives: bool = True) -> Path:
    cfg = config("eeg_ds004504")
    pattern = cfg["derivatives_pattern"] if use_derivatives else cfg["raw_pattern"]
    return Path(bids_root) / pattern.format(sub=subject)


def load_subject(bids_root: Path, subject: str, use_derivatives: bool = True) -> mne.io.BaseRaw:
    return mne.io.read_raw_eeglab(
        subject_path(bids_root, subject, use_derivatives), preload=True, verbose=False
    )
