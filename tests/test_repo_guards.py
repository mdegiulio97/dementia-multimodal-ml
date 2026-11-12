"""Guards that keep individual-level ADNI data out of version control."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_no_adni.py"


def _run(*files: Path | str) -> int:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, files)], cwd=ROOT, capture_output=True
    ).returncode


def test_check_no_adni_rejects_rid_csv(tmp_path):
    f = tmp_path / "table.csv"
    pd.DataFrame({"RID": [1, 2], "MMSE": [28, 27]}).to_csv(f, index=False)
    assert _run(f) == 1


def test_check_no_adni_rejects_ptid_parquet(tmp_path):
    f = tmp_path / "table.parquet"
    pd.DataFrame({"PTID": ["002_S_0001"], "MMSE": [28]}).to_parquet(f)
    assert _run(f) == 1


def test_check_no_adni_accepts_synthetic_csv(tmp_path):
    f = tmp_path / "table.csv"
    pd.DataFrame({"subject_id": [1, 2], "MMSE": [28, 27]}).to_csv(f, index=False)
    assert _run(f) == 0


def test_check_no_adni_rejects_results_adni_path():
    assert _run("results/adni/conversion_metrics.json") == 1


def test_check_no_adni_rejects_adnimerge_name(tmp_path):
    f = tmp_path / "ADNIMERGE_subset.txt"
    f.write_text("hello")
    assert _run(f) == 1


def test_gitignore_covers_private_paths():
    text = (ROOT / ".gitignore").read_text()
    for entry in [
        "results/adni/",
        "figures/adni/",
        "models/adni/",
        "data/",
        "docs/superpowers/",
        ".env",
    ]:
        assert entry in text, entry


def test_paths_data_root_env_override(tmp_path, monkeypatch):
    from dementia_ml import paths

    monkeypatch.setenv("DEMENTIA_ML_DATA", str(tmp_path))
    assert paths.data_root() == tmp_path


def test_results_dir_rejects_unknown_kind():
    import pytest

    from dementia_ml import paths

    with pytest.raises(ValueError):
        paths.results_dir("private")


def test_check_no_adni_accepts_source_code_named_adnimerge():
    assert _run("src/dementia_ml/data/adnimerge.py") == 0
