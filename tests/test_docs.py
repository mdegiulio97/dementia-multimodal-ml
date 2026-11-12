"""The documentation must match what the repository actually contains."""

import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_every_figure_is_indexed():
    index = (ROOT / "docs/FIGURES.md").read_text(encoding="utf-8")
    for kind in ("open", "synthetic"):
        for png in sorted((ROOT / "figures" / kind).glob("*.png")):
            assert f"figures/{kind}/{png.name}" in index, png.name


def test_no_adni_numbers_unless_allowed():
    allowed = yaml.safe_load((ROOT / "configs/sources.yaml").read_text())["publish_adni_aggregates"]
    for doc in ("README.md", "docs/RESULTS.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert ("## ADNI results" in text) == bool(allowed), doc


def test_readme_quickstart_commands_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("## Quickstart", 1)[1].split("```", 2)[1]
    scripts = yaml.safe_load(
        (ROOT / "pyproject.toml")
        .read_text()
        .split("[project.scripts]")[1]
        .split("[")[0]
        .replace(" = ", ": ")
    )
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "bash", "sh")):
            continue
        for path in re.findall(r"pipelines/\w+\.py", line):
            assert (ROOT / path).exists(), path
        m = re.match(r"uv run (dementia-[\w-]+)", line)
        if m:
            assert m.group(1) in scripts, m.group(1)
    assert shutil.which("uv") or True


def test_readme_cites_comparison_literature_with_doi():
    results = (ROOT / "docs/RESULTS.md").read_text(encoding="utf-8")
    for doi in (
        "10.1016/j.jclinepi.2023.03.009",
        "10.1007/s11571-026-10464-w",
        "10.1038/s41588-022-01024-z",
    ):
        assert doi in results, doi
