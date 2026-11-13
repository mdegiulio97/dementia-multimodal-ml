"""Open-data MR screen of proteins and metabolites on Alzheimer's disease.

    uv run python pipelines/mr_candidates.py                      # primary: Bellenguez 2022
    uv run python pipelines/mr_candidates.py --outcome kunkle     # sensitivity: no UK Biobank

Downloads (once) the sources in configs/sources.yaml into $DEMENTIA_ML_DATA/mr.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from dementia_ml import paths
from dementia_ml.mr import run, sources
from dementia_ml.paths import data_root

SEPARATORS = {" ": " ", "\\t": "\t", "tab": "\t"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/mr.yaml"))
    ap.add_argument("--outcome", choices=["bellenguez", "kunkle"], default="bellenguez")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dest = data_root() / "mr"

    exposures = {
        "pqtl": sources.load_pqtl(sources.download("pqtl", dest)),
        "mqtl": sources.load_mqtl(sources.download("mqtl", dest)),
    }
    snps = set(pd.concat(exposures.values())["snp"])
    gwas_cfg = sources.config("gwas_ad")
    if args.outcome == "bellenguez":
        gwas_path, columns, sep, suffix = sources.download("gwas_ad", dest), None, "\t", ""
        accession, version = gwas_cfg["accession"], gwas_cfg["version"]
    else:
        sens = gwas_cfg["sensitivity_outcome"]
        gwas_path = dest / sens["url"].rsplit("/", 1)[-1]
        if not gwas_path.exists() or gwas_path.stat().st_size != sens["size_bytes"]:
            raise SystemExit(f"download {sens['url']} to {gwas_path} first")
        columns, sep, suffix = sens["columns"], SEPARATORS.get(sens["separator"], " "), "_kunkle"
        accession, version = sens["accession"], f"size {sens['size_bytes']}"
    outcome = sources.cached_outcome(
        dest / f"outcome_at_instruments{suffix}.parquet",
        snps,
        lambda s: sources.load_outcome(gwas_path, s, columns=columns, sep=sep),
    )

    names = {k: sources.config(k)["paper_doi"] for k in ("pqtl", "mqtl")}
    names["outcome"] = accession
    versions = {k: sources.config(k)["version"] for k in ("pqtl", "mqtl")} | {"outcome": version}
    res = run.run(
        exposures, outcome, cfg, dataset_names=names, dataset_versions=versions, suffix=suffix
    )
    summary = {"outcome": args.outcome, "n_selected": len(res["selected"])}
    if args.outcome == "kunkle":
        res_dir = paths.results_dir("open")
        primary = pd.read_csv(res_dir / "mr_candidates.csv")
        selected = (res_dir / "mr_candidates_selected.txt").read_text().split()
        cmp = run.compare_outcomes(
            primary,
            res["table"],
            selected,
            paths.figures_dir("open") / "mr_bellenguez_vs_kunkle.png",
        )
        (res_dir / "mr_outcome_comparison.json").write_text(json.dumps(cmp, indent=2))
        summary["comparison"] = cmp
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
