"""Open-data MR screen of proteins and metabolites on Alzheimer's disease.

    uv run python pipelines/mr_candidates.py

Downloads (once) the sources in configs/sources.yaml into $DEMENTIA_ML_DATA/mr.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from dementia_ml.mr import run, sources
from dementia_ml.paths import data_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/mr.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dest = data_root() / "mr"

    exposures = {
        "pqtl": sources.load_pqtl(sources.download("pqtl", dest)),
        "mqtl": sources.load_mqtl(sources.download("mqtl", dest)),
    }
    snps = set(pd.concat(exposures.values())["snp"])
    gwas_path = sources.download("gwas_ad", dest)
    outcome = sources.cached_outcome(
        dest / "outcome_at_instruments.parquet",
        snps,
        lambda s: sources.load_outcome(gwas_path, s),
    )

    names = {k: sources.config(k)["paper_doi"] for k in ("pqtl", "mqtl")}
    names["outcome"] = sources.config("gwas_ad")["accession"]
    versions = {k: sources.config(k)["version"] for k in ("pqtl", "mqtl", "gwas_ad")}
    res = run.run(exposures, outcome, cfg, dataset_names=names, dataset_versions=versions)
    print(
        json.dumps({"selected": res["selected"][:30], "n_selected": len(res["selected"])}, indent=1)
    )


if __name__ == "__main__":
    main()
