"""Download and normalise the open MR inputs listed in configs/sources.yaml.

Normalised exposure columns: trait, snp, effect_allele, other_allele, eaf,
beta, se, pval, cis, chrom, pos. Outcome columns: snp, effect_allele,
other_allele, eaf, beta, se, pval.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dementia_ml.paths import REPO_ROOT

SOURCES_FILE = REPO_ROOT / "configs" / "sources.yaml"
EXPOSURE_COLUMNS = [
    "trait",
    "snp",
    "effect_allele",
    "other_allele",
    "eaf",
    "beta",
    "se",
    "pval",
    "cis",
    "chrom",
    "pos",
]


def config(key: str) -> dict:
    return yaml.safe_load(SOURCES_FILE.read_text())[key]


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(key: str, dest: Path) -> Path:
    """Download the file for ``key`` into ``dest`` (once) and verify checksum/size."""
    cfg = config(key)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / cfg["url"].rsplit("/", 1)[-1].replace("%", "_")
    if not path.exists():
        req = urllib.request.Request(cfg["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, path.open("wb") as fh:  # noqa: S310
            while chunk := r.read(1 << 20):
                fh.write(chunk)
    if cfg.get("size_bytes") and path.stat().st_size != cfg["size_bytes"]:
        raise OSError(f"{path.name}: size {path.stat().st_size} != {cfg['size_bytes']}")
    for algo in ("sha256", "md5"):
        if cfg.get(algo) and _digest(path, algo) != cfg[algo]:
            raise OSError(f"{path.name}: {algo} mismatch")
    return path


def load_pqtl(path: Path) -> pd.DataFrame:
    """UKB-PPP (Sun et al. 2023) Supplementary Table 9; MHC variants excluded."""
    raw = pd.read_excel(path, sheet_name="ST9", header=4)
    vid = (
        raw["Variant ID (CHROM:GENPOS (hg37):A0:A1:imp:v1)"].astype(str).str.split(":", expand=True)
    )
    out = pd.DataFrame(
        {
            "trait": raw["Assay Target"].astype(str),
            "snp": raw["rsID"].astype(str),
            "effect_allele": vid[3].str.upper(),
            "other_allele": vid[2].str.upper(),
            "eaf": raw["A1FREQ (discovery)"].astype(float),
            "beta": raw["BETA (discovery, wrt. A1)"].astype(float),
            "se": raw["SE (discovery)"].astype(float),
            "pval": np.power(10.0, -raw["log10(p) (discovery)"].astype(float)),
            "cis": raw["cis/trans"].astype(str).str.lower().eq("cis"),
            "chrom": pd.to_numeric(raw["CHROM"], errors="coerce"),
            "pos": vid[1].astype(int),
        }
    )
    return out[raw["MHC"].astype(int).eq(0).to_numpy()].reset_index(drop=True)


def load_mqtl(path: Path) -> pd.DataFrame:
    """Karjalainen et al. 2024 Supplementary Table 5 (lead SNP-metabolite associations)."""
    raw = pd.read_excel(path, sheet_name="TableS5_all_lead_SNP_assoc", header=1)
    raw = raw.dropna(subset=["Rsid", "Effect"])
    return pd.DataFrame(
        {
            "trait": raw["Metabolite"].astype(str),
            "snp": raw["Rsid"].astype(str),
            "effect_allele": raw["Allele1"].astype(str).str.upper(),
            "other_allele": raw["Allele2"].astype(str).str.upper(),
            "eaf": raw["Freq1"].astype(float),
            "beta": raw["Effect"].astype(float),
            "se": raw["StdErr"].astype(float),
            "pval": pd.to_numeric(raw["P-value"], errors="coerce"),
            "cis": False,
            "chrom": pd.to_numeric(raw["Chromosome"], errors="coerce"),
            "pos": pd.to_numeric(raw["Lead variant position"], errors="coerce"),
        }
    ).reset_index(drop=True)


OUTCOME_COLUMNS = {
    "hm_rsid": "snp",
    "hm_effect_allele": "effect_allele",
    "hm_other_allele": "other_allele",
    "hm_effect_allele_frequency": "eaf",
    "hm_beta": "beta",
    "standard_error": "se",
    "p_value": "pval",
}


def load_outcome(path: Path, snps: set[str], chunksize: int = 2_000_000) -> pd.DataFrame:
    """Bellenguez et al. 2022 harmonised GWAS (GCST90027158), filtered to ``snps``."""
    parts = []
    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=lambda c: c.rstrip("\r") in OUTCOME_COLUMNS,  # CRLF file: last header ends in \r
        chunksize=chunksize,
        dtype=str,
        lineterminator="\n",
    )
    for chunk in reader:
        chunk.columns = [c.rstrip("\r") for c in chunk.columns]
        chunk = chunk.apply(lambda s: s.str.rstrip("\r"))
        hit = chunk[chunk["hm_rsid"].isin(snps)]
        if len(hit):
            parts.append(hit)
    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=list(OUTCOME_COLUMNS))
    )
    out = out.rename(columns=OUTCOME_COLUMNS)
    for c in ("eaf", "beta", "se", "pval"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates("snp").reset_index(drop=True)


def cached_outcome(cache: Path, snps: set[str], loader) -> pd.DataFrame:
    """Outcome rows for ``snps``, reading the big GWAS file only when needed.

    The cache remembers which SNPs were searched (``<cache>.searched.txt``), so SNPs
    absent from the GWAS do not force a new read on every run.
    """
    searched_file = cache.with_suffix(".searched.txt")
    if cache.exists() and searched_file.exists():
        searched = set(searched_file.read_text().split())
        if snps <= searched:
            return pd.read_parquet(cache)
        snps = snps | searched
    out = loader(snps)
    out.to_parquet(cache)
    searched_file.write_text("\n".join(sorted(snps)))
    return out
