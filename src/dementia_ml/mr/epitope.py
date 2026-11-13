"""Epitope-effect check for cis-pQTL instruments.

A variant that changes the protein sequence can alter how the affinity reagent (here Olink
antibodies) binds the protein without changing its amount. Such a cis-pQTL may then reflect
measurement, not biology. We flag instruments that are, or are in strong LD (r2 >= 0.8, 1000
Genomes EUR) with, a protein-altering variant in the gene encoding the protein.

Annotations come from the public Ensembl REST API (VEP and LD endpoints).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

PROTEIN_ALTERING = {
    "missense_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "frameshift_variant",
    "inframe_insertion",
    "inframe_deletion",
    "protein_altering_variant",
    "splice_donor_variant",
    "splice_acceptor_variant",
}
ENSEMBL = "https://rest.ensembl.org"


def protein_altering_in_gene(vep: list[dict], gene: str) -> bool:
    for record in vep:
        for tc in record.get("transcript_consequences", []):
            if tc.get("gene_symbol") == gene and PROTEIN_ALTERING & set(
                tc.get("consequence_terms", [])
            ):
                return True
    return False


def verdict(instrument_altering: bool, proxies_altering: list[str]) -> str:
    if instrument_altering:
        return "instrument is protein-altering"
    if proxies_altering:
        return "protein-altering proxy in LD"
    return "no protein-altering variant in LD"


class EnsemblClient:
    """Small cached client for the Ensembl REST API (rate-limited, responses kept on disk)."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(timeout=180, headers={"Content-Type": "application/json"})

    def _get(self, path: str):
        key = self.cache_dir / (
            path.strip("/").replace("/", "_").replace("?", "_").replace(":", "_") + ".json"
        )
        if key.exists():
            return json.loads(key.read_text())
        for attempt in range(5):
            try:
                r = self.http.get(ENSEMBL + path)
            except httpx.TimeoutException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                continue
            r.raise_for_status()
            key.write_text(r.text)
            time.sleep(0.1)
            return r.json()
        raise RuntimeError(f"Ensembl did not answer {path} after 5 attempts")

    def vep(self, rsid: str) -> list[dict]:
        return self._get(f"/vep/human/id/{rsid}?content-type=application/json")

    def ld_proxies(self, rsid: str, r2: float = 0.8) -> list[str]:
        res = self._get(
            f"/ld/human/{rsid}/1000GENOMES:phase_3:EUR?r2={r2}&window_size=250"
            "&content-type=application/json"
        )
        return sorted({x["variation2"] for x in res if float(x.get("r2", 0)) >= r2})
