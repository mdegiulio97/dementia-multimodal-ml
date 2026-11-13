from dementia_ml.mr import epitope


def _vep(gene, terms):
    return [{"transcript_consequences": [{"gene_symbol": gene, "consequence_terms": terms}]}]


def test_missense_in_target_gene_is_flagged():
    assert epitope.protein_altering_in_gene(_vep("CD33", ["missense_variant"]), "CD33")


def test_missense_in_another_gene_is_not_flagged():
    assert not epitope.protein_altering_in_gene(_vep("SIGLEC7", ["missense_variant"]), "CD33")


def test_intronic_or_regulatory_is_not_flagged():
    vep = _vep("CD33", ["intron_variant", "upstream_gene_variant"])
    assert not epitope.protein_altering_in_gene(vep, "CD33")


def test_splice_and_frameshift_count_as_protein_altering():
    assert epitope.protein_altering_in_gene(_vep("ACE", ["splice_donor_variant"]), "ACE")
    assert epitope.protein_altering_in_gene(_vep("ACE", ["frameshift_variant"]), "ACE")


def test_verdict():
    assert (
        epitope.verdict(instrument_altering=True, proxies_altering=[])
        == "instrument is protein-altering"
    )
    assert epitope.verdict(False, ["rs1"]) == "protein-altering proxy in LD"
    assert epitope.verdict(False, []) == "no protein-altering variant in LD"


def test_client_retries_after_timeout(tmp_path, monkeypatch):
    import httpx

    client = epitope.EnsemblClient(tmp_path)
    calls = {"n": 0}

    def fake_get(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=[{"ok": 1}], request=httpx.Request("GET", url))

    monkeypatch.setattr(client.http, "get", fake_get)
    monkeypatch.setattr(epitope.time, "sleep", lambda s: None)
    assert client._get("/x") == [{"ok": 1}] and calls["n"] == 2
