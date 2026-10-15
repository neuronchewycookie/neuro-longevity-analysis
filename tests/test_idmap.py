import pandas as pd

from data_analysis.idmap import map_mouse_symbols, map_orthologs, mouse_id_candidates


class FakeMyGene:
    """Offline stand-in for MyGeneCache: answers from a {(scopes, query): hits} dict."""

    def __init__(self, answers):
        self.answers = answers

    def querymany(self, queries, scopes, species, fields):
        return {str(q): self.answers.get((scopes, str(q)), []) for q in queries}


def test_map_mouse_symbols_uses_deglists_then_mygene():
    table = pd.DataFrame({"ensembl_gene_id": ["ENSMUSG01"], "symbol": ["Gpr17"]})
    mg = FakeMyGene({
        ("symbol,alias", "Hist3h2a"): [
            {"symbol": "H2ac25", "_score": 18, "ensembl": [{"gene": "ENSMSIG99"}, {"gene": "ENSMUSG02"}]}
        ],
    })
    out = map_mouse_symbols(["Gpr17", "Hist3h2a", "Nope"], table, mg).set_index("gene")
    assert out.loc["Gpr17", "id_source"] == "deglists"
    assert out.loc["Hist3h2a", "ensembl_gene_id"] == "ENSMUSG02"  # non-reference strain ID dropped
    assert out.loc["Hist3h2a", "current_symbol"] == "H2ac25"
    assert out.loc["Nope", "id_source"] == "unmapped"


def test_retired_ids_are_candidates():
    table = pd.DataFrame({"ensembl_gene_id": ["ENSMUSG00000112148"], "symbol": ["Lilrb4a"]})
    row = map_mouse_symbols(["Lilrb4a"], table, FakeMyGene({})).iloc[0]
    assert mouse_id_candidates(row) == ["ENSMUSG00000112148", "ENSMUSG00000062593"]


def test_map_orthologs_statuses():
    mouse = pd.DataFrame({
        "gene": ["C1qa", "C4b", "H2-K1", "Trem2", "Pcdhb9", "Ccl6"],
        "ensembl_gene_id": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "current_symbol": ["C1qa", "C4b", "H2-K1", "Trem2", "Pcdhb9", "Ccl6"],
    })
    hg = "ensembl.gene"
    mg = FakeMyGene({
        (hg, "E1"): [{"homologene": {"genes": [[10090, 1], [9606, 712]]}}],
        (hg, "E2"): [{"homologene": {"genes": [[9606, 720], [9606, 721]]}}],
        (hg, "E3"): [{"homologene": {"genes": [[9606, 3105]]}}],
        ("symbol", "TREM2"): [{"symbol": "TREM2", "entrezgene": 54209}],
        ("symbol", "PCDHB9"): [{"symbol": "PCDHB9", "entrezgene": 56127}],
        ("entrezgene", "712"): [{"symbol": "C1QA", "ensembl": {"gene": "ENSG00000173372"}}],
        ("entrezgene", "54209"): [{"symbol": "TREM2", "ensembl": [{"gene": "ENSG00000095970"}]}],
    })
    out = map_orthologs(mouse, "human", mg).set_index("gene")
    assert out.loc["C1qa", "human_status"] == "one_to_one"
    assert out.loc["C1qa", "human_ensembl"] == "ENSG00000173372"
    assert out.loc["C4b", "human_status"] == "one_to_many"
    assert out.loc["H2-K1", "human_status"] == "excluded_mhc"
    assert out.loc["Trem2", "human_status"] == "symbol_match"
    assert out.loc["Pcdhb9", "human_status"] == "no_ortholog"  # clustered Pcdh: no symbol fallback
    assert out.loc["Ccl6", "human_status"] == "no_ortholog"
