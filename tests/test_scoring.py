import numpy as np
import pandas as pd
import pytest

from data_analysis.scoring import signature_for, vision_score, zscore_score


@pytest.fixture
def log_expr():
    rng = np.random.default_rng(0)
    genes = [f"g{i}" for i in range(200)]
    return pd.DataFrame(rng.normal(5, 2, size=(200, 6)), index=genes, columns=[f"s{j}" for j in range(6)])


def test_vision_matches_znormalized_columns(log_expr):
    # VISION's correction equals scoring column-z-normalized expression and scaling by sqrt(n+m)
    up, down = ["g1", "g2", "g3"], ["g4"]
    bg = log_expr.index
    z = (log_expr - log_expr.mean()) / log_expr.std(ddof=0)
    expected = (z.loc[up].sum() - z.loc[down].sum()) / 4 * np.sqrt(4)
    assert np.allclose(vision_score(log_expr, up, down, bg), expected)


def test_vision_is_invariant_to_sample_scaling(log_expr):
    # adding a constant to one sample (global depth shift on log scale) must not change its score
    shifted = log_expr.copy()
    shifted["s0"] += 3
    up, down = ["g1", "g2"], ["g3"]
    assert np.allclose(vision_score(shifted, up, down, shifted.index), vision_score(log_expr, up, down, log_expr.index))


def test_zscore_score_centered(log_expr):
    cas = zscore_score(log_expr, ["g1", "g2"], ["g3"])
    assert abs(cas.mean()) < 1e-9


def test_signature_for_uses_dataset_ids():
    mapping = pd.DataFrame({
        "gene": ["A", "B", "C"], "direction": ["up", "down", "up"],
        "walker_id": ["E1", "E2", "E3"], "in_walker": [True, True, False],
        "human_ensembl": ["H1;H9", None, "H3"], "in_human": [True, False, True],
    })
    w = signature_for("walker", mapping)
    assert (w.up, w.down) == (["E1"], ["E2"])
    h = signature_for("human_cud", mapping)
    assert h.up == ["H1", "H3"] and h.symbol["H1"] == "A"


def test_matched_random_signatures_keep_size_split_and_bins(log_expr):
    from data_analysis.scoring import matched_random_signatures

    mean_expr = log_expr.mean(axis=1)
    up, down = ["g1", "g2", "g3", "g4"], ["g5"]
    rng = np.random.default_rng(1)
    draws = matched_random_signatures(mean_expr, up, down, n_draws=50, rng=rng, n_bins=10)
    bins = pd.qcut(mean_expr.rank(method="first"), 10, labels=False)
    for u, d in draws:
        assert (len(u), len(d)) == (4, 1)
        assert not set(u + d) & set(up + down)
        assert list(bins[u + d]) == list(bins[up + down])


def test_mouse_signature_uses_retired_ids_and_drops_absent():
    from data_analysis.scoring import mouse_signature

    genes = pd.DataFrame({"gene": ["Lilrb4a", "Gfap", "Nope"], "direction": ["up", "up", "down"]})
    ids = pd.DataFrame({"gene": ["Lilrb4a", "Gfap", "Nope"],
                        "ensembl_gene_id": ["ENSMUSG00000112148", "ENSMUSG01", "ENSMUSG02"],
                        "retired_ensembl_ids": ["ENSMUSG00000062593", None, None]})
    sig = mouse_signature(genes, ids, "walker", present={"ENSMUSG00000062593", "ENSMUSG01"})
    assert (sig.up, sig.down) == (["ENSMUSG00000062593", "ENSMUSG01"], [])
    hahn = mouse_signature(genes, ids, "hahn", present={"Gfap"})
    assert hahn.up == ["Gfap"]
