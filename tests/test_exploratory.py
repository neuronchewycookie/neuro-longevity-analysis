import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_analysis.config import load_config
from data_analysis.stats import fixed_effect_meta, hedges_g_variance, mixed_term

_path = Path(__file__).resolve().parents[1] / "scripts" / "10_exploratory_pooled_meta.py"
_spec = importlib.util.spec_from_file_location("exploratory", _path)
exploratory = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exploratory)


def test_mixed_term_recovers_treatment_effect():
    # 30 animals x 4 regions, animal-level noise, true treatment effect +0.8
    rng = np.random.default_rng(0)
    rows = []
    for a in range(30):
        treated, animal_effect = a % 2, rng.normal(0, 0.5)
        for r in range(4):
            rows.append({"animal": f"a{a}", "region": f"r{r}", "treated": treated,
                         "y": 0.8 * treated + 0.3 * r + animal_effect + rng.normal(0, 0.3)})
    _, est, se, lo, hi, p = mixed_term(pd.DataFrame(rows), "y ~ treated + C(region)", "animal", "treated")
    assert lo < 0.8 < hi and p < 0.05 and se > 0


def test_fixed_effect_meta_matches_hand_calculation():
    m = fixed_effect_meta(np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    assert np.isclose(m["fixed_estimate"], 0.5)
    assert np.isclose(m["fixed_se"], np.sqrt(1 / 8))
    assert np.isclose(m["q"], 2.0)          # 4*(0.5)^2 + 4*(0.5)^2
    assert np.isclose(m["i2"], 0.5)         # (2 - 1) / 2


def test_hedges_g_variance_positive_and_shrinks_with_n():
    assert hedges_g_variance(0.5, 20, 20) < hedges_g_variance(0.5, 5, 5)


def test_design_labels():
    assert exploratory.design_of("C24") == "24 h withdrawal"
    assert exploratory.design_of("HS") == "30 d + context"
    assert exploratory.design_of("CC") == "30 d + drug (relapse)"
    assert exploratory.design_of("SH") == "first acute exposure"


def test_marker_genes_do_not_overlap_cas():
    mapping_path = load_config()["paths"]["tables"] / "cas_gene_mapping.csv"
    if not mapping_path.is_file():
        pytest.skip("run scripts/00_cas_gene_mapping.py first")
    cas = set(pd.read_csv(mapping_path)["gene"])
    markers = {g for genes in exploratory.MARKERS.values() for g in genes}
    assert not markers & cas
