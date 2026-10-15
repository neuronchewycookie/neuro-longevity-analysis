import numpy as np
import pandas as pd

from data_analysis.stats import (
    bh_fdr,
    bootstrap_g_ci,
    empirical_p,
    group_slopes,
    hedges_g,
    hedges_g_rows,
    slope_difference,
)


def test_hedges_g_known_value():
    # means differ by 1 pooled SD; small-sample correction 1 - 3/(4*8-9)
    a, b = np.array([1.0, 2, 3, 4]), np.array([0.0, 1, 2, 3])
    sd = np.std(a, ddof=1)
    assert np.isclose(hedges_g(a, b), (1 / sd) * (1 - 3 / 23))


def test_hedges_g_rows_matches_scalar():
    rng = np.random.default_rng(0)
    t, c = rng.normal(size=(5, 6)), rng.normal(size=(5, 7))
    assert np.allclose(hedges_g_rows(t, c), [hedges_g(t[i], c[i]) for i in range(5)])


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 20), rng.normal(0, 1, 20)
    lo, hi = bootstrap_g_ci(a, b, 500, np.random.default_rng(1))
    assert lo < hedges_g(a, b) < hi


def test_empirical_p_bounds():
    null = np.linspace(-1, 1, 99)
    assert empirical_p(5.0, null) == 1 / 100
    assert empirical_p(0.0, null) == 1.0


def test_bh_fdr_keeps_nan():
    out = bh_fdr(pd.Series([0.01, np.nan, 0.04]))
    assert np.isnan(out[1]) and out[0] <= out[2]


def test_group_slopes_and_difference_recover_truth():
    rng = np.random.default_rng(0)
    x = np.tile(np.arange(10.0), 2)
    grp = np.repeat(["a", "b"], 10)
    y = np.where(grp == "a", 2 * x, 0.5 * x) + rng.normal(0, 0.1, 20)
    df = pd.DataFrame({"y": y, "x": x, "g": grp})
    slopes = group_slopes(df, "y", "x", "g").set_index("g")
    assert np.isclose(slopes.loc["a", "slope"], 2, atol=0.05)
    diff, p = slope_difference(df, "y", "x", "g", "a", "b")
    assert np.isclose(diff, 1.5, atol=0.05) and p < 1e-6


def test_direction_concordance_counts_aging_like_genes():
    from data_analysis.stats import direction_concordance

    lfc = pd.DataFrame({
        "region": ["A"] * 4,
        "log2FoldChange": [1.0, 0.5, -0.2, -1.0],
        "direction": ["up", "up", "up", "down"],
    })
    out = direction_concordance(lfc, ["region"]).iloc[0]
    assert (out["n_genes"], out["n_aging_like"]) == (4, 3)
