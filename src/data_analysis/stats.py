"""Effect sizes, bootstrap CIs, permutation tests and FDR correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (NaNs kept as NaN)."""
    out = pd.Series(np.nan, index=pvalues.index)
    ok = pvalues.notna()
    if ok.any():
        out[ok] = multipletests(pvalues[ok], method="fdr_bh")[1]
    return out


def group_slopes(df: pd.DataFrame, y: str, x: str, group: str, alpha: float = 0.05) -> pd.DataFrame:
    """Per-group slopes of y on x from one joint linear model y ~ x + group + x:group.

    Parametrized as y ~ 0 + group + group:x so each coefficient is a group's slope; standard
    errors use the pooled residual variance, as least-squares-means trends (R lsmeans::lstrends)
    do in Hahn et al. 2023.
    """
    data = df[[y, x, group]].dropna().copy()
    data[group] = data[group].astype(str)
    fit = smf.ols(f"{y} ~ 0 + C({group}) + C({group}):{x}", data=data).fit()
    ci = fit.conf_int(alpha=alpha)
    rows = []
    for name in sorted(data[group].unique()):
        term = f"C({group})[{name}]:{x}"
        rows.append({
            group: name,
            "n": int((data[group] == name).sum()),
            "slope": fit.params[term],
            "slope_se": fit.bse[term],
            "slope_ci_low": ci.loc[term, 0],
            "slope_ci_high": ci.loc[term, 1],
            "slope_p": fit.pvalues[term],
            "intercept": fit.params[f"C({group})[{name}]"],
        })
    out = pd.DataFrame(rows)
    out["slope_fdr"] = bh_fdr(out["slope_p"])
    return out


def group_spearman(df: pd.DataFrame, y: str, x: str, group: str) -> pd.DataFrame:
    """Spearman correlation of y with x within each group, with BH FDR across groups."""
    rows = []
    for name, sub in df.groupby(group, observed=True):
        rho, p = stats.spearmanr(sub[x], sub[y])
        rows.append({group: str(name), "spearman_rho": rho, "spearman_p": p})
    out = pd.DataFrame(rows)
    out["spearman_fdr"] = bh_fdr(out["spearman_p"])
    return out


def slope_difference(df: pd.DataFrame, y: str, x: str, group: str, a: str, b: str) -> tuple[float, float]:
    """Difference of slopes (a - b) and its two-sided p-value from the joint model in `group_slopes`."""
    data = df[[y, x, group]].dropna().copy()
    data[group] = data[group].astype(str)
    fit = smf.ols(f"{y} ~ 0 + C({group}) + C({group}):{x}", data=data).fit()
    test = fit.t_test(f"C({group})[{a}]:{x} - C({group})[{b}]:{x} = 0")
    return float(np.squeeze(test.effect)), float(np.squeeze(test.pvalue))


def hedges_g(treat: np.ndarray, ctrl: np.ndarray) -> float:
    """Bias-corrected standardized mean difference (treat - ctrl) with pooled SD."""
    treat, ctrl = np.asarray(treat, float), np.asarray(ctrl, float)
    n1, n2 = len(treat), len(ctrl)
    pooled = np.sqrt(((n1 - 1) * treat.var(ddof=1) + (n2 - 1) * ctrl.var(ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return (treat.mean() - ctrl.mean()) / pooled * correction


def hedges_g_rows(treat: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """Hedges' g for each row of two (k x n1) and (k x n2) arrays; same formula as `hedges_g`."""
    n1, n2 = treat.shape[1], ctrl.shape[1]
    pooled = np.sqrt(((n1 - 1) * treat.var(axis=1, ddof=1) + (n2 - 1) * ctrl.var(axis=1, ddof=1)) / (n1 + n2 - 2))
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(pooled > 0, (treat.mean(axis=1) - ctrl.mean(axis=1)) / pooled * correction, np.nan)


def bootstrap_g_ci(treat: np.ndarray, ctrl: np.ndarray, n_boot: int, rng: np.random.Generator,
                   level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of Hedges' g, resampling each group with replacement."""
    treat, ctrl = np.asarray(treat, float), np.asarray(ctrl, float)
    t = treat[rng.integers(0, len(treat), size=(n_boot, len(treat)))]
    c = ctrl[rng.integers(0, len(ctrl), size=(n_boot, len(ctrl)))]
    gs = hedges_g_rows(t, c)
    gs = gs[np.isfinite(gs)]
    tail = (1 - level) / 2 * 100
    return float(np.percentile(gs, tail)), float(np.percentile(gs, 100 - tail))


def compare_groups(treat: np.ndarray, ctrl: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    """Mean difference, Hedges' g with bootstrap CI, Welch t-test and Mann-Whitney U p-values."""
    treat, ctrl = np.asarray(treat, float), np.asarray(ctrl, float)
    lo, hi = bootstrap_g_ci(treat, ctrl, n_boot, rng)
    return {
        "n_treat": len(treat),
        "n_ctrl": len(ctrl),
        "mean_treat": treat.mean(),
        "mean_ctrl": ctrl.mean(),
        "diff": treat.mean() - ctrl.mean(),
        "g": hedges_g(treat, ctrl),
        "g_ci_low": lo,
        "g_ci_high": hi,
        "welch_p": stats.ttest_ind(treat, ctrl, equal_var=False).pvalue,
        "mwu_p": stats.mannwhitneyu(treat, ctrl, alternative="two-sided").pvalue,
    }


def empirical_p(observed: float, null: np.ndarray) -> float:
    """Two-sided empirical p-value: share of |null| >= |observed|, with the +1 correction."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null))
