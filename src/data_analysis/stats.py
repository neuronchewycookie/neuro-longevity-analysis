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


def direction_concordance(lfc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Share of genes moving in their aging direction, with a two-sided binomial test vs 0.5.

    `lfc` needs columns log2FoldChange and direction ('up'/'down'). Genes are treated as
    independent, which co-regulated CAS genes are not, so p-values are optimistic.
    """
    data = lfc.dropna(subset=["log2FoldChange"]).copy()
    sign = np.where(data["direction"] == "up", 1, -1)
    data["aging_like"] = np.sign(data["log2FoldChange"]) == sign
    rows = []
    for key, sub in data.groupby(by, observed=True):
        k, n = int(sub["aging_like"].sum()), len(sub)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))),
                     "n_genes": n, "n_aging_like": k, "frac_aging_like": k / n,
                     "binom_p": stats.binomtest(k, n, 0.5).pvalue})
    out = pd.DataFrame(rows)
    out["binom_fdr"] = bh_fdr(out["binom_p"])
    return out


def mixed_term(data: pd.DataFrame, formula: str, groups: str, term: str, reml: bool = True):
    """Fit a random-intercept MixedLM and return (fit, estimate, SE, CI low, CI high, p) for `term`."""
    fit = smf.mixedlm(formula, data=data, groups=data[groups]).fit(reml=reml)
    ci = fit.conf_int().loc[term]
    return fit, float(fit.params[term]), float(fit.bse[term]), float(ci[0]), float(ci[1]), float(fit.pvalues[term])


def lrt_pvalue(full, reduced) -> tuple[float, int]:
    """Likelihood-ratio test of two nested ML fits; returns (p, df)."""
    df = len(full.fe_params) - len(reduced.fe_params)
    stat = 2 * (full.llf - reduced.llf)
    return float(stats.chi2.sf(max(stat, 0.0), df)), int(df)


def fixed_effect_meta(estimates: np.ndarray, ses: np.ndarray) -> dict:
    """Inverse-variance fixed-effect pooling with Cochran's Q, I^2 and DerSimonian-Laird random effects."""
    est, se = np.asarray(estimates, float), np.asarray(ses, float)
    w = 1 / se**2
    pooled = float(np.sum(w * est) / np.sum(w))
    pooled_se = float(np.sqrt(1 / np.sum(w)))
    q = float(np.sum(w * (est - pooled) ** 2))
    k = len(est)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    tau2 = max(0.0, (q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
    w_re = 1 / (se**2 + tau2)
    re = float(np.sum(w_re * est) / np.sum(w_re))
    re_se = float(np.sqrt(1 / np.sum(w_re)))
    z = pooled / pooled_se
    return {
        "k": k,
        "fixed_estimate": pooled, "fixed_se": pooled_se,
        "fixed_ci_low": pooled - 1.96 * pooled_se, "fixed_ci_high": pooled + 1.96 * pooled_se,
        "fixed_p": float(2 * stats.norm.sf(abs(z))),
        "q": q, "q_p": float(stats.chi2.sf(q, k - 1)) if k > 1 else np.nan, "i2": i2, "tau2": float(tau2),
        "random_estimate": re, "random_ci_low": re - 1.96 * re_se, "random_ci_high": re + 1.96 * re_se,
        "random_p": float(2 * stats.norm.sf(abs(re / re_se))),
    }


def hedges_g_variance(g: float, n1: int, n2: int) -> float:
    """Large-sample variance of Hedges' g."""
    return (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))
