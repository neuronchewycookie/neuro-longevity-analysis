"""Group comparisons of CAS within regions (shared by the cocaine and heroin analyses)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_analysis.loaders import Dataset
from data_analysis.normalize import log_expression
from data_analysis.scoring import (
    Signature,
    expressed_genes,
    matched_random_signatures,
    vision_score,
    zscore_score,
)
from data_analysis.stats import bh_fdr, compare_groups, empirical_p, hedges_g_rows

log = logging.getLogger(__name__)


def _score(log_expr: pd.DataFrame, up: list[str], down: list[str], background: pd.Index, method: str) -> pd.Series:
    return vision_score(log_expr, up, down, background) if method == "vision" else zscore_score(log_expr, up, down)


def _null_scores(log_expr: pd.DataFrame, draws: list[tuple[list[str], list[str]]], background: pd.Index,
                 method: str) -> np.ndarray:
    """Scores of every random signature (draws x samples), vectorized for the VISION method."""
    if method != "vision":
        return np.vstack([_score(log_expr, u, d, background, method).to_numpy() for u, d in draws])
    x = log_expr.to_numpy()
    row = {g: i for i, g in enumerate(log_expr.index)}
    bg = x[[row[g] for g in background]]
    mu, var = bg.mean(axis=0), bg.var(axis=0)
    n, m = len(draws[0][0]), len(draws[0][1])
    up_idx = np.array([[row[g] for g in u] for u, _ in draws]).reshape(len(draws), n)
    down_idx = np.array([[row[g] for g in d] for _, d in draws]).reshape(len(draws), m)
    raw = (x[up_idx].sum(axis=1) - x[down_idx].sum(axis=1)) / (n + m)  # draws x samples
    return (raw - (n - m) / (n + m) * mu) / np.sqrt(var / (n + m))


def compare_cas(
    ds: Dataset,
    signature: Signature,
    comparisons: list[tuple[str, str]],
    cas_cfg: dict,
    rng: np.random.Generator,
    group_col: str = "group",
    exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per region: score CAS, then test each (treat, ctrl) group pair.

    For every comparison the table reports Hedges' g with a bootstrap CI, Welch and
    Mann-Whitney p-values, and an empirical p-value against `n_permutations` expression-matched
    random signatures of the same size and up/down split (scored on the same samples).
    BH FDR is applied across all region x comparison rows.

    `exclude` drops samples (e.g. QC-flagged) before scoring, for sensitivity analyses.
    Returns (stats, scores).
    """
    method = cas_cfg["method"]
    n_perm, n_boot = cas_cfg["n_permutations"], cas_cfg["n_bootstrap"]
    rows, score_frames = [], []

    for region in ds.regions:
        sub = ds.subset(region=region)
        if exclude:
            sub = sub.subset(pd.Series(~sub.meta.index.isin(list(exclude)), index=sub.meta.index))
        background = expressed_genes(sub, cas_cfg["min_mean_count"], cas_cfg.get("min_mean_value", 1.0))
        keep = set(background)
        up = [g for g in signature.up if g in keep]
        down = [g for g in signature.down if g in keep]
        log_expr = log_expression(sub)
        groups = sub.meta[group_col].astype(str)

        cas = _score(log_expr, up, down, background, method)
        frame = sub.meta.copy()
        frame["cas"] = cas
        score_frames.append(frame)

        draws = matched_random_signatures(log_expr.loc[background].mean(axis=1), up, down, n_perm, rng)
        null_scores = _null_scores(log_expr, draws, background, method)
        pos = {sample: i for i, sample in enumerate(log_expr.columns)}

        for treat, ctrl in comparisons:
            t_idx, c_idx = groups.index[groups == treat], groups.index[groups == ctrl]
            if len(t_idx) < 2 or len(c_idx) < 2:
                log.warning("%s %s vs %s: too few samples (%d, %d); skipped", region, treat, ctrl, len(t_idx), len(c_idx))
                continue
            result = compare_groups(cas[t_idx], cas[c_idx], n_boot, rng)
            ti, ci = [pos[i] for i in t_idx], [pos[i] for i in c_idx]
            null_g = hedges_g_rows(null_scores[:, ti], null_scores[:, ci])
            result.update({
                "region": region,
                "comparison": f"{treat} vs {ctrl}",
                "treat": treat,
                "ctrl": ctrl,
                "n_up": len(up),
                "n_down": len(down),
                "perm_p": empirical_p(result["g"], null_g),
                "null_g_sd": float(np.nanstd(null_g)),
            })
            rows.append(result)

    stats = pd.DataFrame(rows)
    stats["welch_fdr"] = bh_fdr(stats["welch_p"])
    stats["mwu_fdr"] = bh_fdr(stats["mwu_p"])
    stats["perm_fdr"] = bh_fdr(stats["perm_p"])
    stats.insert(0, "dataset", ds.name)
    stats["method"] = method
    front = ["dataset", "region", "comparison", "n_treat", "n_ctrl", "diff", "g", "g_ci_low", "g_ci_high",
             "welch_p", "welch_fdr", "mwu_p", "mwu_fdr", "perm_p", "perm_fdr"]
    stats = stats[front + [c for c in stats.columns if c not in front]]
    scores = pd.concat(score_frames)
    scores.insert(0, "dataset", ds.name)
    return stats, scores


def add_aging_equivalent(stats: pd.DataFrame, region_map: dict[str, str], hahn_slopes: pd.DataFrame) -> pd.DataFrame:
    """Convert CAS differences into months of Hahn aging: diff / slope of the matched Hahn region.

    Regions without a Hahn counterpart use the mean slope across all 15 Hahn regions and are
    marked `slope_source = 'mean_15_regions (approx.)'`.
    """
    slope = dict(zip(hahn_slopes["region"], hahn_slopes["slope"]))
    mean_slope = float(hahn_slopes["slope"].mean())
    out = stats.copy()
    out["hahn_region"] = out["region"].map(region_map)
    out["hahn_slope"] = out["hahn_region"].map(slope).fillna(mean_slope)
    out["slope_source"] = np.where(out["hahn_region"].isna(), "mean_15_regions (approx.)",
                                   "hahn_" + out["hahn_region"].fillna("") + " (approx.)")
    out.loc[out["region"].map(region_map).isin(["cp"]), "slope_source"] = "hahn_cp"
    out["months_equivalent"] = out["diff"] / out["hahn_slope"]
    return out
