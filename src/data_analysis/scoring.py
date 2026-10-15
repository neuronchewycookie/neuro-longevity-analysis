"""Common Aging Score (CAS) computation.

Two methods, selected by `cas.method` in config.yaml:

- "vision" (Hahn et al. 2023, STAR Methods; VISION, DeTomaso et al. 2019):
    per sample j, with log-normalized expression e_gj,
        s_j   = (sum_{g in UP} e_gj - sum_{g in DOWN} e_gj) / (n + m)
        E[R_j] = (n - m) / (n + m) * mean_j        mean_j, var_j: over background genes in sample j
        var(R_j) = var_j / (n + m)
        CAS_j = (s_j - E[R_j]) / sqrt(var(R_j))
  i.e. the score is z-normalized against a random signature of the same size and sign
  balance, within each sample. It does not depend on which other samples are scored.

- "zscore" (PIPELINE_SPEC section 5): z-score each gene across the samples of one region,
  then CAS = mean(z[UP]) - mean(z[DOWN]). Depends on the sample set (always one region).

Genes with low expression in a region (mean raw count < `min_mean_count`, or mean value
< `min_mean_value` for non-count data) are dropped from both the signature and the
VISION background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_analysis.loaders import Dataset
from data_analysis.normalize import log_expression

METHODS = ("vision", "zscore")

# Column in results/tables/cas_gene_mapping.csv holding each dataset's gene ID, and the
# boolean column telling whether that gene is usable in the dataset.
MAPPING_COLUMNS = {
    "hahn": ("gene", "in_hahn"),
    "walker": ("walker_id", "in_walker"),
    "heroin": ("heroin_id", "in_heroin"),
    "rat_reward": ("rat_symbol", "in_rat"),
    "human_cud": ("human_ensembl", "in_human"),
}


@dataclass
class Signature:
    """Up/down gene IDs in a dataset's own ID space, plus the CAS symbol for each."""

    up: list[str]
    down: list[str]
    symbol: dict[str, str]  # dataset gene ID -> mouse CAS symbol


def signature_for(dataset: str, mapping: pd.DataFrame) -> Signature:
    """Translate the CAS mapping table into one dataset's gene IDs."""
    id_col, ok_col = MAPPING_COLUMNS[dataset]
    usable = mapping[mapping[ok_col]].copy()
    if dataset == "human_cud":  # several Ensembl IDs may be listed; take the first
        usable[id_col] = usable[id_col].str.split(";").str[0]
    ids = dict(zip(usable[id_col], usable["gene"]))
    up = usable.loc[usable["direction"] == "up", id_col].tolist()
    down = usable.loc[usable["direction"] == "down", id_col].tolist()
    return Signature(up, down, ids)


def expressed_genes(ds: Dataset, min_mean_count: float, min_mean_value: float) -> pd.Index:
    """Genes whose mean (raw count, or normalized value) across `ds` samples passes the filter."""
    threshold = min_mean_count if ds.is_count else min_mean_value
    return ds.counts.index[ds.counts.mean(axis=1) >= threshold]


def vision_score(log_expr: pd.DataFrame, up: list[str], down: list[str], background: pd.Index) -> pd.Series:
    """VISION signed signature score, z-normalized per sample against random signatures."""
    n, m = len(up), len(down)
    if n + m == 0:
        raise ValueError("empty signature")
    bg = log_expr.loc[background]
    mu, var = bg.mean(axis=0), bg.var(axis=0, ddof=0)
    raw = (log_expr.loc[up].sum(axis=0) - log_expr.loc[down].sum(axis=0)) / (n + m)
    expected = (n - m) / (n + m) * mu
    return (raw - expected) / np.sqrt(var / (n + m))


def zscore_score(log_expr: pd.DataFrame, up: list[str], down: list[str]) -> pd.Series:
    """mean(z[UP]) - mean(z[DOWN]) with genes z-scored across the given samples."""
    sub = log_expr.loc[up + down]
    sd = sub.std(axis=1, ddof=1).replace(0, np.nan)
    z = sub.sub(sub.mean(axis=1), axis=0).div(sd, axis=0)
    up_mean = z.loc[up].mean(axis=0) if up else 0.0
    down_mean = z.loc[down].mean(axis=0) if down else 0.0
    return up_mean - down_mean


def compute_cas(ds: Dataset, signature: Signature, cas_cfg: dict, method: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every sample, region by region.

    Returns (scores, genes):
      scores: one row per sample with region, cas, n_up, n_down (genes used in that region)
      genes:  one row per region x CAS gene with `used` and the reason when dropped
    """
    method = method or cas_cfg["method"]
    if method not in METHODS:
        raise ValueError(f"unknown CAS method {method!r}; choose from {METHODS}")
    min_count = cas_cfg["min_mean_count"]
    min_value = cas_cfg.get("min_mean_value", 1.0)

    score_frames, gene_rows = [], []
    for region in ds.regions:
        sub = ds.subset(region=region)
        expressed = expressed_genes(sub, min_count, min_value)
        keep = set(expressed)
        up = [g for g in signature.up if g in keep]
        down = [g for g in signature.down if g in keep]
        for gene in signature.up + signature.down:
            present = gene in sub.counts.index
            gene_rows.append({
                "dataset": ds.name, "region": region, "gene_id": gene, "gene": signature.symbol[gene],
                "direction": "up" if gene in signature.up else "down",
                "used": gene in keep,
                "reason": "" if gene in keep else ("not in data" if not present else "low expression"),
            })

        log_expr = log_expression(sub)
        if method == "vision":
            cas = vision_score(log_expr, up, down, expressed)
        else:
            cas = zscore_score(log_expr, up, down)
        frame = sub.meta.copy()
        frame["cas"] = cas
        frame["n_up"], frame["n_down"] = len(up), len(down)
        score_frames.append(frame)

    scores = pd.concat(score_frames)
    scores.insert(0, "dataset", ds.name)
    scores["method"] = method
    return scores, pd.DataFrame(gene_rows)


def matched_random_signatures(mean_expr: pd.Series, up: list[str], down: list[str], n_draws: int,
                              rng: np.random.Generator, n_bins: int = 20) -> list[tuple[list[str], list[str]]]:
    """Random signatures whose genes match the real signature's expression levels.

    Background genes (index of `mean_expr`, excluding the signature) are split into `n_bins`
    quantile bins of mean log expression. Each real gene is replaced by a random gene from
    its own bin (no repeats within a draw), so every random signature has the same size,
    up/down split and expression profile as the real one.
    """
    genes = list(up) + list(down)
    bins = pd.Series(pd.qcut(mean_expr.rank(method="first"), n_bins, labels=False), index=mean_expr.index)
    pool = {b: mean_expr.index[(bins == b) & ~mean_expr.index.isin(genes)].to_numpy() for b in range(n_bins)}
    gene_bins = bins.loc[genes].to_numpy()
    draws = []
    for _ in range(n_draws):
        picked = np.empty(len(genes), dtype=object)
        for b in np.unique(gene_bins):
            slots = np.flatnonzero(gene_bins == b)
            picked[slots] = rng.choice(pool[b], size=len(slots), replace=False)
        draws.append((list(picked[:len(up)]), list(picked[len(up):])))
    return draws
