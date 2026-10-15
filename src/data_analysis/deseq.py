"""Per-region differential expression with pydeseq2, cached as parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from data_analysis.loaders import Dataset

log = logging.getLogger(__name__)


def prefilter(counts: pd.DataFrame, min_count: int, min_samples: int) -> pd.DataFrame:
    """Keep genes with at least `min_count` reads in at least `min_samples` samples."""
    return counts[(counts >= min_count).sum(axis=1) >= min_samples]


def region_contrasts(ds: Dataset, region: str, comparisons: list[tuple[str, str]], group_col: str = "group",
                     min_count: int = 10, n_cpus: int = 4) -> dict[str, pd.DataFrame]:
    """Fit `~group` on one region and return DESeq2 Wald results for each (treat, ctrl) contrast.

    Genes are prefiltered to >= min_count reads in at least as many samples as the smallest group.
    Results have columns baseMean, log2FoldChange, lfcSE, stat, pvalue, padj (genome-wide BH).
    """
    sub = ds.subset(region=region)
    meta = sub.meta[[group_col]].astype(str)
    counts = prefilter(sub.counts, min_count, int(meta[group_col].value_counts().min()))
    dds = DeseqDataSet(counts=counts.T, metadata=meta, design=f"~{group_col}",
                       inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.deseq2()
    out = {}
    for treat, ctrl in comparisons:
        stat = DeseqStats(dds, contrast=[group_col, treat, ctrl], inference=DefaultInference(n_cpus=n_cpus), quiet=True)
        stat.summary()
        out[f"{treat} vs {ctrl}"] = stat.results_df
    return out


def cached_contrasts(ds: Dataset, comparisons: list[tuple[str, str]], cache_dir: str | Path,
                     **kwargs) -> pd.DataFrame:
    """All regions x contrasts as one long table, reusing `{dataset}_deseq2.parquet` when present."""
    cache = Path(cache_dir) / f"{ds.name}_deseq2.parquet"
    wanted = {f"{t} vs {c}" for t, c in comparisons}
    if cache.is_file():
        table = pd.read_parquet(cache)
        if set(table["comparison"]) >= wanted and set(table["region"]) >= set(map(str, ds.regions)):
            return table[table["comparison"].isin(wanted)]
    frames = []
    for region in ds.regions:
        log.info("%s %s: DESeq2", ds.name, region)
        for comparison, res in region_contrasts(ds, region, comparisons, **kwargs).items():
            frames.append(res.assign(region=str(region), comparison=comparison).rename_axis("gene_id").reset_index())
    table = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache, index=False)
    return table
