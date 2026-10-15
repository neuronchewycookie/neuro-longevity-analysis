"""Step 3 - gene-level view of the cocaine CAS results (Walker et al. 2018).

DESeq2 (pydeseq2, design ~group) per region; for the 4 comparisons of Step 2 the 82 CAS genes'
log2 fold changes are extracted. Concordance = share of CAS genes moving in their aging
direction (up genes up, down genes down), tested against 0.5 with a binomial test.

Outputs:
  data/processed/deseq2/walker_deseq2.parquet      full DESeq2 results (cache, not committed)
  results/tables/walker_cas_genes_log2fc.csv       CAS genes x region x comparison
  results/tables/table_walker_cas_concordance.csv  concordance per region x comparison (+ by gene group)
  results/figures/fig4_cas_genes_log2fc_heatmap.png
"""

from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.deseq import cached_contrasts
from data_analysis.loaders import load_processed
from data_analysis.loaders.walker import REGIONS
from data_analysis.scoring import signature_for
from data_analysis.signatures import CAS_GENE_GROUPS, OTHER_GROUP, cas_gene_group
from data_analysis.stats import direction_concordance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("walker_gene_level")

COMPARISONS = [("C24", "S24"), ("CS", "SS"), ("CC", "SS"), ("SC", "SS")]


def cas_lfc(deseq: pd.DataFrame, dataset: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """DESeq2 rows for CAS genes, labelled with symbol, aging direction and display group."""
    sig = signature_for(dataset, mapping)
    direction = {g: "up" for g in sig.up} | {g: "down" for g in sig.down}
    out = deseq[deseq["gene_id"].isin(direction)].copy()
    out["gene"] = out["gene_id"].map(sig.symbol)
    out["direction"] = out["gene_id"].map(direction)
    out["gene_group"] = out["gene"].map(cas_gene_group)
    return out


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]

    walker = load_processed("walker", cfg)
    mapping = pd.read_csv(tables / "cas_gene_mapping.csv")
    deseq = cached_contrasts(walker, COMPARISONS, cfg["paths"]["processed"] / "deseq2",
                             min_count=cfg["cas"]["min_mean_count"])

    lfc = cas_lfc(deseq, "walker", mapping)
    keep = ["region", "comparison", "gene", "gene_id", "direction", "gene_group", "baseMean",
            "log2FoldChange", "lfcSE", "pvalue", "padj"]
    lfc[keep].to_csv(tables / "walker_cas_genes_log2fc.csv", index=False)

    overall = direction_concordance(lfc, ["region", "comparison"]).assign(gene_group="all CAS genes")
    by_group = direction_concordance(lfc, ["region", "comparison", "gene_group"])
    concordance = pd.concat([overall, by_group], ignore_index=True)
    concordance.to_csv(tables / "table_walker_cas_concordance.csv", index=False)

    comparisons = [f"{t} vs {c}" for t, c in COMPARISONS]
    lfc["column"] = lfc["region"] + " | " + lfc["comparison"]
    col_order = [f"{r} | {c}" for r in REGIONS for c in comparisons]
    group_order = [*CAS_GENE_GROUPS, OTHER_GROUP]
    genes = lfc.drop_duplicates("gene")[["gene", "gene_group", "direction"]]
    genes["g_rank"] = genes["gene_group"].map({g: i for i, g in enumerate(group_order)})
    gene_order = genes.sort_values(["g_rank", "direction", "gene"], ascending=[True, False, True])["gene"].tolist()
    fig = plots.gene_lfc_heatmap(lfc, gene_order, col_order, dict(zip(genes["gene"], genes["gene_group"])),
                                 dict(zip(genes["gene"], genes["direction"])),
                                 "Cocaine SA: log2FC of CAS genes (DESeq2, per region)")
    plots.save(fig, figures / "fig4_cas_genes_log2fc_heatmap.png")

    print(overall.pivot(index="region", columns="comparison", values="frac_aging_like").round(2).to_string())
    print(overall[overall["binom_fdr"] < 0.05][["region", "comparison", "n_aging_like", "n_genes",
                                                "frac_aging_like", "binom_fdr"]].round(4).to_string(index=False))
    sig_genes = lfc[lfc["padj"] < 0.05]
    print(f"\nCAS genes with DESeq2 padj < 0.05: {len(sig_genes)} rows")
    print(sig_genes[["region", "comparison", "gene", "direction", "log2FoldChange", "padj"]]
          .sort_values("padj").round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
