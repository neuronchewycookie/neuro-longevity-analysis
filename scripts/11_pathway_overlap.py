"""EXPLORATORY (post hoc): which biological processes does the 24 h drug response share with aging?

Genome-wide preranked GSEA (gseapy) on
  - aging: Spearman rho of log-CPM with age in Hahn regions cp, hi, cor and the mean over 15 regions
  - drugs: DESeq2 Wald statistics for cocaine (Walker) and heroin (Browne), all regions x comparisons
using a pathway list fixed before running (4 categories: inflammation/immune, mitochondria/energy,
synapse, proteostasis), plus all 50 Hallmark sets for a genome-wide correlation check.
Drug regions are paired with Hahn regions as in the CAS analysis (Cpu/dStri->cp, Hipp/vHPC->hi,
PFC->cor, others -> mean of 15 regions).

Reads existing caches and tables only; writes new pathway_* files.
"""

from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")

import gseapy
import numpy as np
import pandas as pd
from scipy import stats as sstats

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import load_processed
from data_analysis.normalize import log_cpm
from data_analysis.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pathway_overlap")

MSIGDB = "MSigDB_2025.1.Mm"
GMT_FILES = ["mh.all.v2025.1.Mm.symbols.gmt", "m5.go.bp.v2025.1.Mm.symbols.gmt"]

# Fixed before running. Two names were replaced by their closest MSigDB 2025.1 equivalents.
PATHWAYS = {
    "inflammation / immune": [
        "HALLMARK_INFLAMMATORY_RESPONSE", "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_COMPLEMENT", "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    ],
    "mitochondria / energy": [
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_FATTY_ACID_METABOLISM", "HALLMARK_GLYCOLYSIS",
        "GOBP_MITOCHONDRIAL_RESPIRATORY_CHAIN_COMPLEX_ASSEMBLY",
    ],
    "synapse": [
        "GOBP_SYNAPTIC_SIGNALING", "GOBP_CHEMICAL_SYNAPTIC_TRANSMISSION_POSTSYNAPTIC",
        "GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY", "GOBP_SYNAPSE_ORGANIZATION",
    ],
    "proteostasis": [
        "HALLMARK_UNFOLDED_PROTEIN_RESPONSE", "HALLMARK_MTORC1_SIGNALING", "GOBP_PROTEIN_FOLDING",
        "GOBP_PROTEASOME_MEDIATED_UBIQUITIN_DEPENDENT_PROTEIN_CATABOLIC_PROCESS",
        "GOBP_PROCESS_UTILIZING_AUTOPHAGIC_MECHANISM",
    ],
}
RENAMED = {
    "GOBP_CHEMICAL_SYNAPTIC_TRANSMISSION_POSTSYNAPTIC": "GOBP_CHEMICAL_SYNAPTIC_TRANSMISSION (not in 2025.1)",
    "GOBP_PROCESS_UTILIZING_AUTOPHAGIC_MECHANISM": "GOBP_AUTOPHAGY (GO:0006914 renamed)",
}
CATEGORY = {p: c for c, ps in PATHWAYS.items() for p in ps}
HAHN_REF = {"Cpu": "cp", "dStri": "cp", "Hipp": "hi", "vHPC": "hi", "PFC": "cor"}  # others -> mean15
DRUGS = {"walker": "cocaine", "heroin": "heroin"}
GSEA = dict(min_size=15, max_size=1100, permutation_num=1000, seed=42, threads=4, no_plot=True, outdir=None)


def read_gmt(path) -> dict[str, list[str]]:
    sets = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            name, _, *genes = line.rstrip("\n").split("\t")
            sets[name] = genes
    return sets


def prerank(ranks: pd.Series, gene_sets: dict) -> pd.DataFrame:
    """gseapy prerank -> one row per gene set: nes, pval, fdr (gseapy), n_genes in set."""
    ranks = ranks.dropna()
    ranks = ranks[~ranks.index.duplicated()].sort_values(ascending=False)
    res = gseapy.prerank(rnk=ranks, gene_sets=gene_sets, **GSEA).res2d
    out = pd.DataFrame({
        "pathway": res["Term"].astype(str),
        "nes": pd.to_numeric(res["NES"], errors="coerce"),
        "pval": pd.to_numeric(res["NOM p-val"], errors="coerce"),
        "gseapy_fdr": pd.to_numeric(res["FDR q-val"], errors="coerce"),
        "n_genes": res["Tag %"].astype(str).str.split("/").str[1].astype(float),
    })
    out["prespecified"] = out["pathway"].isin(CATEGORY)
    out["category"] = out["pathway"].map(CATEGORY).fillna("hallmark (other)")
    out["fdr_prespecified"] = np.nan
    pre = out["prespecified"]
    out.loc[pre, "fdr_prespecified"] = bh_fdr(out.loc[pre, "pval"]).to_numpy()
    return out


def hahn_age_ranks(cfg: dict) -> dict[str, pd.Series]:
    """Genome-wide Spearman rho of log-CPM with age, per Hahn region; plus the mean over 15 regions."""
    hahn = load_processed("hahn", cfg)
    per_region = {}
    for region in hahn.regions:
        sub = hahn.subset(region=region)
        keep = sub.counts.mean(axis=1) >= cfg["cas"]["min_mean_count"]
        expr = log_cpm(sub.counts)[keep]
        age_rank = sstats.rankdata(sub.meta["age_months"].to_numpy())
        gene_rank = expr.rank(axis=1).to_numpy()
        a = (age_rank - age_rank.mean()) / age_rank.std()
        g = (gene_rank - gene_rank.mean(axis=1, keepdims=True)) / gene_rank.std(axis=1, keepdims=True)
        per_region[str(region)] = pd.Series((g * a).mean(axis=1), index=expr.index)
    table = pd.DataFrame(per_region)
    refs = {r: table[r].dropna() for r in ("cp", "hi", "cor")}
    refs["mean15"] = table.dropna().mean(axis=1)
    return refs


def drug_ranks(cfg: dict, name: str) -> dict[tuple[str, str], pd.Series]:
    """DESeq2 Wald statistic per (region, comparison), mapped Ensembl -> symbol (largest |stat| kept)."""
    deseq = pd.read_parquet(cfg["paths"]["processed"] / "deseq2" / f"{name}_deseq2.parquet")
    ids = pd.read_parquet(cfg["paths"]["processed"] / "idmap" / "mouse_symbol_ensembl.parquet")
    deseq = deseq.merge(ids, left_on="gene_id", right_on="ensembl_gene_id").dropna(subset=["stat"])
    out = {}
    for (region, comparison), sub in deseq.groupby(["region", "comparison"]):
        sub = sub.reindex(sub["stat"].abs().sort_values(ascending=False).index).drop_duplicates("symbol")
        out[(region, comparison)] = sub.set_index("symbol")["stat"]
    return out


def design_of(comparison: str) -> str:
    treat = comparison.split(" vs ")[0]
    if treat.endswith("24"):
        return "24 h withdrawal"
    if treat[0] == "S":
        return "first acute exposure"
    return "30 d + drug (relapse)" if treat[1] == treat[0] else "30 d + context"


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    folder = cfg["paths"]["raw"] / MSIGDB

    all_sets = {}
    for f in GMT_FILES:
        all_sets.update(read_gmt(folder / f))
    missing = [p for p in CATEGORY if p not in all_sets]
    if missing:
        raise ValueError(f"pre-specified pathways not found: {missing}")
    gene_sets = {p: all_sets[p] for p in list(CATEGORY) + [s for s in all_sets if s.startswith("HALLMARK_")]}
    pd.DataFrame([{"category": CATEGORY[p], "pathway": p, "n_genes_in_set": len(all_sets[p]),
                   "replaces": RENAMED.get(p, "")} for p in CATEGORY]).to_csv(tables / "table_pathway_list.csv", index=False)

    aging = []
    for ref, ranks in hahn_age_ranks(cfg).items():
        log.info("GSEA aging %s (%d genes)", ref, len(ranks))
        aging.append(prerank(ranks, gene_sets).assign(aging_ref=ref))
    aging = pd.concat(aging, ignore_index=True).assign(analysis="exploratory (post hoc)")
    aging.to_csv(tables / "table_pathway_aging_nes.csv", index=False)

    drugs = []
    for name, drug in DRUGS.items():
        for (region, comparison), ranks in drug_ranks(cfg, name).items():
            log.info("GSEA %s %s %s (%d genes)", drug, region, comparison, len(ranks))
            drugs.append(prerank(ranks, gene_sets).assign(drug=drug, region=region, comparison=comparison,
                                                          design=design_of(comparison),
                                                          aging_ref=HAHN_REF.get(region, "mean15")))
    drugs = pd.concat(drugs, ignore_index=True).assign(analysis="exploratory (post hoc)")
    drugs.to_csv(tables / "table_pathway_drug_nes.csv", index=False)

    # Overlap: same direction as aging in the paired Hahn reference
    paired = drugs.merge(aging[["aging_ref", "pathway", "nes", "fdr_prespecified"]],
                         on=["aging_ref", "pathway"], suffixes=("_drug", "_aging"))
    paired["same_direction"] = np.sign(paired["nes_drug"]) == np.sign(paired["nes_aging"])
    pre = paired[paired["prespecified"]]
    overlap = (pre.groupby(["drug", "design", "category"])
               .agg(n_pathway_x_region=("pathway", "size"), frac_same_direction=("same_direction", "mean"),
                    mean_nes_drug=("nes_drug", "mean"), mean_nes_aging=("nes_aging", "mean"),
                    n_drug_fdr_lt_05=("fdr_prespecified_drug", lambda s: int((s < 0.05).sum())))
               .reset_index().assign(analysis="exploratory (post hoc)"))
    hallmark_corr = []
    for (drug, region, comparison), sub in paired[paired["pathway"].str.startswith("HALLMARK_")].groupby(
            ["drug", "region", "comparison"]):
        rho, p = sstats.spearmanr(sub["nes_drug"], sub["nes_aging"])
        hallmark_corr.append({"drug": drug, "region": region, "comparison": comparison,
                              "design": design_of(comparison), "n_hallmark": len(sub),
                              "spearman_rho_nes": rho, "p": p})
    hallmark_corr = pd.DataFrame(hallmark_corr)
    overlap.to_csv(tables / "table_pathway_overlap.csv", index=False)
    hallmark_corr.to_csv(tables / "table_pathway_hallmark_correlation.csv", index=False)

    plots.save(plots.pathway_heatmap(aging, drugs, PATHWAYS, design="24 h withdrawal"),
               figures / "fig_pathway_overlap.png")

    pd.set_option("display.width", 220)
    print(aging[aging["prespecified"]].pivot(index="pathway", columns="aging_ref", values="nes").round(2).to_string())
    print(overlap.query("design == '24 h withdrawal'").round(2).to_string(index=False))
    print(hallmark_corr.groupby(["drug", "design"])["spearman_rho_nes"].agg(["mean", "min", "max"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
