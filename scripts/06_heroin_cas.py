"""Heroin self-administration x CAS (Browne et al., GSE228031), side by side with cocaine.

Reuses the cocaine analysis unchanged with groups renamed C -> H: per-region CAS comparisons
(Hedges' g, bootstrap CI, Welch / Mann-Whitney, matched random-signature permutation, BH FDR,
months-of-aging conversion, QC-excluded sensitivity run) and the DESeq2 concordance of CAS genes.
Regions are matched to Walker as dStri<->Cpu, vHPC<->Hipp, PFC, NAc, VTA, BLA.

Requires a passed Hahn positive control and the cocaine results (scripts/03, scripts/04).

Outputs:
  results/tables/table_heroin_cas_stats.csv (+ _excl_flagged)
  results/tables/heroin_cas_scores.csv
  results/tables/heroin_cas_genes_log2fc.csv
  results/tables/table_heroin_cas_concordance.csv
  results/tables/table_cocaine_vs_heroin.csv
  results/figures/fig5_cocaine_vs_heroin.png
  results/figures/fig5b_heroin_cas_boxplots.png
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from data_analysis import plots
from data_analysis.comparisons import add_aging_equivalent, compare_cas
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.deseq import cached_contrasts
from data_analysis.loaders import load_processed
from data_analysis.loaders.heroin import GROUPS, WALKER_REGION
from data_analysis.loaders.walker import REGIONS as WALKER_REGIONS
from data_analysis.scoring import signature_for
from data_analysis.stats import direction_concordance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("heroin_cas")

COMPARISONS = [("H24", "S24"), ("HS", "SS"), ("HH", "SS"), ("SH", "SS")]
# Heroin region -> Hahn region for the months-of-aging conversion (same mapping as cocaine)
HAHN_REGION = {"dStri": "cp", "vHPC": "hi", "PFC": "cor"}
# Drug-neutral names so cocaine and heroin comparisons line up in one table
DESIGN = {
    "C24 vs S24": "24 h withdrawal", "H24 vs S24": "24 h withdrawal",
    "CS vs SS": "30 d + context", "HS vs SS": "30 d + context",
    "CC vs SS": "30 d + drug (relapse)", "HH vs SS": "30 d + drug (relapse)",
    "SC vs SS": "first acute exposure", "SH vs SS": "first acute exposure",
}

_scripts = Path(__file__).resolve().parent


def _sibling(name: str):
    """Import a function module from a numbered sibling script (e.g. 03_walker_cas.py)."""
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _scripts / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qc_flagged = _sibling("03_walker_cas.py").qc_flagged
cas_lfc = _sibling("04_walker_gene_level.py").cas_lfc


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]

    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    heroin = load_processed("heroin", cfg)
    mapping = pd.read_csv(tables / "cas_gene_mapping.csv")
    signature = signature_for("heroin", mapping)
    hahn_slopes = pd.read_csv(tables / "table_hahn_cas_slopes.csv")
    flagged = qc_flagged(cfg, "heroin")
    log.info("QC-flagged heroin samples (sensitivity run excludes): %s", sorted(flagged))

    # CAS comparisons: primary (all samples) and sensitivity (QC-flagged removed)
    results = {}
    for label, exclude in {"all": None, "excl_flagged": flagged}.items():
        rng = np.random.default_rng(cfg["seed"])
        stats, scores = compare_cas(heroin, signature, COMPARISONS, cfg["cas"], rng, exclude=exclude)
        stats = add_aging_equivalent(stats, HAHN_REGION, hahn_slopes)
        stats["run"] = label
        results[label] = (stats, scores)
    stats, scores = results["all"]
    stats.to_csv(tables / "table_heroin_cas_stats.csv", index=False)
    results["excl_flagged"][0].to_csv(tables / "table_heroin_cas_stats_excl_flagged.csv", index=False)
    scores.reset_index().to_csv(tables / "heroin_cas_scores.csv", index=False)

    # Gene level: DESeq2 concordance of CAS genes with their aging direction
    deseq = cached_contrasts(heroin, COMPARISONS, cfg["paths"]["processed"] / "deseq2",
                             min_count=cfg["cas"]["min_mean_count"])
    lfc = cas_lfc(deseq, "heroin", mapping)
    lfc.to_csv(tables / "heroin_cas_genes_log2fc.csv", index=False)
    concordance = direction_concordance(lfc, ["region", "comparison"])
    concordance.to_csv(tables / "table_heroin_cas_concordance.csv", index=False)

    # Cocaine vs heroin side by side, regions in Walker naming
    cocaine = pd.read_csv(tables / "table_walker_cas_stats.csv").assign(drug="cocaine")
    coc_conc = pd.read_csv(tables / "table_walker_cas_concordance.csv")
    coc_conc = coc_conc[coc_conc["gene_group"] == "all CAS genes"]
    her = stats.assign(drug="heroin", region=stats["region"].map(WALKER_REGION))
    her_conc = concordance.assign(region=concordance["region"].map(WALKER_REGION))
    both = pd.concat([cocaine, her], ignore_index=True)
    both["design"] = both["comparison"].map(DESIGN)
    conc_both = pd.concat([coc_conc.assign(drug="cocaine"), her_conc.assign(drug="heroin")])
    both = both.merge(conc_both[["drug", "region", "comparison", "frac_aging_like"]],
                      on=["drug", "region", "comparison"], how="left")
    both["region_pair"] = both["region"].map(
        {r: r if r not in {"Cpu", "Hipp"} else {"Cpu": "Cpu / dStri", "Hipp": "Hipp / vHPC"}[r] for r in WALKER_REGIONS})
    cols = ["drug", "region_pair", "design", "comparison", "n_treat", "n_ctrl", "g", "g_ci_low", "g_ci_high",
            "welch_fdr", "perm_fdr", "frac_aging_like", "months_equivalent"]
    both[cols].to_csv(tables / "table_cocaine_vs_heroin.csv", index=False)

    designs = list(dict.fromkeys(DESIGN.values()))
    heat = both.assign(region=both["region_pair"], comparison=both["drug"] + " | " + both["design"])
    rows = [r if r not in {"Cpu", "Hipp"} else {"Cpu": "Cpu / dStri", "Hipp": "Hipp / vHPC"}[r] for r in WALKER_REGIONS]
    col_order = [f"{drug} | {d}" for d in designs for drug in ("cocaine", "heroin")]
    plots.save(plots.effect_heatmap(heat, rows, col_order, "Cocaine vs heroin SA: CAS effect size by region"),
               figures / "fig5_cocaine_vs_heroin.png")
    drug = {g: ("Heroin SA" if g.startswith("H") else "Saline SA") for g in GROUPS}
    plots.save(plots.group_boxplots(scores, list(map(str, heroin.regions)), GROUPS, drug,
                                    "Browne: CAS per region and group (heroin)"),
               figures / "fig5b_heroin_cas_boxplots.png")

    show = ["region", "comparison", "n_treat", "n_ctrl", "g", "g_ci_low", "g_ci_high", "welch_fdr", "perm_fdr"]
    print(stats[show].round(3).to_string(index=False))
    print(concordance.pivot(index="region", columns="comparison", values="frac_aging_like").round(2).to_string())
    sens = results["excl_flagged"][0].set_index(["region", "comparison"])["g"]
    delta = (stats.set_index(["region", "comparison"])["g"] - sens).abs()
    print(f"\nSensitivity (QC-flagged removed): max |delta g| = {delta.max():.2f} at {delta.idxmax()}")
    print(heroin.meta.assign(cas=scores["cas"]).groupby(["region", "group"], observed=True)["cas"]
          .mean().unstack()[["S24", "SS"]].round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
