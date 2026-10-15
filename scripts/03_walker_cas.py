"""Step 2 - core analysis: cocaine self-administration x CAS (Walker et al. 2018, GSE110344).

Per region (6) and comparison (4): Hedges' g with bootstrap CI, Welch / Mann-Whitney p,
empirical p against expression-matched random signatures, BH FDR across all 24 tests, and the
CAS difference expressed as months of Hahn aging. A sensitivity run repeats everything without
QC-flagged samples (PCA isolation or low library size from scripts/01_load_all.py).

Requires a passed Hahn positive control (scripts/02_hahn_positive_control.py).

Outputs:
  results/tables/table_walker_cas_stats.csv               primary (all samples)
  results/tables/table_walker_cas_stats_excl_flagged.csv  sensitivity (QC-flagged removed)
  results/tables/walker_cas_scores.csv
  results/figures/fig2_walker_cas_heatmap.png
  results/figures/fig3_walker_cas_boxplots.png
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from data_analysis import plots
from data_analysis.comparisons import add_aging_equivalent, compare_cas
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import load_processed
from data_analysis.loaders.walker import GROUPS, REGIONS
from data_analysis.scoring import signature_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("walker_cas")

COMPARISONS = [("C24", "S24"), ("CS", "SS"), ("CC", "SS"), ("SC", "SS")]
COMPARISON_LABELS = {
    "C24 vs S24": "chronic intake, 24 h withdrawal",
    "CS vs SS": "30 d withdrawal + context re-exposure",
    "CC vs SS": "30 d withdrawal + cocaine (relapse)",
    "SC vs SS": "first acute cocaine (control comparison)",
}
# Walker region -> Hahn region used for the months-of-aging conversion (PIPELINE_SPEC Step 2)
HAHN_REGION = {"Cpu": "cp", "Hipp": "hi", "PFC": "cor"}


def qc_flagged(cfg: dict, dataset: str) -> set[str]:
    tables = cfg["paths"]["tables"]
    pca = pd.read_csv(tables / "qc_pca.csv")
    lib = pd.read_csv(tables / "library_sizes.csv")
    flagged = set(pca.loc[(pca["dataset"] == dataset) & pca["outlier"], "sample"])
    flagged |= set(lib.loc[(lib["dataset"] == dataset) & lib["low_library"], "sample"])
    return flagged


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]

    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    walker = load_processed("walker", cfg)
    signature = signature_for("walker", pd.read_csv(tables / "cas_gene_mapping.csv"))
    hahn_slopes = pd.read_csv(tables / "table_hahn_cas_slopes.csv")
    flagged = qc_flagged(cfg, "walker")
    log.info("QC-flagged Walker samples (sensitivity run excludes): %s", sorted(flagged))

    runs = {"all": None, "excl_flagged": flagged}
    results = {}
    for label, exclude in runs.items():
        rng = np.random.default_rng(cfg["seed"])
        stats, scores = compare_cas(walker, signature, COMPARISONS, cfg["cas"], rng, exclude=exclude)
        stats = add_aging_equivalent(stats, HAHN_REGION, hahn_slopes)
        stats["description"] = stats["comparison"].map(COMPARISON_LABELS)
        stats["run"] = label
        results[label] = (stats, scores)

    stats, scores = results["all"]
    stats.to_csv(tables / "table_walker_cas_stats.csv", index=False)
    results["excl_flagged"][0].to_csv(tables / "table_walker_cas_stats_excl_flagged.csv", index=False)
    scores.reset_index().to_csv(tables / "walker_cas_scores.csv", index=False)

    cols = [f"{t} vs {c}" for t, c in COMPARISONS]
    plots.save(plots.effect_heatmap(stats, REGIONS, cols, "Cocaine SA: CAS effect size by region"),
               figures / "fig2_walker_cas_heatmap.png")
    drug = {g: ("Cocaine SA" if g.startswith("C") else "Saline SA") for g in GROUPS}
    plots.save(plots.group_boxplots(scores, REGIONS, GROUPS, drug, "Walker 2018: CAS per region and group"),
               figures / "fig3_walker_cas_boxplots.png")

    show = ["region", "comparison", "n_treat", "n_ctrl", "g", "g_ci_low", "g_ci_high", "welch_p", "welch_fdr",
            "perm_p", "perm_fdr", "months_equivalent"]
    print(stats[show].round(3).to_string(index=False))
    sens = results["excl_flagged"][0].set_index(["region", "comparison"])["g"]
    delta = (stats.set_index(["region", "comparison"])["g"] - sens).abs()
    print(f"\nSensitivity (QC-flagged removed): max |delta g| = {delta.max():.2f} at {delta.idxmax()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
