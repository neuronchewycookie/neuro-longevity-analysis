"""Step 1 - positive control: reproduce Hahn et al. 2023 CAS aging velocities.

For each of 15 regions, CAS is scored per sample and regressed on age with the paper's model
(cas ~ age + region + age:region). The run passes only if every region's slope is positive
(FDR < threshold) and the fastest regions match the paper. Later steps must not be trusted
when this gate fails.

Outputs:
  results/tables/hahn_cas_scores.csv            per-sample CAS (all methods)
  results/tables/table_hahn_cas_slopes.csv      per-region slope ("CAS change per month"), rho, ranks
  results/tables/hahn_cas_genes_used.csv        CAS genes used / dropped per region
  results/tables/hahn_positive_control_gate.json
  results/figures/fig1_hahn_cas_vs_age.png
  results/figures/fig1b_hahn_cas_slopes.png
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import load_processed
from data_analysis.loaders.hahn import REGIONS
from data_analysis.scoring import METHODS, compute_cas, signature_for
from data_analysis.stats import group_slopes, group_spearman, slope_difference

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("hahn_positive_control")


def slope_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Slopes from the paper's joint model (all samples) plus male-only slopes and Spearman rho."""
    all_sex = group_slopes(scores, "cas", "age_months", "region")
    male = group_slopes(scores[scores["sex"] == "Male"], "cas", "age_months", "region")
    male = male[["region", "slope", "slope_ci_low", "slope_ci_high"]].add_prefix("male_").rename(
        columns={"male_region": "region"})
    rho = group_spearman(scores, "cas", "age_months", "region")
    out = all_sex.merge(male, on="region").merge(rho, on="region")
    out["region_name"] = out["region"].map(REGIONS)
    out["rank"] = out["slope"].rank(ascending=False).astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def evaluate_gate(scores: pd.DataFrame, slopes: pd.DataFrame, pc: dict) -> dict:
    """Paper-consistency checks. The fastest-region check is a tie test, not a rank cut-off:
    ranks flip on differences far smaller than the slope uncertainty (decided 2026-10-15)."""
    rank = dict(zip(slopes["region"], slopes["rank"]))
    top = slopes.loc[slopes["rank"] == 1, "region"].iloc[0]
    target = pc["fastest_region"]
    diff, p = (0.0, 1.0) if top == target else slope_difference(scores, "cas", "age_months", "region", top, target)
    checks = {
        "all_slopes_positive_fdr": bool(((slopes["slope"] > 0) & (slopes["slope_fdr"] < pc["fdr"])).all()),
        "all_spearman_positive_fdr": bool(((slopes["spearman_rho"] > 0) & (slopes["spearman_fdr"] < pc["fdr"])).all()),
        f"{target}_tied_with_fastest": p >= pc["tie_alpha"],
        **{f"{r}_rank_le_{pc['fast_regions_max_rank']}": rank[r] <= pc["fast_regions_max_rank"]
           for r in pc["fast_regions"]},
    }
    cortex = slopes[slopes["region"].isin(["cor", "vis", "ent"])]["slope"].mean()
    fastest = slopes["slope"].max()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "ranks": rank,
        "info": {"fastest_region": top, f"fastest_minus_{target}": round(diff, 4), "tie_p": round(p, 3),
                 "cortex_mean_slope_over_fastest": round(cortex / fastest, 3),
                 "paper_reports_cortex_over_cc": "about 1/3"},
    }


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    primary = cfg["cas"]["method"]

    hahn = load_processed("hahn", cfg)
    signature = signature_for("hahn", pd.read_csv(tables / "cas_gene_mapping.csv"))

    all_scores, gate = [], None
    for method in METHODS:
        scores, genes = compute_cas(hahn, signature, cfg["cas"], method=method)
        scores["region"] = scores["region"].astype(str)
        all_scores.append(scores.reset_index())
        slopes = slope_table(scores)
        result = evaluate_gate(scores, slopes, cfg["positive_control"])
        log.info("%s: gate %s | ranks cc=%d cp=%d cer=%d | cortex/fastest=%.2f", method,
                 "PASSED" if result["passed"] else "FAILED", result["ranks"]["cc"], result["ranks"]["cp"],
                 result["ranks"]["cer"], result["info"]["cortex_mean_slope_over_fastest"])
        if method == primary:
            gate = result
            slopes.to_csv(tables / "table_hahn_cas_slopes.csv", index=False)
            genes.to_csv(tables / "hahn_cas_genes_used.csv", index=False)
            plots.save(plots.cas_vs_age_grid(scores, slopes, REGIONS,
                                             f"Hahn 2023: CAS vs age per region ({method})"),
                       figures / "fig1_hahn_cas_vs_age.png")
            plots.save(plots.slope_ranking(slopes, REGIONS, f"CAS aging velocity ({method})"),
                       figures / "fig1b_hahn_cas_slopes.png")
            print(slopes[["rank", "region", "region_name", "n", "slope", "slope_ci_low", "slope_ci_high",
                          "slope_fdr", "male_slope", "spearman_rho"]].round(4).to_string(index=False))
        else:
            slopes.to_csv(tables / f"table_hahn_cas_slopes_{method}.csv", index=False)

    pd.concat(all_scores).to_csv(tables / "hahn_cas_scores.csv", index=False)
    gate["method"] = primary
    (tables / "hahn_positive_control_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    print(json.dumps(gate["checks"], indent=1))
    if not gate["passed"]:
        log.error("positive control FAILED for method %s; do not run downstream steps", primary)
        return 1
    log.info("positive control PASSED (%s)", primary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
