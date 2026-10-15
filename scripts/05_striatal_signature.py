"""Striatum-specific aging signature applied to the cocaine and heroin dorsal striatum.

Signatures (from Hahn 2023 supplement-11, sheet "Age-correlated genes"):
  - CAS                 the 82 common aging genes (reference)
  - striatal_all        caudate-putamen age-correlated genes, adj. P < 0.05, direction = sign of rho
  - striatal_specific   striatal_all minus the CAS genes
Each signature is cross-checked on Hahn's own caudate-putamen samples (our Spearman rho vs the
published rho), scored with the configured CAS method, and compared in Walker Cpu and heroin
dStri with the same statistics as the CAS analysis. Score differences are converted to months
with each signature's own slope in Hahn caudate putamen.

Outputs:
  results/tables/table_striatal_signature_genes.csv
  results/tables/table_striatal_crossval.csv
  results/tables/table_striatal_signature_stats.csv
  results/figures/fig_striatal_signature_heatmap.png
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy import stats as sstats

from data_analysis import plots
from data_analysis.comparisons import compare_cas
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.idmap import MyGeneCache, load_mouse_id_table, map_mouse_symbols
from data_analysis.loaders import load_processed
from data_analysis.normalize import log_cpm
from data_analysis.scoring import compute_cas, mouse_signature
from data_analysis.signatures import load_age_correlations, load_cas_signature, load_striatal_signature
from data_analysis.stats import bh_fdr, group_slopes

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("striatal_signature")

STRIATUM = {"walker": ("Cpu", [("C24", "S24"), ("CS", "SS"), ("CC", "SS"), ("SC", "SS")]),
            "heroin": ("dStri", [("H24", "S24"), ("HS", "SS"), ("HH", "SS"), ("SH", "SS")])}
DRUG = {"walker": "cocaine", "heroin": "heroin"}
DESIGN = {"24": "24 h withdrawal", "S vs": "30 d + context", "relapse": "30 d + drug (relapse)", "first": "first acute exposure"}


def design_of(comparison: str) -> str:
    treat = comparison.split(" vs ")[0]
    if treat.endswith("24"):
        return "24 h withdrawal"
    if treat[0] == "S":
        return "first acute exposure"
    return "30 d + drug (relapse)" if treat[1] == treat[0] else "30 d + context"


def crossvalidate(hahn, signature: pd.DataFrame) -> pd.DataFrame:
    """Our Spearman rho (log-CPM vs age, Hahn cp samples) next to the published rho."""
    cp = hahn.subset(region="cp")
    expr = log_cpm(cp.counts)
    age = cp.meta["age_months"].to_numpy()
    rows = []
    for gene in signature["gene"]:
        if gene not in expr.index:
            rows.append({"gene": gene, "rho_ours": np.nan, "p_ours": np.nan})
            continue
        rho, p = sstats.spearmanr(expr.loc[gene].to_numpy(), age)
        rows.append({"gene": gene, "rho_ours": rho, "p_ours": p})
    return signature.merge(pd.DataFrame(rows), on="gene")


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    supplement = cfg["data"]["hahn"]["supplement"]

    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    cas = load_cas_signature(supplement, cfg["cas"]["expected_n_up"], cfg["cas"]["expected_n_down"])
    striatal = load_striatal_signature(supplement)
    cas_genes = set(cas["gene"])
    striatal["in_cas"] = striatal["gene"].isin(cas_genes)
    signatures = {
        "CAS": cas[["gene", "direction"]],
        "striatal_all": striatal[["gene", "direction"]],
        "striatal_specific": striatal.loc[~striatal["in_cas"], ["gene", "direction"]],
    }

    hahn = load_processed("hahn", cfg)
    cv = crossvalidate(hahn, striatal)
    cache = cfg["paths"]["processed"] / "idmap"
    mouse_ids = map_mouse_symbols(pd.concat([cas["gene"], striatal["gene"]]).drop_duplicates(),
                                  load_mouse_id_table(cfg["data"]["heroin"]["deglists"], cache),
                                  MyGeneCache(cache / "mygene_cache.json"))
    cv.merge(mouse_ids, on="gene", how="left").to_csv(tables / "table_striatal_signature_genes.csv", index=False)

    tested = cv.dropna(subset=["rho_ours"])
    crossval = []
    for name, sig in signatures.items():
        if name == "CAS":
            continue
        sub = tested[tested["gene"].isin(sig["gene"])]
        crossval.append({
            "signature": name, "n_genes": len(sig), "n_in_hahn_counts": len(sub),
            "sign_agreement": float((np.sign(sub["rho"]) == np.sign(sub["rho_ours"])).mean()),
            "rho_correlation": float(sstats.pearsonr(sub["rho"], sub["rho_ours"])[0]),
            "frac_p_lt_0.05_ours": float((sub["p_ours"] < 0.05).mean()),
        })
    crossval = pd.DataFrame(crossval)
    crossval.to_csv(tables / "table_striatal_crossval.csv", index=False)

    datasets = {name: load_processed(name, cfg) for name in ("hahn", "walker", "heroin")}
    rows = []
    for sig_name, sig_df in signatures.items():
        sigs = {name: mouse_signature(sig_df, mouse_ids, name, set(ds.counts.index)) for name, ds in datasets.items()}
        scores, _ = compute_cas(hahn.subset(region="cp"), sigs["hahn"], cfg["cas"])
        slope = group_slopes(scores, "cas", "age_months", "region").iloc[0]
        for name, (region, comparisons) in STRIATUM.items():
            ds = datasets[name].subset(region=region)
            st, _ = compare_cas(ds, sigs[name], comparisons, cfg["cas"], np.random.default_rng(cfg["seed"]))
            st = st.assign(signature=sig_name, drug=DRUG[name], hahn_cp_slope=slope["slope"],
                           hahn_cp_slope_p=slope["slope_p"], n_sig_up=len(sigs[name].up), n_sig_down=len(sigs[name].down))
            st["months_equivalent"] = st["diff"] / st["hahn_cp_slope"]
            rows.append(st)
    result = pd.concat(rows, ignore_index=True)
    result["design"] = result["comparison"].map(design_of)
    result["welch_fdr_step"] = bh_fdr(result["welch_p"])
    result["perm_fdr_step"] = bh_fdr(result["perm_p"])
    result.to_csv(tables / "table_striatal_signature_stats.csv", index=False)

    heat = result.assign(region=result["signature"] + " | " + result["drug"],
                         comparison=result["design"], welch_fdr=result["welch_fdr_step"])
    row_order = [f"{s} | {d}" for s in signatures for d in ("cocaine", "heroin")]
    col_order = ["24 h withdrawal", "30 d + context", "30 d + drug (relapse)", "first acute exposure"]
    plots.save(plots.effect_heatmap(heat, row_order, col_order,
                                    "Dorsal striatum: CAS vs striatal aging signatures (Cpu / dStri)"),
               figures / "fig_striatal_signature_heatmap.png")

    print(crossval.round(3).to_string(index=False))
    print(result[["signature", "drug", "design", "n_sig_up", "n_sig_down", "g", "g_ci_low", "g_ci_high",
                  "welch_fdr_step", "perm_p", "months_equivalent"]].round(3).to_string(index=False))
    print(result.drop_duplicates("signature")[["signature", "hahn_cp_slope", "hahn_cp_slope_p"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
