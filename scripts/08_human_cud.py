"""Human validation: CAS in cocaine use disorder (CUD) vs control postmortem brain (GSE214267).

Per region (caudate, NAc):
  - CUD vs control with the CAS statistics (Hedges' g, bootstrap CI, Welch / Mann-Whitney,
    matched random-signature permutation), BH FDR across regions
  - covariate model  CAS ~ diagnosis + age + PMI + race   (all donors are male; GEO has no RIN)
  - age positive control: Spearman CAS vs age (all donors and controls only)
Across regions, 44 of 45 donors gave both regions (donor IDs inferred from covariates), so a
mixed model with a donor random intercept is also fitted:
  CAS ~ diagnosis * region + age + PMI + race + (1 | donor)
Sensitivity: in NAc, the 7 low-library samples (plate column 10) are all CUD, so diagnosis is
confounded with plate batch; every model is refitted without QC-flagged samples.

Outputs:
  results/tables/table_human_cud_stats.csv
  results/tables/table_human_cud_models.csv
  results/tables/table_human_age_control.csv
  results/tables/human_cas_scores.csv
  results/figures/fig7_human_cud.png
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
import statsmodels.formula.api as smf
from scipy import stats as sstats

from data_analysis import plots
from data_analysis.comparisons import compare_cas
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import load_processed
from data_analysis.scoring import signature_for
from data_analysis.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("human_cud")

COVARIATES = "age + pmi + C(race)"
DX = 'C(diagnosis, Treatment("Control"))'  # control is the reference level
DX_CUD = f"{DX}[T.CUD]"


def _qc_flagged(cfg: dict) -> set[str]:
    spec = importlib.util.spec_from_file_location("walker_cas", Path(__file__).resolve().parent / "03_walker_cas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.qc_flagged(cfg, "human_cud")


def diagnosis_term(fit, name: str = DX_CUD) -> dict:
    ci = fit.conf_int().loc[name]
    return {"coef_cud": fit.params[name], "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues[name]}


def fit_models(scores: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for region, sub in scores.groupby("region"):
        fit = smf.ols(f"cas ~ {DX} + {COVARIATES}", data=sub).fit()
        age = fit.params["age"], fit.pvalues["age"]
        rows.append({"run": label, "model": f"OLS {region}", "n": int(fit.nobs), **diagnosis_term(fit),
                     "age_coef": age[0], "age_p": age[1]})
    mixed = smf.mixedlm(f"cas ~ {DX} * C(region) + {COVARIATES}", data=scores,
                        groups=scores["subject_id"]).fit(reml=True)
    rows.append({"run": label, "model": "Mixed (donor random intercept), CUD effect in caudate", "n": int(mixed.nobs),
                 **diagnosis_term(mixed), "age_coef": mixed.params["age"], "age_p": mixed.pvalues["age"],
                 "interaction_p": mixed.pvalues[f"{DX_CUD}:C(region)[T.NAc]"],
                 "donor_var": float(mixed.cov_re.iloc[0, 0])})
    return rows


def age_control(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, sub in scores.groupby("region"):
        for subset, s in {"all donors": sub, "controls only": sub[sub["diagnosis"] == "Control"]}.items():
            rho, p = sstats.spearmanr(s["age"], s["cas"])
            rows.append({"region": region, "subset": subset, "n": len(s), "spearman_rho": rho, "p": p,
                         "age_range": f"{s['age'].min():.0f}-{s['age'].max():.0f}"})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]

    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    human = load_processed("human_cud", cfg)
    signature = signature_for("human_cud", pd.read_csv(tables / "cas_gene_mapping.csv"))
    flagged = _qc_flagged(cfg)
    log.info("QC-flagged human samples (sensitivity run excludes): %s", sorted(flagged))

    stats_rows, model_rows, all_scores = [], [], None
    for label, exclude in {"all": None, "excl_flagged": flagged}.items():
        st, scores = compare_cas(human, signature, [("CUD", "Control")], cfg["cas"],
                                 np.random.default_rng(cfg["seed"]), group_col="diagnosis", exclude=exclude)
        stats_rows.append(st.assign(run=label))
        scores = scores.assign(diagnosis=scores["diagnosis"].astype(str), region=scores["region"].astype(str))
        model_rows += fit_models(scores, label)
        if label == "all":
            all_scores = scores

    stats = pd.concat(stats_rows, ignore_index=True)
    models = pd.DataFrame(model_rows)
    models["p_fdr_within_run"] = models.groupby("run")["p"].transform(bh_fdr)
    ages = age_control(all_scores)
    stats.to_csv(tables / "table_human_cud_stats.csv", index=False)
    models.to_csv(tables / "table_human_cud_models.csv", index=False)
    ages.to_csv(tables / "table_human_age_control.csv", index=False)
    all_scores.reset_index().to_csv(tables / "human_cas_scores.csv", index=False)
    plots.save(plots.human_cud_panels(all_scores, "Human postmortem: CAS in cocaine use disorder (GSE214267)"),
               figures / "fig7_human_cud.png")

    print(stats[["run", "region", "n_treat", "n_ctrl", "n_up", "n_down", "g", "g_ci_low", "g_ci_high",
                 "welch_p", "welch_fdr", "perm_p"]].round(3).to_string(index=False))
    print(models.round(4).to_string(index=False))
    print(ages.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
