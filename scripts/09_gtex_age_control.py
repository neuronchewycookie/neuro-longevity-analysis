"""Human age positive control for the CAS on GTEx v10 basal ganglia (20-79 years).

Question: does the CAS rise with age in human caudate, putamen and nucleus accumbens? The CUD
cohort (18-52 years) showed no age trend, so this tells whether that reflects the narrow age
range or a failure of the mouse signature to transfer to human brain.

Per tissue:
  - Spearman CAS vs age (10-year bin midpoints), BH FDR over the 3 tissues
  - OLS  CAS ~ age + sex + RIN + ischemic time + Hardy scale
  - the same Spearman restricted to 20-49 years (the CUD cohort's range)
  - Hardy-scale sensitivity: the age effect within donors with similar death circumstances
Across tissues (donors gave up to 3 tissues):
  - mixed model  CAS_z ~ age + tissue + sex + RIN + ischemic time + Hardy + (1 | donor)
Decision rule, fixed before running: the human CAS "tracks age" if the Spearman rho is positive
with FDR < 0.05 in at least 2 of the 3 tissues.

Reads data/raw/GTEx_v10_brain/ and existing tables only; writes new gtex_* files.
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sstats

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders.gtex import load_gtex
from data_analysis.normalize import log_expression
from data_analysis.scoring import expressed_genes, signature_for, vision_score
from data_analysis.stats import bh_fdr, mixed_term

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gtex_age_control")

COVARIATES = "C(sex) + rin + ischemic_time + C(hardy)"
TISSUE_ORDER = ["caudate", "putamen", "nucleus_accumbens"]


def score_tissues(ds, signature, cas_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, genes = [], []
    for tissue in TISSUE_ORDER:
        sub = ds.subset(region=tissue)
        bg = expressed_genes(sub, cas_cfg["min_mean_count"], 1.0)
        keep = set(bg)
        up, down = [g for g in signature.up if g in keep], [g for g in signature.down if g in keep]
        frame = sub.meta.copy()
        frame["cas"] = vision_score(log_expression(sub), up, down, bg)
        frames.append(frame)
        genes.append({"region": tissue, "n_up": len(up), "n_down": len(down),
                      "n_signature_mapped": len(signature.up) + len(signature.down)})
    return pd.concat(frames), pd.DataFrame(genes)


HARDY_SUBSETS = {
    "all donors": None,
    "excluding ventilator deaths (Hardy 0)": ["1", "2", "3", "4", "missing"],
    "fast deaths only (Hardy 1-2)": ["1", "2"],
    "fast natural deaths only (Hardy 2)": ["2"],
}


def hardy_sensitivity(scores: pd.DataFrame) -> pd.DataFrame:
    """Age effect within death-circumstance subsets: CAS is higher after slow or ventilator deaths,
    and those deaths are more common in older donors, so the age effect is re-checked within
    donors who died in similar circumstances. Covariates drop Hardy, which is fixed or near-fixed here."""
    rows = []
    for label, keep in HARDY_SUBSETS.items():
        sub = scores if keep is None else scores[scores["hardy"].astype(str).isin(keep)]
        for tissue in TISSUE_ORDER:
            x = sub[sub["region"] == tissue]
            rho, p = sstats.spearmanr(x["age"], x["cas"])
            fit = smf.ols("cas ~ age + C(sex) + rin + ischemic_time",
                          data=x.dropna(subset=["rin", "ischemic_time"])).fit()
            rows.append({"subset": label, "region": tissue, "n": len(x), "rho": rho, "p": p,
                         "age_coef_per_year": fit.params["age"], "age_p": fit.pvalues["age"]})
    return pd.DataFrame(rows)


def figure(scores: pd.DataFrame, corr: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, tissue in zip(axes, TISSUE_ORDER):
        sub = scores[scores["region"] == tissue]
        jitter = (np.arange(len(sub)) % 7 - 3) * 0.6
        ax.scatter(sub["age"] + jitter, sub["cas"], s=10, color=plots.SERIES[0], alpha=0.45, linewidths=0)
        means = sub.groupby("age")["cas"].agg(["mean", "sem", "count"])
        ax.errorbar(means.index, means["mean"], yerr=1.96 * means["sem"], color=plots.TEXT, linewidth=2,
                    marker="o", markersize=5, capsize=3)
        ax.axvspan(18, 52, color=plots.SERIES[1], alpha=0.08, linewidth=0)
        r = corr[(corr["region"] == tissue) & (corr["subset"] == "all ages")].iloc[0]
        ax.set_title(f"{tissue.replace('_', ' ')}  (n={len(sub)})\nSpearman rho {r['rho']:.2f}, FDR {r['fdr']:.2g}",
                     fontsize=9, color=plots.TEXT)
        ax.set_xlabel("Age (10-year bin midpoint)", fontsize=8, color=plots.TEXT_MUTED)
        ax.set_xticks([25, 35, 45, 55, 65, 75])
        plots._style_axes(ax)
    axes[0].set_ylabel("CAS (human orthologs)", fontsize=9, color=plots.TEXT_MUTED)
    fig.suptitle("GTEx v10: does the CAS rise with age in human basal ganglia?  "
                 "(dots = donors, line = mean ± 95% CI, shaded = CUD cohort age range)", fontsize=10, color=plots.TEXT)
    fig.tight_layout()
    return fig


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    ds = load_gtex(cfg["paths"]["raw"])
    signature = signature_for("human_cud", pd.read_csv(tables / "cas_gene_mapping.csv"))
    scores, used = score_tissues(ds, signature, cfg["cas"])
    log.info("GTEx: %d genes x %d samples, %d donors", *ds.counts.shape, ds.meta["subject_id"].nunique())

    corr_rows = []
    for tissue in TISSUE_ORDER:
        sub = scores[scores["region"] == tissue]
        for subset, s in {"all ages": sub, "20-49 (CUD cohort range)": sub[sub["age"] < 50]}.items():
            rho, p = sstats.spearmanr(s["age"], s["cas"])
            corr_rows.append({"region": tissue, "subset": subset, "n": len(s), "rho": rho, "p": p})
    corr = pd.DataFrame(corr_rows)
    corr["fdr"] = corr.groupby("subset")["p"].transform(bh_fdr)

    model_rows = []
    for tissue in TISSUE_ORDER:
        sub = scores[scores["region"] == tissue].dropna(subset=["rin", "ischemic_time"])
        fit = smf.ols(f"cas ~ age + {COVARIATES}", data=sub).fit()
        ci = fit.conf_int().loc["age"]
        model_rows.append({"model": f"OLS {tissue}", "n": int(fit.nobs), "age_coef_per_year": fit.params["age"],
                           "ci_low": ci[0], "ci_high": ci[1], "p": fit.pvalues["age"]})
    pooled = scores.dropna(subset=["rin", "ischemic_time"]).copy()
    pooled["cas_z"] = pooled.groupby("region")["cas"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    fit, est, se, lo, hi, p = mixed_term(pooled, f"cas_z ~ age + C(region) + {COVARIATES}", "subject_id", "age")
    model_rows.append({"model": "Mixed (donor random intercept), all tissues, CAS_z", "n": int(fit.nobs),
                       "age_coef_per_year": est, "ci_low": lo, "ci_high": hi, "p": p,
                       "per_decade_sd": est * 10, "converged": bool(fit.converged)})
    models = pd.DataFrame(model_rows)

    hardy = hardy_sensitivity(scores)

    passed = int(((corr["subset"] == "all ages") & (corr["rho"] > 0) & (corr["fdr"] < 0.05)).sum())
    verdict = {"rule": "rho > 0 and FDR < 0.05 in >= 2 of 3 tissues", "tissues_passing": passed,
               "human_cas_tracks_age": passed >= 2}

    scores.reset_index().to_csv(tables / "gtex_cas_scores.csv", index=False)
    corr.to_csv(tables / "table_gtex_age_control.csv", index=False)
    models.to_csv(tables / "table_gtex_age_models.csv", index=False)
    used.to_csv(tables / "table_gtex_cas_genes_used.csv", index=False)
    hardy.to_csv(tables / "table_gtex_age_hardy_sensitivity.csv", index=False)
    (tables / "gtex_age_control_verdict.json").write_text(json.dumps(verdict, indent=1), encoding="utf-8")
    plots.save(figure(scores, corr), figures / "fig_gtex_age_control.png")

    print(used.to_string(index=False))
    print(corr.round(4).to_string(index=False))
    print(models.round(4).to_string(index=False))
    print(hardy.round(4).to_string(index=False))
    print(scores.groupby(["region", "age_bin"])["cas"].agg(["mean", "count"]).round(2).unstack(0).to_string())
    print(json.dumps(verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
