"""Natural reward vs drug, by sex: CAS in rat NAc shell (GSE174737).

The four files (sex x reinforcer) were normalized separately and form separate PCA clusters,
so every comparison stays within one file:
  morphine files: Morphine Day 1 / Day 30 vs Saline
  sucrose files:  Sucrose Day 1 / Day 30 vs Cue-only
Statistics per comparison match the mouse CAS analysis (Hedges' g, bootstrap CI, Welch /
Mann-Whitney, matched random-signature permutation); BH FDR across the 8 comparisons.

For the 2-way ANOVAs, each animal's CAS is expressed as change from its own file's control
group in control-SD units (delta_z), which removes file-level offsets. On treated animals:
  delta_z ~ reinforcer * time + sex      (reinforcer x time)
  delta_z ~ sex * reinforcer + time      (sex x reinforcer)
Months-of-aging conversion is not done: rat FPKM scores are not on the Hahn mouse scale.

Outputs:
  results/tables/table_rat_cas_stats.csv
  results/tables/table_rat_anova.csv
  results/tables/rat_cas_scores.csv
  results/figures/fig6_rat_reward_sex.png
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from data_analysis import plots
from data_analysis.comparisons import compare_cas
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import load_processed
from data_analysis.scoring import signature_for
from data_analysis.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rat_reward_sex")

CONTROL = {"Morphine": "Saline", "Sucrose": "Cue"}


def file_comparisons(reinforcer: str) -> list[tuple[str, str]]:
    return [(f"{reinforcer}_D1", CONTROL[reinforcer]), (f"{reinforcer}_D30", CONTROL[reinforcer])]


def control_standardized(scores: pd.DataFrame) -> pd.DataFrame:
    """delta_z = (CAS - mean of same-file control) / SD of same-file control, for treated animals."""
    frames = []
    for _, sub in scores.groupby("file"):
        ctrl = sub[sub["group"].astype(str).isin(CONTROL.values())]["cas"]
        treated = sub[~sub["group"].astype(str).isin(CONTROL.values())].copy()
        treated["delta_z"] = (treated["cas"] - ctrl.mean()) / ctrl.std(ddof=1)
        frames.append(treated)
    out = pd.concat(frames)
    out["time"] = out["time_point"].map({"Day 1": "D1", "Day 30": "D30"})
    out["label"] = out["reinforcer"] + " " + out["time"]
    return out


def anova_table(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, formula in {"reinforcer x time": "delta_z ~ C(reinforcer) * C(time) + C(sex)",
                           "sex x reinforcer": "delta_z ~ C(sex) * C(reinforcer) + C(time)"}.items():
        fit = smf.ols(formula, data=delta).fit()
        table = sm.stats.anova_lm(fit, typ=2).reset_index().rename(columns={"index": "term", "PR(>F)": "p"})
        rows.append(table.assign(model=model, n=int(fit.nobs)))
    return pd.concat(rows, ignore_index=True)[["model", "term", "df", "sum_sq", "F", "p", "n"]]


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]

    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    rat = load_processed("rat_reward", cfg)
    signature = signature_for("rat_reward", pd.read_csv(tables / "cas_gene_mapping.csv"))

    stats_rows, score_rows = [], []
    for file in sorted(rat.meta["file"].unique()):
        sub = rat.subset(file=file)
        reinforcer = sub.meta["reinforcer"].iloc[0]
        st, sc = compare_cas(sub, signature, file_comparisons(reinforcer), cfg["cas"],
                             np.random.default_rng(cfg["seed"]))
        stats_rows.append(st.assign(file=file, sex=sub.meta["sex"].iloc[0], reinforcer=reinforcer))
        score_rows.append(sc)
    stats = pd.concat(stats_rows, ignore_index=True)
    for col in ("welch", "mwu", "perm"):
        stats[f"{col}_fdr"] = bh_fdr(stats[f"{col}_p"])
    stats["label"] = stats["treat"].str.replace("_", " ")
    scores = pd.concat(score_rows)
    stats.to_csv(tables / "table_rat_cas_stats.csv", index=False)
    scores.reset_index().to_csv(tables / "rat_cas_scores.csv", index=False)

    delta = control_standardized(scores)
    anova = anova_table(delta)
    anova.to_csv(tables / "table_rat_anova.csv", index=False)

    plots.save(plots.rat_reward_panels(stats, delta, "Rat NAc shell: morphine vs sucrose, by sex (CAS)"),
               figures / "fig6_rat_reward_sex.png")

    print(stats[["sex", "comparison", "n_treat", "n_ctrl", "n_up", "n_down", "g", "g_ci_low", "g_ci_high",
                 "welch_fdr", "perm_fdr"]].round(3).to_string(index=False))
    print(anova.round(4).to_string(index=False))
    print(delta.groupby(["reinforcer", "time", "sex"])["delta_z"].agg(["mean", "count"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
