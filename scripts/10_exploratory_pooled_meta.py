"""EXPLORATORY (post hoc) analyses of the cocaine and heroin CAS results.

These analyses were planned after the primary results were seen. They do not change the
primary conclusions and every output is labelled exploratory.

A. Pooled mixed model per dataset x design: CAS z-scored within region (all samples of that
   region), CAS_z ~ treated + region + (1 | animal); LRT for treated x region; BH over 8 models;
   sensitivity run without QC-flagged samples.
B. Cocaine + heroin meta-analysis per design: inverse-variance fixed effect of the two pooled
   estimates from A (DerSimonian-Laird random effects for reference), Cochran Q / I^2, BH over 4.
   A per-region pooling of Hedges' g is reported as a descriptive table only.
C. Specificity: marker scores built from genes NOT in the CAS (microglia abundance, inflammatory
   activation). (a) drug effect on each marker score; (b) CAS model adjusted for both scores;
   (c) Hahn CAS-marker correlation per region, as a reference for over-adjustment.
D. Scoring robustness: Hedges' g re-computed with the spec's z-score method, VST instead of
   log-CPM, and expression filters of 1 and 50 counts.

Reads existing results only; writes new files named *exploratory*.
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
from scipy import stats as sstats

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.idmap import MyGeneCache, load_mouse_id_table, map_mouse_symbols
from data_analysis.loaders import load_processed
from data_analysis.normalize import log_cpm, log_expression
from data_analysis.scoring import expressed_genes, mouse_signature, signature_for, vision_score, zscore_score
from data_analysis.stats import bh_fdr, fixed_effect_meta, hedges_g, hedges_g_variance, lrt_pvalue, mixed_term

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("exploratory")

DATASETS = {
    "walker": {"drug": "cocaine", "animal": "animal_id",
               "comparisons": [("C24", "S24"), ("CS", "SS"), ("CC", "SS"), ("SC", "SS")]},
    "heroin": {"drug": "heroin", "animal": "mouse_id",
               "comparisons": [("H24", "S24"), ("HS", "SS"), ("HH", "SS"), ("SH", "SS")]},
}
DESIGNS = ["24 h withdrawal", "30 d + context", "30 d + drug (relapse)", "first acute exposure"]
WALKER_NAME = {"BLA": "BLA", "dStri": "Cpu", "NAc": "NAc", "PFC": "PFC", "VTA": "VTA", "vHPC": "Hipp"}

# Marker genes chosen to NOT overlap the 82 CAS genes (checked at run time).
MARKERS = {
    "microglia": ["P2ry12", "Tmem119", "Csf1r", "Sall1", "Siglech", "Olfml3", "Fcrls", "Gpr34"],
    "inflammation": ["Il1b", "Tnf", "Ccl2", "Ccl3", "Ccl4", "Cxcl10", "Nfkbia", "Socs3"],
}


def design_of(treat: str) -> str:
    if treat.endswith("24"):
        return DESIGNS[0]
    if treat[0] == "S":
        return DESIGNS[3]
    return DESIGNS[2] if treat[1] == treat[0] else DESIGNS[1]


def qc_flagged(cfg: dict, dataset: str) -> set[str]:
    spec = importlib.util.spec_from_file_location("walker_cas", Path(__file__).resolve().parent / "03_walker_cas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.qc_flagged(cfg, dataset)


def within_region_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_z"] = out.groupby("region")[col].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    return out


def load_scores(cfg: dict, name: str) -> pd.DataFrame:
    s = pd.read_csv(cfg["paths"]["tables"] / f"{name}_cas_scores.csv")
    s = s.rename(columns={DATASETS[name]["animal"]: "animal"})
    s["animal"] = s["animal"].astype(str)
    s["region"] = s["region"].astype(str)
    return s[["sample", "region", "animal", "group", "cas"]].assign(dataset=name)


# ----------------------------------------------------------------------------------------- C
def marker_scores(cfg: dict, ds, markers: dict[str, list[str]], mouse_ids: pd.DataFrame, name: str):
    """VISION scores (all genes 'up') of each marker set, per region; plus a gene usage table."""
    present = set(ds.counts.index)
    sigs = {k: mouse_signature(pd.DataFrame({"gene": v, "direction": "up"}), mouse_ids, name, present)
            for k, v in markers.items()}
    frames, genes = [], []
    for region in ds.regions:
        sub = ds.subset(region=region)
        bg = expressed_genes(sub, cfg["cas"]["min_mean_count"], cfg["cas"].get("min_mean_value", 1.0))
        keep = set(bg)
        expr = log_expression(sub)
        frame = pd.DataFrame(index=sub.meta.index)
        for key, sig in sigs.items():
            used = [g for g in sig.up if g in keep]
            for gene in markers[key]:
                gid = next((i for i, s in sig.symbol.items() if s == gene), None)
                genes.append({"dataset": name, "region": str(region), "marker_set": key, "gene": gene,
                              "used": gid in keep if gid else False,
                              "reason": "" if gid and gid in keep else ("not in data" if gid is None else "low expression")})
            frame[key] = vision_score(expr, used, [], bg) if used else np.nan
        frames.append(frame)
    return pd.concat(frames), pd.DataFrame(genes)


# ----------------------------------------------------------------------------------------- D
def vst(counts: pd.DataFrame) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference

    meta = pd.DataFrame({"c": ["a"] * counts.shape[1]}, index=counts.columns)
    dds = DeseqDataSet(counts=counts.T, metadata=meta, design="~1", inference=DefaultInference(n_cpus=4), quiet=True)
    dds.vst(use_design=False)
    return pd.DataFrame(dds.layers["vst_counts"].T, index=counts.index, columns=counts.columns)


def scoring_robustness(cfg: dict, name: str, ds, mapping: pd.DataFrame) -> pd.DataFrame:
    sig = signature_for(name, mapping)
    rows = []
    for region in ds.regions:
        sub = ds.subset(region=region)
        groups = sub.meta["group"].astype(str)
        variants = {}
        for label, min_count in (("current", cfg["cas"]["min_mean_count"]), ("filter >= 1", 1), ("filter >= 50", 50)):
            bg = expressed_genes(sub, min_count, 1.0)
            up, down = [g for g in sig.up if g in set(bg)], [g for g in sig.down if g in set(bg)]
            variants[label] = vision_score(log_cpm(sub.counts), up, down, bg)
            if label == "current":
                variants["z-score method (spec)"] = zscore_score(log_cpm(sub.counts), up, down)
                variants["VST instead of log-CPM"] = vision_score(vst(sub.counts.loc[bg]), up, down, bg)
        for treat, ctrl in DATASETS[name]["comparisons"]:
            for label, cas in variants.items():
                rows.append({"dataset": name, "region": str(region), "design": design_of(treat),
                             "comparison": f"{treat} vs {ctrl}", "variant": label,
                             "g": hedges_g(cas[groups == treat], cas[groups == ctrl])})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    gate = json.loads((tables / "hahn_positive_control_gate.json").read_text(encoding="utf-8"))
    if not gate["passed"] or gate["method"] != cfg["cas"]["method"]:
        log.error("Hahn positive control has not passed for method %s; run 02 first", cfg["cas"]["method"])
        return 1

    mapping = pd.read_csv(tables / "cas_gene_mapping.csv")
    cas_genes = set(mapping["gene"])
    overlap = sorted({g for v in MARKERS.values() for g in v} & cas_genes)
    if overlap:
        raise ValueError(f"marker genes overlap the CAS: {overlap}")
    cache = cfg["paths"]["processed"] / "idmap"
    mouse_ids = map_mouse_symbols([g for v in MARKERS.values() for g in v],
                                  load_mouse_id_table(cfg["data"]["heroin"]["deglists"], cache),
                                  MyGeneCache(cache / "mygene_cache.json"))

    # Per-sample table: CAS + marker scores, z-scored within region
    per_sample, marker_genes = [], []
    for name in DATASETS:
        scores = load_scores(cfg, name)
        ds = load_processed(name, cfg)
        mk, genes = marker_scores(cfg, ds, MARKERS, mouse_ids, name)
        marker_genes.append(genes)
        per_sample.append(within_region_z(scores.merge(mk, left_on="sample", right_index=True),
                                          ["cas", *MARKERS]))
    data = pd.concat(per_sample, ignore_index=True)
    pd.concat(marker_genes).to_csv(tables / "table_exploratory_marker_genes.csv", index=False)

    # A (+ C a/b) -------------------------------------------------------------------------
    flagged = {name: qc_flagged(cfg, name) for name in DATASETS}
    model_rows, spec_rows = [], []
    for run in ("all", "excl_flagged"):
        for name, info in DATASETS.items():
            d = data[data["dataset"] == name]
            if run == "excl_flagged":
                d = d[~d["sample"].isin(flagged[name])]
            for treat, ctrl in info["comparisons"]:
                sub = d[d["group"].isin([treat, ctrl])].assign(treated=lambda x: (x["group"] == treat).astype(int))
                base = dict(analysis="exploratory (post hoc)", run=run, dataset=name, drug=info["drug"],
                            design=design_of(treat), comparison=f"{treat} vs {ctrl}",
                            n_samples=len(sub), n_animals=sub["animal"].nunique())
                fit, est, se, lo, hi, p = mixed_term(sub, "cas_z ~ treated + C(region)", "animal", "treated")
                full = mixed_term(sub, "cas_z ~ treated * C(region)", "animal", "treated", reml=False)[0]
                reduced = mixed_term(sub, "cas_z ~ treated + C(region)", "animal", "treated", reml=False)[0]
                het_p, het_df = lrt_pvalue(full, reduced)
                model_rows.append({**base, "estimate": est, "se": se, "ci_low": lo, "ci_high": hi, "p": p,
                                   "region_heterogeneity_p": het_p, "lrt_df": het_df,
                                   "animal_var": float(fit.cov_re.iloc[0, 0]), "converged": bool(fit.converged)})
                if run != "all":
                    continue
                for outcome in MARKERS:
                    _, e, s_, l, h, pp = mixed_term(sub.dropna(subset=[f"{outcome}_z"]),
                                                    f"{outcome}_z ~ treated + C(region)", "animal", "treated")
                    spec_rows.append({**base, "model": f"(a) drug effect on {outcome} score",
                                      "estimate": e, "se": s_, "ci_low": l, "ci_high": h, "p": pp})
                adj = sub.dropna(subset=[f"{m}_z" for m in MARKERS])
                _, e, s_, l, h, pp = mixed_term(adj, "cas_z ~ treated + C(region) + microglia_z + inflammation_z",
                                                "animal", "treated")
                spec_rows.append({**base, "model": "(b) CAS adjusted for microglia + inflammation",
                                  "estimate": e, "se": s_, "ci_low": l, "ci_high": h, "p": pp})
                spec_rows.append({**base, "model": "(b) CAS unadjusted (= A)", "estimate": est, "se": se,
                                  "ci_low": lo, "ci_high": hi, "p": p})
    models = pd.DataFrame(model_rows)
    models["fdr"] = models.groupby("run")["p"].transform(bh_fdr)
    models.to_csv(tables / "table_exploratory_mixed_model.csv", index=False)
    specificity = pd.DataFrame(spec_rows)
    specificity["fdr"] = specificity.groupby("model")["p"].transform(bh_fdr)
    specificity.to_csv(tables / "table_exploratory_specificity.csv", index=False)

    # B -------------------------------------------------------------------------------------
    primary = models[models["run"] == "all"]
    meta_rows = []
    for design in DESIGNS:
        sub = primary[primary["design"] == design].sort_values("drug")
        m = fixed_effect_meta(sub["estimate"].to_numpy(), sub["se"].to_numpy())
        meta_rows.append({"analysis": "exploratory (post hoc)", "design": design,
                          "cocaine_estimate": sub.loc[sub["drug"] == "cocaine", "estimate"].iloc[0],
                          "heroin_estimate": sub.loc[sub["drug"] == "heroin", "estimate"].iloc[0], **m})
    meta = pd.DataFrame(meta_rows)
    meta["fixed_fdr"] = bh_fdr(meta["fixed_p"])
    meta.to_csv(tables / "table_exploratory_meta.csv", index=False)

    region_rows = []
    coc = pd.read_csv(tables / "table_walker_cas_stats.csv")
    her = pd.read_csv(tables / "table_heroin_cas_stats.csv").assign(region=lambda x: x["region"].map(WALKER_NAME))
    for frame in (coc, her):
        frame["design"] = frame["comparison"].str.split(" vs ").str[0].map(design_of)
    for (region, design), _ in coc.groupby(["region", "design"]):
        pair = pd.concat([coc[(coc.region == region) & (coc.design == design)],
                          her[(her.region == region) & (her.design == design)]])
        var = [hedges_g_variance(g, a, b) for g, a, b in zip(pair["g"], pair["n_treat"], pair["n_ctrl"])]
        m = fixed_effect_meta(pair["g"].to_numpy(), np.sqrt(var))
        region_rows.append({"analysis": "exploratory (post hoc), descriptive only", "region": region,
                            "design": design, "g_cocaine": pair["g"].iloc[0], "g_heroin": pair["g"].iloc[1],
                            "g_pooled_fixed": m["fixed_estimate"], "ci_low": m["fixed_ci_low"],
                            "ci_high": m["fixed_ci_high"], "i2": m["i2"]})
    pd.DataFrame(region_rows).to_csv(tables / "table_exploratory_meta_by_region.csv", index=False)

    # C (c): Hahn CAS vs marker scores, per region ------------------------------------------
    hahn = load_processed("hahn", cfg)
    hahn_cas = pd.read_csv(tables / "hahn_cas_scores.csv").query("method == @cfg['cas']['method']")
    hmk, _ = marker_scores(cfg, hahn, MARKERS, mouse_ids, "hahn")
    h = hahn_cas.merge(hmk, left_on="sample", right_index=True)
    corr_rows = []
    for region, sub in h.groupby("region"):
        row = {"analysis": "exploratory (post hoc)", "region": region, "n": len(sub)}
        for m in MARKERS:
            ok = sub[m].notna()
            row[f"rho_cas_{m}"], row[f"p_cas_{m}"] = sstats.spearmanr(sub.loc[ok, "cas"], sub.loc[ok, m]) if ok.sum() > 3 else (np.nan, np.nan)
            row[f"rho_age_{m}"], _ = sstats.spearmanr(sub.loc[ok, "age_months"], sub.loc[ok, m]) if ok.sum() > 3 else (np.nan, np.nan)
        corr_rows.append(row)
    pd.DataFrame(corr_rows).to_csv(tables / "table_exploratory_hahn_marker_corr.csv", index=False)

    # D -------------------------------------------------------------------------------------
    robust = pd.concat([scoring_robustness(cfg, name, load_processed(name, cfg), mapping) for name in DATASETS],
                       ignore_index=True)
    cur = robust[robust["variant"] == "current"].set_index(["dataset", "region", "comparison"])["g"]
    summary = []
    for variant, sub in robust[robust["variant"] != "current"].groupby("variant"):
        alt = sub.set_index(["dataset", "region", "comparison"])["g"]
        summary.append({"variant": variant, "r_with_current": cur.corr(alt),
                        "same_sign": float((np.sign(alt) == np.sign(cur)).mean()),
                        "max_abs_diff": float((alt - cur).abs().max()),
                        "all_24h_positive": bool((alt[alt.index.get_level_values(2).str.contains("24")] > 0).mean() >= 11 / 12)})
    robust.assign(analysis="exploratory (post hoc)").to_csv(tables / "table_exploratory_scoring_robustness.csv", index=False)
    pd.DataFrame(summary).to_csv(tables / "table_exploratory_scoring_robustness_summary.csv", index=False)

    # Figures --------------------------------------------------------------------------------
    plots.save(plots.exploratory_forest(primary, meta, DESIGNS), figures / "fig_exploratory_forest.png")
    plots.save(plots.exploratory_specificity(specificity, DESIGNS), figures / "fig_exploratory_specificity.png")

    pd.set_option("display.width", 200)
    print("A:\n", models[["run", "drug", "design", "n_animals", "estimate", "ci_low", "ci_high", "p", "fdr",
                          "region_heterogeneity_p", "converged"]].round(3).to_string(index=False))
    print("B:\n", meta[["design", "cocaine_estimate", "heroin_estimate", "fixed_estimate", "fixed_ci_low",
                        "fixed_ci_high", "fixed_p", "fixed_fdr", "i2"]].round(3).to_string(index=False))
    print("C:\n", specificity[["drug", "design", "model", "estimate", "ci_low", "ci_high", "p"]]
          .round(3).to_string(index=False))
    print("C(c):\n", pd.DataFrame(corr_rows).round(2).to_string(index=False))
    print("D:\n", pd.DataFrame(summary).round(3).to_string(index=False))
    print(pd.concat(marker_genes).groupby(["dataset", "marker_set"])["used"].mean().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
