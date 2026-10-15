"""Step 0 - load all datasets, cache them as parquet, and run sample-level QC.

Outputs:
  data/processed/{dataset}_{counts,meta}.parquet, {dataset}_info.json
  results/tables/sample_counts.csv       samples per dataset x region x group (x sex)
  results/tables/library_sizes.csv       per-sample library size (count datasets only)
  results/tables/qc_pca.csv              per-region PC1/PC2 with outlier flags
  results/figures/qc_library_size.png
  results/figures/qc_pca_{dataset}.png
Outliers are flagged only; no sample is removed.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from data_analysis import plots
from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.loaders import DATASETS, load_raw
from data_analysis.qc import library_sizes, pca_by_region

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("load_all")

# Per dataset: columns that define a design cell (besides region), shown in QC tables.
GROUP_COLUMNS = {
    "walker": ["group"],
    "hahn": ["sex", "age_months"],
    "heroin": ["group"],
    "rat_reward": ["group", "sex"],
    "human_cud": ["diagnosis"],
}


def tidy_sample_counts(ds, group_cols: list[str]) -> pd.DataFrame:
    """Rows: dataset, region, group, sex, n (group joins non-sex design columns)."""
    table = ds.sample_counts(["region", *group_cols])
    design = [c for c in group_cols if c != "sex"]
    out = pd.DataFrame({
        "dataset": ds.name,
        "region": table["region"].astype(str),
        "group": table[design].astype(str).agg("_".join, axis=1),
        "sex": table["sex"] if "sex" in table else "Male",
        "n": table["n"],
    })
    if ds.name == "hahn":
        out["group"] = out["group"] + "mo"
    return out


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)
    tables, figures = cfg["paths"]["tables"], cfg["paths"]["figures"]
    qc_cfg = cfg["qc"]

    counts, libs, pcas = [], [], []
    for name in DATASETS:
        log.info("loading %s", name)
        ds = load_raw(name, cfg)
        if ds is None:
            continue
        ds.to_parquet(cfg["paths"]["processed"])
        log.info("%s: %d genes x %d samples cached", name, *ds.counts.shape)

        group_cols = GROUP_COLUMNS[name]
        counts.append(tidy_sample_counts(ds, group_cols))
        if ds.is_count:
            libs.append(library_sizes(ds, extra=group_cols, z_threshold=qc_cfg["outlier_robust_z"]))
        pca = pca_by_region(ds, n_top=qc_cfg["pca_top_genes"], z_threshold=qc_cfg["outlier_robust_z"],
                            extra=group_cols)
        pcas.append(pca)
        plots.save(plots.pca_outlier_grid(pca, f"{name}: per-region PCA (top {qc_cfg['pca_top_genes']} genes)"),
                   figures / f"qc_pca_{name}.png")

    sample_counts = pd.concat(counts, ignore_index=True)
    lib = pd.concat(libs, ignore_index=True)
    pca = pd.concat(pcas, ignore_index=True)
    sample_counts.to_csv(tables / "sample_counts.csv", index=False)
    lib.to_csv(tables / "library_sizes.csv", index=False)
    pca.to_csv(tables / "qc_pca.csv", index=False)
    plots.save(plots.library_size_panels(lib), figures / "qc_library_size.png")

    print(sample_counts.groupby("dataset")["n"].sum().to_string())
    print(lib.groupby("dataset")["lib_size"].describe()[["min", "50%", "max"]].div(1e6).round(1).to_string())
    flagged = pca[pca["outlier"]]
    if len(flagged):
        log.warning("%d samples flagged as PCA outliers (kept):\n%s", len(flagged),
                    flagged[["dataset", "region", "sample", "knn_z"]].round(1).to_string(index=False))
    low = lib[lib["low_library"]]
    if len(low):
        log.warning("%d samples flagged with low library size (kept):\n%s", len(low),
                    low[["dataset", "region", "sample", "lib_size", "log_lib_z"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
