"""Map the 82 CAS genes to every dataset's gene IDs and report coverage.

Outputs:
  results/tables/cas_gene_mapping.csv   one row per CAS gene, IDs + presence per dataset
  results/tables/cas_gene_coverage.csv  n_up / n_down usable per dataset
"""

from __future__ import annotations

import logging

import pandas as pd

from data_analysis.config import ensure_output_dirs, load_config
from data_analysis.idmap import MyGeneCache, build_cas_mapping, load_mouse_id_table, mouse_id_candidates
from data_analysis.signatures import load_cas_signature

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cas_gene_mapping")


def dataset_gene_ids(cfg) -> dict[str, set[str]]:
    """Gene identifiers present in each dataset (first column / gene_name only)."""
    data = cfg["data"]
    rat_files = [data["rat"]["normalized"][k] for k in ("male_sucrose", "female_sucrose")]
    human_file = next(data["human"]["raw_dir"].glob(data["human"]["counts_glob"]))
    return {
        "walker": set(pd.read_csv(data["walker"]["counts"], sep="\t", usecols=[0]).iloc[:, 0]),
        "hahn": set(pd.read_csv(data["hahn"]["counts"], sep="\t", usecols=[0]).iloc[:, 0]),
        "heroin": set(pd.read_csv(data["heroin"]["counts"], usecols=[0]).iloc[:, 0]),
        # Male_Morphine has a shifted gene_name column, so take symbols from the sucrose files.
        "rat": set().union(*(pd.read_csv(f, sep="\t", usecols=["gene_name"])["gene_name"] for f in rat_files)),
        "human": set(pd.read_csv(human_file, sep="\t", comment="#", usecols=[0]).iloc[:, 0]),
    }


def add_presence(mapping: pd.DataFrame, ids: dict[str, set[str]]) -> pd.DataFrame:
    out = mapping.copy()
    candidates = out.apply(mouse_id_candidates, axis=1)
    out["in_hahn"] = out["gene"].isin(ids["hahn"])
    for ds in ("walker", "heroin"):
        out[f"{ds}_id"] = candidates.map(lambda c, s=ids[ds]: next((i for i in c if i in s), None))
        out[f"in_{ds}"] = out[f"{ds}_id"].notna()
    usable = ["one_to_one", "symbol_match"]
    out["in_rat"] = out["rat_symbol"].isin(ids["rat"]) & out["rat_status"].isin(usable)
    out["in_human"] = out["human_ensembl"].fillna("").map(
        lambda s: any(e in ids["human"] for e in s.split(";") if e)
    ) & out["human_status"].isin(usable)
    return out


def coverage(mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in ("hahn", "walker", "heroin", "rat", "human"):
        used = mapping[mapping[f"in_{ds}"]]
        rows.append({
            "dataset": ds,
            "n_up": int((used["direction"] == "up").sum()),
            "n_down": int((used["direction"] == "down").sum()),
            "n_total": len(used),
            "missing": ";".join(mapping.loc[~mapping[f"in_{ds}"], "gene"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ensure_output_dirs(cfg)
    cache_dir = cfg["paths"]["processed"] / "idmap"

    signature = load_cas_signature(
        cfg["data"]["hahn"]["supplement"], cfg["cas"]["expected_n_up"], cfg["cas"]["expected_n_down"]
    )
    id_table = load_mouse_id_table(cfg["data"]["heroin"]["deglists"], cache_dir)
    mg = MyGeneCache(cache_dir / "mygene_cache.json")

    mapping = add_presence(build_cas_mapping(signature, id_table, mg), dataset_gene_ids(cfg))
    cov = coverage(mapping)

    mapping.to_csv(cfg["paths"]["tables"] / "cas_gene_mapping.csv", index=False)
    cov.to_csv(cfg["paths"]["tables"] / "cas_gene_coverage.csv", index=False)

    print(mapping["id_source"].value_counts().to_string())
    print(mapping[["rat_status", "human_status"]].apply(pd.Series.value_counts).fillna(0).astype(int).to_string())
    print(cov.to_string(index=False))
    for ds, row in cov.set_index("dataset").iterrows():
        if row["missing"]:
            log.warning("%s: %d CAS genes not usable: %s", ds, 82 - row["n_total"], row["missing"])


if __name__ == "__main__":
    main()
