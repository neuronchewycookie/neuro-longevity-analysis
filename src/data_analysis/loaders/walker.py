"""Loader for Walker et al. 2018 cocaine self-administration (GSE110344, mouse)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_analysis.geo import parse_series_matrix
from data_analysis.loaders import Dataset, check_expected

REGIONS = ["BLA", "Cpu", "Hipp", "NAc", "PFC", "VTA"]
GROUPS = ["S24", "C24", "SS", "SC", "CS", "CC"]

# (self-administration, challenge) -> group, per PIPELINE_SPEC 3.1
GROUP_MAP = {
    ("Saline", "None"): "S24",
    ("Cocaine", "None"): "C24",
    ("Saline", "Saline"): "SS",
    ("Saline", "Cocaine"): "SC",
    ("Cocaine", "Saline"): "CS",
    ("Cocaine", "Cocaine"): "CC",
}
N_SAMPLES = 230
N_PER_CELL = (3, 8)


def load_metadata(series_matrix: str | Path) -> pd.DataFrame:
    """Sample metadata indexed by count-table column name (`{Region}-{AnimalID}`)."""
    sm = parse_series_matrix(series_matrix)
    meta = pd.DataFrame({
        "gsm": sm.index,
        "sample": sm["title"].values,
        "region_name": sm["region"].values,
        "self_administration": sm["self_administration"].values,
        "challenge": sm["challenge"].values,
    })
    meta[["region", "animal_id"]] = meta["sample"].str.split("-", n=1, expand=True)

    pairs = list(zip(meta["self_administration"], meta["challenge"]))
    unknown = sorted(set(pairs) - set(GROUP_MAP))
    if unknown:
        raise ValueError(f"unknown (self-administration, challenge) combinations: {unknown}")
    meta["group"] = pd.Categorical([GROUP_MAP[p] for p in pairs], categories=GROUPS)
    meta["drug"] = meta["self_administration"]
    meta["withdrawal"] = meta["challenge"].map(lambda c: "24h" if c == "None" else "30d")
    meta["region"] = pd.Categorical(meta["region"], categories=REGIONS)
    return meta.set_index("sample")


def load_counts(counts: str | Path) -> pd.DataFrame:
    """Raw counts, genes (Ensembl IDs) x samples."""
    df = pd.read_csv(counts, sep="\t", index_col="Gene")
    if df.isna().any().any():
        raise ValueError("missing values in Walker count table")
    return df.astype("int64")


def validate(ds: Dataset) -> pd.DataFrame:
    """Check sample numbers against PIPELINE_SPEC; return the region x group count table."""
    check_expected("walker total samples", len(ds.meta), N_SAMPLES)
    table = pd.crosstab(ds.meta["region"], ds.meta["group"])
    check_expected("walker regions", list(table.index), REGIONS)
    check_expected("walker groups", list(table.columns), GROUPS)
    lo, hi = N_PER_CELL
    if not ((table >= lo) & (table <= hi)).all().all():
        raise AssertionError(f"walker region x group n outside {lo}-{hi}:\n{table}")
    check_expected("walker VTA CS n", int(table.loc["VTA", "CS"]), 3)
    return table


def load_walker(cfg: dict) -> Dataset:
    paths = cfg["data"]["walker"]
    meta = load_metadata(paths["series_matrix"])
    counts = load_counts(paths["counts"])
    missing = sorted(set(meta.index) ^ set(counts.columns))
    if missing:
        raise ValueError(f"walker samples not shared by counts and metadata: {missing}")
    ds = Dataset("walker", counts[meta.index], meta, gene_id="ensembl")
    validate(ds)
    return ds
