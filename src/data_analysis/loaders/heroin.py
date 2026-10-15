"""Loader for Browne et al. heroin self-administration (GSE228031, mouse)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_analysis.loaders import Dataset, check_expected

REGIONS = ["BLA", "dStri", "NAc", "PFC", "VTA", "vHPC"]
GROUPS = ["S24", "H24", "SS", "SH", "HS", "HH"]  # Walker design with C -> H
N_SAMPLES = 305
N_PER_CELL = (7, 11)

# Matching Walker region names (PIPELINE_SPEC Step 5)
WALKER_REGION = {"BLA": "BLA", "dStri": "Cpu", "NAc": "NAc", "PFC": "PFC", "VTA": "VTA", "vHPC": "Hipp"}


def load_metadata(annotation: str | Path) -> pd.DataFrame:
    """Sample metadata indexed by count-matrix column name (Sample_ID)."""
    ann = pd.read_csv(annotation, index_col=0)
    meta = ann.rename(columns={"Sample_ID": "sample", "Mouse_ID": "mouse_id", "Group": "group", "Region": "region"})
    if meta["sample"].duplicated().any():
        raise ValueError("duplicated Sample_ID in heroin annotation")
    unknown_groups = sorted(set(meta["group"]) - set(GROUPS))
    unknown_regions = sorted(set(meta["region"]) - set(REGIONS))
    if unknown_groups or unknown_regions:
        raise ValueError(f"unknown heroin groups {unknown_groups} / regions {unknown_regions}")

    meta["drug"] = meta["group"].str[0].map({"S": "Saline", "H": "Heroin"})
    meta["withdrawal"] = meta["group"].map(lambda g: "24h" if g.endswith("24") else "30d")
    meta["walker_region"] = meta["region"].map(WALKER_REGION)
    meta["group"] = pd.Categorical(meta["group"], categories=GROUPS)
    meta["region"] = pd.Categorical(meta["region"], categories=REGIONS)
    return meta.set_index("sample")[["region", "walker_region", "group", "drug", "withdrawal", "mouse_id"]]


def load_counts(counts: str | Path) -> pd.DataFrame:
    """Raw counts, genes (Ensembl IDs) x samples."""
    df = pd.read_csv(counts, index_col=0)
    df.index.name = "gene"
    if df.isna().any().any():
        raise ValueError("missing values in heroin count matrix")
    return df.astype("int64")


def validate(ds: Dataset) -> pd.DataFrame:
    """Check sample numbers against PIPELINE_SPEC; return the region x group count table."""
    check_expected("heroin total samples", len(ds.meta), N_SAMPLES)
    table = pd.crosstab(ds.meta["region"], ds.meta["group"])
    check_expected("heroin regions", list(table.index), REGIONS)
    check_expected("heroin groups", list(table.columns), GROUPS)
    lo, hi = N_PER_CELL
    if not ((table >= lo) & (table <= hi)).all().all():
        raise AssertionError(f"heroin region x group n outside {lo}-{hi}:\n{table}")
    return table


def load_heroin(cfg: dict) -> Dataset:
    paths = cfg["data"]["heroin"]
    meta = load_metadata(paths["annotation"])
    counts = load_counts(paths["counts"])
    missing = sorted(set(meta.index) ^ set(counts.columns))
    if missing:
        raise ValueError(f"heroin samples not shared by counts and annotation: {missing}")
    ds = Dataset("heroin", counts[meta.index], meta, gene_id="ensembl")
    validate(ds)
    return ds
