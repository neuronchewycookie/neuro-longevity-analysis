"""Loader for Hahn et al. 2023 brain aging bulk-seq (GSE212336, mouse)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_analysis.geo import parse_series_matrix
from data_analysis.loaders import Dataset, check_expected

log = logging.getLogger(__name__)

# Region codes used in column names (BulkSeq_{code}_{mouseID}), per PIPELINE_SPEC 3.2
REGIONS = {
    "cc": "Corpus callosum",
    "cer": "Cerebellum",
    "cor": "Motor cortex",
    "cp": "Caudate putamen",
    "ent": "Entorhinal cortex",
    "hi": "Hippocampus (anterior)",
    "hi2": "Hippocampus (posterior)",
    "hy": "Hypothalamus",
    "med": "Medulla",
    "olf": "Olfactory bulb",
    "plx": "Choroid Plexus",
    "pon": "Pons",
    "svz": "Subventricular zone",
    "th": "Thalamus",
    "vis": "Visual Cortex",
}
AGES = {"Male": [3, 12, 15, 18, 21, 26, 28], "Female": [3, 12, 15, 18, 21]}
N_SAMPLES = 847
SAMPLE_NAME_COLUMN = "description_3"  # third !Sample_description row holds BulkSeq_{region}_{id}


def load_metadata(series_matrix: str | Path) -> pd.DataFrame:
    """Sample metadata indexed by count-table column name (`BulkSeq_{region}_{mouseID}`)."""
    sm = parse_series_matrix(series_matrix)
    meta = pd.DataFrame({
        "gsm": sm.index,
        "sample": sm[SAMPLE_NAME_COLUMN].values,
        "region_name": sm["source_name_ch1"].values,
        "sex": sm["sex"].values,
        "age_months": sm["age_months"].astype(int).values,
        "mouse_id": sm["mouse_id_number"].values,
    })
    parts = meta["sample"].str.split("_", expand=True)
    if parts.shape[1] != 3 or (parts[0] != "BulkSeq").any():
        raise ValueError("unexpected Hahn sample name format; expected BulkSeq_{region}_{id}")
    meta["region"] = parts[1]

    unknown = sorted(set(meta["region"]) - set(REGIONS))
    if unknown:
        raise ValueError(f"unknown Hahn region codes: {unknown}")
    mismatch = meta[meta["region"].map(REGIONS) != meta["region_name"]]
    if len(mismatch):
        raise ValueError(f"region code and source name disagree for {len(mismatch)} samples")
    id_mismatch = meta[parts[2] != meta["mouse_id"]]
    if len(id_mismatch):
        raise ValueError(f"mouse ID in sample name and metadata disagree for {len(id_mismatch)} samples")

    meta["region"] = pd.Categorical(meta["region"], categories=list(REGIONS))
    return meta.set_index("sample")


def load_counts(counts: str | Path) -> pd.DataFrame:
    """Raw counts, genes (symbols) x samples."""
    df = pd.read_csv(counts, sep="\t", index_col=0)
    df.index.name = "gene"
    if df.isna().any().any():
        raise ValueError("missing values in Hahn count table")
    dup = df.index.duplicated()
    if dup.any():
        log.warning("Hahn counts: %d duplicated gene symbols summed: %s",
                    dup.sum(), sorted(set(df.index[dup]))[:20])
        df = df.groupby(level=0).sum()
    return df.astype("int64")


def validate(ds: Dataset) -> pd.DataFrame:
    """Check sample numbers and age design; return the region x sex x age count table."""
    check_expected("hahn total samples", len(ds.meta), N_SAMPLES)
    check_expected("hahn regions", ds.regions, sorted(REGIONS))
    for sex, ages in AGES.items():
        observed = sorted(ds.meta.loc[ds.meta["sex"] == sex, "age_months"].unique())
        check_expected(f"hahn {sex} ages", observed, ages)
    return ds.sample_counts(["region", "sex", "age_months"])


def load_hahn(cfg: dict) -> Dataset:
    paths = cfg["data"]["hahn"]
    meta = load_metadata(paths["series_matrix"])
    counts = load_counts(paths["counts"])
    missing = sorted(set(meta.index) ^ set(counts.columns))
    if missing:
        raise ValueError(f"hahn samples not shared by counts and metadata: {missing[:10]}")
    ds = Dataset("hahn", counts[meta.index], meta, gene_id="symbol")
    validate(ds)
    return ds
