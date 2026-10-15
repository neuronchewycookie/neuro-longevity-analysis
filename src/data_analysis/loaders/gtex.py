"""Loader for GTEx v10 brain basal ganglia (caudate, putamen, nucleus accumbens), human.

Used only as a human age positive control for the CAS: donors span 20-79 years, but public
ages are 10-year bins. Gene IDs are versioned Ensembl IDs and are stripped to the base ID.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis.loaders import Dataset

FOLDER = "GTEx_v10_brain"
TISSUES = {
    "caudate": "gene_reads_v10_brain_caudate_basal_ganglia.gct.gz",
    "putamen": "gene_reads_v10_brain_putamen_basal_ganglia.gct.gz",
    "nucleus_accumbens": "gene_reads_v10_brain_nucleus_accumbens_basal_ganglia.gct.gz",
}
SUBJECTS = "GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt"
SAMPLES = "GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
AGE_MIDPOINT = {"20-29": 25, "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70-79": 75}
SEX = {1: "Male", 2: "Female"}


def read_gct(path: str | Path) -> pd.DataFrame:
    """GCT v1.2 counts -> genes (unversioned Ensembl ID) x samples, integer counts.

    Versions are dropped; if two versioned IDs share a base ID (PAR_Y copies), their counts are summed.
    """
    with gzip.open(path, "rt") as fh:
        version = fh.readline().strip()
        if version != "#1.2":
            raise ValueError(f"{Path(path).name}: unexpected GCT version {version!r}")
        n_genes, n_samples = map(int, fh.readline().split()[:2])
        df = pd.read_csv(fh, sep="\t", index_col=0)
    df = df.drop(columns="Description")
    if df.shape != (n_genes, n_samples):
        raise ValueError(f"{Path(path).name}: shape {df.shape} does not match header {(n_genes, n_samples)}")
    df.index = df.index.str.split(".").str[0]
    df.index.name = "gene"
    return df.groupby(level=0).sum().astype("int64")


def donor_id(sample_id: str) -> str:
    """GTEX-1117F-0011-R5a-SM-GI4VW -> GTEX-1117F."""
    return "-".join(sample_id.split("-")[:2])


def load_gtex(raw_dir: str | Path) -> Dataset:
    """Counts for the three basal ganglia tissues with donor and sample covariates.

    meta columns: region (tissue), subject_id, sex, age_bin, age (bin midpoint), hardy (death
    circumstance, 'missing' when blank), rin, ischemic_time (minutes).
    """
    folder = Path(raw_dir) / FOLDER
    blocks = {tissue: read_gct(folder / name) for tissue, name in TISSUES.items()}
    shared = sorted(set.intersection(*(set(b.index) for b in blocks.values())))
    counts = pd.concat([b.loc[shared] for b in blocks.values()], axis=1)

    region = pd.Series({s: t for t, b in blocks.items() for s in b.columns}, name="region")
    subjects = pd.read_csv(folder / SUBJECTS, sep="\t").set_index("SUBJID")
    samples = pd.read_csv(folder / SAMPLES, sep="\t", usecols=["SAMPID", "SMRIN", "SMTSISCH"],
                          low_memory=False).set_index("SAMPID")

    meta = pd.DataFrame(index=counts.columns)
    meta.index.name = "sample"
    meta["region"] = region.loc[meta.index]
    meta["subject_id"] = [donor_id(s) for s in meta.index]
    subj = subjects.loc[meta["subject_id"]]
    meta["sex"] = subj["SEX"].map(SEX).to_numpy()
    meta["age_bin"] = subj["AGE"].to_numpy()
    meta["age"] = meta["age_bin"].map(AGE_MIDPOINT).astype(float)
    meta["hardy"] = subj["DTHHRDY"].map(lambda v: "missing" if pd.isna(v) else str(int(v))).to_numpy()
    meta["rin"] = samples.loc[meta.index, "SMRIN"].to_numpy(dtype=float)
    meta["ischemic_time"] = samples.loc[meta.index, "SMTSISCH"].to_numpy(dtype=float)

    if meta["age"].isna().any():
        raise ValueError(f"unknown GTEx age bins: {sorted(set(meta.loc[meta['age'].isna(), 'age_bin']))}")
    if meta.groupby(["subject_id", "region"]).size().max() > 1:
        raise ValueError("a GTEx donor has more than one sample in the same tissue")
    return Dataset("gtex_brain", counts, meta, gene_id="ensembl")
