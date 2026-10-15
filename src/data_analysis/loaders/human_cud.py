"""Loader for human cocaine use disorder postmortem brain (GSE214267)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from data_analysis.geo import parse_series_matrix
from data_analysis.loaders import Dataset, check_expected

log = logging.getLogger(__name__)

REGION_CODES = {"Caudate nucleus": "Caudate", "Nucleus accumbens": "NAc"}
DIAGNOSIS = {"Cocaine": "CUD", "Control": "Control"}
N_SAMPLES = 89
N_SUBJECTS = 45
N_PAIRED_SUBJECTS = 44  # subjects with both caudate and NAc samples
FILE_PATTERN = re.compile(r"^(GSM\d+)_(\w+)\.counts\.txt$")

# GEO gives no donor ID. Samples sharing all of these come from the same donor.
SUBJECT_KEYS = ["diagnosis", "age", "pmi", "race"]


def read_featurecounts(path: str | Path) -> pd.Series:
    """Counts (7th column) of one featureCounts output, indexed by Ensembl gene ID."""
    df = pd.read_csv(path, sep="\t", comment="#", index_col=0)
    return df.iloc[:, 5].astype("int64")


def load_counts(raw_dir: str | Path, pattern: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Combine per-sample featureCounts files into genes x samples (columns named by GSM).

    Also returns {GSM: well} parsed from file names (GSM*_{well}.counts.txt).
    """
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern} in {raw_dir}")
    series, wells = {}, {}
    for path in files:
        match = FILE_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"unexpected count file name: {path.name}")
        gsm, wells[gsm] = match.group(1), match.group(2)
        series[gsm] = read_featurecounts(path)
    index = next(iter(series.values())).index
    for gsm, s in series.items():
        if not s.index.equals(index):
            raise ValueError(f"{gsm}: gene order differs from the first count file")
    counts = pd.DataFrame(series)
    counts.index.name = "gene"
    return counts, wells


def load_metadata(series_matrix: str | Path) -> pd.DataFrame:
    """Sample metadata indexed by GSM; no RIN or subject ID is provided by GEO."""
    sm = parse_series_matrix(series_matrix)
    unknown = sorted(set(sm["brain_region"]) - set(REGION_CODES)) + sorted(set(sm["group"]) - set(DIAGNOSIS))
    if unknown:
        raise ValueError(f"unknown human region/group values: {unknown}")
    meta = pd.DataFrame({
        "well": sm["title"],
        "region_name": sm["brain_region"],
        "region": sm["brain_region"].map(REGION_CODES),
        "diagnosis": pd.Categorical(sm["group"].map(DIAGNOSIS), categories=["Control", "CUD"]),
        "sex": sm["sex"].str.capitalize(),
        "race": sm["race"],
        "age": sm["age"].astype(float),
        "pmi": sm["pmi"].astype(float),
    }, index=sm.index)
    meta.index.name = "sample"
    meta["subject_id"] = infer_subject_ids(meta)
    return meta


def infer_subject_ids(meta: pd.DataFrame) -> pd.Series:
    """Infer donor IDs (S01, S02, ...) from identical diagnosis/age/PMI/race.

    This is an inferred ID, not one provided by GEO. Raises if a donor would have two samples
    from the same region, which would mean the covariates do not identify donors uniquely.
    """
    keys = meta[SUBJECT_KEYS].astype(str).agg("|".join, axis=1)
    codes = {k: f"S{i:02d}" for i, k in enumerate(sorted(keys.unique()), start=1)}
    subject = keys.map(codes)

    repeats = meta.assign(subject_id=subject).groupby(["subject_id", "region"]).size()
    if (repeats > 1).any():
        raise ValueError(f"inferred donors with repeated regions: {repeats[repeats > 1].index.tolist()}")
    n_regions = subject.value_counts()
    if len(n_regions) != N_SUBJECTS or (n_regions == 2).sum() != N_PAIRED_SUBJECTS:
        log.warning("human donors inferred: %d (paired %d), expected %d (paired %d)",
                    len(n_regions), (n_regions == 2).sum(), N_SUBJECTS, N_PAIRED_SUBJECTS)
    return subject.rename("subject_id")


def validate(ds: Dataset) -> pd.DataFrame:
    """Check sample numbers; return the region x diagnosis count table."""
    check_expected("human total samples", len(ds.meta), N_SAMPLES)
    table = pd.crosstab(ds.meta["region"], ds.meta["diagnosis"])
    check_expected("human regions", sorted(table.index), ["Caudate", "NAc"])
    return table


def load_human_cud(cfg: dict) -> Dataset | None:
    """Return the human dataset, or None (with a warning) when the series_matrix is missing."""
    paths = cfg["data"]["human"]
    if not Path(paths["series_matrix"]).is_file():
        log.warning("GSE214267 series_matrix not found at %s; skipping human data", paths["series_matrix"])
        return None
    meta = load_metadata(paths["series_matrix"])
    counts, wells = load_counts(paths["raw_dir"], paths["counts_glob"])
    missing = sorted(set(meta.index) ^ set(counts.columns))
    if missing:
        raise ValueError(f"human samples not shared by counts and metadata: {missing}")
    bad = sorted(gsm for gsm, well in wells.items() if meta.loc[gsm, "well"] != well)
    if bad:
        raise ValueError(f"count file well names disagree with series_matrix titles: {bad}")
    ds = Dataset("human_cud", counts[meta.index], meta, gene_id="ensembl")
    validate(ds)
    return ds
