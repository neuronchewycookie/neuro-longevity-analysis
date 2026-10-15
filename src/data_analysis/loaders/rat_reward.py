"""Loader for morphine vs sucrose NAc shell transcriptomes (GSE174737, rat)."""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path

import pandas as pd

from data_analysis.geo import parse_series_matrix
from data_analysis.loaders import Dataset, check_expected

log = logging.getLogger(__name__)

# config key -> (sex, reinforcer); each file holds the reinforcer and its own control group
FILES = {
    "male_morphine": ("Male", "Morphine"),
    "female_morphine": ("Female", "Morphine"),
    "male_sucrose": ("Male", "Sucrose"),
    "female_sucrose": ("Female", "Sucrose"),
}
CONTROL = {"Morphine": "Saline", "Sucrose": "Cue"}
GROUPS = ["Saline", "Morphine_D1", "Morphine_D30", "Cue", "Sucrose_D1", "Sucrose_D30"]
N_SAMPLES = 92  # 93 GEO samples minus one exact duplicate column (Male_Morphine 233 == 231)
N_PER_CELL = (7, 8)
REGION = "NAc_shell"


def read_normalized(path: str | Path) -> pd.DataFrame:
    """Read one transcript-level file as rows (t_name, gene_name, samples...).

    Male_Morphine repeats t_name in every data row, so data rows are one field longer
    than the header; the duplicated column is dropped after checking it matches.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        rows = list(reader)
    widths = {len(r) for r in rows}
    if widths == {len(header)}:
        fixed = rows
    elif widths == {len(header) + 1}:
        if any(r[0] != r[1] for r in rows):
            raise ValueError(f"{Path(path).name}: extra column is not a repeated t_name")
        log.warning("%s: data rows repeat t_name; dropped the extra column", Path(path).name)
        fixed = [[r[0], *r[2:]] for r in rows]
    else:
        raise ValueError(f"{Path(path).name}: inconsistent row widths {sorted(widths)} vs header {len(header)}")
    df = pd.DataFrame(fixed, columns=header)
    values = df.columns[2:]
    df[values] = df[values].astype(float)
    return df


def sum_to_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Sum transcript values per gene_name -> genes x samples."""
    if df["t_name"].duplicated().any():
        raise ValueError("duplicated transcript IDs")
    return df.drop(columns="t_name").groupby("gene_name").sum()


def load_metadata(series_matrix: str | Path) -> pd.DataFrame:
    """Series-matrix metadata with derived reinforcer / group columns, indexed by GSM."""
    sm = parse_series_matrix(series_matrix)
    meta = pd.DataFrame({"title": sm["title"], "sex": sm["sex"]}, index=sm.index)
    meta["agent"] = sm["agent"].fillna("Saline")  # saline controls have no agent
    meta["time_point"] = sm["time_point"].where(sm["time_point"].isin(["Day 1", "Day 30"]))
    meta["reinforcer"] = meta["agent"].map(
        {"Morphine": "Morphine", "Saline": "Morphine", "Sucrose": "Sucrose", "Cue": "Sucrose"}
    )
    day = meta["time_point"].map({"Day 1": "D1", "Day 30": "D30"})
    meta["group"] = meta["agent"].where(day.isna(), meta["agent"] + "_" + day)
    unknown = sorted(set(meta["group"]) - set(GROUPS))
    if unknown:
        raise ValueError(f"unknown rat groups: {unknown}")
    return meta


def match_columns(meta: pd.DataFrame, sex: str, reinforcer: str, columns: list[str]) -> dict[str, str]:
    """Map file columns to GSM accessions within one (sex, reinforcer) file.

    Titles usually equal column names; GEO renamed one duplicated animal ID with a suffix
    (e.g. column 'Blue4' in Male_Sucrose is titled 'Blue4_Suc').
    """
    stratum = meta[(meta["sex"] == sex) & (meta["reinforcer"] == reinforcer)]
    by_title = dict(zip(stratum["title"], stratum.index))
    mapping = {}
    for col in columns:
        hits = [t for t in (col, f"{col}_Suc", f"{col}_Mor") if t in by_title]
        if len(hits) != 1:
            raise ValueError(f"{sex} {reinforcer}: column {col!r} matches titles {hits}")
        if hits[0] != col:
            log.info("%s %s: column %s matched title %s", sex, reinforcer, col, hits[0])
        mapping[col] = by_title[hits[0]]
    if len(set(mapping.values())) != len(mapping) or len(mapping) != len(stratum):
        raise ValueError(f"{sex} {reinforcer}: {len(columns)} columns vs {len(stratum)} metadata samples")
    return mapping


def drop_duplicate_columns(values: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop sample columns that are exact copies of an earlier column (first occurrence kept).

    GEO's Male_Morphine file stores animals 231 and 233 (both Morphine Day 30) with identical
    values for every gene; counting both would treat one animal as two.
    """
    seen: dict[bytes, str] = {}
    duplicates = []
    for col in values.columns:
        key = values[col].to_numpy().tobytes()
        if key in seen:
            duplicates.append(col)
            log.warning("rat: sample %s is an exact copy of %s; dropped (kept %s)", col, seen[key], seen[key])
        else:
            seen[key] = col
    keep = [c for c in values.columns if c not in duplicates]
    return values[keep], meta.loc[keep]


def validate(ds: Dataset) -> pd.DataFrame:
    """Check sample numbers against PIPELINE_SPEC; return the sex x group count table."""
    check_expected("rat total samples", len(ds.meta), N_SAMPLES)
    table = pd.crosstab(ds.meta["group"], ds.meta["sex"]).reindex(GROUPS)
    lo, hi = N_PER_CELL
    if table.isna().any().any() or not ((table >= lo) & (table <= hi)).all().all():
        raise AssertionError(f"rat group x sex n outside {lo}-{hi}:\n{table}")
    return table


def load_rat_reward(cfg: dict) -> Dataset:
    """Gene-level values (transcripts summed per gene_name, not log-transformed) for 92 rats.

    Sample keys are '{file key}:{column}' because animal IDs repeat across files.
    Only genes present in all four files are kept, and exact duplicate sample columns are dropped.
    """
    paths = cfg["data"]["rat"]
    meta_all = load_metadata(paths["series_matrix"])

    blocks, metas = [], []
    for key, (sex, reinforcer) in FILES.items():
        genes = sum_to_genes(read_normalized(paths["normalized"][key]))
        gsm = match_columns(meta_all, sex, reinforcer, list(genes.columns))
        sample_ids = [f"{key}:{col}" for col in genes.columns]
        block_meta = meta_all.loc[[gsm[c] for c in genes.columns]].copy()
        block_meta.insert(0, "gsm", block_meta.index)
        block_meta["file"] = key
        block_meta["animal_id"] = list(genes.columns)
        block_meta.index = sample_ids
        genes.columns = sample_ids
        blocks.append(genes)
        metas.append(block_meta)

    shared = sorted(set.intersection(*(set(b.index) for b in blocks)))
    dropped = sum(len(b) for b in blocks) - len(shared) * len(blocks)
    if dropped:
        log.warning("rat: %d gene rows not shared by all four files were dropped", dropped)
    values = pd.concat([b.loc[shared] for b in blocks], axis=1)
    values.index.name = "gene"

    meta = pd.concat(metas)
    meta.index.name = "sample"
    meta["region"] = REGION
    meta["group"] = pd.Categorical(meta["group"], categories=GROUPS)
    values, meta = drop_duplicate_columns(values, meta)
    ds = Dataset("rat_reward", values, meta, gene_id="symbol", is_count=False)
    validate(ds)
    return ds
