"""Shared parser for GEO series_matrix files."""

from __future__ import annotations

import gzip
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

SAMPLE_PREFIX = "!Sample_"
CHARACTERISTICS_KEY = "characteristics_ch1"


def _open_text(path: Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _split_line(line: str) -> list[str]:
    return [cell.strip().strip('"') for cell in line.rstrip("\r\n").split("\t")]


def read_sample_rows(path: str | Path) -> list[tuple[str, list[str]]]:
    """Return every `!Sample_*` header row as (key without prefix, per-sample values), in file order."""
    rows: list[tuple[str, list[str]]] = []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith(SAMPLE_PREFIX):
                key, *values = _split_line(line)
                rows.append((key[len(SAMPLE_PREFIX):], values))
    if not rows:
        raise ValueError(f"no {SAMPLE_PREFIX} rows found in {path}")
    n_samples = {len(values) for _, values in rows}
    if len(n_samples) != 1:
        raise ValueError(f"inconsistent sample counts across rows in {path}: {sorted(n_samples)}")
    return rows


def normalize_key(key: str) -> str:
    """Lowercase a characteristic key and collapse non-alphanumerics to '_' (e.g. 'age (months)' -> 'age_months')."""
    return re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_")


def parse_characteristic(cell: str) -> tuple[str, str] | None:
    """Split a 'key: value' characteristic cell; return None for empty cells."""
    if not cell:
        return None
    key, sep, value = cell.partition(":")
    if not sep:
        raise ValueError(f"characteristic cell without 'key: value' form: {cell!r}")
    return normalize_key(key), value.strip()


def parse_series_matrix(path: str | Path) -> pd.DataFrame:
    """Parse sample metadata from a GEO series_matrix into one row per sample.

    - Index is the GSM accession (`geo_accession`).
    - Single `!Sample_*` rows keep their name (e.g. `title`, `source_name_ch1`).
    - Repeated rows are numbered from 1 (e.g. `description_1`, `description_2`, ...).
    - Characteristics are parsed per cell by their own key, because GEO may put different
      keys on the same row for different samples. Each key becomes a column; samples
      lacking a key get NA.
    """
    rows = read_sample_rows(path)
    n_samples = len(rows[0][1])

    occurrences: dict[str, int] = defaultdict(int)
    for key, _ in rows:
        occurrences[key] += 1

    columns: dict[str, list] = {}
    seen: dict[str, int] = defaultdict(int)
    characteristics: list[dict[str, str]] = [{} for _ in range(n_samples)]

    for key, values in rows:
        if key == CHARACTERISTICS_KEY:
            for i, cell in enumerate(values):
                parsed = parse_characteristic(cell)
                if parsed is None:
                    continue
                char_key, value = parsed
                if char_key in characteristics[i] and characteristics[i][char_key] != value:
                    raise ValueError(f"sample {i}: conflicting values for characteristic {char_key!r}")
                characteristics[i][char_key] = value
            continue
        seen[key] += 1
        name = f"{key}_{seen[key]}" if occurrences[key] > 1 else key
        columns[name] = values

    meta = pd.DataFrame(columns)
    char_df = pd.DataFrame(characteristics)
    clashes = set(char_df.columns) & set(meta.columns)
    if clashes:
        raise ValueError(f"characteristic keys clash with sample fields: {sorted(clashes)}")
    meta = pd.concat([meta, char_df], axis=1)

    if meta["geo_accession"].duplicated().any():
        raise ValueError(f"duplicated GSM accessions in {path}")
    return meta.set_index("geo_accession")
