"""Per-dataset loaders returning (counts, metadata)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Expression matrix plus aligned sample metadata.

    counts: genes x samples (raw counts, or normalized values when `is_count` is False)
    meta:   one row per sample, index equals `counts.columns`; always has a `region` column
    gene_id: 'ensembl' or 'symbol' — what `counts.index` holds
    """

    name: str
    counts: pd.DataFrame
    meta: pd.DataFrame
    gene_id: str
    is_count: bool = True

    def __post_init__(self) -> None:
        if not self.counts.columns.equals(self.meta.index):
            raise ValueError(f"{self.name}: counts columns and meta index are not aligned")
        if self.counts.index.duplicated().any():
            raise ValueError(f"{self.name}: duplicated gene IDs in counts")
        if "region" not in self.meta.columns:
            raise ValueError(f"{self.name}: meta needs a 'region' column")

    @property
    def regions(self) -> list[str]:
        return sorted(self.meta["region"].unique())

    def subset(self, mask: pd.Series | None = None, **equals) -> "Dataset":
        """Return samples matching a boolean mask and/or column == value filters (e.g. region='Cpu')."""
        keep = pd.Series(True, index=self.meta.index) if mask is None else mask.reindex(self.meta.index)
        for column, value in equals.items():
            keep &= self.meta[column] == value
        meta = self.meta[keep]
        return Dataset(self.name, self.counts[meta.index], meta, self.gene_id, self.is_count)

    def sample_counts(self, by: list[str]) -> pd.DataFrame:
        """Number of samples per combination of `by` columns, as a tidy table."""
        return self.meta.groupby(by, observed=True).size().rename("n").reset_index()

    def to_parquet(self, directory: str | Path) -> None:
        """Write `{name}_counts.parquet`, `{name}_meta.parquet` and `{name}_info.json`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.counts.to_parquet(directory / f"{self.name}_counts.parquet")
        self.meta.to_parquet(directory / f"{self.name}_meta.parquet")
        info = {"name": self.name, "gene_id": self.gene_id, "is_count": self.is_count}
        (directory / f"{self.name}_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")

    @classmethod
    def from_parquet(cls, directory: str | Path, name: str) -> "Dataset":
        """Read a dataset written by `to_parquet`."""
        directory = Path(directory)
        info = json.loads((directory / f"{name}_info.json").read_text(encoding="utf-8"))
        counts = pd.read_parquet(directory / f"{name}_counts.parquet")
        meta = pd.read_parquet(directory / f"{name}_meta.parquet")
        return cls(info["name"], counts, meta, info["gene_id"], info["is_count"])


DATASETS = ["walker", "hahn", "heroin", "rat_reward", "human_cud"]


def load_raw(name: str, cfg: dict) -> Dataset | None:
    """Load one dataset from data/raw with its loader (slow; validates against the spec)."""
    from data_analysis.loaders import hahn, heroin, human_cud, rat_reward, walker

    loaders = {
        "walker": walker.load_walker,
        "hahn": hahn.load_hahn,
        "heroin": heroin.load_heroin,
        "rat_reward": rat_reward.load_rat_reward,
        "human_cud": human_cud.load_human_cud,
    }
    return loaders[name](cfg)


def load_processed(name: str, cfg: dict) -> Dataset:
    """Load one dataset from the parquet cache written by scripts/01_load_all.py."""
    directory = cfg["paths"]["processed"]
    if not (directory / f"{name}_info.json").is_file():
        raise FileNotFoundError(f"{name} not cached in {directory}; run scripts/01_load_all.py first")
    return Dataset.from_parquet(directory, name)


def check_expected(name: str, actual: object, expected: object) -> None:
    """Raise with a readable message when a sanity check fails."""
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
