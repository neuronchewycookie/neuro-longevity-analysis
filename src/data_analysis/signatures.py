"""Aging signatures: Hahn et al. 2023 CAS genes with direction, striatal signature."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Sheet names in NIHMS1924184-supplement-11.xlsx (note the trailing space on the DEG sheet).
DEG_SHEET = "DEG detection across regions "
AGE_SHEET = "Age-correlated genes"
HEADER_ROW = 1  # row 0 holds the sheet title and caption

GENE_COL = "Gene Symbol"
CLASS_COL = "Signature class"
REGION_COL = "Region"
PADJ_COL = "Adj. Pval"
CAS_CLASS = "CAS gene"
STRIATUM_REGION = "Caudate putamen"

# Down-regulated CAS genes reported in PIPELINE_SPEC (used as a cross-check).
EXPECTED_DOWN = frozenset({"Gpr17", "Chordc1", "Hsph1", "P4ha1", "Ahsa1", "Dnajb1", "Hist3h2a"})


def _rho_column(columns: pd.Index) -> str:
    """Find the Spearman column; its header contains a mis-encoded apostrophe."""
    matches = [c for c in columns if str(c).startswith("Spearman")]
    if len(matches) != 1:
        raise ValueError(f"expected one Spearman column, found {matches}")
    return matches[0]


def load_age_correlations(supplement: str | Path) -> pd.DataFrame:
    """Return the age-correlation sheet as columns: gene, region, rho, pval, padj."""
    df = pd.read_excel(supplement, sheet_name=AGE_SHEET, header=HEADER_ROW)
    df = df.rename(columns={
        GENE_COL: "gene",
        REGION_COL: "region",
        _rho_column(df.columns): "rho",
        "P val": "pval",
        PADJ_COL: "padj",
    })
    df["gene"] = df["gene"].astype(str).str.strip()
    return df[["gene", "region", "rho", "pval", "padj"]]


def _direction_from_rho(rho: pd.Series) -> pd.Series:
    return rho.map(lambda r: "up" if r > 0 else "down")


def load_cas_signature(
    supplement: str | Path,
    expected_n_up: int = 75,
    expected_n_down: int = 7,
) -> pd.DataFrame:
    """Return the 82 Common Aging Score genes with their aging direction.

    Genes come from the DEG sheet (`Signature class == "CAS gene"`); direction is the sign of
    the Spearman age correlation, which must agree across every region where the gene is listed.

    Columns: gene, direction ('up'/'down'), n_regions, mean_rho, min_rho, max_rho.
    Raises ValueError if counts or directions differ from expectations.
    """
    deg = pd.read_excel(supplement, sheet_name=DEG_SHEET, header=HEADER_ROW)
    genes = deg.loc[deg[CLASS_COL] == CAS_CLASS, GENE_COL].astype(str).str.strip()
    if genes.duplicated().any():
        raise ValueError(f"duplicated CAS genes: {sorted(genes[genes.duplicated()])}")

    age = load_age_correlations(supplement)
    age = age[age["gene"].isin(genes)]
    missing = sorted(set(genes) - set(age["gene"]))
    if missing:
        raise ValueError(f"CAS genes without age correlation: {missing}")

    summary = age.groupby("gene")["rho"].agg(
        n_regions="count", mean_rho="mean", min_rho="min", max_rho="max"
    )
    mixed = summary.index[(summary["min_rho"] < 0) & (summary["max_rho"] > 0)]
    if len(mixed):
        raise ValueError(f"CAS genes with inconsistent age direction across regions: {sorted(mixed)}")

    summary["direction"] = _direction_from_rho(summary["mean_rho"])
    signature = (
        summary.reindex(genes.tolist())
        .reset_index()
        .loc[:, ["gene", "direction", "n_regions", "mean_rho", "min_rho", "max_rho"]]
    )

    n_up = int((signature["direction"] == "up").sum())
    n_down = int((signature["direction"] == "down").sum())
    if (n_up, n_down) != (expected_n_up, expected_n_down):
        raise ValueError(
            f"CAS direction counts {n_up} up / {n_down} down, expected {expected_n_up} / {expected_n_down}"
        )
    down = set(signature.loc[signature["direction"] == "down", "gene"])
    if down != EXPECTED_DOWN:
        raise ValueError(
            f"down genes differ from spec: extra {sorted(down - EXPECTED_DOWN)}, "
            f"missing {sorted(EXPECTED_DOWN - down)}"
        )
    return signature


def split_directions(signature: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (up_genes, down_genes) from a signature table with gene/direction columns."""
    up = signature.loc[signature["direction"] == "up", "gene"].tolist()
    down = signature.loc[signature["direction"] == "down", "gene"].tolist()
    return up, down


def load_striatal_signature(
    supplement: str | Path,
    padj_threshold: float = 0.05,
    exclude_genes: set[str] | None = None,
) -> pd.DataFrame:
    """Return caudate-putamen age-correlated genes (padj < threshold) with direction from rho sign.

    Pass the CAS genes as `exclude_genes` to build the striatum-specific signature.
    Columns: gene, direction, rho, padj.
    """
    age = load_age_correlations(supplement)
    cp = age[(age["region"] == STRIATUM_REGION) & (age["padj"] < padj_threshold)].copy()
    if cp.empty:
        raise ValueError(f"no genes for region {STRIATUM_REGION!r} at padj < {padj_threshold}")
    if cp["gene"].duplicated().any():
        raise ValueError("duplicated genes within the caudate putamen age-correlation list")
    if exclude_genes:
        cp = cp[~cp["gene"].isin(exclude_genes)]
    cp["direction"] = _direction_from_rho(cp["rho"])
    return cp[["gene", "direction", "rho", "padj"]].reset_index(drop=True)
