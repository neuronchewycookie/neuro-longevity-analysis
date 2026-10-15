"""Expression normalization: DESeq2 VST (pydeseq2) or log2(CPM + 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_analysis.loaders import Dataset


def log_cpm(counts: pd.DataFrame, prior: float = 1.0) -> pd.DataFrame:
    """log2(counts per million + prior), genes x samples."""
    lib = counts.sum(axis=0)
    if (lib == 0).any():
        raise ValueError(f"samples with zero library size: {lib.index[lib == 0].tolist()}")
    return np.log2(counts.div(lib, axis=1) * 1e6 + prior)


def log1p_values(values: pd.DataFrame) -> pd.DataFrame:
    """log2(x + 1) for already-normalized values (e.g. rat FPKM)."""
    if (values < 0).any().any():
        raise ValueError("negative values cannot be log-transformed")
    return np.log2(values + 1)


def log_expression(ds: Dataset) -> pd.DataFrame:
    """Log-scale expression: log2(CPM + 1) for counts, log2(x + 1) for normalized values."""
    return log_cpm(ds.counts) if ds.is_count else log1p_values(ds.counts)
