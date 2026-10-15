"""Sample-level QC: library sizes and per-region PCA with outlier flags (flag only, never drop)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_analysis.loaders import Dataset
from data_analysis.normalize import log_expression

MAD_SCALE = 1.4826  # makes the MAD consistent with the SD for normal data


def library_sizes(ds: Dataset, extra: list[str] | None = None, z_threshold: float = 3.5) -> pd.DataFrame:
    """Per-sample total counts with region (and optional metadata columns).

    `low_library` flags samples whose log10 library size is below the region median by more
    than `z_threshold` robust SDs.
    """
    cols = ["region", *(extra or [])]
    out = ds.meta[cols].copy()
    out.insert(0, "dataset", ds.name)
    out["lib_size"] = ds.counts.sum(axis=0)
    out["n_detected_genes"] = (ds.counts > 0).sum(axis=0)
    log_lib = np.log10(out["lib_size"].clip(lower=1))
    out["log_lib_z"] = log_lib.groupby(out["region"], observed=True).transform(robust_z)
    out["low_library"] = out["log_lib_z"] < -z_threshold
    return out.reset_index()


def robust_z(x: pd.Series) -> pd.Series:
    """(x - median) / (1.4826 * MAD); zero spread gives all-zero scores."""
    mad = (x - x.median()).abs().median() * MAD_SCALE
    if mad == 0:
        return x * 0.0
    return (x - x.median()) / mad


def pca_scores(log_expr: pd.DataFrame, n_top: int = 500, n_pcs: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """PCA on the `n_top` most variable genes; returns (sample x PC scores, explained variance ratio)."""
    expressed = log_expr[log_expr.var(axis=1) > 0]
    top = expressed.loc[expressed.var(axis=1).sort_values(ascending=False).index[:n_top]]
    centered = top.sub(top.mean(axis=1), axis=0).to_numpy().T  # samples x genes
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :n_pcs] * s[:n_pcs]
    explained = s**2 / np.sum(s**2)
    cols = [f"PC{i + 1}" for i in range(n_pcs)]
    return pd.DataFrame(scores, index=top.columns, columns=cols), explained[:n_pcs]


def knn_distance(scores: pd.DataFrame, k: int = 3) -> pd.Series:
    """Mean Euclidean distance from each sample to its k nearest other samples."""
    x = scores.to_numpy()
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(d, np.inf)
    k = min(k, len(x) - 1)
    return pd.Series(np.sort(d, axis=1)[:, :k].mean(axis=1), index=scores.index)


def pca_by_region(ds: Dataset, n_top: int = 500, z_threshold: float = 3.5,
                  extra: list[str] | None = None, n_pcs: int = 10, k: int = 3) -> pd.DataFrame:
    """PC1/PC2 per region with isolation-based outlier flags.

    A sample is flagged when its mean distance to its `k` nearest neighbours (in the first
    `n_pcs` PCs) is more than `z_threshold` robust SDs above the region median. Unlike a
    z-score on PC1/PC2 directly, this does not flag every member of a tight sub-cluster
    (e.g. a batch), only samples that sit apart from all others.
    """
    log_expr = log_expression(ds)
    frames = []
    for region in ds.regions:
        idx = ds.meta.index[ds.meta["region"] == region]
        scores, explained = pca_scores(log_expr[idx], n_top=n_top, n_pcs=min(n_pcs, len(idx) - 1))
        dist = knn_distance(scores, k=k)
        frame = ds.meta.loc[idx, ["region", *(extra or [])]].copy()
        frame[["PC1", "PC2"]] = scores[["PC1", "PC2"]]
        frame["PC1_var"], frame["PC2_var"] = explained[:2]
        frame["knn_dist"] = dist
        frame["knn_z"] = robust_z(dist)
        frame["outlier"] = frame["knn_z"] > z_threshold
        frames.append(frame)
    out = pd.concat(frames)
    out.insert(0, "dataset", ds.name)
    return out.reset_index()
