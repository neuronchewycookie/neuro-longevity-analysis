import numpy as np
import pandas as pd
import pytest

from data_analysis.config import load_config
from data_analysis.loaders import DATASETS, Dataset, load_processed
from data_analysis.normalize import log_cpm
from data_analysis.qc import knn_distance, robust_z


def toy_dataset() -> Dataset:
    counts = pd.DataFrame({"s1": [10, 0, 5], "s2": [20, 1, 5], "s3": [0, 3, 7]}, index=["g1", "g2", "g3"])
    meta = pd.DataFrame({"region": pd.Categorical(["A", "A", "B"]), "group": ["x", "y", "x"]}, index=counts.columns)
    return Dataset("toy", counts, meta, gene_id="symbol")


def test_log_cpm_columns_sum_to_million_before_log():
    out = log_cpm(toy_dataset().counts)
    assert np.allclose((2 ** out - 1).sum(), 1e6)


def test_robust_z_zero_spread():
    assert (robust_z(pd.Series([1.0, 1.0, 1.0])) == 0).all()


def test_knn_flags_isolated_sample_not_tight_cluster():
    # two tight clusters (a batch split) plus one isolated sample
    pts = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [10, 10], [10.1, 10], [10, 10.1], [10.1, 10.1], [30, -30]]
    scores = pd.DataFrame(pts, columns=["PC1", "PC2"], index=[f"s{i}" for i in range(len(pts))])
    z = robust_z(knn_distance(scores, k=3))
    assert z.idxmax() == "s8"
    assert (z.drop("s8") < 3.5).all()


def test_parquet_roundtrip(tmp_path):
    ds = toy_dataset()
    ds.to_parquet(tmp_path)
    back = Dataset.from_parquet(tmp_path, "toy")
    pd.testing.assert_frame_equal(back.counts, ds.counts)
    pd.testing.assert_frame_equal(back.meta, ds.meta)
    assert (back.gene_id, back.is_count) == ("symbol", True)


@pytest.mark.parametrize("name", DATASETS)
def test_processed_cache_matches_spec_sizes(name):
    cfg = load_config()
    if not (cfg["paths"]["processed"] / f"{name}_info.json").is_file():
        pytest.skip("run scripts/01_load_all.py first")
    expected = {"walker": 230, "hahn": 847, "heroin": 305, "rat_reward": 93, "human_cud": 89}
    ds = load_processed(name, cfg)
    assert ds.counts.shape[1] == expected[name]
