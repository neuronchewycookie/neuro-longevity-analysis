import pandas as pd

from data_analysis.deseq import prefilter


def test_prefilter_requires_min_count_in_enough_samples():
    counts = pd.DataFrame({"s1": [10, 0, 50], "s2": [12, 30, 0], "s3": [0, 0, 40]}, index=["a", "b", "c"])
    assert list(prefilter(counts, min_count=10, min_samples=2).index) == ["a", "c"]
