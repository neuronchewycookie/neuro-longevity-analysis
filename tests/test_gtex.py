import gzip

import pytest

from data_analysis.config import load_config
from data_analysis.loaders.gtex import FOLDER, donor_id, load_gtex, read_gct


def test_read_gct_strips_versions_and_sums_duplicates(tmp_path):
    path = tmp_path / "toy.gct.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("#1.2\n3\t2\n")
        fh.write("Name\tDescription\tGTEX-A-1\tGTEX-B-1\n")
        fh.write("ENSG01.5\tG1\t1\t2\n")
        fh.write("ENSG02.3\tG2\t3\t4\n")
        fh.write("ENSG02.3_PAR_Y\tG2\t1\t1\n")
    df = read_gct(path)
    assert list(df.index) == ["ENSG01", "ENSG02"]
    assert df.loc["ENSG02"].tolist() == [4, 5]


def test_donor_id():
    assert donor_id("GTEX-1117F-0011-R5a-SM-GI4VW") == "GTEX-1117F"


def test_real_gtex_layout():
    cfg = load_config()
    if not (cfg["paths"]["raw"] / FOLDER).is_dir():
        pytest.skip("GTEx files not downloaded")
    ds = load_gtex(cfg["paths"]["raw"])
    assert ds.meta["region"].value_counts().to_dict() == {"caudate": 300, "nucleus_accumbens": 285, "putamen": 254}
    assert ds.counts.index.str.startswith("ENSG").all()
    assert set(ds.meta["age"]) <= {25, 35, 45, 55, 65, 75}
