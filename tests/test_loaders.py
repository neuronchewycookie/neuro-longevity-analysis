import pytest

from data_analysis.config import load_config
from data_analysis.loaders import hahn, heroin, human_cud, rat_reward, walker


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def walker_ds(cfg):
    return walker.load_walker(cfg)


@pytest.fixture(scope="module")
def hahn_ds(cfg):
    return hahn.load_hahn(cfg)


def test_walker_shape_and_ids(walker_ds):
    assert walker_ds.counts.shape == (43346, 230)
    assert walker_ds.counts.index.str.startswith("ENSMUSG").all()
    assert (walker_ds.counts.to_numpy() >= 0).all()


def test_walker_groups(walker_ds):
    table = walker.validate(walker_ds)
    assert table.to_numpy().sum() == 230
    assert table.loc["VTA", "CS"] == 3
    # 24 h groups have no challenge, 30 d groups do
    meta = walker_ds.meta
    assert set(meta.loc[meta["group"].isin(["S24", "C24"]), "withdrawal"]) == {"24h"}
    assert set(meta.loc[meta["group"].isin(["SS", "SC", "CS", "CC"]), "withdrawal"]) == {"30d"}


def test_walker_region_names(walker_ds):
    pairs = set(zip(walker_ds.meta["region"], walker_ds.meta["region_name"]))
    assert ("Cpu", "Dorsal striatum") in pairs
    assert ("Hipp", "Ventral hippocampus") in pairs
    assert len(pairs) == 6


def test_hahn_shape_and_design(hahn_ds):
    assert hahn_ds.counts.shape == (21076, 847)
    assert len(hahn_ds.regions) == 15
    assert hahn_ds.meta.loc[hahn_ds.meta["sex"] == "Female", "age_months"].max() == 21
    assert hahn_ds.meta.loc[hahn_ds.meta["sex"] == "Male", "age_months"].max() == 28


def test_hahn_has_all_cas_genes(hahn_ds, cfg):
    from data_analysis.signatures import load_cas_signature

    cas = load_cas_signature(cfg["data"]["hahn"]["supplement"])
    assert set(cas["gene"]) <= set(hahn_ds.counts.index)


def test_subset_keeps_alignment(hahn_ds):
    cp = hahn_ds.subset(region="cp")
    assert len(cp.meta) == 58
    assert cp.counts.columns.equals(cp.meta.index)


@pytest.fixture(scope="module")
def heroin_ds(cfg):
    return heroin.load_heroin(cfg)


@pytest.fixture(scope="module")
def rat_ds(cfg):
    return rat_reward.load_rat_reward(cfg)


@pytest.fixture(scope="module")
def human_ds(cfg):
    return human_cud.load_human_cud(cfg)


def test_heroin_design(heroin_ds):
    table = heroin.validate(heroin_ds)
    assert table.to_numpy().sum() == 305
    assert heroin_ds.counts.index.str.startswith("ENSMUSG").all()
    assert set(heroin_ds.meta["walker_region"]) == set(walker.REGIONS)


def test_rat_design(rat_ds):
    table = rat_reward.validate(rat_ds)
    assert table.to_numpy().sum() == 92
    assert not rat_ds.is_count
    assert rat_ds.meta["gsm"].is_unique
    # the same animal ID in two files stays two samples
    assert {"male_morphine:Blue4", "male_sucrose:Blue4"} <= set(rat_ds.meta.index)


def test_rat_male_morphine_column_shift_fixed(rat_ds):
    mm = rat_ds.subset(file="male_morphine")
    assert len(mm.meta) == 23  # 24 columns in GEO, one exact duplicate dropped
    assert (mm.counts.to_numpy() >= 0).all()  # no gene names left in the value columns


def test_rat_controls_have_no_time_point(rat_ds):
    controls = rat_ds.meta[rat_ds.meta["group"].isin(["Saline", "Cue"])]
    assert controls["time_point"].isna().all()


def test_human_design(human_ds):
    table = human_cud.validate(human_ds)
    assert table.loc["Caudate"].to_dict() == {"Control": 20, "CUD": 25}
    assert table.loc["NAc"].to_dict() == {"Control": 19, "CUD": 25}
    assert human_ds.counts.index.str.startswith("ENSG").all()
    assert human_ds.meta[["age", "pmi"]].notna().all().all()


def test_human_inferred_subjects_pair_regions(human_ds):
    per_subject = human_ds.meta.groupby("subject_id")["region"].agg(sorted)
    assert len(per_subject) == 45
    assert (per_subject.map(len) == 2).sum() == 44
    # a donor never mixes diagnoses or ages across its two samples
    assert (human_ds.meta.groupby("subject_id")[["diagnosis", "age"]].nunique(dropna=False) == 1).all().all()


def test_rat_duplicate_sample_dropped(rat_ds):
    # GEO's Male_Morphine file stores 231 and 233 with identical values; only 231 is kept
    assert "male_morphine:231" in rat_ds.meta.index
    assert "male_morphine:233" not in rat_ds.meta.index
    assert not rat_ds.counts.T.duplicated().any()
    table = rat_reward.validate(rat_ds)
    assert table.loc["Morphine_D30", "Male"] == 7


def test_drop_duplicate_columns_keeps_first():
    import pandas as pd

    values = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [3.0, 4.0]})
    meta = pd.DataFrame({"region": ["x", "x", "x"]}, index=["a", "b", "c"])
    v, m = rat_reward.drop_duplicate_columns(values, meta)
    assert list(v.columns) == ["a", "c"] and list(m.index) == ["a", "c"]
