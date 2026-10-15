import gzip

import pytest

from data_analysis.config import load_config
from data_analysis.geo import normalize_key, parse_characteristic, parse_series_matrix

EXPECTED_SAMPLES = {"walker": 230, "hahn": 847, "heroin": 305, "rat": 93, "human": 89}


def test_normalize_key():
    assert normalize_key("age (months)") == "age_months"
    assert normalize_key("Self-Administration") == "self_administration"


def test_parse_characteristic():
    assert parse_characteristic("time point: Day 30") == ("time_point", "Day 30")
    assert parse_characteristic("") is None


def test_characteristics_parsed_by_key_not_row(tmp_path):
    # Different keys on the same row (as in GSE174737) must land in their own columns.
    text = "\n".join([
        '!Sample_title\t"a"\t"b"',
        '!Sample_geo_accession\t"GSM1"\t"GSM2"',
        '!Sample_characteristics_ch1\t"Sex: Male"\t"Sex: Female"',
        '!Sample_characteristics_ch1\t"time point: Day 1"\t"agent: Cue"',
        '!Sample_characteristics_ch1\t"agent: Morphine"\t""',
        "!series_matrix_table_begin",
    ])
    path = tmp_path / "toy_series_matrix.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(text)
    meta = parse_series_matrix(path)
    assert meta.loc["GSM1", "agent"] == "Morphine"
    assert meta.loc["GSM2", "agent"] == "Cue"
    assert meta["time_point"].isna().tolist() == [False, True]


@pytest.mark.parametrize("dataset,n", EXPECTED_SAMPLES.items())
def test_real_series_matrix_sample_counts(dataset, n):
    meta = parse_series_matrix(load_config()["data"][dataset]["series_matrix"])
    assert len(meta) == n


def test_hahn_sample_names_in_third_description():
    meta = parse_series_matrix(load_config()["data"]["hahn"]["series_matrix"])
    assert meta["description_3"].str.startswith("BulkSeq_").all()
    assert meta["description_3"].is_unique
