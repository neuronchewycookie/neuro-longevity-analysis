import pytest

from data_analysis.config import load_config
from data_analysis.signatures import (
    EXPECTED_DOWN,
    load_cas_signature,
    load_striatal_signature,
    split_directions,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cas(cfg):
    return load_cas_signature(
        cfg["data"]["hahn"]["supplement"],
        expected_n_up=cfg["cas"]["expected_n_up"],
        expected_n_down=cfg["cas"]["expected_n_down"],
    )


def test_cas_has_82_genes_75_up_7_down(cas):
    up, down = split_directions(cas)
    assert len(cas) == 82
    assert (len(up), len(down)) == (75, 7)
    assert set(down) == EXPECTED_DOWN


def test_cas_wrong_expectation_raises(cfg):
    with pytest.raises(ValueError, match="direction counts"):
        load_cas_signature(cfg["data"]["hahn"]["supplement"], expected_n_up=80, expected_n_down=2)


def test_striatal_signature_excluding_cas(cfg, cas):
    full = load_striatal_signature(cfg["data"]["hahn"]["supplement"])
    specific = load_striatal_signature(cfg["data"]["hahn"]["supplement"], exclude_genes=set(cas["gene"]))
    assert (full["padj"] < 0.05).all()
    assert len(specific) < len(full)
    assert not set(specific["gene"]) & set(cas["gene"])
