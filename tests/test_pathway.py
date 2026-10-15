import importlib.util
from pathlib import Path

import pytest

from data_analysis.config import load_config

_path = Path(__file__).resolve().parents[1] / "scripts" / "11_pathway_overlap.py"
_spec = importlib.util.spec_from_file_location("pathway_overlap", _path)
pathway = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pathway)


def test_read_gmt(tmp_path):
    gmt = tmp_path / "toy.gmt"
    gmt.write_text("SET_A\thttp://x\tG1\tG2\nSET_B\tdesc\tG3\n", encoding="utf-8")
    assert pathway.read_gmt(gmt) == {"SET_A": ["G1", "G2"], "SET_B": ["G3"]}


def test_design_labels_match_cas_analysis():
    assert pathway.design_of("C24 vs S24") == "24 h withdrawal"
    assert pathway.design_of("HH vs SS") == "30 d + drug (relapse)"
    assert pathway.design_of("CS vs SS") == "30 d + context"
    assert pathway.design_of("SH vs SS") == "first acute exposure"


def test_prespecified_list_is_four_categories_without_duplicates():
    names = [p for ps in pathway.PATHWAYS.values() for p in ps]
    assert len(pathway.PATHWAYS) == 4
    assert len(names) == len(set(names))
    assert set(pathway.RENAMED) <= set(names)


def test_prespecified_pathways_exist_in_downloaded_msigdb():
    folder = load_config()["paths"]["raw"] / pathway.MSIGDB
    if not folder.is_dir():
        pytest.skip("MSigDB files not downloaded")
    sets = {}
    for f in pathway.GMT_FILES:
        sets.update(pathway.read_gmt(folder / f))
    assert set(pathway.CATEGORY) <= set(sets)
