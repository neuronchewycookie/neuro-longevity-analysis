from pathlib import Path

from data_analysis.config import load_config, missing_data_files


def test_config_loads_with_resolved_paths():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert (cfg["root"] / "config.yaml").is_file()
    assert isinstance(cfg["data"]["walker"]["counts"], Path)
    assert cfg["data"]["walker"]["counts"].is_absolute()
    assert cfg["data"]["human"]["counts_glob"] == "GSM*.counts.txt"


def test_all_declared_data_files_exist():
    missing = missing_data_files(load_config())
    assert not missing, f"missing data files: {[str(p) for p in missing]}"


def test_human_count_files_present():
    cfg = load_config()
    human = cfg["data"]["human"]
    assert len(list(human["raw_dir"].glob(human["counts_glob"]))) == 89
