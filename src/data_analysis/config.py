"""Load config.yaml and resolve its paths against the project root."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"
PATH_SECTIONS = ("paths", "data")


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until a directory holding config.yaml is found.

    The DATA_ANALYSIS_ROOT environment variable overrides the search.
    """
    env_root = os.environ.get("DATA_ANALYSIS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    start = (start or Path(__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / CONFIG_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"{CONFIG_NAME} not found above {start}")


def _resolve_paths(node: Any, root: Path) -> Any:
    """Turn every string in a path section into an absolute Path, except *_glob patterns."""
    if isinstance(node, dict):
        return {
            key: value if key.endswith("_glob") else _resolve_paths(value, root)
            for key, value in node.items()
        }
    if isinstance(node, str):
        return (root / node).resolve()
    return node


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return the parsed config with `root` added and path sections resolved to Path objects."""
    root = find_project_root()
    with open(root / CONFIG_NAME, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for section in PATH_SECTIONS:
        cfg[section] = _resolve_paths(cfg[section], root)
    cfg["root"] = root
    return cfg


def iter_data_files(cfg: dict[str, Any]) -> list[Path]:
    """List every input file declared under `data` (directories are skipped)."""
    files: list[Path] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, Path) and node.suffix:
            files.append(node)

    walk(cfg["data"])
    return files


def missing_data_files(cfg: dict[str, Any]) -> list[Path]:
    """Return declared input files that do not exist on disk."""
    return [path for path in iter_data_files(cfg) if not path.is_file()]


def ensure_output_dirs(cfg: dict[str, Any]) -> None:
    """Create processed/tables/figures output directories if missing."""
    for key in ("processed", "tables", "figures"):
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
