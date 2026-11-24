"""Config loading. One YAML per run; nothing is passed as a CLI flag."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict[str, Any]

    @property
    def run_name(self) -> str:
        return self.data["run_name"]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def load(path: str | Path) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    stem = path.stem
    if data.get("run_name") != stem:
        raise ValueError(f"run_name {data.get('run_name')!r} != filename stem {stem!r}")
    return Config(path=path, data=data)


def provenance(config: Config) -> dict[str, Any]:
    """Everything needed to reproduce this run, written into the run directory."""
    import platform
    import sys

    import numpy
    import pandas
    import pyarrow

    return {
        "run_name": config.run_name,
        "config_path": str(config.path),
        "resolved_config": config.data,
        "git_sha": _git_sha(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "pyarrow": pyarrow.__version__,
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "datasketch": _version("datasketch"),
            "transformers": _version("transformers"),
        },
    }


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _version(package: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
