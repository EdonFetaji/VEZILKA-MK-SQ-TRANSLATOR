"""Experiment bookkeeping shared by the training, evaluation and registry scripts.

An experiment is identified by an ID of the form E{NN}[suffix]_{short_name}
(e.g. E02_gazette_only). Its frozen settings live in configs/<ID>.yaml, its row in
experiments/registry.csv, and every output file it writes carries the provenance
returned by provenance(): experiment ID, git commit, config sha256 and the sha256
of every dataset manifest in data/manifests/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
MANIFESTS = ROOT / "data" / "manifests"
REGISTRY = ROOT / "experiments" / "registry.csv"
NOT_RECORDED = "NOT RECORDED"
EXP_ID = re.compile(r"E\d{2}[a-z]?_[a-z0-9_]+")
HF_OWNER = "EdonFetaji"
HF_PREFIX = "mk-sq-nllb600m-"
PROVENANCE_COLUMNS = ("exp_id", "git_commit", "config_sha256", "manifest_gazette_sha256",
                      "manifest_verbis_sha256", "manifest_flores_plus_sha256", "manifest_ntrex_sha256")
# paths whose uncommitted changes would make a run unmappable to a commit
TRACKED_FOR_RUNS = ("scripts", "configs", "data/manifests", "src")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def content_sha256(rows) -> str:
    """Order-sensitive sha256 of rows of strings (e.g. (mk, sq) pairs): independent of file format and metadata."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(list(r), ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def hf_repo_for(exp_id: str) -> str:
    """EdonFetaji/mk-sq-nllb600m-<id in lowercase with hyphens>."""
    return f"{HF_OWNER}/{HF_PREFIX}{exp_id.lower().replace('_', '-')}"


def validate_id(exp_id: str) -> None:
    if not EXP_ID.fullmatch(exp_id):
        raise ValueError(f"experiment ID {exp_id!r} does not match E{{NN}}_{{short_name}} (e.g. E02_gazette_only)")


def config_path(exp_id: str) -> Path:
    return CONFIGS / f"{exp_id}.yaml"


def load_config(exp_id: str) -> dict[str, Any]:
    validate_id(exp_id)
    path = config_path(exp_id)
    if not path.exists():
        raise FileNotFoundError(f"no config for {exp_id}: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("exp_id") != exp_id:
        raise ValueError(f"{path} declares exp_id {cfg.get('exp_id')!r}, expected {exp_id!r}")
    return cfg


def registry_rows() -> list[dict[str, str]]:
    with open(REGISTRY, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def registry_row(exp_id: str) -> dict[str, str]:
    for row in registry_rows():
        if row["exp_id"] == exp_id:
            return row
    raise KeyError(f"{exp_id} is not registered in {REGISTRY.relative_to(ROOT)}")


def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def git_head() -> str:
    return git("rev-parse", "HEAD")


def uncommitted(paths=TRACKED_FOR_RUNS) -> list[str]:
    """Uncommitted (modified, staged or untracked, not ignored) files under the given paths."""
    # not git(): its .strip() would eat the leading status space of the first line (" M path" -> "M path")
    r = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", *paths],
                       capture_output=True, text=True, check=True)
    return [line[3:] for line in r.stdout.splitlines() if line.strip()]


def manifest_sha256s() -> dict[str, str]:
    return {p.stem: sha256_file(p) for p in sorted(MANIFESTS.glob("*.json"))}


def load_manifest(name: str) -> dict[str, Any]:
    return json.loads((MANIFESTS / f"{name}.json").read_text(encoding="utf-8"))


def provenance(exp_id: str, commit: str | None = None) -> dict[str, Any]:
    m = manifest_sha256s()
    return {"exp_id": exp_id, "git_commit": commit or git_head(),
            "config_path": str(config_path(exp_id).relative_to(ROOT)),
            "config_sha256": sha256_file(config_path(exp_id)),
            "manifest_sha256": m}


def provenance_columns(prov: dict[str, Any]) -> dict[str, str]:
    """The provenance as constant CSV columns (same names in every CSV an experiment writes)."""
    m = prov["manifest_sha256"]
    return {"exp_id": prov["exp_id"], "git_commit": prov["git_commit"], "config_sha256": prov["config_sha256"],
            "manifest_gazette_sha256": m.get("gazette", NOT_RECORDED),
            "manifest_verbis_sha256": m.get("verbis", NOT_RECORDED),
            "manifest_flores_plus_sha256": m.get("flores_plus", NOT_RECORDED),
            "manifest_ntrex_sha256": m.get("ntrex", NOT_RECORDED)}


def recorded(value: Any) -> Any:
    """None for "NOT RECORDED" (and empty) values, the value otherwise."""
    return None if value in (None, "", NOT_RECORDED) or str(value).startswith(NOT_RECORDED) else value


_TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| ")


def redact_log_examples(text: str) -> tuple[str, int]:
    """Drop example-translation records (source/output sentences) from a progress.log.

    A record is a timestamped line plus its continuation lines. Records whose message contains
    "SRC:" (the smoke-test and random-example printouts) quote corpus sentences, which contain
    Verbis terms; each is replaced by one placeholder line. Everything else is kept verbatim.
    """
    out, removed, skipping = [], 0, False
    for line in text.splitlines():
        if _TS.match(line):
            skipping = "SRC:" in line
            if skipping:
                removed += 1
                head = line.split(" | ")
                out.append(" | ".join(head[:3] + ["[example translations removed from this copy]"]))
                continue
        elif skipping:
            continue
        out.append(line)
    return "\n".join(out) + "\n", removed
