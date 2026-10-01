#!/usr/bin/env python
"""Copy an experiment's small artifacts into experiments/<ID>/ with sha256 checksums.

    python scripts/registry/import_hf_artifacts.py E01_gazette_verbis_mix
    python scripts/registry/import_hf_artifacts.py E00_base

For a trained experiment, the files are downloaded from its PRIVATE results repo at the
revision recorded in experiments/registry.csv (hf_revision); the originals stay where they
are. For E00_base (evaluated inside E01), the base-model rows are filtered out of E01's
copied result files. Every copy is listed in experiments/<ID>/SOURCES.json with its origin
and sha256; existing copies are never overwritten with different content. Each copied text
file is scanned for Verbis entry text before it is kept.

New runs of the training script do this themselves at the end (step 13); this script is
for experiments that finished before that existed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mksq import experiment as X  # noqa: E402

EXPERIMENTS = X.ROOT / "experiments"
SMALL = {  # experiments/<ID>/<path> <- <repo>/<path>   (same mapping as the training script's step 13)
    "run_info.json": "eval/run_info.json", "train_log.csv": "eval/train_log.csv", "dev_curve.csv": "eval/dev_curve.csv",
    "progress.log": "progress.log",
    "results/gazette_test_results.csv": "eval/gazette_test_results.csv", "results/flores_results.csv": "eval/flores_results.csv",
    "results/ntrex_results.csv": "eval/ntrex_results.csv",
    "analysis/gazette_test_overlap_results.csv": "eval/gazette_test_overlap_results.csv",
    "analysis/terminology_results.csv": "eval/terminology_results.csv",
    "analysis/number_fidelity_results.csv": "eval/number_fidelity_results.csv",
}


def verbis_strings() -> set[str]:
    """Verbis entry texts (>= 10 characters) from the private repo, to make sure none is copied into git."""
    man = X.load_manifest("verbis")
    out: set[str] = set()
    for f in man["files"]:
        df = pl.read_parquet(hf_hub_download(man["storage_repo"], f["path"], revision=man["storage_revision"]))
        for c in df.columns:
            if df[c].dtype == pl.Utf8:
                out |= {s.strip().lower() for s in df[c].drop_nulls().to_list() if len(s.strip()) >= 10}
    return out


def place(src: Path, dest: Path) -> str:
    sha = X.sha256_file(src)
    if dest.exists():
        if X.sha256_file(dest) != sha:
            sys.exit(f"ERROR: {dest} exists with different content -- not overwritten")
        return sha
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    assert X.sha256_file(dest) == sha
    return sha


def import_trained(exp_id: str) -> None:
    row, cfg = X.registry_row(exp_id), X.load_config(exp_id)
    repo, rev = cfg["outputs"]["hf_results_repo"], row["hf_revision"]
    if not X.recorded(rev) or rev == "pending":
        sys.exit(f"ERROR: {exp_id} has no recorded hf_revision in experiments/registry.csv")
    if HfApi().model_info(repo).private is not True:
        sys.exit(f"ERROR: {repo} is not private")
    dest = EXPERIMENTS / exp_id
    banned = verbis_strings()
    files = [{"path": "config.yaml", "source": f"configs/{exp_id}.yaml",
              "sha256": place(X.config_path(exp_id), dest / "config.yaml")}]
    for rel_dest, remote in SMALL.items():
        local = Path(hf_hub_download(repo, remote, revision=rev))
        entry = {"path": rel_dest, "source": f"hf://{repo}/{remote}", "hf_revision": rev}
        if rel_dest == "progress.log":  # example translations quote corpus sentences (they contain Verbis terms)
            redacted, n = X.redact_log_examples(local.read_text(encoding="utf-8"))
            tmp = Path(tempfile.mkdtemp()) / "progress.log"
            tmp.write_text(redacted, encoding="utf-8")
            entry.update(source_sha256=X.sha256_file(local), redacted=f"{n} example-translation records removed "
                         "(mksq.experiment.redact_log_examples); all other lines verbatim")
            local = tmp
        text = local.read_text(encoding="utf-8", errors="replace").lower()
        hits = [s for s in banned if s in text]
        if hits:
            sys.exit(f"ERROR: {remote} contains {len(hits)} Verbis entry strings -- not copied")
        entry["sha256"] = place(local, dest / rel_dest)
        files.append(entry)
        print(f"  {rel_dest:<45} <- {remote}")
    (dest / "SOURCES.json").write_text(json.dumps({
        "exp_id": exp_id, "copied_from": f"hf://{repo}@{rev}", "originals_unchanged": True,
        "verbis_content_scan": f"{len(banned)} Verbis strings checked, none present", "files": files,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {(dest / 'SOURCES.json').relative_to(X.ROOT)} ({len(files)} files)")


def derive_base(exp_id: str = "E00_base", parent: str = "E01_gazette_verbis_mix") -> None:
    """E00's results = the base-model rows of E01's result and analysis files."""
    src_dir, dest = EXPERIMENTS / parent, EXPERIMENTS / exp_id
    files = []
    for rel_path in [p for p in SMALL if p.startswith(("results/", "analysis/"))]:
        src = src_dir / rel_path
        df = pl.read_csv(src)
        base = df.filter(pl.col("model") == "base")
        if "bleu_delta_vs_base" in base.columns:
            base = base.drop("bleu_delta_vs_base", "chrf_delta_vs_base")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / src.name
            base.write_csv(out)
            sha = place(out, dest / rel_path)
        files.append({"path": rel_path, "source": str(src.relative_to(X.ROOT)), "source_sha256": X.sha256_file(src),
                      "filter": "model == 'base'" + (" (delta columns dropped)" if "results/" in rel_path else ""),
                      "rows": base.height, "sha256": sha})
        print(f"  {rel_path:<45} <- {src.relative_to(X.ROOT)} ({base.height} rows)")
    (dest / "SOURCES.json").write_text(json.dumps({
        "exp_id": exp_id, "derived_from": parent,
        "note": "E00 was evaluated inside E01's final evaluation (same test sets, same decoding).",
        "ci_source": f"experiments/{parent}/results.json (significance.base_vs_finetuned_hf.*.baseline)",
        "files": files}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {(dest / 'SOURCES.json').relative_to(X.ROOT)}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/registry/import_hf_artifacts.py <EXPERIMENT_ID>")
    exp_id = sys.argv[1]
    X.validate_id(exp_id)
    if exp_id == "E00_base":
        derive_base()
    else:
        import_trained(exp_id)


if __name__ == "__main__":
    main()
