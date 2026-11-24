#!/usr/bin/env python
"""(Re)build the committed dataset manifests in data/manifests/ from their primary sources.

    python scripts/registry/build_manifests.py

Reads only: Hugging Face file metadata (LFS sha256) at pinned revisions, the NTREX files at a
pinned GitHub commit, and the E01 run record in experiments/E01_gazette_verbis_mix/results.json.
Nothing in a manifest is estimated: facts the sources do not contain are "NOT RECORDED".
Files that exist only on the GPU box (work/data/processed/*) are listed with sha256
"NOT RECORDED"; fill them with scripts/registry/hash_work_dir.py (read-only) run there.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mksq.experiment import MANIFESTS, NOT_RECORDED, ROOT  # noqa: E402

GAZETTE = ("EdonFetaji/slvesnik-mk-sq", "07af5bbbb2a2c5277e3eb9c032a86682148fb269")
FLORES = ("openlanguagedata/flores_plus", "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06")
NTREX = ("MicrosoftTranslator/NTREX", "468c6b69c7f6a75d31d4743d9daba2af566cc18d")
VERBIS_REPO = "EdonFetaji/mk-sq-nllb600m-gazette-verbis"
E01 = ROOT / "experiments" / "E01_gazette_verbis_mix"
UNTRACKED = "UNTRACKED"  # by decision: external public test sets are pinned by revision, their file hashes are not tracked
UNTRACKED_NOTE = ("File hashes deliberately not tracked for this external test set (project decision); the dataset is "
                  "pinned by revision instead.")
WORK_ONLY = "NOT RECORDED (file exists only in work/ on the GPU box; run scripts/registry/hash_work_dir.py there)"


def hf_files(api: HfApi, repo: str, repo_type: str, revision: str, paths: list[str], track: bool = True) -> list[dict]:
    out = []
    for p in api.get_paths_info(repo, paths, repo_type=repo_type, revision=revision, expand=True):
        if not track:
            out.append({"path": p.path, "size_bytes": p.size, "sha256": UNTRACKED})
            continue
        lfs = getattr(p, "lfs", None)
        if lfs:
            digest = lfs.sha256
        else:  # a regular git file: hash the bytes at this revision
            local = hf_hub_download(repo, p.path, repo_type=repo_type, revision=revision)
            digest = hashlib.sha256(Path(local).read_bytes()).hexdigest()
        out.append({"path": p.path, "size_bytes": p.size, "sha256": digest,
                    "git_blob_id": getattr(p, "blob_id", None),
                    "last_commit": getattr(getattr(p, "last_commit", None), "oid", None)})
    return out


def write(name: str, obj: dict) -> None:
    path = MANIFESTS / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    R = json.loads((E01 / "results.json").read_text(encoding="utf-8"))
    d = R["data"]
    first_run = R["runtime"]["invocations"][0]["started"]

    g_info = api.dataset_info(GAZETTE[0], revision=GAZETTE[1])
    g_files = [f for f in api.list_repo_files(GAZETTE[0], repo_type="dataset", revision=GAZETTE[1])
               if f.endswith(".parquet")]
    write("gazette", {
        "dataset": "gazette", "name": "Sluzben vesnik MK-SQ legal parallel corpus",
        "source": f"https://huggingface.co/datasets/{GAZETTE[0]}", "access": "public",
        "licence": (g_info.card_data or {}).get("license", NOT_RECORDED),
        "revision": GAZETTE[1],
        "revision_note": (f"Pinned for E02 onwards. E01 did not record the revision it loaded; the repo HEAD "
                          f"({GAZETTE[1][:7]}, last modified {g_info.last_modified}) predates E01's first "
                          f"invocation ({first_run} UTC)."),
        "revision_used_by": {"E01_gazette_verbis_mix": NOT_RECORDED},
        "row_counts": {"train": d["gazette"]["train"], "validation": d["gazette"]["dev"], "test": d["gazette"]["test"]},
        "split_method": "issue-disjoint",
        "issue_overlap": d["gazette"]["issue_overlap"],
        "files": hf_files(api, GAZETTE[0], "dataset", GAZETTE[1], g_files),
        "derived_files_as_used_by_E01": [
            {"path": f"work/data/processed/{n}", "sha256": WORK_ONLY}
            for n in ("gazette_train.parquet", "gazette_dev.parquet", "gazette_test.parquet", "train_clean.parquet",
                      "train_bidirectional.parquet", "leakage_report.json")],
        "produced_by": ["scripts/data/prepare_corpus.py (configs/prepare_corpus_v1.yaml)", "scripts/data/repair_corpus.py",
                        "scripts/data/prune_columns.py", "scripts/data/redact_corpus.py", "scripts/data/publish_hf.py",
                        "gazzette-parallel-corpus/01_*.ipynb"],
        "known_issues": [
            "Formulaic text: after digit normalisation, "
            f"{R['overlap_summary']['pct_template_or_near_duplicate']}% of test pairs are exact templates or 5-gram "
            "near duplicates of training sentences despite the issue-disjoint split (E01 overlap audit).",
            f"{d['leakage_removed']['gazette']} training pairs share an exact (lowercased, whitespace-normalised) side "
            "with dev/test/FLORES+/NTREX and are removed by the training script before training.",
            "OCR-derived text; some segments contain OCR artefacts.",
        ],
    })

    v_paths = ["data/verbis/verbis_mk_sq.parquet", "data/verbis/verbis_terms.parquet",
               "data/verbis/verbis_defs_scored.parquet", "data/verbis/verbis_defs.parquet"]
    v_rev = api.model_info(VERBIS_REPO).sha
    write("verbis", {
        "dataset": "verbis", "name": "Verbis MK-SQ terminology dictionary (APJ)",
        "source": d["verbis"]["source_api"], "publisher": d["verbis"]["publisher"],
        "access": f"private (stored only in hf://{VERBIS_REPO}/data/verbis/, a private repo)",
        "licence": "Redistribution not granted: requires the Agency's permission (scripts/data/verbis_harvest.py). "
                   "Never commit Verbis data or example entries.",
        "crawl_date": NOT_RECORDED, "categories_crawled": NOT_RECORDED,
        "storage_repo": VERBIS_REPO, "storage_revision": v_rev,
        "row_counts": {"entries": d["verbis"]["entries"], "term_pairs": d["verbis"]["term_pairs_kept"],
                       "definition_pairs_scored": d["verbis"]["definition_pairs_scored"],
                       "definition_pairs_kept": d["verbis"]["definition_pairs_kept"]},
        "filters": {"labse_model": "sentence-transformers/LaBSE", "labse_min_score": d["verbis"]["labse_threshold"]},
        "files": hf_files(api, VERBIS_REPO, "model", v_rev, v_paths),
        "local_cache": {"path": "data/verbis_mk_sq.parquet", "note": "downloaded from storage_repo by the training script"},
        "produced_by": ["scripts/data/verbis_harvest.py (raw entries; its docstring names it verbis_dataset.py)",
                        "scripts/train/train_nllb_gazette_verbis_ct2.py step 4 (term/definition pairs, LaBSE scores)"],
        "known_issues": [
            "Redistribution not granted; stored privately.",
            "Homoglyph repair applied at harvest: each field forced into a single script (Cyrillic/Latin).",
            "Crawl date and categories were not recorded; the local harvest cache (scripts/verbis_cache/, "
            f"{d['verbis']['local_cache_aggregate']['unique_entries']} entries) does not match the "
            f"{d['verbis']['entries']} entries used, so it cannot establish them.",
            "verbis_harvest.py's docstring states an entry total that differs from the dataset used.",
            "No standalone audit script exists; cleaning and LaBSE scoring happen in the training script's step 4.",
        ],
    })

    f_info = api.dataset_info(FLORES[0], revision=FLORES[1])
    write("flores_plus", {
        "dataset": "flores_plus", "name": "FLORES+ devtest (mkd_Cyrl / als_Latn)",
        "source": f"https://huggingface.co/datasets/{FLORES[0]}", "access": "gated (accept terms on Hugging Face)",
        "licence": (f_info.card_data or {}).get("license", NOT_RECORDED), "revision": FLORES[1],
        "revision_note": (f"Pinned for E02 onwards. E01 did not record the revision; the repo HEAD "
                          f"(last modified {f_info.last_modified}) predates E01's first invocation."),
        "revision_used_by": {"E01_gazette_verbis_mix": NOT_RECORDED},
        "row_counts": {"devtest": d["flores"]["sentences"]},
        "files": hf_files(api, FLORES[0], "dataset", FLORES[1], ["devtest/mkd_Cyrl.jsonl", "devtest/als_Latn.jsonl"],
                          track=False),
        "hash_policy": UNTRACKED_NOTE,
        "produced_by": ["scripts/train/train_nllb_gazette_verbis_ct2.py step 3 (joined on id)"],
        "known_issues": ["General domain (Wikimedia sources): measures general-domain retention, not legal quality."],
    })

    files = []
    for code in ("mkd", "sqi"):
        path = f"NTREX-128/newstest2019-ref.{code}.txt"
        data = urllib.request.urlopen(f"https://raw.githubusercontent.com/{NTREX[0]}/{NTREX[1]}/{path}", timeout=60).read()
        files.append({"path": path, "size_bytes": len(data), "sha256": UNTRACKED,
                      "lines": data.decode("utf-8-sig").rstrip("\n").count("\n") + 1})
    lic = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{NTREX[0]}/license", timeout=30))
    write("ntrex", {
        "dataset": "ntrex", "name": "NTREX-128 (mkd / sqi)", "source": f"https://github.com/{NTREX[0]}",
        "access": "public", "licence": lic["license"]["spdx_id"], "revision": NTREX[1],
        "revision_note": "Pinned for E02 onwards. E01 downloaded from branch main without recording the commit.",
        "revision_used_by": {"E01_gazette_verbis_mix": "NOT RECORDED (branch main)"},
        "row_counts": {"test": d["ntrex"]["sentences"]}, "files": files, "hash_policy": UNTRACKED_NOTE,
        "produced_by": ["scripts/train/train_nllb_gazette_verbis_ct2.py step 3 (line i of mkd aligned with line i of sqi)"],
        "known_issues": ["News domain: measures general-domain retention, not legal quality."],
    })


if __name__ == "__main__":
    main()
