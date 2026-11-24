#!/usr/bin/env python
"""Build the experiment report for E01_gazette_verbis_mix from recorded artifacts only.

    .venv/bin/python experiments/E01_gazette_verbis_mix/build_report.py

Nothing is trained or translated. The script
  1. downloads every source artifact from the PRIVATE results repo, pinned to the
     revision it reads at start, and records size / sha256 / last commit (MANIFEST.json);
  2. parses the recorded facts (run_info.json, trainer_state.json, progress.log, DONE, CSVs);
  3. computes paired bootstrap significance on the cached translations
     (sacrebleu PairedTest, 1000 resamples, SACREBLEU_SEED=42);
  4. writes results.json, figures/*.png (from the CSVs), tables/*.tex (booktabs), REPORT.md;
  5. self-checks: every number in REPORT.md appears in results.json, and no Verbis
     entry text appears in any output file.
Missing facts are written as "NOT RECORDED".
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ["SACREBLEU_SEED"] = "42"  # read by sacrebleu.significance.PairedTest at construction

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402
from sacrebleu.significance import PairedTest  # noqa: E402

EXPERIMENT_ID = "E01_gazette_verbis_mix"
BASELINE_ID = "E00_base"
RESULTS_REPO = "EdonFetaji/mk-sq-nllb600m-gazette-verbis"
GAZETTE_REPO = "EdonFetaji/slvesnik-mk-sq"
FLORES_REPO = "openlanguagedata/flores_plus"
NTREX_REPO = "MicrosoftTranslator/NTREX"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
NR = "NOT RECORDED"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG, TAB = HERE / "figures", HERE / "tables"

HF_FILES = [
    "DONE", "progress.log", "config.json", "generation_config.json",
    "eval/run_info.json", "eval/train_log.csv", "eval/dev_curve.csv", "eval/training_curves.png",
    "eval/gazette_test_results.csv", "eval/flores_results.csv", "eval/ntrex_results.csv",
    "eval/gazette_test_overlap_results.csv", "eval/terminology_results.csv", "eval/number_fidelity_results.csv",
    "eval/translations_gazette_test.parquet", "eval/translations_flores.parquet", "eval/translations_ntrex.parquet",
    "eval/dev_eval_ids.parquet",
    "checkpoints/checkpoint-3000/trainer_state.json", "checkpoints/checkpoint-4000/trainer_state.json",
    "checkpoints/checkpoint-4115/trainer_state.json",
    "data/verbis/verbis_mk_sq.parquet", "data/verbis/verbis_terms.parquet",
    "data/verbis/verbis_defs_scored.parquet", "data/verbis/verbis_defs.parquet",
]
LOCAL_FILES = ["scripts/train_nllb_gazette_verbis_ct2.py", "scripts/analyze_gazette_results.py",
               "scripts/smoke_test_nllb_run.py", "scripts/verbis_harvest.py", "scripts/requirements-train.txt"]
TESTSETS = (("gazette_test", "Gazette test"), ("flores", "FLORES+ devtest"), ("ntrex", "NTREX-128"))
DIRECTIONS = (("mk_sq", "MK→SQ"), ("sq_mk", "SQ→MK"))
MODELS = (("base", "Base NLLB-600M (E00)"), ("finetuned_hf", "Fine-tuned, HF bf16 (E01)"),
          ("finetuned_ct2", "Fine-tuned, CT2 int8 (E01)"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


# ============================================================================
# 1. Fetch sources + manifest
# ============================================================================
def fetch(api: HfApi) -> tuple[dict, dict, list]:
    info = api.model_info(RESULTS_REPO)
    if info.private is not True:
        sys.exit(f"ERROR: {RESULTS_REPO} is not private -- refusing to read or cite it")
    rev = info.sha
    paths, manifest = {}, []
    last = {p.path: p for p in api.get_paths_info(RESULTS_REPO, HF_FILES, expand=True, revision=rev)}
    for f in HF_FILES:
        local = Path(hf_hub_download(RESULTS_REPO, f, revision=rev))
        paths[f] = local
        lc = getattr(last.get(f), "last_commit", None)
        manifest.append({"artifact": f, "location": f"hf://{RESULTS_REPO}/{f}", "hf_revision": rev,
                         "last_commit": getattr(lc, "oid", None),
                         "last_commit_date": str(getattr(lc, "date", None)) if lc else None,
                         "size_bytes": local.stat().st_size, "sha256": sha256(local)})
    git_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for f in LOCAL_FILES:
        p = ROOT / f
        log = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", f], capture_output=True,
                             text=True).stdout.strip()
        manifest.append({"artifact": f, "location": f"git:{f}", "git_head": git_head, "last_commit": log or None,
                         "size_bytes": p.stat().st_size, "sha256": sha256(p)})
    cache = sorted((ROOT / "scripts/verbis_cache").glob("*.json"))
    h = hashlib.sha256()
    for p in cache:
        h.update(p.name.encode())
        h.update(bytes.fromhex(sha256(p)))
    manifest.append({"artifact": "scripts/verbis_cache/ (28 API pages)", "location": "local:scripts/verbis_cache/",
                     "n_files": len(cache), "size_bytes": sum(p.stat().st_size for p in cache),
                     "sha256_of_name_and_file_hashes": h.hexdigest(), "tracked_in_git": False})
    # dataset revisions and licences (primary sources)
    g = api.dataset_info(GAZETTE_REPO)
    fl = api.dataset_info(FLORES_REPO)
    ntrex = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{NTREX_REPO}/commits/main", timeout=30))
    lic = json.load(urllib.request.urlopen(f"https://api.github.com/repos/{NTREX_REPO}/license", timeout=30))
    lic_text = urllib.request.urlopen(
        f"https://raw.githubusercontent.com/{NTREX_REPO}/{ntrex['sha']}/{lic['path']}", timeout=30).read()
    external = {
        "gazette": {"repo": GAZETTE_REPO, "current_sha": g.sha, "last_modified": str(g.last_modified),
                    "license": (g.card_data or {}).get("license")},
        "flores": {"repo": FLORES_REPO, "current_sha": fl.sha, "last_modified": str(fl.last_modified),
                   "license": (fl.card_data or {}).get("license")},
        "ntrex": {"repo": f"github.com/{NTREX_REPO}", "current_sha": ntrex["sha"],
                  "commit_date": ntrex["commit"]["committer"]["date"], "license": lic["license"]["spdx_id"],
                  "license_file": lic["path"]},
    }
    manifest += [
        {"artifact": "Gazette dataset card (licence, revision)", "location": f"hf://datasets/{GAZETTE_REPO}",
         "hf_revision": g.sha},
        {"artifact": "FLORES+ dataset card (licence)", "location": f"hf://datasets/{FLORES_REPO}",
         "hf_revision": fl.sha, "license": external["flores"]["license"]},
        {"artifact": f"NTREX {lic['path']}", "location": f"https://github.com/{NTREX_REPO}/blob/{ntrex['sha']}/{lic['path']}",
         "git_revision": ntrex["sha"], "size_bytes": len(lic_text), "sha256": hashlib.sha256(lic_text).hexdigest()},
    ]
    meta = {"results_repo": RESULTS_REPO, "results_repo_revision": rev,
            "results_repo_last_modified": str(info.last_modified), "report_git_head": git_head,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return paths, {"meta": meta, "external": external}, manifest


# ============================================================================
# 2. Parse recorded facts
# ============================================================================
def parse_progress(text: str) -> dict:
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if "| run started" in ln]
    final = lines[starts[-1]:] if starts else lines
    ts = lambda ln: ln[:19]  # noqa: E731
    steps = {}
    for ln in final:
        m = re.search(r"\| step +(\d+) (.+?) \| ===== done in (\d+)h(\d+)m(\d+)s", ln)
        if m:
            steps[int(m.group(1))] = {"name": m.group(2).strip(),
                                      "seconds": int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))}
    invocations = []
    for i in starts:
        seg_end = next((j for j in starts if j > i), len(lines))
        seg = lines[i:seg_end]
        failed = next((ln for ln in seg if "| ERROR" in ln and "FAILED" in ln), None)
        invocations.append({"started": ts(lines[i]), "outcome": "failed" if failed else "completed",
                            "error": failed.split("FAILED", 1)[1].strip(" :") if failed else None})
    throughput = {}
    for ln in final:
        m = re.search(r"\| (\w+) / (\w+) / (\w+): (\d+) / (\d+) sentences \(([\d.]+) sent/s", ln)
        if m and m.group(4) == m.group(5):
            throughput[f"{m.group(1)}/{m.group(2)}/{m.group(3)}"] = float(m.group(6))
    warnings_training = [ln for ln in final if "| WARNING" in ln and "step  8" in ln and "save_safetensors" not in ln]
    grab = lambda pat, cast=str: next((cast(m.group(1)) for ln in lines if (m := re.search(pat, ln))), None)  # noqa: E731
    ct2_fallback = any("CTranslate2 cannot use this GPU" in ln for ln in final)
    return {
        "invocations": invocations, "final_invocation_started": ts(lines[starts[-1]]) if starts else None,
        "final_invocation_last_line": ts(final[-1]) if final else None,
        "step_seconds": steps, "eval_throughput_sent_per_s": throughput,
        "training_warnings": len(warnings_training),
        "gazette_split_names": re.findall(r"Gazette split '(\w+)'", "\n".join(lines))[:3],
        "verbis_entries": grab(r"Verbis: (\d+) entries", int),
        "labse_median": grab(r"median score ([\d.]+)", float),
        "projection_s_per_step": grab(r"projection after 200 steps: ([\d.]+) s/step", float),
        "ct2_device_fallback_to_cpu": ct2_fallback,
        "base_model_snapshot": grab(r"models--facebook--nllb-200-distilled-600M/snapshots/([0-9a-f]{40})"),
        "tokens_dropped_line": grab(r"(train: \d+ -> \d+ examples \(\d+ dropped)"),
    }


def verbis_cache_aggregate() -> dict:
    ids, cats, created = set(), {}, []
    for p in (ROOT / "scripts/verbis_cache").glob("*.json"):
        for r in json.load(open(p, encoding="utf-8")).get("data", []):
            if r["id"] in ids:
                continue
            ids.add(r["id"])
            c = r.get("category")
            key = str(c.get("id")) if isinstance(c, dict) else str(c)
            cats[key] = cats.get(key, 0) + 1
            if r.get("created_at"):
                created.append(r["created_at"])
    return {"unique_entries": len(ids), "entries_per_category_id": dict(sorted(cats.items())),
            "entry_created_at_range": [min(created), max(created)] if created else None}


def issue_overlap(api: HfApi, revision: str) -> dict:
    """Issue-level overlap between Gazette test and train, from the dataset's own parquet files."""
    files = [f for f in api.list_repo_files(GAZETTE_REPO, repo_type="dataset", revision=revision) if f.endswith(".parquet")]
    keys = {}
    for split in ("train", "test"):
        parts = [f for f in files if f"/{split}-" in f or f.startswith(f"{split}-")]
        cols = [pl.read_parquet(hf_hub_download(GAZETTE_REPO, f, repo_type="dataset", revision=revision), columns=["issue_key"])
                for f in parts]
        keys[split] = set(pl.concat(cols)["issue_key"].drop_nulls().to_list())
    return {"revision": revision, "test_issues": len(keys["test"]), "train_issues": len(keys["train"]),
            "test_issues_in_train": len(keys["test"] & keys["train"])}


def csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
            except (ValueError, TypeError):
                pass
    return rows


# ============================================================================
# 3. Significance (paired bootstrap on cached translations)
# ============================================================================
def paired(baseline: tuple[str, list], system: tuple[str, list], refs: list) -> dict:
    out = {}
    test = PairedTest([baseline, system], metrics={"bleu": BLEU(), "chrf": CHRF(word_order=2)},
                      references=[refs], test_type="bs", n_samples=N_BOOTSTRAP)
    sigs, scores = test()
    names = [k for k in scores if k != "System"]  # sacrebleu keys results by metric display name, in metrics order
    for metric, name in zip(("bleu", "chrf"), names, strict=True):
        bl, sy = scores[name]
        for r in (bl, sy):  # sacrebleu returns numpy float32 for bootstrap statistics
            r.score, r.mean, r.ci = float(r.score), float(r.mean), float(r.ci)
            r.p_value = None if r.p_value is None else float(r.p_value)
        out[metric] = {
            "baseline": {"score": round(bl.score, 2), "bootstrap_mean": round(bl.mean, 2),
                         "ci95": [round(bl.mean - bl.ci, 2), round(bl.mean + bl.ci, 2)]},
            "system": {"score": round(sy.score, 2), "bootstrap_mean": round(sy.mean, 2),
                       "ci95": [round(sy.mean - sy.ci, 2), round(sy.mean + sy.ci, 2)]},
            "delta": round(sy.score - bl.score, 2), "p_value": round(sy.p_value, 4),
            "significant": bool(sy.p_value < ALPHA), "signature": str(sigs[name]),
        }
    return out


def significance(paths: dict) -> dict:
    res = {"base_vs_finetuned_hf": {}, "finetuned_hf_vs_finetuned_ct2": {}}
    for ts, _ in TESTSETS:
        tr = pl.read_parquet(paths[f"eval/translations_{ts}.parquet"])
        for d, _ in DIRECTIONS:
            blk = tr.filter(pl.col("direction") == d)
            refs = blk["reference"].to_list()
            print(f"  bootstrap {ts} {d} ({blk.height} sentences)", flush=True)
            res["base_vs_finetuned_hf"][f"{ts}/{d}"] = paired(("base", blk["base"].to_list()),
                                                              ("finetuned_hf", blk["finetuned_hf"].to_list()), refs)
            res["finetuned_hf_vs_finetuned_ct2"][f"{ts}/{d}"] = paired(
                ("finetuned_hf", blk["finetuned_hf"].to_list()), ("finetuned_ct2", blk["finetuned_ct2"].to_list()), refs)
    return res


# ============================================================================
# 4. Assemble results.json
# ============================================================================
def assemble(paths: dict, head: dict) -> dict:
    ri = json.loads(paths["eval/run_info.json"].read_text())
    st = json.loads(paths["checkpoints/checkpoint-4115/trainer_state.json"].read_text())
    prog = parse_progress(paths["progress.log"].read_text(encoding="utf-8"))
    ta, ds, probe, summ = ri["training_args"], ri["dataset_sizes"], ri["memory_probe"], ri["training_summary"]
    tl = pl.read_csv(paths["eval/train_log.csv"])
    main = {ts: csv_rows(paths[f"eval/{ts}_results.csv"]) for ts, _ in TESTSETS}
    cache = verbis_cache_aggregate()
    v_entries = pl.read_parquet(paths["data/verbis/verbis_mk_sq.parquet"]).height
    v_scored = pl.read_parquet(paths["data/verbis/verbis_defs_scored.parquet"])
    assert v_entries == prog["verbis_entries"], (v_entries, prog["verbis_entries"])  # log and data agree
    assert v_scored.height == ds["verbis_defs_scored"]
    prog["verbis_entries"] = v_entries
    prog["labse_median"] = round(float(v_scored["labse_score"].median()), 3)
    cache_matches = cache["unique_entries"] == prog["verbis_entries"]
    comp = ds["train_composition"]
    total = sum(comp.values())
    R = {
        "experiment": {"id": EXPERIMENT_ID, "baseline_id": BASELINE_ID,
                       "paper_arm": "M7-mix + D3", "baseline_arm": "M0"},
        **head,
        "code": {"training_commit": ri["git"]["commit"], "training_commit_dirty": ri["git"]["dirty"],
                 "report_git_head": head["meta"]["report_git_head"]},
        "data": {
            "gazette": {"repo": GAZETTE_REPO, "revision_used_by_run": NR,
                        "current_sha": head["external"]["gazette"]["current_sha"],
                        "current_last_modified": head["external"]["gazette"]["last_modified"],
                        "splits_loaded": prog["gazette_split_names"],
                        "train": ds["gazette_train"], "dev": ds["gazette_dev"], "test": ds["gazette_test"],
                        "dev_origin": "published 'validation' split" if "validation" in prog["gazette_split_names"] else NR},
            "verbis": {"source_api": "https://api.verbis.gov.mk/api/words", "publisher": "APJ (Agjencia e Zbatimit të Gjuhës)",
                       "entries": prog["verbis_entries"], "term_pairs_kept": ds["verbis_terms"],
                       "definition_pairs_scored": ds["verbis_defs_scored"], "definition_pairs_kept": ds["verbis_defs"],
                       "definition_keep_pct": round(100 * ds["verbis_defs"] / ds["verbis_defs_scored"], 1),
                       "labse_threshold": ri["constants"]["LABSE_MIN_SCORE"], "labse_median": prog["labse_median"],
                       "crawl_date": NR if not cache_matches else "see cache", "categories_crawled": NR if not cache_matches else cache["entries_per_category_id"],
                       "local_cache_aggregate": cache, "local_cache_matches_dataset": cache_matches,
                       "licence": "redistribution requires the Agency's permission (scripts/verbis_harvest.py); stored privately"},
            "flores": {"repo": FLORES_REPO, "split": "devtest", "sentences": ds["flores_devtest"],
                       "licence": head["external"]["flores"]["license"], "revision_used_by_run": NR,
                       "licence_source_revision": head["external"]["flores"]["current_sha"]},
            "ntrex": {"repo": head["external"]["ntrex"]["repo"], "sentences": ds["ntrex"],
                      "licence": head["external"]["ntrex"]["license"], "revision_used_by_run": NR + " (downloaded from branch main)",
                      "licence_source_revision": head["external"]["ntrex"]["current_sha"]},
            "leakage_removed": ds["leakage_removed"],
            "leakage_removed_total": sum(ds["leakage_removed"].values()),
            "gazette_train_after_leakage": ds["gazette_train"] - ds["leakage_removed"]["gazette"],
            "gazette_leakage_pct": round(100 * ds["leakage_removed"]["gazette"] / ds["gazette_train"], 2),
            "train_clean_pairs": ds["train_clean_pairs"], "train_bidirectional": ds["train_bidirectional"],
            "length_filter_max_tokens": ri["constants"]["MAX_TOKENS"],
            "length_filter_dropped": ds["train_dropped_over_max_tokens"],
            "length_filter_dropped_pct": round(100 * ds["train_dropped_over_max_tokens"] / ds["train_bidirectional"], 2),
            "train_tokenized": ds["train_tokenized"],
            "composition_examples": comp,
            "composition_pct": {k: round(100 * v / total, 1) for k, v in comp.items()},
            "direction_split_examples": ds["train_bidirectional"] // 2,
        },
        "training": {
            "base_model": ri["constants"]["BASE_MODEL"], "base_model_snapshot": prog["base_model_snapshot"] or NR,
            "method": "full-parameter supervised fine-tuning, bidirectional (MK→SQ and SQ→MK in one model)",
            "optimizer": ta["optim"], "learning_rate": ta["learning_rate"], "lr_scheduler": ta["lr_scheduler_type"],
            "warmup_steps": ta["warmup_steps"], "epochs": ta["num_train_epochs"],
            "label_smoothing": ta["label_smoothing_factor"], "weight_decay": ta["weight_decay"],
            "max_grad_norm": ta["max_grad_norm"], "bf16": ta["bf16"], "tf32": ta["tf32"],
            "per_device_batch": ta["per_device_train_batch_size"], "grad_accumulation": ta["gradient_accumulation_steps"],
            "effective_batch": ta["per_device_train_batch_size"] * ta["gradient_accumulation_steps"],
            "gradient_checkpointing": ta["gradient_checkpointing"], "sampling": ta.get("train_sampling_strategy", NR),
            "seed": ta["seed"], "data_seed": ta["data_seed"], "eval_steps": ta["eval_steps"],
            "save_total_limit": ta["save_total_limit"], "metric_for_best_model": ta["metric_for_best_model"],
            "probe_attempts": probe["attempts"], "probe_peak_gb": probe["peak_gb"],
            "total_steps": st["global_step"], "max_steps": st["max_steps"], "total_flos": st["total_flos"],
            "best_step": st["best_global_step"], "best_dev_chrf_mean": round(st["best_metric"], 2),
            "training_time_h": summ["total_training_time_h"],
            "training_step_seconds": prog["step_seconds"].get(8, {}).get("seconds"),
            "projection_s_per_step": prog["projection_s_per_step"],
            "train_log_rows": tl.height,
            "samples_per_sec_median": round(float(tl["samples_per_sec"].median()), 1),
            "tokens_per_sec_median": round(float(tl["tokens_per_sec"].median()), 0),
            "gpu_mem_peak_gb": float(tl["gpu_mem_peak_gb"].max()),
            "gpu_temp_max_c": float(tl["gpu_temp_c"].max()), "gpu_power_max_w": float(tl["gpu_power_w"].max()),
            "loss_first_logged": float(tl["loss"][0]), "loss_last_logged": float(tl["loss"][-1]),
            "training_warnings": prog["training_warnings"],
            "dev_curve": csv_rows(paths["eval/dev_curve.csv"]),
            "dev_eval_size_per_direction": ri["constants"]["DEV_EVAL_SIZE"],
        },
        "environment": {"hardware": {"gpu": ri["gpu"]["gpu_name"], "vram_total_gb": ri["gpu"]["vram_total_gb"],
                                     "compute_capability": ri["gpu"]["compute_capability"],
                                     "driver": ri["gpu"]["driver"], "cpu": NR, "ram": NR},
                        "versions": ri["versions"], "cuda_torch": ri["gpu"]["cuda_torch"],
                        "package_manager": NR, "torch_stated_in_brief": "2.11+cu128", "ct2_device": "cpu (int8)" if prog["ct2_device_fallback_to_cpu"] else "cuda"},
        "runtime": {"invocations": prog["invocations"], "final_invocation_started": prog["final_invocation_started"],
                    "final_invocation_runtime": re.search(r"runtime of this invocation: (\S+)",
                                                          paths["DONE"].read_text()).group(1),
                    "step_seconds": prog["step_seconds"], "eval_throughput_sent_per_s": prog["eval_throughput_sent_per_s"],
                    "analysis_script_runtime": NR},
        "evaluation": {"beams": ri["constants"]["FINAL_BEAMS"], "max_new_tokens": ri["constants"]["MAX_NEW_TOKENS"],
                       "bleu_signature": main["gazette_test"][0]["bleu_signature"],
                       "chrf_signature": main["gazette_test"][0]["chrf_signature"]},
        "results_main": {ts: [{k: r[k] for k in ("model", "direction", "n_sentences", "bleu", "chrf",
                                                 "bleu_delta_vs_base", "chrf_delta_vs_base")} for r in rows]
                         for ts, rows in main.items()},
        "overlap": csv_rows(paths["eval/gazette_test_overlap_results.csv"]),
        "terminology": csv_rows(paths["eval/terminology_results.csv"]),
        "number_fidelity": csv_rows(paths["eval/number_fidelity_results.csv"]),
        "interpolation": "not run yet",
        "analysis_constants": {"normalisation": "lowercase, digits -> 0, whitespace collapsed",
                               "shingle_chars": 5, "near_dup_jaccard": 0.8, "minhash_perms": 128, "lsh_bands": 16,
                               "min_term_chars": 5, "rare_term_max_train_sentences": 5, "prefix_min_chars": 4,
                               "prefix_trim_chars": 2, "bootstrap_resamples": N_BOOTSTRAP,
                               "bootstrap_seed": BOOTSTRAP_SEED, "alpha": ALPHA,
                               "ci_level_pct": 95},
    }
    return R


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    print("fetching sources ...", flush=True)
    paths, head, manifest = fetch(api)
    R = assemble(paths, head)
    R["data"]["gazette"]["issue_overlap"] = issue_overlap(api, head["external"]["gazette"]["current_sha"])
    print("paired bootstrap ...", flush=True)
    R["significance"] = significance(paths)
    for key, blk in R["significance"]["base_vs_finetuned_hf"].items():  # consistency with the recorded CSVs
        ts, d = key.split("/")
        rec = next(r for r in R["results_main"][ts] if r["model"] == "finetuned_hf" and r["direction"] == d)
        assert abs(blk["bleu"]["system"]["score"] - rec["bleu"]) < 0.011, (key, blk["bleu"], rec)
        assert abs(blk["chrf"]["system"]["score"] - rec["chrf"]) < 0.011, (key, blk["chrf"], rec)
    (HERE / "MANIFEST.json").write_text(json.dumps({"generated_at": R["meta"]["generated_at"],
                                                    "artifacts": manifest}, indent=2, ensure_ascii=False) + "\n")
    render(R, paths)
    self_check(R, paths)
    print(f"done: {rel(HERE / 'REPORT.md')}")


# ============================================================================
# 5. Derived summaries (all from the recorded CSVs / results)
# ============================================================================
def derive(R: dict) -> None:
    ov = R["overlap"]
    shares = {c: next(r for r in ov if r["class"] == c) for c in ("exact_template", "near_duplicate", "novel")}
    R["overlap_summary"] = {
        "n_pairs": {c: r["n_pairs"] for c, r in shares.items()},
        "pct_pairs": {c: r["pct_pairs"] for c, r in shares.items()},
        "pct_template_or_near_duplicate": round(shares["exact_template"]["pct_pairs"] + shares["near_duplicate"]["pct_pairs"], 2),
    }
    delta = {}
    for c in ("exact_template", "near_duplicate", "novel", "all"):
        for d, _ in DIRECTIONS:
            b = next(r for r in ov if r["class"] == c and r["model"] == "base" and r["direction"] == d)
            f = next(r for r in ov if r["class"] == c and r["model"] == "finetuned_hf" and r["direction"] == d)
            if c == "all":  # the recorded deltas (computed from unrounded scores) in eval/gazette_test_results.csv
                m = next(x for x in R["results_main"]["gazette_test"] if x["model"] == "finetuned_hf" and x["direction"] == d)
                delta[f"{c}/{d}"] = {"bleu": m["bleu_delta_vs_base"], "chrf": m["chrf_delta_vs_base"]}
            else:
                delta[f"{c}/{d}"] = {"bleu": round(f["bleu"] - b["bleu"], 2), "chrf": round(f["chrf"] - b["chrf"], 2)}
    R["overlap_summary"]["finetuned_hf_minus_base"] = delta
    tsr = {}
    for d, _ in DIRECTIONS:
        row = {r["model"]: r for r in R["terminology"] if r["direction"] == d}
        tsr[d] = {"finetuned_hf_minus_reference": round(row["finetuned_hf"]["tsr"] - row["reference"]["tsr"], 2),
                  "rare_finetuned_hf_minus_reference": round(row["finetuned_hf"]["rare_tsr"] - row["reference"]["rare_tsr"], 2),
                  "finetuned_hf_minus_base": round(row["finetuned_hf"]["tsr"] - row["base"]["tsr"], 2),
                  "rare_finetuned_hf_minus_base": round(row["finetuned_hf"]["rare_tsr"] - row["base"]["rare_tsr"], 2)}
    R["terminology_summary"] = tsr
    sig = R["significance"]["base_vs_finetuned_hf"]
    ext = {k: v for k, v in sig.items() if not k.startswith("gazette_test")}
    R["forgetting_summary"] = {
        "n_tests": 2 * len(ext), "n_significant_declines": sum(v[m]["significant"] and v[m]["delta"] < 0
                                                               for v in ext.values() for m in ("bleu", "chrf")),
        "bleu_delta_range": [min(v["bleu"]["delta"] for v in ext.values()), max(v["bleu"]["delta"] for v in ext.values())],
        "chrf_delta_range": [min(v["chrf"]["delta"] for v in ext.values()), max(v["chrf"]["delta"] for v in ext.values())],
        "non_significant": [f"{k}/{m}" for k, v in ext.items() for m in ("bleu", "chrf") if not v[m]["significant"]],
    }
    q = R["significance"]["finetuned_hf_vs_finetuned_ct2"]
    R["quantisation_summary"] = {
        "max_abs_bleu_delta": max(abs(v["bleu"]["delta"]) for v in q.values()),
        "max_abs_chrf_delta": max(abs(v["chrf"]["delta"]) for v in q.values()),
        "n_tests": 2 * len(q), "n_significant": sum(v[m]["significant"] for v in q.values() for m in ("bleu", "chrf")),
        "significant": [f"{k}/{m}" for k, v in q.items() for m in ("bleu", "chrf") if v[m]["significant"]],
    }
    st = R["runtime"]["step_seconds"]
    R["runtime"]["training_step_h"] = round(st[8]["seconds"] / 3600, 2) if 8 in st else None
    R["runtime"]["final_eval_step_h"] = round(st[12]["seconds"] / 3600, 2) if 12 in st else None
    R["runtime"]["ct2_conversion_s"] = st[10]["seconds"] if 10 in st else None
    tp = R["runtime"]["eval_throughput_sent_per_s"]
    R["runtime"]["ct2_gazette_sent_per_s"] = [tp.get("gazette_test/finetuned_ct2/mk_sq"), tp.get("gazette_test/finetuned_ct2/sq_mk")]
    R["runtime"]["hf_gazette_sent_per_s"] = [tp.get("gazette_test/finetuned_hf/mk_sq"), tp.get("gazette_test/finetuned_hf/sq_mk")]
    R["training"]["logging_steps"] = 50
    R["report_constants"] = {"n_test_sets": 3, "n_directions": 2, "n_systems": 3, "min_p_resolution": round(1 / (N_BOOTSTRAP + 1), 4),
                             "dev_eval_greedy_beams": 1, "hf_ct2_tests_per_set": 4}


# ============================================================================
# 6. Figures (from the CSVs only)
# ============================================================================
C = {"base": "#2a78d6", "finetuned_hf": "#eb6834", "finetuned_ct2": "#1baf7a", "reference": "#52514e"}  # fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"base": "Base (E00)", "finetuned_hf": "Fine-tuned HF", "finetuned_ct2": "Fine-tuned CT2 int8", "reference": "Human reference"}


def _style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def figures(R: dict, paths: dict) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK, "text.color": INK,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    tl = pl.read_csv(paths["eval/train_log.csv"])
    best = R["training"]["best_step"]
    # training curves: loss and learning rate in separate panels (never a dual axis)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(tl["step"], tl["loss"], color=C["finetuned_hf"], lw=2)
    a1.set_ylabel("training loss\n(label-smoothed, mean of 50 steps)")
    a2.plot(tl["step"], tl["learning_rate"], color=C["finetuned_hf"], lw=2)
    a2.set_ylabel("learning rate")
    a2.set_xlabel("optimizer step")
    for ax in (a1, a2):
        _style(ax)
        ax.axvline(best, color=MUTED, lw=1, ls=":")
    a1.set_title(f"E01 training (best checkpoint = step {best}, dotted)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=200)
    plt.close(fig)
    # dev curve
    dc = pl.read_csv(paths["eval/dev_curve.csv"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, metric, name in ((a1, "chrf", "chrF++"), (a2, "bleu", "BLEU")):
        for d, lab, ls in (("mk_sq", "MK→SQ", "-"), ("sq_mk", "SQ→MK", "--")):
            ax.plot(dc["step"], dc[f"{metric}_{d}"], color=C["finetuned_hf"], ls=ls, lw=2, marker="o", ms=5, label=lab)
            ax.annotate(lab, (dc["step"][-1], dc[f"{metric}_{d}"][-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8)
        if metric == "chrf":
            ax.plot(dc["step"], dc["chrf_mean"], color=C["reference"], lw=1.5, marker="s", ms=4, label="mean")
        ax.axvline(best, color=MUTED, lw=1, ls=":")
        ax.set_title(f"Gazette dev {name} (greedy, 500 sentences per direction)", loc="left")
        ax.set_xlabel("optimizer step")
        _style(ax)
        if metric == "chrf":
            ax.annotate("mean", (dc["step"][-1], dc["chrf_mean"][-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8)
        ax.set_xlim(dc["step"].min() - 200, dc["step"].max() + 700)
    fig.tight_layout()
    fig.savefig(FIG / "dev_curve.png", dpi=200)
    plt.close(fig)
    # overlap breakdown
    ov = pl.read_csv(paths["eval/gazette_test_overlap_results.csv"])
    classes = ["exact_template", "near_duplicate", "novel"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    shares = [ov.filter((pl.col("class") == c) & (pl.col("model") == "base") & (pl.col("direction") == "mk_sq"))["pct_pairs"][0]
              for c in classes]
    axes[0].barh(["exact template", "near duplicate", "novel"][::-1], shares[::-1], color=C["reference"], height=0.6)
    for y, v in enumerate(shares[::-1]):
        axes[0].text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].set_xlim(0, max(shares) * 1.25)
    axes[0].set_title("Share of Gazette test pairs", loc="left")
    axes[0].set_xlabel("% of 10,174 pairs")
    for ax, (d, lab) in zip(axes[1:], DIRECTIONS, strict=True):
        x = range(len(classes))
        for k, m in enumerate(("base", "finetuned_hf")):
            vals = [ov.filter((pl.col("class") == c) & (pl.col("model") == m) & (pl.col("direction") == d))["chrf"][0]
                    for c in classes]
            ax.bar([i + (k - 0.5) * 0.38 for i in x], vals, width=0.36, color=C[m], label=LABEL[m], edgecolor="#fcfcfb", lw=1)
            for i, v in zip(x, vals, strict=True):
                ax.text(i + (k - 0.5) * 0.38, v + 1, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(list(x), ["exact\ntemplate", "near\nduplicate", "novel"])
        ax.set_ylim(0, 100)
        ax.set_title(f"chrF++ by overlap class, {lab}", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "overlap_breakdown.png", dpi=200)
    plt.close(fig)
    # terminology
    te = pl.read_csv(paths["eval/terminology_results.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col, title in ((axes[0], "tsr", "All terms"), (axes[1], "rare_tsr", "Rare terms (≤ 5 train sentences)")):
        for j, (d, lab) in enumerate(DIRECTIONS):
            ref = te.filter((pl.col("model") == "reference") & (pl.col("direction") == d))[col][0]
            ax.hlines(ref, j - 0.42, j + 0.42, color=C["reference"], lw=2, ls="--", label=LABEL["reference"] if j == 0 else None)
            for k, m in enumerate(("base", "finetuned_hf", "finetuned_ct2")):
                v = te.filter((pl.col("model") == m) & (pl.col("direction") == d))[col][0]
                ax.bar(j + (k - 1) * 0.27, v, width=0.25, color=C[m], hatch="//" if m == "finetuned_ct2" else None,
                       edgecolor="#fcfcfb", lw=1, label=LABEL[m] if j == 0 else None)
                ax.text(j + (k - 1) * 0.27, v + 1.2, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks([0, 1], [lab for _, lab in DIRECTIONS])
        ax.set_ylim(0, 100)
        ax.set_title(f"Term Success Rate, {title}", loc="left")
        _style(ax)
    axes[0].set_ylabel("TSR (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "terminology.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 7. Tables (Markdown + LaTeX booktabs from the same rows)
# ============================================================================
def f2(x) -> str:
    return "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def pfmt(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("→", r"$\rightarrow$").replace("≤", r"$\leq$").replace("Δ", r"$\Delta$")
            .replace("±", r"$\pm$").replace("–", "--"))


def write_table(name: str, caption: str, header: list[str], rows: list[list], source: str) -> str:
    numeric = lambda i: all(isinstance(r[i], (int, float)) or r[i] is None or re.fullmatch(r"[<\[\d.,\- \]]+|–", str(r[i]))  # noqa: E731
                            for r in rows)
    align = "l" + "".join("r" if numeric(i) else "l" for i in range(1, len(header)))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(f2(c)) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex_escape(caption)} Source: \texttt{{{tex_escape(source)}}}.}}",
              rf"\label{{tab:e01-{name}}}", r"\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(f2(c) for c in r) + " |" for r in rows]
    return "\n".join(md) + f"\n\nSource: {source}\n"


def tables(R: dict) -> dict:
    T = {}
    dn = dict(DIRECTIONS)
    tn = dict(TESTSETS)
    mn = {"base": "Base (E00)", "finetuned_hf": "Fine-tuned HF (E01)", "finetuned_ct2": "Fine-tuned CT2 int8 (E01)",
          "reference": "Human reference"}
    rows = []
    for ts, _ in TESTSETS:
        for d, _ in DIRECTIONS:
            sig = R["significance"]["base_vs_finetuned_hf"][f"{ts}/{d}"]
            for m, _ in MODELS:
                r = next(x for x in R["results_main"][ts] if x["model"] == m and x["direction"] == d)
                p_b = pfmt(sig["bleu"]["p_value"]) if m == "finetuned_hf" else "–"
                p_c = pfmt(sig["chrf"]["p_value"]) if m == "finetuned_hf" else "–"
                rows.append([tn[ts], dn[d], mn[m], r["n_sentences"], r["bleu"], r["chrf"],
                             r["bleu_delta_vs_base"] if m != "base" else None, r["chrf_delta_vs_base"] if m != "base" else None,
                             p_b, p_c])
    T["main"] = write_table("main_results", "Main results: BLEU and chrF++ (beam 4). Deltas are against the base model; p-values from a paired bootstrap (1,000 resamples, seed 42), base vs fine-tuned HF.",
                            ["Test set", "Dir.", "System", "n", "BLEU", "chrF++", "ΔBLEU", "ΔchrF++", "p (BLEU)", "p (chrF++)"],
                            rows, "eval/gazette_test_results.csv, eval/flores_results.csv, eval/ntrex_results.csv; significance: results.json")
    rows = []
    for key, v in R["significance"]["base_vs_finetuned_hf"].items():
        ts, d = key.split("/")
        for m, lab in (("bleu", "BLEU"), ("chrf", "chrF++")):
            b, s_ = v[m]["baseline"], v[m]["system"]
            rows.append([tn[ts], dn[d], lab, b["score"], f"[{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}]", s_["score"],
                         f"[{s_['ci95'][0]:.2f}, {s_['ci95'][1]:.2f}]", v[m]["delta"], pfmt(v[m]["p_value"])])
    T["sig"] = write_table("significance", "Paired bootstrap, base vs fine-tuned HF: scores with 95% confidence intervals of the bootstrap mean.",
                           ["Test set", "Dir.", "Metric", "Base", "Base 95% CI", "FT", "FT 95% CI", "Δ", "p"], rows,
                           "eval/translations_{gazette_test,flores,ntrex}.parquet (recomputed); results.json#significance")
    rows = []
    for c in ("exact_template", "near_duplicate", "novel", "all"):
        for d, _ in DIRECTIONS:
            b = next(r for r in R["overlap"] if r["class"] == c and r["model"] == "base" and r["direction"] == d)
            f = next(r for r in R["overlap"] if r["class"] == c and r["model"] == "finetuned_hf" and r["direction"] == d)
            k = next(r for r in R["overlap"] if r["class"] == c and r["model"] == "finetuned_ct2" and r["direction"] == d)
            dd = R["overlap_summary"]["finetuned_hf_minus_base"][f"{c}/{d}"]
            rows.append([c.replace("_", " "), dn[d], b["n_pairs"], b["pct_pairs"], b["bleu"], f["bleu"], k["bleu"], dd["bleu"],
                         b["chrf"], f["chrf"], k["chrf"], dd["chrf"]])
    T["overlap"] = write_table("overlap", "Gazette test by overlap with the Gazette training sentences (normalised: lowercase, digits to 0, whitespace collapsed; near duplicate = character 5-gram Jaccard ≥ 0.8).",
                               ["Class", "Dir.", "Pairs", "% pairs", "BLEU base", "BLEU FT", "BLEU CT2", "ΔBLEU",
                                "chrF++ base", "chrF++ FT", "chrF++ CT2", "ΔchrF++"], rows, "eval/gazette_test_overlap_results.csv")
    rows = [[mn[r["model"]], dn[r["direction"]], r["n_sentences_with_terms"], r["n_term_occurrences"], r["tsr"],
             r["n_rare_occurrences"], r["rare_tsr"]] for r in R["terminology"]]
    T["terms"] = write_table("terminology", "Term Success Rate (micro, %) with Verbis terms (source term ≥ 5 characters, whole-word match); rare = contained in ≤ 5 Gazette training sentences.",
                             ["System", "Dir.", "Sentences with terms", "Term occurrences", "TSR", "Rare occurrences", "Rare TSR"],
                             rows, "eval/terminology_results.csv")
    rows = [[mn[r["model"]], dn[r["direction"]], r["n_sentences"], r["match_rate_all"], r["n_with_numbers"],
             r["match_rate_with_numbers"]] for r in R["number_fidelity"]]
    T["numbers"] = write_table("number_fidelity", "Number fidelity: share of sentences whose multiset of digit sequences in the output equals that of the source (%).",
                               ["System", "Dir.", "Sentences", "Match (all)", "Sentences with numbers", "Match (with numbers)"],
                               rows, "eval/number_fidelity_results.csv")
    rows = []
    for key, v in R["significance"]["finetuned_hf_vs_finetuned_ct2"].items():
        ts, d = key.split("/")
        rows.append([tn[ts], dn[d], v["bleu"]["baseline"]["score"], v["bleu"]["system"]["score"], v["bleu"]["delta"],
                     pfmt(v["bleu"]["p_value"]), v["chrf"]["baseline"]["score"], v["chrf"]["system"]["score"],
                     v["chrf"]["delta"], pfmt(v["chrf"]["p_value"])])
    T["quant"] = write_table("quantisation", "Quantisation: fine-tuned HF (bf16, GPU) vs CTranslate2 int8 (CPU); Δ = CT2 − HF; paired bootstrap p-values.",
                             ["Test set", "Dir.", "BLEU HF", "BLEU CT2", "ΔBLEU", "p", "chrF++ HF", "chrF++ CT2", "ΔchrF++", "p"],
                             rows, "eval/*_results.csv, eval/translations_*.parquet; results.json#significance")
    rows = [[r["step"], r["bleu_mk_sq"], r["chrf_mk_sq"], r["bleu_sq_mk"], r["chrf_sq_mk"], r["chrf_mean"]]
            for r in R["training"]["dev_curve"]]
    T["dev"] = write_table("dev_curve", "Gazette dev evaluation during training (greedy, 500 sentences per direction).",
                           ["Step", "BLEU MK→SQ", "chrF++ MK→SQ", "BLEU SQ→MK", "chrF++ SQ→MK", "chrF++ mean"], rows,
                           "eval/dev_curve.csv")
    t = R["training"]
    rows = [["Base model", t["base_model"]], ["Base model snapshot", t["base_model_snapshot"]],
            ["Method", "full-parameter SFT, bidirectional"], ["Optimizer", t["optimizer"]],
            ["Learning rate", f"{t['learning_rate']:g}"], ["Schedule", f"{t['lr_scheduler']}, warmup {t['warmup_steps']} steps"],
            ["Epochs", str(t["epochs"])], ["Label smoothing", str(t["label_smoothing"])],
            ["Weight decay", str(t["weight_decay"])], ["Max grad norm", str(t["max_grad_norm"])], ["Precision", "bf16 mixed precision, TF32 matmul"],
            ["Batch (per device × accumulation)", f"{t['per_device_batch']} × {t['grad_accumulation']} = {t['effective_batch']}"],
            ["Gradient checkpointing", str(t["gradient_checkpointing"])], ["Batch sampling", t["sampling"]],
            ["Max length (tokens, drop not truncate)", str(R["data"]["length_filter_max_tokens"])],
            ["Seed / data seed", f"{t['seed']} / {t['data_seed']}"], ["Total optimizer steps", f"{t['total_steps']:,}"],
            ["Evaluation / checkpoint interval", f"{t['eval_steps']} steps"],
            ["Checkpoint selection", f"{t['metric_for_best_model']} (mean dev chrF++ of both directions)"],
            ["Best step / dev chrF++ mean", f"{t['best_step']} / {t['best_dev_chrf_mean']}"],
            ["Training time (h)", str(t["training_time_h"])]]
    T["hparams"] = write_table("hyperparameters", "Training configuration as recorded.", ["Setting", "Value"], rows,
                               "eval/run_info.json, checkpoints/checkpoint-4115/trainer_state.json")
    dd = R["data"]
    comp, pct = dd["composition_examples"], dd["composition_pct"]
    rows = [["Gazette train / dev / test (pairs)", f"{dd['gazette']['train']:,} / {dd['gazette']['dev']:,} / {dd['gazette']['test']:,}"],
            ["Removed as leakage: Gazette / Verbis terms / Verbis definitions",
             f"{dd['leakage_removed']['gazette']:,} / {dd['leakage_removed']['verbis_terms']:,} / {dd['leakage_removed']['verbis_defs']:,}"],
            ["Clean training pairs", f"{dd['train_clean_pairs']:,}"],
            ["Bidirectional examples", f"{dd['train_bidirectional']:,}"],
            ["Dropped by length filter (> 192 tokens)", f"{dd['length_filter_dropped']:,} ({dd['length_filter_dropped_pct']}%)"],
            ["Tokenised training examples", f"{dd['train_tokenized']:,}"]]
    rows += [[f"Composition: {k.replace('_', ' ')}", f"{v:,} ({pct[k]}%)"] for k, v in comp.items()]
    T["data"] = write_table("data", "Training data after leakage removal, bidirectional expansion and length filtering.",
                            ["Quantity", "Value"], rows, "eval/run_info.json (dataset_sizes), progress.log")
    return T


# ============================================================================
# 8. REPORT.md
# ============================================================================
def render(R: dict, paths: dict) -> None:
    derive(R)
    figures(R, paths)
    T = tables(R)
    (HERE / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str) + "\n")
    (HERE / "REPORT.md").write_text(report_text(R, T), encoding="utf-8")


def row(R, ts, m, d):
    return next(x for x in R["results_main"][ts] if x["model"] == m and x["direction"] == d)


def ovr(R, c, m, d):
    return next(x for x in R["overlap"] if x["class"] == c and x["model"] == m and x["direction"] == d)


def term(R, m, d):
    return next(x for x in R["terminology"] if x["model"] == m and x["direction"] == d)


def num(R, m, d):
    return next(x for x in R["number_fidelity"] if x["model"] == m and x["direction"] == d)


def report_text(R: dict, T: dict) -> str:
    d, t, rt, env = R["data"], R["training"], R["runtime"], R["environment"]
    sig = R["significance"]["base_vs_finetuned_hf"]
    fs, qs, osu = R["forgetting_summary"], R["quantisation_summary"], R["overlap_summary"]
    g = {dd: row(R, "gazette_test", "base", dd) for dd, _ in DIRECTIONS}
    gf = {dd: row(R, "gazette_test", "finetuned_hf", dd) for dd, _ in DIRECTIONS}
    nb = {dd: ovr(R, "novel", "base", dd) for dd, _ in DIRECTIONS}
    nf = {dd: ovr(R, "novel", "finetuned_hf", dd) for dd, _ in DIRECTIONS}
    tsr = {(m, dd): term(R, m, dd) for m in ("reference", "base", "finetuned_hf", "finetuned_ct2") for dd, _ in DIRECTIONS}
    nfid = {(m, dd): num(R, m, dd) for m in ("reference", "base", "finetuned_hf", "finetuned_ct2") for dd, _ in DIRECTIONS}
    pct = d["composition_pct"]
    nonsig = ", ".join(x.replace("/", " ").replace("mk_sq", "MK→SQ").replace("sq_mk", "SQ→MK").replace("flores", "FLORES+")
                       .replace("ntrex", "NTREX-128").replace("bleu", "BLEU").replace("chrf", "chrF++") for x in fs["non_significant"]) or "none"
    qsig = ", ".join(x.replace("/", " ").replace("mk_sq", "MK→SQ").replace("sq_mk", "SQ→MK").replace("gazette_test", "Gazette test")
                     .replace("flores", "FLORES+").replace("ntrex", "NTREX-128").replace("bleu", "BLEU").replace("chrf", "chrF++")
                     for x in qs["significant"]) or "none"
    inv = rt["invocations"]
    dev = t["dev_curve"]
    dev_best_mk = max(dev, key=lambda r: r["chrf_mk_sq"])
    dev_best_sq = max(dev, key=lambda r: r["chrf_sq_mk"])
    rare_gap = R["terminology_summary"]
    ex = R["external"]
    L = []
    w = L.append
    w(f"# Experiment report: {R['experiment']['id']}\n")
    w(f"Combined lexicon-aware system (closest research-design arm: {R['experiment']['paper_arm']}); baseline "
      f"{R['experiment']['baseline_id']} (arm {R['experiment']['baseline_arm']}). Generated {R['meta']['generated_at']} by "
      f"`experiments/{EXPERIMENT_ID}/build_report.py` from results repo revision `{R['meta']['results_repo_revision']}` "
      f"and git commit `{R['meta']['report_git_head']}`. Every number below is in `results.json`; every artifact is listed "
      "with its sha256 in `MANIFEST.json`.\n")
    # 1
    w("## 1. Summary\n")
    w(f"NLLB-200-distilled-600M was fully fine-tuned in both directions (Macedonian↔Albanian) in a single run on "
      f"{d['train_tokenized']:,} tokenised examples drawn from the Gazette parallel corpus ({pct['gazette']}%), Verbis term "
      f"pairs ({pct['verbis_terms']}%) and LaBSE-filtered Verbis definition pairs ({pct['verbis_defs']}%), for one epoch "
      f"({t['total_steps']:,} optimizer steps, {t['training_time_h']} h on one RTX 5080). "
      f"On the full Gazette test set ({g['mk_sq']['n_sentences']:,} pairs), BLEU rose from {g['mk_sq']['bleu']} to "
      f"{gf['mk_sq']['bleu']} (MK→SQ) and from {g['sq_mk']['bleu']} to {gf['sq_mk']['bleu']} (SQ→MK), and chrF++ from "
      f"{g['mk_sq']['chrf']} to {gf['mk_sq']['chrf']} and from {g['sq_mk']['chrf']} to {gf['sq_mk']['chrf']} "
      f"(paired bootstrap p = {pfmt(sig['gazette_test/mk_sq']['bleu']['p_value'])} for every in-domain comparison). "
      f"On the novel subset ({osu['pct_pairs']['novel']}% of test pairs, with no template or near-duplicate match in "
      f"training), BLEU rose from {nb['mk_sq']['bleu']} to {nf['mk_sq']['bleu']} (MK→SQ) and from {nb['sq_mk']['bleu']} to "
      f"{nf['sq_mk']['bleu']} (SQ→MK). "
      f"On the general-domain sets FLORES+ and NTREX-128, BLEU fell by {abs(fs['bleu_delta_range'][1])} to "
      f"{abs(fs['bleu_delta_range'][0])} points; {fs['n_significant_declines']} of {fs['n_tests']} comparisons were "
      f"significant declines (not significant: {nonsig}). "
      f"The CTranslate2 int8 export stayed within {qs['max_abs_bleu_delta']} BLEU of the HF model. "
      f"The main caveats are a single seed, a test set in which {osu['pct_template_or_near_duplicate']}% of pairs are "
      "templates or near duplicates of training sentences, and the combination of Gazette and Verbis data in one run, "
      "which prevents attributing any effect to the lexicon.\n")
    # 2
    w("## 2. Research questions\n")
    w("The wording of the research questions is defined in the research design and is NOT RECORDED in the sources "
      "used for this report; only the mapping given for this experiment is used here.\n")
    w("- **RQ1** — informed: the in-domain effect of fine-tuning on Gazette + Verbis data (§5.1, §5.2).")
    w("- **RQ5** — informed: retention of general-domain quality on FLORES+ and NTREX-128 (§5.1, §6).")
    w("- **RQ6** — partly informed: terminology and number behaviour (§5.3, §5.4) and int8 deployment (§5.5).")
    w("- **RQ2–RQ4** — not answerable from this run: Gazette and Verbis data were combined in one training mixture, so the "
      "contribution of the lexical resource cannot be isolated from that of the parallel corpus. This requires the "
      "Gazette-only arm (§9).\n")
    # 3
    w("## 3. Data\n")
    w("### 3.1 Gazette parallel corpus\n")
    w(f"- Source: Hugging Face dataset `{d['gazette']['repo']}` (licence recorded on the dataset card: "
      f"{ex['gazette']['license']}). Revision used by the run: {d['gazette']['revision_used_by_run']}. At report time the "
      f"dataset was at `{d['gazette']['current_sha']}`, last modified {d['gazette']['current_last_modified']}, which "
      f"precedes the first run invocation ({inv[0]['started']} UTC); this is a timing fact, not a record of the revision used.")
    w(f"- Splits loaded: {', '.join(d['gazette']['splits_loaded'])}. Sizes: train {d['gazette']['train']:,}, "
      f"dev {d['gazette']['dev']:,}, test {d['gazette']['test']:,} pairs. Dev was the {d['gazette']['dev_origin']} (no hold-out was created).")
    io_ = d["gazette"]["issue_overlap"]
    w(f"- Split method: issue-disjoint. Computed from the dataset at `{io_['revision']}`: {io_['test_issues']} test issues "
      f"(issue_key), {io_['test_issues_in_train']} of which occur among the {io_['train_issues']:,} train issues.\n")
    w("### 3.2 Verbis dictionary\n")
    w(f"- Source: {d['verbis']['publisher']}, API `{d['verbis']['source_api']}` (`scripts/verbis_harvest.py`). The "
      f"dataset used by the run (`data/verbis/verbis_mk_sq.parquet` in the private repo) had {d['verbis']['entries']:,} entries.")
    w(f"- Crawl date: {d['verbis']['crawl_date']}. Categories crawled: {d['verbis']['categories_crawled']}. The local "
      f"harvest cache (`scripts/verbis_cache/`) holds {d['verbis']['local_cache_aggregate']['unique_entries']:,} unique "
      f"entries, which does not match the {d['verbis']['entries']:,} entries used, so it cannot establish either fact. "
      "The harvest script's docstring states a total that also differs from the dataset used.")
    w("- Cleaning: the harvest script forces each field into a single script (Cyrillic/Latin homoglyph repair). The "
      "training script stripped trailing parenthetical qualifiers and Albanian inflection endings from headwords, "
      "lowercased all-capital headwords longer than 4 characters, dropped empty sides and removed duplicate pairs.")
    w(f"- Term pairs kept: {d['verbis']['term_pairs_kept']:,}. Definition pairs: {d['verbis']['definition_pairs_scored']:,} "
      f"scored with LaBSE, {d['verbis']['definition_pairs_kept']:,} kept at LaBSE ≥ {d['verbis']['labse_threshold']} "
      f"({d['verbis']['definition_keep_pct']}%; median score {d['verbis']['labse_median']}).")
    w(f"- Licence status: {d['verbis']['licence']}. No Verbis entry is reproduced in this report.\n")
    w("### 3.3 External test sets\n")
    w(f"- FLORES+ devtest (`{d['flores']['repo']}`, mkd_Cyrl / als_Latn): {d['flores']['sentences']:,} sentences; licence "
      f"{', '.join(d['flores']['licence'])} (dataset card at `{d['flores']['licence_source_revision']}`). Revision used by the run: {d['flores']['revision_used_by_run']}.")
    w(f"- NTREX-128 (`{d['ntrex']['repo']}`, mkd / sqi): {d['ntrex']['sentences']:,} sentences; licence {d['ntrex']['licence']} "
      f"({ex['ntrex']['license_file']} at `{d['ntrex']['licence_source_revision']}`). Revision used by the run: {d['ntrex']['revision_used_by_run']}.\n")
    w("### 3.4 Leakage removal, length filter and composition\n")
    w(f"Training pairs whose Macedonian or Albanian side (lowercased, whitespace-normalised) occurs in Gazette dev, Gazette "
      f"test, FLORES+ or NTREX-128 were removed: {d['leakage_removed']['gazette']:,} Gazette pairs "
      f"({d['gazette_leakage_pct']}% of Gazette train), {d['leakage_removed']['verbis_terms']:,} Verbis term pairs and "
      f"{d['leakage_removed']['verbis_defs']:,} definition pairs. Each of the {d['train_clean_pairs']:,} remaining pairs was used "
      f"in both directions ({d['direction_split_examples']:,} examples per direction), and "
      f"{d['length_filter_dropped']:,} examples ({d['length_filter_dropped_pct']}%) longer than "
      f"{d['length_filter_max_tokens']} tokens were dropped.\n")
    w(T["data"])
    # 4
    w("## 4. Model and training\n")
    w(f"The base model `{t['base_model']}` (snapshot `{t['base_model_snapshot']}`) was trained with {t['method']}. "
      f"The per-device batch size was chosen by a memory probe on worst-case 192-token batches: batch "
      f"{t['probe_attempts'][0]['batch']} ran out of memory (peak {t['probe_attempts'][0]['peak_gb']} GB), batch "
      f"{t['probe_attempts'][1]['batch']} succeeded (peak {t['probe_attempts'][1]['peak_gb']} GB), giving "
      f"{t['per_device_batch']} × {t['grad_accumulation']} = {t['effective_batch']} examples per optimizer step. "
      f"Training ran for {t['total_steps']:,} steps in {t['training_time_h']} h (median {t['samples_per_sec_median']} "
      f"samples/s, {t['tokens_per_sec_median']:,.0f} non-padding tokens/s; peak allocated GPU memory "
      f"{t['gpu_mem_peak_gb']} GB; maximum GPU temperature {t['gpu_temp_max_c']} °C; {t['training_warnings']} training "
      f"warnings). The mean training loss fell from {t['loss_first_logged']} (first {t['logging_steps']} steps) to "
      f"{t['loss_last_logged']} (last logged interval). The checkpoint with the highest mean dev chrF++ (evaluated every "
      f"{t['eval_steps']:,} steps and at the final step) was step {t['best_step']:,} ({t['best_dev_chrf_mean']}).\n")
    w(f"Hardware: {env['hardware']['gpu']} ({env['hardware']['vram_total_gb']} GB, compute capability "
      f"{env['hardware']['compute_capability']}, driver {env['hardware']['driver']}); CPU {env['hardware']['cpu']}; RAM "
      f"{env['hardware']['ram']}. Software: Python {env['versions']['python']}, torch {env['versions']['torch']}, "
      f"transformers {env['versions']['transformers']}, datasets {env['versions']['datasets']}, accelerate "
      f"{env['versions']['accelerate']}, ctranslate2 {env['versions']['ctranslate2']}, sacrebleu {env['versions']['sacrebleu']}, "
      f"sentence-transformers {env['versions']['sentence_transformers']}. Training code: commit `{R['code']['training_commit']}` "
      f"(clean working tree: {not R['code']['training_commit_dirty']}).\n")
    w(T["hparams"])
    # 5
    w("## 5. Results\n")
    w(f"All test-set scores use beam {R['evaluation']['beams']} and at most {R['evaluation']['max_new_tokens']} new tokens. "
      f"BLEU signature `{R['evaluation']['bleu_signature']}`; chrF++ signature `{R['evaluation']['chrf_signature']}`.\n")
    w("### 5.1 Main results\n")
    w(T["main"])
    w(f"Significance is reported as the paired-bootstrap p-value with {R['analysis_constants']['bootstrap_resamples']:,} "
      f"resamples (seed {R['analysis_constants']['bootstrap_seed']}); {R['report_constants']['min_p_resolution']} is the "
      "smallest value this number of resamples can produce. Confidence intervals are listed in the next table.\n")
    w(T["sig"])
    w("### 5.2 Overlap-controlled results (Gazette test)\n")
    w(f"Of the {osu['n_pairs']['exact_template'] + osu['n_pairs']['near_duplicate'] + osu['n_pairs']['novel']:,} test pairs, "
      f"{osu['n_pairs']['exact_template']:,} ({osu['pct_pairs']['exact_template']}%) were exact templates of a training "
      f"sentence (identical after digit normalisation), {osu['n_pairs']['near_duplicate']:,} "
      f"({osu['pct_pairs']['near_duplicate']}%) were near duplicates and {osu['n_pairs']['novel']:,} "
      f"({osu['pct_pairs']['novel']}%) were novel. Significance was not computed per class.\n")
    w(T["overlap"])
    w("![Overlap breakdown](figures/overlap_breakdown.png)\n")
    w("### 5.3 Terminology (Term Success Rate)\n")
    w("A term occurrence counts as a success when every token of a listed Verbis target term is matched by a hypothesis "
      f"token sharing its first max({R['analysis_constants']['prefix_min_chars']}, length − "
      f"{R['analysis_constants']['prefix_trim_chars']}) characters. The human reference, scored the same way, gives the "
      "realistic ceiling.\n")
    w(T["terms"])
    w("![Terminology](figures/terminology.png)\n")
    w("### 5.4 Number fidelity\n")
    w(T["numbers"])
    w("### 5.5 Quantisation (HF bf16 vs CTranslate2 int8)\n")
    w(f"CTranslate2 could not use the GPU and ran on CPU (int8). Across the {qs['n_tests']} paired comparisons the "
      f"largest absolute difference was {qs['max_abs_bleu_delta']} BLEU and {qs['max_abs_chrf_delta']} chrF++; "
      f"{qs['n_significant']} reached p < {R['analysis_constants']['alpha']} ({qsig}). CPU decoding was slower: "
      f"{rt['ct2_gazette_sent_per_s'][0]} and {rt['ct2_gazette_sent_per_s'][1]} sentences/s on the Gazette test against "
      f"{rt['hf_gazette_sent_per_s'][0]} and {rt['hf_gazette_sent_per_s'][1]} for the HF model on the GPU.\n")
    w(T["quant"])
    w("### 5.6 Weight interpolation (WiSE-FT)\n")
    w(f"{R['interpolation'].capitalize()}: no `eval/interp_*` artifact exists and `scripts/interpolate_and_eval.py` is not in the repository.\n")
    w("### 5.7 Training dynamics\n")
    w(T["dev"])
    w(f"Mean dev chrF++ rose from {dev[0]['chrf_mean']} at step {dev[0]['step']:,} to {dev[-1]['chrf_mean']} at step "
      f"{dev[-1]['step']:,}. MK→SQ dev chrF++ peaked at step {dev_best_mk['step']:,} ({dev_best_mk['chrf_mk_sq']}) and "
      f"SQ→MK at step {dev_best_sq['step']:,} ({dev_best_sq['chrf_sq_mk']}); the selected checkpoint (step "
      f"{t['best_step']:,}) is therefore the best mean, not the best MK→SQ checkpoint.\n")
    w("![Dev curve](figures/dev_curve.png)\n\n![Training curves](figures/training_curves.png)\n")
    # 6
    w("## 6. Interpretation\n")
    w(f"- **In-domain gain beyond overlap.** The gain was largest on exact templates, but it held on the novel subset: "
      f"+{R['overlap_summary']['finetuned_hf_minus_base']['novel/mk_sq']['bleu']} BLEU (MK→SQ) and "
      f"+{R['overlap_summary']['finetuned_hf_minus_base']['novel/sq_mk']['bleu']} BLEU (SQ→MK) over the base model. "
      "The full-test gain is therefore not an artefact of template overlap, although its size on the full test set is "
      "inflated by it.")
    w(f"- **General-domain degradation.** On FLORES+ and NTREX-128 the fine-tuned model scored below the base model in "
      f"every comparison (BLEU {fs['bleu_delta_range'][0]} to {fs['bleu_delta_range'][1]}; chrF++ "
      f"{fs['chrf_delta_range'][0]} to {fs['chrf_delta_range'][1]}), with {fs['n_significant_declines']} of "
      f"{fs['n_tests']} declines significant. This is consistent with catastrophic forgetting of general-domain ability.")
    ts_ = R["terminology_summary"]
    w(f"- **Terminology.** The fine-tuned model's TSR ({tsr[('finetuned_hf', 'mk_sq')]['tsr']}% MK→SQ, "
      f"{tsr[('finetuned_hf', 'sq_mk')]['tsr']}% SQ→MK) reached the human reference rate "
      f"({tsr[('reference', 'mk_sq')]['tsr']}%, {tsr[('reference', 'sq_mk')]['tsr']}%; differences "
      f"{ts_['mk_sq']['finetuned_hf_minus_reference']:+} and {ts_['sq_mk']['finetuned_hf_minus_reference']:+} points), "
      f"up from {tsr[('base', 'mk_sq')]['tsr']}% and {tsr[('base', 'sq_mk')]['tsr']}% for the base model.")
    w(f"- **Rare terms.** On rare terms the fine-tuned model remained below the reference: "
      f"{tsr[('finetuned_hf', 'mk_sq')]['rare_tsr']}% vs {tsr[('reference', 'mk_sq')]['rare_tsr']}% (MK→SQ, "
      f"{tsr[('finetuned_hf', 'mk_sq')]['n_rare_occurrences']} occurrences) and "
      f"{tsr[('finetuned_hf', 'sq_mk')]['rare_tsr']}% vs {tsr[('reference', 'sq_mk')]['rare_tsr']}% (SQ→MK, "
      f"{tsr[('finetuned_hf', 'sq_mk')]['n_rare_occurrences']} occurrences), the larger gap being in SQ→MK "
      f"({rare_gap['sq_mk']['rare_finetuned_hf_minus_reference']} points).")
    w(f"- **Numbers.** The fine-tuned model reproduced the source's digit sequences in "
      f"{nfid[('finetuned_hf', 'mk_sq')]['match_rate_with_numbers']}% (MK→SQ) and "
      f"{nfid[('finetuned_hf', 'sq_mk')]['match_rate_with_numbers']}% (SQ→MK) of sentences with numbers, above both the "
      f"base model and the human reference ({nfid[('reference', 'mk_sq')]['match_rate_with_numbers']}%, "
      f"{nfid[('reference', 'sq_mk')]['match_rate_with_numbers']}%). Because the reference itself departs from the "
      "source's digits, this measures digit copying rather than correctness.")
    w(f"- **Quantisation.** int8 CTranslate2 conversion changed scores by at most {qs['max_abs_bleu_delta']} BLEU; "
      f"{qs['n_significant']} of {qs['n_tests']} differences were statistically significant but small, which is "
      "negligible relative to the fine-tuning effect.\n")
    # 7
    w("## 7. Limitations and threats to validity\n")
    w(f"- A single training run with seed {t['seed']}; no variance across seeds is available.")
    w(f"- Formulaic overlap: {osu['pct_template_or_near_duplicate']}% of Gazette test pairs are templates or near duplicates "
      "of training sentences, despite the issue-disjoint split; full-test scores overstate generalisation.")
    w("- TSR depends on the whole-word source match and the prefix heuristic; the human reference itself uses the listed "
      f"Verbis target only {tsr[('reference', 'mk_sq')]['tsr']}% (MK→SQ) and {tsr[('reference', 'sq_mk')]['tsr']}% (SQ→MK) "
      "of the time, so TSR is not a correctness measure. Rare-term results rest on few occurrences.")
    w("- Number fidelity mixes formatting differences (e.g. thousands separators, OCR artefacts) with genuine errors.")
    w("- Gazette and Verbis were mixed in one run, so no effect can be attributed to the lexicon.")
    w("- Verbis redistribution is not granted; the data are stored privately and the experiment cannot be released with them.")
    w("- The Gazette, FLORES+ and NTREX revisions used by the run were not logged (§3).")
    w("- Dev evaluation during training used greedy decoding on 500 sentences per direction; test evaluation used beam 4.\n")
    # 8
    w("## 8. Reproducibility\n")
    w(f"- Training code: commit `{R['code']['training_commit']}`; report code: commit `{R['meta']['report_git_head']}`.")
    w(f"- Results repo (private): `{RESULTS_REPO}` at revision `{R['meta']['results_repo_revision']}`.")
    w(f"- Environment as recorded: Python {env['versions']['python']}, torch {env['versions']['torch']} (CUDA "
      f"{env['cuda_torch']}). The brief for this report specified torch 2.11+cu128; the recorded version is the one "
      f"above. Package manager / environment tool: {env['package_manager']} (the shell prompt showed an environment "
      "named `mksq`).")
    w(f"- Runtimes: training step (memory probe, model loading, training and dev evaluations) {rt['training_step_h']} h, "
      f"of which training {t['training_time_h']} h; CTranslate2 conversion {rt['ct2_conversion_s']} s, final "
      f"evaluation {rt['final_eval_step_h']} h, whole final invocation {rt['final_invocation_runtime']}; analysis script "
      f"{rt['analysis_script_runtime']}. The first invocation ({inv[0]['started']} UTC) prepared all data and then "
      f"failed at training ({inv[0]['error']}); the second ({inv[-1]['started']} UTC) loaded the prepared data from "
      "disk and completed.")
    w("\n```bash\ncd ~/VEZILKA-MK-SQ-TRANSLATOR\n"
      f"git checkout {R['code']['training_commit']}\n"
      "python3 scripts/smoke_test_nllb_run.py --quick           # preflight, ~5 min\n"
      "python3 scripts/train_nllb_gazette_verbis_ct2.py        # training + export + evaluation\n"
      f"git checkout {R['meta']['report_git_head']}\n"
      "python3 scripts/analyze_gazette_results.py              # overlap, terminology, number fidelity\n"
      f"python3 experiments/{EXPERIMENT_ID}/build_report.py    # this report (bootstrap, tables, figures)\n```\n")
    # 9
    w("## 9. Next steps\n")
    w("- **E02_gazette_only (arm M1):** the same configuration without Verbis data, to isolate the lexical contribution "
      "(RQ2–RQ4).")
    w("- Manual review of 50 TSR misses and 50 number mismatches, to separate heuristic failures, formatting "
      "differences and genuine errors.\n")
    return "\n".join(L) + "\n"


# ============================================================================
# 9. Self-check
# ============================================================================
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)*(?![\w])")


def _numbers_in(obj, out: set) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _numbers_in(k, out)
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        for nd in (0, 1, 2, 3, 4):
            out.add(f"{obj:.{nd}f}")
        out.add(str(obj))
        out.add(str(abs(obj)))
        if isinstance(obj, float) and obj.is_integer():
            out.add(str(int(obj)))
    elif isinstance(obj, str):
        out.update(m.group(0).replace(",", "") for m in _NUM.finditer(obj))


def self_check(R: dict, paths: dict) -> None:
    allowed: set = set()
    _numbers_in(R, allowed)
    allowed.update(a.lstrip("-") for a in list(allowed))
    text = (HERE / "REPORT.md").read_text(encoding="utf-8")
    body = []
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("![") or line.startswith("|---"):
            continue
        line = re.sub(r"`[^`]*`", "", line)  # code spans: hashes, paths, signatures
        line = re.sub(r"§\d+(\.\d+)?", "", line)  # section references
        line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?", "", line)  # timestamps
        body.append(line)
    bad = sorted({m.group(0) for line in body for m in _NUM.finditer(line) if m.group(0).replace(",", "") not in allowed})
    if bad:
        sys.exit(f"SELF-CHECK FAILED: numbers in REPORT.md not found in results.json: {bad}")
    outputs = [HERE / "REPORT.md", HERE / "results.json", HERE / "MANIFEST.json", *TAB.glob("*.tex")]
    blob = "\n".join(p.read_text(encoding="utf-8").lower() for p in outputs)
    verbis = pl.read_parquet(paths["data/verbis/verbis_mk_sq.parquet"])
    strings = {s.lower().strip() for c in verbis.columns for s in verbis[c].drop_nulls().to_list() if len(s.strip()) >= 10}
    for f in ("data/verbis/verbis_terms.parquet", "data/verbis/verbis_defs.parquet"):
        v = pl.read_parquet(paths[f])
        strings |= {s.lower().strip() for c in ("mk", "sq") for s in v[c].drop_nulls().to_list() if len(s.strip()) >= 10}
    leaked = sorted(s for s in strings if s in blob)
    if leaked:
        sys.exit(f"SELF-CHECK FAILED: {len(leaked)} Verbis strings appear in the outputs")
    print(f"self-check passed: {len(set(m.group(0) for line in body for m in _NUM.finditer(line)))} distinct numbers "
          f"traced to results.json; {len(strings):,} Verbis strings checked, none present in {len(outputs)} output files")


if __name__ == "__main__":
    main()
