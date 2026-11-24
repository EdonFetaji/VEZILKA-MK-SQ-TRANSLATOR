#!/usr/bin/env python
"""Build the experiment report for E02_gazette_only from recorded artifacts only.

    uv run python experiments/E02_gazette_only/build_report.py

Nothing is trained or translated. The script
  1. downloads E02's source artifacts from its PRIVATE results repo at the revision in
     experiments/registry.csv, and E01's cached translations at E01's registry revision, and
     records size / sha256 / last commit of each (MANIFEST.json);
  2. reads the recorded facts (run_info.json, trainer_state.json, progress.log, CSVs) of E02,
     and E01's committed files in experiments/E01_gazette_verbis_mix/;
  3. computes paired bootstrap significance on the cached translations (sacrebleu PairedTest,
     1000 resamples, SACREBLEU_SEED=42): base vs E02, E02 HF vs E02 CT2, and E01 vs E02;
  4. writes results.json, figures/*.png, tables/*.tex (booktabs) and REPORT.md;
  5. self-checks: every number in REPORT.md appears in results.json, and no Verbis entry text
     appears in any output file.
E02 differs from E01 only in data.verbis.use_for_training (checked against both configs).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ["SACREBLEU_SEED"] = "42"  # read by sacrebleu.significance.PairedTest at construction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from huggingface_hub import HfApi, hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from mksq import experiment as X

EXPERIMENT_ID = "E02_gazette_only"
REFERENCE_ID = "E01_gazette_verbis_mix"
BASELINE_ID = "E00_base"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
ALPHA = 0.05
NR = "NOT RECORDED"

ROOT = X.ROOT
E01_DIR = ROOT / "experiments" / REFERENCE_ID
FIG, TAB = HERE / "figures", HERE / "tables"
BEST_STEP_CKPT = "checkpoints/checkpoint-3650/trainer_state.json"
HF_FILES = [
    "progress.log", "config.json", "generation_config.json", "provenance.json",
    "eval/run_info.json", "eval/train_log.csv", "eval/dev_curve.csv",
    "eval/gazette_test_results.csv", "eval/flores_results.csv", "eval/ntrex_results.csv",
    "eval/gazette_test_overlap_results.csv", "eval/terminology_results.csv", "eval/number_fidelity_results.csv",
    "eval/translations_gazette_test.parquet", "eval/translations_flores.parquet", "eval/translations_ntrex.parquet",
    BEST_STEP_CKPT, "data/verbis/verbis_terms.parquet", "data/verbis/verbis_defs.parquet",
]
E01_HF_FILES = ["eval/translations_gazette_test.parquet", "eval/translations_flores.parquet", "eval/translations_ntrex.parquet"]
TESTSETS = (("gazette_test", "Gazette test"), ("flores", "FLORES+ devtest"), ("ntrex", "NTREX-128"))
DIRECTIONS = (("mk_sq", "MK→SQ"), ("sq_mk", "SQ→MK"))


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


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
# 1. Fetch sources + manifest
# ============================================================================
def fetch(api: HfApi, repo: str, rev: str, files: list[str]) -> tuple[dict, list]:
    if api.model_info(repo).private is not True:
        sys.exit(f"ERROR: {repo} is not private -- refusing to read or cite it")
    paths, manifest = {}, []
    last = {p.path: p for p in api.get_paths_info(repo, files, expand=True, revision=rev)}
    for f in files:
        local = Path(hf_hub_download(repo, f, revision=rev))
        paths[f] = local
        lc = getattr(last.get(f), "last_commit", None)
        manifest.append({"artifact": f, "location": f"hf://{repo}/{f}", "hf_revision": rev,
                         "last_commit": getattr(lc, "oid", None),
                         "last_commit_date": str(getattr(lc, "date", None)) if lc else None,
                         "size_bytes": local.stat().st_size, "sha256": X.sha256_file(local)})
    return paths, manifest


def local_manifest(files: list[Path]) -> list[dict]:
    head = X.git_head()
    out = []
    for p in files:
        log = X.git("log", "-1", "--format=%H", "--", rel(p)) if p.is_relative_to(ROOT) else ""
        out.append({"artifact": rel(p), "location": f"git:{rel(p)}", "git_head": head, "last_commit": log or None,
                    "size_bytes": p.stat().st_size, "sha256": X.sha256_file(p)})
    return out


# ============================================================================
# 2. Parse recorded facts
# ============================================================================
def parse_progress(text: str) -> dict:
    lines = text.splitlines()
    ts = lambda ln: ln[:19]
    starts = [i for i, ln in enumerate(lines) if "| run started" in ln or ln[22:].startswith("INFO    | start | QUICK PREFLIGHT")]
    invocations = []
    for k, i in enumerate(starts):
        seg = lines[i:starts[k + 1] if k + 1 < len(starts) else len(lines)]
        quick = "QUICK PREFLIGHT" in lines[i]
        commit = next((m.group(1) for ln in seg if (m := re.search(r'"git_commit": "([0-9a-f]{40})"', ln))), None)
        failed = next((ln for ln in seg if "| ERROR" in ln and "FAILED" in ln), None)
        steps_done = [int(m.group(1)) for ln in seg if (m := re.search(r"\| step +(\d+) .+? \| ===== done in", ln))]
        resumed = next((int(m.group(1)) for ln in seg if (m := re.search(r"resuming from .*checkpoint-(\d+)", ln))), None)
        trained = [int(m.group(1)) for ln in seg if (m := re.search(r"\| (?:step |dev @ step )(\d+)(?:/\d+ \(|:)", ln))]
        trained_to = max(trained, default=None)
        err = None
        if failed:
            err = failed.split("FAILED", 1)[1].strip(" :")
            err = re.sub(r"^after \d+h\d+m\d+s: ", "", err)
            err = re.sub(r"[0-9a-f]{64}", "<sha256>", err.split(". ")[0])[:90]
        invocations.append({"started": ts(lines[i]), "kind": "smoke test (--quick)" if quick else "full run",
                            "git_commit": commit or NR, "outcome": "failed" if failed else "completed",
                            "error": err, "steps_completed": steps_done, "resumed_from_step": resumed,
                            "last_logged_train_step": trained_to})
    dev = {}
    for ln in lines:
        m = re.search(r"dev @ step (\d+): mk→sq BLEU ([\d.]+) chrF\+\+ ([\d.]+) \| sq→mk BLEU ([\d.]+) chrF\+\+ ([\d.]+) \| "
                      r"mean chrF\+\+ ([\d.]+)", ln)
        if m:  # last occurrence wins (a step re-evaluated after a resume)
            dev[int(m.group(1))] = {"step": int(m.group(1)), "bleu_mk_sq": float(m.group(2)), "chrf_mk_sq": float(m.group(3)),
                                    "bleu_sq_mk": float(m.group(4)), "chrf_sq_mk": float(m.group(5)),
                                    "chrf_mean": float(m.group(6)), "source": "progress.log"}
    final = lines[starts[-1]:]
    step_s = {}
    for ln in final:
        m = re.search(r"\| step +(\d+) (.+?) \| ===== done in (\d+)h(\d+)m(\d+)s", ln)
        if m:
            step_s[int(m.group(1))] = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
    throughput = {}
    for ln in final:
        m = re.search(r"\| (\w+) / (\w+) / (\w+): (\d+) / (\d+) sentences \(([\d.]+) sent/s", ln)
        if m and m.group(4) == m.group(5):
            throughput[f"{m.group(1)}/{m.group(2)}/{m.group(3)}"] = float(m.group(6))
    return {"invocations": invocations, "dev_from_log": [dev[k] for k in sorted(dev)], "final_step_seconds": step_s,
            "eval_throughput_sent_per_s": throughput,
            "ct2_device_fallback_to_cpu": any("CTranslate2 cannot use this GPU" in ln for ln in final),
            "final_evaluation_finished": next((ts(ln) for ln in final if "step 12 FINAL EVALUATION | ===== done" in ln), NR)}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff() -> dict:
    a, b = flatten(X.load_config(REFERENCE_ID)), flatten(X.load_config(EXPERIMENT_ID))
    diff = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    cmp_ = X.load_config(EXPERIMENT_ID)["comparison"]
    allowed = cmp_["experimental_differences"] + cmp_["bookkeeping_differences"]
    covered = lambda k: any(k == p or k.startswith(p + ".") for p in allowed)
    undeclared = [k for k in diff if not covered(k)]
    if undeclared:
        sys.exit(f"ERROR: E01 and E02 configs differ in undeclared keys: {undeclared}")
    experimental = [k for k in diff if any(k == p or k.startswith(p + ".") for p in cmp_["experimental_differences"])]
    return {"against": cmp_["against"], "experimental_differences": experimental,
            "values": {k: {"E01": a.get(k), "E02": b.get(k)} for k in experimental},
            "bookkeeping_differences": [k for k in diff if k not in experimental], "undeclared": undeclared}


# ============================================================================
# 3. Significance (paired bootstrap on cached translations)
# ============================================================================
def paired(baseline: tuple[str, list], system: tuple[str, list], refs: list) -> dict:
    out = {}
    test = PairedTest([baseline, system], metrics={"bleu": BLEU(), "chrf": CHRF(word_order=2)},
                      references=[refs], test_type="bs", n_samples=N_BOOTSTRAP)
    sigs, scores = test()
    names = [k for k in scores if k != "System"]
    for metric, name in zip(("bleu", "chrf"), names, strict=True):
        bl, sy = scores[name]
        for r in (bl, sy):
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


def significance(paths: dict, e01_paths: dict) -> dict:
    res = {"base_vs_finetuned_hf": {}, "finetuned_hf_vs_finetuned_ct2": {}, "e01_finetuned_hf_vs_e02_finetuned_hf": {}}
    for ts, _ in TESTSETS:
        tr = pl.read_parquet(paths[f"eval/translations_{ts}.parquet"])
        t1 = pl.read_parquet(e01_paths[f"eval/translations_{ts}.parquet"])
        for d, _ in DIRECTIONS:
            blk, b1 = tr.filter(pl.col("direction") == d), t1.filter(pl.col("direction") == d)
            refs = blk["reference"].to_list()
            # E01 and E02 must have translated the same sentences in the same order (and the same base outputs)
            assert b1["source"].to_list() == blk["source"].to_list() and b1["reference"].to_list() == refs, (ts, d)
            assert b1["base"].to_list() == blk["base"].to_list(), (ts, d)
            print(f"  bootstrap {ts} {d} ({blk.height} sentences)", flush=True)
            k = f"{ts}/{d}"
            res["base_vs_finetuned_hf"][k] = paired(("base", blk["base"].to_list()), ("finetuned_hf", blk["finetuned_hf"].to_list()), refs)
            res["finetuned_hf_vs_finetuned_ct2"][k] = paired(("finetuned_hf", blk["finetuned_hf"].to_list()),
                                                             ("finetuned_ct2", blk["finetuned_ct2"].to_list()), refs)
            res["e01_finetuned_hf_vs_e02_finetuned_hf"][k] = paired(("E01", b1["finetuned_hf"].to_list()),
                                                                    ("E02", blk["finetuned_hf"].to_list()), refs)
    return res


# ============================================================================
# 4. Assemble results.json
# ============================================================================
def assemble(paths: dict, meta: dict) -> dict:
    ri = json.loads(paths["eval/run_info.json"].read_text())
    ri1 = json.loads((E01_DIR / "run_info.json").read_text())
    st = json.loads(paths[BEST_STEP_CKPT].read_text())
    prog = parse_progress(paths["progress.log"].read_text(encoding="utf-8"))
    ta, ds, ds1, summ = ri["training_args"], ri["dataset_sizes"], ri1["dataset_sizes"], ri["training_summary"]
    tl = pl.read_csv(paths["eval/train_log.csv"])
    dc_csv = csv_rows(paths["eval/dev_curve.csv"])
    analysis_commit = {r["git_commit"] for f in ("gazette_test_overlap", "terminology", "number_fidelity")
                       for r in csv_rows(paths[f"eval/{f}_results.csv"])}
    assert len(analysis_commit) == 1, analysis_commit
    main = {ts: csv_rows(paths[f"eval/{ts}_results.csv"]) for ts, _ in TESTSETS}
    main1 = {ts: csv_rows(E01_DIR / "results" / f"{ts}_results.csv") for ts, _ in TESTSETS}
    keep = ("model", "direction", "n_sentences", "bleu", "chrf", "bleu_delta_vs_base", "chrf_delta_vs_base")
    st1 = json.loads((E01_DIR / "results.json").read_text())["training"]
    return {
        "experiment": {"id": EXPERIMENT_ID, "paper_arm": "M1", "reference_id": REFERENCE_ID, "reference_arm": "M7-mix + D3",
                       "baseline_id": BASELINE_ID, "baseline_arm": "M0"},
        "meta": meta,
        "config_comparison": config_diff(),
        "code": {"training_commit_final_invocation": ri["git"]["commit"], "training_commit_dirty": ri["git"]["dirty"],
                 "analysis_commit": analysis_commit.pop(), "report_git_head": meta["report_git_head"],
                 "code_diff_between_training_commits": X.git("diff", "--stat", "bd3c440", ri["git"]["commit"]).splitlines()},
        "data": {
            "gazette": {"train": ds["gazette_train"], "dev": ds["gazette_dev"], "test": ds["gazette_test"]},
            "flores_devtest": ds["flores_devtest"], "ntrex": ds["ntrex"],
            "leakage_removed_gazette": ds["leakage_removed"]["gazette"],
            "train_clean_pairs": ds["train_clean_pairs"], "train_bidirectional": ds["train_bidirectional"],
            "train_dropped_over_max_tokens": ds["train_dropped_over_max_tokens"],
            "train_dropped_pct": round(100 * ds["train_dropped_over_max_tokens"] / ds["train_bidirectional"], 2),
            "train_tokenized": ds["train_tokenized"], "max_tokens": ri["config"]["data"]["max_tokens"], "composition": ds["train_composition"],
            "verbis_loaded_for_evaluation_only": {"terms": ds["verbis_terms"], "definitions_kept": ds["verbis_defs"]},
            "e01": {"train_tokenized": ds1["train_tokenized"], "composition": ds1["train_composition"],
                    "leakage_removed_gazette": ds1["leakage_removed"]["gazette"]},
            "gazette_examples_same_as_e01": ds["train_composition"]["gazette"] == ds1["train_composition"]["gazette"],
            "e02_share_of_e01_examples_pct": round(100 * ds["train_tokenized"] / ds1["train_tokenized"], 1),
        },
        "training": {
            "base_model": ri["config"]["base_model"]["name"], "base_model_revision": ri["config"]["base_model"]["revision"],
            "optimizer": ta["optim"], "learning_rate": ta["learning_rate"], "lr_scheduler": ta["lr_scheduler_type"],
            "warmup_steps": ta["warmup_steps"], "epochs": ta["num_train_epochs"], "label_smoothing": ta["label_smoothing_factor"],
            "weight_decay": ta["weight_decay"], "max_grad_norm": ta["max_grad_norm"],
            "per_device_batch": ta["per_device_train_batch_size"], "grad_accumulation": ta["gradient_accumulation_steps"],
            "effective_batch": ta["per_device_train_batch_size"] * ta["gradient_accumulation_steps"],
            "gradient_checkpointing": ta["gradient_checkpointing"], "seed": ta["seed"], "data_seed": ta["data_seed"],
            "eval_steps": ta["eval_steps"], "metric_for_best_model": ta["metric_for_best_model"],
            "total_steps": st["global_step"], "max_steps": st["max_steps"],
            "best_step": st["best_global_step"], "best_dev_chrf_mean": round(st["best_metric"], 2),
            "training_time_h": summ["total_training_time_h"], "peak_gpu_mem_gb": summ["peak_gpu_mem_gb"],
            "samples_per_sec_median": round(float(tl["samples_per_sec"].median()), 1),
            "loss_first_logged": float(tl["loss"][0]), "loss_last_logged": float(tl["loss"][-1]),
            "dev_curve": prog["dev_from_log"],
            "dev_curve_csv_steps": [r["step"] for r in dc_csv],
            "dev_curve_csv_missing_steps": [r["step"] for r in prog["dev_from_log"] if r["step"] not in {x["step"] for x in dc_csv}],
            "e01": {"total_steps": st1["total_steps"], "best_step": st1["best_step"], "best_dev_chrf_mean": st1["best_dev_chrf_mean"],
                    "training_time_h": st1["training_time_h"], "dev_curve": st1["dev_curve"]},
        },
        "environment": {"gpu": ri["gpu"]["gpu_name"], "vram_total_gb": ri["gpu"]["vram_total_gb"], "versions": ri["versions"],
                        "cuda_torch": ri["gpu"]["cuda_torch"],
                        "ct2_device": "cpu (int8)" if prog["ct2_device_fallback_to_cpu"] else "cuda"},
        "runtime": {"invocations": prog["invocations"], "final_step_seconds": prog["final_step_seconds"],
                    "final_evaluation_finished": prog["final_evaluation_finished"],
                    "eval_throughput_sent_per_s": prog["eval_throughput_sent_per_s"]},
        "evaluation": {"bleu_signature": main["gazette_test"][0]["bleu_signature"],
                       "chrf_signature": main["gazette_test"][0]["chrf_signature"]},
        "results_main": {ts: [{k: r[k] for k in keep} for r in rows] for ts, rows in main.items()},
        "results_e01": {ts: [{k: r[k] for k in keep} for r in rows] for ts, rows in main1.items()},
        "overlap": csv_rows(paths["eval/gazette_test_overlap_results.csv"]),
        "terminology": csv_rows(paths["eval/terminology_results.csv"]),
        "number_fidelity": csv_rows(paths["eval/number_fidelity_results.csv"]),
        "e01_overlap": csv_rows(E01_DIR / "analysis/gazette_test_overlap_results.csv"),
        "e01_terminology": csv_rows(E01_DIR / "analysis/terminology_results.csv"),
        "e01_number_fidelity": csv_rows(E01_DIR / "analysis/number_fidelity_results.csv"),
        "analysis_constants": {"bootstrap_resamples": N_BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "alpha": ALPHA,
                               "ci_level_pct": 95, "min_p_resolution": round(1 / (N_BOOTSTRAP + 1), 4)},
    }


def pick(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def derive(R: dict) -> None:
    sig = R["significance"]
    e = sig["e01_finetuned_hf_vs_e02_finetuned_hf"]
    ext = {k: v for k, v in e.items() if not k.startswith("gazette_test")}
    ind = {k: v for k, v in e.items() if k.startswith("gazette_test")}
    rng = lambda blk, m: [min(v[m]["delta"] for v in blk.values()), max(v[m]["delta"] for v in blk.values())]
    R["e01_vs_e02_summary"] = {
        "in_domain": {"bleu_delta_range": rng(ind, "bleu"), "chrf_delta_range": rng(ind, "chrf"),
                      "n_tests": 2 * len(ind), "n_significant": sum(v[m]["significant"] for v in ind.values() for m in ("bleu", "chrf")),
                      "significant": [f"{k}/{m}" for k, v in ind.items() for m in ("bleu", "chrf") if v[m]["significant"]]},
        "general_domain": {"bleu_delta_range": rng(ext, "bleu"), "chrf_delta_range": rng(ext, "chrf"),
                           "n_tests": 2 * len(ext),
                           "n_significant_e02_lower": sum(v[m]["significant"] and v[m]["delta"] < 0 for v in ext.values() for m in ("bleu", "chrf")),
                           "not_significant": [f"{k}/{m}" for k, v in ext.items() for m in ("bleu", "chrf") if not v[m]["significant"]]},
    }
    b = sig["base_vs_finetuned_hf"]
    bext = {k: v for k, v in b.items() if not k.startswith("gazette_test")}
    R["forgetting_summary"] = {
        "e02": {"bleu_delta_range": rng(bext, "bleu"), "chrf_delta_range": rng(bext, "chrf"), "n_tests": 2 * len(bext),
                "n_significant_declines": sum(v[m]["significant"] and v[m]["delta"] < 0 for v in bext.values() for m in ("bleu", "chrf"))},
        "e01": {k: R["e01_reference_summaries"]["forgetting_summary"][k] for k in ("bleu_delta_range", "chrf_delta_range")},
    }
    q = sig["finetuned_hf_vs_finetuned_ct2"]
    R["quantisation_summary"] = {
        "max_abs_bleu_delta": max(abs(v["bleu"]["delta"]) for v in q.values()),
        "max_abs_chrf_delta": max(abs(v["chrf"]["delta"]) for v in q.values()),
        "n_tests": 2 * len(q), "n_significant": sum(v[m]["significant"] for v in q.values() for m in ("bleu", "chrf"))}
    ov = {}
    for c in ("exact_template", "near_duplicate", "novel", "all"):
        for d, _ in DIRECTIONS:
            r2, r1 = pick(R["overlap"], **{"class": c, "model": "finetuned_hf", "direction": d}), pick(
                R["e01_overlap"], **{"class": c, "model": "finetuned_hf", "direction": d})
            ov[f"{c}/{d}"] = {"bleu": round(r2["bleu"] - r1["bleu"], 2), "chrf": round(r2["chrf"] - r1["chrf"], 2)}
    shares = {c: pick(R["overlap"], **{"class": c, "model": "base", "direction": "mk_sq"}) for c in ("exact_template", "near_duplicate", "novel")}
    shares1 = {c: pick(R["e01_overlap"], **{"class": c, "model": "base", "direction": "mk_sq"}) for c in shares}
    R["overlap_summary"] = {"n_pairs": {c: r["n_pairs"] for c, r in shares.items()}, "pct_pairs": {c: r["pct_pairs"] for c, r in shares.items()},
                            "e01_pct_pairs": {c: r["pct_pairs"] for c, r in shares1.items()},
                            "pct_template_or_near_duplicate": round(shares["exact_template"]["pct_pairs"] + shares["near_duplicate"]["pct_pairs"], 2),
                            "e02_minus_e01_finetuned_hf": ov}
    tsr = {}
    for d, _ in DIRECTIONS:
        t2 = {r["model"]: r for r in R["terminology"] if r["direction"] == d}
        t1 = {r["model"]: r for r in R["e01_terminology"] if r["direction"] == d}
        tsr[d] = {"e02_minus_e01": round(t2["finetuned_hf"]["tsr"] - t1["finetuned_hf"]["tsr"], 2),
                  "rare_e02_minus_e01": round(t2["finetuned_hf"]["rare_tsr"] - t1["finetuned_hf"]["rare_tsr"], 2),
                  "e02_minus_reference": round(t2["finetuned_hf"]["tsr"] - t2["reference"]["tsr"], 2),
                  "e01_minus_reference": round(t1["finetuned_hf"]["tsr"] - t1["reference"]["tsr"], 2),
                  "rare_e02_successes": t2["finetuned_hf"]["n_rare_success"], "rare_e01_successes": t1["finetuned_hf"]["n_rare_success"],
                  "reference_identical": t1["reference"]["tsr"] == t2["reference"]["tsr"]}
    R["terminology_summary"] = tsr
    nf = {}
    for d, _ in DIRECTIONS:
        n2, n1 = pick(R["number_fidelity"], model="finetuned_hf", direction=d), pick(R["e01_number_fidelity"], model="finetuned_hf", direction=d)
        nf[d] = {"e02_minus_e01_with_numbers": round(n2["match_rate_with_numbers"] - n1["match_rate_with_numbers"], 2)}
    R["number_summary"] = nf
    tp = R["runtime"]["eval_throughput_sent_per_s"]
    R["runtime"]["ct2_gazette_sent_per_s"] = [tp.get("gazette_test/finetuned_ct2/mk_sq"), tp.get("gazette_test/finetuned_ct2/sq_mk")]
    R["runtime"]["final_eval_h"] = round(R["runtime"]["final_step_seconds"].get(12, 0) / 3600, 2)
    R["report_constants"] = {"n_test_sets": 3, "n_directions": 2, "n_invocations": len(R["runtime"]["invocations"]),
                             "quick_preflights": sum(i["kind"].startswith("smoke") for i in R["runtime"]["invocations"])}


# ============================================================================
# 5. Figures (from the CSVs / progress.log only)
# ============================================================================
C = {"base": "#2a78d6", "E01": "#8a5cd1", "E02": "#eb6834", "ct2": "#1baf7a", "reference": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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
    t = R["training"]
    # training curves: E02 loss and learning rate, E01 loss for reference
    tl, tl1 = pl.read_csv(paths["eval/train_log.csv"]), pl.read_csv(E01_DIR / "train_log.csv")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(tl1["step"], tl1["loss"], color=C["E01"], lw=1.5, label="E01 (Gazette + Verbis)")
    a1.plot(tl["step"], tl["loss"], color=C["E02"], lw=2, label="E02 (Gazette only)")
    a1.set_ylabel("training loss\n(label-smoothed, mean of 50 steps)")
    a1.legend(frameon=False, fontsize=8)
    a2.plot(tl["step"], tl["learning_rate"], color=C["E02"], lw=2)
    a2.set_ylabel("learning rate (E02)")
    a2.set_xlabel("optimizer step")
    for ax in (a1, a2):
        _style(ax)
        ax.axvline(t["best_step"], color=MUTED, lw=1, ls=":")
    a1.set_title(f"Training (E02 best checkpoint = step {t['best_step']}, dotted)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=200)
    plt.close(fig)
    # dev curve, E01 vs E02 (E02 points from progress.log: dev_curve.csv lacks the last step)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, col, title in zip(axes, ("chrf_mk_sq", "chrf_sq_mk", "chrf_mean"), ("MK→SQ", "SQ→MK", "mean of both"), strict=True):
        for exp, dev in (("E01", t["e01"]["dev_curve"]), ("E02", t["dev_curve"])):
            ax.plot([r["step"] for r in dev], [r[col] for r in dev], color=C[exp], lw=2, marker="o", ms=4, label=exp)
        ax.set_title(f"Gazette dev chrF++, {title}", loc="left")
        ax.set_xlabel("optimizer step")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "dev_curve.png", dpi=200)
    plt.close(fig)
    # E02 minus E01 per test set and direction, with significance marked
    e = R["significance"]["e01_finetuned_hf_vs_e02_finetuned_hf"]
    keys = [f"{ts}/{d}" for ts, _ in TESTSETS for d, _ in DIRECTIONS]
    short = {"gazette_test": "Gazette", "flores": "FLORES+", "ntrex": "NTREX-128"}
    labels = [f"{short[k.split('/')[0]]}\n{dict(DIRECTIONS)[k.split('/')[1]]}" for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, m, name in ((axes[0], "bleu", "BLEU"), (axes[1], "chrf", "chrF++")):
        vals = [e[k][m]["delta"] for k in keys]
        ax.bar(range(len(keys)), vals, color=[C["E02"] if e[k][m]["significant"] else GRID for k in keys], width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.08 if v >= 0 else -0.08), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(keys)), labels, fontsize=7)
        ax.set_title(f"E02 − E01, {name} (filled: p < {ALPHA}, paired bootstrap)", loc="left")
        _style(ax)
    lo = min(min(e[k][m]["delta"] for k in keys) for m in ("bleu", "chrf"))
    for ax in axes:
        ax.set_ylim(lo - 0.8, 1.2)
    fig.tight_layout()
    fig.savefig(FIG / "e01_vs_e02.png", dpi=200)
    plt.close(fig)
    # terminology, E00 / E01 / E02 with the reference line
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col, title in ((axes[0], "tsr", "All terms"), (axes[1], "rare_tsr", "Rare terms (≤ 5 train sentences)")):
        for j, (d, lab) in enumerate(DIRECTIONS):
            ref = pick(R["terminology"], model="reference", direction=d)[col]
            ax.hlines(ref, j - 0.42, j + 0.42, color=C["reference"], lw=2, ls="--", label="Human reference" if j == 0 else None)
            for k, (name, rows, m) in enumerate((("Base (E00)", R["terminology"], "base"), ("E01 HF", R["e01_terminology"], "finetuned_hf"),
                                                 ("E02 HF", R["terminology"], "finetuned_hf"))):
                v = pick(rows, model=m, direction=d)[col]
                ax.bar(j + (k - 1) * 0.27, v, width=0.25, color=(C["base"], C["E01"], C["E02"])[k], edgecolor="#fcfcfb", lw=1,
                       label=name if j == 0 else None)
                ax.text(j + (k - 1) * 0.27, v + 1.2, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks([0, 1], [lab for _, lab in DIRECTIONS])
        ax.set_ylim(0, 100)
        ax.set_title(f"Term Success Rate, {title}", loc="left")
        _style(ax)
    axes[0].set_ylabel("TSR (%)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "terminology.png", dpi=200)
    plt.close(fig)


# ============================================================================
# 6. Tables (Markdown + LaTeX booktabs from the same rows)
# ============================================================================
def f2(x) -> str:
    return "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))


def pfmt(p: float) -> str:
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("→", r"$\rightarrow$").replace("≤", r"$\leq$").replace("Δ", r"$\Delta$")
            .replace("−", "--").replace("–", "--"))


def write_table(name: str, caption: str, header: list[str], rows: list[list], source: str) -> str:
    numeric = lambda i: all(isinstance(r[i], (int, float)) or r[i] is None or re.fullmatch(r"[<\[\d.,\-+ \]]+|–", str(r[i]))
                            for r in rows)
    align = "l" + "".join("r" if numeric(i) else "l" for i in range(1, len(header)))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(f2(c)) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{tex_escape(caption)} Source: \texttt{{{tex_escape(source)}}}.}}",
              rf"\label{{tab:e02-{name.replace('_', '-')}}}", r"\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    md = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    md += ["| " + " | ".join(f2(c) for c in r) + " |" for r in rows]
    return "\n".join(md) + f"\n\nSource: {source}\n"


def tables(R: dict) -> dict:
    T = {}
    dn, tn = dict(DIRECTIONS), dict(TESTSETS)
    sb, se = R["significance"]["base_vs_finetuned_hf"], R["significance"]["e01_finetuned_hf_vs_e02_finetuned_hf"]
    rows = []
    for ts, _ in TESTSETS:
        for d, _ in DIRECTIONS:
            k = f"{ts}/{d}"
            b = pick(R["results_main"][ts], model="base", direction=d)
            e1 = pick(R["results_e01"][ts], model="finetuned_hf", direction=d)
            e2 = pick(R["results_main"][ts], model="finetuned_hf", direction=d)
            rows.append([tn[ts], dn[d], b["bleu"], e1["bleu"], e2["bleu"], pfmt(sb[k]["bleu"]["p_value"]), se[k]["bleu"]["delta"],
                         pfmt(se[k]["bleu"]["p_value"]), b["chrf"], e1["chrf"], e2["chrf"], pfmt(sb[k]["chrf"]["p_value"]),
                         se[k]["chrf"]["delta"], pfmt(se[k]["chrf"]["p_value"])])
    T["main"] = write_table(
        "main_results", "BLEU and chrF++ (beam 4) of the base model (E00), E01 (Gazette + Verbis) and E02 (Gazette only), HF bf16 models. "
        "p vs E00: paired bootstrap, base vs E02; Δ and p vs E01: E02 − E01 (1,000 resamples, seed 42).",
        ["Test set", "Dir.", "BLEU E00", "BLEU E01", "BLEU E02", "p vs E00", "ΔBLEU vs E01", "p vs E01",
         "chrF++ E00", "chrF++ E01", "chrF++ E02", "p vs E00", "ΔchrF++ vs E01", "p vs E01"], rows,
        "eval/*_results.csv (E02), experiments/E01_gazette_verbis_mix/results/*.csv; significance: results.json")
    rows = []
    for k, v in se.items():
        ts, d = k.split("/")
        for m, lab in (("bleu", "BLEU"), ("chrf", "chrF++")):
            a, s_ = v[m]["baseline"], v[m]["system"]
            rows.append([tn[ts], dn[d], lab, a["score"], f"[{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}]", s_["score"],
                         f"[{s_['ci95'][0]:.2f}, {s_['ci95'][1]:.2f}]", v[m]["delta"], pfmt(v[m]["p_value"])])
    T["sig"] = write_table("significance_e01_vs_e02", "Paired bootstrap, E01 vs E02 (HF bf16): scores with 95% confidence intervals of the bootstrap mean.",
                           ["Test set", "Dir.", "Metric", "E01", "E01 95% CI", "E02", "E02 95% CI", "Δ (E02 − E01)", "p"], rows,
                           "eval/translations_*.parquet of both experiments (recomputed); results.json#significance")
    rows = []
    for c in ("exact_template", "near_duplicate", "novel", "all"):
        for d, _ in DIRECTIONS:
            q = {"class": c, "direction": d}
            b, f1, f2_ = pick(R["overlap"], model="base", **q), pick(R["e01_overlap"], model="finetuned_hf", **q), pick(R["overlap"], model="finetuned_hf", **q)
            dd = R["overlap_summary"]["e02_minus_e01_finetuned_hf"][f"{c}/{d}"]
            rows.append([c.replace("_", " "), dn[d], b["n_pairs"], b["pct_pairs"], b["bleu"], f1["bleu"], f2_["bleu"], dd["bleu"],
                         b["chrf"], f1["chrf"], f2_["chrf"], dd["chrf"]])
    T["overlap"] = write_table("overlap", "Gazette test by overlap with the Gazette training sentences (same classes as E01: normalised, near duplicate = character 5-gram Jaccard ≥ 0.8). Δ = E02 − E01; not significance-tested.",
                               ["Class", "Dir.", "Pairs", "% pairs", "BLEU E00", "BLEU E01", "BLEU E02", "ΔBLEU",
                                "chrF++ E00", "chrF++ E01", "chrF++ E02", "ΔchrF++"], rows,
                               "eval/gazette_test_overlap_results.csv (E02), experiments/E01_gazette_verbis_mix/analysis/")
    rows = []
    for d, _ in DIRECTIONS:
        for name, src, m in (("Human reference", R["terminology"], "reference"), ("Base (E00)", R["terminology"], "base"),
                             ("E01 HF", R["e01_terminology"], "finetuned_hf"), ("E02 HF", R["terminology"], "finetuned_hf"),
                             ("E02 CT2 int8", R["terminology"], "finetuned_ct2")):
            r = pick(src, model=m, direction=d)
            rows.append([name, dn[d], r["n_term_occurrences"], r["n_success"], r["tsr"], r["n_rare_occurrences"], r["n_rare_success"], r["rare_tsr"]])
    T["terms"] = write_table("terminology", "Term Success Rate (micro, %) with Verbis terms (source term ≥ 5 characters, whole-word match); rare = in ≤ 5 Gazette training sentences. E02 never saw Verbis data.",
                             ["System", "Dir.", "Term occurrences", "Successes", "TSR", "Rare occurrences", "Rare successes", "Rare TSR"], rows,
                             "eval/terminology_results.csv (E02), experiments/E01_gazette_verbis_mix/analysis/terminology_results.csv")
    rows = []
    for d, _ in DIRECTIONS:
        for name, src, m in (("Human reference", R["number_fidelity"], "reference"), ("Base (E00)", R["number_fidelity"], "base"),
                             ("E01 HF", R["e01_number_fidelity"], "finetuned_hf"), ("E02 HF", R["number_fidelity"], "finetuned_hf"),
                             ("E02 CT2 int8", R["number_fidelity"], "finetuned_ct2")):
            r = pick(src, model=m, direction=d)
            rows.append([name, dn[d], r["match_rate_all"], r["n_with_numbers"], r["match_rate_with_numbers"]])
    T["numbers"] = write_table("number_fidelity", "Number fidelity: share of sentences whose multiset of digit sequences in the output equals the source's (%).",
                               ["System", "Dir.", "Match (all)", "Sentences with numbers", "Match (with numbers)"], rows,
                               "eval/number_fidelity_results.csv (E02), experiments/E01_gazette_verbis_mix/analysis/")
    rows = []
    for k, v in R["significance"]["finetuned_hf_vs_finetuned_ct2"].items():
        ts, d = k.split("/")
        rows.append([tn[ts], dn[d], v["bleu"]["baseline"]["score"], v["bleu"]["system"]["score"], v["bleu"]["delta"], pfmt(v["bleu"]["p_value"]),
                     v["chrf"]["baseline"]["score"], v["chrf"]["system"]["score"], v["chrf"]["delta"], pfmt(v["chrf"]["p_value"])])
    T["quant"] = write_table("quantisation", "E02 quantisation: HF bf16 (GPU) vs CTranslate2 int8 (CPU); Δ = CT2 − HF; paired bootstrap p-values.",
                             ["Test set", "Dir.", "BLEU HF", "BLEU CT2", "ΔBLEU", "p", "chrF++ HF", "chrF++ CT2", "ΔchrF++", "p"], rows,
                             "eval/*_results.csv, eval/translations_*.parquet; results.json#significance")
    e1dev = {r["step"]: r for r in R["training"]["e01"]["dev_curve"]}
    rows = [[r["step"], r["bleu_mk_sq"], r["chrf_mk_sq"], r["bleu_sq_mk"], r["chrf_sq_mk"], r["chrf_mean"],
             e1dev[r["step"]]["chrf_mean"] if r["step"] in e1dev else None] for r in R["training"]["dev_curve"]]
    T["dev"] = write_table("dev_curve", "E02 Gazette dev evaluation during training (greedy, 500 sentences per direction), with E01's mean at the same step. "
                           "Read from progress.log: eval/dev_curve.csv lacks the final step.",
                           ["Step", "BLEU MK→SQ", "chrF++ MK→SQ", "BLEU SQ→MK", "chrF++ SQ→MK", "chrF++ mean", "E01 chrF++ mean"], rows,
                           "progress.log (dev @ step lines); E01: experiments/E01_gazette_verbis_mix/results.json")
    t, dd = R["training"], R["data"]
    rows = [["Base model", f"{t['base_model']}@{t['base_model_revision'][:7]}"], ["Optimizer", t["optimizer"]],
            ["Learning rate", f"{t['learning_rate']:g}"], ["Schedule", f"{t['lr_scheduler']}, warmup {t['warmup_steps']} steps"],
            ["Epochs", str(t["epochs"])], ["Label smoothing", str(t["label_smoothing"])], ["Weight decay", str(t["weight_decay"])],
            ["Max grad norm", str(t["max_grad_norm"])], ["Precision", "bf16 mixed precision, TF32 matmul"],
            ["Batch (per device × accumulation)", f"{t['per_device_batch']} × {t['grad_accumulation']} = {t['effective_batch']}"],
            ["Gradient checkpointing", str(t["gradient_checkpointing"])], ["Seed / data seed", f"{t['seed']} / {t['data_seed']}"],
            ["Total optimizer steps (E01)", f"{t['total_steps']:,} ({t['e01']['total_steps']:,})"],
            ["Checkpoint selection", f"{t['metric_for_best_model']} every {t['eval_steps']} steps and at the end"],
            ["Best step / dev chrF++ mean (E01)", f"{t['best_step']} / {t['best_dev_chrf_mean']} ({t['e01']['best_step']} / {t['e01']['best_dev_chrf_mean']})"],
            ["Training time, h (E01)", f"{t['training_time_h']} ({t['e01']['training_time_h']})"]]
    T["hparams"] = write_table("hyperparameters", "E02 training configuration as recorded (identical to E01 by config comparison).", ["Setting", "Value"], rows,
                               "eval/run_info.json, checkpoints/checkpoint-3650/trainer_state.json")
    rows = [["Gazette train / dev / test (pairs)", f"{dd['gazette']['train']:,} / {dd['gazette']['dev']:,} / {dd['gazette']['test']:,}"],
            ["Gazette pairs removed as leakage", f"{dd['leakage_removed_gazette']:,}"],
            ["Clean training pairs", f"{dd['train_clean_pairs']:,}"], ["Bidirectional examples", f"{dd['train_bidirectional']:,}"],
            [f"Dropped by length filter (> {dd['max_tokens']} tokens)", f"{dd['train_dropped_over_max_tokens']:,} ({dd['train_dropped_pct']}%)"],
            ["Tokenised training examples (E01)", f"{dd['train_tokenized']:,} ({dd['e01']['train_tokenized']:,})"],
            ["Gazette examples, E02 = E01", str(dd["gazette_examples_same_as_e01"])]]
    T["data"] = write_table("data", "E02 training data: the Gazette part of E01's mixture, without Verbis.", ["Quantity", "Value"], rows,
                            "eval/run_info.json (dataset_sizes) of both experiments")
    rows = [[i["started"], i["kind"], (i["git_commit"] or NR)[:7], i["outcome"],
             (f"resumed at step {i['resumed_from_step']}; " if i["resumed_from_step"] else "")
             + (f"trained to step {i['last_logged_train_step']}; " if i["last_logged_train_step"] else "")
             + (i["error"] or "")] for i in R["runtime"]["invocations"]]
    T["inv"] = write_table("invocations", "Invocations of the training script for E02 (times as logged).",
                           ["Started", "Kind", "Commit", "Outcome", "Notes"], rows, "progress.log")
    return T


# ============================================================================
# 7. REPORT.md
# ============================================================================
def name_of(key: str) -> str:
    for a, b in (("gazette_test", "Gazette test"), ("flores", "FLORES+"), ("ntrex", "NTREX-128"), ("mk_sq", "MK→SQ"),
                 ("sq_mk", "SQ→MK"), ("bleu", "BLEU"), ("chrf", "chrF++")):
        key = key.replace(a, b)
    return key.replace("/", " ")


def report_text(R: dict, T: dict) -> str:
    d, t, rt, cc = R["data"], R["training"], R["runtime"], R["config_comparison"]
    s, fs, qs, osu, ts_ = R["e01_vs_e02_summary"], R["forgetting_summary"], R["quantisation_summary"], R["overlap_summary"], R["terminology_summary"]
    term = lambda src, m, dd: pick(R[src], model=m, direction=dd)
    inv = rt["invocations"]
    full = [i for i in inv if i["kind"] == "full run"]
    L = []
    w = L.append
    w(f"# Experiment report: {R['experiment']['id']}\n")
    w(f"Gazette-only control (research-design arm {R['experiment']['paper_arm']}), compared with "
      f"{R['experiment']['reference_id']} (arm {R['experiment']['reference_arm']}) and the base model {R['experiment']['baseline_id']} "
      f"(arm {R['experiment']['baseline_arm']}). Generated {R['meta']['generated_at']} by `experiments/{EXPERIMENT_ID}/build_report.py` "
      f"from results repo revision `{R['meta']['results_repo_revision']}` and git commit `{R['meta']['report_git_head']}`. "
      "Every number below is in `results.json`; every artifact is listed with its sha256 in `MANIFEST.json`.\n")
    w("## 1. Summary\n")
    w(f"E02 repeats E01 exactly except that no Verbis data was used for training (the only experimental difference between "
      f"the two configs: `{', '.join(cc['experimental_differences'])}`). It was trained on the same "
      f"{d['composition']['gazette']:,} bidirectional Gazette examples that formed the Gazette part of E01's mixture, for "
      f"{t['total_steps']:,} optimizer steps ({t['training_time_h']} h).")
    w(f"On the Gazette test set E02 was marginally better than E01: E02 − E01 = {s['in_domain']['bleu_delta_range'][0]:+} to "
      f"{s['in_domain']['bleu_delta_range'][1]:+} BLEU and {s['in_domain']['chrf_delta_range'][0]:+} to "
      f"{s['in_domain']['chrf_delta_range'][1]:+} chrF++ ({s['in_domain']['n_significant']} of {s['in_domain']['n_tests']} "
      f"differences significant at p < {ALPHA}). Verbis training brought no in-domain benefit; the differences are a "
      "fraction of a point against a fine-tuning gain of more than 20 BLEU.")
    w(f"On the general-domain sets FLORES+ and NTREX-128, E02 scored below E01 in every comparison: "
      f"{s['general_domain']['bleu_delta_range'][0]} to {s['general_domain']['bleu_delta_range'][1]} BLEU and "
      f"{s['general_domain']['chrf_delta_range'][0]} to {s['general_domain']['chrf_delta_range'][1]} chrF++, "
      f"{s['general_domain']['n_significant_e02_lower']} of {s['general_domain']['n_tests']} significant. Relative to the base model, "
      f"E02 lost {abs(fs['e02']['bleu_delta_range'][1])} to {abs(fs['e02']['bleu_delta_range'][0])} BLEU on these sets "
      f"(E01: {abs(fs['e01']['bleu_delta_range'][1])} to {abs(fs['e01']['bleu_delta_range'][0])}). Adding Verbis data thus "
      "reduced, but did not prevent, the loss of general-domain quality.")
    w(f"Term Success Rate on the Gazette test was {term('terminology', 'finetuned_hf', 'mk_sq')['tsr']}% (MK→SQ) and "
      f"{term('terminology', 'finetuned_hf', 'sq_mk')['tsr']}% (SQ→MK) for E02 against "
      f"{term('e01_terminology', 'finetuned_hf', 'mk_sq')['tsr']}% and {term('e01_terminology', 'finetuned_hf', 'sq_mk')['tsr']}% for E01: "
      "Verbis training changed overall terminology use by less than a point. The only visible difference is on rare MK→SQ "
      f"terms, where E01 was ahead by {abs(ts_['mk_sq']['rare_e02_minus_e01'])} points ({ts_['mk_sq']['rare_e01_successes']} vs "
      f"{ts_['mk_sq']['rare_e02_successes']} of {term('terminology', 'finetuned_hf', 'mk_sq')['n_rare_occurrences']} occurrences); "
      f"on rare SQ→MK terms the two were equal ({ts_['sq_mk']['rare_e02_successes']} of "
      f"{term('terminology', 'finetuned_hf', 'sq_mk')['n_rare_occurrences']}).\n")
    w("## 2. What E02 isolates\n")
    w(f"`configs/{EXPERIMENT_ID}.yaml` and `configs/{REFERENCE_ID}.yaml` were compared key by key. The only experimental "
      f"difference is `{cc['experimental_differences'][0]}` (E01: {cc['values'][cc['experimental_differences'][0]]['E01']}, "
      f"E02: {cc['values'][cc['experimental_differences'][0]]['E02']}); the other differing keys are bookkeeping "
      f"({', '.join(f'`{k}`' for k in cc['bookkeeping_differences'])}), all declared in the `comparison` block. The run "
      f"also confirmed at start that the Gazette, FLORES+ and NTREX row counts and the Gazette-test and dev-subset content "
      f"fingerprints equal E01's (`data.expected`), and E01 and E02 produced identical base-model translations on all three test "
      f"sets (asserted by this script). Verbis was still loaded, for the terminology metric only "
      f"({d['verbis_loaded_for_evaluation_only']['terms']:,} term pairs).\n")
    w("This isolates the effect of the Verbis data together with the change it implies in training length: E02 had "
      f"{d['train_tokenized']:,} training examples ({d['e02_share_of_e01_examples_pct']}% of E01's {d['e01']['train_tokenized']:,}), "
      f"so one epoch was {t['total_steps']:,} instead of {t['e01']['total_steps']:,} optimizer steps, and the linear learning-rate "
      "schedule decayed correspondingly sooner. The experiment cannot separate the content of the Verbis examples from the extra "
      "training steps they caused.\n")
    w("## 3. Data\n")
    w(T["data"])
    w("## 4. Training and run history\n")
    w(f"Training used the hyperparameters below, identical to E01. The final invocation started at commit "
      f"`{R['code']['training_commit_final_invocation']}` with a clean working tree. The run took {len(full)} full invocations "
      f"(plus {R['report_constants']['quick_preflights']} quick preflights) because of three failures unrelated to the model: a data "
      "check comparing a fingerprint to a row count (fixed in `bd3c440`), a crash appending to `dev_curve.csv` after the step-2000 "
      "dev evaluation (fixed in `ccd58f9`) and a rejected (invalid) Hugging Face token. The crash happened before checkpoint 2000 was saved, "
      "so the final invocation resumed from checkpoint 1000; steps 1–1000 were trained at `bd3c440`, steps 1001–"
      f"{t['total_steps']} at `{R['code']['training_commit_final_invocation'][:7]}`. Between these commits only the dev-curve writer "
      f"changed (`{R['code']['code_diff_between_training_commits'][0].strip()}`). Data preparation (steps 2–7) ran at `bd3c440` and "
      "was loaded from disk afterwards.\n")
    w(T["inv"])
    w(f"The analysis step (overlap, terminology, numbers) failed in the final invocation on a variable-shadowing bug and was "
      f"re-run alone at commit `{R['code']['analysis_commit']}` (the training commit plus that fix) on the GPU machine's cached "
      "translations and processed data. The step that writes `DONE` and copies artifacts into `experiments/` never ran; the "
      "artifacts were imported by `scripts/registry/import_hf_artifacts.py` (see `SOURCES.json`).\n")
    w(T["hparams"])
    w(f"Hardware: {R['environment']['gpu']} ({R['environment']['vram_total_gb']} GB). Software: Python "
      f"{R['environment']['versions']['python']}, torch {R['environment']['versions']['torch']}, transformers "
      f"{R['environment']['versions']['transformers']}, ctranslate2 {R['environment']['versions']['ctranslate2']}, sacrebleu "
      f"{R['environment']['versions']['sacrebleu']}. Training ran at a median {t['samples_per_sec_median']} samples/s with a peak of "
      f"{t['peak_gpu_mem_gb']} GB GPU memory; the mean training loss fell from {t['loss_first_logged']} to {t['loss_last_logged']}.\n")
    w("## 5. Results\n")
    w(f"Beam 4 on all test sets. BLEU signature `{R['evaluation']['bleu_signature']}`; chrF++ signature "
      f"`{R['evaluation']['chrf_signature']}`. Paired bootstrap with {N_BOOTSTRAP:,} resamples (seed {BOOTSTRAP_SEED}); "
      f"{R['analysis_constants']['min_p_resolution']} is the smallest p-value this can produce.\n")
    w("### 5.1 Main results: E00, E01, E02\n")
    w(T["main"])
    w("![E02 minus E01](figures/e01_vs_e02.png)\n")
    w(T["sig"])
    w("### 5.2 Overlap-controlled results (Gazette test)\n")
    w(f"The overlap classes are computed against each model's own Gazette training sentences, which are the same in E01 and E02: "
      f"{osu['pct_pairs']['exact_template']}% exact templates, {osu['pct_pairs']['near_duplicate']}% near duplicates and "
      f"{osu['pct_pairs']['novel']}% novel pairs (E01: {osu['e01_pct_pairs']['exact_template']}%, "
      f"{osu['e01_pct_pairs']['near_duplicate']}%, {osu['e01_pct_pairs']['novel']}%).\n")
    w(T["overlap"])
    w("### 5.3 Terminology (Term Success Rate)\n")
    w("Same Verbis term list and matching rule as E01; the human-reference row is identical in both experiments "
      f"({ts_['mk_sq']['reference_identical'] and ts_['sq_mk']['reference_identical']}). TSR differences between E01 and E02 "
      "were not significance-tested.\n")
    w(T["terms"])
    w("![Terminology](figures/terminology.png)\n")
    w("### 5.4 Number fidelity\n")
    w(T["numbers"])
    w("### 5.5 Quantisation (HF bf16 vs CTranslate2 int8)\n")
    w(f"As in E01, CTranslate2 could not use the GPU and ran on CPU (int8). Across the {qs['n_tests']} paired comparisons the largest "
      f"absolute difference was {qs['max_abs_bleu_delta']} BLEU and {qs['max_abs_chrf_delta']} chrF++; {qs['n_significant']} "
      f"reached p < {ALPHA}.\n")
    w(T["quant"])
    w("### 5.6 Training dynamics\n")
    w(T["dev"])
    w(f"`eval/dev_curve.csv` holds steps {', '.join(str(x) for x in t['dev_curve_csv_steps'])} only; the dev evaluation at step "
      f"{', '.join(str(x) for x in t['dev_curve_csv_missing_steps'])} (the selected checkpoint) is recorded in `progress.log` and "
      "`trainer_state.json`, and the table and figure use the log.\n")
    w("![Dev curve](figures/dev_curve.png)\n\n![Training curves](figures/training_curves.png)\n")
    w("## 6. Interpretation\n")
    w(f"- **In-domain: no benefit from Verbis.** On the Gazette test E02 is at most "
      f"{max(abs(x) for x in s['in_domain']['bleu_delta_range'])} BLEU ahead of E01, and close to E01 on the novel subset too "
      f"(E02 − E01: {R['overlap_summary']['e02_minus_e01_finetuned_hf']['novel/mk_sq']['bleu']:+} and "
      f"{R['overlap_summary']['e02_minus_e01_finetuned_hf']['novel/sq_mk']['bleu']:+} BLEU). The in-domain gain reported for E01 "
      "comes from the Gazette data.")
    w(f"- **General domain: Verbis limits forgetting.** Both fine-tuned systems fall below the base model on FLORES+ and NTREX-128, "
      f"but E02 falls further ({fs['e02']['bleu_delta_range'][0]} to {fs['e02']['bleu_delta_range'][1]} BLEU vs E00, against "
      f"{fs['e01']['bleu_delta_range'][0]} to {fs['e01']['bleu_delta_range'][1]} for E01). The Verbis term and definition pairs are "
      "general vocabulary and short, non-gazette sentences, which plausibly keep the model closer to general-domain usage. The "
      "shorter E02 schedule is the other difference; fewer optimizer steps would not by themselves be expected to cause more "
      "forgetting, but this was not tested.")
    w(f"- **Terminology.** Training on the dictionary did not raise the overall TSR (E02 − E01: {ts_['mk_sq']['e02_minus_e01']:+} "
      f"MK→SQ, {ts_['sq_mk']['e02_minus_e01']:+} SQ→MK): Gazette fine-tuning alone already brings the model to about the "
      f"reference rate (E02 − reference: {ts_['mk_sq']['e02_minus_reference']:+} and {ts_['sq_mk']['e02_minus_reference']:+}). "
      f"Rare MK→SQ terms are the only place a lexicon effect shows ({ts_['mk_sq']['rare_e01_successes']} vs "
      f"{ts_['mk_sq']['rare_e02_successes']} successes), on too few occurrences to rely on.")
    w(f"- **Numbers and quantisation** behave as in E01 (number match with numbers: E02 − E01 = "
      f"{R['number_summary']['mk_sq']['e02_minus_e01_with_numbers']:+} and {R['number_summary']['sq_mk']['e02_minus_e01_with_numbers']:+} "
      f"points; int8 within {qs['max_abs_bleu_delta']} BLEU).\n")
    w("## 7. Limitations\n")
    w(f"- One seed per system ({t['seed']}); the E01–E02 differences on FLORES+/NTREX-128 are significant under the paired bootstrap, "
      "which accounts for test-set sampling but not for training variance.")
    w("- E01 vs E02 changes both the data and the number of optimizer steps (§2).")
    w("- E02 was trained across two invocations at two commits (resumed from checkpoint 1000); the code difference concerns only "
      "dev-curve logging, and the resume restored optimizer, scheduler and RNG state from the checkpoint.")
    w("- TSR on rare terms rests on few occurrences and was not significance-tested; TSR is a matching heuristic, not a correctness measure.")
    w(f"- {osu['pct_template_or_near_duplicate']}% of Gazette test pairs are templates or near duplicates of training sentences.\n")
    w("## 8. Reproducibility\n")
    w(f"- Results repo (private): `{R['meta']['results_repo']}` at `{R['meta']['results_repo_revision']}`; E01 translations from "
      f"`{R['meta']['e01_results_repo']}` at `{R['meta']['e01_results_repo_revision']}`.")
    w(f"- Training: commit `{R['code']['training_commit_final_invocation']}` (steps 1–1000 at `bd3c440`); analysis: "
      f"`{R['code']['analysis_commit']}`; report: `{R['meta']['report_git_head']}`.")
    w("\n```bash\n# GPU machine\npython3 scripts/train/smoke_test_nllb_run.py E02_gazette_only --quick\n"
      "python3 scripts/train/train_nllb_gazette_verbis_ct2.py E02_gazette_only\n"
      "# local, after the run\nuv run python scripts/registry/import_hf_artifacts.py E02_gazette_only\n"
      f"uv run python experiments/{EXPERIMENT_ID}/build_report.py\n```\n")
    return "\n".join(L) + "\n"


# ============================================================================
# 8. Self-check
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


def verbis_strings(paths: dict) -> set[str]:
    man = X.load_manifest("verbis")
    out: set[str] = set()
    sources = [hf_hub_download(man["storage_repo"], f["path"], revision=man["storage_revision"]) for f in man["files"]]
    sources += [paths["data/verbis/verbis_terms.parquet"], paths["data/verbis/verbis_defs.parquet"]]
    for p in sources:
        df = pl.read_parquet(p)
        for c in df.columns:
            if df[c].dtype == pl.Utf8:
                out |= {s.strip().lower() for s in df[c].drop_nulls().to_list() if len(s.strip()) >= 10}
    return out


def self_check(R: dict, paths: dict) -> None:
    allowed: set = set()
    _numbers_in(R, allowed)
    allowed.update(a.lstrip("-+") for a in list(allowed))
    body = []
    for line in (HERE / "REPORT.md").read_text(encoding="utf-8").splitlines():
        if line.startswith(("#", "![", "|---")):
            continue
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"§\d+(\.\d+)?", "", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?", "", line)
        line = re.sub(r"\b(steps? )?1[–-]1000\b|1001[–-]", "", line)  # resume ranges, stated from the log
        body.append(line)
    bad = sorted({m.group(0) for line in body for m in _NUM.finditer(line) if m.group(0).replace(",", "") not in allowed})
    if bad:
        sys.exit(f"SELF-CHECK FAILED: numbers in REPORT.md not found in results.json: {bad}")
    outputs = [HERE / "REPORT.md", HERE / "results.json", HERE / "MANIFEST.json", *TAB.glob("*.tex")]
    blob = "\n".join(p.read_text(encoding="utf-8").lower() for p in outputs)
    strings = verbis_strings(paths)
    leaked = [s for s in strings if s in blob]
    if leaked:
        sys.exit(f"SELF-CHECK FAILED: {len(leaked)} Verbis strings appear in the outputs")
    print(f"self-check passed: {len({m.group(0) for line in body for m in _NUM.finditer(line)})} distinct numbers traced to "
          f"results.json; {len(strings):,} Verbis strings checked, none present in {len(outputs)} output files")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    row, row1 = X.registry_row(EXPERIMENT_ID), X.registry_row(REFERENCE_ID)
    repo, rev = X.load_config(EXPERIMENT_ID)["outputs"]["hf_results_repo"], row["hf_revision"]
    repo1, rev1 = row1["hf_model_repo"], row1["hf_revision"]
    print("fetching sources ...", flush=True)
    paths, manifest = fetch(api, repo, rev, HF_FILES)
    e01_paths, manifest1 = fetch(api, repo1, rev1, E01_HF_FILES)
    local = [E01_DIR / p for p in ("run_info.json", "train_log.csv", "results.json", "results/gazette_test_results.csv",
                                   "results/flores_results.csv", "results/ntrex_results.csv", "analysis/gazette_test_overlap_results.csv",
                                   "analysis/terminology_results.csv", "analysis/number_fidelity_results.csv")]
    local += [X.config_path(EXPERIMENT_ID), X.config_path(REFERENCE_ID), Path(__file__).resolve()]
    meta = {"results_repo": repo, "results_repo_revision": rev, "e01_results_repo": repo1, "e01_results_repo_revision": rev1,
            "report_git_head": X.git_head(), "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    R = assemble(paths, meta)
    e01R = json.loads((E01_DIR / "results.json").read_text())
    R["e01_reference_summaries"] = {"forgetting_summary": e01R["forgetting_summary"]}
    print("paired bootstrap ...", flush=True)
    R["significance"] = significance(paths, e01_paths)
    for key, blk in R["significance"]["base_vs_finetuned_hf"].items():  # consistency with the recorded CSVs
        ts, dd = key.split("/")
        rec = pick(R["results_main"][ts], model="finetuned_hf", direction=dd)
        assert abs(blk["bleu"]["system"]["score"] - rec["bleu"]) < 0.011 and abs(blk["chrf"]["system"]["score"] - rec["chrf"]) < 0.011, key
        rec1 = pick(R["results_e01"][ts], model="finetuned_hf", direction=dd)
        e = R["significance"]["e01_finetuned_hf_vs_e02_finetuned_hf"][key]
        assert abs(e["bleu"]["baseline"]["score"] - rec1["bleu"]) < 0.011 and abs(e["chrf"]["baseline"]["score"] - rec1["chrf"]) < 0.011, key
    derive(R)
    (HERE / "MANIFEST.json").write_text(json.dumps({"generated_at": meta["generated_at"],
                                                    "artifacts": manifest + manifest1 + local_manifest(local)},
                                                   indent=2, ensure_ascii=False) + "\n")
    figures(R, paths)
    T = tables(R)
    (HERE / "results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str) + "\n")
    (HERE / "REPORT.md").write_text(report_text(R, T), encoding="utf-8")
    self_check(R, paths)
    print(f"done: {rel(HERE / 'REPORT.md')}")


if __name__ == "__main__":
    main()
