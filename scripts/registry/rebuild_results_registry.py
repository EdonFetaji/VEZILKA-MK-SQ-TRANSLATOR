#!/usr/bin/env python
"""Rebuild paper/results_registry.csv -- the single table every number in the paper comes from --
and regenerate paper/tables/*.tex and paper/figures/*.png from it.

    python scripts/registry/rebuild_results_registry.py

Collects, for every experiment with status "done" in experiments/registry.csv:
  experiments/<ID>/results/*.csv    BLEU / chrF++ (and deltas) per test set x direction x system
  experiments/<ID>/analysis/*.csv   overlap classes, Term Success Rate, number fidelity
  experiments/<ID>/results.json     paired-bootstrap means, 95% CIs and p-values (when present)
Every registry row names exactly one source file and its sha256; check_traceability.py
re-derives every row from that file. Tables and figures read the registry only.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mksq import experiment as X  # noqa: E402

PAPER = X.ROOT / "paper"
REGISTRY_OUT = PAPER / "results_registry.csv"
COLUMNS = ["exp_id", "paper_arm", "test_set", "subset", "direction", "model_variant", "metric", "value",
           "ci_low", "ci_high", "n", "source_file", "source_sha256"]
RESULT_FILES = {"gazette_test_results.csv": "gazette_test", "flores_results.csv": "flores_devtest", "ntrex_results.csv": "ntrex"}
SIG_TESTSETS = {"gazette_test": "gazette_test", "flores": "flores_devtest", "ntrex": "ntrex"}
BASE_OWNER = "E00_base"  # base-model rows belong to E00, wherever they were computed


def _num(v):
    if v in (None, ""):
        return ""
    f = float(v)
    return int(f) if f.is_integer() and "." not in str(v) else f


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_from_file(exp: dict, path: Path) -> list[dict]:
    """All registry rows derived from one source file (the checker calls this too)."""
    rel = str(path.relative_to(X.ROOT))
    sha = X.sha256_file(path)
    out = []

    def add(test_set, subset, direction, variant, metric, value, n="", ci=("", ""), owner=None):
        o = X.registry_row(owner) if owner else exp
        out.append({"exp_id": o["exp_id"], "paper_arm": o["paper_arm"], "test_set": test_set, "subset": subset,
                    "direction": direction, "model_variant": variant, "metric": metric, "value": _num(value),
                    "ci_low": _num(ci[0]), "ci_high": _num(ci[1]), "n": _num(n), "source_file": rel, "source_sha256": sha})

    is_base_exp = exp["exp_id"] == BASE_OWNER
    keep = (lambda m: m == "base") if is_base_exp else (lambda m: m != "base")
    name = path.name
    if path.parent.name == "results" and name in RESULT_FILES:
        for r in _read(path):
            if not keep(r["model"]):
                continue
            for metric in ("bleu", "chrf", "bleu_delta_vs_base", "chrf_delta_vs_base"):
                if metric in r:
                    add(RESULT_FILES[name], "full", r["direction"], r["model"], metric, r[metric], r["n_sentences"])
    elif name == "gazette_test_overlap_results.csv":
        shares = set()
        for r in _read(path):
            if (r["class"], r["direction"]) not in shares and not is_base_exp:
                shares.add((r["class"], r["direction"]))
                add("gazette_test", r["class"], r["direction"], "-", "share_of_pairs_pct", r["pct_pairs"], r["n_pairs"])
            if keep(r["model"]):
                for metric in ("bleu", "chrf"):
                    add("gazette_test", r["class"], r["direction"], r["model"], metric, r[metric], r["n_pairs"])
    elif name == "terminology_results.csv":
        for r in _read(path):
            if keep(r["model"]) or (r["model"] == "reference" and not is_base_exp):
                add("gazette_test", "all_terms", r["direction"], r["model"], "tsr", r["tsr"], r["n_term_occurrences"])
                add("gazette_test", "rare_terms", r["direction"], r["model"], "tsr", r["rare_tsr"], r["n_rare_occurrences"])
    elif name == "number_fidelity_results.csv":
        for r in _read(path):
            if keep(r["model"]) or (r["model"] == "reference" and not is_base_exp):
                add("gazette_test", "all_sentences", r["direction"], r["model"], "number_match_pct", r["match_rate_all"],
                    r["n_sentences"])
                add("gazette_test", "with_numbers", r["direction"], r["model"], "number_match_pct",
                    r["match_rate_with_numbers"], r["n_with_numbers"])
    elif name == "results.json":
        R = json.loads(path.read_text(encoding="utf-8"))
        n_by = {(ts, r["direction"]): r["n_sentences"] for ts, rows in R.get("results_main", {}).items() for r in rows}
        for group, (bl_name, sys_name) in (("base_vs_finetuned_hf", ("base", "finetuned_hf")),
                                           ("finetuned_hf_vs_finetuned_ct2", ("finetuned_hf", "finetuned_ct2"))):
            for key, blk in R.get("significance", {}).get(group, {}).items():
                ts, d = key.split("/")
                n = n_by.get((ts, d), "")
                for metric in ("bleu", "chrf"):
                    b, s = blk[metric]["baseline"], blk[metric]["system"]
                    if group == "base_vs_finetuned_hf":  # the baseline block of this group is E00's
                        add(SIG_TESTSETS[ts], "full", d, bl_name, f"{metric}_bootstrap_mean", b["bootstrap_mean"], n,
                            b["ci95"], owner=BASE_OWNER)
                    if not (group == "finetuned_hf_vs_finetuned_ct2" and sys_name == "finetuned_hf"):
                        add(SIG_TESTSETS[ts], "full", d, sys_name, f"{metric}_bootstrap_mean", s["bootstrap_mean"], n, s["ci95"])
                    add(SIG_TESTSETS[ts], "full", d, sys_name, f"{metric}_p_vs_{bl_name}", blk[metric]["p_value"], n)
    return out


def source_files(exp_id: str) -> list[Path]:
    d = X.ROOT / "experiments" / exp_id
    files = sorted((d / "results").glob("*.csv")) + sorted((d / "analysis").glob("*.csv"))
    if (d / "results.json").exists():
        files.append(d / "results.json")
    return files


def build_rows() -> list[dict]:
    rows = []
    for exp in X.registry_rows():
        if exp["status"] != "done":
            continue
        for f in source_files(exp["exp_id"]):
            rows.extend(rows_from_file(exp, f))
    rows.sort(key=lambda r: (r["exp_id"], r["test_set"], r["subset"], r["direction"], r["model_variant"], r["metric"]))
    return rows


def write_registry(rows: list[dict]) -> None:
    PAPER.mkdir(exist_ok=True)
    with open(REGISTRY_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {REGISTRY_OUT.relative_to(X.ROOT)} ({len(rows)} rows)")


# ============================================================================
# Tables and figures: generated from paper/results_registry.csv only
# ============================================================================
DIR = {"mk_sq": "MK→SQ", "sq_mk": "SQ→MK"}
TS = {"gazette_test": "Gazette test", "flores_devtest": "FLORES+ devtest", "ntrex": "NTREX-128"}
VAR = {"base": "Base (E00)", "finetuned_hf": "FT HF", "finetuned_ct2": "FT CT2 int8", "reference": "Human reference"}
C = {"base": "#2a78d6", "finetuned_hf": "#eb6834", "finetuned_ct2": "#1baf7a", "reference": "#52514e"}


def _lookup(reg: list[dict]):
    idx = {(r["exp_id"], r["test_set"], r["subset"], r["direction"], r["model_variant"], r["metric"]): r for r in reg}

    def get(exp, ts, subset, d, var, metric, field="value"):
        r = idx.get((exp, ts, subset, d, var, metric))
        return None if r is None or r[field] == "" else float(r[field])
    return get


def _tex(name: str, caption: str, header: list[str], rows: list[list[str]]) -> None:
    esc = lambda s: (str(s).replace("_", r"\_").replace("%", r"\%").replace("→", r"$\rightarrow$")  # noqa: E731
                     .replace("≤", r"$\leq$").replace("Δ", r"$\Delta$"))
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}",
             r"\toprule", " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{esc(caption)} Generated from \texttt{{paper/results\_registry.csv}}.}}",
              rf"\label{{tab:{name.replace('_', '-')}}}", r"\end{table}", ""]
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def fmt(x, nd=2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def exps_with(reg, variant) -> list[str]:
    return sorted({r["exp_id"] for r in reg if r["model_variant"] == variant and r["metric"] == "chrf"})


def tables_and_figures(reg: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    get = _lookup(reg)
    trained = [e for e in exps_with(reg, "finetuned_hf")]
    systems = [("E00_base", "base")] + [(e, v) for e in trained for v in ("finetuned_hf", "finetuned_ct2")]
    # main results
    rows = []
    for ts in TS:
        for d in DIR:
            for exp, var in systems:
                b, c = get(exp, ts, "full", d, var, "bleu"), get(exp, ts, "full", d, var, "chrf")
                if b is None:
                    continue
                lo, hi = (get(exp, ts, "full", d, var, "chrf_bootstrap_mean", f) for f in ("ci_low", "ci_high"))
                p = get(exp, ts, "full", d, var, "chrf_p_vs_base")
                rows.append([TS[ts], DIR[d], f"{exp} {VAR[var]}", fmt(b), fmt(c),
                             f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None else "–", fmt(p, 3) if p is not None else "–"])
    _tex("main_results", "BLEU and chrF++ (beam 4) with the chrF++ paired-bootstrap 95% CI and p-value against the base model.",
         ["Test set", "Dir.", "System", "BLEU", "chrF++", "chrF++ 95% CI", "p vs base"], rows)
    # overlap
    rows = []
    for cls in ("exact_template", "near_duplicate", "novel", "all"):
        for d in DIR:
            share = next((get(e, "gazette_test", cls, d, "-", "share_of_pairs_pct") for e in trained), None)
            rows.append([cls.replace("_", " "), DIR[d], fmt(share)] +
                        [fmt(get(exp, "gazette_test", cls, d, var, "chrf")) for exp, var in systems])
    _tex("overlap", "Gazette test chrF++ by overlap with the training sentences.",
         ["Class", "Dir.", "% pairs"] + [f"{e} {VAR[v]}" for e, v in systems], rows)
    # terminology
    rows = []
    for subset in ("all_terms", "rare_terms"):
        for d in DIR:
            ref = next((get(e, "gazette_test", subset, d, "reference", "tsr") for e in trained), None)
            rows.append([subset.replace("_", " "), DIR[d], fmt(ref)] +
                        [fmt(get(exp, "gazette_test", subset, d, var, "tsr")) for exp, var in systems])
    _tex("terminology", "Term Success Rate (%) on the Gazette test; the human reference is the realistic ceiling.",
         ["Terms", "Dir.", "Reference"] + [f"{e} {VAR[v]}" for e, v in systems], rows)
    # number fidelity
    rows = []
    for subset in ("all_sentences", "with_numbers"):
        for d in DIR:
            ref = next((get(e, "gazette_test", subset, d, "reference", "number_match_pct") for e in trained), None)
            rows.append([subset.replace("_", " "), DIR[d], fmt(ref)] +
                        [fmt(get(exp, "gazette_test", subset, d, var, "number_match_pct")) for exp, var in systems])
    _tex("number_fidelity", "Number fidelity (%): sentences whose digit sequences match the source.",
         ["Sentences", "Dir.", "Reference"] + [f"{e} {VAR[v]}" for e, v in systems], rows)
    print(f"wrote paper/tables/ ({len(list((PAPER / 'tables').glob('*.tex')))} tables)")

    # figure: main chrF++ with CIs
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "savefig.facecolor": "#fcfcfb"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    width = 0.8 / len(systems)
    for ax, ts in zip(axes, TS, strict=True):
        for k, (exp, var) in enumerate(systems):
            xs, vals, errs = [], [], [[], []]
            for j, d in enumerate(DIR):
                v = get(exp, ts, "full", d, var, "chrf")
                lo, hi = (get(exp, ts, "full", d, var, "chrf_bootstrap_mean", f) for f in ("ci_low", "ci_high"))
                mean = get(exp, ts, "full", d, var, "chrf_bootstrap_mean")
                xs.append(j + (k - (len(systems) - 1) / 2) * width)
                vals.append(v or 0)
                errs[0].append(max(0.0, (mean - lo)) if lo is not None else 0)
                errs[1].append(max(0.0, (hi - mean)) if hi is not None else 0)
            ax.bar(xs, vals, width=width * 0.92, color=C[var], hatch="//" if var == "finetuned_ct2" else None,
                   edgecolor="#fcfcfb", label=f"{exp.split('_')[0]} {VAR[var]}", yerr=errs, capsize=2, ecolor="#52514e")
        ax.set_xticks([0, 1], list(DIR.values()))
        ax.set_title(TS[ts], loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("chrF++ (95% bootstrap CI)")
    h, labels = axes[0].get_legend_handles_labels()
    fig.legend(h, labels, frameon=False, loc="lower center", ncol=len(systems), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PAPER / "figures" / "main_results.png", dpi=200)
    plt.close(fig)
    # figure: TSR
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, subset in zip(axes, ("all_terms", "rare_terms"), strict=True):
        for j, d in enumerate(DIR):
            ref = next((get(e, "gazette_test", subset, d, "reference", "tsr") for e in trained), None)
            if ref is not None:
                ax.hlines(ref, j - 0.42, j + 0.42, color=C["reference"], ls="--", lw=2,
                          label=VAR["reference"] if j == 0 else None)
            for k, (exp, var) in enumerate(systems):
                v = get(exp, "gazette_test", subset, d, var, "tsr")
                ax.bar(j + (k - (len(systems) - 1) / 2) * width, v or 0, width=width * 0.92, color=C[var],
                       hatch="//" if var == "finetuned_ct2" else None, edgecolor="#fcfcfb",
                       label=f"{exp.split('_')[0]} {VAR[var]}" if j == 0 else None)
        ax.set_xticks([0, 1], list(DIR.values()))
        ax.set_title(f"Term Success Rate, {subset.replace('_', ' ')}", loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#e4e3df")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("TSR (%)")
    h, labels = axes[0].get_legend_handles_labels()
    fig.legend(h, labels, frameon=False, loc="lower center", ncol=len(systems) + 1, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(PAPER / "figures" / "terminology.png", dpi=200)
    plt.close(fig)
    print("wrote paper/figures/ (main_results.png, terminology.png)")


def read_registry() -> list[dict]:
    with open(REGISTRY_OUT, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    write_registry(build_rows())
    tables_and_figures(read_registry())  # tables and figures read the committed registry file, nothing else


if __name__ == "__main__":
    main()
