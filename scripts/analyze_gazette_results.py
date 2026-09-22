#!/usr/bin/env python
"""Error analysis of the Gazette test results -- no retraining, no GPU.

    cd ~/VEZILKA-MK-SQ-TRANSLATOR
    python3 scripts/analyze_gazette_results.py

Reads the cached translations (work/eval/translations_gazette_test.parquet,
written by step 12 of train_nllb_gazette_verbis_ct2.py) and the processed data
in work/data/processed/, and writes:

  1. work/eval/gazette_test_overlap_results.csv
     Overlap audit, Gazette test vs the Gazette training sentences the model
     actually saw (train_clean.parquet, source == "gazette"). Sentences are
     normalised (lowercase, every digit -> 0, whitespace collapsed); each test
     pair is classed by its closest train sentence on either side:
       exact_template  identical after normalisation (differs only in numbers)
       near_duplicate  character 5-gram Jaccard >= 0.8 (MinHash/LSH candidates,
                       verified with the exact Jaccard of the shingle sets)
       novel           everything else
     BLEU and chrF++ per class x model x direction. Issue-level overlap
     (issue_key in test and train) is printed.
  2. work/eval/terminology_results.csv
     Term Success Rate with the Verbis terms (source term >= 5 characters,
     matched on word boundaries); a hypothesis succeeds if every target-term
     token is matched by some hypothesis token by prefix,
     hyp.startswith(term[:max(4, len(term) - 2)]). Micro-TSR, plus TSR on rare
     terms (<= 5 training sentences contain them).
  3. work/eval/number_fidelity_results.csv
     Share of sentences whose multiset of digit sequences in the hypothesis
     equals the source's -- over all sentences, and over sentences with numbers.

"reference" rows score the human translation the same way: the ceiling the
metric allows on this data. The three CSVs are uploaded to eval/ of the
PRIVATE results repo after model_info().private is confirmed; if the repo is
not private nothing is uploaded and the script exits non-zero.
"""

from __future__ import annotations

import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sacrebleu.metrics import BLEU, CHRF

HF_RESULTS_REPO = "EdonFetaji/mk-sq-nllb600m-gazette-verbis"  # PRIVATE
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
PROC = WORK / "data" / "processed"
EVAL = WORK / "eval"
TRANSLATIONS = EVAL / "translations_gazette_test.parquet"
OUT_OVERLAP = EVAL / "gazette_test_overlap_results.csv"
OUT_TERMS = EVAL / "terminology_results.csv"
OUT_NUMBERS = EVAL / "number_fidelity_results.csv"
OUT_PAIRS = EVAL / "gazette_test_overlap_pairs.parquet"  # per-pair classes, local only

MODELS = ("base", "finetuned_hf", "finetuned_ct2")
DIRECTIONS = (("mk_sq", "mk", "sq"), ("sq_mk", "sq", "mk"))  # (name, source side, target side)

SHINGLE = 5
NEAR_DUP = 0.8
NUM_PERM, BANDS = 128, 16  # 16 bands x 8 rows: candidate threshold ~ (1/16)^(1/8) = 0.71 < 0.8
MAX_CANDIDATES = 200  # verified per query, most band collisions first

MIN_TERM_CHARS = 5
RARE_MAX_TRAIN = 5

_DIGIT = re.compile(r"\d")
_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"\w+")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def table(df: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=40,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        return str(df)


def write_csv(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.write_csv(tmp)
    os.replace(tmp, path)
    say(f"wrote {path.relative_to(ROOT)}")


def scores(hyps: list[str], refs: list[str]) -> tuple[float | None, float | None]:
    if not hyps:
        return None, None
    return (round(BLEU().corpus_score(hyps, [refs]).score, 2),
            round(CHRF(word_order=2).corpus_score(hyps, [refs]).score, 2))


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def load_inputs():
    for p in (TRANSLATIONS, PROC / "gazette_test.parquet", PROC / "gazette_train.parquet",
              PROC / "train_clean.parquet", PROC / "verbis_terms.parquet"):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found -- run train_nllb_gazette_verbis_ct2.py through step 12 first")
    trans = pl.read_parquet(TRANSLATIONS)
    test = pl.read_parquet(PROC / "gazette_test.parquet")
    models = [m for m in MODELS if m in trans.columns and trans[m].null_count() == 0]
    for m in MODELS:
        if m not in models:
            say(f"WARNING: {m} translations missing or incomplete in {TRANSLATIONS.name} -- skipped")
    if not models:
        sys.exit("ERROR: no complete model column in the translations file")
    # step 12 writes the test pairs in order, once per direction: row i of each block is test pair i
    blocks = {}
    for d, s_col, t_col in DIRECTIONS:
        blk = trans.filter(pl.col("direction") == d)
        if (blk.height != test.height or blk["source"].to_list() != test[s_col].to_list()
                or blk["reference"].to_list() != test[t_col].to_list()):
            sys.exit(f"ERROR: {TRANSLATIONS.name} ({d}) does not line up with gazette_test.parquet")
        blocks[d] = blk
    say(f"test pairs: {test.height} | models: {', '.join(models)}")
    return test, blocks, models


# ----------------------------------------------------------------------------
# 1. Overlap audit
# ----------------------------------------------------------------------------
def normalise(text: str) -> str:
    return " ".join(_DIGIT.sub("0", text.lower()).split())


def shingle_hashes(text: str) -> np.ndarray:
    grams = {text[i:i + SHINGLE] for i in range(max(1, len(text) - SHINGLE + 1))}
    return np.unique(np.fromiter((zlib.crc32(g.encode()) for g in grams), dtype=np.uint64, count=len(grams)))


class MinHasher:
    """MinHash over 32-bit shingle hashes: (a*h + b) mod p, p prime > 2^32, a < 2^31 so nothing overflows."""

    P = np.uint64(4294967311)

    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.a = rng.integers(1, 2 ** 31, NUM_PERM, dtype=np.uint64)[:, None]
        self.b = rng.integers(0, 2 ** 32, NUM_PERM, dtype=np.uint64)[:, None]
        self.rows = NUM_PERM // BANDS

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        return ((self.a * hashes[None, :] + self.b) % self.P).min(axis=1)

    def band_keys(self, sig: np.ndarray) -> list[tuple[int, bytes]]:
        return [(i, sig[i * self.rows:(i + 1) * self.rows].tobytes()) for i in range(BANDS)]


def nearest_train(test_texts: list[str], train_texts: list[str]) -> list[float]:
    """Best (verified) Jaccard of each test text against the train texts, via LSH candidates."""
    mh = MinHasher()
    train_sh = [shingle_hashes(t) for t in train_texts]
    buckets: dict = defaultdict(list)
    for idx, sh in enumerate(train_sh):
        for key in mh.band_keys(mh.signature(sh)):
            buckets[key].append(idx)
    best = []
    for text in test_texts:
        sh = shingle_hashes(text)
        hits = Counter()
        for key in mh.band_keys(mh.signature(sh)):
            hits.update(buckets.get(key, ()))
        top = 0.0
        for idx, _ in hits.most_common(MAX_CANDIDATES):
            other = train_sh[idx]
            inter = np.intersect1d(sh, other, assume_unique=True).size
            top = max(top, inter / (sh.size + other.size - inter))
            if top >= 1.0:
                break
        best.append(top)
    return best


def overlap_audit(test: pl.DataFrame, blocks: dict, models: list[str]) -> pl.DataFrame:
    say("\n=== 1. OVERLAP AUDIT: Gazette test vs Gazette train (sentences the model trained on) ===")
    train = pl.read_parquet(PROC / "train_clean.parquet").filter(pl.col("source") == "gazette")
    classes = ["novel"] * test.height
    best_j = np.zeros(test.height)
    for side in ("mk", "sq"):
        train_norm = sorted({normalise(t) for t in train[side].to_list()})
        test_norm = [normalise(t) for t in test[side].to_list()]
        exact = set(train_norm)
        say(f"{side}: {train.height} train sentences -> {len(train_norm)} unique after normalisation; "
            f"indexing 5-gram MinHash ({NUM_PERM} perms, {BANDS} bands)")
        todo = [i for i, t in enumerate(test_norm) if t not in exact]
        for i, t in enumerate(test_norm):
            if t in exact:
                classes[i] = "exact_template"
                best_j[i] = 1.0
        for i, j in zip(todo, nearest_train([test_norm[i] for i in todo], train_norm), strict=True):
            best_j[i] = max(best_j[i], j)
            if j >= NEAR_DUP and classes[i] == "novel":
                classes[i] = "near_duplicate"
    pairs = test.select([c for c in ("pair_id", "issue_key", "mk", "sq") if c in test.columns]).with_columns(
        overlap_class=pl.Series(classes), best_train_jaccard=pl.Series(best_j.round(4)))
    tmp = OUT_PAIRS.with_name(OUT_PAIRS.name + ".tmp")
    pairs.write_parquet(tmp, compression="zstd")
    os.replace(tmp, OUT_PAIRS)

    rows = []
    order = ["exact_template", "near_duplicate", "novel", "all"]
    for cls in order:
        idx = list(range(test.height)) if cls == "all" else [i for i, c in enumerate(classes) if c == cls]
        for model in (*models, ):
            for d, _, _ in DIRECTIONS:
                blk = blocks[d]
                hyp, ref = blk[model].to_list(), blk["reference"].to_list()
                b, c = scores([hyp[i] for i in idx], [ref[i] for i in idx])
                rows.append({"class": cls, "n_pairs": len(idx), "pct_pairs": round(100 * len(idx) / test.height, 2),
                             "model": model, "direction": d, "bleu": b, "chrf": c})
    out = pl.DataFrame(rows)
    counts = pl.DataFrame({"class": order[:3], "n_pairs": [classes.count(c) for c in order[:3]]}).with_columns(
        pct=(100 * pl.col("n_pairs") / test.height).round(2))
    say(table(counts))
    say(table(out))

    if "issue_key" in test.columns:
        tr_issues = set(pl.read_parquet(PROC / "gazette_train.parquet", columns=["issue_key"])["issue_key"].to_list())
        te_issues = set(test["issue_key"].drop_nulls().to_list())
        shared = te_issues & tr_issues
        say(f"issue overlap (issue_key): {len(te_issues)} test issues, {len(shared)} also in train"
            + (f" -- e.g. {sorted(shared)[:5]}" if shared else " -- the split is issue-disjoint"))
    else:
        say("issue overlap: gazette_test.parquet has no issue/document column")
    write_csv(out, OUT_OVERLAP)
    return out


# ----------------------------------------------------------------------------
# 2. Terminology accuracy
# ----------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class TermMatcher:
    """Finds which source terms occur in a sentence (whole-word token sequences)."""

    def __init__(self, terms: list[str]):
        self.index: dict = defaultdict(list)
        for tid, term in enumerate(terms):
            toks = tuple(words(term))
            if toks:
                self.index[toks[0]].append((toks, tid))

    def find(self, text: str) -> set[int]:
        toks = words(text)
        found = set()
        for i, t in enumerate(toks):
            for seq, tid in self.index.get(t, ()):
                if tuple(toks[i:i + len(seq)]) == seq:
                    found.add(tid)
        return found


def term_hit(hyp_tokens: list[str], target: str) -> bool:
    need = words(target)
    return bool(need) and all(any(h.startswith(t[:max(4, len(t) - 2)]) for h in hyp_tokens) for t in need)


def terminology(blocks: dict, models: list[str]) -> pl.DataFrame:
    say("\n=== 2. TERMINOLOGY ACCURACY (Term Success Rate, Verbis terms) ===")
    terms = pl.read_parquet(PROC / "verbis_terms.parquet").select(pl.col("mk").str.to_lowercase(),
                                                                  pl.col("sq").str.to_lowercase())
    train = pl.read_parquet(PROC / "train_clean.parquet").filter(pl.col("source") == "gazette")
    rows = []
    for d, s_col, t_col in DIRECTIONS:
        grouped = (terms.filter(pl.col(s_col).str.len_chars() >= MIN_TERM_CHARS)
                   .group_by(s_col, maintain_order=True).agg(pl.col(t_col).unique()))
        src_terms, targets = grouped[s_col].to_list(), grouped[t_col].to_list()
        matcher = TermMatcher(src_terms)
        train_freq = Counter()
        for sent in train[s_col].to_list():
            train_freq.update(matcher.find(sent))
        blk = blocks[d]
        found = [matcher.find(s) for s in blk["source"].to_list()]
        n_occ = sum(len(f) for f in found)
        n_sent = sum(1 for f in found if f)
        say(f"{d}: {len(src_terms)} source terms (>= {MIN_TERM_CHARS} chars); {n_occ} occurrences in "
            f"{n_sent} of {blk.height} test sentences")
        for model in ("reference", *models):
            hyps = blk[model].to_list()
            ok = ok_rare = n_rare = 0
            for hyp, tids in zip(hyps, found, strict=True):
                if not tids:
                    continue
                htoks = words(hyp)
                for tid in tids:
                    hit = any(term_hit(htoks, tgt) for tgt in targets[tid])  # any listed translation counts
                    ok += hit
                    if train_freq[tid] <= RARE_MAX_TRAIN:
                        n_rare += 1
                        ok_rare += hit
            rows.append({"model": model, "direction": d, "n_sentences_with_terms": n_sent,
                         "n_term_occurrences": n_occ, "n_success": ok,
                         "tsr": round(100 * ok / n_occ, 2) if n_occ else None,
                         "n_rare_occurrences": n_rare, "n_rare_success": ok_rare,
                         "rare_tsr": round(100 * ok_rare / n_rare, 2) if n_rare else None})
    out = pl.DataFrame(rows)
    say(table(out))
    write_csv(out, OUT_TERMS)
    return out


# ----------------------------------------------------------------------------
# 3. Number fidelity
# ----------------------------------------------------------------------------
def number_fidelity(blocks: dict, models: list[str]) -> pl.DataFrame:
    say("\n=== 3. NUMBER FIDELITY (multiset of digit sequences, source vs hypothesis) ===")
    rows = []
    for d, _, _ in DIRECTIONS:
        blk = blocks[d]
        src_nums = [Counter(_NUMBER.findall(s)) for s in blk["source"].to_list()]
        with_nums = [i for i, c in enumerate(src_nums) if c]
        for model in ("reference", *models):
            match = [Counter(_NUMBER.findall(h)) == c for h, c in zip(blk[model].to_list(), src_nums, strict=True)]
            rows.append({"model": model, "direction": d, "n_sentences": len(match),
                         "match_rate_all": round(100 * sum(match) / len(match), 2),
                         "n_with_numbers": len(with_nums),
                         "match_rate_with_numbers": round(100 * sum(match[i] for i in with_nums) / len(with_nums), 2)
                         if with_nums else None})
    out = pl.DataFrame(rows)
    say(table(out))
    write_csv(out, OUT_NUMBERS)
    return out


# ----------------------------------------------------------------------------
# Upload (private repo only)
# ----------------------------------------------------------------------------
def upload(paths: list[Path]) -> bool:
    from huggingface_hub import CommitOperationAdd, HfApi, get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        say("ERROR: no Hugging Face token (set HF_TOKEN or run `hf auth login`) -- nothing uploaded")
        return False
    api = HfApi(token=token)
    try:
        private = api.model_info(HF_RESULTS_REPO).private
    except Exception as e:  # noqa: BLE001
        say(f"ERROR: could not verify that {HF_RESULTS_REPO} is private ({e}) -- nothing uploaded")
        return False
    if private is not True:
        say(f"ERROR: PRIVACY GUARD: {HF_RESULTS_REPO} is not private (private={private!r}) -- nothing uploaded")
        sys.exit(2)
    try:
        api.create_commit(HF_RESULTS_REPO, repo_type="model", commit_message="Gazette test error analysis",
                          operations=[CommitOperationAdd(path_in_repo=f"eval/{p.name}", path_or_fileobj=str(p))
                                      for p in paths])
    except Exception as e:  # noqa: BLE001
        say(f"ERROR: upload failed ({type(e).__name__}: {e}) -- the CSVs are in {EVAL}")
        return False
    say(f"uploaded {', '.join('eval/' + p.name for p in paths)} to https://huggingface.co/{HF_RESULTS_REPO}")
    return True


def main() -> None:
    random.seed(SEED)
    say(f"Gazette test error analysis -- {datetime.now().astimezone().isoformat(timespec='seconds')}")
    test, blocks, models = load_inputs()
    overlap_audit(test, blocks, models)
    terminology(blocks, models)
    number_fidelity(blocks, models)
    say()
    sys.exit(0 if upload([OUT_OVERLAP, OUT_TERMS, OUT_NUMBERS]) else 1)


if __name__ == "__main__":
    main()
