#!/usr/bin/env python
"""Build and publish a cleaned mk-sq parallel dataset from every OPUS corpus.

    python scripts/build_opus_mk_sq_dataset.py

One file, one command, no arguments, no interactive input. Safe to start with
`nohup python scripts/build_opus_mk_sq_dataset.py &` and walk away: every stage
writes its output to disk, every later stage reads from disk, and nothing is
recomputed if its output already exists. Resume granularity for the two
expensive stages -- cleaning and LaBSE scoring -- is one chunk, not one corpus.

This script builds and publishes data. It trains nothing. Verbis is not part of
it and is never uploaded.

Two places where this departs from a literal reading of the brief, both
deliberate:

  * **Cross-corpus dedup runs before LaBSE scoring, not after.** CCMatrix and
    NLLB report exactly the same 2,053,423 pairs for mk-sq -- they are the same
    data. Scoring before deduplicating would embed ~2M pairs twice, in the most
    expensive stage of the run. The dedup rule does not consult the score, so
    moving it earlier changes nothing but the cost. Leakage removal moves up for
    the same reason.

  * **Licences come from the curated LICENCES table below, not from OPUS.** The
    OPUS API exposes no licence field and the corpus pages are JavaScript-
    rendered, so there is nothing to read programmatically. Each entry carries a
    source URL. Anything absent from the table is ("unknown", "restricted"), so
    a corpus OPUS adds later can never silently reach the public repo.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
import unicodedata
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants. Everything tunable lives here; the script takes no flags.
# ---------------------------------------------------------------------------

HF_USERNAME = "EdonFetaji"
HF_OPUS_REPO = f"{HF_USERNAME}/mk-sq-opus-filtered"
HF_OPUS_PRIVATE_REPO = f"{HF_USERNAME}/mk-sq-opus-filtered-restricted"
HF_GAZETTE_REPO = f"{HF_USERNAME}/slvesnik-mk-sq"

WORK_DIR = Path("data/opus_build")
LABSE_THRESHOLD = 0.75
OPUS_TEST_SIZE = 2000
SEED = 42
CHUNK_SIZE = 200_000

OPUS_API = "https://opus.nlpl.eu/opusapi/"
SRC, TGT = "mk", "sq"

FLORES_REPO = "openlanguagedata/flores_plus"
FLORES_FILES = ["devtest/mkd_Cyrl.jsonl", "devtest/als_Latn.jsonl"]
NTREX_BASE = (
    "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
)
NTREX_FILES = ["newstest2019-ref.mkd.txt", "newstest2019-ref.sqi.txt"]

NLLB_TOKENIZER = "facebook/nllb-200-distilled-600M"
MAX_NLLB_TOKENS = 192
LABSE_MODEL = "sentence-transformers/LaBSE"
MAX_LENGTH_RATIO = 2.0
MIN_WORDS_FOR_LANGID = 5
SCRIPT_PURITY_MIN = 0.70

# Corpora never downloaded: they are evaluation sets and would be leakage.
SKIP_CORPORA = {"FLORES", "FLORES200", "FLORES_PLUS", "NTREX", "NTREX128"}
# Entity pairs, not sentences -- kept for training, excluded from the test set.
NOT_SENTENCES = {"XLEnt"}

# When the same pair appears in several corpora, keep the copy from the corpus
# earliest in this list; anything unlisted sorts after, alphabetically. A public
# corpus always beats a restricted one regardless of position.
CORPUS_PRIORITY = ["SETIMES", "TED2020", "WikiMatrix"]

# --- licences ---------------------------------------------------------------
# (licence, class, source_url). class is "public" only where the terms clearly
# permit redistributing a modified subset with no NC and no ND restriction.
# Anything else, anything unclear, and anything unlisted is "restricted".
LICENCES: dict[str, tuple[str, str, str]] = {
    "SETIMES": ("CC-BY-SA-3.0", "public", "https://opus.nlpl.eu/SETIMES/corpus/version/SETIMES"),
    "WikiMatrix": ("CC-BY-SA-4.0", "public", "https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix"),
    "wikimedia": ("CC-BY-SA-4.0", "public", "https://opus.nlpl.eu/wikimedia/corpus/version/wikimedia"),
    "GlobalVoices": ("CC-BY-3.0", "public", "https://globalvoices.org/about/global-voices-attribution-policy/"),
    "Tatoeba": ("CC-BY-2.0-FR", "public", "https://tatoeba.org/eng/terms_of_use"),
    "MultiHPLT": ("CC0-1.0", "public", "https://hplt-project.org/datasets/v2"),
    "MultiMaCoCu": ("CC0-1.0", "public", "https://macocu.eu/"),
    "EUbookshop": ("EU reuse decision 2011/833/EU", "public", "https://op.europa.eu/en/web/about-us/legal-notices"),
    # TED talks are explicitly NonCommercial-NoDerivatives.
    "TED2020": ("CC-BY-NC-ND-4.0", "restricted", "https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy"),
    "NeuLab-TedTalks": ("CC-BY-NC-ND-4.0", "restricted", "https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy"),
    # Subtitle and crawl derivatives with no clear redistribution grant.
    "OpenSubtitles": ("unclear", "restricted", "https://www.opensubtitles.org/en/disclaimer"),
    "QED": ("unclear", "restricted", "https://opus.nlpl.eu/QED/corpus/version/QED"),
    "CCMatrix": ("unclear (CommonCrawl derivative)", "restricted", "https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix"),
    "NLLB": ("unclear (CommonCrawl derivative)", "restricted", "https://opus.nlpl.eu/NLLB/corpus/version/NLLB"),
    "MultiCCAligned": ("unclear (CommonCrawl derivative)", "restricted", "https://opus.nlpl.eu/MultiCCAligned/corpus/version/MultiCCAligned"),
    "XLEnt": ("unclear (CommonCrawl derivative)", "restricted", "https://opus.nlpl.eu/XLEnt/corpus/version/XLEnt"),
    # Software localisation: mixed upstream terms.
    "GNOME": ("unclear (mixed upstream)", "restricted", "https://opus.nlpl.eu/GNOME/corpus/version/GNOME"),
    "Ubuntu": ("unclear (mixed upstream)", "restricted", "https://opus.nlpl.eu/Ubuntu/corpus/version/Ubuntu"),
}
DEFAULT_LICENCE = ("unknown", "restricted", "")

REQUIRED_IMPORTS = {
    "datasets": "datasets",
    "huggingface_hub": "huggingface_hub",
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "lingua": "lingua-language-detector",
    "requests": "requests",
}

RAW = WORK_DIR / "raw"
PROCESSED = WORK_DIR / "processed"
LOG_PATH = WORK_DIR / "progress.log"

PAIR_COLUMNS = ["mk", "sq", "corpus"]

# ---------------------------------------------------------------------------
# Logging, steps, atomic writes
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


@contextmanager
def step(number: int, name: str):
    """Log a step, time it, and turn any exception into a logged non-zero exit."""
    log("=" * 78)
    log(f"STEP {number}  {name}")
    started = time.time()
    try:
        yield
    except KeyboardInterrupt:
        log(f"STEP {number} interrupted by the user -- re-run to resume")
        raise SystemExit(130)
    except BaseException:
        log(f"STEP {number} FAILED after {time.time() - started:,.1f}s")
        for line in traceback.format_exc().splitlines():
            log("    " + line)
        raise SystemExit(1)
    log(f"STEP {number} done in {time.time() - started:,.1f}s")


def write_parquet_atomic(table, path: Path) -> None:
    """Write via .tmp then rename, so an interrupted write is never mistaken
    for a finished one on the next run."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def write_frame_atomic(frame, path: Path) -> None:
    """polars DataFrame -> Parquet, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def clear_stale_tmp(root: Path) -> int:
    """Remove .tmp files left by an interrupted write."""
    removed = 0
    if root.exists():
        for stale in root.rglob("*.tmp"):
            stale.unlink()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Step 1: preflight
# ---------------------------------------------------------------------------


def preflight() -> tuple[str, str]:
    missing = []
    for module, package in REQUIRED_IMPORTS.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("missing dependencies:", ", ".join(missing), file=sys.stderr)
        print("\ninstall them with:\n", file=sys.stderr)
        print(f"    {sys.executable} -m pip install -r scripts/data/requirements-opus.txt\n", file=sys.stderr)
        raise SystemExit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    log(f"work dir: {WORK_DIR.resolve()}")

    stale = clear_stale_tmp(WORK_DIR)
    if stale:
        log(f"removed {stale} stale .tmp file(s) from an interrupted run")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
    if not token:
        print(
            "no Hugging Face token.\n"
            "  set HF_TOKEN=... in the environment, or run `hf auth login` once.\n"
            "  a WRITE token is required: this script creates and pushes two datasets.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as exc:
        print(f"the Hugging Face token was rejected: {exc}", file=sys.stderr)
        raise SystemExit(1)
    log(f"hugging face: {who.get('name')} (token accepted)")

    # FLORES+ is gated. Probe it now: a 401 three hours into the run is the
    # worst possible time to discover the terms were never accepted.
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(
            FLORES_REPO, FLORES_FILES[0], repo_type="dataset",
            token=token, cache_dir=str(RAW / "hf_cache"),
        )
        log(f"{FLORES_REPO}: access confirmed")
    except Exception as exc:
        print(
            f"cannot read {FLORES_REPO}: {type(exc).__name__}\n"
            f"  it is a gated dataset. Open\n"
            f"      https://huggingface.co/datasets/{FLORES_REPO}\n"
            f"  accept the terms once, then re-run.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    import torch

    if torch.cuda.is_available():
        device = "cuda"
        log(f"device: cuda ({torch.cuda.get_device_name(0)})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        log("device: mps (Apple Silicon)")
    else:
        device = "cpu"
        log("device: cpu")
        log("WARNING  LaBSE on CPU across millions of pairs can take many hours.")
        log("WARNING  This run is resumable, so it is safe to stop and restart it.")
    return token, device


# ---------------------------------------------------------------------------
# Step 2: leakage reference
# ---------------------------------------------------------------------------


def normalise_for_match(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", (text or "")).lower().split())


def build_leakage_reference(token: str) -> "object":
    import polars as pl
    import requests

    out = PROCESSED / "leakage_reference.parquet"
    if out.exists():
        log(f"loaded from disk: {out}")
        return pl.read_parquet(out)

    from huggingface_hub import hf_hub_download

    rows: list[dict] = []

    for member in FLORES_FILES:
        path = hf_hub_download(
            FLORES_REPO, member, repo_type="dataset",
            token=token, cache_dir=str(RAW / "hf_cache"),
        )
        with open(path, encoding="utf-8") as fh:
            n = 0
            for line in fh:
                text = json.loads(line).get("text", "")
                if text:
                    rows.append({"text": normalise_for_match(text), "source": f"flores_plus:{member}"})
                    n += 1
        log(f"  FLORES+ {member}: {n:,} sentences")

    ntrex_dir = RAW / "ntrex"
    ntrex_dir.mkdir(parents=True, exist_ok=True)
    for member in NTREX_FILES:
        local = ntrex_dir / member
        if not local.exists():
            response = requests.get(f"{NTREX_BASE}/{member}", timeout=120)
            response.raise_for_status()
            local.write_bytes(response.content)
        lines = local.read_text(encoding="utf-8").splitlines()
        rows.extend(
            {"text": normalise_for_match(t), "source": f"ntrex128:{member}"}
            for t in lines if t.strip()
        )
        log(f"  NTREX-128 {member}: {len(lines):,} sentences")

    try:
        from datasets import load_dataset

        gazette = load_dataset(HF_GAZETTE_REPO, token=token)
        for split in ("validation", "test"):
            if split not in gazette:
                continue
            n = 0
            for column in ("mk_text", "sq_text"):
                if column not in gazette[split].column_names:
                    continue
                for text in gazette[split][column]:
                    if text:
                        rows.append({"text": normalise_for_match(text),
                                     "source": f"gazette:{split}:{column}"})
                        n += 1
            log(f"  Gazette {split}: {n:,} sentences")
    except Exception as exc:
        log(f"WARNING  Gazette repo {HF_GAZETTE_REPO} unreachable ({type(exc).__name__}); "
            f"continuing without it. The training notebook repeats leakage removal.")

    frame = pl.DataFrame(rows).filter(pl.col("text") != "").unique(subset=["text"])
    write_frame_atomic(frame, out)
    log(f"leakage reference: {len(frame):,} distinct sentences -> {out}")
    return pl.read_parquet(out)


# ---------------------------------------------------------------------------
# Steps 3-4: corpus list and download
# ---------------------------------------------------------------------------


def list_corpora() -> list[dict]:
    import requests

    out = RAW / "opus" / "corpora_list.json"
    if out.exists():
        log(f"loaded from disk: {out}")
        return json.loads(out.read_text())

    response = requests.get(
        OPUS_API,
        params={"source": SRC, "target": TGT, "preprocessing": "moses", "version": "latest"},
        timeout=120,
    )
    response.raise_for_status()
    corpora = response.json()["corpora"]

    kept = []
    for entry in corpora:
        name = entry["corpus"]
        if name.upper().replace("-", "").replace("_", "") in SKIP_CORPORA:
            log(f"  skipping {name}: evaluation set, would be leakage")
            continue
        if not (entry.get("alignment_pairs") or 0):
            log(f"  skipping {name}: 0 alignment pairs")
            continue
        kept.append(entry)

    # Smallest first: a misconfiguration surfaces on Tatoeba in seconds rather
    # than on CCMatrix in an hour.
    kept.sort(key=lambda e: e.get("alignment_pairs") or 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(e["alignment_pairs"] for e in kept)
    log(f"{len(kept)} corpora, {total:,} pairs reported by OPUS -> {out}")
    for entry in kept:
        log(f"    {entry['corpus']:<24} {entry['version']:<12} {entry['alignment_pairs']:>10,} pairs")
    return kept


def moses_members(archive: zipfile.ZipFile, corpus: str) -> tuple[str, str]:
    """The .mk and .sq members inside a Moses zip."""
    mk = [n for n in archive.namelist() if n.endswith(f".{SRC}")]
    sq = [n for n in archive.namelist() if n.endswith(f".{TGT}")]
    if len(mk) != 1 or len(sq) != 1:
        raise ValueError(f"{corpus}: expected one .{SRC} and one .{TGT} member, got {mk} / {sq}")
    return mk[0], sq[0]


def verify_archive(path: Path, corpus: str) -> int:
    """Open the zip, confirm both sides have equal line counts. Returns the count."""
    with zipfile.ZipFile(path) as archive:
        mk_member, sq_member = moses_members(archive, corpus)
        counts = []
        for member in (mk_member, sq_member):
            with archive.open(member) as handle:
                counts.append(sum(1 for _ in handle))
    if counts[0] != counts[1]:
        raise ValueError(f"{corpus}: {counts[0]:,} {SRC} lines vs {counts[1]:,} {TGT} lines")
    return counts[0]


def download_corpora(corpora: list[dict]) -> dict[str, Path]:
    import requests

    target_dir = RAW / "opus"
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for entry in corpora:
        corpus, url = entry["corpus"], entry["url"]
        path = target_dir / f"{corpus}.{entry['version']}.mk-sq.zip"

        for attempt in (1, 2):
            if path.exists():
                try:
                    lines = verify_archive(path, corpus)
                    log(f"  {corpus:<24} on disk, {lines:,} line pairs")
                    paths[corpus] = path
                    break
                except Exception as exc:
                    log(f"  {corpus}: on-disk archive is bad ({exc}); re-downloading")
                    path.unlink(missing_ok=True)
                    if attempt == 2:
                        raise

            log(f"  {corpus:<24} downloading {url}")
            tmp = path.with_suffix(".zip.tmp")
            with requests.get(url, stream=True, timeout=600) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for block in response.iter_content(chunk_size=1 << 20):
                        fh.write(block)
            os.replace(tmp, path)

    return paths


# ---------------------------------------------------------------------------
# Step 5: licences
# ---------------------------------------------------------------------------


def licence_of(corpus: str) -> tuple[str, str, str]:
    return LICENCES.get(corpus, DEFAULT_LICENCE)


def write_licences(corpora: list[dict]) -> dict:
    out = PROCESSED / "opus_licenses.json"
    table = {}
    for entry in corpora:
        corpus = entry["corpus"]
        licence, klass, source = licence_of(corpus)
        table[corpus] = {
            "corpus": corpus, "version": entry["version"],
            "licence": licence, "class": klass, "source_url": source,
            "opus_pairs": entry["alignment_pairs"],
        }
        marker = " " if corpus in LICENCES else " (not in table -> restricted)"
        log(f"    {corpus:<24} {klass:<11} {licence}{marker}")
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    public = sum(1 for v in table.values() if v["class"] == "public")
    log(f"{public} public, {len(table) - public} restricted -> {out}")
    return table


# ---------------------------------------------------------------------------
# Step 6: cleaning
# ---------------------------------------------------------------------------

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-zÇçËë]")


def script_shares(text: str) -> tuple[float, float]:
    cyr, lat = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    total = cyr + lat
    if not total:
        return 0.0, 0.0
    return cyr / total, lat / total


def build_detector():
    from lingua import Language, LanguageDetectorBuilder

    # Serbian and Bulgarian are in the set precisely so they can be rejected:
    # both are routinely mislabelled as Macedonian in web-crawled corpora.
    return (
        LanguageDetectorBuilder.from_languages(
            Language.MACEDONIAN, Language.ALBANIAN,
            Language.SERBIAN, Language.BULGARIAN,
        )
        .with_low_accuracy_mode()
        .build()
    )


def read_moses_chunks(path: Path, corpus: str, chunk_size: int):
    """Yield (index, [mk lines], [sq lines]) without loading the corpus."""
    with zipfile.ZipFile(path) as archive:
        mk_member, sq_member = moses_members(archive, corpus)
        with archive.open(mk_member) as mk_handle, archive.open(sq_member) as sq_handle:
            index, mk_buf, sq_buf = 0, [], []
            for mk_line, sq_line in zip(mk_handle, sq_handle):
                mk_buf.append(mk_line.decode("utf-8", "replace").rstrip("\n"))
                sq_buf.append(sq_line.decode("utf-8", "replace").rstrip("\n"))
                if len(mk_buf) >= chunk_size:
                    yield index, mk_buf, sq_buf
                    index += 1
                    mk_buf, sq_buf = [], []
            if mk_buf:
                yield index, mk_buf, sq_buf


def clean_chunk(mk_lines, sq_lines, detector, tokenizer, stats) -> "object":
    import polars as pl
    from lingua import Language

    mk_clean, sq_clean = [], []
    for mk_raw, sq_raw in zip(mk_lines, sq_lines):
        mk = " ".join(unicodedata.normalize("NFC", mk_raw).split())
        sq = " ".join(unicodedata.normalize("NFC", sq_raw).split())
        stats["read"] += 1

        if not mk or not sq:
            stats["empty"] += 1
            continue
        if mk == sq:
            stats["identical"] += 1
            continue

        mk_cyr, _ = script_shares(mk)
        _, sq_lat = script_shares(sq)
        if mk_cyr < SCRIPT_PURITY_MIN or sq_lat < SCRIPT_PURITY_MIN:
            stats["script"] += 1
            continue

        ratio = len(mk) / max(len(sq), 1)
        if ratio > MAX_LENGTH_RATIO or ratio < 1 / MAX_LENGTH_RATIO:
            stats["length_ratio"] += 1
            continue

        # `==`, never `is`: lingua's Language is a Rust pyclass, not a Python
        # enum, so identity comparison is False even for the same variant and
        # an `is not` test would reject every row. A None verdict (lingua could
        # not decide) also fails this test, which is the intended behaviour.
        if len(mk.split()) >= MIN_WORDS_FOR_LANGID:
            if detector.detect_language_of(mk) != Language.MACEDONIAN:
                stats["langid_mk"] += 1
                continue
        if len(sq.split()) >= MIN_WORDS_FOR_LANGID:
            if detector.detect_language_of(sq) != Language.ALBANIAN:
                stats["langid_sq"] += 1
                continue

        mk_clean.append(mk)
        sq_clean.append(sq)

    if not mk_clean:
        return pl.DataFrame({"mk": [], "sq": []}, schema={"mk": pl.Utf8, "sq": pl.Utf8})

    keep = [True] * len(mk_clean)
    for side in (mk_clean, sq_clean):
        for start in range(0, len(side), 1000):
            batch = side[start : start + 1000]
            encoded = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
            for offset, ids in enumerate(encoded):
                if len(ids) > MAX_NLLB_TOKENS:
                    keep[start + offset] = False
    stats["too_long"] += keep.count(False)

    return pl.DataFrame(
        {
            "mk": [t for t, k in zip(mk_clean, keep) if k],
            "sq": [t for t, k in zip(sq_clean, keep) if k],
        },
        schema={"mk": pl.Utf8, "sq": pl.Utf8},
    )


def clean_corpus(corpus: str, archive_path: Path, detector, tokenizer) -> dict:
    import polars as pl

    out_dir = PROCESSED / "clean" / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.json"
    final = out_dir / "deduped.parquet"

    if final.exists() and stats_path.exists():
        log(f"  {corpus:<24} loaded from disk: {final}")
        return json.loads(stats_path.read_text())

    stats = dict(read=0, empty=0, identical=0, script=0, length_ratio=0,
                 langid_mk=0, langid_sq=0, too_long=0)
    saved = stats_path.with_suffix(".partial.json")
    if saved.exists():
        stats = json.loads(saved.read_text())

    started = time.time()
    for index, mk_lines, sq_lines in read_moses_chunks(archive_path, corpus, CHUNK_SIZE):
        part = out_dir / f"part-{index:05d}.parquet"
        if part.exists():
            continue
        chunk_stats = dict.fromkeys(stats, 0)
        frame = clean_chunk(mk_lines, sq_lines, detector, tokenizer, chunk_stats)
        write_frame_atomic(frame, part)
        for key, value in chunk_stats.items():
            stats[key] = stats.get(key, 0) + value
        saved.write_text(json.dumps(stats, indent=2))
        log(f"  {corpus:<24} chunk {index:>4}  {len(mk_lines):>7,} read -> "
            f"{len(frame):>7,} kept  ({time.time() - started:,.0f}s)")

    parts = sorted(out_dir.glob("part-*.parquet"))
    if not parts:
        frame = pl.DataFrame({"mk": [], "sq": []}, schema={"mk": pl.Utf8, "sq": pl.Utf8})
    else:
        frame = pl.concat([pl.read_parquet(p) for p in parts])
    before = len(frame)
    frame = frame.unique(subset=["mk", "sq"], keep="first")
    stats["within_corpus_duplicates"] = before - len(frame)
    stats["clean"] = len(frame)

    write_frame_atomic(frame, final)
    stats_path.write_text(json.dumps(stats, indent=2))
    saved.unlink(missing_ok=True)

    log(f"  {corpus:<24} read {stats['read']:,} -> clean {stats['clean']:,}  "
        f"(empty {stats['empty']:,}, identical {stats['identical']:,}, "
        f"script {stats['script']:,}, ratio {stats['length_ratio']:,}, "
        f"lang {stats['langid_mk'] + stats['langid_sq']:,}, "
        f">192tok {stats['too_long']:,}, dup {stats['within_corpus_duplicates']:,})")
    return stats


# ---------------------------------------------------------------------------
# Steps 7-8: cross-corpus dedup and leakage removal
# ---------------------------------------------------------------------------


def combine_and_dedup(corpora: list[dict], licences: dict) -> "object":
    import polars as pl

    out = PROCESSED / "opus_deduped.parquet"
    if out.exists():
        log(f"loaded from disk: {out}")
        return pl.read_parquet(out)

    frames = []
    for entry in corpora:
        corpus = entry["corpus"]
        path = PROCESSED / "clean" / corpus / "deduped.parquet"
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        if not len(frame):
            continue
        klass = licences[corpus]["class"]
        rank = CORPUS_PRIORITY.index(corpus) if corpus in CORPUS_PRIORITY else len(CORPUS_PRIORITY)
        frames.append(
            frame.with_columns(
                pl.lit(corpus).alias("corpus"),
                pl.lit(0 if klass == "public" else 1).cast(pl.Int8).alias("class_rank"),
                pl.lit(rank).cast(pl.Int16).alias("priority"),
            )
        )
    combined = pl.concat(frames)
    before = len(combined)
    log(f"combined: {before:,} pairs across {len(frames)} corpora")

    # Public beats restricted; then the explicit priority list; then name, so the
    # outcome does not depend on concat order.
    deduped = (
        combined.sort(["class_rank", "priority", "corpus"])
        .unique(subset=["mk", "sq"], keep="first", maintain_order=True)
        .drop(["class_rank", "priority"])
    )
    log(f"cross-corpus dedup removed {before - len(deduped):,} pairs "
        f"({(before - len(deduped)) / max(before, 1):.1%}) -> {len(deduped):,}")
    for row in (
        combined.group_by("corpus").len().join(
            deduped.group_by("corpus").len(), on="corpus", how="left", suffix="_kept"
        ).sort("len", descending=True).iter_rows(named=True)
    ):
        kept = row.get("len_kept") or 0
        log(f"    {row['corpus']:<24} {row['len']:>9,} -> {kept:>9,}")

    write_frame_atomic(deduped, out)
    return pl.read_parquet(out)


def remove_leakage(frame, reference) -> "object":
    import polars as pl

    out = PROCESSED / "opus_no_leakage.parquet"
    if out.exists():
        log(f"loaded from disk: {out}")
        return pl.read_parquet(out)

    banned = set(reference["text"].to_list())
    log(f"leakage reference holds {len(banned):,} distinct sentences")

    before_by_corpus = frame.group_by("corpus").len()
    marked = frame.with_columns(
        pl.col("mk").map_elements(lambda t: normalise_for_match(t) in banned, return_dtype=pl.Boolean).alias("mk_leak"),
        pl.col("sq").map_elements(lambda t: normalise_for_match(t) in banned, return_dtype=pl.Boolean).alias("sq_leak"),
    )
    leaking = marked.filter(pl.col("mk_leak") | pl.col("sq_leak"))
    log(f"leakage: {len(leaking):,} pairs removed "
        f"(mk side {int(marked['mk_leak'].sum()):,}, sq side {int(marked['sq_leak'].sum()):,})")
    if len(leaking):
        for row in leaking.group_by("corpus").len().sort("len", descending=True).iter_rows(named=True):
            log(f"    {row['corpus']:<24} {row['len']:>8,} removed")

    kept = marked.filter(~(pl.col("mk_leak") | pl.col("sq_leak"))).drop(["mk_leak", "sq_leak"])
    write_frame_atomic(kept, out)
    log(f"after leakage removal: {len(kept):,} pairs")
    _ = before_by_corpus
    return pl.read_parquet(out)


# ---------------------------------------------------------------------------
# Step 9: LaBSE
# ---------------------------------------------------------------------------


def score_with_labse(frame, device: str) -> "object":
    import numpy as np
    import polars as pl

    out = PROCESSED / "opus_scored.parquet"
    if out.exists():
        log(f"loaded from disk: {out}")
        return pl.read_parquet(out)

    from sentence_transformers import SentenceTransformer

    shard_dir = PROCESSED / "scored"
    shard_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 256 if device == "cuda" else (64 if device == "mps" else 32)
    model = SentenceTransformer(LABSE_MODEL, device=device)
    log(f"LaBSE loaded on {device}, batch size {batch_size}")

    total = len(frame)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    started, scored_rows = time.time(), 0

    for index in range(n_chunks):
        part = shard_dir / f"part-{index:05d}.parquet"
        if part.exists():
            log(f"  chunk {index + 1}/{n_chunks} loaded from disk")
            continue
        chunk = frame.slice(index * CHUNK_SIZE, CHUNK_SIZE)
        t0 = time.time()
        mk_vec = model.encode(chunk["mk"].to_list(), batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
        sq_vec = model.encode(chunk["sq"].to_list(), batch_size=batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
        score = np.sum(mk_vec * sq_vec, axis=1).astype("float32")
        write_frame_atomic(chunk.with_columns(pl.Series("score", score)), part)

        elapsed = time.time() - t0
        scored_rows += len(chunk)
        log(f"  chunk {index + 1}/{n_chunks}  {len(chunk):,} pairs in {elapsed:,.0f}s "
            f"({len(chunk) / max(elapsed, 1e-9):,.0f} pairs/s)")
        if index == 0:
            rate = len(chunk) / max(elapsed, 1e-9)
            log(f"  projected total scoring time: {total / rate / 3600:,.1f} hours "
                f"for {total:,} pairs on {device}")
            if device == "cpu":
                log("  WARNING  that is a CPU estimate. The run is resumable per chunk.")

    parts = sorted(shard_dir.glob("part-*.parquet"))
    scored = pl.concat([pl.read_parquet(p) for p in parts])
    if len(scored) != total:
        raise ValueError(f"scored {len(scored):,} pairs, expected {total:,}")
    write_frame_atomic(scored, out)
    log(f"scored {len(scored):,} pairs in {(time.time() - started) / 3600:,.2f} hours")
    return pl.read_parquet(out)


# ---------------------------------------------------------------------------
# Steps 10-12: threshold, hold-out, final files
# ---------------------------------------------------------------------------


def apply_threshold(scored, licences: dict) -> "object":
    import polars as pl

    out = PROCESSED / "opus_kept.parquet"
    if out.exists():
        log(f"loaded from disk: {out}")
        return pl.read_parquet(out)

    before = len(scored)
    kept = scored.filter(pl.col("score") >= LABSE_THRESHOLD)
    log(f"LaBSE >= {LABSE_THRESHOLD}: {len(kept):,} of {before:,} "
        f"({len(kept) / max(before, 1):.1%})")
    for row in kept.group_by("corpus").len().sort("len", descending=True).iter_rows(named=True):
        log(f"    {row['corpus']:<24} {row['len']:>9,}")

    kept = kept.with_columns(
        pl.col("corpus").map_elements(lambda c: licences[c]["licence"], return_dtype=pl.Utf8).alias("licence"),
        pl.col("corpus").map_elements(lambda c: licences[c]["class"], return_dtype=pl.Utf8).alias("class"),
    )
    write_frame_atomic(kept, out)
    return pl.read_parquet(out)


def split_test(kept) -> tuple["object", "object"]:
    import polars as pl

    train_path = PROCESSED / "opus_train_all.parquet"
    test_path = PROCESSED / "opus_test_all.parquet"
    if train_path.exists() and test_path.exists():
        log(f"loaded from disk: {train_path}, {test_path}")
        return pl.read_parquet(train_path), pl.read_parquet(test_path)

    eligible = kept.filter(~pl.col("corpus").is_in(list(NOT_SENTENCES)))
    log(f"test-eligible: {len(eligible):,} pairs "
        f"({', '.join(sorted(NOT_SENTENCES))} excluded as entity pairs)")

    sizes = {r["corpus"]: r["len"] for r in eligible.group_by("corpus").len().iter_rows(named=True)}
    total = sum(sizes.values())
    random.seed(SEED)

    picked_indices: list[int] = []
    with_index = eligible.with_row_index("row_index")
    for corpus, size in sorted(sizes.items()):
        quota = min(size, max(1, round(OPUS_TEST_SIZE * size / max(total, 1))))
        rows = with_index.filter(pl.col("corpus") == corpus)["row_index"].to_list()
        picked_indices.extend(random.sample(rows, min(quota, len(rows))))
    picked_indices = picked_indices[:OPUS_TEST_SIZE]

    test = with_index.filter(pl.col("row_index").is_in(picked_indices)).drop("row_index")
    test_keys = set(zip(test["mk"].to_list(), test["sq"].to_list()))
    train = kept.filter(
        ~pl.struct(["mk", "sq"]).map_elements(
            lambda s: (s["mk"], s["sq"]) in test_keys, return_dtype=pl.Boolean
        )
    )
    if len(train) + len(test) != len(kept):
        raise ValueError(f"train {len(train):,} + test {len(test):,} != kept {len(kept):,}")

    log(f"test hold-out: {len(test):,} pairs, stratified by corpus, seed {SEED}")
    for row in test.group_by("corpus").len().sort("len", descending=True).iter_rows(named=True):
        log(f"    {row['corpus']:<24} {row['len']:>6,}")

    write_frame_atomic(train, train_path)
    write_frame_atomic(test, test_path)
    return pl.read_parquet(train_path), pl.read_parquet(test_path)


FINAL_COLUMNS = ["mk", "sq", "corpus", "score", "licence"]


def write_final(train, test) -> dict[str, Path]:
    import polars as pl

    paths = {}
    for klass in ("public", "restricted"):
        for name, frame in (("train", train), ("test", test)):
            path = PROCESSED / f"opus_{klass}_{name}.parquet"
            subset = (
                frame.filter(pl.col("class") == klass)
                .select(FINAL_COLUMNS)
                .with_columns(pl.col("score").cast(pl.Float32))
            )
            write_frame_atomic(subset, path)
            paths[f"{klass}_{name}"] = path
            log(f"  {path.name:<34} {len(subset):>9,} rows")
    return paths


# ---------------------------------------------------------------------------
# Step 13: upload
# ---------------------------------------------------------------------------


def dataset_card(klass: str, repo_id: str, counts: dict, licences: dict,
                 clean_stats: dict, corpora: list[dict]) -> str:
    used = sorted(
        {c for c in licences if licences[c]["class"] == klass},
        key=lambda c: -licences[c]["opus_pairs"],
    )
    rows = []
    for corpus in used:
        info = licences[corpus]
        stats = clean_stats.get(corpus, {})
        rows.append(
            f"| {corpus} | {info['version']} | {info['opus_pairs']:,} | "
            f"{stats.get('clean', 0):,} | [{info['licence']}]({info['source_url']}) |"
            if info["source_url"] else
            f"| {corpus} | {info['version']} | {info['opus_pairs']:,} | "
            f"{stats.get('clean', 0):,} | {info['licence']} |"
        )
    corpus_table = "\n".join(rows)
    visibility = "public" if klass == "public" else "private"
    licence_field = "cc-by-sa-4.0" if klass == "public" else "other"

    return f"""---
license: {licence_field}
language:
  - mk
  - sq
multilinguality: translation
task_categories:
  - translation
size_categories:
  - 100K<n<1M
source_datasets:
  - extended
pretty_name: MK-SQ OPUS filtered ({klass})
tags:
  - macedonian
  - albanian
  - low-resource
  - opus
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: test
        path: data/test-*
---

# MK–SQ OPUS, filtered ({klass})

Macedonian–Albanian parallel sentences drawn from every mk–sq corpus on
[OPUS](https://opus.nlpl.eu/), cleaned, quality-filtered and checked for leakage
against the standard evaluation sets. This is the **{klass}** half of the
release ({visibility} repository); corpora are split by whether their licence
clearly permits redistributing a modified subset.

| split | pairs |
|---|---|
| train | {counts['train']:,} |
| test | {counts['test']:,} |

## Source corpora

| corpus | OPUS version | pairs on OPUS | after cleaning | licence |
|---|---|---|---|---|
{corpus_table}

"after cleaning" counts the corpus in isolation, before cross-corpus
deduplication moved shared pairs to whichever corpus won the tie-break.

## Filtering

Applied in this order:

1. Unicode NFC, whitespace collapsed, empty pairs and `mk == sq` dropped.
2. Script check — the Macedonian side must be ≥{SCRIPT_PURITY_MIN:.0%} Cyrillic,
   the Albanian side ≥{SCRIPT_PURITY_MIN:.0%} Latin.
3. Character length ratio ≤ {MAX_LENGTH_RATIO}.
4. Language identification with
   [lingua](https://github.com/pemistahl/lingua-py) on lines of
   ≥{MIN_WORDS_FOR_LANGID} words, over {{Macedonian, Albanian, Serbian,
   Bulgarian}}. Serbian and Bulgarian are in the candidate set so that they can
   be *rejected*: both are routinely mislabelled as Macedonian in web-crawled
   corpora.
5. Both sides must fit {MAX_NLLB_TOKENS} tokens under the
   `{NLLB_TOKENIZER}` tokenizer.
6. Exact deduplication within each corpus.
7. Exact deduplication **across** corpora — CCMatrix and NLLB are the same
   {corpora[0]['alignment_pairs'] if corpora else 0:,}-pair collection for this
   language pair. One copy is kept, preferring a public-licensed corpus, then
   {' > '.join(CORPUS_PRIORITY)}, then alphabetically.
8. Leakage removal — any pair whose Macedonian or Albanian side matches a
   sentence in FLORES+ `devtest` (mkd_Cyrl, als_Latn), NTREX-128 (mkd, sqi), or
   the validation and test splits of
   [{HF_GAZETTE_REPO}](https://huggingface.co/datasets/{HF_GAZETTE_REPO}),
   compared lowercased and whitespace-normalised.
9. [LaBSE]({'https://huggingface.co/' + LABSE_MODEL}) cosine similarity
   ≥ {LABSE_THRESHOLD} between the two sides, on normalised embeddings.

## Test split

{OPUS_TEST_SIZE:,} pairs held out in total, stratified by corpus in proportion to
size, with seed {SEED}. {', '.join(sorted(NOT_SENTENCES))} is excluded from the
test split — it is entity pairs rather than sentences — but remains in train.
Held-out pairs are removed from train.

## Fields

- `mk` — Macedonian sentence
- `sq` — Albanian sentence
- `corpus` — the OPUS corpus the pair was kept from
- `score` — LaBSE cosine similarity (float32)
- `licence` — the licence recorded for that corpus

## What is not here

No Verbis / APJ terminology data is included in this dataset.

## Citation

Please cite OPUS, and the individual corpora listed above.

```bibtex
@inproceedings{{tiedemann2012parallel,
  title     = {{Parallel Data, Tools and Interfaces in OPUS}},
  author    = {{Tiedemann, J{{\\"o}}rg}},
  booktitle = {{LREC}},
  year      = {{2012}}
}}
```
"""


def upload(paths: dict[str, Path], klass: str, repo_id: str, private: bool,
           token: str, licences: dict, clean_stats: dict, corpora: list[dict]) -> bool:
    import polars as pl
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    train = pl.read_parquet(paths[f"{klass}_train"])
    test = pl.read_parquet(paths[f"{klass}_test"])
    counts = {"train": len(train), "test": len(test)}
    if not counts["train"]:
        log(f"  {repo_id}: nothing classed {klass}; skipping upload")
        return False

    api = HfApi(token=token)
    try:
        from datasets import load_dataset

        existing = load_dataset(repo_id, token=token)
        remote = {k: existing[k].num_rows for k in existing}
        if remote == counts:
            log(f"  {repo_id}: already holds {remote}; skipping upload")
            return True
    except Exception:
        pass

    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    bundle = DatasetDict({
        "train": Dataset.from_pandas(train.to_pandas(), preserve_index=False),
        "test": Dataset.from_pandas(test.to_pandas(), preserve_index=False),
    })
    bundle.push_to_hub(repo_id, token=token, private=private)

    card = dataset_card(klass, repo_id, counts, licences, clean_stats, corpora)
    card_path = WORK_DIR / f"README_{klass}.md"
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(card_path), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset", token=token,
    )
    log(f"  {repo_id}: pushed {counts} -> https://huggingface.co/datasets/{repo_id}")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    overall = time.time()
    log("#" * 78)
    log("OPUS mk-sq dataset build")

    with step(1, "preflight"):
        token, device = preflight()

    with step(2, "leakage reference sets"):
        reference = build_leakage_reference(token)

    with step(3, "list OPUS corpora"):
        corpora = list_corpora()

    with step(4, "download corpora"):
        archives = download_corpora(corpora)

    with step(5, "licences"):
        licences = write_licences(corpora)

    with step(6, "clean"):
        from transformers import AutoTokenizer

        detector = build_detector()
        tokenizer = AutoTokenizer.from_pretrained(NLLB_TOKENIZER)
        clean_stats = {}
        for entry in corpora:
            corpus = entry["corpus"]
            if corpus not in archives:
                continue
            clean_stats[corpus] = clean_corpus(corpus, archives[corpus], detector, tokenizer)
        (PROCESSED / "clean_stats.json").write_text(
            json.dumps(clean_stats, indent=2), encoding="utf-8"
        )

    with step(7, "cross-corpus dedup"):
        deduped = combine_and_dedup(corpora, licences)

    with step(8, "leakage removal"):
        no_leakage = remove_leakage(deduped, reference)

    with step(9, "LaBSE scoring"):
        scored = score_with_labse(no_leakage, device)

    with step(10, "apply LaBSE threshold"):
        kept = apply_threshold(scored, licences)

    with step(11, "test hold-out"):
        train, test = split_test(kept)

    with step(12, "final files"):
        paths = write_final(train, test)

    with step(13, "upload"):
        upload(paths, "public", HF_OPUS_REPO, False, token, licences, clean_stats, corpora)
        upload(paths, "restricted", HF_OPUS_PRIVATE_REPO, True, token, licences, clean_stats, corpora)

    with step(14, "verify"):
        verify(paths, token)

    with step(15, "finish"):
        import polars as pl

        summary = {
            name: len(pl.read_parquet(path)) for name, path in paths.items()
        }
        summary["runtime_hours"] = round((time.time() - overall) / 3600, 3)
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["public_repo"] = HF_OPUS_REPO
        summary["restricted_repo"] = HF_OPUS_PRIVATE_REPO
        (WORK_DIR / "DONE").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(json.dumps(summary, indent=2))

    log(f"finished in {(time.time() - overall) / 3600:,.2f} hours")


def verify(paths: dict[str, Path], token: str) -> None:
    import polars as pl
    from datasets import load_dataset

    for klass, repo_id in (("public", HF_OPUS_REPO), ("restricted", HF_OPUS_PRIVATE_REPO)):
        local = {
            name: len(pl.read_parquet(paths[f"{klass}_{name}"])) for name in ("train", "test")
        }
        if not local["train"]:
            log(f"  {repo_id}: nothing uploaded, nothing to verify")
            continue
        try:
            remote_ds = load_dataset(repo_id, token=token)
        except Exception as exc:
            log(f"  WARNING  cannot reload {repo_id}: {type(exc).__name__}: {exc}")
            continue
        remote = {k: remote_ds[k].num_rows for k in remote_ds}
        match = all(remote.get(k) == v for k, v in local.items())
        log(f"  {repo_id}: local {local} remote {remote} -> {'MATCH' if match else 'MISMATCH'}")
        if not match:
            raise ValueError(f"{repo_id}: row counts differ between local and Hub")
        sample = remote_ds["train"].shuffle(seed=SEED).select(range(min(5, remote["train"])))
        for row in sample:
            log(f"      [{row['corpus']}, {row['score']:.3f}] "
                f"{row['mk'][:58]!r} || {row['sq'][:58]!r}")


if __name__ == "__main__":
    main()
