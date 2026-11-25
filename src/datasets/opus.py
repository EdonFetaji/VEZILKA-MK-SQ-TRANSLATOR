"""General-domain OPUS MK–SQ pairs (dataset tags G, G_REPLAY, G_REST)."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pyarrow as pa

from src.datasets.base import (
    CACHE_ROOT,
    WRITE_CHUNK,
    DatasetLoader,
    RecordKind,
    hub_files,
    parquet_batches,
    sha256,
)
from src.pipeline.registry import register_loader

#: Repository columns carried into ``meta`` (``pair_id`` becomes ``doc_id``).
META_COLUMNS = (
    "unit_type",
    "domain",
    "translation_origin",
    "pairing_method",
    "english_mediated",
    "release_scope",
    "flags",
)

# ── tag G (TG-D0, plan §06): the TG-10 training set ─────────────────────────────
#: Human-translated sources: every row, both directions.
G_HUMAN = (
    "SETIMES", "TED2020", "ELRC-5067-SciPar", "GlobalVoices", "EUbookshop", "NeuLab-TedTalks", "Tatoeba",
    "ELRC-wikipedia_health",
)
#: Mined sources: LaBSE-ranked, never below the floor, one direction each. ``None`` = no cap.
G_MINED_TOP = {"NLLB": 900_000, "MultiHPLT": 300_000, "MultiMaCoCu": None}
G_LABSE_FLOOR = 0.80
#: OpenSubtitles: random sample, preferring lines of 3+ words (both sides), one direction each.
G_SUBS, G_SUBS_ROWS = "OpenSubtitles", 100_000
#: Dev/test sets a training pair must not repeat (plan TG-D0), as (loader, split).
G_HELD_OUT = (("flores", "dev"), ("flores", "devtest"), ("ntrex", "test"), ("gazette", "validation"),
              ("gazette", "test"))
LABSE_MODEL = "sentence-transformers/LaBSE"
SCORE_BLOCK = 65_536  # mined rows per cached LaBSE part file (resumable scoring)


def norm(text: str) -> str:
    """Casefolded, whitespace-normalised sentence, for the dev/test overlap check."""
    return " ".join(str(text).split()).casefold()


def select_g(
    sources: np.ndarray,
    scores: np.ndarray,
    held_out: np.ndarray,
    subs_ok: np.ndarray,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Membership, directions and the single shuffle of tag G. Pure (unit-tested).

    ``sources``: source_id per repository row; ``scores``: LaBSE per row (NaN if unscored);
    ``held_out``: row repeats a dev/test sentence; ``subs_ok``: OpenSubtitles row of 3+ words.
    Returns (row_index, direction) per example, in the frozen order, and per-source stats.
    """
    rng = np.random.default_rng(seed)
    keep = ~held_out
    stats: dict[str, Any] = {}
    both: list[np.ndarray] = []
    one: list[np.ndarray] = []
    for src in G_HUMAN:
        rows = np.flatnonzero((sources == src) & keep)
        both.append(rows)
        stats[src] = {"available": int((sources == src).sum()), "selected": len(rows), "directions": "both"}
    for src, top in G_MINED_TOP.items():
        is_src = sources == src
        ok = np.flatnonzero(is_src & keep & (scores >= G_LABSE_FLOOR))
        ok = ok[np.argsort(-scores[ok], kind="stable")]
        rows = np.sort(ok if top is None else ok[:top])
        one.append(rows)
        stats[src] = {"available": int(is_src.sum()), "held_out": int((is_src & held_out).sum()),
                      "pass_floor": len(ok), "target": top, "selected": len(rows),
                      "min_labse": float(scores[rows].min()) if len(rows) else None, "directions": "one each"}
    is_subs = (sources == G_SUBS) & keep
    pref, rest = np.flatnonzero(is_subs & subs_ok), np.flatnonzero(is_subs & ~subs_ok)
    pick = rng.permutation(pref)[:G_SUBS_ROWS]
    if len(pick) < G_SUBS_ROWS:
        pick = np.concatenate([pick, rng.permutation(rest)[: G_SUBS_ROWS - len(pick)]])
    one.append(np.sort(pick))
    stats[G_SUBS] = {"available": int((sources == G_SUBS).sum()), "three_plus_words": len(pref),
                     "selected": len(pick), "directions": "one each"}

    both_rows, one_rows = np.concatenate(both), np.concatenate(one)
    flip = rng.permutation(len(one_rows)) < len(one_rows) // 2  # exactly 50/50, seeded
    rows = np.concatenate([both_rows, both_rows, one_rows])
    dirs = np.concatenate([np.zeros(len(both_rows), np.int8), np.ones(len(both_rows), np.int8),
                           flip.astype(np.int8)])  # 0 = mk-sq, 1 = sq-mk
    order = rng.permutation(len(rows))  # the one shuffle
    stats["_total"] = {"rows": len(both_rows) + len(one_rows), "examples": len(rows),
                       "held_out_rows_removed": int(held_out.sum())}
    return rows[order], dirs[order], stats


class Labse:
    """LaBSE cosine between the two sides (normalised embeddings), fp16 on CUDA."""

    def __init__(self, batch_size: int = 512, model: str = LABSE_MODEL) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=self.device)
        if self.device == "cuda":
            self.model.half()
        self.batch_size, self.name = batch_size, model

    def __call__(self, mk: list[str], sq: list[str]) -> np.ndarray:
        kw = {"batch_size": self.batch_size, "convert_to_numpy": True, "normalize_embeddings": True,
              "show_progress_bar": False}
        a = self.model.encode(mk, **kw).astype(np.float32)
        b = self.model.encode(sq, **kw).astype(np.float32)
        return (a * b).sum(axis=1)


def frozen_dir(revision: str | None, seed: int, root: Path | None = None) -> Path:
    return Path(root or CACHE_ROOT) / "frozen" / "G" / f"{(revision or 'main')[:12]}-s{seed}"


def held_out_sentences(allow_missing: bool = False, log=print) -> tuple[set[str], dict[str, int]]:
    from src.pipeline.registry import get_loader

    seen: set[str] = set()
    counts: dict[str, int] = {}
    for name, split in G_HELD_OUT:
        try:
            n = 0
            for batch in get_loader(name)().stream_batch_from_disk(split, batch_size=4096):
                for r in batch:
                    seen.add(norm(r["mk"]))
                    seen.add(norm(r["sq"]))
                    n += 1
            counts[f"{name}/{split}"] = n
        except Exception as e:
            if not allow_missing:
                raise RuntimeError(f"held-out set {name}/{split} unavailable ({type(e).__name__}: {e}); fix "
                                   "access (hf auth login, FLORES+ terms) or pass --allow-missing-eval") from e
            log(f"WARNING held-out set {name}/{split} skipped: {type(e).__name__}: {e}")
            counts[f"{name}/{split}"] = -1
    return seen, counts


def freeze_g(
    loader: OpusLoader,
    seed: int = 1,
    scorer=None,
    allow_missing_eval: bool = False,
    held_out: set[str] | None = None,
    force: bool = False,
    batch_size: int = 512,
    log=print,
) -> Path:
    """Build and freeze tag G: ``<cache>/frozen/G/<rev>-s<seed>/train.arrow`` + ``train.json``.

    One sequential pass over the cached repository file collects source ids, the dev/test overlap and
    OpenSubtitles word counts, and LaBSE-scores every NLLB / MultiHPLT / MultiMaCoCu pair (cached in
    ``labse/part-*.npy`` per 65,536 mined rows, so an interrupted run resumes). Then :func:`select_g`
    picks rows, directions and the order, and the examples are written in that order.
    """
    src_path = loader.download("train")
    src_meta = json.loads(src_path.with_suffix(".json").read_text())
    out_dir = frozen_dir(src_meta.get("revision") or loader.opt("revision"), seed,
                         Path(loader.options.get("cache_dir", CACHE_ROOT)))
    out = out_dir / "train.arrow"
    if out.exists() and not force:
        log(f"G already frozen: {out}")
        return out
    parts = out_dir / "labse"
    parts.mkdir(parents=True, exist_ok=True)
    n_rows = int(src_meta["rows"])
    if held_out is None:
        held_out, held_counts = held_out_sentences(allow_missing_eval, log)
    else:
        held_counts = {"given": len(held_out)}
    mined = set(G_MINED_TOP)

    sources = np.empty(n_rows, dtype=object)
    overlap = np.zeros(n_rows, dtype=bool)
    subs_ok = np.zeros(n_rows, dtype=bool)
    scores = np.full(n_rows, np.nan, dtype=np.float32)
    buf_idx: list[int] = []
    buf_mk: list[str] = []
    buf_sq: list[str] = []
    block = 0
    t0 = time.monotonic()

    def flush() -> None:
        nonlocal block, scorer
        part = parts / f"part-{block:05d}.npy"
        if part.exists():
            vals = np.load(part)
        else:
            if scorer is None:  # loaded only if some block isn't cached yet
                scorer = Labse(batch_size=batch_size)
            vals = np.asarray(scorer(buf_mk, buf_sq), dtype=np.float32)
            tmp = part.with_name(part.name + ".tmp.npy")
            np.save(tmp, vals)
            os.replace(tmp, part)
        if len(vals) != len(buf_idx):
            raise RuntimeError(f"{part}: {len(vals)} scores for {len(buf_idx)} rows (stale cache? delete {parts})")
        scores[np.asarray(buf_idx)] = vals
        block += 1
        done = block * SCORE_BLOCK
        log(f"  LaBSE block {block}: ~{done:,} mined rows scored ({time.monotonic() - t0:.0f}s)")
        buf_idx.clear()
        buf_mk.clear()
        buf_sq.clear()

    i = 0
    with pa.OSFile(str(src_path)) as f:
        for chunk in pa.ipc.open_stream(f):
            src_ids = chunk.column("source_id").to_pylist()
            mk = chunk.column("mk").to_pylist()
            sq = chunk.column("sq").to_pylist()
            for s, a, b in zip(src_ids, mk, sq, strict=True):
                sources[i] = s
                overlap[i] = norm(a) in held_out or norm(b) in held_out
                if s == G_SUBS:
                    subs_ok[i] = len(a.split()) >= 3 and len(b.split()) >= 3
                elif s in mined:
                    buf_idx.append(i)
                    buf_mk.append(a)
                    buf_sq.append(b)
                    if len(buf_idx) == SCORE_BLOCK:
                        flush()
                i += 1
    if buf_idx:
        flush()
    if i != n_rows:
        raise RuntimeError(f"{src_path}: read {i} rows, sidecar says {n_rows}")

    rows, dirs, stats = select_g(sources, scores, overlap, subs_ok, seed)
    log(f"G: {stats['_total']['rows']:,} rows -> {stats['_total']['examples']:,} examples")

    table = pa.ipc.open_stream(pa.memory_map(str(src_path))).read_all()  # mapped, not copied
    tmp = out.with_name(out.name + ".tmp")
    names = ("mk-sq", "sq-mk")
    with pa.OSFile(str(tmp), "wb") as sink:
        writer = None
        for lo in range(0, len(rows), WRITE_CHUNK):
            idx = rows[lo : lo + WRITE_CHUNK]
            t = table.take(pa.array(idx)).select(["mk", "sq", "source_id", "doc_id", "meta"])
            sc = scores[idx]
            t = (t.append_column("direction", pa.array([names[d] for d in dirs[lo : lo + WRITE_CHUNK]]))
                 .append_column("labse", pa.array(sc, mask=np.isnan(sc), type=pa.float32()))
                 .append_column("row_index", pa.array(idx, type=pa.int64()))
                 .append_column("example_id", pa.array([f"G:{lo + k:07d}" for k in range(len(idx))])))
            if writer is None:
                writer = pa.ipc.new_stream(sink, t.schema)
            writer.write_table(t, max_chunksize=WRITE_CHUNK)
        if writer is not None:
            writer.close()
    os.replace(tmp, out)
    sidecar = {
        "tag": "G", "seed": seed, "rows": stats["_total"]["rows"], "examples": len(rows),
        "sha256": sha256(out), "sources": stats, "labse_model": getattr(scorer, "name", LABSE_MODEL),
        "labse_floor": G_LABSE_FLOOR, "held_out_sets": held_counts,
        "opus": {k: src_meta.get(k) for k in ("repo", "revision", "sha256", "rows")},
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    log(f"froze G -> {out}")
    return out


@register_loader("opus")
class OpusLoader(DatasetLoader):
    """``EdonFetaji/opus-mk-sq-clean-dedup``, default config. Rows are selected by ``source_id``.

    ``source_id`` is the OPUS corpus name (``SETIMES``, ``NLLB``, …; the repository's
    ``opus:<corpus>:<version>`` without prefix and version). The default config has no document
    ids, so ``doc_id`` is the row's ``pair_id`` (row-level bootstrap units).

    Selection rules (see ``research_plan/tg_datasets.csv``, tag G):

    * ``tag``: ``G``, ``G_REPLAY`` or ``G_REST`` (the frozen TG-D0 membership)
    * ``source_ids``: list of OPUS sources, e.g. ``[SETIMES, TED2020]``
    * ``labse_floor``: minimum LaBSE score (0.80 for NLLB, MultiHPLT and MultiMaCoCu; never lowered)
    * ``top_k``: per-source cap by LaBSE rank (NLLB 900,000; MultiHPLT 300,000)
    * ``stream_from`` / ``stream_to``: a slice of TG-10's single seed-42 shuffle (size-check branches)

    QED, WikiMatrix, wikimedia, MultiCCAligned and GNOME are never in G.
    """

    kind = RecordKind.PARALLEL
    splits = ("train",)
    select_rules: ClassVar[tuple[str, ...]] = (
        "tag",
        "source_ids",
        "labse_floor",
        "top_k",
        "stream_from",
        "stream_to",
        "seed",
    )
    repo = "EdonFetaji/opus-mk-sq-clean-dedup"
    revision = "0920a1440aa07997eea26582ef23a0bd41c9b8df"

    def _read(self, split: str):
        paths, sha = hub_files(self.opt("repo"), self.opt("revision"), prefix="data/train-")
        self._source = {"repo": self.opt("repo"), "revision": sha, "config": "default"}
        for c in parquet_batches(paths, columns=["pair_id", "source_id", "mk", "sq", *META_COLUMNS]):
            meta = zip(*(c[k] for k in META_COLUMNS))
            yield {
                "mk": c["mk"],
                "sq": c["sq"],
                "source_id": [s.split(":")[1] for s in c["source_id"]],
                "doc_id": c["pair_id"],
                "meta": [json.dumps(dict(zip(META_COLUMNS, m)), ensure_ascii=False) for m in meta],
            }

    def frozen_path(self, tag: str = "G", seed: int = 1) -> Path:
        if tag != "G":
            raise NotImplementedError(f"tag {tag!r} is not frozen yet (only G, for TG-10)")
        side = self.path("train").with_suffix(".json")
        rev = json.loads(side.read_text()).get("revision") if side.exists() else None
        return frozen_dir(rev or self.opt("revision"), seed, Path(self.options.get("cache_dir", CACHE_ROOT)))

    def select(self, ds, **rules: Any):
        """``tag: G`` → the frozen TG-D0 file (already in its one shuffled order, with ``direction``).

        ``source_ids`` filters it (order kept); ``stream_from`` / ``stream_to`` slice it by example.
        ``ds`` (the whole repository) is not used: G's membership lives in the frozen file.
        """
        unknown = set(rules) - {"tag", "source_ids", "stream_from", "stream_to", "seed"}
        if unknown:
            raise NotImplementedError(f"opus select rules {sorted(unknown)} are not implemented yet")
        tag = rules.get("tag")
        if tag is None:
            raise ValueError("opus select needs a tag (G)")
        path = self.frozen_path(tag, int(rules.get("seed", 1)))
        if not (path / "train.arrow").exists():
            raise FileNotFoundError(f"{path / 'train.arrow'} missing: run `python training_pipeline.py freeze G`")
        import datasets

        out = datasets.Dataset.from_file(str(path / "train.arrow"))
        if rules.get("source_ids"):
            keep = set(rules["source_ids"])
            out = out.filter(lambda b: [s in keep for s in b["source_id"]], batched=True,
                             input_columns=None, batch_size=65_536)
        if "stream_from" in rules or "stream_to" in rules:
            out = out.select(range(int(rules.get("stream_from", 0)), int(rules.get("stream_to", len(out)))))
        return self.validate(out)
