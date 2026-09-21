"""Cross-split near-duplicate sweep (constraint C4).

Exact deduplication is not enough for gazette text. The same enacting formula
recurs across issues and years with a changed date, number or ministry name, so
two segments can differ by three characters and still be the same segment for
every purpose that matters to an evaluation.

The sweep is MinHash/LSH over character n-grams, run **between** splits: every
dev/test segment is compared against the whole of train, and the ones that come
back above the Jaccard threshold are removed from the evaluation splits. Train
itself is never touched -- the point is an honest measurement, not a smaller
training set.

Both sides are indexed separately. A dev/test pair is dropped if *either* its
Macedonian or its Albanian side is a near-duplicate of the corresponding side of
any train pair, because either one alone is enough to leak the answer.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

WHITESPACE = None  # str.split() default


def normalise(text: str) -> str:
    """NFKC, casefolded, whitespace-collapsed. Shared by both sides."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text: str, k: int) -> list[bytes]:
    """Character k-grams of the normalised text, as the bytes MinHash wants.

    A text shorter than k yields itself as a single shingle rather than nothing,
    so short segments are still comparable instead of silently matching nothing.
    """
    clean = normalise(text)
    if not clean:
        return []
    if len(clean) <= k:
        return [clean.encode("utf-8")]
    return [clean[i : i + k].encode("utf-8") for i in range(len(clean) - k + 1)]


def _minhash(text: str, k: int, num_perm: int) -> MinHash | None:
    grams = shingles(text, k)
    if not grams:
        return None
    m = MinHash(num_perm=num_perm)
    m.update_batch(grams)
    return m


@dataclass
class SweepResult:
    """Per-side outcome of one cross-split sweep."""

    side: str
    threshold: float
    num_perm: int
    ngram: int
    train_indexed: int
    queried: int
    matched: int
    max_jaccard: pd.Series = field(repr=False, default_factory=pd.Series)
    match_key: pd.Series = field(repr=False, default_factory=pd.Series)


def sweep_side(
    train_texts: pd.Series,
    eval_texts: pd.Series,
    *,
    side: str,
    threshold: float = 0.85,
    num_perm: int = 128,
    ngram: int = 5,
    progress: bool = True,
) -> SweepResult:
    """Index `train_texts`, query `eval_texts`, return the per-row best match.

    `train_texts` and `eval_texts` are indexed by a stable row key (the pair's
    ``sq_component_id``); the returned Series carry the same index as
    `eval_texts`.
    """
    from tqdm import tqdm

    def bar(iterable, desc, total):
        return tqdm(iterable, desc=desc, total=total, disable=not progress, unit="seg")

    train_hashes: dict[str, MinHash] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    with lsh.insertion_session() as session:
        for key, text in bar(train_texts.items(), f"{side}: index train", len(train_texts)):
            m = _minhash(text, ngram, num_perm)
            if m is None:
                continue
            train_hashes[key] = m
            session.insert(key, m)

    best_jaccard = np.zeros(len(eval_texts), dtype="float64")
    best_key: list[str | None] = [None] * len(eval_texts)
    matched = 0
    for position, (_, text) in enumerate(bar(eval_texts.items(), f"{side}: query eval", len(eval_texts))):
        m = _minhash(text, ngram, num_perm)
        if m is None:
            continue
        candidates = lsh.query(m)
        if not candidates:
            continue
        # LSH is a candidate filter; the estimate below is what decides.
        scores = [(m.jaccard(train_hashes[c]), c) for c in candidates]
        score, key = max(scores)
        best_jaccard[position] = score
        best_key[position] = key
        if score >= threshold:
            matched += 1

    return SweepResult(
        side=side,
        threshold=threshold,
        num_perm=num_perm,
        ngram=ngram,
        train_indexed=len(train_hashes),
        queried=len(eval_texts),
        matched=matched,
        max_jaccard=pd.Series(best_jaccard, index=eval_texts.index),
        match_key=pd.Series(best_key, index=eval_texts.index, dtype="object"),
    )


def content_fingerprint(mk: str, sq: str) -> str:
    """A stable 16-hex digest of the normalised pair, for provenance logging."""
    payload = (normalise(mk) + "\x1f" + normalise(sq)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]
