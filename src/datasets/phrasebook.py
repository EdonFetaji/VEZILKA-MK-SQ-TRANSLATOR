"""Goethe Verlag book2 conversational phrasebook (part of W_GEN, research only)."""

from __future__ import annotations

import json

from src.datasets.base import DatasetLoader, RecordKind, parquet_batches
from src.pipeline.registry import register_loader


@register_loader("phrasebook")
class PhrasebookLoader(DatasetLoader):
    """Private HF repo ``repo`` holding ``phrasebook_mk_sq_unique.parquet`` (1,779 unique phrase
    pairs; needs ``mk, sq, pair_ids, lesson_numbers``). Option ``path`` reads a local copy instead.
    All rights reserved: research only.

    ``doc_id`` is the first lesson the phrase appears in. Pairs with grammatical-person mismatches
    should be dropped, but the file has no flag for them yet: list their ``pair_ids`` in option
    ``drop_pair_ids`` (TODO at TG-D0). Phrases are full utterances, so they are trained with the
    native ``translation`` task.
    """

    kind = RecordKind.PARALLEL
    restricted = True  # research only: private HF repo
    splits = ("train",)
    repo = "EdonFetaji/mksq-phrasebook"  # private; TODO confirm name
    revision = None  # TODO pin once uploaded

    def _read(self, split: str):
        drop = set(self.options.get("drop_pair_ids", ()))
        for c in parquet_batches(self.source_files(), columns=["mk", "sq", "pair_ids", "lesson_numbers"]):
            rows = [
                r for r in zip(c["mk"], c["sq"], c["pair_ids"], c["lesson_numbers"]) if not drop & set(r[2])
            ]
            yield {
                "mk": [r[0] for r in rows],
                "sq": [r[1] for r in rows],
                "source_id": ["phrasebook"] * len(rows),
                "doc_id": [f"lesson:{min(r[3])}" for r in rows],
                "meta": [json.dumps({"pair_ids": r[2]}, ensure_ascii=False) for r in rows],
            }
