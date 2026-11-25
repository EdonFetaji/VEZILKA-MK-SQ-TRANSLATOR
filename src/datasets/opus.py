"""General-domain OPUS MK–SQ pairs (dataset tags G, G_REPLAY, G_REST)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind, hub_files, parquet_batches
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

    def select(self, ds, **rules: Any):
        raise NotImplementedError("OpusLoader.select")
