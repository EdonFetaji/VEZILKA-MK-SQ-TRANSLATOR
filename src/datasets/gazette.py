"""Official Gazette (Службен весник ↔ Gazeta Zyrtare) parallel corpus."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind, hub_files, parquet_batches
from src.pipeline.registry import register_loader

#: Repository columns carried into ``meta``.
META_COLUMNS = (
    "id",
    "year",
    "issue_no",
    "segment_type",
    "is_high_quality",
    "redacted",
    "cometkiwi22",
    "metricx24_qe",
    "sonar_cosine",
)


@register_loader("gazette")
class GazetteLoader(DatasetLoader):
    """``EdonFetaji/slvesnik-mk-sq-clean``, with issue-disjoint ``train``/``validation``/``test`` splits.

    ``doc_id`` is the Gazette issue (its ``source_pdf``, the unit the repository splits on).
    Selection rule ``slice`` names one part of the training pool, split by whole issues and
    stratified by year at TG-D0:

    * ``L_MAIN`` (68%, TG-20 and TG-21), ``L_ARMS`` (16%, every TG-30 arm) and
      ``L_FRESH`` (16%, TG-31 only; ``max_rows`` caps it at the matched 28,161 rows)
    * ``novel_only: true`` on ``test`` gives GAZ_TEST_NOVEL (templates and near-duplicates removed)

    Training pairs that repeat a dev or test sentence (GAZ_REMOVED) are never returned.
    """

    kind = RecordKind.PARALLEL
    splits = ("train", "validation", "test")
    select_rules: ClassVar[tuple[str, ...]] = ("slice", "max_rows", "novel_only")
    repo = "EdonFetaji/slvesnik-mk-sq-clean"
    revision = "e9ed9fb67e6b5cc015354d64987d57bc790cd683"

    def _read(self, split: str):
        paths, sha = hub_files(self.opt("repo"), self.opt("revision"), prefix=f"data/{split}-")
        self._source = {"repo": self.opt("repo"), "revision": sha}
        for c in parquet_batches(paths, columns=["mk", "sq", "source_pdf", *META_COLUMNS]):
            meta = zip(*(c[k] for k in META_COLUMNS))
            yield {
                "mk": c["mk"],
                "sq": c["sq"],
                "source_id": ["gazette"] * len(c["mk"]),
                "doc_id": c["source_pdf"],
                "meta": [json.dumps(dict(zip(META_COLUMNS, m)), ensure_ascii=False) for m in meta],
            }

    def select(self, ds, **rules: Any):
        raise NotImplementedError("GazetteLoader.select")
