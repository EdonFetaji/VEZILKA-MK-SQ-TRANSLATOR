"""Official Gazette (Службен весник ↔ Gazeta Zyrtare) parallel corpus."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("gazette")
class GazetteLoader(DatasetLoader):
    """``EdonFetaji/slvesnik-mk-sq-clean``, with issue-disjoint ``train``/``validation``/``test`` splits.

    ``doc_id`` is the Gazette issue. Selection rule ``slice`` names one part of the training pool,
    split by whole issues and stratified by year at TG-D0:

    * ``L_MAIN`` (68%, TG-20 and TG-21), ``L_ARMS`` (16%, every TG-30 arm) and
      ``L_FRESH`` (16%, TG-31 only; ``max_rows`` caps it at the matched 28,161 rows)
    * ``novel_only: true`` on ``test`` gives GAZ_TEST_NOVEL (templates and near-duplicates removed)

    Training pairs that repeat a dev or test sentence (GAZ_REMOVED) are never returned.
    """

    kind = RecordKind.PARALLEL
    splits = ("train", "validation", "test")
    select_rules: ClassVar[tuple[str, ...]] = ("slice", "max_rows", "novel_only")
    repo = "EdonFetaji/slvesnik-mk-sq-clean"

    def load(self, split: str):
        raise NotImplementedError("GazetteLoader.load")

    def select(self, ds, **rules: Any):
        raise NotImplementedError("GazetteLoader.select")
