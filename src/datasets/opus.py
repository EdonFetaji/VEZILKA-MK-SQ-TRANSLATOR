"""General-domain OPUS MK–SQ pairs (dataset tags G, G_REPLAY, G_REST)."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("opus")
class OpusLoader(DatasetLoader):
    """``EdonFetaji/opus-mk-sq-clean-dedup``, default config. Rows are selected by ``source_id``.

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

    def load(self, split: str):
        raise NotImplementedError("OpusLoader.load")

    def select(self, ds, **rules: Any):
        raise NotImplementedError("OpusLoader.select")
