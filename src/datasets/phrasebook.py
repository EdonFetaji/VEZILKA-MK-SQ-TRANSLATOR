"""Goethe Verlag book2 conversational phrasebook (part of W_GEN, research only)."""

from __future__ import annotations

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("phrasebook")
class PhrasebookLoader(DatasetLoader):
    """Local ``mksq_phrasebook/`` (1,779 unique phrase pairs). All rights reserved: research only.

    Pairs with grammatical-person mismatches are dropped. Phrases are full utterances, so they are
    trained with the native ``translation`` task.
    """

    kind = RecordKind.PARALLEL
    splits = ("train",)

    def load(self, split: str):
        raise NotImplementedError("PhrasebookLoader.load")
