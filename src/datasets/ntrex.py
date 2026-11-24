"""NTREX-128 Macedonian and Albanian references (evaluation only)."""

from __future__ import annotations

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("ntrex")
class NtrexLoader(DatasetLoader):
    """NTREX-128 at a pinned revision (1,997 sentences). Kept for continuity with the NLLB pilot.
    Never trained on.
    """

    kind = RecordKind.PARALLEL
    splits = ("test",)

    def load(self, split: str):
        raise NotImplementedError("NtrexLoader.load")
