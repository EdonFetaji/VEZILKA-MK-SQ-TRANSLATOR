"""MUSE MK–SQ word pairs built through English (part of W_GEN)."""

from __future__ import annotations

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("muse")
class MuseLoader(DatasetLoader):
    """``EdonFetaji/mk-sq-dictionary-muse`` (7,388 pairs, CC BY-NC 4.0). Has no definitions.

    Cleaning: drop identical strings, numbers and names. ``domain`` is ``general``.
    Used by TG-36, and by TG-90 only if TG-36 beat TG-32.
    """

    kind = RecordKind.LEXICAL
    splits = ("train",)
    repo = "EdonFetaji/mk-sq-dictionary-muse"

    def load(self, split: str):
        raise NotImplementedError("MuseLoader.load")
