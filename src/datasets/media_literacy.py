"""Media Literacy Dictionary: term + definition in both languages (the fifth domain)."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("media_literacy")
class MediaLiteracyLoader(DatasetLoader):
    """Local ``media-literacy-vocabulary`` (1,081 valid linked entries; 2k rows as stored).

    No licence found, so it is treated as copyrighted and kept research-only (never in TG-90's
    public mix). Selection rules are the same as for VERBIS; ``domain`` is ``media_literacy``.
    """

    kind = RecordKind.LEXICAL
    splits = ("train",)
    select_rules: ClassVar[tuple[str, ...]] = ("exclude", "only", "def_labse_floor")

    def load(self, split: str):
        raise NotImplementedError("MediaLiteracyLoader.load")

    def select(self, ds, **rules: Any):
        raise NotImplementedError("MediaLiteracyLoader.select")
