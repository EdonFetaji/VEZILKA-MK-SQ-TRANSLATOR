"""FLORES+ mkd_Cyrl–als_Latn (evaluation only)."""

from __future__ import annotations

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("flores")
class FloresLoader(DatasetLoader):
    """FLORES+ at a pinned revision. ``dev`` (997, FLORES_DEV, selection) and ``devtest``
    (1,012, FLORES_DEVTEST, the primary general-quality test). Never trained on.
    """

    kind = RecordKind.PARALLEL
    splits = ("dev", "devtest")
    repo = "openlanguagedata/flores_plus"

    def load(self, split: str):
        raise NotImplementedError("FloresLoader.load")
