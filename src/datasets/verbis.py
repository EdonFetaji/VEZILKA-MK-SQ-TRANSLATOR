"""VERBIS termbase: term + definition in both languages, four domains."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader


@register_loader("verbis")
class VerbisLoader(DatasetLoader):
    """Private repo ``verbis-contextual-dictionary`` (15,925 entries). Redistribution restricted.

    Domains: legal-administrative, finance-economic, health or judicial (label to confirm), general.
    Records arrive with headwords separated from notation and definitions that contain the target
    term masked.

    Selection rules:

    * ``exclude: V_HOLDOUT`` drops the 10% hold-out concepts (every training task uses this)
    * ``only: V_HOLDOUT`` returns just the hold-out concepts (TERM_HOLDOUT scoring)
    * ``domains``: list of domains to keep (RQ4)
    * ``def_labse_floor``: minimum LaBSE between the two definitions (0.70 for V_DEF_PAR)

    Never write VERBIS rows into the repository or into logs.
    """

    kind = RecordKind.LEXICAL
    splits = ("train",)
    select_rules: ClassVar[tuple[str, ...]] = ("exclude", "only", "domains", "def_labse_floor")

    def load(self, split: str):
        raise NotImplementedError("VerbisLoader.load")

    def select(self, ds, **rules: Any):
        raise NotImplementedError("VerbisLoader.select")
