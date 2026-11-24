"""Term + someone else's definition → target term (V_DEF_SHUF, TG-35 control)."""

from __future__ import annotations

from typing import Any

from src.pipeline.registry import register_task
from src.train_tasks.base import Direction
from src.train_tasks.term_definition import TermDefinitionTask


@register_task("shuffled_definition")
class ShuffledDefinitionTask(TermDefinitionTask):
    """Same as ``term_definition``, but each entry gets the definition of another entry of similar
    length in the same domain (fixed permutation, seed 42). Tests whether the definition's meaning
    mattered or only the extra input tokens.

    The permutation is built once over the whole lexicon (``build_permutation``), before formatting.
    """

    def build_permutation(self, dataset: Any, seed: int = 42) -> dict[str, str]:
        """Map ``entry_id`` → donor ``entry_id``: same domain, similar definition length, never itself."""
        raise NotImplementedError("ShuffledDefinitionTask.build_permutation")

    def definition_for(self, record: dict[str, Any], direction: Direction) -> str:
        raise NotImplementedError("ShuffledDefinitionTask.definition_for")
