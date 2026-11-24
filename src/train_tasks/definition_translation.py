"""Definition → definition in the other language (V_DEF_PAR, TG-34)."""

from __future__ import annotations

from typing import Any

from src.datasets.base import RecordKind
from src.pipeline.registry import register_task
from src.train_tasks.base import Direction, TrainTask


@register_task("definition_translation")
class DefinitionTranslationTask(TrainTask):
    """Input: ``mk_def`` (or ``sq_def``) in the native translation prompt. Output: the other definition.

    Only pairs with LaBSE ≥ 0.70 (``select: {def_labse_floor: 0.70}`` on the loader). The token
    budget is matched to TG-33's word task (≈1.7 passes), set with ``tokens_m`` in the config.
    """

    accepts = RecordKind.LEXICAL

    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        raise NotImplementedError("DefinitionTranslationTask.format")
