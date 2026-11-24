"""Term + source-language definition → target term (V_DEF_CTX, TG-33, the central test)."""

from __future__ import annotations

from typing import Any

from src.datasets.base import RecordKind
from src.pipeline.registry import register_task
from src.train_tasks.base import Direction, TrainTask


@register_task("term_definition")
class TermDefinitionTask(TrainTask):
    """Input: source term plus its definition in the *source* language. Output: the target term only.

    Everything except the definition must match ``term_pair``: same instruction wording, entries
    and repeats. That is the fairness condition the TG-33 vs TG-32 comparison rests on.
    """

    accepts = RecordKind.LEXICAL

    def definition_for(self, record: dict[str, Any], direction: Direction) -> str:
        """The definition shown in the prompt. ``shuffled_definition`` overrides this."""
        src = direction.split("-")[0]
        return record[f"{src}_def"]

    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        raise NotImplementedError("TermDefinitionTask.format")
