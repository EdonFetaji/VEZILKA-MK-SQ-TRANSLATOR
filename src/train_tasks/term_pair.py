"""Bare term pairs (V_TERM, TG-32; also W_GEN in TG-36)."""

from __future__ import annotations

from typing import Any

from src.datasets.base import RecordKind
from src.pipeline.registry import register_task
from src.train_tasks.base import Direction, TrainTask


@register_task("term_pair")
class TermPairTask(TrainTask):
    """Input: source term (with a short separate instruction, not the sentence prompt).
    Output: target term.

    Example (mk-sq): ``застареност`` → ``parashkrim``.
    Each entry is shown ``repeats`` times per direction (3 in the plan), set in the config.
    """

    accepts = RecordKind.LEXICAL

    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        raise NotImplementedError("TermPairTask.format")
