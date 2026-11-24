"""Sentence translation with TranslateGemma's native prompt."""

from __future__ import annotations

from typing import Any

from src.datasets.base import RecordKind
from src.pipeline.registry import register_task
from src.train_tasks.base import Direction, TrainTask


@register_task("translation")
class TranslationTask(TrainTask):
    """Input: source sentence. Output: its translation.

    Uses TranslateGemma's own chat-template message (≈80 prompt tokens), so training matches how
    the model is prompted at test time::

        [{"role": "user", "content": [{"type": "text",
                                       "source_lang_code": "mk", "target_lang_code": "sq",
                                       "text": record["mk"]}]}]

    Completion: ``record["sq"]`` (or the reverse for ``sq-mk``).
    Used for G, G_REPLAY, L_MAIN, L_ARMS, L_FRESH, phrasebook rows and every evaluation set.
    """

    accepts = RecordKind.PARALLEL

    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        raise NotImplementedError("TranslationTask.format")
