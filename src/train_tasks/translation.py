"""Sentence translation with TranslateGemma's native prompt."""

from __future__ import annotations

from typing import Any

from src.datasets.base import RecordKind
from src.pipeline.registry import register_task
from src.train_tasks.base import LANG_CODE, Direction, TrainTask, split_direction

#: Target-side stop marker: the end of TranslateGemma's model turn.
END_OF_TURN = "<end_of_turn>"


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

    def __init__(self, tokenizer: Any = None, **options: Any) -> None:
        super().__init__(tokenizer, **options)
        self._bos: str | None = None  # leading BOS to strip (the tokenizer adds its own), set lazily

    def messages(self, record: dict[str, Any], direction: Direction) -> list[dict[str, Any]]:
        src, tgt = split_direction(direction)
        return [{"role": "user", "content": [{"type": "text", "source_lang_code": LANG_CODE[src],
                                              "target_lang_code": LANG_CODE[tgt], "text": record[src]}]}]

    def _strip_bos(self) -> str:
        """The template's ``<bos>`` if the tokenizer also adds one when TRL tokenizes the prompt string."""
        if self._bos is None:
            tok = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            bos = getattr(tok, "bos_token", None)
            adds = bool(bos) and tok("x")["input_ids"][:1] == [tok.bos_token_id]
            self._bos = bos if adds else ""
        return self._bos

    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        """Prompt: TranslateGemma's own chat template (``apply_chat_template``, generation prompt on),
        never retyped. Completion: the target sentence + ``<end_of_turn>``."""
        if self.tokenizer is None:
            raise ValueError("TranslationTask needs the model's processor (its chat template)")
        prompt = self.tokenizer.apply_chat_template(
            self.messages(record, direction), tokenize=False, add_generation_prompt=True
        )
        bos = self._strip_bos()
        if bos and prompt.startswith(bos):  # exactly one <bos> after TRL tokenizes the string
            prompt = prompt[len(bos):]
        return {"prompt": prompt, "completion": record[split_direction(direction)[1]] + END_OF_TURN}
