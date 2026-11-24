"""The training-task interface.

A task turns one loader record into instruction-tuning examples in TRL's prompt–completion format::

    {"prompt": [{"role": "user", "content": ...}], "completion": "<target text>"}

The trainer computes loss on the completion only ("loss on target tokens only", plan §05).

Directions
----------
``mk-sq`` and ``sq-mk`` give one example per record; ``both`` gives one per direction;
``alternating`` gives one per record, flipping direction by row index (mined and subtitle rows in G).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from src.datasets.base import RecordKind

Direction = Literal["mk-sq", "sq-mk"]
Directions = Literal["mk-sq", "sq-mk", "both", "alternating"]
DIRECTIONS: tuple[str, ...] = ("mk-sq", "sq-mk", "both", "alternating")

#: Language codes as TranslateGemma's chat template expects them.
LANG_CODE = {"mk": "mk", "sq": "sq"}


def split_direction(direction: Direction) -> tuple[str, str]:
    src, tgt = direction.split("-")
    return src, tgt


class TrainTask(ABC):
    """Base class for every training task. Register with ``@register_task("<name>")``."""

    name: ClassVar[str]
    accepts: ClassVar[RecordKind]

    def __init__(self, tokenizer: Any = None, **options: Any) -> None:
        self.tokenizer = tokenizer
        self.options = options

    @abstractmethod
    def format(self, record: dict[str, Any], direction: Direction) -> dict[str, Any]:
        """One prompt–completion example for ``record`` in ``direction``."""

    def directions_for(self, directions: Directions, index: int) -> list[Direction]:
        if directions == "both":
            return ["mk-sq", "sq-mk"]
        if directions == "alternating":
            return ["mk-sq" if index % 2 == 0 else "sq-mk"]
        if directions in ("mk-sq", "sq-mk"):
            return [directions]
        raise ValueError(f"unknown directions {directions!r}; expected one of {DIRECTIONS}")

    def examples(
        self, record: dict[str, Any], directions: Directions, index: int = 0
    ) -> list[dict[str, Any]]:
        return [self.format(record, d) for d in self.directions_for(directions, index)]
