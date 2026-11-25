"""NTREX-128 Macedonian and Albanian references (evaluation only)."""

from __future__ import annotations

import json
import urllib.request

from src.datasets.base import DatasetLoader, RecordKind
from src.pipeline.registry import register_loader

URL = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{commit}/NTREX-128/newstest2019-ref.{code}.txt"


def _lines(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8-sig")
    # split on "\n" only: str.splitlines() also splits on U+2028 etc. and would misalign the sides
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


@register_loader("ntrex")
class NtrexLoader(DatasetLoader):
    """NTREX-128 at a pinned revision (1,997 sentences). Kept for continuity with the NLLB pilot.
    Never trained on.

    Line *i* of the Macedonian reference is aligned with line *i* of the Albanian one. The files
    carry no document ids, so ``doc_id`` is the line number.
    """

    kind = RecordKind.PARALLEL
    splits = ("test",)
    revision = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"

    def _read(self, split: str):
        commit = self.opt("revision")
        self._source = {"repo": "github.com/MicrosoftTranslator/NTREX", "revision": commit}
        mk, sq = _lines(URL.format(commit=commit, code="mkd")), _lines(URL.format(commit=commit, code="sqi"))
        if len(mk) != len(sq):
            raise ValueError(f"NTREX-128 sides differ: mkd={len(mk)} lines, sqi={len(sq)} lines")
        yield {
            "mk": mk,
            "sq": sq,
            "source_id": ["ntrex"] * len(mk),
            "doc_id": [str(i) for i in range(len(mk))],
            "meta": [json.dumps({"line": i}) for i in range(len(mk))],
        }
