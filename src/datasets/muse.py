"""MUSE MK–SQ word pairs built through English (part of W_GEN)."""

from __future__ import annotations

from src.datasets.base import DatasetLoader, RecordKind, hub_files, parquet_batches
from src.pipeline.registry import register_loader


def keep_pair(mk: str, sq: str) -> bool:
    """Cleaning rule: drop identical strings, anything with a digit, and names (both sides capitalised)."""
    if mk.casefold() == sq.casefold():
        return False
    if any(ch.isdigit() for ch in mk + sq):
        return False
    return not (mk[:1].isupper() and sq[:1].isupper())


@register_loader("muse")
class MuseLoader(DatasetLoader):
    """``EdonFetaji/mk-sq-dictionary-muse`` (7,388 pairs, CC BY-NC 4.0). Has no definitions.

    Cleaning (``keep_pair``): drop identical strings, numbers and names. ``domain`` is ``general``;
    ``concept_id`` is the English pivot word. Used by TG-36, and by TG-90 only if TG-36 beat TG-32.
    """

    kind = RecordKind.LEXICAL
    splits = ("train",)
    repo = "EdonFetaji/mk-sq-dictionary-muse"
    revision = "7980380d28b3b5fcbb0e67f46666f45d47653511"

    def _read(self, split: str):
        paths, sha = hub_files(self.opt("repo"), self.opt("revision"))
        self._source = {"repo": self.opt("repo"), "revision": sha}
        row = 0
        for c in parquet_batches(paths, columns=["mk", "sq", "en"]):
            rows = []
            for mk, sq, en in zip(c["mk"], c["sq"], c["en"]):
                if keep_pair(mk, sq):
                    rows.append((f"muse:{row}", f"muse:{en.casefold()}", mk, sq))
                row += 1
            yield {
                "entry_id": [r[0] for r in rows],
                "concept_id": [r[1] for r in rows],
                "domain": ["general"] * len(rows),
                "mk_term": [r[2] for r in rows],
                "sq_term": [r[3] for r in rows],
                "mk_def": [""] * len(rows),
                "sq_def": [""] * len(rows),
                "source_id": ["muse"] * len(rows),
            }
