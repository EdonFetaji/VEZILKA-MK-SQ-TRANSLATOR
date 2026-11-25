"""Media Literacy Dictionary: term + definition in both languages (the fifth domain)."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind, concept_key, parquet_batches
from src.pipeline.registry import register_loader


@register_loader("media_literacy")
class MediaLiteracyLoader(DatasetLoader):
    """Private HF repo ``repo`` holding ``media_literacy.parquet`` (1,081 valid linked entries of
    1,084 stored; needs ``record_id, mk, sq, mk_description, sq_description, quality_flags``).
    Option ``path`` reads a local copy instead.

    The 3 self-referencing links (``quality_flags`` contains ``mk_link_self_reference``) are
    dropped. ``concept_id`` is the casefolded Macedonian term, as for VERBIS.

    No licence found, so it is treated as copyrighted and kept research-only (never in TG-90's
    public mix). Selection rules are the same as for VERBIS; ``domain`` is ``media_literacy``.
    """

    kind = RecordKind.LEXICAL
    restricted = True  # research only: private HF repo
    splits = ("train",)
    select_rules: ClassVar[tuple[str, ...]] = ("exclude", "only", "def_labse_floor")
    repo = "EdonFetaji/media-literacy-vocabulary"  # private; TODO confirm name
    revision = None  # TODO pin once uploaded

    def _read(self, split: str):
        cols = ["record_id", "mk", "sq", "mk_description", "sq_description", "quality_flags"]
        for c in parquet_batches(self.source_files(), columns=cols):
            keep = [i for i, f in enumerate(c["quality_flags"]) if "mk_link_self_reference" not in (f or "")]
            col = {k: [c[k][i] for i in keep] for k in cols}
            yield {
                "entry_id": [f"media_literacy:{r}" for r in col["record_id"]],
                "concept_id": [f"media_literacy:{concept_key(t)}" for t in col["mk"]],
                "domain": ["media_literacy"] * len(keep),
                "mk_term": col["mk"],
                "sq_term": col["sq"],
                "mk_def": [d or "" for d in col["mk_description"]],
                "sq_def": [d or "" for d in col["sq_description"]],
                "source_id": ["media_literacy"] * len(keep),
            }

    def select(self, ds, **rules: Any):
        raise NotImplementedError("MediaLiteracyLoader.select")
