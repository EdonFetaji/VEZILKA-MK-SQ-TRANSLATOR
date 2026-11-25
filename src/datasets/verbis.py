"""VERBIS termbase: term + definition in both languages, four domains."""

from __future__ import annotations

from typing import Any, ClassVar

from src.datasets.base import DatasetLoader, RecordKind, concept_key, parquet_batches
from src.pipeline.registry import register_loader

#: VERBIS API ``category_id`` → ``domain`` (3 is "Shëndetësore", health).
DOMAINS = {1: "legal_administrative", 2: "finance_economic", 3: "health", 4: "general"}


@register_loader("verbis")
class VerbisLoader(DatasetLoader):
    """VERBIS v2 aligned export (15,925 entries). Redistribution restricted.

    Private HF repo ``repo`` (the v2 aligned export, ``verbis_mk_sq_v2_aligned.parquet``: needs
    ``verbis_id, category_id, mk, sq, mk_description, sq_description``). Option ``path`` reads a
    local copy instead, for development.

    Domains: legal-administrative, finance-economic, health, general (VERBIS ``category_id`` 1–4).
    ``entry_id`` is ``verbis:<verbis_id>``; ``concept_id`` is the casefolded Macedonian headword,
    so every sense of a term (in any domain) is held out together.
    Records arrive with headwords separated from notation and definitions that contain the target
    term masked.

    Selection rules:

    * ``exclude: V_HOLDOUT`` drops the 10% hold-out concepts (every training task uses this)
    * ``only: V_HOLDOUT`` returns just the hold-out concepts (TERM_HOLDOUT scoring)
    * ``domains``: list of domains to keep (RQ4)
    * ``def_labse_floor``: minimum LaBSE between the two definitions (0.70 for V_DEF_PAR)

    Never write VERBIS rows into the repository or into logs.
    """

    kind = RecordKind.LEXICAL
    restricted = True  # research only: private HF repo
    splits = ("train",)
    select_rules: ClassVar[tuple[str, ...]] = ("exclude", "only", "domains", "def_labse_floor")
    repo = "EdonFetaji/verbis-contextual-dictionary"  # private; TODO confirm name
    revision = None  # TODO pin once uploaded

    def _read(self, split: str):
        cols = ["verbis_id", "category_id", "mk", "sq", "mk_description", "sq_description"]
        for c in parquet_batches(self.source_files(), columns=cols):
            yield {
                "entry_id": [f"verbis:{i}" for i in c["verbis_id"]],
                "concept_id": [f"verbis:{concept_key(t)}" for t in c["mk"]],
                "domain": [DOMAINS[k] for k in c["category_id"]],
                "mk_term": c["mk"],
                "sq_term": c["sq"],
                "mk_def": [d or "" for d in c["mk_description"]],
                "sq_def": [d or "" for d in c["sq_description"]],
                "source_id": ["verbis"] * len(c["mk"]),
            }

    def select(self, ds, **rules: Any):
        raise NotImplementedError("VerbisLoader.select")
