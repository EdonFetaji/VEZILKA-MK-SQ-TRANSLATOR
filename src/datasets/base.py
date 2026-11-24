"""The dataset loader interface.

A loader turns one raw source (a Hugging Face repo, a private repo, a local folder) into a
``datasets.Dataset`` with a fixed set of columns, so tasks never see source-specific fields.

There are two record kinds:

* ``PARALLEL`` – sentence pairs. Columns: ``mk, sq, source_id, doc_id, meta``.
  ``doc_id`` is the unit for issue-disjoint splits and document-level bootstrap (a Gazette issue,
  an OPUS document); ``meta`` is a JSON string for anything else (LaBSE score, english_mediated, …).
* ``LEXICAL`` – dictionary entries. Columns:
  ``entry_id, concept_id, domain, mk_term, sq_term, mk_def, sq_def, source_id``.
  ``concept_id`` groups senses and both directions, so V_HOLDOUT can be sampled by concept.
  Definitions may be empty (MUSE has none).

A run config names a loader, a split and optional selection rules::

    - {loader: opus, split: train, select: {source_ids: [SETIMES, TED2020]}, task: translation}

Loader options (repo, revision, local path) can be set per entry under ``options:``; defaults live
on the implementing class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import datasets


class RecordKind(str, Enum):
    PARALLEL = "parallel"
    LEXICAL = "lexical"


PARALLEL_COLUMNS = ("mk", "sq", "source_id", "doc_id", "meta")
LEXICAL_COLUMNS = (
    "entry_id",
    "concept_id",
    "domain",
    "mk_term",
    "sq_term",
    "mk_def",
    "sq_def",
    "source_id",
)
COLUMNS = {RecordKind.PARALLEL: PARALLEL_COLUMNS, RecordKind.LEXICAL: LEXICAL_COLUMNS}


class DatasetLoader(ABC):
    """Base class for every dataset loader.

    Subclasses set ``kind`` and ``splits`` and are registered with
    ``@register_loader("<name>")`` (which sets ``name``).
    """

    name: ClassVar[str]
    kind: ClassVar[RecordKind]
    splits: ClassVar[tuple[str, ...]] = ("train",)
    #: Selection rules this loader understands (keys of ``select:`` in a config).
    select_rules: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def load(self, split: str) -> datasets.Dataset:
        """Return ``split`` with exactly the columns of ``COLUMNS[self.kind]``."""

    def select(self, ds: datasets.Dataset, **rules: Any) -> datasets.Dataset:
        """Apply the config's ``select:`` rules (a dataset tag such as L_ARMS or G).

        The default accepts no rules. Subclasses that override it should list their rule names in
        ``select_rules`` so config validation can reject typos early.
        """
        if rules:
            raise ValueError(f"loader {self.name!r} accepts no selection rules, got {sorted(rules)}")
        return ds

    def describe(self) -> dict[str, Any]:
        """Provenance for the run manifest: repo, pinned revision, row counts, file hashes."""
        return {"loader": self.name, "kind": self.kind.value, "options": self.options}

    def validate(self, ds: datasets.Dataset) -> datasets.Dataset:
        """Check that ``ds`` has the required columns for this loader's kind."""
        missing = set(COLUMNS[self.kind]) - set(ds.column_names)
        if missing:
            raise ValueError(f"loader {self.name!r} returned a dataset missing {sorted(missing)}")
        return ds
