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

Disk, then stream
-----------------
Each split is written **once** to ``<cache>/<loader>/<revision>/<split>.arrow`` (Arrow IPC stream
format, uncompressed) plus a ``<split>.json`` sidecar with the row count, file hash and resolved
source revision. The cache root is ``$MKSQ_DATA_CACHE`` or ``data/cache``.

* ``download(split)`` builds the file chunk by chunk (``_read`` yields column dicts), so the raw
  source is never held in RAM whole. The write is atomic: a crash leaves no half-written file.
* ``stream_batch_from_disk(split, batch_size)`` reads that file sequentially, one stored chunk
  (``WRITE_CHUNK`` rows) at a time, and yields one batch of records at a time. RAM holds the current
  chunk and batch only: past chunks are released once yielded, and a batch's Python objects are
  freed as soon as the caller drops it. (Not memory-mapped on purpose: mapped pages stay resident
  and count towards a container's memory limit.)
* ``load(split)`` returns the same file as a memory-mapped ``datasets.Dataset`` (no copy), for
  ``select`` / ``map`` / TRL.

Shuffling is not done while streaming: the order on disk is the order you get. Shuffle once when
the frozen tag is built (TG-D0), write it, then stream it sequentially, which is the fastest read.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

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
#: Every standard column is a string (``meta`` is a JSON string; empty definitions are "").
SCHEMAS = {kind: pa.schema([(c, pa.string()) for c in cols]) for kind, cols in COLUMNS.items()}

CACHE_ROOT = Path(os.environ.get("MKSQ_DATA_CACHE", "data/cache"))
DEFAULT_BATCH_SIZE = 1024
#: Rows per stored chunk: the unit ``stream_batch_from_disk`` reads, so it bounds streaming RAM.
WRITE_CHUNK = 16_384


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hub_files(
    repo: str, revision: str | None, prefix: str = "", suffix: str = ".parquet"
) -> tuple[list[str], str]:
    """Download the files of a HF dataset repo under ``prefix`` ending in ``suffix``.

    Returns the local paths (sorted by repo path) and the resolved commit sha. Uses the HF cache,
    so a second call is free. Gated/private repos need ``HF_TOKEN`` or ``hf auth login``.
    """
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(repo, revision=revision)
    names = sorted(
        s.rfilename for s in info.siblings if s.rfilename.startswith(prefix) and s.rfilename.endswith(suffix)
    )
    if not names:
        raise FileNotFoundError(f"{repo}@{info.sha}: no files matching {prefix}*{suffix}")
    paths = [hf_hub_download(repo, n, repo_type="dataset", revision=info.sha) for n in names]
    return paths, info.sha


REPO_ROOT = Path(__file__).resolve().parents[2]


def local_file(path: str | Path) -> Path:
    """Resolve a local source file; relative paths are taken from the repository root."""
    p = Path(path).expanduser()
    p = p if p.is_absolute() else REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; set the loader's `path` option")
    return p


def concept_key(term: str) -> str:
    """Senses of one headword share a concept: casefolded, whitespace-normalised term."""
    return " ".join(term.casefold().split())


def parquet_batches(paths: Iterable[str | Path], columns: list[str] | None = None, batch_size: int = 65_536):
    """Yield ``{column: list}`` chunks from parquet files without reading any file whole."""
    import pyarrow.parquet as pq

    for p in paths:
        for rb in pq.ParquetFile(p).iter_batches(batch_size=batch_size, columns=columns):
            yield rb.to_pydict()


class DatasetLoader(ABC):
    """Base class for every dataset loader.

    Subclasses set ``kind`` and ``splits``, implement ``_read`` and are registered with
    ``@register_loader("<name>")`` (which sets ``name``).
    """

    name: ClassVar[str]
    kind: ClassVar[RecordKind]
    splits: ClassVar[tuple[str, ...]] = ("train",)
    #: Selection rules this loader understands (keys of ``select:`` in a config).
    select_rules: ClassVar[tuple[str, ...]] = ()
    #: Pinned source revision (HF commit / git commit). ``None``: resolved at download, recorded.
    revision: ClassVar[str | None] = None
    #: Licence-restricted source (no redistribution). Any run that trains on it, or descends from
    #: such a run, is stored only in the private HF repo (``src.pipeline.hub``).
    restricted: ClassVar[bool] = False

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._source: dict[str, Any] = {}

    # ── to implement ──────────────────────────────────────────────────────────

    @abstractmethod
    def _read(self, split: str) -> Iterator[dict[str, list]]:
        """Yield ``split`` as chunks of ``{column: list}`` with exactly ``COLUMNS[self.kind]``.

        Read the raw source in pieces (``parquet_batches``) rather than whole. Record where the
        data came from in ``self._source`` (``repo``, ``revision``, ``path``, ``sha256``, ...).
        """

    # ── disk ──────────────────────────────────────────────────────────────────

    def opt(self, key: str, default: Any = None) -> Any:
        """A loader option from the config, falling back to the class attribute, then ``default``."""
        return self.options.get(key, getattr(self, key, default))

    def source_files(self) -> list[str]:
        """The source parquet files: option ``path`` (a local file, for development) if set, else
        every ``.parquet`` under option/attribute ``prefix`` in HF repo ``repo`` at ``revision``.
        Records the provenance in ``self._source``.
        """
        if "path" in self.options:
            src = local_file(self.options["path"])
            self._source = {"path": str(src), "source_sha256": sha256(src)}
            return [str(src)]
        paths, sha = hub_files(self.opt("repo"), self.opt("revision"), prefix=self.opt("prefix", ""))
        self._source = {"repo": self.opt("repo"), "revision": sha, "files": [Path(p).name for p in paths]}
        return paths

    def path(self, split: str) -> Path:
        self._check_split(split)
        root = Path(self.options.get("cache_dir", CACHE_ROOT))
        version = "local" if "path" in self.options else (self.opt("revision") or "main")
        return root / self.name / version / f"{split}.arrow"

    def download(self, split: str, force: bool = False) -> Path:
        """Write ``split`` to disk once and return its path. Cached: later calls only check it exists."""
        path = self.path(split)
        if path.exists() and not force:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".arrow.tmp")
        schema, cols, rows = SCHEMAS[self.kind], COLUMNS[self.kind], 0
        with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
            for chunk in self._read(split):
                missing = set(cols) - set(chunk)
                if missing:
                    raise ValueError(f"loader {self.name!r} chunk is missing {sorted(missing)}")
                batch = pa.record_batch([pa.array(chunk[c], pa.string()) for c in cols], schema=schema)
                if batch.num_rows:
                    writer.write_table(pa.Table.from_batches([batch]), max_chunksize=WRITE_CHUNK)
                    rows += batch.num_rows
        os.replace(tmp, path)
        sidecar = {"loader": self.name, "split": split, "rows": rows, "sha256": sha256(path), **self._source}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n")
        return path

    def stream_batch_from_disk(
        self, split: str, batch_size: int = DEFAULT_BATCH_SIZE, start: int = 0, stop: int | None = None
    ) -> Iterator[list[dict[str, str]]]:
        """Yield ``split`` from disk as lists of ``batch_size`` records (dicts of the standard columns).

        The file is read sequentially, one stored chunk (≤ ``WRITE_CHUNK`` rows) at a time, so RAM
        holds at most the current chunk plus the batch being built; earlier chunks are released as
        soon as their rows have been yielded. ``start`` / ``stop`` are row offsets, for resuming
        after a checkpoint or reading one slice. Downloads the split first if it isn't on disk yet.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        buf: pa.Table | None = None  # rows read but not yet yielded (< batch_size + one chunk)
        pos = 0  # row offset of the next stored chunk
        with pa.OSFile(str(self.download(split))) as source:
            for chunk in pa.ipc.open_stream(source):
                n, pos = chunk.num_rows, pos + chunk.num_rows
                lo, hi = max(start - (pos - n), 0), n if stop is None else min(n, stop - (pos - n))
                if hi > lo:
                    piece = pa.Table.from_batches([chunk.slice(lo, hi - lo)])
                    buf = piece if buf is None else pa.concat_tables([buf, piece])
                    while buf.num_rows >= batch_size:
                        yield buf.slice(0, batch_size).to_pylist()
                        buf = buf.slice(batch_size)
                if stop is not None and pos >= stop:
                    break
        if buf is not None and buf.num_rows:
            yield buf.to_pylist()

    def num_rows(self, split: str) -> int:
        return json.loads(self.download(split).with_suffix(".json").read_text())["rows"]

    def load(self, split: str) -> datasets.Dataset:
        """Return ``split`` as a memory-mapped ``datasets.Dataset`` with the columns of ``COLUMNS[self.kind]``."""
        import datasets

        return self.validate(datasets.Dataset.from_file(str(self.download(split))))

    def _check_split(self, split: str) -> None:
        if split not in self.splits:
            raise ValueError(f"loader {self.name!r} has no split {split!r}; known: {self.splits}")

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
        out: dict[str, Any] = {
            "loader": self.name,
            "kind": self.kind.value,
            "options": self.options,
            "splits": {},
        }
        for split in self.splits:
            sidecar = self.path(split).with_suffix(".json")
            if sidecar.exists():
                out["splits"][split] = json.loads(sidecar.read_text())
        return out

    def validate(self, ds: datasets.Dataset) -> datasets.Dataset:
        """Check that ``ds`` has the required columns for this loader's kind."""
        missing = set(COLUMNS[self.kind]) - set(ds.column_names)
        if missing:
            raise ValueError(f"loader {self.name!r} returned a dataset missing {sorted(missing)}")
        return ds
