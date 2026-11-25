# Dataset loaders (`src/datasets/`)

Each loader turns one raw source into a split with a fixed set of columns, written **once** to
disk and then **streamed** batch by batch. Tasks (`src/train_tasks/`) only ever see the standard
columns, never source-specific fields. Research context: `research_plan/TranslateGemma_MK-SQ_Experiment_Plan.md`
(§06 Data) and `research_plan/tg_datasets.csv` (every dataset tag, source by source).

## Quick use

```python
from src.pipeline.registry import get_loader

L = get_loader("opus")()                     # options: get_loader("verbis")(path="local.parquet")
L.download("train")                          # once; later calls only check the file exists
for batch in L.stream_batch_from_disk("train", batch_size=1024):   # list[dict] of records
    ...                                      # previous batch is freed when you drop it
L.stream_batch_from_disk("train", 1024, start=600_000, stop=1_000_000)  # resume / slice by row
L.num_rows("train"); L.describe()            # counts + provenance for the run manifest
ds = L.load("train")                         # same file as a datasets.Dataset (no copy)
```

From a run config, loaders are named by their registry name; `options:` go to the constructor:

```yaml
- {loader: opus, split: train, select: {tag: G}, task: translation, options: {revision: <sha>}}
```

## Record kinds and columns

Every column is a **string** (`base.SCHEMAS`). `meta` is a JSON string; missing definitions are `""`.

| Kind | Columns |
| --- | --- |
| `PARALLEL` | `mk, sq, source_id, doc_id, meta` |
| `LEXICAL` | `entry_id, concept_id, domain, mk_term, sq_term, mk_def, sq_def, source_id` |

* `doc_id`: the unit for disjoint splits and document-level bootstrap.
* `concept_id`: groups the senses and both directions of one term, so V_HOLDOUT is sampled by concept.

### One shape for the tasks

Every source names its columns differently; the loaders rename them so tasks never see source
names. The renaming happens **once, when `download` writes the split**, not on every read: the
file on disk is already in the standard shape, so every batch from `stream_batch_from_disk` (and
every row of `load`) has exactly the columns above, and reading does no per-batch reshaping.

| Dataset | Its own columns | Become |
| --- | --- | --- |
| OPUS | `mk`, `sq`, `source_id` = `opus:SETIMES:v2`, `pair_id`, provenance fields | `mk`, `sq`, `source_id` = `SETIMES`, `doc_id` = `pair_id`, provenance in `meta` |
| Gazette | `mk`, `sq`, `source_pdf`, `year`, `issue_no`, quality scores | `mk`, `sq`, `doc_id` = `source_pdf`, the rest in `meta` |
| FLORES+ | two files (`mkd_Cyrl`, `als_Latn`), each with `text` | joined on `id` into `mk`, `sq` |
| NTREX | two text files, one sentence per line | `mk`, `sq` |
| Phrasebook | `mk`, `sq`, `pair_ids`, `lesson_numbers` | `mk`, `sq`, `doc_id` = first lesson |
| VERBIS | `mk`, `sq`, `mk_description`, `sq_description`, `category_id` | `mk_term`, `sq_term`, `mk_def`, `sq_def`, `domain` |
| Media literacy | `mk`, `sq`, `mk_description`, `sq_description` | `mk_term`, `sq_term`, `mk_def`, `sq_def` |
| MUSE | `mk`, `sq`, `en` | `mk_term`, `sq_term`, `mk_def` = `sq_def` = `""`, `concept_id` from `en` |

The convention tasks rely on:

* **Parallel tasks** (translation) read only `record["mk"]` and `record["sq"]`.
* **Lexical tasks** (term pairs, term + definition, definition translation, shuffled definitions)
  read only `mk_term`, `sq_term`, `mk_def`, `sq_def`. For a direction, build the name:
  `src, tgt = split_direction(d)` then `record[f"{src}_term"]`, `record[f"{src}_def"]`,
  `record[f"{tgt}_term"]`.
* **A missing definition is `""`**, never absent or `None` (MUSE has none). Definition tasks skip
  records whose definition is empty.
* The names are `mk_def` / `sq_def`, not "description": the sources' `*_description` columns are
  renamed.
* **Mismatches fail early.** Config validation rejects a lexical task on a parallel loader (and the
  reverse), and `download` raises if a loader's chunk lacks a standard column, before training starts.

## How the base class works (`base.py`)

A subclass implements only `_read(split)`: a generator of column dicts (`{column: list}`) with
exactly the standard columns. Read the source in pieces (`parquet_batches`), never whole, and
record provenance in `self._source`. Everything else lives in `DatasetLoader`:

| Method | What it does |
| --- | --- |
| `download(split, force=False)` | Writes `_read`'s chunks to `<cache>/<loader>/<version>/<split>.arrow` (Arrow IPC **stream** format, uncompressed, chunks of `WRITE_CHUNK` = 16,384 rows), atomically via `.tmp` + rename. Writes a `<split>.json` sidecar: `rows`, `sha256` of the arrow file, and `_source` (repo + resolved commit, or local path + source sha256). |
| `stream_batch_from_disk(split, batch_size=1024, start=0, stop=None)` | Generator. Reads the file **sequentially**, one stored chunk at a time, and yields `list[dict]` batches. RAM = one chunk + one batch. Downloads first if needed. |
| `load(split)` | `datasets.Dataset.from_file` on the same arrow file (memory-mapped, no copy), validated. |
| `select(ds, **rules)` | Config `select:` rules. Base accepts none; subclasses list theirs in `select_rules`. |
| `describe()` | Loader, kind, options and every split's sidecar. |
| `source_files()` | Option `path` (local file) if set, else every `.parquet` under `prefix` in HF repo `repo` at `revision`. |
| `opt(key)` | Config option, else class attribute (`repo`, `revision`, `prefix`). |

Cache layout: root is `$MKSQ_DATA_CACHE` or `data/cache` (gitignored). `<version>` is the pinned
`revision`, `main` when unpinned (the sidecar still records the resolved commit), or `local` when
the `path` option is used, so a local copy never masks the hub copy. Delete a folder or pass
`force=True` to rebuild.

Helpers: `hub_files(repo, revision, prefix, suffix)` downloads through the HF cache and returns
`(paths, commit_sha)`; `local_file(path)` resolves paths relative to the repository root;
`concept_key(term)` = casefolded, whitespace-normalised term.

### Design decisions (keep them)

* **Not memory-mapped for streaming.** Mapped pages stay resident; streaming OPUS through mmap
  pushed RSS to the full 2.2 GB file. Sequential reads keep it at **97 MB**.
* **No shuffling while streaming.** Order on disk is the order you get. Shuffle once when a frozen
  tag is built (TG-D0), write it, stream it sequentially.
* **IPC stream format** (not file format) so `datasets.Dataset.from_file` can read it.
* Measured on OPUS (4,268,571 rows): write 45 s, peak 1.09 GB (the source parquet is one
  500k-row row group, decoded at once; one-time cost); full stream 14 s, peak 97 MB; resume at row
  4M in 0.2 s.

## The loaders

| Name | Kind | Splits | Source | Pinned revision |
| --- | --- | --- | --- | --- |
| `opus` | parallel | train | HF `EdonFetaji/opus-mk-sq-clean-dedup`, `data/train-*` (default config) | `0920a144…` |
| `gazette` | parallel | train, validation, test | HF `EdonFetaji/slvesnik-mk-sq-clean` (`data/{split}-*`), public | `e9ed9fb6…` |
| `flores` | parallel | dev, devtest | HF `openlanguagedata/flores_plus` (gated) | `5fec6c13…` |
| `ntrex` | parallel | test | GitHub `MicrosoftTranslator/NTREX` raw files | `468c6b69…` |
| `phrasebook` | parallel | train | HF private `EdonFetaji/mksq-phrasebook` | **none yet** |
| `verbis` | lexical | train | HF private `EdonFetaji/verbis-contextual-dictionary` | **none yet** |
| `media_literacy` | lexical | train | HF private `EdonFetaji/media-literacy-vocabulary` | **none yet** |
| `muse` | lexical | train | HF `EdonFetaji/mk-sq-dictionary-muse` | `7980380d…` |

Per-loader mapping to the standard columns:

* **opus**: `source_id` = corpus name from `opus:<corpus>:<version>` (`SETIMES`, `NLLB`, …, what
  configs list under `source_ids`). The default config has no document ids, so `doc_id` = `pair_id`.
  `meta`: `unit_type, domain, translation_origin, pairing_method, english_mediated, release_scope, flags`.
  No LaBSE score yet (TG-D0 adds it). 4,268,571 rows, 17 corpora.
* **gazette**: `doc_id` = `source_pdf` (the issue; the repo's splits are issue-disjoint by it).
  `source_id` = `gazette`. `meta`: `id, year, issue_no, segment_type, is_high_quality, redacted,
  cometkiwi22, metricx24_qe, sonar_cosine`. 269,896 / 13,350 / 10,166 rows; 2,085 / 90 / 90 issues,
  no issue shared between train and validation/test (checked).
* **flores**: the `mkd_Cyrl` and `als_Latn` files of the split, joined on `id` (fails if ids differ).
  `doc_id` = article `url`. Needs the gated terms accepted on the token's account.
* **ntrex**: line *i* of `newstest2019-ref.mkd.txt` ↔ line *i* of `.sqi.txt`, split on `\n` only
  (U+2028 inside lines would misalign `splitlines()`). `doc_id` = line number. 1,997 rows.
* **phrasebook**: `doc_id` = `lesson:<first lesson>`; `meta` = `pair_ids`. Person-mismatch pairs
  are not flagged in the data: list them in option `drop_pair_ids`. 1,779 rows.
* **verbis**: `entry_id` = `verbis:<verbis_id>`; `concept_id` = `verbis:<casefolded mk term>` (all
  senses of a headword, in any domain, held out together); `domain` from `category_id`
  1 `legal_administrative`, 2 `finance_economic`, 3 `health` (Shëndetësore), 4 `general`.
  Defs from `mk_description` / `sq_description`. 15,925 rows.
* **media_literacy**: drops the 3 rows whose `quality_flags` contain `mk_link_self_reference`;
  `concept_id` like VERBIS; `domain` = `media_literacy`. 1,081 rows.
* **muse**: `keep_pair` drops identical strings, anything with a digit, and names (both sides
  capitalised); currently drops none. `concept_id` = English pivot word; no definitions. 7,388 rows.

## Rules that apply to the data

* **Never print, log or commit VERBIS rows** (redistribution not granted). Media literacy and the
  phrasebook are research-only too; keep all three out of anything public, including TG-90's mix.
* FLORES+ and NTREX file hashes are deliberately UNTRACKED; they are pinned by revision only.

### `restricted`: which runs stay private

`DatasetLoader.restricted` (a `ClassVar[bool]`, default `False`) marks a source that must not be
redistributed. It is `True` for `verbis`, `media_literacy` and `phrasebook`. The training pipeline
(`src/pipeline/hub.py`) reads it to choose the HF repo for a run with `output.repo: auto`:

* **private** (`EdonFetaji/translategemma-mk-sq-private`): if any `data.mixture` or `data.shared`
  entry uses a restricted loader, or the run's parent (`model.init_from: run:<ID>`, followed up the
  chain) was private;
* **public** (`EdonFetaji/translategemma-mk-sq`) otherwise.

`output.repo: public` on a restricted run is a config error at push time. A new loader for
licensed data must set `restricted = True`.
* Test and dev sets are never trained on.

## Open work

* `select()` is `NotImplementedError` in opus, gazette, verbis and media_literacy. Its rules
  (`tag: G / G_REPLAY / G_REST`, `slice: L_MAIN / L_ARMS / L_FRESH`, `exclude/only: V_HOLDOUT`,
  LaBSE floors, `top_k`, `stream_from/stream_to`) depend on the TG-D0 freeze, which isn't built yet.
* Pin `revision` for phrasebook, verbis and media_literacy (copy the commit sha from the HF repo,
  or from a sidecar after a download).
* FLORES+ (gated) and the three private repos have not been tested end to end: the dev machine's
  HF token is invalid (`hf auth login`). They were tested from local copies via `path`.
* `.venv` had pyarrow 17 with datasets 5, which breaks `import datasets` (and so `load()`);
  `pyproject.toml` asks for `pyarrow>=21`, run `uv lock && uv sync`. Streaming does not need `datasets`.

## Adding a loader

1. New module: subclass `DatasetLoader`, set `kind`, `splits`, and `repo` / `revision` (or read
   option `path`), implement `_read` yielding chunks with exactly `COLUMNS[kind]`.
2. Decorate with `@register_loader("<name>")` and import the module in `__init__.py`.
3. If it takes `select:` rules, list them in `select_rules` (config validation rejects others).
