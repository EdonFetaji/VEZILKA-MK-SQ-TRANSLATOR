#!/usr/bin/env python
"""Release stage 3 of 3 -- Hugging Face layout, dataset card, upload.

    python scripts/publish_hf.py data/release/02_redacted.parquet \
        --out data/release/hf --repo-id YOUR-USERNAME/slvesnik-mk-sq

Writes a directory that is a valid Hugging Face dataset repository: the three
splits as Parquet under `data/`, plus a `README.md` dataset card whose YAML
frontmatter declares the features and split sizes.

Every fact the card cannot derive from the data is written as an explicit
`<FILL_IN ...>` placeholder rather than a plausible guess. `--check` lists what
is still unfilled, and the upload refuses to run while any remain.

Uploading is a separate, explicit step: this script prints the command and only
performs it when given `--push`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mksq.release import PUBLISHED_COLUMNS  # noqa: E402

PLACEHOLDER = re.compile(r"<FILL_IN [^>]*>")

HF_DTYPE = {
    "string": "string", "int16": "int16", "bool": "bool",
    "float": "float32", "double": "float64",
}


def card(table: pa.Table, counts: dict[str, int], repo_id: str, redaction: dict) -> str:
    total = sum(counts.values())
    span = pc.min_max(table.column("year")).as_py()
    features = "\n".join(
        f"  - name: {f.name}\n    dtype: {HF_DTYPE.get(str(f.type), str(f.type))}"
        for f in table.schema
    )
    splits = "\n".join(f"  - name: {k}\n    num_examples: {v}" for k, v in counts.items())
    split_rows = "\n".join(f"| {k} | {v:,} |" for k, v in counts.items())
    field_rows = "\n".join(
        f"| `{n}` | {d} |" for n, d in PUBLISHED_COLUMNS.items() if n in table.column_names
    )
    dropped = redaction.get("person_rows_dropped", 0)
    replacements = redaction.get("replacements", {})

    return f"""---
license: cc-by-4.0
language:
  - mk
  - sq
multilinguality: translation
task_categories:
  - translation
size_categories:
  - 100K<n<1M
source_datasets:
  - original
pretty_name: Sluzben Vesnik MK-SQ Legal Parallel Corpus
tags:
  - legal
  - macedonian
  - albanian
  - low-resource
  - official-gazette
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/validation-*
      - split: test
        path: data/test-*
dataset_info:
  features:
{features}
  splits:
{splits}
---

# Службен весник MK–SQ Legal Parallel Corpus

{total:,} Macedonian–Albanian sentence pairs from the Official Gazette of the
Republic of North Macedonia (Службен весник), {span['min']}–{span['max']}.

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["test"][0]["mk_text"], ds["test"][0]["sq_text"])
```

## Splits

The split is **issue-disjoint**: assignment happens at the level of the source
PDF, so no gazette issue contributes to more than one split. Gazette text is
formulaic and repeats across issues, so a sentence-level random split leaks
badly. Deduplication is global — one row per normalised pair across the whole
corpus — so no evaluation text also appears in train.

| split | pairs |
|---|---|
{split_rows}

## Fields

| field | meaning |
|---|---|
{field_rows}
| `redacted` | the pair carries at least one redaction placeholder |

## Personal data

The gazette's classified section publishes material that identifies individuals.
Two treatments were applied before release.

**Structured identifiers were replaced with placeholders.** Each is the same
string on both sides of the pair, so replacement keeps the alignment intact:

| placeholder | replaced | occurrences |
|---|---|---|
| `[ID-NUMBER]` | 13-digit national / tax numbers | {replacements.get('national_id', 0):,} |
| `[DOC-NUMBER]` | passport-style document numbers | {replacements.get('passport', 0):,} |

**Rows whose content is a person's identity were removed** — {dropped:,} pairs:
lost-document notices, sanctions-list entries and "born on" records. These are
mostly name, and names cannot be located reliably by regex across Cyrillic and
Latin at once, so redacting the numbers would have left the name behind while
looking sanitised.

Detection was a recall-oriented heuristic. It is not a guarantee in either
direction; if you find residual personal data, please open a discussion.

## Known limitations

- **The source is OCR.** Segments from scanned issues carry recognition errors.
  A character-level repair was applied to the Albanian side: `[` mis-decoded as
  `ë` (24,991 occurrences) and Cyrillic homoglyphs inside Latin words.
- **Alignment is automatic.** No subset has been manually verified.
- **`quality_confidence` is a heuristic**, not a human judgement.
- **`sonar_cosine` is null on `is_high_quality` rows**, which were never scored.
- The split is issue-disjoint but **not chronological**; issues are assigned by a
  hash of the filename, so all years appear in all three splits.

## Provenance

Built from the published PDF issues of Службен весник: layout analysis, OCR where
there is no text layer, then bilingual component matching. `source_pdf`,
`issue_key`, `mk_page` and `sq_page` locate every pair in its source issue.

## Licence

Two layers, and they are not the same.

**Contents.** The Macedonian and Albanian text is drawn from the Official Gazette
of the Republic of North Macedonia. Official texts of a political, legislative,
administrative and judicial nature, **and their official translations**, are
excluded from protection under the Macedonian Law on Copyright and Related
Rights. The Albanian side of the gazette is such an official translation, so no
copyright is claimed over either side, and none is granted here.

**This compilation.** The selection, extraction, alignment, SONAR scoring,
redaction, splits and derived columns represent substantial investment and are
released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution is to this compilation, not to the gazette text.

The code that produced it is licensed separately; see the project repository.

## Citation

```bibtex
@misc{{slvesnik_mk_sq,
  title        = {{Службен весник MK--SQ Legal Parallel Corpus}},
  author       = {{Edon Fetaji, VEZILKA}},
  year         = {{09.2026}},
  howpublished = {{Hugging Face Datasets}},
  url          = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

## Contact

Corrections and takedown requests: fetaji.ed@gmail.com
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-id", default="<FILL_IN username/dataset-name>")
    parser.add_argument("--check", action="store_true", help="list unfilled placeholders and exit")
    parser.add_argument("--push", action="store_true", help="upload to the Hub")
    parser.add_argument("--private", action="store_true", help="create the repo private")
    args = parser.parse_args()

    if args.check:
        return check(args.out)

    if args.input is None:
        parser.error("an input Parquet is required unless --check is given")

    import json

    table = pq.read_table(args.input)
    print(f"read {args.input}  ({table.num_rows:,} rows x {table.num_columns} columns)")

    sidecar = args.input.with_suffix(".redaction.json")
    redaction = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if not redaction:
        print(f"  note: no {sidecar.name} beside the input -- the card's personal-data")
        print("        section will show zeros. Run scripts/data/redact_corpus.py first.")

    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        part = table.filter(pc.equal(table.column("split"), split))
        if part.num_rows == 0:
            continue
        path = data_dir / f"{split}-00000-of-00001.parquet"
        pq.write_table(part, path, compression="zstd")
        counts[split] = part.num_rows
        print(f"  {split:<11} {part.num_rows:>8,} rows -> {path.name} "
              f"({path.stat().st_size / 1e6:,.1f} MB)")

    readme = args.out / "README.md"
    readme.write_text(card(table, counts, args.repo_id, redaction), encoding="utf-8")
    print(f"\nwrote {readme}")

    remaining = check(args.out)
    if args.push:
        if remaining:
            raise SystemExit("\nrefusing to push while placeholders are unfilled")
        push(args.out, args.repo_id, args.private)
    else:
        print("\nTo upload:" if not remaining else
              "\nTo upload once the placeholders above are filled in:")
        print("    hf auth login")
        print(f"    hf upload {args.repo_id} {args.out} . --repo-type dataset")
        print(f"\n  or re-run this script with:  --repo-id {args.repo_id} --push")


def check(out: Path) -> int:
    readme = out / "README.md"
    if not readme.exists():
        print(f"no card at {readme}")
        return 1
    found = PLACEHOLDER.findall(readme.read_text(encoding="utf-8"))
    if found:
        print(f"\n{len(found)} placeholder(s) still to fill in {readme}:")
        for f in found:
            print(f"  {f[:96]}")
    else:
        print(f"\nno placeholders left in {readme} -- ready to push")
    return len(found)


def push(out: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset")
    print(f"\npushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
