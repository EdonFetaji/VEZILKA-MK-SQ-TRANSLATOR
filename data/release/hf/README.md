---
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
  - name: pair_id
    dtype: string
  - name: mk_text
    dtype: string
  - name: sq_text
    dtype: string
  - name: year
    dtype: int16
  - name: issue_no
    dtype: int16
  - name: issue_key
    dtype: string
  - name: mk_component_type
    dtype: string
  - name: sq_component_type
    dtype: string
  - name: mk_page
    dtype: int16
  - name: sq_page
    dtype: int16
  - name: mk_component
    dtype: int16
  - name: sq_component
    dtype: int16
  - name: is_high_quality
    dtype: bool
  - name: quality_confidence
    dtype: float32
  - name: sonar_cosine
    dtype: float32
  - name: gale_church_inferred
    dtype: bool
  - name: mk_ocr_confidence
    dtype: float32
  - name: sq_ocr_confidence
    dtype: float32
  - name: source_pdf
    dtype: string
  - name: split
    dtype: string
  - name: redacted
    dtype: bool
  splits:
  - name: train
    num_examples: 270083
  - name: validation
    num_examples: 13355
  - name: test
    num_examples: 10174
---

# Службен весник MK–SQ Legal Parallel Corpus

293,612 Macedonian–Albanian sentence pairs from the Official Gazette of the
Republic of North Macedonia (Службен весник), 2001–2025.

```python
from datasets import load_dataset

ds = load_dataset("EdonFetaji/slvesnik-mk-sq")
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
| train | 270,083 |
| validation | 13,355 |
| test | 10,174 |

## Fields

| field | meaning |
|---|---|
| `pair_id` | stable sortable id, {issue_key}-{sq_page}-{sq_component} |
| `mk_text` | Macedonian side |
| `sq_text` | Albanian side |
| `year` | publication year |
| `issue_no` | gazette issue number within the year |
| `issue_key` | '{year}-{issue_no:04d}', the unit the split respects |
| `mk_component_type` | plain_text | title | table | ... (layout model) |
| `sq_component_type` | as above, Albanian side |
| `mk_page` | page of the source PDF the Macedonian side came from |
| `sq_page` | page of the source PDF the Albanian side came from |
| `mk_component` | component index within that page |
| `sq_component` | component index within that page |
| `is_high_quality` | the extraction pipeline's own strict flag |
| `quality_confidence` | continuous alignment score in [0, 1] |
| `sonar_cosine` | SONAR similarity; null on is_high_quality rows, never scored |
| `gale_church_inferred` | the link came from length alone, not content |
| `mk_ocr_confidence` | OCR score where the page was OCR'd (~4% of rows) |
| `sq_ocr_confidence` | as above, Albanian side |
| `source_pdf` | source PDF object name; dictionary-encoded, near-free |
| `split` | train | validation | test, issue-disjoint |
| `redacted` | the pair carries at least one redaction placeholder |

## Personal data

The gazette's classified section publishes material that identifies individuals.
Two treatments were applied before release.

**Structured identifiers were replaced with placeholders.** Each is the same
string on both sides of the pair, so replacement keeps the alignment intact:

| placeholder | replaced | occurrences |
|---|---|---|
| `[ID-NUMBER]` | 13-digit national / tax numbers | 3,565 |
| `[DOC-NUMBER]` | passport-style document numbers | 15 |

**Rows whose content is a person's identity were removed** — 2,016 pairs:
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
@misc{slvesnik_mk_sq,
  title        = {Службен весник MK--SQ Legal Parallel Corpus},
  author       = {Edon Fetaji, VEZILKA},
  year         = {09.2026},
  howpublished = {Hugging Face Datasets},
  url          = {https://huggingface.co/datasets/EdonFetaji/slvesnik-mk-sq}
}
```

## Contact

Corrections and takedown requests: fetaji.ed@gmail.com
