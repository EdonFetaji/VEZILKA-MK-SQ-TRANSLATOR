"""Explicit pyarrow schemas. Nothing in this project infers a schema from data.

`RAW_SCHEMA` is the 46-column shape that notebook `01_3` wrote. It is read-only:
the raw extractor output is never mutated, only read.

`CORPUS_SCHEMA` is what `scripts/prepare_corpus.py` writes -- the raw columns
unchanged, plus the chronology, split, near-duplicate and token-length columns
this project adds. Every added column is listed in `ADDED_FIELDS` so a diff
against the raw shape is a one-liner.
"""

from __future__ import annotations

import pyarrow as pa

# --- exactly as 01_3 wrote it ------------------------------------------------
RAW_FIELDS: list[tuple[str, pa.DataType]] = [
    ("slv_identifier", pa.string()),
    ("filename", pa.string()),
    ("mk_component_id", pa.string()),
    ("sq_component_id", pa.string()),
    ("mk_text", pa.string()),
    ("sq_text", pa.string()),
    ("mk_component_type", pa.string()),
    ("sq_component_type", pa.string()),
    ("mk_ocr_confidence", pa.float64()),
    ("sq_ocr_confidence", pa.float64()),
    ("match_confidence", pa.float64()),
    ("is_high_quality", pa.bool_()),
    ("year", pa.int16()),
    ("source_object", pa.string()),
    ("schema_shape", pa.string()),
    ("gale_church_inferred", pa.bool_()),
    ("matcher_term", pa.float64()),
    ("text_term", pa.float64()),
    ("quality_confidence", pa.float64()),
    ("mk_chars", pa.int64()),
    ("sq_chars", pa.int64()),
    ("mk_words", pa.int64()),
    ("sq_words", pa.int64()),
    ("length_ratio", pa.float64()),
    ("mk_cyrillic_share", pa.float64()),
    ("sq_latin_share", pa.float64()),
    ("script_purity", pa.float64()),
    ("numeric_overlap_raw", pa.float64()),
    ("numeric_overlap", pa.float64()),
    ("mk_language", pa.string()),
    ("sq_language", pa.string()),
    ("lingua_agreement", pa.float64()),
    ("either_empty", pa.bool_()),
    ("very_short", pa.bool_()),
    ("copy_through", pa.bool_()),
    ("copy_through_norm", pa.bool_()),
    ("dup_key", pa.string()),
    ("dup_group_id", pa.int64()),
    ("dup_rank", pa.int64()),
    ("dup_count", pa.int64()),
    ("split", pa.string()),
    ("split_bucket", pa.int16()),
    ("dup_leaks_into_train", pa.bool_()),
    ("sonar_cosine", pa.float64()),
    ("sonar_text_too_short", pa.bool_()),
    ("ref_number_contradiction", pa.bool_()),
]

RAW_SCHEMA = pa.schema(RAW_FIELDS)

# Columns of the raw shape this project supersedes. They are dropped rather than
# overwritten, so a stale hash-split value can never be mistaken for the
# chronological one (constraint C3).
SUPERSEDED = ["split", "split_bucket", "dup_leaks_into_train"]

# --- what prepare_corpus.py adds ---------------------------------------------
ADDED_FIELDS: list[tuple[str, pa.DataType]] = [
    # chronology, recovered from filename + slv_identifier
    ("issue_no", pa.int32()),          # gazette issue number within the year
    ("issue_month", pa.int8()),        # month from slv_identifier, for cross-checks
    ("issue_key", pa.string()),        # "{year}-{issue_no:04d}", the stable issue id
    ("issue_order", pa.int32()),       # 0-based chronological rank of the issue
    # split (chronological, issue-disjoint)
    ("split", pa.string()),            # train | dev | test
    # near-duplicate sweep (cross-split, MinHash/LSH over char n-grams)
    ("near_dup_max_jaccard", pa.float64()),  # best estimated Jaccard against train, else null
    # NLLB token lengths, precomputed so length filtering never re-tokenises
    ("mk_nllb_tokens", pa.int32()),
    ("sq_nllb_tokens", pa.int32()),
]

CORPUS_FIELDS = [f for f in RAW_FIELDS if f[0] not in SUPERSEDED] + ADDED_FIELDS
CORPUS_SCHEMA = pa.schema(CORPUS_FIELDS)

# The sidecar written for every segment the near-duplicate sweep removes.
DROPPED_SCHEMA = pa.schema(
    [
        ("mk_component_id", pa.string()),
        ("sq_component_id", pa.string()),
        ("issue_key", pa.string()),
        ("year", pa.int16()),
        ("split", pa.string()),                 # the split it would have been in
        ("side", pa.string()),                  # mk | sq | both -- which side matched
        ("near_dup_max_jaccard", pa.float64()),
        ("train_match_sq_component_id", pa.string()),
        ("mk_text", pa.string()),
        ("sq_text", pa.string()),
        ("train_match_mk_text", pa.string()),
        ("train_match_sq_text", pa.string()),
    ]
)

PARTITION_COLUMNS = ["split", "year"]


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reorder and cast `table` to `schema`. Raises if a column is missing."""
    missing = [f.name for f in schema if f.name not in table.column_names]
    if missing:
        raise ValueError(f"table is missing {missing}")
    arrays = [table.column(f.name).cast(f.type) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema)
