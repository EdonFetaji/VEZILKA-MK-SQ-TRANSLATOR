"""Shared definitions for the three-stage public release.

    1. scripts/prune_columns.py   derive the publishable columns, delete the rest
    2. scripts/redact_corpus.py   replace identifiers with placeholders
    3. scripts/publish_hf.py      Hugging Face layout, dataset card, upload command

Each stage reads a Parquet file and writes a new one; none of them ever modifies
its input. Running them in order is the release.
"""

from __future__ import annotations

import re

import pyarrow as pa

# ---------------------------------------------------------------------------
# Stage 1: columns
# ---------------------------------------------------------------------------

PUBLISHED_COLUMNS: dict[str, str] = {
    "pair_id": "stable sortable id, {issue_key}-{sq_page}-{sq_component}",
    "mk_text": "Macedonian side",
    "sq_text": "Albanian side",
    "year": "publication year",
    "issue_no": "gazette issue number within the year",
    "issue_key": "'{year}-{issue_no:04d}', the unit the split respects",
    "mk_component_type": "plain_text | title | table | ... (layout model)",
    "sq_component_type": "as above, Albanian side",
    "mk_page": "page of the source PDF the Macedonian side came from",
    "sq_page": "page of the source PDF the Albanian side came from",
    "mk_component": "component index within that page",
    "sq_component": "component index within that page",
    "is_high_quality": "the extraction pipeline's own strict flag",
    "quality_confidence": "continuous alignment score in [0, 1]",
    "sonar_cosine": "SONAR similarity; null on is_high_quality rows, never scored",
    "gale_church_inferred": "the link came from length alone, not content",
    "mk_ocr_confidence": "OCR score where the page was OCR'd (~4% of rows)",
    "sq_ocr_confidence": "as above, Albanian side",
    "source_pdf": "source PDF object name; dictionary-encoded, near-free",
    "split": "train | validation | test, issue-disjoint",
}

# Why each deleted column is deleted. This is the audit trail, not a comment.
DELETED_COLUMNS: dict[str, str] = {
    # no information: constant across every row after 01_2/01_3 filtering
    "schema_shape": "constant ('12col')",
    "sq_language": "constant ('ALBANIAN')",
    "either_empty": "constant (False)",
    "copy_through": "constant (False)",
    "copy_through_norm": "constant (False)",
    "dup_rank": "constant (1)",
    "dup_leaks_into_train": "constant (False)",
    # build infrastructure, not data
    "source_object": "leaks the GCS bucket path sl_vesnik_izdanija/DATA_PARQUET/...",
    "split_bucket": "implementation detail of the hash split",
    "dup_group_id": "internal factorize() id, meaningless outside the build",
    "dup_key": "internal sha1 of the normalised pair",
    "slv_identifier": "documented in 01_1 as NOT a stable key; publishing invites a bad join",
    "filename": "superseded by source_pdf, issue_key and issue_no",
    "mk_component_id": "superseded by mk_page and mk_component",
    "sq_component_id": "superseded by sq_page and sq_component",
    # recomputable from the text in one line; pure download weight
    "mk_chars": "len(mk_text)",
    "sq_chars": "len(sq_text)",
    "mk_words": "len(mk_text.split())",
    "sq_words": "len(sq_text.split())",
    "length_ratio": "min/max of the two character counts",
    "mk_cyrillic_share": "character class ratio",
    "sq_latin_share": "character class ratio",
    "script_purity": "character class ratio",
    "numeric_overlap": "shared-numeral ratio",
    "numeric_overlap_raw": "shared-numeral ratio",
    "very_short": "a length threshold",
    "matcher_term": "a component of quality_confidence",
    "text_term": "a component of quality_confidence",
    "mk_language": "lingua verdict, 2 distinct values, 8.6% null",
    "lingua_agreement": "derived from the two language verdicts",
    "sonar_text_too_short": "a length threshold on the shorter side",
    "dup_count": "copies within the pre-filter corpus; not meaningful post-filter",
    "ref_number_contradiction": "75% null; recomputable from the text",
    # documented as unusable
    "match_confidence": (
        "01_1 documents it as mixing Hybrid scores with Gale-Church normal-tails on "
        "incomparable scales; quality_confidence supersedes it"
    ),
}

ARROW_TYPES: dict[str, pa.DataType] = {
    "pair_id": pa.string(), "mk_text": pa.string(), "sq_text": pa.string(),
    "year": pa.int16(), "issue_no": pa.int16(), "issue_key": pa.string(),
    "mk_component_type": pa.string(), "sq_component_type": pa.string(),
    "mk_page": pa.int16(), "sq_page": pa.int16(),
    "mk_component": pa.int16(), "sq_component": pa.int16(),
    "is_high_quality": pa.bool_(), "quality_confidence": pa.float32(),
    "sonar_cosine": pa.float32(), "gale_church_inferred": pa.bool_(),
    "mk_ocr_confidence": pa.float32(), "sq_ocr_confidence": pa.float32(),
    "source_pdf": pa.string(), "split": pa.string(),
    "redacted": pa.bool_(),
}

SPLIT_RENAME = {"train": "train", "dev": "validation", "test": "test"}

# ---------------------------------------------------------------------------
# Stage 2: redaction
# ---------------------------------------------------------------------------

# Structured identifiers only. Each is script-independent -- it appears as the same
# string on the Macedonian and the Albanian side -- so replacing it with a
# placeholder keeps the pair aligned. Order matters: the longest pattern first, so
# a 13-digit number is not eaten by the passport rule.
REDACTIONS: list[tuple[str, re.Pattern, str]] = [
    ("national_id", re.compile(r"(?<!\d)\d{13}(?!\d)"), "[ID-NUMBER]"),
    ("passport", re.compile(r"\b[A-ZА-Шa-zа-ш]{1,2}\d{6,9}\b"), "[DOC-NUMBER]"),
]

# Rows whose *content is a person's identity* -- lost-document notices, sanctions
# entries, "born on" records. Redacting the numbers out of these leaves the name,
# and names cannot be found reliably with a regex across two scripts. These are
# dropped rather than redacted badly.
PERSON_CONTEXT: list[tuple[str, re.Pattern, str]] = [
    ("mk_text", re.compile(r"на\s+име\b", re.I), "issued in the name of"),
    ("mk_text", re.compile(r"\bизгубен[аиo]?\b", re.I), "lost document notice"),
    ("mk_text", re.compile(r"\bроден[аи]?\s+на\b", re.I), "born on"),
    ("mk_text", re.compile(r"лична\s+карта", re.I), "identity card"),
    ("mk_text", re.compile(r"\bпасош", re.I), "passport"),
    ("mk_text", re.compile(r"\b(?:диплома|свидетелств)", re.I), "diploma / certificate"),
    ("sq_text", re.compile(r"\b[ie]\s+humbur\b", re.I), "lost"),
    ("sq_text", re.compile(r"shpallet\s+i\s+pavlefsh", re.I), "declared invalid"),
    ("sq_text", re.compile(r"\bpasaport", re.I), "passport"),
    ("sq_text", re.compile(r"\bletërnjoftim", re.I), "identity card"),
    ("sq_text", re.compile(r"\bdiplom", re.I), "diploma"),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace structured identifiers with placeholders. Returns (text, counts)."""
    if not text:
        return text, {}
    counts: dict[str, int] = {}
    for label, pattern, placeholder in REDACTIONS:
        text, n = pattern.subn(placeholder, text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts


def person_context_hits(mk: str, sq: str) -> list[str]:
    """Which person-context cues fire on this pair, if any."""
    hits = []
    for column, pattern, label in PERSON_CONTEXT:
        if pattern.search((mk if column == "mk_text" else sq) or ""):
            hits.append(label)
    return hits


def schema_for(names: list[str]) -> pa.Schema:
    return pa.schema([(n, ARROW_TYPES[n]) for n in names])
