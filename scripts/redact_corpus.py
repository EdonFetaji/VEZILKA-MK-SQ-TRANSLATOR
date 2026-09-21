#!/usr/bin/env python
"""Release stage 2 of 3 -- replace personal identifiers with placeholders.

    python scripts/redact_corpus.py data/release/01_pruned.parquet \
        data/release/02_redacted.parquet

Two different problems, handled two different ways.

**Structured identifiers are redacted in place.** A 13-digit national/tax number
and a passport-style document number are the same string on both sides of the
pair, so replacing each with a placeholder keeps the alignment intact and the
segment usable:

    `Број на пасош: D9004878.`   ->  `Број на пасош: [DOC-NUMBER].`
    `даночен број: 4054022506320` -> `даночен број: [ID-NUMBER]`

**Rows whose content *is* a person's identity are dropped.** Lost-document
notices, sanctions-list entries and "born on" records are mostly name, and names
cannot be located reliably by regex across Cyrillic and Latin at once. Redacting
the numbers out of them would leave the name behind and look sanitised. Dropping
is the honest option; `--keep-person-rows` overrides it if you would rather keep
them and say so in the card.

The input is never modified. A sidecar records what was replaced and what was
dropped, so the redaction is reviewable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mksq.release import PERSON_CONTEXT, REDACTIONS, person_context_hits, redact  # noqa: E402

PLACEHOLDERS = [placeholder for _, _, placeholder in REDACTIONS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--keep-person-rows", action="store_true",
        help="keep lost-document / sanctions rows instead of dropping them",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.output.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input")

    table = pq.read_table(args.input)
    df = table.to_pandas()
    print(f"read {args.input}  ({len(df):,} rows x {table.num_columns} columns)")

    # --- 1. drop the rows that are a person's identity ---------------------------
    reasons = [person_context_hits(mk, sq) for mk, sq in zip(df["mk_text"], df["sq_text"])]
    is_person_row = [bool(r) for r in reasons]
    reason_counts = Counter(label for r in reasons for label in r)

    print(f"\nperson-context rows: {sum(is_person_row):,} ({sum(is_person_row) / len(df):.2%})")
    for label, n in reason_counts.most_common():
        print(f"    {n:>6,}  {label}")

    dropped_examples = [
        {"pair_id": df["pair_id"].iloc[i], "reasons": reasons[i],
         "mk_text": df["mk_text"].iloc[i][:160]}
        for i, flag in enumerate(is_person_row) if flag
    ][:25]

    if args.keep_person_rows:
        print("  --keep-person-rows: kept")
    else:
        df = df.loc[[not f for f in is_person_row]].reset_index(drop=True)
        print(f"  dropped -> {len(df):,} rows remain")

    # --- 2. redact structured identifiers in what is left -------------------------
    totals: Counter = Counter()
    rows_touched = 0
    for column in ("mk_text", "sq_text"):
        new_values, per_row = [], []
        for text in df[column]:
            replaced, counts = redact(text)
            new_values.append(replaced)
            per_row.append(bool(counts))
            totals.update(counts)
        df[column] = new_values
        per_column = sum(per_row)
        rows_touched += per_column
        print(f"\n{column}: {per_column:,} rows carried an identifier")

    df["redacted"] = (
        df["mk_text"].str.contains("|".join(re.escape(p) for p in PLACEHOLDERS), regex=True)
        | df["sq_text"].str.contains("|".join(re.escape(p) for p in PLACEHOLDERS), regex=True)
    )
    print("\nreplacements:", dict(totals) or "none")
    print(f"rows carrying a placeholder: {int(df['redacted'].sum()):,}")

    # --- 3. verify nothing slipped through ----------------------------------------
    leftover = {}
    for label, pattern, _ in REDACTIONS:
        n = sum(len(pattern.findall(t or "")) for t in df["mk_text"]) + sum(
            len(pattern.findall(t or "")) for t in df["sq_text"]
        )
        leftover[label] = n
    if any(leftover.values()):
        raise SystemExit(f"\nREFUSING TO WRITE: identifiers survived redaction: {leftover}")
    print("verified: no structured identifier survives in either side")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    schema = table.schema.append(pa.field("redacted", pa.bool_()))
    out = pa.Table.from_arrays(
        [pa.array(df[f.name], f.type) for f in schema], schema=schema
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.output, compression="zstd")

    sidecar = args.output.with_suffix(".redaction.json")
    sidecar.write_text(
        json.dumps(
            {
                "rows_in": table.num_rows,
                "rows_out": out.num_rows,
                "person_rows_dropped": 0 if args.keep_person_rows else sum(is_person_row),
                "person_context_counts": dict(reason_counts),
                "replacements": dict(totals),
                "rows_with_placeholder": int(df["redacted"].sum()),
                "placeholders": PLACEHOLDERS,
                "dropped_examples": dropped_examples,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:,.1f} MB)")
    print(f"wrote {sidecar}  (what was replaced, what was dropped)")
    print(f"rows: {table.num_rows:,} in, {out.num_rows:,} out")


if __name__ == "__main__":
    main()
