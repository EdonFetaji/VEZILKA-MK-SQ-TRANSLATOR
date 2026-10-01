#!/usr/bin/env python
"""Repair OCR character substitutions in the Albanian side of the pair table.

    python scripts/repair_corpus.py IN.parquet OUT.parquet
    python scripts/repair_corpus.py IN.parquet OUT.parquet --no-mixed-script
    python scripts/repair_corpus.py IN.parquet --dry-run

Two defects, both learned by any model fine-tuned on the corpus if left alone:

  `[` for `ë`   27,035 occurrences against 233 `]` -- 116:1, where the Macedonian
                side runs 7:7. 91% are word-internal and 98.7% of the words
                containing one resolve to a word attested elsewhere in the corpus
                (t[->të, n[->në, p[r->për, s[->së).

  Cyrillic      `Е` (U+0415) for `E`, `а` (U+0430) for `a`, `Ё` for `Ë`. Folded
  homoglyphs    only inside mixed-script words, so an Albanian sentence quoting a
                Macedonian title in Cyrillic is left alone.

Both repairs are conservative, and the script refuses to write unless the
conservation identity holds: every `[` removed is matched by an `ë` added.

The input is never modified. Only `sq_text` changes; every other column, the row
order and the schema are carried through untouched.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mksq.repair import repair_bracket_e, repair_sq  # noqa: E402

RE_BRACKET = re.compile(r"\[")
RE_EDIA = re.compile(r"[ëË]")
RE_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def count(texts: list[str], pattern: re.Pattern) -> int:
    return sum(len(pattern.findall(t)) for t in texts if t)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="the pair table to read")
    parser.add_argument("output", type=Path, nargs="?", help="where to write the repaired copy")
    parser.add_argument("--column", default="sq_text", help="the column to repair")
    parser.add_argument(
        "--no-mixed-script", action="store_true",
        help="do the `[` -> `ë` fix only, leaving Cyrillic homoglyphs alone",
    )
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        parser.error("an output path is required unless --dry-run is given")
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input -- give a different output path")

    table = pq.read_table(args.input)
    if args.column not in table.column_names:
        parser.error(f"{args.input} has no column {args.column!r}")
    print(f"read {args.input}  ({table.num_rows:,} rows x {table.num_columns} columns)")

    original = table.column(args.column).to_pylist()
    repaired = [repair_sq(t, mixed_script=not args.no_mixed_script) for t in original]
    bracket_only = [repair_bracket_e(t) if t else t for t in original]

    # The conservation identity holds for the bracket fix *in isolation*. The
    # homoglyph fold also produces `ë` (from `Ё`/`ё`), so measuring the two
    # together would compare one repair's brackets against both repairs' `ë`.
    brackets_removed = count(original, RE_BRACKET) - count(bracket_only, RE_BRACKET)
    e_added = count(bracket_only, RE_EDIA) - count(original, RE_EDIA)
    e_from_homoglyphs = count(repaired, RE_EDIA) - count(bracket_only, RE_EDIA)
    rows_bracket = sum(a != b for a, b in zip(original, bracket_only))
    rows_total = sum(a != b for a, b in zip(original, repaired))

    print(f"\ncolumn                     {args.column}")
    print(f"  '[' before / after       {count(original, RE_BRACKET):>9,} / {count(repaired, RE_BRACKET):,}")
    print(f"  ë/Ë before / after       {count(original, RE_EDIA):>9,} / {count(repaired, RE_EDIA):,}")
    print(f"  Cyrillic before / after  {count(original, RE_CYRILLIC):>9,} / {count(repaired, RE_CYRILLIC):,}")
    print(f"  rows changed by '[' fix  {rows_bracket:>9,}")
    print(f"  rows changed in total    {rows_total:>9,}  ({rows_total / max(table.num_rows, 1):.2%})")

    # The identity that says nothing but `[` -> `ë` happened to the brackets.
    if brackets_removed != e_added:
        raise SystemExit(
            f"\nREFUSING TO WRITE: the '[' fix removed {brackets_removed:,} brackets but "
            f"added {e_added:,} 'ë'.\nIt changed something other than the substitution "
            f"it claims to."
        )
    print(f"\n  conservation OK: '[' fix removed {brackets_removed:,} brackets, "
          f"added {e_added:,} 'ë'")
    if not args.no_mixed_script:
        print(f"  homoglyph fold additionally produced {e_from_homoglyphs:,} 'ë' (from Ё/ё)")

    if "split" in table.column_names:
        splits = table.column("split").to_pylist()
        per_split: dict[str, int] = {}
        for split, before, after in zip(splits, original, repaired):
            if before != after:
                per_split[split] = per_split.get(split, 0) + 1
        print("  rows changed by split:", per_split or "none")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    index = table.column_names.index(args.column)
    out = table.set_column(index, args.column, pa.array(repaired, pa.string()))
    assert out.schema.equals(table.schema), "schema changed"
    assert out.num_rows == table.num_rows, "row count changed"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.output, compression="zstd")

    back = pq.read_table(args.output)
    assert back.num_rows == table.num_rows, "row count changed on the round trip"
    assert back.schema.equals(table.schema), "schema changed on the round trip"
    assert back.column(args.column).to_pylist() == repaired, "the repaired column did not survive"
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:,.1f} MB)")
    print(f"round trip OK: {back.num_rows:,} rows, {back.num_columns} columns, schema unchanged")


if __name__ == "__main__":
    main()
