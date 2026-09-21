#!/usr/bin/env python
"""Release stage 1 of 3 -- derive the publishable columns and delete the rest.

    python scripts/prune_columns.py data/processed/pairs_sonar_repaired.parquet \
        data/release/01_pruned.parquet

Of the 46 columns in the working table this keeps 20 and deletes 34, each with a
recorded reason (`src/mksq/release.py :: DELETED_COLUMNS`, echoed into a JSON
sidecar so the decision is auditable rather than buried in a script).

Four columns are *derived* before the delete, because the information in them is
worth keeping but the carrier is not:

    mk_component_id  ->  mk_page, mk_component     (294 bytes -> two int16)
    sq_component_id  ->  sq_page, sq_component
    filename         ->  issue_no, issue_key, source_pdf
                     ->  pair_id, asserted unique

The input is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mksq.release import (  # noqa: E402
    DELETED_COLUMNS,
    PUBLISHED_COLUMNS,
    SPLIT_RENAME,
    schema_for,
)


def derive(df):
    """Attach the columns that replace the ones about to be deleted."""
    for side in ("mk", "sq"):
        parts = df[f"{side}_component_id"].str.extract(r"_p(\d+)_c(\d+)$")
        if parts[0].isna().any():
            raise ValueError(f"{side}_component_id does not parse as _pNNNN_cNNNN")
        df[f"{side}_page"] = parts[0].astype("int16")
        df[f"{side}_component"] = parts[1].astype("int16")

    issue_no = df["filename"].str.extract(r"_(\d+)\.pdf$", expand=False)
    if issue_no.isna().any():
        raise ValueError("a filename carries no issue number")
    df["issue_no"] = issue_no.astype("int16")
    df["issue_key"] = df["year"].astype(str) + "-" + df["issue_no"].map("{:04d}".format)
    df["pair_id"] = (
        df["issue_key"] + "-"
        + df["sq_page"].map("{:04d}".format) + "-"
        + df["sq_component"].map("{:04d}".format)
    )
    if not df["pair_id"].is_unique:
        raise ValueError(f"pair_id is not unique ({int(df['pair_id'].duplicated().sum()):,} clashes)")

    df["source_pdf"] = df["filename"]
    df["split"] = df["split"].map(SPLIT_RENAME)
    if df["split"].isna().any():
        raise ValueError("an unrecognised split label")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args()

    if args.output.resolve() == args.input.resolve():
        parser.error("refusing to overwrite the input")

    table = pq.read_table(args.input)
    incoming = list(table.column_names)
    print(f"read {args.input}  ({table.num_rows:,} rows x {len(incoming)} columns)")

    df = derive(table.to_pandas())

    names = list(PUBLISHED_COLUMNS)
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise SystemExit(f"cannot build the release: missing {missing}")

    deleted = [c for c in incoming if c not in names]
    unexplained = [c for c in deleted if c not in DELETED_COLUMNS]
    if unexplained:
        raise SystemExit(
            f"refusing to delete columns with no recorded reason: {unexplained}\n"
            f"add them to DELETED_COLUMNS in src/mksq/release.py first"
        )

    print(f"\nkeeping {len(names)}:")
    for n in names:
        print(f"  + {n:<22} {PUBLISHED_COLUMNS[n]}")
    print(f"\ndeleting {len(deleted)}:")
    for c in deleted:
        print(f"  - {c:<22} {DELETED_COLUMNS[c]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    schema = schema_for(names)
    out = pa.Table.from_arrays(
        [pa.array(df[n], schema.field(n).type) for n in names], schema=schema
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, args.output, compression="zstd")

    back = pq.read_table(args.output)
    assert back.num_rows == table.num_rows, "row count changed"
    assert back.column_names == names, "column set changed on the round trip"
    sidecar = args.output.with_suffix(".columns.json")
    sidecar.write_text(
        json.dumps(
            {"kept": PUBLISHED_COLUMNS,
             "deleted": {c: DELETED_COLUMNS[c] for c in deleted}},
            indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}  ({args.output.stat().st_size / 1e6:,.1f} MB)")
    print(f"wrote {sidecar}")
    print(f"round trip OK: {back.num_rows:,} rows, {back.num_columns} columns")


if __name__ == "__main__":
    main()
