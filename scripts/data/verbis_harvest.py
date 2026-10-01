#!/usr/bin/env python3
"""
verbis_dataset.py — build the MK–SQ term/definition dataset in one command.

    python verbis_dataset.py
    python verbis_dataset.py --out data/verbis_mk_sq.parquet --category 1

Output: a Parquet file with exactly four columns, one row per dictionary entry.

    mk              Macedonian term
    mk_description  Macedonian definition
    sq              Albanian term
    sq_description  Albanian definition

Source: https://api.verbis.gov.mk/api/words  (APJ / Agjencia e Zbatimit të Gjuhës)
10,241 entries total; 11 requests at per_page=1000.

Categories: 1 legal-administrative · 2 finance-economic · 3 health · 4 general.
Omit --category for all of them.

Redistribution of the result requires the Agency's permission. Research use and
redistribution are different questions; this script only does the first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.verbis.gov.mk/api/words"

CONTACT = "fetaji.ed@gmail.com"  # <- put a real institutional address here
HEADERS = {
    "User-Agent": (
        f"UNT-Research-Crawler/1.0 (Mother Teresa University, Skopje; "
        f"academic MT research; {CONTACT})"
    ),
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.verbis.gov.mk/",
    "Origin": "https://www.verbis.gov.mk",
}

# The live data mixes scripts: the Albanian headword "АGREGATOR" opens with a
# Cyrillic А (U+0410), not a Latin A. Left alone this quietly breaks tokenisation
# and every string match downstream, so each field is forced into one script.
CYR_TO_LAT = str.maketrans("АВЕКМНОРСТУХаеорсухіјѕ", "ABEKMHOPCTYXaeopcyxijs")
LAT_TO_CYR = str.maketrans("ABEKMHOPCTYXaeopcyx", "АВЕКМНОРСТУХаеорсух")


def clean(text: object, script: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\xa0", " ").strip()
    s = " ".join(s.split())
    return s.translate(LAT_TO_CYR if script == "cyrl" else CYR_TO_LAT)


def crawl(category: int | None, per_page: int, rate: float) -> list[dict]:
    session = requests.Session()
    session.headers.update(HEADERS)

    rows: dict[int, dict] = {}
    page, last_page, total = 1, None, None

    while True:
        params: dict[str, object] = {"per_page": per_page, "page": page}
        if category is not None:
            params["category"] = category

        for attempt in range(5):
            try:
                r = session.get(BASE, params=params, timeout=60)
            except requests.RequestException as exc:
                print(f"  network error ({exc}), retrying", file=sys.stderr)
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                break
            print(f"  HTTP {r.status_code}, retrying", file=sys.stderr)
            time.sleep(2**attempt)
        else:
            sys.exit(f"gave up on page {page}")

        payload = r.json()
        data = payload.get("data", [])
        meta = payload.get("meta", {})
        if total is None:
            total = meta.get("total")
            last_page = meta.get("last_page")
            print(f"API reports total={total}, last_page={last_page}")

        if not data:
            break
        for rec in data:
            rid = rec.get("id")
            if rid is not None:
                rows[rid] = rec

        print(f"  page {page}/{last_page}  held {len(rows)}")

        if last_page is not None and page >= int(last_page):
            break
        if not (payload.get("links") or {}).get("next"):
            break
        page += 1
        time.sleep(1.0 / rate)

    if total is not None and len(rows) < int(total):
        print(
            f"WARNING: collected {len(rows)} of {total} reported entries",
            file=sys.stderr,
        )
    return list(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="verbis_mk_sq.parquet")
    ap.add_argument("--category", type=int, default=None, help="1 legal, 2 finance, 3 health, 4 general")
    ap.add_argument("--per-page", type=int, default=1000, help="1000 is the safe ceiling; 5000+ returns a 520")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second")
    ap.add_argument("--keep-empty", action="store_true", help="keep rows missing a term or a definition")
    ap.add_argument("--raw", default=None, help="also dump the untouched API records to this .jsonl")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    records = crawl(args.category, args.per_page, args.rate)
    print(f"\n{len(records)} unique records retrieved")

    if args.raw:
        Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
        with open(args.raw, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"raw records -> {args.raw}")

    mk, mk_d, sq, sq_d = [], [], [], []
    dropped = 0
    for rec in records:
        a = clean(rec.get("mkd"), "cyrl")
        b = clean(rec.get("description_mkd"), "cyrl")
        c = clean(rec.get("alb"), "latn")
        d = clean(rec.get("description_alb"), "latn")
        if not args.keep_empty and not (a and b and c and d):
            dropped += 1
            continue
        mk.append(a)
        mk_d.append(b)
        sq.append(c)
        sq_d.append(d)

    table = pa.table(
        {
            "mk": pa.array(mk, pa.string()),
            "mk_description": pa.array(mk_d, pa.string()),
            "sq": pa.array(sq, pa.string()),
            "sq_description": pa.array(sq_d, pa.string()),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", compression_level=3)

    print(f"\nwrote {out}")
    print(f"  rows      {table.num_rows}")
    print(f"  dropped   {dropped} (incomplete)")
    print(f"  columns   {table.column_names}")
    print(f"  size      {out.stat().st_size / 1024:.1f} KiB")

    if table.num_rows:
        print("\nfirst row:")
        for col in table.column_names:
            val = str(table[col][0])
            print(f"  {col:<16} {val[:110]}{'…' if len(val) > 110 else ''}")


if __name__ == "__main__":
    main()