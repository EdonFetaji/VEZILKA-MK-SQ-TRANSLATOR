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
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.verbis.gov.mk/api/words"

CONTACT = "wffwefwefwefwefwefwef"  # <- put a real institutional address here
HEADERS = {
    "User-Agent": (
        f"Academic-Research-Crawler/1.0  "
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


# --------------------------------------------------------------------------------------
# Per-category harvest with a raw page cache (v2).
#
#   python verbis_harvest.py --categories 1,2,3,4 --cache-dir CACHE --out OUT.parquet
#   python verbis_harvest.py --categories 1,2,3,4 --cache-dir CACHE --out OUT.parquet --offline
#
# Each category is crawled on its own, so every row's category is known by construction and
# is also checked against the record's own `category.id`. MK and SQ come from the same API
# record, so they cannot shift against each other. Every page response is stored untouched
# in CACHE (cat{N}_page{P}.json + .meta.json); cached pages are never fetched again, and
# --offline rebuilds the output from the cache alone. Rows with an empty field are kept and
# flagged in `missing_fields`, never dropped.
# --------------------------------------------------------------------------------------

V2_COLUMNS = [
    "verbis_id", "category_id", "category_mk", "category_sq",
    "mk", "mk_description", "sq", "sq_description",
    "missing_fields", "status", "created_at", "updated_at",
    "mk_raw", "sq_raw", "mk_description_raw", "sq_description_raw",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_json(session: requests.Session, params: dict) -> tuple[dict, str, int, bytes]:
    for attempt in range(5):
        try:
            r = session.get(BASE, params=params, timeout=60)
        except requests.RequestException as exc:
            print(f"  network error ({exc}), retrying", file=sys.stderr)
            time.sleep(2**attempt)
            continue
        if r.status_code == 200:
            return r.json(), r.url, r.status_code, r.content
        print(f"  HTTP {r.status_code} for {r.url}, retrying", file=sys.stderr)
        time.sleep(2**attempt)
    sys.exit(f"gave up on {params}")


def _cached_page(session, cache: Path, name: str, params: dict, offline: bool, rate: float) -> dict:
    """Return one API page, from cache if present; otherwise fetch it and store it untouched."""
    body_path, meta_path = cache / f"{name}.json", cache / f"{name}.meta.json"
    if body_path.exists() and meta_path.exists():
        return json.loads(body_path.read_bytes())
    if offline:
        sys.exit(f"--offline: {body_path} is not in the cache")
    payload, url, status, content = _get_json(session, params)
    body_path.write_bytes(content)
    meta_path.write_text(json.dumps({
        "request_url": url, "params": params, "http_status": status,
        "retrieved_at_utc": _utc_now(), "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    time.sleep(1.0 / rate)
    return payload


def crawl_categories_cached(categories: list[int], per_page: int, rate: float,
                            cache: Path, offline: bool) -> tuple[list[dict], dict]:
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)
    report: dict = {"categories": {}}

    # Totals without a category filter (category omitted, and category=0), for the cross-check.
    for name, params in (("all_nofilter_total", {"per_page": 1, "page": 1}),
                         ("all_cat0_total", {"per_page": 1, "page": 1, "category": 0})):
        p = _cached_page(session, cache, name, params, offline, rate)
        report[name] = (p.get("meta") or {}).get("total")

    by_id: dict[int, dict] = {}
    for cat in categories:
        page, last_page, total, seen = 1, None, None, 0
        while True:
            params = {"per_page": per_page, "page": page, "category": cat}
            payload = _cached_page(session, cache, f"cat{cat}_page{page:03d}", params, offline, rate)
            meta = payload.get("meta") or {}
            if total is None:
                total, last_page = meta.get("total"), meta.get("last_page")
            data = payload.get("data") or []
            for rec in data:
                rcat = (rec.get("category") or {}).get("id")
                if rcat != cat:
                    sys.exit(f"record id={rec.get('id')} has category {rcat}, requested {cat}")
                rid = rec["id"]
                if rid in by_id and by_id[rid] != rec:
                    sys.exit(f"id {rid} returned twice with different content")
                by_id[rid] = rec
                seen += 1
            print(f"  category {cat}: page {page}/{last_page}  records so far {seen}")
            if not data or (last_page is not None and page >= int(last_page)):
                break
            page += 1
        cat_ids = {r["id"] for r in by_id.values() if r["category"]["id"] == cat}
        report["categories"][cat] = {"api_total": total, "records": len(cat_ids),
                                     "pages": page, "records_incl_repeats": seen}
        if total is not None and len(cat_ids) != int(total):
            sys.exit(f"category {cat}: collected {len(cat_ids)} unique records, API total {total}")

    report["sum_of_categories"] = sum(c["records"] for c in report["categories"].values())
    report["unique_ids"] = len(by_id)
    return [by_id[k] for k in sorted(by_id)], report


def build_v2_table(records: list[dict]):
    import pyarrow as pa

    cols: dict[str, list] = {c: [] for c in V2_COLUMNS}
    for rec in records:
        cat = rec.get("category") or {}
        raw = {k: rec.get(k) for k in ("mkd", "alb", "description_mkd", "description_alb")}
        vals = {
            "mk": clean(raw["mkd"], "cyrl"), "mk_description": clean(raw["description_mkd"], "cyrl"),
            "sq": clean(raw["alb"], "latn"), "sq_description": clean(raw["description_alb"], "latn"),
        }
        cols["verbis_id"].append(int(rec["id"]))
        cols["category_id"].append(int(cat.get("id")))
        cols["category_mk"].append(cat.get("mkd"))
        cols["category_sq"].append(cat.get("alb"))
        for k, v in vals.items():
            cols[k].append(v)
        cols["missing_fields"].append(";".join(k for k, v in vals.items() if not v))
        cols["status"].append(None if rec.get("status") is None else bool(rec.get("status")))
        cols["created_at"].append(rec.get("created_at"))
        cols["updated_at"].append(rec.get("updated_at"))
        cols["mk_raw"].append(raw["mkd"])
        cols["sq_raw"].append(raw["alb"])
        cols["mk_description_raw"].append(raw["description_mkd"])
        cols["sq_description_raw"].append(raw["description_alb"])

    types = {"verbis_id": pa.int64(), "category_id": pa.int16(), "status": pa.bool_()}
    return pa.table({c: pa.array(v, types.get(c, pa.string())) for c, v in cols.items()})


def main_v2(args) -> None:
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    cats = [int(c) for c in args.categories.split(",")]
    cache = Path(args.cache_dir)
    records, report = crawl_categories_cached(cats, args.per_page, args.rate, cache, args.offline)
    table = build_v2_table(records)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", compression_level=3)
    pcsv.write_csv(table, out.with_suffix(".csv"))
    legal = table.filter(pc.equal(table["category_id"], 1))
    legal_path = out.with_name(out.stem + "_legal.parquet")
    pq.write_table(legal, legal_path, compression="zstd", compression_level=3)

    metas = sorted(cache.glob("*.meta.json"))
    times = [json.loads(m.read_text(encoding="utf-8"))["retrieved_at_utc"] for m in metas]
    missing = pc.not_equal(table["missing_fields"], "")
    report.update({
        "source": BASE,
        "crawl_first_utc": min(times) if times else None,
        "crawl_last_utc": max(times) if times else None,
        "per_page": args.per_page,
        "rows": table.num_rows,
        "rows_with_missing_fields": int(pc.sum(missing).as_py() or 0),
        "legal_rows": legal.num_rows,
        "homoglyph_repair": "clean(): each field forced into one script; *_raw keep the API strings",
        "cache_pages": {m.name.removesuffix(".meta.json"): json.loads(m.read_text(encoding="utf-8"))["sha256"]
                        for m in metas},
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (out, out.with_suffix(".csv"), legal_path)},
    })
    (out.parent / "harvest_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out} ({table.num_rows} rows), {legal_path} ({legal.num_rows} rows)")
    for c, r in report["categories"].items():
        print(f"  category {c}: {r['records']} records (API total {r['api_total']})")
    print(f"  sum {report['sum_of_categories']} | unfiltered total {report['all_nofilter_total']}"
          f" | category=0 total {report['all_cat0_total']}")
    print(f"  rows with a missing field: {report['rows_with_missing_fields']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", default=None,
                    help="v2 mode: comma-separated categories, e.g. 1,2,3,4 (needs --cache-dir)")
    ap.add_argument("--cache-dir", default=None, help="v2 mode: directory for raw API pages")
    ap.add_argument("--offline", action="store_true", help="v2 mode: rebuild from --cache-dir only")
    ap.add_argument("--out", default="verbis_mk_sq.parquet")
    ap.add_argument("--category", type=int, default=None, help="1 legal, 2 finance, 3 health, 4 general")
    ap.add_argument("--per-page", type=int, default=1000, help="1000 is the safe ceiling; 5000+ returns a 520")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second")
    ap.add_argument("--keep-empty", action="store_true", help="keep rows missing a term or a definition")
    ap.add_argument("--raw", default=None, help="also dump the untouched API records to this .jsonl")
    args = ap.parse_args()

    if args.categories:
        if not args.cache_dir:
            ap.error("--categories needs --cache-dir")
        main_v2(args)
        return

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