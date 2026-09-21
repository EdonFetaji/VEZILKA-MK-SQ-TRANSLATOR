#!/usr/bin/env python
"""D0 preparation: chronological split, cross-split near-duplicate sweep, token lengths.

    python scripts/prepare_corpus.py configs/prepare_corpus_v1.yaml

Reads the raw pair table, never writes to it. Writes a Hive-partitioned corpus
under ``output.corpus_dir``, a sidecar of every segment the near-duplicate sweep
removed, and a report directory carrying the resolved config, library versions
and every number that belongs in the paper's corpus section.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mksq import chronology, config as config_module, neardup, schema, tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(config_path: str) -> None:
    started = time.time()
    config = config_module.load(config_path)
    report_dir = ROOT / config["output"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(report_dir / "run.log")

    log(f"run {config.run_name}")
    (report_dir / "provenance.json").write_text(
        json.dumps(config_module.provenance(config), indent=2, ensure_ascii=False)
    )

    # --- 1. read the raw table, unchanged --------------------------------------
    raw_path = ROOT / config["input"]["path"]
    table = pq.read_table(raw_path)
    log(f"read {raw_path.name}: {table.num_rows:,} rows x {table.num_columns} columns")
    expected = config["input"].get("expected_rows")
    if expected is not None and table.num_rows != expected:
        log(f"WARNING: expected {expected:,} rows, got {table.num_rows:,}")
    table = schema.conform(table, schema.RAW_SCHEMA)
    df = table.to_pandas()
    for side in ("mk_text", "sq_text"):
        df[side] = df[side].fillna("")

    # --- 2. chronology and the chronological split (C3) -------------------------
    df = chronology.add_chronology(df)
    log(f"chronology: {df['issue_key'].nunique():,} issues, "
        f"{df['year'].min()}-{df['year'].max()}, order verified against month")

    old_split = df["split"].copy()
    df, boundaries = chronology.chronological_split(
        df,
        train_frac=config["split"]["train_frac"],
        dev_frac=config["split"]["dev_frac"],
        test_frac=config["split"]["test_frac"],
    )
    log("\nchronological split (issue-disjoint, oldest -> newest):\n" + boundaries.to_string())
    agreement = float((old_split == df["split"]).mean())
    log(f"\nrows whose split is unchanged from the old hash split: {agreement:.1%}")
    boundaries.to_csv(report_dir / "split_boundaries.csv")

    # --- 3. cross-split near-duplicate sweep (C4) -------------------------------
    df["near_dup_max_jaccard"] = np.nan
    dropped = pd.DataFrame()
    if config["near_dup"]["enabled"]:
        df, dropped, sweep_report = run_sweep(df, config, log)
        (report_dir / "near_dup.json").write_text(
            json.dumps(sweep_report, indent=2, ensure_ascii=False)
        )

    # --- 4. NLLB token lengths ---------------------------------------------------
    if config["tokens"]["enabled"]:
        settings = config["tokens"]
        tokens.assert_language_codes(settings["model_name"])
        df["mk_nllb_tokens"] = tokens.token_lengths(
            df["mk_text"], model_name=settings["model_name"],
            lang=settings["mk_lang"], batch_size=settings["batch_size"],
        ).astype("int32")
        df["sq_nllb_tokens"] = tokens.token_lengths(
            df["sq_text"], model_name=settings["model_name"],
            lang=settings["sq_lang"], batch_size=settings["batch_size"],
        ).astype("int32")
        log(f"\nNLLB token lengths ({settings['model_name']}):")
        log(df[["mk_nllb_tokens", "sq_nllb_tokens"]]
            .quantile([0.5, 0.9, 0.95, 0.99, 1.0]).round(1).to_string())
    else:
        df["mk_nllb_tokens"] = np.int32(-1)
        df["sq_nllb_tokens"] = np.int32(-1)

    # --- 5. write ----------------------------------------------------------------
    out = schema.conform(pa.Table.from_pandas(df, preserve_index=False), schema.CORPUS_SCHEMA)
    corpus_dir = ROOT / config["output"]["corpus_dir"]
    if corpus_dir.exists():
        import shutil

        shutil.rmtree(corpus_dir)
    pq.write_to_dataset(
        out,
        root_path=str(corpus_dir),
        partition_cols=schema.PARTITION_COLUMNS,
        compression=config["output"]["compression"],
        existing_data_behavior="overwrite_or_ignore",
    )
    log(f"\nwrote {out.num_rows:,} rows to {corpus_dir}"
        f" partitioned by {'/'.join(schema.PARTITION_COLUMNS)}")

    if len(dropped):
        dropped_table = schema.conform(
            pa.Table.from_pandas(dropped, preserve_index=False), schema.DROPPED_SCHEMA
        )
        dropped_path = ROOT / config["output"]["dropped_path"]
        dropped_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(dropped_table, dropped_path, compression=config["output"]["compression"])
        log(f"wrote {len(dropped):,} dropped segments to {dropped_path}")

    verify(corpus_dir, out, log)
    write_corpus_stats(df, report_dir, log)
    log(f"\ndone in {time.time() - started:,.0f}s")
    log.close()


def run_sweep(df: pd.DataFrame, config, log):
    """Drop every dev/test pair whose mk or sq side near-duplicates a train pair."""
    settings = config["near_dup"]
    key = df["sq_component_id"]
    if not key.is_unique:
        raise ValueError("sq_component_id is not unique -- it cannot key the sweep")

    is_train = df["split"] == "train"
    train_index = pd.Index(key[is_train])
    eval_index = pd.Index(key[~is_train])
    log(f"\nnear-duplicate sweep: {len(train_index):,} train vs {len(eval_index):,} dev+test, "
        f"char {settings['ngram']}-grams, Jaccard >= {settings['threshold']}, "
        f"{settings['num_perm']} permutations")

    results, flags, report = {}, {}, {"threshold": settings["threshold"],
                                      "num_perm": settings["num_perm"],
                                      "ngram": settings["ngram"], "sides": {}}
    for side in settings["sides"]:
        column = f"{side}_text"
        result = neardup.sweep_side(
            pd.Series(df.loc[is_train, column].to_numpy(), index=train_index),
            pd.Series(df.loc[~is_train, column].to_numpy(), index=eval_index),
            side=side,
            threshold=settings["threshold"],
            num_perm=settings["num_perm"],
            ngram=settings["ngram"],
        )
        results[side] = result
        flags[side] = result.max_jaccard >= settings["threshold"]
        log(f"  {side}: {result.matched:,} of {result.queried:,} dev/test segments "
            f"({result.matched / max(result.queried, 1):.2%}) near-duplicate a train segment")
        report["sides"][side] = {
            "train_indexed": result.train_indexed,
            "queried": result.queried,
            "matched": result.matched,
            "matched_share": result.matched / max(result.queried, 1),
        }

    any_side = pd.concat(flags.values(), axis=1).any(axis=1)
    max_jaccard = pd.concat([r.max_jaccard for r in results.values()], axis=1).max(axis=1)
    side_label = pd.Series(
        np.select(
            [flags[s] & ~pd.concat([flags[o] for o in flags if o != s], axis=1).any(axis=1)
             for s in flags],
            list(flags),
            default="both",
        ),
        index=any_side.index,
    ).where(any_side, None)

    keyed = df.set_index(key, drop=False)
    keyed["near_dup_max_jaccard"] = max_jaccard
    drop_mask = keyed.index.isin(any_side[any_side].index)

    dropped = keyed.loc[drop_mask].copy()
    dropped["side"] = side_label.reindex(dropped.index)
    dropped["train_match_sq_component_id"] = _best_match_key(results, flags, dropped.index)
    train_by_key = df.set_index(key)
    dropped["train_match_mk_text"] = dropped["train_match_sq_component_id"].map(train_by_key["mk_text"])
    dropped["train_match_sq_text"] = dropped["train_match_sq_component_id"].map(train_by_key["sq_text"])

    kept = keyed.loc[~drop_mask].reset_index(drop=True)
    per_split = dropped.groupby("split").size()
    log(f"  dropped from dev/test: {len(dropped):,} segments "
        f"({len(dropped) / max(len(eval_index), 1):.2%} of dev+test, "
        f"{len(dropped) / len(df):.2%} of the corpus)")
    log("  " + per_split.to_string().replace("\n", "\n  "))
    report["dropped_total"] = int(len(dropped))
    report["dropped_per_split"] = {k: int(v) for k, v in per_split.items()}
    report["eval_rows_before"] = int(len(eval_index))
    report["eval_rows_after"] = int(len(eval_index) - len(dropped))
    return kept, dropped.reset_index(drop=True), report


def _best_match_key(results, flags, index) -> pd.Series:
    """The train key behind the highest-scoring side, per dropped row."""
    stacked = pd.concat({s: r.max_jaccard for s, r in results.items()}, axis=1).reindex(index)
    winner = stacked.idxmax(axis=1)
    keys = pd.concat({s: r.match_key for s, r in results.items()}, axis=1).reindex(index)
    return pd.Series(
        [keys.loc[i, winner.loc[i]] if pd.notna(winner.loc[i]) else None for i in index],
        index=index, dtype="object",
    )


def verify(corpus_dir: Path, written: pa.Table, log) -> None:
    """Re-read what was written and check it is what we think it is."""
    back = pq.read_table(corpus_dir)
    if back.num_rows != written.num_rows:
        raise AssertionError(f"round trip: {back.num_rows:,} rows, wrote {written.num_rows:,}")
    df = back.to_pandas()
    if df.groupby("issue_key")["split"].nunique().max() > 1:
        raise AssertionError("round trip: an issue spans two splits")
    bounds = df.groupby("split")["issue_order"].agg(["min", "max"])
    for earlier, later in (("train", "dev"), ("dev", "test")):
        if bounds.loc[earlier, "max"] >= bounds.loc[later, "min"]:
            raise AssertionError(f"round trip: {earlier} and {later} overlap in time")
    if df["dup_key"].duplicated().any():
        raise AssertionError("round trip: an exact duplicate survived")
    log(f"round trip OK: {back.num_rows:,} rows, {back.num_columns} columns, "
        f"{len(list(corpus_dir.rglob('*.parquet')))} partition files")


def write_corpus_stats(df: pd.DataFrame, report_dir: Path, log) -> None:
    """The numbers that go straight into the paper's corpus table."""
    per_split = df.groupby("split").agg(
        pairs=("split", "size"),
        issues=("issue_key", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        mk_tokens_median=("mk_nllb_tokens", "median"),
        sq_tokens_median=("sq_nllb_tokens", "median"),
        mk_tokens_p95=("mk_nllb_tokens", lambda s: s.quantile(0.95)),
        sq_tokens_p95=("sq_nllb_tokens", lambda s: s.quantile(0.95)),
        high_quality=("is_high_quality", "sum"),
        mean_quality=("quality_confidence", "mean"),
    ).reindex(["train", "dev", "test"])
    per_split["hq_rate"] = (per_split["high_quality"] / per_split["pairs"]).round(3)
    per_split.round(3).to_csv(report_dir / "corpus_stats_by_split.csv")

    per_year = pd.crosstab(df["year"], df["split"]).reindex(columns=["train", "dev", "test"], fill_value=0)
    per_year.to_csv(report_dir / "corpus_stats_by_year.csv")

    log("\ncorpus statistics by split:\n" + per_split.round(3).to_string())


class Logger:
    def __init__(self, path: Path):
        self.handle = path.open("w")

    def __call__(self, message: str) -> None:
        print(message)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
