"""Chronology recovery and the chronological, issue-disjoint split (constraint C3).

The pair table has no date column. What it has is enough:

* ``filename``       ``<md5>__СЛУЖБЕН_ВЕСНИК_НА_Р(С)М_<issue_no>.pdf``
* ``slv_identifier`` ``<year>-<month>-<rank>`` -- ``rank`` is a creation-time rank
  inside the month folder and is **not** stable, but ``year`` and ``month`` are.

``(year, issue_no)`` is unique over all issues and, verified against the
independent ``month`` field, sorting by it produces zero month inversions in
every year of the corpus. That makes it a sound chronological order even though
no day-level date is recoverable.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ISSUE_NO_RE = re.compile(r"_(\d+)\.pdf$")
IDENTIFIER_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d+)$")


def add_chronology(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``issue_no``, ``issue_month``, ``issue_key`` and ``issue_order``.

    Raises if either field fails to parse, or if the recovered order contradicts
    the month field -- a silent chronology error would invalidate every split
    built on top of it.
    """
    out = df.copy()

    issue_no = out["filename"].str.extract(ISSUE_NO_RE, expand=False)
    if issue_no.isna().any():
        bad = out.loc[issue_no.isna(), "filename"].unique()[:5]
        raise ValueError(f"{issue_no.isna().sum():,} rows: no issue number in filename, e.g. {list(bad)}")
    out["issue_no"] = issue_no.astype("int32")

    parts = out["slv_identifier"].str.extract(IDENTIFIER_RE)
    if parts[0].isna().any():
        bad = out.loc[parts[0].isna(), "slv_identifier"].unique()[:5]
        raise ValueError(f"slv_identifier not <year>-<month>-<rank>, e.g. {list(bad)}")
    if not (parts[0].astype("int16") == out["year"]).all():
        raise ValueError("year column disagrees with the year in slv_identifier")
    out["issue_month"] = parts[1].astype("int8")

    out["issue_key"] = out["year"].astype(str) + "-" + out["issue_no"].map("{:04d}".format)

    issues = (
        out[["issue_key", "year", "issue_no", "issue_month"]]
        .drop_duplicates("issue_key")
        .sort_values(["year", "issue_no"])
        .reset_index(drop=True)
    )
    if issues["issue_key"].duplicated().any():
        raise ValueError("issue_key is not unique per (year, issue_no)")
    _assert_month_monotone(issues)

    order = pd.Series(issues.index.to_numpy("int32"), index=issues["issue_key"])
    out["issue_order"] = out["issue_key"].map(order).astype("int32")
    return out


def _assert_month_monotone(issues: pd.DataFrame) -> None:
    """Within each year, issue_no must not run backwards against the month."""
    for year, group in issues.groupby("year"):
        months = group.sort_values("issue_no")["issue_month"].to_numpy()
        # pairs i < j where the later issue number carries the earlier month
        inversions = int(sum(np.sum(months[i + 1:] < months[i]) for i in range(len(months))))
        if inversions:
            raise ValueError(
                f"{year}: {inversions} month inversions when ordering by issue_no -- "
                "issue numbers are not chronological in this year"
            )


def chronological_split(
    df: pd.DataFrame,
    train_frac: float,
    dev_frac: float,
    test_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign train/dev/test by walking issues oldest-first (C3).

    Boundaries fall on issue boundaries, so an issue never spans two splits, and
    the cut points are chosen on the cumulative *pair* count rather than the
    issue count -- issues vary in size by two orders of magnitude.

    Returns ``(df_with_split, boundaries)``.
    """
    total = train_frac + dev_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"fractions sum to {total}, not 1.0")

    per_issue = (
        df.groupby("issue_order", sort=True)
        .agg(issue_key=("issue_key", "first"), year=("year", "first"), pairs=("issue_order", "size"))
        .reset_index()
        .sort_values("issue_order")
    )
    cumulative = per_issue["pairs"].cumsum() / len(df)

    # An issue belongs to the first split whose cumulative ceiling it has not passed.
    train_end = train_frac
    dev_end = train_frac + dev_frac
    split_of_issue = np.where(
        cumulative.shift(fill_value=0.0) < train_end,
        "train",
        np.where(cumulative.shift(fill_value=0.0) < dev_end, "dev", "test"),
    )
    per_issue["split"] = split_of_issue

    out = df.copy()
    out["split"] = out["issue_order"].map(
        pd.Series(per_issue["split"].to_numpy(), index=per_issue["issue_order"])
    )
    if out["split"].isna().any():
        raise ValueError("an issue was not assigned a split")
    if out.groupby("issue_key")["split"].nunique().max() > 1:
        raise ValueError("an issue landed in two splits")
    _assert_ordered(out)

    boundaries = (
        per_issue.groupby("split")
        .agg(
            issues=("issue_key", "size"),
            pairs=("pairs", "sum"),
            first_issue=("issue_key", "first"),
            last_issue=("issue_key", "last"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .reindex(["train", "dev", "test"])
    )
    boundaries["pct_of_pairs"] = (boundaries["pairs"] / len(df)).round(4)
    return out, boundaries


def _assert_ordered(df: pd.DataFrame) -> None:
    """Every train issue precedes every dev issue, which precedes every test issue."""
    bounds = df.groupby("split")["issue_order"].agg(["min", "max"])
    for earlier, later in (("train", "dev"), ("dev", "test")):
        if earlier in bounds.index and later in bounds.index:
            if bounds.loc[earlier, "max"] >= bounds.loc[later, "min"]:
                raise ValueError(f"{earlier} and {later} overlap in time")
