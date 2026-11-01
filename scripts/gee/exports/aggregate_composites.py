"""
======================================================================

aggregate_composites.py

Purpose
-------
Aggregate composite-level GEE exports (one row per reporting unit per
16-day composite) into a monthly feature panel.

This is the LOCAL half of the composite-resolution extraction design.
The GEE side exports raw composites; this module assigns each composite
to a calendar month and reduces to a monthly value, RECORDING how many
composites contributed to each month.

Why composite-resolution rather than monthly-in-GEE
---------------------------------------------------
MOD13Q1 is a 16-day composite product. Filtering a calendar month
yields 2 composites usually, sometimes 3, depending on where the cycle
lands. Pre-aggregating inside GEE hides that variance. Aggregating here
makes it an auditable column (`n_composites`) so the inconsistent
temporal support can be quantified and reported as a methodology point
rather than silently absorbed.

Composite-to-month assignment
-----------------------------
A composite is assigned to the month of its START date (the
`system:time_start` GEE exports). This matches the convention used when
filtering by calendar month and is documented in the methodology. A
composite may physically span a month boundary; assignment by start
date is the declared, consistent rule.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import pandas as pd


def aggregate_to_monthly(
    composites: pd.DataFrame,
    value_col: str,
    date_col: str = "composite_date",
    unit_col: str = "FNID",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate composite-level rows to a monthly panel.

    Parameters
    ----------
    composites : DataFrame
        One row per (unit, composite). Must contain unit_col, date_col,
        value_col.
    value_col : str
        The feature value column (e.g. "ndvi").
    date_col : str
        Composite start-date column (parseable to datetime).
    unit_col : str
        Reporting-unit identifier column.
    agg : str
        Monthly reduction. "mean" for NDVI/LST (average of composites);
        "sum" is NOT used here - rainfall is summed inside GEE at daily
        resolution, not composite-aggregated locally.

    Returns
    -------
    DataFrame with columns:
        unit_col, year, month, value_col, n_composites, n_valid
    where n_composites is how many composites fell in the month and
    n_valid is how many had a non-null value.
    """

    df = composites.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        n = df[date_col].isna().sum()
        raise ValueError(f"{n} composite rows have unparseable {date_col}.")

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month

    grouped = df.groupby([unit_col, "year", "month"])

    out = grouped.agg(
        **{
            value_col: (value_col, agg),
            "n_composites": (value_col, "size"),
            "n_valid": (value_col, "count"),   # count excludes NaN
        }
    ).reset_index()

    return out


def report_temporal_support(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the composites-per-month distribution - the temporal
    support variance the methodology needs to document.
    """

    dist = (
        monthly["n_composites"]
        .value_counts()
        .sort_index()
        .rename_axis("composites_in_month")
        .reset_index(name="unit_months")
    )
    return dist


def assert_panel_complete(
    monthly: pd.DataFrame,
    expected_units: int,
    start: str,
    end: str,
    unit_col: str = "FNID",
) -> None:
    """
    Assert the monthly panel is a complete units x months grid and flag
    months with zero valid composites for any unit.
    """

    months = pd.period_range(start=start, end=end, freq="M")
    expected_rows = expected_units * len(months)

    actual = len(monthly)
    if actual != expected_rows:
        raise RuntimeError(
            f"Panel has {actual:,} rows, expected "
            f"{expected_rows:,} ({expected_units} units x "
            f"{len(months)} months). Missing unit-months indicate "
            f"composites absent for some months - investigate before "
            f"the panel reaches the model."
        )

    zero_valid = (monthly["n_valid"] == 0).sum()
    if zero_valid:
        print(
            f"\u26a0 {zero_valid:,} unit-months have zero valid "
            f"composites (fully masked). Values are null and will need "
            f"handling downstream."
        )


# ---------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------

def interpolate_masked(
    monthly: pd.DataFrame,
    value_col: str,
    unit_col: str = "FNID",
    limit: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill fully-masked unit-months (n_valid == 0, value is NaN) by
    linear interpolation WITHIN each unit's own time series.

    NDVI varies smoothly month to month within a reporting unit, so a
    single interpolated month is a defensible, literature-standard fill.
    Interpolation is strictly within-unit (a unit is never filled from
    another unit) and limited to runs of at most `limit` consecutive
    months, so a genuine multi-month gap is NOT silently bridged.

    Every filled cell is flagged in a new boolean column
    `<value_col>_interpolated` so interpolated values remain auditable
    and can be excluded from evaluation.

    Returns
    -------
    (filled, report)
        filled  : the monthly panel with NaNs interpolated where
                  possible and the interpolated-flag column added.
        report  : one row per originally-masked unit-month, recording
                  FNID, year, month, whether it was filled, and the
                  filled value. This is the C3 data-quality artifact.
    """

    df = monthly.copy()
    flag_col = f"{value_col}_interpolated"

    # Order within each unit by calendar time so interpolation follows
    # the real sequence, not row order.
    df = df.sort_values([unit_col, "year", "month"]).reset_index(drop=True)

    was_missing = df[value_col].isna()

    # Interpolate within each unit, capped at `limit` consecutive NaNs.
    df[value_col] = (
        df.groupby(unit_col, group_keys=False)[value_col]
          .apply(lambda s: s.interpolate(
              method="linear", limit=limit, limit_area="inside"
          ))
    )

    now_present = df[value_col].notna()
    df[flag_col] = was_missing & now_present

    # Report every originally-masked cell and its resolution.
    report = df.loc[was_missing, [unit_col, "year", "month", value_col, flag_col]].copy()
    report = report.rename(columns={
        value_col: f"{value_col}_filled",
        flag_col: "interpolated",
    })
    report["still_missing"] = report[f"{value_col}_filled"].isna()

    return df, report


def assert_no_nulls(
    monthly: pd.DataFrame,
    value_col: str,
    unit_col: str = "FNID",
) -> None:
    """
    Assert the feature column has no remaining nulls. Raises with the
    offending unit-months if any survive (e.g. a gap longer than the
    interpolation limit, or a leading/trailing gap interpolation cannot
    reach).
    """

    remaining = monthly[monthly[value_col].isna()]
    if len(remaining):
        sample = remaining[[unit_col, "year", "month"]].head(10)
        raise RuntimeError(
            f"{len(remaining):,} unit-months still null in "
            f"'{value_col}' after interpolation. First rows:\n{sample}\n"
            f"These are gaps interpolation could not fill (too long, or "
            f"at a series edge). Decide handling explicitly before the "
            f"panel reaches the model."
        )
