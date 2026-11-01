"""
======================================================================

build_climatology.py

D18 stage 1: reduce a long baseline monthly series to a per-unit,
per-calendar-month climatology lookup.

Two climatology types (D18):
  - ANOMALY features (CHIRPS rainfall, LST): per unit, per calendar
    month -> long-term MEAN and standard deviation. A modelling-window
    value is later expressed as (value - mean) / sd, a standardized
    anomaly.
  - VCI (from NDVI): per unit, per calendar month -> 5th and 95th
    PERCENTILES over the long record. A modelling-window NDVI is later
    normalized to [0,1] against that robust range (NOT raw min/max,
    which is outlier-sensitive - D18).

The long baseline series is the INPUT (a downloaded monthly export over
the D18 baseline window: CHIRPS 1991-2020, MODIS ~2000-present). This
module reduces it to the lookup and DISCARDS the long series; only the
compact climatology is retained (D18).

Baseline windows differ by product availability, not choice (D18):
CHIRPS supports 1991-2020; MODIS begins ~2000-2001. Each feature's
climatology is built over its own window.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_anomaly_climatology(
    baseline_monthly: pd.DataFrame,
    value_col: str,
    unit_col: str = "FNID",
    baseline_label: str = "",
) -> pd.DataFrame:
    """
    Per-unit, per-calendar-month mean and SD over the baseline series.

    Returns a lookup with columns:
        unit_col, month, clim_mean, clim_sd, n_years
    n_years records how many baseline observations informed each cell -
    a per-cell data-quality figure worth retaining (D18 notes baselines
    differ in length by product).
    """

    df = baseline_monthly.copy()
    if "month" not in df.columns:
        raise ValueError("baseline_monthly must have a 'month' column.")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    grouped = df.groupby([unit_col, "month"])[value_col]
    clim = grouped.agg(
        clim_mean="mean",
        clim_sd=lambda s: s.std(ddof=0),
        n_years="count",
    ).reset_index()

    clim["baseline"] = baseline_label
    return clim


def build_vci_climatology(
    baseline_monthly: pd.DataFrame,
    value_col: str = "ndvi",
    unit_col: str = "FNID",
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
    baseline_label: str = "",
) -> pd.DataFrame:
    """
    Per-unit, per-calendar-month robust percentile range for VCI.

    Returns a lookup with columns:
        unit_col, month, ndvi_lo, ndvi_hi, n_years
    where ndvi_lo/ndvi_hi are the 5th/95th percentiles (D18: robust to
    outliers / residual cloud contamination, unlike raw min/max).
    """

    df = baseline_monthly.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    grouped = df.groupby([unit_col, "month"])[value_col]
    clim = grouped.agg(
        ndvi_lo=lambda s: np.nanpercentile(s.dropna(), lo_pct) if s.notna().any() else np.nan,
        ndvi_hi=lambda s: np.nanpercentile(s.dropna(), hi_pct) if s.notna().any() else np.nan,
        n_years="count",
    ).reset_index()

    clim["baseline"] = baseline_label
    return clim


# ---------------------------------------------------------------------
# Applying a climatology to the modelling window
# ---------------------------------------------------------------------

def apply_anomaly(
    panel: pd.DataFrame,
    climatology: pd.DataFrame,
    value_col: str,
    out_col: str,
    unit_col: str = "FNID",
    min_sd: float = 1e-6,
) -> pd.DataFrame:
    """
    Standardized anomaly = (value - clim_mean) / clim_sd, per unit per
    calendar month. Where clim_sd is ~0 (a unit-month with no baseline
    variation), the anomaly is set to 0 rather than dividing by zero.
    """

    df = panel.merge(
        climatology[[unit_col, "month", "clim_mean", "clim_sd"]],
        on=[unit_col, "month"], how="left", validate="many_to_one",
    )

    missing = df["clim_mean"].isna().sum()
    if missing:
        raise RuntimeError(
            f"{missing} panel rows have no climatology entry. Every "
            f"unit-month must have a baseline; investigate coverage."
        )

    # Distinguish the two NaN sources so a cloudy (null-input) cell is
    # NOT silently turned into a real anomaly:
    #   - flat-baseline cell  (clim_sd ~ 0, value present): anomaly = 0,
    #     because there is no variation to standardize against.
    #   - null-input cell     (value is NaN, e.g. QC-masked cloud): the
    #     anomaly STAYS NaN, to be handled downstream (D11 discipline),
    #     never fabricated as 0.
    value_present = df[value_col].notna()
    flat_baseline = df["clim_sd"] <= min_sd

    sd = df["clim_sd"].where(df["clim_sd"] > min_sd, np.nan)
    df[out_col] = (df[value_col] - df["clim_mean"]) / sd

    # zero ONLY the flat-baseline cells that have a real input value.
    df.loc[flat_baseline & value_present, out_col] = 0.0
    # null-input cells remain NaN (do not fill).

    n_null_input = int((~value_present).sum())
    if n_null_input:
        print(f"  [apply_anomaly] {n_null_input} null-input (e.g. cloudy) "
              f"cells kept NaN for downstream handling, not zero-filled.")

    return df.drop(columns=["clim_mean", "clim_sd"])


def apply_vci(
    panel: pd.DataFrame,
    climatology: pd.DataFrame,
    value_col: str = "ndvi",
    out_col: str = "vci",
    unit_col: str = "FNID",
) -> pd.DataFrame:
    """
    VCI = (NDVI - lo) / (hi - lo), clipped to [0, 1], per unit per
    calendar month against the robust percentile range.
    """

    df = panel.merge(
        climatology[[unit_col, "month", "ndvi_lo", "ndvi_hi"]],
        on=[unit_col, "month"], how="left", validate="many_to_one",
    )

    missing = df["ndvi_lo"].isna().sum()
    if missing:
        raise RuntimeError(
            f"{missing} panel rows have no VCI climatology entry."
        )

    span = (df["ndvi_hi"] - df["ndvi_lo"]).where(
        lambda s: s > 1e-6, np.nan
    )
    df[out_col] = ((df[value_col] - df["ndvi_lo"]) / span).clip(0.0, 1.0)
    df[out_col] = df[out_col].fillna(0.5)   # degenerate range -> neutral

    return df.drop(columns=["ndvi_lo", "ndvi_hi"])
