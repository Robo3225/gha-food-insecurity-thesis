"""
======================================================================

build_lags.py

D19 leakage-safe lagged-feature generation.

Generates within-unit lagged columns for a feature (SPI-3 per D19:
lags 1, 3, 6, 9, 12). Each lag is a SEPARATE column so SHAP and TFT
variable selection can identify the operative lag empirically (D19 -
the one-year offset must be discovered, not asserted).

LEAKAGE INVARIANT (D19 s571): a lagged value at time t is the feature
value at t-k for the SAME unit, and NEVER a future value or a value
from another unit. This module enforces that as a hard invariant:
  - lags are computed within-unit (groupby), never across units
  - the shift is strictly positive k (backward in time), never
    negative
  - the first k months of each unit's series are NaN (no fabricated
    history), and this is asserted, not silently filled

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# D19: SPI-3 lag set.
SPI_LAGS = (1, 3, 6, 9, 12)


def add_lags(
    panel: pd.DataFrame,
    value_col: str,
    lags=SPI_LAGS,
    unit_col: str = "FNID",
    time_cols=("year", "month"),
) -> pd.DataFrame:
    """
    Add one column per lag: <value_col>_lag{k}.

    The panel MUST be a contiguous monthly series per unit (no gaps),
    because a plain shift(k) assumes row adjacency equals month
    adjacency. This is verified before shifting; a gap would make a
    shift(k) silently span the wrong interval.
    """

    df = panel.sort_values([unit_col, *time_cols]).reset_index(drop=True)

    # --- invariant 1: contiguous monthly series per unit -------------
    _assert_contiguous(df, unit_col, time_cols)

    # --- generate lags strictly within unit, strictly backward -------
    for k in lags:
        if k <= 0:
            raise ValueError(f"Lag {k} is not positive; a lag must look "
                             f"backward (t-k), never forward.")
        df[f"{value_col}_lag{k}"] = (
            df.groupby(unit_col)[value_col].shift(k)
        )

    # --- invariant 2: first k rows per unit are NaN (no fabrication) --
    _assert_lag_edges(df, value_col, lags, unit_col)

    return df


def _assert_contiguous(df, unit_col, time_cols):
    """Every unit's series must be consecutive months with no gaps."""
    d = df.copy()
    d["_ym"] = d[time_cols[0]] * 12 + (d[time_cols[1]] - 1)
    diffs = d.groupby(unit_col)["_ym"].diff().dropna()
    bad = diffs[diffs != 1]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} non-contiguous month steps found. A lag shift "
            f"assumes contiguous months; gaps would make shift(k) span "
            f"the wrong interval. Fill or investigate the series first."
        )


def _assert_lag_edges(df, value_col, lags, unit_col):
    """
    For each unit, the first k values of lag{k} must be NaN (no history
    to look back to) and the rest must be non-fabricated. This catches a
    shift that wrapped across a unit boundary (which would put a real
    value where a NaN belongs).
    """
    for k in lags:
        col = f"{value_col}_lag{k}"
        # count leading NaNs per unit for this lag
        def leading_nan(s):
            arr = s.values
            n = 0
            for v in arr:
                if pd.isna(v):
                    n += 1
                else:
                    break
            return n
        per_unit_leading = df.groupby(unit_col)[col].apply(leading_nan)
        # every unit with >= k rows must have EXACTLY k leading NaNs;
        # units shorter than k are entirely NaN.
        unit_len = df.groupby(unit_col).size()
        expected = unit_len.clip(upper=k)
        mismatch = per_unit_leading[per_unit_leading != expected]
        if len(mismatch):
            raise RuntimeError(
                f"Lag {k} ({col}): {len(mismatch)} units have the wrong "
                f"number of leading NaNs. A shift may have leaked across a "
                f"unit boundary (cross-unit contamination)."
            )
