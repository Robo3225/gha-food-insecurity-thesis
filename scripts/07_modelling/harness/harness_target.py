"""
======================================================================

harness_target.py  (M10.1 — horizon-shifted target construction)

THE foundational correctness piece. A feature row at (unit, t) predicts
the outcome at (unit, t+horizon). The label is therefore phase3plus at
t+horizon, NOT the row's own phase3plus at t. Scoring against the row's
own phase would be target-leakage (the target scoring itself -> PR-AUC
1.0), the exact invalid-but-beautiful result M10 exists to prevent.

This module builds, per horizon, a within-unit forward-shifted target:

    y_h(unit, t) = phase3plus(unit, t + h)

with leakage-safe construction (within-unit, contiguous-time asserted)
and NaN where t+h exceeds the series (no fabricated future label).

Both models AND baselines are scored against y_h — never the feature-
time phase. Persistence (phase at t) vs y_h (phase at t+h) is then a
genuine, beatable baseline, not the target compared to itself.

Logged as D23 (horizon-target construction).

NO MODELLING.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import TARGET_BINARY, TARGET_ORDINAL, UNIT_COL


def add_horizon_targets(df: pd.DataFrame, horizons=(1, 3)) -> pd.DataFrame:
    """
    Add y_binary_h{h} and y_ordinal_h{h} columns: the target shifted
    FORWARD by h months within each unit (so a row at t carries the
    label observed at t+h). Rows whose t+h exceeds the unit's series get
    NaN (no fabricated future) and are excluded from train/test.

    Requires a contiguous monthly series per unit (asserted), so the
    row-shift equals a month-shift.
    """
    df = df.sort_values([UNIT_COL, "t_index"]).reset_index(drop=True)

    # contiguity assertion (per unit, consecutive t_index)
    d = df.copy()
    diffs = d.groupby(UNIT_COL)["t_index"].diff().dropna()
    bad = diffs[diffs != 1]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} non-contiguous month steps; a forward shift "
            f"assumes contiguous months. Fill/investigate first."
        )

    for h in horizons:
        # shift(-h) within unit pulls the value h rows AHEAD (t+h) to row t
        df[f"y_binary_h{h}"] = (
            df.groupby(UNIT_COL)[TARGET_BINARY].shift(-h)
        )
        df[f"y_ordinal_h{h}"] = (
            df.groupby(UNIT_COL)[TARGET_ORDINAL].shift(-h)
        )
        # rows where t+h ran off the end are NaN (correct — no future label)

    return df


def assert_target_not_leaking(df: pd.DataFrame, h: int) -> None:
    """
    Sanity assertion: the horizon target must NOT equal the feature-time
    phase for a nontrivial fraction (if it did, the shift silently failed
    and we're back to target-scoring-itself). Allow equality where the
    phase genuinely didn't change, but a PERFECT match means no shift.
    """
    col = f"y_binary_h{h}"
    valid = df[df[col].notna()]
    if len(valid) == 0:
        raise RuntimeError(f"h{h}: no valid horizon targets (all NaN).")
    frac_equal = (valid[col] == valid[TARGET_BINARY]).mean()
    # phase is autocorrelated so high equality is expected, but exactly
    # 1.0 means the shift did nothing (the bug we're guarding against).
    if frac_equal >= 0.9999:
        raise RuntimeError(
            f"h{h}: horizon target equals feature-time phase for "
            f"{frac_equal:.4f} of rows — the shift did not apply. This is "
            f"the target-leakage bug (target scoring itself)."
        )
    print(f"  \u2713 h{h}: horizon target distinct from feature-time phase "
          f"(equal on {frac_equal:.1%} — autocorrelation, not leakage; "
          f"{int(valid[col].sum())} positives)")


if __name__ == "__main__":
    from harness_data import load_all
    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
