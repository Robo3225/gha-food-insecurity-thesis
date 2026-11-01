"""
======================================================================

harness_phase_history.py  (M10.2 / D26 — leakage-safe phase-history features)

Config B (D26) adds IPC phase history to the biophysical predictors:
the observed phase3plus at t (current), t-4, t-8 (tri-annual cadence).

Leakage safety (D23 discipline — the load-bearing part):
  - phase history at feature-time t uses ONLY observations at t and earlier
    (strictly backward), NEVER the t+h target.
  - within-unit, contiguity-asserted (row-shift == month-shift).
  - first rows per unit are NaN by construction (no fabricated history).
  - the walk-forward split is keyed on PREDICTED time t+h (D23), so a test
    row's phase history is entirely pre-test-window observation — no reach
    across the split boundary.

Current phase (t) is INCLUDED (D26): it is the same value persistence uses
(making the comparison information-matched) and is genuinely observable at t
(the target is at t+h, a different, later period).

Which phase is lagged: the forward-filled phase3plus (D10) — the operational
"last known IPC phase" a forecaster carries forward between tri-annual
assessments, not the raw-observed-only (which is NaN between assessments).
This matches D26's "a forecaster knows the current and recent phase".

PHASE_HISTORY_LAGS = (0, 4, 8)  # 0 = current phase at t

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import UNIT_COL, TARGET_BINARY

PHASE_HISTORY_LAGS = (0, 4, 8)   # months; 0 = current phase at feature-time t
PHASE_HISTORY_COLS = [f"phase_hist_lag{k}" for k in PHASE_HISTORY_LAGS]


def add_phase_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add phase_hist_lag{0,4,8}: the phase3plus observed at t, t-4, t-8 within
    unit. lag0 is the current phase (a copy of phase3plus at t). Strictly
    backward; contiguity-asserted; leading rows NaN.
    """
    df = df.sort_values([UNIT_COL, "t_index"]).reset_index(drop=True)

    # contiguity per unit (row-shift == month-shift)
    diffs = df.groupby(UNIT_COL)["t_index"].diff().dropna()
    bad = diffs[diffs != 1]
    if len(bad):
        raise RuntimeError(
            f"{len(bad)} non-contiguous month steps; phase-history shift "
            f"assumes contiguous months. Fill/investigate first."
        )

    for k in PHASE_HISTORY_LAGS:
        col = f"phase_hist_lag{k}"
        if k == 0:
            df[col] = df[TARGET_BINARY]          # current phase at t
        else:
            df[col] = df.groupby(UNIT_COL)[TARGET_BINARY].shift(k)  # t-k, backward

    return df


def assert_phase_history_safe(df: pd.DataFrame, horizons=(1, 3)) -> None:
    """
    Verify phase history never equals the FUTURE target (would be leakage).
    lag0 (current phase at t) vs y_binary_h{h} (phase at t+h) must NOT be
    identical for ~all rows — if it were, the 'history' would be the target.
    High equality is fine (autocorrelation); perfect equality is leakage.
    """
    for h in horizons:
        ycol = f"y_binary_h{h}"
        if ycol not in df.columns:
            continue
        valid = df[df[ycol].notna() & df["phase_hist_lag0"].notna()]
        if len(valid) == 0:
            continue
        frac_eq = (valid["phase_hist_lag0"] == valid[ycol]).mean()
        if frac_eq >= 0.9999:
            raise RuntimeError(
                f"h{h}: current-phase feature equals the t+h target for "
                f"{frac_eq:.4f} of rows — phase history IS the target "
                f"(leakage). Check the shift/horizon construction."
            )
        print(f"  \u2713 phase-history leakage check h{h}: current phase != "
              f"t+h target (equal on {frac_eq:.1%} — autocorrelation, safe)")


if __name__ == "__main__":
    from harness_data import load_all
    from harness_target import add_horizon_targets
    df = load_all()
    df = add_horizon_targets(df)
    df = add_phase_history(df)
    assert_phase_history_safe(df)
    print("phase-history cols:", PHASE_HISTORY_COLS)
    print("non-null current phase:", df["phase_hist_lag0"].notna().sum())
