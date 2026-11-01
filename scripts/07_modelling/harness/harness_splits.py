"""
======================================================================

harness_splits.py  (M10.1 — walk-forward splits + fail-loud leakage audit)

Expanding-window walk-forward validation with THREE test windows
(2021, 2022, 2023) and TWO horizons (1-month, 3-month), per the design.

The central M10 principle: the harness must be mathematically
INCAPABLE of running a leaky split. Every split passes a named, loud
assertion battery before it can be used. A leaky split raises; it never
silently proceeds.

Leakage assertions:
  1. CHRONOLOGY: no training observation post-dates any test observation.
  2. BI-DIRECTIONAL LAGS: lag features never cross the train/test
     boundary in either direction.
  3. (autocorrelation + calibration leakage are enforced in the baseline
     / calibration modules, asserted there.)

Expanding window: test window Y trains on ALL data strictly before Y's
test period. The horizon h means predicting phase at t+h from features
at t, so the train/test boundary accounts for h to prevent the label
of a training row being observed inside the test period.

NO MODELLING.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import LAGS, TIME_COL, UNIT_COL


# test windows: calendar years, as t_index ranges (months since 2018-01)
# 2021 = t 36..47, 2022 = 48..59, 2023 = 60..71
TEST_WINDOWS = {
    2021: (36, 47),
    2022: (48, 59),
    2023: (60, 71),
}
HORIZONS = (1, 3)
MAX_LAG = max(LAGS)   # 12


def make_split(df: pd.DataFrame, test_year: int, horizon: int) -> dict:
    """
    Build one expanding-window split for (test_year, horizon).

    Training rows: feature-time t such that the PREDICTED time (t+horizon)
    falls strictly before the test window start. This guarantees no
    training label is observed within the test period.

    Test rows: feature-time t such that predicted time (t+horizon) falls
    within the test window.
    """
    t_lo, t_hi = TEST_WINDOWS[test_year]

    # predicted-time of a feature-row is t + horizon
    pred_t = df["t_index"] + horizon

    # test: predicted time inside the window
    test_mask = (pred_t >= t_lo) & (pred_t <= t_hi)
    # train: predicted time strictly before the window start
    train_mask = pred_t < t_lo

    split = {
        "test_year": test_year,
        "horizon": horizon,
        "train_idx": df.index[train_mask].to_numpy(),
        "test_idx": df.index[test_mask].to_numpy(),
        "t_lo": t_lo, "t_hi": t_hi,
    }
    return split


def assert_no_leakage(df: pd.DataFrame, split: dict) -> None:
    """
    Fail-loud leakage battery. Raises on any violation. A split that
    passes this is chronologically and lag-safe by construction.
    """
    tr = df.loc[split["train_idx"]]
    te = df.loc[split["test_idx"]]
    h = split["horizon"]

    if len(tr) == 0 or len(te) == 0:
        raise RuntimeError(
            f"[{split['test_year']}/h{h}] empty train ({len(tr)}) or "
            f"test ({len(te)}) set."
        )

    # --- 1. CHRONOLOGY (on PREDICTED time) ---------------------------
    # every training predicted-time must be < every test predicted-time.
    tr_pred_max = (tr["t_index"] + h).max()
    te_pred_min = (te["t_index"] + h).min()
    if not (tr_pred_max < te_pred_min):
        raise RuntimeError(
            f"[{split['test_year']}/h{h}] CHRONOLOGY LEAK: max train "
            f"predicted-time {tr_pred_max} >= min test predicted-time "
            f"{te_pred_min}. A training label is observed in/after the "
            f"test period."
        )

    # --- 2. BI-DIRECTIONAL LAG boundary ------------------------------
    # A lag feature at feature-time t uses observations at t-1..t-MAX_LAG.
    # Forward (test not pulling scored train values): a test row's lags
    #   reach back to t-MAX_LAG; those are FEATURE observations (biophysical
    #   inputs), not labels, and they precede the test row — allowed. What
    #   must NOT happen: a training row's PREDICTED time reaching into the
    #   test window. Enforced by chronology above (pred_t < t_lo for train).
    # Backward (train lags not reaching test-period observations): a
    #   training row at feature-time t has lags at t-k (earlier still), so
    #   training lags reach BACKWARD, never forward into test. Assert the
    #   max training FEATURE-time + 0 (its own lags are all <= its t) does
    #   not enter the test feature-period.
    tr_feat_max = tr["t_index"].max()
    te_feat_min = te["t_index"].min()
    # training feature-times (and their backward lags) must not overlap the
    # test feature-period. Because lags only look backward, the only way a
    # training row touches the test period is if its OWN feature-time does.
    if not (tr_feat_max < te_feat_min):
        # this can legitimately be tight for horizon>1; the binding
        # guarantee is the predicted-time chronology above. But feature-time
        # overlap would mean a training row shares a timestamp with a test
        # row — assert separation to be safe.
        raise RuntimeError(
            f"[{split['test_year']}/h{h}] LAG-BOUNDARY LEAK: max train "
            f"feature-time {tr_feat_max} >= min test feature-time "
            f"{te_feat_min}. Training features overlap the test period."
        )

    # --- 3. no shared (unit, feature-time) rows across train/test -----
    tr_keys = set(zip(tr[UNIT_COL], tr["t_index"]))
    te_keys = set(zip(te[UNIT_COL], te["t_index"]))
    overlap = tr_keys & te_keys
    if overlap:
        raise RuntimeError(
            f"[{split['test_year']}/h{h}] {len(overlap)} (unit,time) rows "
            f"appear in BOTH train and test."
        )


def all_splits(df: pd.DataFrame, verbose: bool = True) -> list:
    """Generate + leakage-audit every (test_year × horizon) split."""
    splits = []
    for hz in HORIZONS:
        for yr in TEST_WINDOWS:
            s = make_split(df, yr, hz)
            assert_no_leakage(df, s)          # fail-loud; never silent
            splits.append(s)
            if verbose:
                print(f"  \u2713 split {yr}/h{hz}: "
                      f"train={len(s['train_idx']):,}  "
                      f"test={len(s['test_idx']):,}  (leakage-audited)")
    return splits


if __name__ == "__main__":
    from harness_data import load_all
    df = load_all()
    print("\nwalk-forward splits (expanding window, leakage-audited):")
    all_splits(df)
