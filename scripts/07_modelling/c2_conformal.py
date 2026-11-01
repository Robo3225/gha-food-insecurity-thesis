"""
======================================================================

c2_conformal.py  (M10.4 / C2 — split-conformal uncertainty calibration)

Split-conformal prediction to give the LightGBM quantile
intervals a DISTRIBUTION-FREE coverage guarantee, replacing the raw
under-covering quantiles (0.55-0.80 vs nominal 0.80). Conformal achieves
nominal coverage by construction — a citable guarantee, not a scaled fudge.

NON-NEGOTIABLE: the conformal calibration set is TRAIN-ONLY —
carved from within each training window, never touching test. This is the
uncertainty-side analogue of the isotonic-calibration leakage guard.
Calibrating intervals on test data is the exact leak that yields beautiful,
meaningless coverage.

Method — Conformalized Quantile Regression (CQR, Romano et al. 2019),
the standard way to conformalize quantile intervals:
  1. Fit lower/upper quantile regressors (P10/P90) on the TRAIN-FIT slice.
  2. On a held-out TRAIN-CALIBRATION slice, compute conformity scores
     E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))  (how far truth falls
     outside the raw interval; negative if inside).
  3. Take the (1-alpha)(1+1/n) empirical quantile of E over the cal set = Q.
  4. Conformalized interval on TEST: [q_lo(x) - Q, q_hi(x) + Q].
     This has guaranteed >= 1-alpha marginal coverage (distribution-free).

Reported per window/horizon: raw coverage vs conformalized coverage, and
the interval widening Q. Train-only throughout.

Runs in gee_env. Structure/leakage verified offline.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_ORDINAL, IS_OBSERVED, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
NOMINAL = 0.80          # target coverage
ALPHA = 1 - NOMINAL     # 0.20
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(df, idx, ycol, predictors):
    sub = df.loc[idx]; sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].to_numpy(), sub.index


def conformal_split(df, split, ycol_ord, predictors):
    """
    CQR on one walk-forward split. Fit P10/P90 on train-FIT; compute
    conformity on train-CAL (train-only, no test); apply the conformal
    widening Q to the TEST intervals. Returns raw vs conformalized coverage.
    """
    import lightgbm as lgb

    # chronological train split: FIT (earlier) + CAL (latest train slice)
    tr_rows = df.loc[split["train_idx"]].copy()
    tr_rows = tr_rows[tr_rows[ycol_ord].notna()].sort_values("t_index")
    n_cal = int(len(tr_rows) * CAL_FRACTION)
    if n_cal < 50:
        return None
    fit_idx = tr_rows.index[:-n_cal]
    cal_idx = tr_rows.index[-n_cal:]           # TRAIN-only calibration set

    Xfit, yfit, _ = _prep(df, fit_idx, ycol_ord, predictors)
    Xcal, ycal, _ = _prep(df, cal_idx, ycol_ord, predictors)
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol_ord, predictors)
    if len(Xte) == 0:
        return None

    def q_model(alpha):
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha,
                              n_estimators=300, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
        m.fit(Xfit, yfit); return m

    lo_a, hi_a = ALPHA / 2, 1 - ALPHA / 2       # 0.10, 0.90 for 80% interval
    m_lo, m_hi = q_model(lo_a), q_model(hi_a)

    # raw intervals
    cal_lo, cal_hi = m_lo.predict(Xcal), m_hi.predict(Xcal)
    te_lo, te_hi = m_lo.predict(Xte), m_hi.predict(Xte)

    # --- conformity scores on TRAIN-CAL (CQR) ---
    E = np.maximum(cal_lo - ycal, ycal - cal_hi)
    n = len(E)
    # (1-alpha)(1+1/n) empirical quantile
    k = int(np.ceil((1 - ALPHA) * (n + 1)))
    k = min(max(k, 1), n)
    Q = np.sort(E)[k - 1]

    # --- conformalized TEST intervals ---
    cte_lo, cte_hi = te_lo - Q, te_hi + Q

    fin = np.isfinite(yte)
    raw_cov = float(np.mean((yte[fin] >= te_lo[fin]) & (yte[fin] <= te_hi[fin])))
    conf_cov = float(np.mean((yte[fin] >= cte_lo[fin]) & (yte[fin] <= cte_hi[fin])))
    raw_width = float(np.mean(te_hi[fin] - te_lo[fin]))
    conf_width = float(np.mean(cte_hi[fin] - cte_lo[fin]))

    return {"raw_cov": raw_cov, "conf_cov": conf_cov, "Q": float(Q),
            "raw_width": raw_width, "conf_width": conf_width,
            "n_cal": n, "n_test": int(fin.sum())}


def run():
    section("M10.4 / C2 — split-conformal calibration of quantile intervals (train-only)")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)

    print(f"\n{'split':<9}{'n_cal':>7}{'n_test':>8}{'raw_cov':>9}{'conf_cov':>10}"
          f"{'Q':>7}{'raw_w':>7}{'conf_w':>8}")
    print("-" * 66)

    results = []
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        out = conformal_split(df, s, f"y_ordinal_h{h}", CONFIG_B)
        if out is None:
            continue
        print(f"{yr}/h{h:<6}{out['n_cal']:>7}{out['n_test']:>8}"
              f"{out['raw_cov']:>9.2f}{out['conf_cov']:>10.2f}"
              f"{out['Q']:>7.2f}{out['raw_width']:>7.2f}{out['conf_width']:>8.2f}")
        results.append({"year": yr, "horizon": h, **out})

    section("READING")
    print(f"- nominal coverage target: {NOMINAL:.0%}")
    print("- raw_cov = under-covering LightGBM quantiles (the C1 problem).")
    print("- conf_cov = after split-conformal (CQR), TRAIN-only calibration.")
    print("  Should sit at/above nominal by construction (distribution-free guarantee).")
    print("- Q = conformal widening; conf_w - raw_w = how much intervals grew.")
    print("- calibration set is TRAIN-only (latest train slice); never test.")
    pd.DataFrame(results).to_csv("outputs/diagnostics/c2_conformal.csv", index=False)
    print("\nsaved: outputs/diagnostics/c2_conformal.csv")
    return results


if __name__ == "__main__":
    run()
