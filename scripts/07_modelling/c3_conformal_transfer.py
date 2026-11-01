"""
======================================================================

c3_conformal_transfer.py  (M10.5 / C3 — conformal under transfer; C2<->C3 capstone)

The uncertainty axis of the C3 decomposition, and the C2<->C3 capstone.
Conformal calibrated on KE+ET, applied to Somalia — the LARGEST
distribution shift in the thesis (cross-country, not merely cross-time).

C2 already showed conformal coverage degrades under the milder TEMPORAL
shift (crisis windows). Pre-registered expectation (C3_preregistration):
cross-country transfer degrades it FURTHER — and that degradation IS the
uncertainty-axis finding, not a failure. It lands the through-line:
prediction (B-C collapse at onset), data-quality (D6/LST gradients), and
uncertainty (this) are ALL hardest under transfer to the most-affected context.

Design (mirrors c2_conformal, transfer partition):
  - Quantile regressors fit on KE+ET FIT slice.
  - Conformity scores on a KE+ET CALIBRATION slice (train-only — here
    "train" = the source countries; Somalia never informs Q).
  - Applied to Somalia test rows; coverage measured on Somalia.
  - Compared to C2's in-pool conformal coverage (the milder shift) to
    quantify the ADDITIONAL degradation from cross-country transfer.
  - Also the weighted (recency) variant, for parity with C2's extension.

Leakage: Q computed only from KE+ET calibration residuals; Somalia never
enters Q. The transfer of the calibration itself is the intended test.

Runs in gee_env. Structure mirrors the verified c2_conformal.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_ORDINAL, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
NOMINAL = 0.80
ALPHA = 1 - NOMINAL
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
TRAIN_COUNTRIES = ["kenya", "ethiopia"]
TEST_COUNTRY = "somalia"
TEST_WINDOWS = {2021: (36, 47), 2022: (48, 59), 2023: (60, 71)}

# C2 in-pool conformal coverage (the milder temporal shift) for comparison
C2_INPOOL_CONF_COV_H3 = {2021: 0.76, 2022: 0.68, 2023: 0.84}


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(sub, ycol, predictors):
    sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].to_numpy(), sub.index


def conformal_transfer(train_df, test_df, ycol_ord, predictors):
    """CQR: fit + calibrate on KE+ET (train-only), apply to Somalia. Returns
    base and recency-weighted coverage on Somalia."""
    import lightgbm as lgb

    tr = train_df[train_df[ycol_ord].notna()].sort_values("t_index")
    n_cal = int(len(tr) * CAL_FRACTION)
    if n_cal < 50:
        return None
    fit_idx = tr.index[:-n_cal]; cal_idx = tr.index[-n_cal:]   # KE+ET only
    Xfit, yfit, _ = _prep(train_df.loc[fit_idx], ycol_ord, predictors)
    cal = train_df.loc[cal_idx]
    Xcal, ycal = cal[predictors].copy(), cal[ycol_ord].to_numpy()
    cal_t = cal["t_index"].to_numpy()
    Xte, yte, _ = _prep(test_df, ycol_ord, predictors)   # Somalia
    if len(Xte) == 0:
        return None

    def qm(a):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                              num_leaves=31, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=RANDOM_SEED,
                              n_jobs=-1, verbose=-1)
        m.fit(Xfit, yfit); return m

    m_lo, m_hi = qm(ALPHA/2), qm(1-ALPHA/2)
    cal_lo, cal_hi = m_lo.predict(Xcal), m_hi.predict(Xcal)
    te_lo, te_hi = m_lo.predict(Xte), m_hi.predict(Xte)

    E = np.maximum(cal_lo - ycal, ycal - cal_hi)
    n = len(E)
    k = min(max(int(np.ceil((1-ALPHA)*(n+1))), 1), n)
    Q_base = np.sort(E)[k-1]

    # recency-weighted (within-KE+ET-train recency)
    age = cal_t.max() - cal_t
    w = np.exp(-np.log(2) * age / 6.0)
    order = np.argsort(E); Es = E[order]; ws = w[order]
    cum = (np.cumsum(ws) - 0.5*ws) / ws.sum()
    Q_wt = float(np.interp(1-ALPHA, cum, Es))

    fin = np.isfinite(yte)
    def cov(Q):
        return float(np.mean((yte[fin] >= te_lo[fin]-Q) & (yte[fin] <= te_hi[fin]+Q)))
    return {"base_cov": cov(Q_base), "wt_cov": cov(Q_wt), "n_test": int(fin.sum())}


def run():
    section("M10.5 / C3 — conformal UNDER TRANSFER (KE+ET -> Somalia); C2<->C3 capstone")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)

    train_all = df[df["country"].isin(TRAIN_COUNTRIES)].copy()
    test_all = df[df["country"] == TEST_COUNTRY].copy()

    print(f"\n{'win':<9}{'transfer_cov':>13}{'wt_cov':>9}{'C2_inpool':>11}"
          f"{'Δ vs in-pool':>14}")
    print("-" * 58)

    results = []
    for h in (3,):   # h=3 is the meaningful horizon (matches C2 comparison)
        ycol = f"y_ordinal_h{h}"
        for yr, (t_lo, t_hi) in TEST_WINDOWS.items():
            pred_t = test_all["t_index"] + h
            te_win = test_all[(pred_t >= t_lo) & (pred_t <= t_hi)].copy()
            tr_win = train_all[(train_all["t_index"] + h) < t_lo].copy()
            if len(te_win) == 0 or len(tr_win) == 0:
                continue
            out = conformal_transfer(tr_win, te_win, ycol, CONFIG_B)
            if out is None:
                continue
            inpool = C2_INPOOL_CONF_COV_H3.get(yr, np.nan)
            delta = out["base_cov"] - inpool if not np.isnan(inpool) else np.nan
            print(f"{yr}/h{h:<5}{out['base_cov']:>13.2f}{out['wt_cov']:>9.2f}"
                  f"{inpool:>11.2f}{delta:>+14.2f}")
            results.append({"year": yr, "horizon": h, "transfer_cov": out["base_cov"],
                            "wt_cov": out["wt_cov"], "c2_inpool_cov": inpool,
                            "delta_vs_inpool": delta, "n_test": out["n_test"]})

    section("READING — uncertainty axis of the C3 decomposition (pre-registered)")
    print(f"- nominal target {NOMINAL:.0%}. transfer_cov = conformal calibrated on")
    print("  KE+ET, applied to Somalia (largest shift in the thesis).")
    print("- Δ vs in-pool = ADDITIONAL degradation from cross-country transfer,")
    print("  beyond C2's temporal-shift degradation. Negative Δ = worse under transfer.")
    print("- Pre-registered: transfer degrades coverage FURTHER -> uncertainty axis")
    print("  of the C3 story (prediction + data-quality + uncertainty all hardest here).")
    print("- calibration TRAIN(=KE+ET)-only; Somalia never informs Q (leakage-safe).")
    pd.DataFrame(results).to_csv("outputs/diagnostics/c3_conformal_transfer.csv", index=False)
    print("\nsaved: outputs/diagnostics/c3_conformal_transfer.csv")
    return results


if __name__ == "__main__":
    run()
