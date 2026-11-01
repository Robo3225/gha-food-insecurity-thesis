"""
======================================================================

c2_conformal_weighted.py  (M10.4 / C2 extension — weighted split-conformal)

Bounded shift-robust attempt (pre-registered extension). Base CQR
under-covers under crisis-period distribution shift because the
calibration slice (pre-crisis) is not exchangeable with the test window
(crisis). This variant up-weights RECENT-WITHIN-TRAIN calibration
residuals, so the conformal quantile Q reflects the most-recent-training
regime — the best available proxy for the shifted test period.

NON-NEGOTIABLE: TRAIN-ONLY. "recent" = recent within the
TRAINING window. The recency weight uses calibration rows' t_index (all
strictly pre-test by construction). Test never enters the weight or the
residuals. This is weighted-conformal (Tibshirani et al. 2019 style),
with weights a function of within-train recency only.

PRE-COMMITTED (before seeing numbers): both outcomes
reportable — closing the gap = shift-robust contribution; not closing it
= the deeper limit "even adaptive conformal can't recover coverage under
crisis-severity shift". No coverage-chasing.

Reports base-conformal vs weighted-conformal coverage side by side.

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
from harness_splits import all_splits
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
NOMINAL = 0.80
ALPHA = 1 - NOMINAL
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS

# recency half-life for the within-train weight, in MONTHS. Fixed a priori
# (not tuned to coverage): ~ one tri-annual assessment interval, so residuals
# from the most recent ~4-8 months of training dominate Q. Train-only.
RECENCY_HALFLIFE_MONTHS = 6.0


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(df, idx, ycol, predictors):
    sub = df.loc[idx]; sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].to_numpy(), sub.index


def _weighted_quantile(values, weights, q):
    """Weighted empirical quantile (for the conformal Q under recency weights)."""
    order = np.argsort(values)
    v = np.asarray(values)[order]; w = np.asarray(weights)[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= np.sum(w)
    return float(np.interp(q, cum, v))


def weighted_conformal_split(df, split, ycol_ord, predictors):
    """
    CQR with base (unweighted) AND recency-weighted conformal Q, both from
    the TRAIN-only calibration slice. Returns coverage for each on TEST.
    """
    import lightgbm as lgb

    tr_rows = df.loc[split["train_idx"]].copy()
    tr_rows = tr_rows[tr_rows[ycol_ord].notna()].sort_values("t_index")
    n_cal = int(len(tr_rows) * CAL_FRACTION)
    if n_cal < 50:
        return None
    fit_idx = tr_rows.index[:-n_cal]
    cal_idx = tr_rows.index[-n_cal:]

    Xfit, yfit, _ = _prep(df, fit_idx, ycol_ord, predictors)
    cal = df.loc[cal_idx]
    Xcal, ycal = cal[predictors].copy(), cal[ycol_ord].to_numpy()
    cal_t = cal["t_index"].to_numpy()          # within-train times (pre-test)
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol_ord, predictors)
    if len(Xte) == 0:
        return None

    def qm(a):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=300,
                              num_leaves=31, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, random_state=RANDOM_SEED,
                              n_jobs=-1, verbose=-1)
        m.fit(Xfit, yfit); return m

    lo_a, hi_a = ALPHA/2, 1-ALPHA/2
    m_lo, m_hi = qm(lo_a), qm(hi_a)
    cal_lo, cal_hi = m_lo.predict(Xcal), m_hi.predict(Xcal)
    te_lo, te_hi = m_lo.predict(Xte), m_hi.predict(Xte)

    E = np.maximum(cal_lo - ycal, ycal - cal_hi)   # conformity scores
    n = len(E)

    # --- base (unweighted) Q ---
    k = min(max(int(np.ceil((1-ALPHA)*(n+1))), 1), n)
    Q_base = np.sort(E)[k-1]

    # --- recency-weighted Q (TRAIN-ONLY): weight by recency within TRAIN ---
    # most recent calibration month gets weight 1; older decays exponentially.
    t_max = cal_t.max()
    age_months = t_max - cal_t                       # >=0, within-train age
    w = np.exp(-np.log(2) * age_months / RECENCY_HALFLIFE_MONTHS)
    Q_wt = _weighted_quantile(E, w, 1 - ALPHA)

    fin = np.isfinite(yte)
    def cov(Q):
        return float(np.mean((yte[fin] >= te_lo[fin]-Q) & (yte[fin] <= te_hi[fin]+Q)))

    return {"base_cov": cov(Q_base), "wt_cov": cov(Q_wt),
            "Q_base": float(Q_base), "Q_wt": float(Q_wt),
            "n_cal": n, "n_test": int(fin.sum())}


def run():
    section("M10.4 / C2 extension — weighted (shift-robust) conformal, TRAIN-ONLY")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)

    print(f"\n{'split':<9}{'n_cal':>7}{'base_cov':>10}{'wt_cov':>9}"
          f"{'Q_base':>8}{'Q_wt':>7}{'improved?':>11}")
    print("-" * 62)

    results = []
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        out = weighted_conformal_split(df, s, f"y_ordinal_h{h}", CONFIG_B)
        if out is None:
            continue
        improved = "yes" if out["wt_cov"] > out["base_cov"] else "no"
        reaches = "→nominal" if out["wt_cov"] >= NOMINAL else ""
        print(f"{yr}/h{h:<6}{out['n_cal']:>7}{out['base_cov']:>10.2f}"
              f"{out['wt_cov']:>9.2f}{out['Q_base']:>8.2f}{out['Q_wt']:>7.2f}"
              f"{improved:>11} {reaches}")
        results.append({"year": yr, "horizon": h, **out, "improved": improved})

    section("READING — pre-committed both outcomes reportable")
    print(f"- nominal target: {NOMINAL:.0%}. Recency half-life: "
          f"{RECENCY_HALFLIFE_MONTHS} mo (fixed a priori, TRAIN-ONLY).")
    print("- wt_cov > base_cov under shift (2021/2022) => shift-robust contribution.")
    print("- wt_cov still < nominal under shift => the DEEPER limit: even adaptive")
    print("  conformal can't fully recover coverage under crisis-severity shift.")
    print("- weight uses within-TRAIN recency only; test never touches Q.")
    pd.DataFrame(results).to_csv("outputs/diagnostics/c2_conformal_weighted.csv", index=False)
    print("\nsaved: outputs/diagnostics/c2_conformal_weighted.csv")
    return results


if __name__ == "__main__":
    run()
