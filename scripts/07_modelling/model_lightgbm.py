"""
======================================================================

model_lightgbm.py  (M10.3 — LightGBM quantile regression, C1 second model)

D16 fallback (kill-switch executed): the TFT failed the gate (discrimination
short of XGBoost in all windows, calibration failed 2 of 3). LightGBM
quantile regression becomes C1's second benchmark model AND C2's uncertainty
source, exactly as pre-registered in D16.

Design — matches the XGBoost harness for like-for-like C1:
  - Binary target phase3plus at t+h (D8/D23), D22 canonical predictors,
    log-transformed density, four-arm ablation A/C/B (D26).
  - Discrimination: a standard LightGBM CLASSIFIER (binary) gives the
    probability for PR-AUC — directly comparable to XGBoost's predict_proba.
  - Uncertainty (C2): LightGBM QUANTILE REGRESSORS (objective="quantile",
    alpha in {0.1,0.5,0.9}) on the ORDINAL phase give P10/P50/P90 intervals,
    replacing the TFT quantiles. Coverage computed the same way.
  - Same walk-forward splits, D25-canonical rows, shared evaluation,
    isotonic calibration on a train-only chronological slice (no test leak).
  - Per-window, no pooled average.

Runs in gee_env (lightgbm installed). Structure/leakage verified offline.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_BINARY, TARGET_ORDINAL, IS_OBSERVED,
    UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_eval import evaluate_binary, evaluate_baselines
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
PHASE3_THRESHOLD = 3
QUANTILES = [0.1, 0.5, 0.9]

CONFIG_A_PREDICTORS = CANONICAL_PREDICTORS
CONFIG_B_PREDICTORS = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
CONFIG_C_PREDICTORS = list(PHASE_HISTORY_COLS)


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(df, idx, ycol, predictors):
    sub = df.loc[idx]
    sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].to_numpy(), sub.index


def train_predict_classifier(df, split, ycol_bin, predictors):
    """LightGBM binary classifier -> calibrated probability for PR-AUC.
    Calibration isotonic on a chronological train-only slice (no test leak)."""
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    Xtr, ytr, tr_idx = _prep(df, split["train_idx"], ycol_bin, predictors)
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol_bin, predictors)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    tr_rows = df.loc[tr_idx].sort_values("t_index")
    n_cal = int(len(tr_rows) * CAL_FRACTION)
    cal_idx = tr_rows.index[-n_cal:] if n_cal else tr_rows.index[:0]
    fit_idx = tr_rows.index[:-n_cal] if n_cal else tr_rows.index
    Xfit, yfit, _ = _prep(df, fit_idx, ycol_bin, predictors)
    Xcal, ycal, _ = _prep(df, cal_idx, ycol_bin, predictors)

    pos = yfit.sum(); neg = len(yfit) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    clf = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xfit, yfit)

    raw_te = clf.predict_proba(Xte)[:, 1]
    if len(Xcal) >= 50 and len(np.unique(ycal)) == 2:
        raw_cal = clf.predict_proba(Xcal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(raw_cal, ycal)
        prob = iso.transform(raw_te)
    else:
        prob = raw_te
    return {"te_idx": te_idx, "y_true": yte, "prob": prob}


def train_predict_quantiles(df, split, ycol_ord, predictors):
    """Three LightGBM quantile regressors on the ORDINAL phase -> P10/P50/P90
    (C2 uncertainty, replacing TFT quantiles)."""
    import lightgbm as lgb
    Xtr, ytr, _ = _prep(df, split["train_idx"], ycol_ord, predictors)
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol_ord, predictors)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None
    preds = {}
    for a in QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=a, n_estimators=300, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        m.fit(Xtr, ytr)
        preds[a] = m.predict(Xte)
    p10, p50, p90 = preds[0.1], preds[0.5], preds[0.9]
    # enforce monotone quantiles (rare crossings sorted)
    stacked = np.sort(np.vstack([p10, p50, p90]).T, axis=1)
    return {"te_idx": te_idx, "y_ord": yte,
            "p10": stacked[:, 0], "p50": stacked[:, 1], "p90": stacked[:, 2]}


def run():
    section("M10.3 — LightGBM (C1 second model, D16 fallback): four-arm + C2 quantiles")
    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)
    fews_rows = build_fews_baseline_rows(df)

    print(f"\n{'split':<9}{'prev':>5}{'A':>7}{'C':>7}{'B':>7}"
          f"{'B-C':>8}{'B-A':>7}{'persist':>8}{'FEWS':>7}{'Bp':>4}{'Bf':>4}{'cov80':>7}")
    print("-" * 82)

    results = []
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        ycb, yco = f"y_binary_h{h}", f"y_ordinal_h{h}"

        pA = train_predict_classifier(df, s, ycb, CONFIG_A_PREDICTORS)
        pB = train_predict_classifier(df, s, ycb, CONFIG_B_PREDICTORS)
        pC = train_predict_classifier(df, s, ycb, CONFIG_C_PREDICTORS)
        if not (pA and pB and pC):
            continue

        A = evaluate_binary(pA["y_true"], pA["prob"])["all"]["pr_auc"]
        Bev = evaluate_binary(pB["y_true"], pB["prob"])["all"]
        B = Bev["pr_auc"]; prev = Bev["prevalence"]
        C = evaluate_binary(pC["y_true"], pC["prob"])["all"]["pr_auc"]

        base = evaluate_baselines(df, s)
        persist = base["persistence"]["all"]["pr_auc"]
        fews = fews_baseline_for_split(df, fews_rows, s, proj_type="near")
        fews_pr = fews["fews"]["all"]["pr_auc"] if fews else np.nan

        # C2 uncertainty: quantile intervals (Config B), coverage of ordinal truth
        cov80 = np.nan
        q = train_predict_quantiles(df, s, yco, CONFIG_B_PREDICTORS)
        if q:
            y_ord = q["y_ord"]; fin = np.isfinite(y_ord)
            cov80 = float(np.mean((y_ord[fin] >= q["p10"][fin]) &
                                  (y_ord[fin] <= q["p90"][fin]))) if fin.any() else np.nan

        bp = "y" if B > persist else "n"
        bf = ("y" if (not np.isnan(fews_pr) and B > fews_pr)
              else ("n" if not np.isnan(fews_pr) else "-"))

        print(f"{yr}/h{h:<6}{prev:>5.2f}{A:>7.3f}{C:>7.3f}{B:>7.3f}"
              f"{B-C:>+8.3f}{B-A:>+7.3f}{persist:>8.3f}"
              f"{(fews_pr if not np.isnan(fews_pr) else float('nan')):>7.3f}"
              f"{bp:>4}{bf:>4}{cov80:>7.2f}")

        results.append({"year": yr, "horizon": h, "prevalence": prev,
                        "A": A, "C": C, "B": B, "B_minus_C": B-C, "B_minus_A": B-A,
                        "persistence": persist, "fews": fews_pr,
                        "B_beats_persist": bp, "B_beats_fews": bf, "coverage80": cov80})

    section("READING (per-window, no pooled)")
    print("- LightGBM (D16 fallback) = C1 second model + C2 uncertainty (quantiles).")
    print("- B-C = biophysical value-added over phase (compare to XGBoost's B-C).")
    print("- cov80 = LightGBM quantile-interval coverage (C2; target 70-90%).")
    print("- h=3 is the meaningful horizon; h=1 persistence-dominated.")
    pd.DataFrame(results).to_csv("outputs/diagnostics/m10_lightgbm_4arm.csv", index=False)
    print("\nsaved: outputs/diagnostics/m10_lightgbm_4arm.csv")
    return results


if __name__ == "__main__":
    run()
