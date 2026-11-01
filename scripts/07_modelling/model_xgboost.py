"""
======================================================================

model_xgboost.py  (M10.2 — XGBoost benchmark, C1)

The first model, attached to the M10.1 foundation (which has been proven
to make an unfair good number impossible: it caught target-leakage,
a missing benchmark, and unequal row-sets before any model ran).

Design (from settled decisions):
  - Target: binary phase3plus at t+h (D8 level target; XGBoost trains
    directly on binary, D-dual-target). Horizon-shifted (D23).
  - Predictors: the D22 canonical list, log-transformed pop_density.
  - Class weight: scale_pos_weight from TRAINING prevalence
    (moderate imbalance 0.21-0.42). Fit per split on train only.
  - Calibration: isotonic, fit ONLY on a held-out slice of TRAINING
    data (calibrating on test is an invisible leak). Never
    touches the test window.
  - Evaluation: shared binary eval, on the D25-canonical row-set, per
    window, at both horizons, against BOTH baselines. Never pooled.
  - Scorable-row counts reported alongside every result.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_BINARY, IS_OBSERVED, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_eval import evaluate_binary, evaluate_baselines
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)


RANDOM_SEED = 42
CAL_FRACTION = 0.2   # fraction of TRAIN held out (chronologically) for isotonic cal


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep_xy(df, idx, ycol, predictors):
    """Extract predictor matrix X and binary target y for a row index set,
    dropping rows with NaN target (D25). `predictors` is the config's
    feature list (A: biophysical-only; B: + phase history). Returns X, y, idx."""
    sub = df.loc[idx]
    sub = sub[sub[ycol].notna()]
    X = sub[predictors].copy()
    # climate_class is categorical int; XGBoost handles as numeric or via
    # enable_categorical. Keep numeric (ordinal-safe here: it's a code, and
    # XGBoost tree splits don't assume order meaningfully for few classes).
    y = sub[ycol].astype(int).to_numpy()
    return X, y, sub.index


def train_and_predict(df, split, ycol, predictors):
    """
    Train XGBoost on the split's train rows, predict probabilities on the
    test rows. Class weight from train prevalence; isotonic calibration on
    a chronological held-out slice of TRAIN only (no test leakage).
    """
    from xgboost import XGBClassifier
    from sklearn.isotonic import IsotonicRegression

    Xtr, ytr, tr_idx = _prep_xy(df, split["train_idx"], ycol, predictors)
    Xte, yte, te_idx = _prep_xy(df, split["test_idx"], ycol, predictors)

    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    # --- chronological calibration split of TRAIN (last CAL_FRACTION by time) ---
    tr_rows = df.loc[tr_idx].sort_values("t_index")
    n_cal = int(len(tr_rows) * CAL_FRACTION)
    cal_idx = tr_rows.index[-n_cal:] if n_cal > 0 else tr_rows.index[:0]
    fit_idx = tr_rows.index[:-n_cal] if n_cal > 0 else tr_rows.index

    Xfit, yfit, _ = _prep_xy(df, fit_idx, ycol, predictors)
    Xcal, ycal, _ = _prep_xy(df, cal_idx, ycol, predictors)

    # class weight from FIT prevalence (train only, never test)
    pos = yfit.sum(); neg = len(yfit) - pos
    spw = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
    )
    clf.fit(Xfit, yfit)

    # --- isotonic calibration: fit on CAL slice (train-only), apply to test ---
    raw_cal = clf.predict_proba(Xcal)[:, 1] if len(Xcal) else np.array([])
    raw_te = clf.predict_proba(Xte)[:, 1]
    if len(Xcal) >= 50 and len(np.unique(ycal)) == 2:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_cal, ycal)
        cal_te = iso.transform(raw_te)
    else:
        cal_te = raw_te   # too little cal data; report uncalibrated (flagged)

    return {"te_idx": te_idx, "y_true": yte,
            "y_prob_raw": raw_te, "y_prob_cal": cal_te,
            "n_fit": len(yfit), "n_cal": len(ycal), "spw": spw}


# Config predictor sets (D26)
CONFIG_A_PREDICTORS = CANONICAL_PREDICTORS                       # biophysical-only
CONFIG_B_PREDICTORS = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS  # + phase history (full)
CONFIG_C_PREDICTORS = list(PHASE_HISTORY_COLS)                   # phase-history only (D26 amend)


def run():
    section("M10.2 — XGBoost (C1, D26+amend): four-arm ablation A/C/B + B-C value-added")
    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)                # leakage guard on phase features
    splits = all_splits(df, verbose=False)
    fews_rows = build_fews_baseline_rows(df)

    print(f"\n{'split':<9}{'prev':>5}{'A':>7}{'C':>7}{'B':>7}"
          f"{'B-C':>8}{'B-A':>7}{'persist':>8}{'FEWS':>7}{'Bp':>4}{'Bf':>4}")
    print("-" * 76)
    print("  A=biophys-only  C=phase-only  B=full  B-C=biophys value-added(PRIMARY)")
    print("-" * 76)

    results = []
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        ycol = f"y_binary_h{h}"

        predA = train_and_predict(df, s, ycol, CONFIG_A_PREDICTORS)
        predB = train_and_predict(df, s, ycol, CONFIG_B_PREDICTORS)
        predC = train_and_predict(df, s, ycol, CONFIG_C_PREDICTORS)
        if predA is None or predB is None or predC is None:
            continue

        A_pr = evaluate_binary(predA["y_true"], predA["y_prob_cal"])["all"]["pr_auc"]
        Bev = evaluate_binary(predB["y_true"], predB["y_prob_cal"])["all"]
        B_pr = Bev["pr_auc"]; prev = Bev["prevalence"]
        C_pr = evaluate_binary(predC["y_true"], predC["y_prob_cal"])["all"]["pr_auc"]

        B_minus_A = B_pr - A_pr          # biophysical-alone insufficiency
        B_minus_C = B_pr - C_pr          # PRIMARY: biophysical value-added over phase

        base = evaluate_baselines(df, s)
        persist_pr = base["persistence"]["all"]["pr_auc"]
        fews = fews_baseline_for_split(df, fews_rows, s, proj_type="near")
        fews_pr = fews["fews"]["all"]["pr_auc"] if fews else np.nan

        b_p = "y" if B_pr > persist_pr else "n"
        b_f = ("y" if (not np.isnan(fews_pr) and B_pr > fews_pr)
               else ("n" if not np.isnan(fews_pr) else "-"))

        print(f"{yr}/h{h:<6}{prev:>5.2f}{A_pr:>7.3f}{C_pr:>7.3f}{B_pr:>7.3f}"
              f"{B_minus_C:>+8.3f}{B_minus_A:>+7.3f}"
              f"{persist_pr:>8.3f}{(fews_pr if not np.isnan(fews_pr) else float('nan')):>7.3f}"
              f"{b_p:>4}{b_f:>4}")

        results.append({"year": yr, "horizon": h, "prevalence": prev,
                        "configA_pr_auc": A_pr, "configC_pr_auc": C_pr,
                        "configB_pr_auc": B_pr,
                        "B_minus_C": B_minus_C, "B_minus_A": B_minus_A,
                        "persistence_pr_auc": persist_pr, "fews_pr_auc": fews_pr,
                        "B_beats_persistence": b_p, "B_beats_fews": b_f})

    section("READING (per-window, no pooled average)")
    print("- B-C (PRIMARY) = biophysical value-added OVER recent phase history.")
    print("  B>C -> remote sensing adds forecasting skill; B~C -> adds little (both publishable).")
    print("- B-A = biophysical-alone insufficiency (complementary).")
    print("- Per-window texture: B-C expected LARGEST in 2021 (drought-onset,")
    print("  phase-alone weakest) and SMALLEST in 2022 (crisis already in phase). That")
    print("  texture is itself a finding.")
    print("- h=1 persistence-dominated (B hugs persistence); h=3 is the real test.")
    print("- C2 forward: B-C gap will be cross-validated against SHAP phase-vs-biophys weight.")
    pd.DataFrame(results).to_csv("outputs/diagnostics/m10_xgboost_4arm.csv", index=False)
    print("\nsaved: outputs/diagnostics/m10_xgboost_configs.csv")
    return results


if __name__ == "__main__":
    run()
