"""
======================================================================

c3_transfer.py  (M10.5 / C3 — cross-country transferability)

Train on Kenya+Ethiopia, test on Somalia (unseen country). The
culmination: measures how the whole apparatus degrades on the hardest
unseen deployment context, DECOMPOSED across the three D17 heterogeneity
axes rather than reported as a single gap.

Design (C3_preregistration):
  - PRIMARY split: pure geographic. Train ALL KE+ET rows, test ALL SO
    rows. Isolates country-transfer (no temporal confound) -> attributable.
  - Four-arm ablation under transfer (A/C/B); CENTRAL measurement = B-C
    on Somalia (does the feature-outcome relationship transfer).
  - Baselines: Somalia's OWN persistence + OWN FEWS projection.
  - Reused leakage-safe machinery: D23 horizon target, D22 canonical
    predictors, phase history, shared evaluation. Somalia features built
    within-SO-unit; KE+ET supplies weights only (intended transfer).
  - Per-window on Somalia (2021/2022/2023), no pooled average.

Continuous-probability scoring throughout (like C1). Conformal under
transfer handled in c3_conformal (C2<->C3 capstone). Degradation
decomposition assembled in the C3 summary from the banked axes.

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
    load_all, CANONICAL_PREDICTORS, TARGET_BINARY, IS_OBSERVED, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_eval import evaluate_binary
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
PHASE3_THRESHOLD = 3
TRAIN_COUNTRIES = ["kenya", "ethiopia"]
TEST_COUNTRY = "somalia"
TEST_WINDOWS = {2021: (36, 47), 2022: (48, 59), 2023: (60, 71)}  # t_index ranges

CONFIG_A = CANONICAL_PREDICTORS
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
CONFIG_C = list(PHASE_HISTORY_COLS)


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(sub, ycol, predictors):
    sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].astype(int).to_numpy(), sub.index


def transfer_train_predict(train_df, test_df, ycol, predictors, which="xgb"):
    """
    Train on ALL KE+ET rows (with valid horizon target); predict on the
    Somalia test-window rows. Class weight + isotonic calibration from a
    chronological slice of the TRAINING (KE+ET) data — never test.
    """
    from sklearn.isotonic import IsotonicRegression

    Xtr, ytr, tr_idx = _prep(train_df, ycol, predictors)
    Xte, yte, te_idx = _prep(test_df, ycol, predictors)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None

    # calibration slice = latest chronological slice of TRAIN (KE+ET), train-only
    tr_sorted = train_df.loc[tr_idx].sort_values("t_index")
    n_cal = int(len(tr_sorted) * CAL_FRACTION)
    cal_idx = tr_sorted.index[-n_cal:] if n_cal else tr_sorted.index[:0]
    fit_idx = tr_sorted.index[:-n_cal] if n_cal else tr_sorted.index
    Xfit, yfit, _ = _prep(train_df.loc[fit_idx], ycol, predictors)
    Xcal, ycal, _ = _prep(train_df.loc[cal_idx], ycol, predictors)

    pos = yfit.sum(); spw = ((len(yfit) - pos) / pos) if pos > 0 else 1.0

    if which == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                          eval_metric="aucpr", random_state=RANDOM_SEED,
                          n_jobs=-1, tree_method="hist")
    else:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)
    m.fit(Xfit, yfit)

    raw_te = m.predict_proba(Xte)[:, 1]
    if len(Xcal) >= 50 and len(np.unique(ycal)) == 2:
        raw_cal = m.predict_proba(Xcal)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(raw_cal, ycal)
        prob = iso.transform(raw_te)
    else:
        prob = raw_te
    return {"te_idx": te_idx, "y_true": yte, "prob": prob}


def somalia_persistence(test_df, ycol_bin):
    """Somalia's OWN persistence: predict phase at t (in-hand) for t+h target.
    No training — pure Somalia autocorrelation. Baseline floor."""
    sub = test_df[test_df[ycol_bin].notna()]
    return sub[TARGET_BINARY].to_numpy().astype(float), sub[ycol_bin].to_numpy(), sub.index


def run():
    section("M10.5 / C3 — TRANSFER: train KE+ET, test Somalia (per-window, four-arm)")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    fews_rows = build_fews_baseline_rows(df)

    train_all = df[df["country"].isin(TRAIN_COUNTRIES)].copy()
    test_all = df[df["country"] == TEST_COUNTRY].copy()
    print(f"train (KE+ET): {len(train_all):,} rows | test (SO): {len(test_all):,} rows")

    print(f"\n{'win':<8}{'model':>6}{'A':>7}{'C':>7}{'B':>7}{'B-C':>8}{'B-A':>7}"
          f"{'SO_pers':>8}{'SO_FEWS':>8}{'Bp':>4}{'Bf':>4}")
    print("-" * 74)

    results = []
    for h in (1, 3):
        ycol_bin = f"y_binary_h{h}"
        for yr, (t_lo, t_hi) in TEST_WINDOWS.items():
            # Somalia test rows whose PREDICTED period (t+h) falls in the window
            pred_t = test_all["t_index"] + h
            te_win = test_all[(pred_t >= t_lo) & (pred_t <= t_hi)].copy()
            # Train on KE+ET rows whose predicted period precedes the test window
            # start (chronology-safe: no training label observed in/after SO test period)
            tr_pred_t = train_all["t_index"] + h
            tr_win = train_all[tr_pred_t < t_lo].copy()
            if len(te_win) == 0 or len(tr_win) == 0:
                continue

            for which in ("xgb", "lgb"):
                pA = transfer_train_predict(tr_win, te_win, ycol_bin, CONFIG_A, which)
                pB = transfer_train_predict(tr_win, te_win, ycol_bin, CONFIG_B, which)
                pC = transfer_train_predict(tr_win, te_win, ycol_bin, CONFIG_C, which)
                if not (pA and pB and pC):
                    continue
                A = evaluate_binary(pA["y_true"], pA["prob"])["all"]["pr_auc"]
                B = evaluate_binary(pB["y_true"], pB["prob"])["all"]["pr_auc"]
                C = evaluate_binary(pC["y_true"], pC["prob"])["all"]["pr_auc"]

                # Somalia's own persistence floor
                yp, ytrue_p, _ = somalia_persistence(te_win, ycol_bin)
                so_pers = evaluate_binary(ytrue_p, yp)["all"]["pr_auc"]

                # Somalia's own FEWS (h=3 only), on the aligned SO rows
                so_fews = np.nan
                if h == 3:
                    split_like = {"test_idx": te_win.index, "horizon": h,
                                  "t_lo": t_lo, "t_hi": t_hi, "test_year": yr}
                    fb = fews_baseline_for_split(df, fews_rows, split_like, "near")
                    so_fews = fb["fews"]["all"]["pr_auc"] if fb else np.nan

                bp = "y" if B > so_pers else "n"
                bf = ("y" if (not np.isnan(so_fews) and B > so_fews)
                      else ("n" if not np.isnan(so_fews) else "-"))

                print(f"{yr}/h{h:<4}{which:>6}{A:>7.3f}{C:>7.3f}{B:>7.3f}"
                      f"{B-C:>+8.3f}{B-A:>+7.3f}{so_pers:>8.3f}"
                      f"{(so_fews if not np.isnan(so_fews) else float('nan')):>8.3f}"
                      f"{bp:>4}{bf:>4}")
                results.append({"year": yr, "horizon": h, "model": which,
                                "A": A, "C": C, "B": B, "B_minus_C": B-C, "B_minus_A": B-A,
                                "so_persistence": so_pers, "so_fews": so_fews,
                                "B_beats_so_pers": bp, "B_beats_so_fews": bf})

    section("READING (per-window transfer, no pooled — pre-registered outcomes)")
    print("- B (transferred model) vs SO_pers: does a KE+ET model beat Somalia's own floor?")
    print("- B vs SO_FEWS (h=3): does it beat Somalia's own operational ceiling?")
    print("- B-C on Somalia = CENTRAL: does biophysical value-added TRANSFER?")
    print("  pre-registered: B-C~in-pool = portable; B-C~0 = only autocorrelation transfers.")
    print("- Compare B-C-on-SO to in-pool B-C (2021 +.117 / 2022 +.036 / 2023 +.021).")
    print("- Degradation to be DECOMPOSED across D17 axes (data-quality / conflict-shift /")
    print("  uncertainty) in the C3 summary, not reported as one gap.")
    pd.DataFrame(results).to_csv("outputs/diagnostics/c3_transfer.csv", index=False)
    print("\nsaved: outputs/diagnostics/c3_transfer.csv")
    return results


if __name__ == "__main__":
    run()
