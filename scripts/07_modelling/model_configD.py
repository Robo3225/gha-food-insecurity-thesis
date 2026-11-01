"""
======================================================================

model_configD.py  (Config D — conflict ablation, D27; in-pool + C3 transfer)

The conflict-value extension. Config D = biophysical + phase + CONFLICT.
Tests the highest-value question C3 exposed: does conflict recover the
2021 Somalia onset collapse (where biophysical value-added B-C -> ~0)?

D27 pre-registration:
  - Config D is an ABLATION ARM, not canonical (no re-freeze; D22-compliant).
  - Conflict features (density, intensity) + lags {1,3,6,9,12} (D19 uniform-
    lag: let the model find conflict's operative lead/lag; displacement and
    market collapse unfold over months). Built within-unit, backward, leakage-safe.
  - Conflict artifact joined at the harness (data/processed/conflict/), panels
    untouched.
  - TWO runs (targeted only):
      (1) IN-POOL: Config D vs Config B, walk-forward, h=3  -> D-B in-pool
      (2) C3 TRANSFER: train KE+ET, test Somalia, h=3, per-window -> D-B on transfer
    Headline = 2021 Somalia onset cell.
  - Both outcomes pre-committed reportable; conflict-transferability sub-question
    (helps-everywhere vs helps-in-pool-not-on-transfer) reported explicitly.

Runs in gee_env. Reuses the leakage-safe harness; conflict is a 4th predictor set.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_BINARY, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_eval import evaluate_binary, evaluate_baselines
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
LAGS = (1, 3, 6, 9, 12)
CONFLICT_BASE = ["conflict_density", "conflict_intensity"]

# resolve the conflict-artifact dir without depending on harness_data.ROOT:
# walk up from cwd / this file to find data/processed/conflict.
def _conflict_dir():
    from pathlib import Path
    for base in [Path.cwd(), Path(__file__).resolve().parent]:
        for up in range(0, 5):
            cand = base
            for _ in range(up):
                cand = cand.parent
            d = cand / "data" / "processed" / "conflict"
            if d.exists():
                return d
    raise RuntimeError("data/processed/conflict not found; run build_conflict_features first")
TRAIN_COUNTRIES = ["kenya", "ethiopia"]
TEST_COUNTRY = "somalia"
TEST_WINDOWS = {2021: (36, 47), 2022: (48, 59), 2023: (60, 71)}


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def load_conflict(df):
    """Join the standalone conflict artifact + build within-unit backward lags
    (leakage-safe, D19 discipline). Returns df with conflict cols + the conflict
    predictor name list."""
    cdir = _conflict_dir()
    frames = []
    for slug in ["kenya", "ethiopia", "somalia"]:
        c = pd.read_csv(cdir / f"{slug}_conflict_features.csv")
        frames.append(c)
    conf = pd.concat(frames, ignore_index=True)
    df = df.merge(conf, on=[UNIT_COL, "year_month"], how="left")

    # fail loud if the join left conflict nulls on rows that should have them
    miss = df[CONFLICT_BASE].isna().any(axis=1).sum()
    if miss:
        # conflict features cover full window; nulls => join key mismatch
        print(f"  WARNING: {miss} rows missing conflict after join (filling 0 — "
              f"verify unit-key alignment)")
        df[CONFLICT_BASE] = df[CONFLICT_BASE].fillna(0.0)

    # within-unit backward lags (same discipline as build_lags / phase history)
    df = df.sort_values([UNIT_COL, "t_index"]).reset_index(drop=True)
    conflict_cols = list(CONFLICT_BASE)
    for base in CONFLICT_BASE:
        for k in LAGS:
            col = f"{base}_lag{k}"
            df[col] = df.groupby(UNIT_COL)[base].shift(k)
            conflict_cols.append(col)
    return df, conflict_cols


def _prep(sub, ycol, predictors):
    sub = sub[sub[ycol].notna()]
    X = sub[predictors].copy()
    # conflict lags are NaN for first k months per unit -> fill 0 (no prior
    # conflict observed = no exposure; distinct from missing). Backward-only.
    X = X.fillna(0.0)
    return X, sub[ycol].astype(int).to_numpy(), sub.index


def train_predict(train_df, test_df, ycol, predictors, which="xgb"):
    from sklearn.isotonic import IsotonicRegression
    Xtr, ytr, tr_idx = _prep(train_df, ycol, predictors)
    Xte, yte, te_idx = _prep(test_df, ycol, predictors)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None
    tr_sorted = train_df.loc[tr_idx].sort_values("t_index")
    n_cal = int(len(tr_sorted) * CAL_FRACTION)
    fit_idx = tr_sorted.index[:-n_cal] if n_cal else tr_sorted.index
    cal_idx = tr_sorted.index[-n_cal:] if n_cal else tr_sorted.index[:0]
    Xfit, yfit, _ = _prep(train_df.loc[fit_idx], ycol, predictors)
    Xcal, ycal, _ = _prep(train_df.loc[cal_idx], ycol, predictors)
    pos = yfit.sum(); spw = ((len(yfit)-pos)/pos) if pos > 0 else 1.0
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
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(Xcal)[:, 1], ycal)
        prob = iso.transform(raw_te)
    else:
        prob = raw_te
    return {"y_true": yte, "prob": prob}


def run():
    section("Config D — conflict ablation (D27): in-pool D-B + C3 transfer D-B")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    df, CONFLICT_COLS = load_conflict(df)
    print(f"conflict predictors ({len(CONFLICT_COLS)}): {CONFLICT_COLS}")

    B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS            # Config B (canonical)
    D = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS + CONFLICT_COLS  # Config D

    # ---------- (1) IN-POOL: D vs B ----------
    section("(1) IN-POOL — Config D vs Config B (h=3, per window)")
    splits = all_splits(df, verbose=False)
    print(f"{'win':<9}{'model':>6}{'B':>7}{'D':>7}{'D-B':>8}{'persist':>8}")
    print("-" * 45)
    inpool = []
    for s in [s for s in splits if s["horizon"] == 3]:
        yr = s["test_year"]
        tr = df.loc[s["train_idx"]]; te = df.loc[s["test_idx"]]
        ycol = "y_binary_h3"
        for which in ("xgb", "lgb"):
            pB = train_predict(tr, te, ycol, B, which)
            pD = train_predict(tr, te, ycol, D, which)
            if not (pB and pD): continue
            bpr = evaluate_binary(pB["y_true"], pB["prob"])["all"]["pr_auc"]
            dpr = evaluate_binary(pD["y_true"], pD["prob"])["all"]["pr_auc"]
            base = evaluate_baselines(df, s)["persistence"]["all"]["pr_auc"]
            print(f"{yr}/h3{'':>2}{which:>6}{bpr:>7.3f}{dpr:>7.3f}{dpr-bpr:>+8.3f}{base:>8.3f}")
            inpool.append({"scope":"in_pool","year":yr,"model":which,
                           "B":bpr,"D":dpr,"D_minus_B":dpr-bpr})

    # ---------- (2) C3 TRANSFER: D vs B, train KE+ET test SO ----------
    section("(2) C3 TRANSFER — train KE+ET, test Somalia (h=3, per window)")
    print("HEADLINE = 2021 Somalia onset (where B-C collapsed to ~0)")
    train_all = df[df["country"].isin(TRAIN_COUNTRIES)]
    test_all = df[df["country"] == TEST_COUNTRY]
    print(f"\n{'win':<9}{'model':>6}{'B':>7}{'D':>7}{'D-B':>8}{'recovers?':>11}")
    print("-" * 48)
    transfer = []
    for yr, (t_lo, t_hi) in TEST_WINDOWS.items():
        te = test_all[(test_all["t_index"]+3 >= t_lo) & (test_all["t_index"]+3 <= t_hi)]
        tr = train_all[(train_all["t_index"]+3) < t_lo]
        if len(te) == 0 or len(tr) == 0: continue
        ycol = "y_binary_h3"
        for which in ("xgb", "lgb"):
            pB = train_predict(tr, te, ycol, B, which)
            pD = train_predict(tr, te, ycol, D, which)
            if not (pB and pD): continue
            bpr = evaluate_binary(pB["y_true"], pB["prob"])["all"]["pr_auc"]
            dpr = evaluate_binary(pD["y_true"], pD["prob"])["all"]["pr_auc"]
            gap = dpr - bpr
            flag = ("RECOVERS" if (yr == 2021 and gap > 0.02) else
                    ("+"if gap>0 else "~/-"))
            print(f"{yr}/h3{'':>2}{which:>6}{bpr:>7.3f}{dpr:>7.3f}{gap:>+8.3f}{flag:>11}")
            transfer.append({"scope":"transfer","year":yr,"model":which,
                             "B":bpr,"D":dpr,"D_minus_B":gap})

    section("READING — pre-registered (D27)")
    print("- D-B in-pool: does conflict add value WHERE THE MODEL TRAINS?")
    print("- D-B on transfer @2021: does conflict RECOVER the Somalia onset collapse?")
    print("- Conflict-transferability sub-question (explicit):")
    print("    D-B>0 in-pool AND on transfer -> conflict helps everywhere (portable)")
    print("    D-B>0 in-pool but ~0 on transfer -> conflict signal itself context-")
    print("    specific (learned on quieter KE+ET, doesn't capture Somalia's regime)")
    print("    -> another C3 heterogeneity instance.")
    pd.DataFrame(inpool + transfer).to_csv("outputs/diagnostics/configD.csv", index=False)
    print("\nsaved: outputs/diagnostics/configD.csv")


if __name__ == "__main__":
    run()
