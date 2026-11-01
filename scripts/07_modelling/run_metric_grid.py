"""
======================================================================

run_metric_grid.py  (P3 — full metric grid for Tables 5.4-5.5)

Approved eval-only re-score (option-1 class). Re-runs the C1
four-arm models AND Config D through the EXISTING model modules — no
re-implementation of training — and emits PR-AUC, ROC-AUC and Brier for
every cell, with per-cell n. Same seed (42), same frozen panels, same
rows => identical probabilities => identical PR-AUC.

CONSISTENCY ANCHOR (condition 1)
---------------------------------------
Every re-emitted four-arm PR-AUC is checked against the published figure
for that cell (m10_xgboost_4arm.csv / m10_lightgbm_4arm.csv). Config D
PR-AUC is checked against configD.csv. Any drift beyond 1e-6 RAISES with
the offending cell — a drift signals row-set/seed mismatch to catch
before trusting ROC/Brier. Nothing is written until every anchor matches.

BASELINE BRIER CAVEAT (condition 2)
------------------------------------------
Persistence and FEWS issue BINARY predictions (0/1). Their Brier ==
misclassification rate and their ROC-AUC is a single-operating-point
value. Emitted, flagged binary_predictor=True; Ch5 §5.5 note states it.

CONFIG D ROW-SET (carried from P3 scoping)
------------------------------------------
Config D is scored on its own ACLED-conflict-joined row-set (D27), so its
B differs from the 4-arm B. Its metrics are reported on D's own rows and
tagged rowset="configD" so the write-up never places D beside the 4-arm
table. Only D_minus_B within the configD panel is a valid comparison.

Calls the real module functions:
  - XGBoost : model_xgboost.train_and_predict -> dict key "y_prob_cal"
  - LightGBM: model_lightgbm.train_predict_classifier -> dict key "prob"
  - Config D: model_configD.{load_conflict, train_predict, TEST_WINDOWS,
              TRAIN_COUNTRIES, TEST_COUNTRY} -> dict key "prob"

NO NEW MODELLING. Frozen panels read-only. Emits one diagnostic CSV.

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

import model_xgboost as MX
import model_lightgbm as ML
import model_configD as MD

CONFIG_A = CANONICAL_PREDICTORS
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
CONFIG_C = list(PHASE_HISTORY_COLS)

PUBLISHED_XGB = "outputs/diagnostics/m10_xgboost_4arm.csv"
PUBLISHED_LGB = "outputs/diagnostics/m10_lightgbm_4arm.csv"
PUBLISHED_CD = "outputs/diagnostics/configD.csv"
ANCHOR_TOL = 1e-6


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prob_of(pred):
    """The four model modules use two dict keys for the calibrated probability:
    XGBoost -> 'y_prob_cal'; LightGBM/ConfigD -> 'prob'. Resolve either."""
    if "y_prob_cal" in pred:
        return pred["y_prob_cal"]
    if "prob" in pred:
        return pred["prob"]
    raise KeyError(f"prediction dict has no probability key: {list(pred)}")


def _metrics(pred, *, model, year, horizon, arm, rowset, published_pr):
    ev = evaluate_binary(pred["y_true"], _prob_of(pred))["all"]
    pr, roc, brier, n, pos = (ev["pr_auc"], ev["roc_auc"], ev["brier"],
                              ev["n"], ev["positives"])
    ok = None
    if published_pr is not None and not np.isnan(published_pr):
        ok = abs(pr - published_pr) <= ANCHOR_TOL
        if not ok:
            raise RuntimeError(
                f"PR-AUC ANCHOR MISMATCH [{model} {year}/h{horizon} {arm} "
                f"{rowset}]: re-score {pr:.6f} vs published {published_pr:.6f} "
                f"(|Δ|={abs(pr-published_pr):.2e} > {ANCHOR_TOL}). Row-set/seed "
                f"drift — do NOT trust ROC/Brier until resolved."
            )
    return {"model": model, "year": year, "horizon": horizon, "arm": arm,
            "rowset": rowset, "n": n, "positives": pos, "pr_auc": pr,
            "roc_auc": roc, "brier": brier, "binary_predictor": False,
            "pr_anchor_ok": ok}


def main():
    section("P3 — METRIC GRID re-score (PR-AUC anchor + ROC-AUC + Brier)")

    pub_xgb = pd.read_csv(PUBLISHED_XGB)
    pub_lgb = pd.read_csv(PUBLISHED_LGB)
    pub_cd = pd.read_csv(PUBLISHED_CD)

    def anchor_4arm(model, year, horizon, arm):
        tbl = pub_xgb if model == "xgb" else pub_lgb
        r = tbl[(tbl["year"] == year) & (tbl["horizon"] == horizon)]
        if len(r) == 0:
            return None
        r = r.iloc[0]
        # xgb file: configA/B/C_pr_auc ; lgb file: bare A/B/C
        for cand in ({"A": ["configA_pr_auc", "A"], "B": ["configB_pr_auc", "B"],
                      "C": ["configC_pr_auc", "C"]}[arm]):
            if cand in r.index and pd.notna(r[cand]):
                return float(r[cand])
        return None

    def anchor_cd(scope, model, year, col):
        r = pub_cd[(pub_cd["scope"] == scope) & (pub_cd["model"] == model)
                   & (pub_cd["year"] == year)]
        return float(r.iloc[0][col]) if len(r) else None

    # ---- shared harness prep (identical to every model's run) ----
    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)
    fews_rows = build_fews_baseline_rows(df)

    rows = []
    arms = {"A": CONFIG_A, "B": CONFIG_B, "C": CONFIG_C}

    # ============ C1 four-arm: XGBoost + LightGBM ============
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        ycol = f"y_binary_h{h}"
        for arm, preds in arms.items():
            pX = MX.train_and_predict(df, s, ycol, preds)
            if pX is not None:
                rows.append(_metrics(pX, model="xgb", year=yr, horizon=h, arm=arm,
                                     rowset="4arm", published_pr=anchor_4arm("xgb", yr, h, arm)))
            pL = ML.train_predict_classifier(df, s, ycol, preds)
            if pL is not None:
                rows.append(_metrics(pL, model="lgb", year=yr, horizon=h, arm=arm,
                                     rowset="4arm", published_pr=anchor_4arm("lgb", yr, h, arm)))

        # baselines (binary predictors) on this split
        base = evaluate_baselines(df, s)
        te = df.loc[base["scored_idx"]]
        y_true = te[ycol].to_numpy()
        persist = te[TARGET_BINARY].to_numpy().astype(float)
        pe = evaluate_binary(y_true, persist)["all"]
        rows.append({"model": "persistence", "year": yr, "horizon": h, "arm": "-",
                     "rowset": "persistence_set", "n": pe["n"], "positives": pe["positives"],
                     "pr_auc": pe["pr_auc"], "roc_auc": pe["roc_auc"], "brier": pe["brier"],
                     "binary_predictor": True, "pr_anchor_ok": None})
        fews = fews_baseline_for_split(df, fews_rows, s, proj_type="near")
        if fews is not None:
            fe = fews["fews"]["all"]
            rows.append({"model": "fews", "year": yr, "horizon": h, "arm": "-",
                         "rowset": "fews_aligned", "n": fews["n_aligned"],
                         "positives": fe["positives"], "pr_auc": fe["pr_auc"],
                         "roc_auc": fe["roc_auc"], "brier": fe["brier"],
                         "binary_predictor": True, "pr_anchor_ok": None})

    # ============ Config D (own conflict-joined row-set) ============
    section("Config D re-score (own D27 conflict row-set)")
    dfD = df.copy()
    dfD, CONFLICT_COLS = MD.load_conflict(dfD)
    B_cols = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
    D_cols = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS + CONFLICT_COLS

    # in-pool
    for s in [s for s in all_splits(dfD, verbose=False) if s["horizon"] == 3]:
        yr = s["test_year"]
        tr = dfD.loc[s["train_idx"]]; te = dfD.loc[s["test_idx"]]
        for which in ("xgb", "lgb"):
            pB = MD.train_predict(tr, te, "y_binary_h3", B_cols, which)
            pD = MD.train_predict(tr, te, "y_binary_h3", D_cols, which)
            if pB:
                rows.append(_metrics(pB, model=which, year=yr, horizon=3, arm="B",
                                     rowset="configD", published_pr=anchor_cd("in_pool", which, yr, "B")))
            if pD:
                rows.append(_metrics(pD, model=which, year=yr, horizon=3, arm="D",
                                     rowset="configD", published_pr=anchor_cd("in_pool", which, yr, "D")))

    # transfer (train KE+ET, test Somalia)
    train_all = dfD[dfD["country"].isin(MD.TRAIN_COUNTRIES)]
    test_all = dfD[dfD["country"] == MD.TEST_COUNTRY]
    for yr, (t_lo, t_hi) in MD.TEST_WINDOWS.items():
        te = test_all[(test_all["t_index"] + 3 >= t_lo) & (test_all["t_index"] + 3 <= t_hi)]
        tr = train_all[(train_all["t_index"] + 3) < t_lo]
        if len(te) == 0 or len(tr) == 0:
            continue
        for which in ("xgb", "lgb"):
            pB = MD.train_predict(tr, te, "y_binary_h3", B_cols, which)
            pD = MD.train_predict(tr, te, "y_binary_h3", D_cols, which)
            if pB:
                rows.append(_metrics(pB, model=which, year=yr, horizon=3, arm="B",
                                     rowset="configD_transfer", published_pr=anchor_cd("transfer", which, yr, "B")))
            if pD:
                rows.append(_metrics(pD, model=which, year=yr, horizon=3, arm="D",
                                     rowset="configD_transfer", published_pr=anchor_cd("transfer", which, yr, "D")))

    out = pd.DataFrame(rows)

    section("PR-AUC ANCHOR CHECK (condition 1)")
    chk = out[out["pr_anchor_ok"].notna()]
    n_ok, n_tot = int(chk["pr_anchor_ok"].sum()), len(chk)
    print(f"model cells checked: {n_tot} | anchors matched: {n_ok}")
    print("✓ all re-scored PR-AUCs match published within 1e-6."
          if n_ok == n_tot else "‼ ANCHOR FAILURE — see raised error above.")

    section("METRIC GRID (per cell)")
    show = out.copy()
    for c in ("pr_auc", "roc_auc", "brier"):
        show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    print(show[["model", "year", "horizon", "arm", "rowset", "n", "positives",
                "pr_auc", "roc_auc", "brier", "binary_predictor"]].to_string(index=False))

    dest = "outputs/diagnostics/m10_metric_grid.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved: {dest}")
    print("\nNOTE (condition 2): persistence & fews are binary predictors — "
          "Brier=misclassification rate, ROC-AUC single-operating-point. "
          "Flagged binary_predictor=True; Ch5 §5.5 states this.")
    print("NOTE: rowset=configD/configD_transfer sit on the D27 conflict row-set; "
          "do NOT place their absolute values beside the 4arm table (only D_minus_B "
          "within-panel is valid).")


if __name__ == "__main__":
    main()
