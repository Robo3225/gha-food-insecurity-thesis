"""
======================================================================

run_p12_rowmatched.py  (P12 — model-vs-baseline row-set equalisation)

Eval-only re-score (identical class to P1 and P3): re-runs
the frozen XGB-B and LGB-B predictions and RE-SCORES them on each baseline's
exact D25 row-set, so every "model beats persistence / FEWS" number is
row-matched. No retraining, no hyperparameter change, no re-freeze.

THE GAP (confirmed against m10_metric_grid.csv)
-------------------------------------------------------
The metric grid scores models on the 4arm rowset (drops only NaN-target),
but baselines on their own rowsets (persistence_set additionally drops
NaN-persistence-input; fews_aligned is the period-matched subset). So
"XGB-B 0.733 beats FEWS 0.514" is a cross-rowset comparison D25 forbids for
model-vs-baseline. P12 fixes it.

WHAT THIS DOES
--------------
For XGB-B and LGB-B (Config B), at h=3 and h=1, in-pool AND Somalia transfer:
  (a) score the model on the PERSISTENCE rowset (evaluate_baselines' scored_idx)
      -> model-vs-persistence is row-matched.
  (b) [h=3] score the model on the FEWS-ALIGNED rowset (the exact rows
      fews_baseline_for_split scores FEWS on) -> model / persistence / FEWS
      all on ONE row-set.
Reports PR-AUC, ROC-AUC, Brier, n, positives per cell, rowset-labelled.

CONSISTENCY ANCHOR
------------------
The model on the 4arm rowset must reproduce the published m10_metric_grid.csv
PR-AUC within 1e-6 (proves predictions are identical to the banked run before
any subset is trusted). Raises on drift.

INTEGRITY
---------
If a margin flips (model no longer beats a baseline on row-matched rows), it
is reported AS-IS. A row-matched loss is the honest number. Rows are never
adjusted to preserve a win.

Calls the real module functions (identical predictions):
  XGBoost : model_xgboost.train_and_predict            -> dict key "y_prob_cal", "te_idx"
  LightGBM: model_lightgbm.train_predict_classifier    -> dict key "prob", "te_idx"
  Transfer: model_configD.train_predict is NOT used (that is the conflict
            rowset); instead c3_transfer.transfer_train_predict is reused so
            the transfer predictions match the c3_transfer.csv banked run.

NO NEW MODELLING. Frozen panels read-only. Emits one diagnostic CSV.

Author: Robert Odingo MSc Thesis
======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, TARGET_BINARY, IS_OBSERVED, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits, TEST_WINDOWS
from harness_eval import evaluate_binary, evaluate_baselines
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

import model_xgboost as MX
import model_lightgbm as ML
import c3_transfer as C3   # reuse transfer_train_predict so transfer preds match c3_transfer.csv

CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
ANCHOR_TOL = 1e-6
PUBLISHED_GRID = "outputs/diagnostics/m10_metric_grid.csv"
TRAIN_COUNTRIES = ["kenya", "ethiopia"]
TEST_COUNTRY = "somalia"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prob_of(pred):
    if "y_prob_cal" in pred:
        return pred["y_prob_cal"]
    if "prob" in pred:
        return pred["prob"]
    raise KeyError(f"no probability key in {list(pred)}")


def _fews_aligned_index(df, fews_rows, split):
    """
    Reproduce fews_baseline_for_split's aligned merge, but RETURN THE DF INDEX
    of the aligned test rows (so the model can be scored on exactly them).
    Mirrors harness_fews lines 164-195 verbatim; adds only the index capture.
    """
    h = split["horizon"]
    if h != 3:
        return None
    te = df.loc[split["test_idx"]].copy()
    te["target_ym_int"] = te["year"].astype(int) * 12 + (te["month"].astype(int) - 1) + h
    fr = fews_rows[fews_rows["proj_type"] == "near"][
        ["slug", "admin_code", "target_ym_int", "fews_pred_binary"]]
    te = te.reset_index().rename(columns={"index": "_df_idx"})
    merged = te.merge(
        fr, left_on=["country", UNIT_COL, "target_ym_int"],
        right_on=["slug", "admin_code", "target_ym_int"], how="inner",
    )
    ycol = f"y_binary_h{h}"
    merged = merged[merged[ycol].notna()]
    merged = merged[merged[TARGET_BINARY].notna()]
    if len(merged) == 0:
        return None
    return merged.set_index("_df_idx")


def _score_model_on_index(pred, df, want_idx, ycol):
    """Restrict model predictions (te_idx-aligned) to want_idx, then evaluate."""
    te_idx = np.asarray(pred["te_idx"])
    prob = np.asarray(_prob_of(pred))
    ptab = pd.Series(prob, index=te_idx)
    common = ptab.index.intersection(want_idx)
    if len(common) == 0:
        return None
    y_true = df.loc[common, ycol].to_numpy()
    y_prob = ptab.loc[common].to_numpy()
    obs = df.loc[common, IS_OBSERVED].to_numpy().astype(bool) if IS_OBSERVED in df.columns else None
    ev = evaluate_binary(y_true, y_prob, observed_mask=obs)["all"]
    return ev, len(common)


def _row(model, scope, year, horizon, rowset, ev, n):
    return {"model": model, "scope": scope, "year": year, "horizon": horizon,
            "rowset": rowset, "n": n, "positives": ev["positives"],
            "pr_auc": ev["pr_auc"], "roc_auc": ev["roc_auc"], "brier": ev["brier"]}


def main():
    section("P12 — model-vs-baseline row-matched re-score (eval-only)")

    grid = pd.read_csv(PUBLISHED_GRID)

    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)
    fews_rows = build_fews_baseline_rows(df)

    rows = []
    train_fns = {"xgb": MX.train_and_predict, "lgb": ML.train_predict_classifier}

    # ================= IN-POOL =================
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        ycol = f"y_binary_h{h}"

        base = evaluate_baselines(df, s)
        persist_idx = pd.Index(base["scored_idx"])          # persistence rowset
        fews_tab = _fews_aligned_index(df, fews_rows, s)    # None at h=1
        fews_idx = fews_tab.index if fews_tab is not None else None

        for m, fn in train_fns.items():
            pred = fn(df, s, ycol, CONFIG_B)
            if pred is None:
                continue

            # --- anchor: 4arm rowset must reproduce the published PR-AUC ---
            all_idx = pd.Index(np.asarray(pred["te_idx"]))
            ev4, n4 = _score_model_on_index(pred, df, all_idx, ycol)
            pub = grid[(grid.model == m) & (grid.year == yr) & (grid.horizon == h)
                       & (grid.arm == "B") & (grid.rowset == "4arm")]
            if len(pub):
                pubpr = float(pub.iloc[0]["pr_auc"])
                if abs(ev4["pr_auc"] - pubpr) > ANCHOR_TOL:
                    raise RuntimeError(
                        f"ANCHOR MISMATCH {m} {yr}/h{h} 4arm: re-score "
                        f"{ev4['pr_auc']:.6f} vs published {pubpr:.6f}")

            # --- (a) persistence rowset ---
            r = _score_model_on_index(pred, df, persist_idx, ycol)
            if r:
                rows.append(_row(m, "in_pool", yr, h, "persistence_set", r[0], r[1]))

            # --- (b) FEWS-aligned rowset (h=3) ---
            if fews_idx is not None:
                r = _score_model_on_index(pred, df, fews_idx, ycol)
                if r:
                    rows.append(_row(m, "in_pool", yr, h, "fews_aligned", r[0], r[1]))

        # persistence + FEWS baselines on the fews-aligned rows (for the 3-way)
        if fews_tab is not None:
            fb = fews_baseline_for_split(df, fews_rows, s, "near")
            if fb:
                rows.append(_row("persistence", "in_pool", yr, h, "fews_aligned",
                                 fb["persistence_on_fews_rows"]["all"], fb["n_aligned"]))
                rows.append(_row("fews", "in_pool", yr, h, "fews_aligned",
                                 fb["fews"]["all"], fb["n_aligned"]))
        # persistence baseline on the persistence rowset
        rows.append(_row("persistence", "in_pool", yr, h, "persistence_set",
                         base["persistence"]["all"], base["n_scorable"]))

    # ================= SOMALIA TRANSFER =================
    section("Somalia transfer — same row-matched re-score")
    train_all = df[df["country"].isin(TRAIN_COUNTRIES)].copy()
    test_all = df[df["country"] == TEST_COUNTRY].copy()

    for h in (1, 3):
        ycol = f"y_binary_h{h}"
        for yr, (t_lo, t_hi) in TEST_WINDOWS.items():
            pred_t = test_all["t_index"] + h
            te_win = test_all[(pred_t >= t_lo) & (pred_t <= t_hi)].copy()
            tr_win = train_all[(train_all["t_index"] + h) < t_lo].copy()
            if len(te_win) == 0 or len(tr_win) == 0:
                continue

            # Somalia persistence rowset: te_win rows with non-NaN target AND non-NaN persist
            sub = te_win[te_win[ycol].notna() & te_win[TARGET_BINARY].notna()]
            so_persist_idx = sub.index
            # transfer 4-arm rowset n = te_win rows with non-NaN target (what c3_transfer scores B on)
            n_4arm_transfer = int(te_win[ycol].notna().sum())

            # FEWS-aligned Somalia rows (h=3)
            so_fews_tab = None
            if h == 3:
                split_like = {"test_idx": te_win.index, "horizon": h,
                              "t_lo": t_lo, "t_hi": t_hi, "test_year": yr}
                so_fews_tab = _fews_aligned_index(df, fews_rows, split_like)

            for m in ("xgb", "lgb"):
                pB = C3.transfer_train_predict(tr_win, te_win, ycol, CONFIG_B, m)
                if pB is None:
                    continue
                te_idx = np.asarray(pB["te_idx"]); prob = np.asarray(pB["prob"])
                ptab = pd.Series(prob, index=te_idx)

                # 4arm-equivalent (all non-NaN-target rows) — anchor to c3_transfer.csv B
                allc = ptab.index  # transfer_train_predict already dropped NaN-target
                ev = evaluate_binary(df.loc[allc, ycol].to_numpy(), ptab.loc[allc].to_numpy())["all"]
                rows.append(_row(m, "transfer", yr, h, "transfer_4arm", ev, len(allc)))

                # (a) Somalia persistence rowset
                common = ptab.index.intersection(so_persist_idx)
                if len(common):
                    ev = evaluate_binary(df.loc[common, ycol].to_numpy(),
                                         ptab.loc[common].to_numpy())["all"]
                    rows.append(_row(m, "transfer", yr, h, "so_persistence_set", ev, len(common)))

                # (b) Somalia FEWS-aligned rowset
                if so_fews_tab is not None:
                    common = ptab.index.intersection(so_fews_tab.index)
                    if len(common):
                        ev = evaluate_binary(df.loc[common, ycol].to_numpy(),
                                             ptab.loc[common].to_numpy())["all"]
                        rows.append(_row(m, "transfer", yr, h, "so_fews_aligned", ev, len(common)))

            # Somalia persistence baseline on its own rowset
            if len(so_persist_idx):
                yt = df.loc[so_persist_idx, ycol].to_numpy()
                pp = df.loc[so_persist_idx, TARGET_BINARY].to_numpy().astype(float)
                rows.append(_row("persistence", "transfer", yr, h, "so_persistence_set",
                                 evaluate_binary(yt, pp)["all"], len(so_persist_idx)))
            # Somalia persistence + FEWS on the fews-aligned rows
            if so_fews_tab is not None:
                fb = fews_baseline_for_split(df, fews_rows,
                    {"test_idx": te_win.index, "horizon": h, "t_lo": t_lo, "t_hi": t_hi,
                     "test_year": yr}, "near")
                if fb:
                    rows.append(_row("persistence", "transfer", yr, h, "so_fews_aligned",
                                     fb["persistence_on_fews_rows"]["all"], fb["n_aligned"]))
                    rows.append(_row("fews", "transfer", yr, h, "so_fews_aligned",
                                     fb["fews"]["all"], fb["n_aligned"]))
            # record the transfer 4arm n explicitly
            rows.append({"model": "_meta_n", "scope": "transfer", "year": yr, "horizon": h,
                         "rowset": "transfer_4arm", "n": n_4arm_transfer, "positives": np.nan,
                         "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan})

    out = pd.DataFrame(rows)
    section("RESULT")
    show = out.copy()
    for c in ("pr_auc", "roc_auc", "brier"):
        show[c] = show[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    print(show.to_string(index=False))
    dest = "outputs/diagnostics/m10_p12_rowmatched.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved: {dest}")

    # --- version string for the TFT appendix ---
    section("TFT gate build — pytorch-forecasting version (Ch5 appendix)")
    try:
        import pytorch_forecasting as pf
        print(f"pytorch_forecasting.__version__ = {pf.__version__}")
    except Exception as e:
        print(f"could not import pytorch_forecasting here: {e}")
    try:
        import torch, lightning
        print(f"torch = {torch.__version__}; lightning = {lightning.__version__}")
    except Exception as e:
        print(f"torch/lightning: {e}")


if __name__ == "__main__":
    main()
