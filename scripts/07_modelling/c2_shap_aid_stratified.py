"""
======================================================================

c2_shap_aid_stratified.py  (C2 confound discussion — D3 aid confound)

Post-hoc SHAP re-slice (in-bounds by construction:
no model retraining, no panel re-freeze, nothing settled reopened). Splits
the tree-SHAP attributions on the humanitarian-assistance flag (fews_ha,
present in the panels as metadata, excluded as a predictor per D15) and
compares biophysical attribution shares in AID-HEAVY vs AID-LIGHT units.

Purpose (D3 confound). The target is post-assistance fews_ipc, so SHAP
attributions may partly reflect WHERE AID WAS DELIVERED rather than solely
where conditions deteriorated. If biophysical attribution is similar across
aid strata, the model keys on deterioration (aid-invariant); if it diverges,
the aid confound is plausibly operating.

CONDITIONS:
  1. SEVERITY CONFOUND reported prominently: fews_ha correlates with severity
     (aid flows to worse-off units), so aid-heavy vs aid-light is PARTLY
     crisis vs non-crisis. A divergence could be the aid confound OR the
     known onset/severity behaviour. SUGGESTIVE about D3, NOT a clean
     causal separation.
  2. Framed as STRENGTHENING the confound DISCUSSION (EU AI Act Art.13
     transparency), NOT resolving the confound.

Reuses the C2 tree-SHAP machinery; only the aggregation changes (group x
fews_ha). Runs in gee_env.

Author: Robert Odingo MSc Thesis
======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import load_all, CANONICAL_PREDICTORS, UNIT_COL
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS
PHASE_FEATURES = set(PHASE_HISTORY_COLS)
STATIC_FEATURES = {"pop_density", "climate_class"}
BIOPHYS_FEATURES = set(CONFIG_B) - PHASE_FEATURES - STATIC_FEATURES
AID_FLAG = "fews_ha"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(df, idx, ycol, predictors):
    sub = df.loc[idx]; sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].astype(int).to_numpy(), sub.index


def _train(df, split, ycol, predictors, which="xgb"):
    Xtr, ytr, tr_idx = _prep(df, split["train_idx"], ycol, predictors)
    if len(Xtr) == 0:
        return None, None, None
    pos = ytr.sum(); spw = ((len(ytr)-pos)/pos) if pos > 0 else 1.0
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
    m.fit(Xtr, ytr)
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol, predictors)
    return m, Xte, te_idx


def _biophys_share(mean_abs, names):
    total = mean_abs.sum()
    if total == 0:
        return np.nan
    bio = sum(v for n, v in zip(names, mean_abs) if n in BIOPHYS_FEATURES)
    return bio / total


def stratified_shares(model, X, aid_mask):
    import shap
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    absv = np.abs(sv); names = list(X.columns)
    out = {}
    for label, mask in [("aid_heavy", aid_mask), ("aid_light", ~aid_mask)]:
        if mask.sum() == 0:
            out[label] = {"biophys_share": np.nan, "n": 0}; continue
        out[label] = {"biophys_share": _biophys_share(absv[mask].mean(axis=0), names),
                      "n": int(mask.sum())}
    return out


def run():
    section("C2 confound (D3) — SHAP stratified by aid presence (fews_ha)")
    print("SUGGESTIVE evidence about the aid confound, NOT a clean separation.")
    print("fews_ha correlates with SEVERITY -> aid-heavy vs aid-light is PARTLY")
    print("crisis vs non-crisis. Divergence could be aid confound OR severity behaviour.\n")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    if AID_FLAG not in df.columns:
        raise RuntimeError(f"{AID_FLAG} not in panels; cannot stratify.")
    splits = all_splits(df, verbose=False)
    print(f"{'win':<9}{'model':>6}{'bio%_aidH':>11}{'bio%_aidL':>11}"
          f"{'d(H-L)':>9}{'n_H':>7}{'n_L':>7}")
    print("-" * 60)
    rows = []
    for s in [s for s in splits if s["horizon"] == 3]:
        yr = s["test_year"]; ycol = "y_binary_h3"
        for which in ("xgb", "lgb"):
            m, Xte, te_idx = _train(df, s, ycol, CONFIG_B, which)
            if m is None or len(Xte) == 0:
                continue
            aid = df.loc[te_idx, AID_FLAG].fillna(0).astype(int).to_numpy() == 1
            sh = stratified_shares(m, Xte, aid)
            bh = sh["aid_heavy"]["biophys_share"]; bl = sh["aid_light"]["biophys_share"]
            d = (bh - bl) if (not np.isnan(bh) and not np.isnan(bl)) else np.nan
            print(f"{yr}/h3{'':>2}{which:>6}{bh:>11.3f}{bl:>11.3f}{d:>+9.3f}"
                  f"{sh['aid_heavy']['n']:>7}{sh['aid_light']['n']:>7}")
            rows.append({"year": yr, "model": which,
                         "biophys_share_aid_heavy": bh, "biophys_share_aid_light": bl,
                         "delta_H_minus_L": d,
                         "n_aid_heavy": sh["aid_heavy"]["n"],
                         "n_aid_light": sh["aid_light"]["n"]})
    section("READING — D3 confound discussion")
    print("- SIMILAR shares (d~0) -> aid-invariant; model keys on deterioration (reassuring).")
    print("- DIVERGENT shares -> consistent with aid confound OR severity behaviour")
    print("  (aid~severity). SUGGESTIVE about D3, NOT a clean causal separation.")
    print("- Strengthens the confound DISCUSSION / Art.13 transparency, NOT a resolution.")
    pd.DataFrame(rows).to_csv("outputs/diagnostics/c2_shap_aid_stratified.csv", index=False)
    print("\nsaved: outputs/diagnostics/c2_shap_aid_stratified.csv")


if __name__ == "__main__":
    run()
