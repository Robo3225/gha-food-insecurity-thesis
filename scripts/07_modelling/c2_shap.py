"""
======================================================================

c2_shap.py  (M10.4 / C2 — SHAP cross-validation of the biophysical finding)

The positive C2 explainability result. The C1 ablation measured the
biophysical value-added (B-C) by RETRAINING with/without biophysical
features. SHAP measures it independently by FEATURE ATTRIBUTION on the
full Config-B model. If the two INDEPENDENT methods agree — biophysical
SHAP weight largest exactly where the B-C ablation gap is largest (2021
drought-onset) — C2's explainability claim is solid, not decorative.

Design:
  - Tree-SHAP (exact, fast) on the Config-B models (XGBoost AND LightGBM,
    both cross-validated the ablation; SHAP on both = strongest agreement).
  - Per window, per horizon; SHAP computed on the D25-canonical TEST rows
    (attribution of the predictions actually scored).
  - Aggregate |SHAP| into groups: BIOPHYSICAL (ndvi, spi3/lst/vci + lags),
    PHASE (phase_hist_lag0/4/8), STATIC (pop_density, climate_class).
  - Report biophysical SHAP SHARE per window, and correlate its across-
    window pattern with the B-C ablation gap. Agreement = convergent
    validation of the central finding.

Leakage note: SHAP is post-hoc attribution of an already-trained,
leakage-safe model on already-scored rows — it introduces no new leakage
surface (no refitting, no test information entering training).

Runs in gee_env (shap installed, or use model built-in importances as
fallback). Structure verified offline.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    load_all, CANONICAL_PREDICTORS, UNIT_COL,
)
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_phase_history import (
    add_phase_history, assert_phase_history_safe, PHASE_HISTORY_COLS,
)

RANDOM_SEED = 42
CAL_FRACTION = 0.2
CONFIG_B = CANONICAL_PREDICTORS + PHASE_HISTORY_COLS

# feature groups for aggregation
PHASE_FEATURES = set(PHASE_HISTORY_COLS)
STATIC_FEATURES = {"pop_density", "climate_class"}
BIOPHYS_FEATURES = set(CONFIG_B) - PHASE_FEATURES - STATIC_FEATURES

# the B-C ablation gap from C1 (per window, h=3) — the number SHAP cross-checks
BC_GAP_H3 = {2021: 0.117, 2022: 0.036, 2023: 0.021}   # XGBoost (LGB near-identical)


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _prep(df, idx, ycol, predictors):
    sub = df.loc[idx]; sub = sub[sub[ycol].notna()]
    return sub[predictors].copy(), sub[ycol].astype(int).to_numpy(), sub.index


def _train_classifier(df, split, ycol, predictors, which="xgb"):
    Xtr, ytr, tr_idx = _prep(df, split["train_idx"], ycol, predictors)
    if len(Xtr) == 0:
        return None, None, None
    tr_rows = df.loc[tr_idx].sort_values("t_index")
    n_cal = int(len(tr_rows) * CAL_FRACTION)
    fit_idx = tr_rows.index[:-n_cal] if n_cal else tr_rows.index
    Xfit, yfit, _ = _prep(df, fit_idx, ycol, predictors)
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
    Xte, yte, te_idx = _prep(df, split["test_idx"], ycol, predictors)
    return m, Xte, te_idx


def _group_shares(mean_abs_shap, feature_names):
    """Aggregate mean|SHAP| into biophysical / phase / static shares."""
    total = mean_abs_shap.sum()
    if total == 0:
        return dict(biophys=np.nan, phase=np.nan, static=np.nan)
    g = {"biophys": 0.0, "phase": 0.0, "static": 0.0}
    for name, val in zip(feature_names, mean_abs_shap):
        if name in PHASE_FEATURES: g["phase"] += val
        elif name in STATIC_FEATURES: g["static"] += val
        else: g["biophys"] += val
    return {k: v/total for k, v in g.items()}


def shap_shares(model, X, which="xgb"):
    """
    Mean |SHAP| per feature -> group shares via exact Tree-SHAP.

    REQUIRES the shap library. Fails LOUD if unavailable — the previous
    silent fallback to built-in importances used DIFFERENT metrics per model
    (XGBoost gain vs LightGBM split-count), which are not comparable and
    produced an artifactual 40%-vs-82% discrepancy. Both models must use the
    same principled attribution (real SHAP) or the comparison is meaningless.
    """
    import shap  # hard requirement; install in the env (pip install shap)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):            # some versions return [neg, pos]
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    return _group_shares(mean_abs, list(X.columns))


def run():
    section("M10.4 / C2 — SHAP cross-validation of biophysical value-added")
    df = load_all(); df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df); assert_phase_history_safe(df)
    splits = all_splits(df, verbose=False)

    print(f"\n{'split':<9}{'model':>7}{'biophys%':>10}{'phase%':>9}{'static%':>9}"
          f"{'B-C(abl)':>10}")
    print("-" * 56)

    rows = []
    for s in splits:
        h, yr = s["horizon"], s["test_year"]
        ycol = f"y_binary_h{h}"
        for which in ("xgb", "lgb"):
            m, Xte, _ = _train_classifier(df, s, ycol, CONFIG_B, which)
            if m is None or len(Xte) == 0:
                continue
            shares = shap_shares(m, Xte, which)
            bc = BC_GAP_H3.get(yr, np.nan) if h == 3 else np.nan
            print(f"{yr}/h{h:<5}{which:>7}{shares['biophys']:>10.3f}"
                  f"{shares['phase']:>9.3f}{shares['static']:>9.3f}"
                  f"{(bc if not np.isnan(bc) else float('nan')):>10.3f}")
            rows.append({"year": yr, "horizon": h, "model": which,
                         "biophys_share": shares["biophys"],
                         "phase_share": shares["phase"],
                         "static_share": shares["static"],
                         "bc_gap_h3": bc})

    section("CROSS-VALIDATION: does biophysical SHAP share track the B-C gap?")
    res = pd.DataFrame(rows)
    # at h=3, correlate biophysical SHAP share with the ablation B-C gap across windows
    h3 = res[(res.horizon == 3) & res.bc_gap_h3.notna()]
    for which in ("xgb", "lgb"):
        sub = h3[h3.model == which].sort_values("year")
        if len(sub) >= 2:
            rho = sub["biophys_share"].corr(sub["bc_gap_h3"], method="spearman")
            print(f"  {which}: biophys SHAP share by year "
                  f"{dict(zip(sub.year, sub.biophys_share.round(3)))} vs "
                  f"B-C {dict(zip(sub.year, sub.bc_gap_h3))} -> Spearman {rho:+.2f}")
    print("\n- If biophys SHAP share is LARGEST in 2021 (where B-C gap is largest),")
    print("  two INDEPENDENT methods (ablation retraining + feature attribution)")
    print("  converge on the same finding -> C2 explainability is SOLID.")
    print("- Cross-model agreement (xgb & lgb SHAP) adds a further independent axis.")
    res.to_csv("outputs/diagnostics/c2_shap.csv", index=False)
    print("\nsaved: outputs/diagnostics/c2_shap.csv")
    return res


if __name__ == "__main__":
    run()
