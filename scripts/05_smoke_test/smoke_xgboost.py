"""
======================================================================

smoke_xgboost.py  (scripts/05_smoke_test/ — DISPOSABLE)

XGBoost control for the smoke test.

Purpose: XGBoost trains in seconds and has no sequence machinery. If it
CAN beat persistence but the TFT cannot, the problem is TFT-specific
(config, tensor roles, convergence). If XGBoost ALSO cannot beat
persistence, the problem is the data/target itself - a far more
important finding. This control is what makes that distinction.

XGBoost sees the panel FLATTENED: no sequences, but lagged NDVI and
calendar features engineered so it has temporal context comparable to
what the TFT gets from its encoder.

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import smoke_common as SC


def make_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    """Engineer lagged NDVI + calendar so XGBoost has temporal context."""
    df = df.sort_values([SC.GROUP, SC.TIME]).copy()
    for L in lags:
        df[f"ndvi_lag{L}"] = df.groupby(SC.GROUP)["ndvi"].shift(L)
        df[f"phase_lag{L}"] = df.groupby(SC.GROUP)[SC.TARGET].shift(L)
    return df


FEATURES_BASE = ["ndvi", "month_sin", "month_cos"]


def run_xgboost_control(df: pd.DataFrame, test_year: int) -> dict:
    """
    Train XGBoost on < test_year, evaluate on test_year. Returns metrics
    alongside the persistence baseline on the same window.
    """
    df = make_lag_features(df)
    lag_cols = [c for c in df.columns if c.startswith(("ndvi_lag", "phase_lag"))]
    features = FEATURES_BASE + lag_cols

    train, test = SC.walk_forward_split(df, test_year)

    # drop rows with NaN target or NaN features (early lags)
    train = train.dropna(subset=[SC.TARGET] + features)
    test_eval = test.dropna(subset=[SC.TARGET] + features)

    Xtr, ytr = train[features], train[SC.TARGET].astype(int)
    Xte, yte = test_eval[features], test_eval[SC.TARGET].astype(int)

    # class imbalance -> scale_pos_weight (NOT SMOTE: temporal integrity)
    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    spw = neg / max(pos, 1)

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1,
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    xgb_res = SC.evaluate(yte, proba, (proba >= 0.5).astype(int),
                          label=f"xgboost_{test_year}")

    # persistence on the SAME evaluation rows
    test_p = SC.persistence_prediction(test_eval)
    pers = test_p["persistence"].fillna(0)
    pers_res = SC.evaluate(test_p[SC.TARGET], pers, pers,
                           label=f"persistence_{test_year}")

    return {"xgboost": xgb_res, "persistence": pers_res,
            "beats_persistence": xgb_res["pr_auc"] > pers_res["pr_auc"]}
