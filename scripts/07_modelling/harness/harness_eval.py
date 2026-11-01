"""
======================================================================

harness_eval.py  (M10.1 — baselines + shared binary evaluation)

Two baselines, auto-reported on every model run:
  - PERSISTENCE  (predict last observed phase >=3): the validity FLOOR.
    A model that cannot beat this is learning the forward-fill
    autocorrelation, not the phenomenon.
  - FEWS NET projection (fews_proj_near >=3): the significance CEILING —
    computed by harness_fews as a period-matched join (D24), NOT here,
    since the projection is not a panel column. This module owns the
    persistence floor and the shared evaluation function.

ONE shared binary evaluation function: XGBoost (binary) and
TFT (ordinal thresholded >=3) are both scored through this identical
function, on the identical test rows, with the identical metrics. This
guarantees C1 compares architectures, not evaluation pipelines.

NO MODELLING.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import (
    TARGET_BINARY, TARGET_ORDINAL, IS_OBSERVED, UNIT_COL,
)

PHASE3_THRESHOLD = 3


# =====================================================================
# Shared binary evaluation — the single scoring path for ALL predictors
# =====================================================================

def evaluate_binary(y_true, y_prob, *, observed_mask=None) -> dict:
    """
    The one evaluation function. y_true is binary phase3plus; y_prob is a
    probability (XGBoost) or a thresholded/derived score (TFT median ->
    binary or its P(phase>=3)). Metrics: PR-AUC (primary, class-imbalanced),
    ROC-AUC, Brier (calibration-sensitive), plus positive count/rate.

    observed_mask (optional): if given, metrics are computed ONLY on
    genuinely-observed rows (is_observed True), so forward-filled labels
    do not inflate the score. Reported both ways when provided.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    def _metrics(yt, yp):
        yt = np.asarray(yt, dtype=float); yp = np.asarray(yp, dtype=float)
        # drop any residual non-finite pairs (defensive; upstream should have
        # removed NaN target/prediction rows, but never pass NaN to sklearn)
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]; yp = yp[finite]
        n = len(yt); pos = int(yt.sum())
        out = {"n": n, "positives": pos, "prevalence": (pos / n) if n else np.nan}
        # PR-AUC / ROC-AUC undefined if only one class present
        if pos == 0 or pos == n:
            out.update(pr_auc=np.nan, roc_auc=np.nan, brier=np.nan,
                       note="single-class window")
            return out
        out["pr_auc"] = average_precision_score(yt, yp)
        out["roc_auc"] = roc_auc_score(yt, yp)
        out["brier"] = brier_score_loss(yt, np.clip(yp, 0, 1))
        return out

    result = {"all": _metrics(y_true, y_prob)}
    if observed_mask is not None:
        m = np.asarray(observed_mask, dtype=bool)
        result["observed_only"] = _metrics(y_true[m], y_prob[m])
    return result


# =====================================================================
# Baselines
# =====================================================================

def persistence_prediction(df: pd.DataFrame, test_idx, horizon: int) -> np.ndarray:
    """
    Persistence: predict that unit's LAST OBSERVED phase (>=3 -> 1) as of
    the feature time (t), for the target at t+horizon. Uses only past
    information (the observed phase at or before t), so it is leakage-safe.

    Implementation: for each test row (unit, t), the persistence
    prediction is the binary phase3plus at feature-time t itself (the most
    recent value the forecaster would have in hand). This is the
    "predict same as last observed" floor.
    """
    te = df.loc[test_idx]
    # phase3plus at feature-time t is the last-in-hand observation
    return te[TARGET_BINARY].to_numpy().astype(float)


def evaluate_baselines(df: pd.DataFrame, split: dict) -> dict:
    """
    Run BOTH baselines through the shared evaluation on a split's test rows.
    y_true is the HORIZON-SHIFTED target y_binary_h{h} (phase at t+h), NOT
    the feature-time phase — scoring against the feature-time phase would be
    target-leakage (D23). Rows with NaN horizon target (series edge) are
    dropped from scoring.
    """
    h = split["horizon"]
    ycol = f"y_binary_h{h}"
    te = df.loc[split["test_idx"]].copy()

    n_before = len(te)
    # 1. drop rows with no horizon target (t+h ran off the series end)
    te = te[te[ycol].notna()]
    n_no_target = n_before - len(te)

    # 2. drop rows where the PERSISTENCE INPUT (phase at feature-time t) is
    #    NaN — a D6 label-coverage gap. You cannot "assume no change" without
    #    a current observation, so persistence is undefined there. For a fair
    #    like-for-like comparison the row-set must be identical across methods
    #    (one shared evaluation, same rows), so these rows are dropped
    #    from scoring entirely. [Canonical-row-set definition = D25, pending.]
    n_pre_persist = len(te)
    te = te[te[TARGET_BINARY].notna()]
    n_no_persist = n_pre_persist - len(te)

    if len(te) == 0:
        raise RuntimeError(f"[{split['test_year']}/h{h}] no scorable rows after "
                           f"dropping NaN target ({n_no_target}) and NaN "
                           f"persistence-input ({n_no_persist}).")

    y_true = te[ycol].to_numpy()
    obs = te[IS_OBSERVED].to_numpy().astype(bool) if IS_OBSERVED in te else None
    persist = te[TARGET_BINARY].to_numpy().astype(float)

    return {
        "persistence": evaluate_binary(y_true, persist, observed_mask=obs),
        "n_scorable": len(te),
        "n_dropped_no_target": n_no_target,
        "n_dropped_no_persist": n_no_persist,
        "scored_idx": te.index.to_numpy(),   # the canonical scored rows for this split
    }


def assert_beats_persistence(model_pr_auc: float, persistence_pr_auc: float,
                             label: str = "model") -> None:
    """
    Autocorrelation guard: a model that does not beat
    persistence is learning the forward-fill structure, not the
    phenomenon. This is a WARNING-level check the harness surfaces loudly
    (not a hard raise — a model may legitimately tie persistence in a
    trivial window — but it must never pass unnoticed).
    """
    if np.isnan(model_pr_auc) or np.isnan(persistence_pr_auc):
        return
    if model_pr_auc <= persistence_pr_auc:
        print(f"  \u26a0 AUTOCORRELATION WARNING: {label} PR-AUC "
              f"{model_pr_auc:.3f} <= persistence {persistence_pr_auc:.3f}. "
              f"Model may be learning forward-fill structure, not signal.")


if __name__ == "__main__":
    print("harness_eval: shared binary evaluation + persistence & FEWS baselines")
    print(f"phase3 threshold: {PHASE3_THRESHOLD}")
