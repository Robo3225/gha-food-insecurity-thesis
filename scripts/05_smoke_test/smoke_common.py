"""
======================================================================

smoke_common.py  (scripts/05_smoke_test/ — DISPOSABLE)

Shared pieces for the smoke test: data prep, stratified subset,
walk-forward split, persistence baseline, PR-AUC metrics.

This is smoke-test scaffolding, not the production harness. It is
allowed to be blunt. Its only job is to feed the five diagnostics.

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, accuracy_score


TARGET = "phase3plus"
GROUP = "canonical_admin_code"
TIME = "time_idx"


# ---------------------------------------------------------------------
# Load + prepare
# ---------------------------------------------------------------------

def load_panel(path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # integer month index per unit, contiguous, for TimeSeriesDataSet
    df["ym"] = pd.to_datetime(df["year_month"], format="%Y_%m")
    df = df.sort_values([GROUP, "ym"]).reset_index(drop=True)
    first = df["ym"].min()
    df[TIME] = (
        (df["ym"].dt.year - first.year) * 12
        + (df["ym"].dt.month - first.month)
    ).astype(int)

    # calendar features — TIME-VARYING KNOWN (always known in advance)
    df["month_num"] = df["ym"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    df["year"] = df["ym"].dt.year
    df[GROUP] = df[GROUP].astype(str)   # group id must be categorical-safe
    return df


def stratified_subset(df: pd.DataFrame, n_units: int, seed: int = 0) -> pd.DataFrame:
    """
    Smoke-test-only subset. The target is spatially concentrated, so a
    random subset risks having no positives. Sample preferentially from
    crisis-active units so the smoke test has a signal to learn.

    Documented as a smoke-test shortcut — NOT a modelling choice.
    """
    pos_by_unit = df.groupby(GROUP)[TARGET].apply(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()
    )
    active = pos_by_unit[pos_by_unit > 0].index.tolist()
    stable = pos_by_unit[pos_by_unit == 0].index.tolist()

    rng = np.random.default_rng(seed)
    # 70% active, 30% stable, so the model sees both regimes
    n_active = min(len(active), int(n_units * 0.7))
    n_stable = min(len(stable), n_units - n_active)
    chosen = (
        list(rng.choice(active, n_active, replace=False))
        + list(rng.choice(stable, n_stable, replace=False))
    )
    return df[df[GROUP].isin(chosen)].copy()


# ---------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------

def walk_forward_split(df: pd.DataFrame, test_year: int):
    """
    Expanding-window split: train on everything strictly BEFORE the
    test year, test on the test year. No future information leaks into
    training because the split is purely temporal.
    """
    train = df[df["year"] < test_year].copy()
    test = df[df["year"] == test_year].copy()
    return train, test


# ---------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------

def persistence_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict each unit-month's Phase 3+ as equal to that unit's PREVIOUS
    month. This is the bar the model must clear: forward-filled labels
    are heavily autocorrelated, so 'same as last month' is strong.
    """
    df = df.sort_values([GROUP, TIME]).copy()
    df["persistence"] = df.groupby(GROUP)[TARGET].shift(1)
    return df


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate(y_true, y_score, y_pred_binary=None, label=""):
    """
    PR-AUC is the headline (class-imbalanced). Accuracy reported too,
    but only for comparison against the persistence accuracy figure.
    """
    y_true = np.asarray(y_true, dtype=float)
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_score = np.asarray(y_score, dtype=float)[mask]

    out = {"label": label, "n": int(mask.sum()),
           "positives": int(y_true.sum()) if len(y_true) else 0}

    # Guard: nothing to score (e.g. prediction/truth join produced no rows)
    if len(y_true) == 0:
        out["pr_auc"] = float("nan")
        out["accuracy"] = float("nan")
        out["empty"] = True
        return out

    # PR-AUC needs both classes present
    if 0 < y_true.sum() < len(y_true):
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["pr_auc"] = float("nan")

    if y_pred_binary is not None:
        yb = np.asarray(y_pred_binary, dtype=float)[mask]
        out["accuracy"] = accuracy_score(y_true, yb)
    return out
