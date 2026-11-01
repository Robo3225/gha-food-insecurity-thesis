"""
======================================================================

c2_uncertainty_flag.py  (M10.4 / C2 — three-tier operational uncertainty flag)

Translates the calibrated (conformal) prediction into an actionable
operational output: a HIGH / MEDIUM / LOW confidence tier per prediction,
for a program officer deciding whether to act on the forecast.

Two inputs to the tier (both honest, both grounded in C2's findings):
  1. INTERVAL WIDTH — the conformalized P10-P90 width. Narrow interval =
     confident point forecast; wide = diffuse. Width thresholds are set
     on TRAIN-only interval widths (percentiles), never test (leakage
     discipline).
  2. SHIFT-AWARE DOWNGRADE — C2's uncertainty finding is that calibration
     DEGRADES under distribution shift (conformal under-covers in crisis
     windows). So a prediction made under detected distribution shift has
     its tier DOWNGRADED: the model's uncertainty estimate is itself less
     trustworthy then, and the flag must say so rather than assert false
     confidence. Shift is detected TRAIN-only (a unit-month whose recent
     feature distribution departs from the training distribution).

This is the EU-AI-Act-style transparency piece: the operational output
signals not just its prediction and interval, but WHEN its own
uncertainty estimate is less reliable — it knows the limits of what it knows.

Tiers (per prediction):
  HIGH   : narrow interval AND no detected shift
  MEDIUM : (moderate interval) OR (narrow interval under mild shift)
  LOW    : wide interval OR prediction under strong shift (calibration
           known-unreliable regime)

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_tiers(interval_width, shift_score, *,
                 width_lo, width_hi, shift_mild, shift_strong):
    """
    Vectorized tier assignment. Thresholds (width_lo/hi, shift_*) are all
    computed on TRAIN data upstream and passed in — this function applies
    them, it does not learn them from test.

    interval_width : conformalized P90-P10 width per prediction (array)
    shift_score    : distribution-shift score per prediction (array); higher
                     = more departure from training distribution
    """
    w = np.asarray(interval_width, float)
    s = np.asarray(shift_score, float)
    n = len(w)
    tier = np.array(["MEDIUM"] * n, dtype=object)

    narrow = w <= width_lo
    wide = w >= width_hi
    mild_shift = (s > shift_mild) & (s <= shift_strong)
    strong_shift = s > shift_strong

    # HIGH: narrow AND no meaningful shift
    tier[narrow & (s <= shift_mild)] = "HIGH"
    # LOW: wide interval, OR strong shift (calibration known-unreliable)
    tier[wide] = "LOW"
    tier[strong_shift] = "LOW"
    # narrow interval but mild shift -> MEDIUM (downgrade from HIGH)
    tier[narrow & mild_shift] = "MEDIUM"
    return tier


def compute_shift_score_train_only(train_feat, test_feat):
    """
    A simple, train-only distribution-shift score per test row: the
    Mahalanobis-like distance of each test row's features from the TRAINING
    feature distribution (mean/cov estimated on TRAIN only). Higher = more
    shifted. Test rows never inform the reference distribution.

    (Simplified to standardized Euclidean here for robustness; the training
    mean/std are the reference, computed train-only.)
    """
    mu = train_feat.mean(axis=0)
    sd = train_feat.std(axis=0).replace(0, 1.0)
    z = (test_feat - mu) / sd
    # per-row shift = RMS standardized deviation across features
    return np.sqrt((z ** 2).mean(axis=1)).to_numpy()


def calibrate_thresholds_train_only(train_widths, train_shift):
    """
    Set tier thresholds from TRAIN distributions (percentiles), never test:
      width_lo = 33rd pct of train interval widths (narrow)
      width_hi = 67th pct (wide)
      shift_mild = 90th pct of train shift scores
      shift_strong = 99th pct
    """
    return dict(
        width_lo=float(np.percentile(train_widths, 33)),
        width_hi=float(np.percentile(train_widths, 67)),
        shift_mild=float(np.percentile(train_shift, 90)),
        shift_strong=float(np.percentile(train_shift, 99)),
    )


def tier_report(tiers, y_true_binary, correct_mask=None):
    """
    Summarize tiers and their reliability: what fraction of each tier's
    predictions were correct / what the positive rate is. A well-designed
    flag has HIGH tier = most reliable, LOW = least — validating that the
    tiers carry real information about prediction trustworthiness.
    """
    df = pd.DataFrame({"tier": tiers, "y": np.asarray(y_true_binary)})
    if correct_mask is not None:
        df["correct"] = np.asarray(correct_mask, float)
    out = {}
    for t in ["HIGH", "MEDIUM", "LOW"]:
        sub = df[df.tier == t]
        rec = {"n": len(sub), "share": len(sub) / len(df) if len(df) else np.nan}
        if correct_mask is not None and len(sub):
            rec["accuracy"] = float(sub["correct"].mean())
        out[t] = rec
    return out


if __name__ == "__main__":
    # illustrative self-test on synthetic inputs
    rng = np.random.default_rng(0)
    n = 1000
    widths = rng.gamma(2, 0.5, n)
    shift = rng.gamma(2, 0.3, n)
    thr = calibrate_thresholds_train_only(widths[:700], shift[:700])
    tiers = assign_tiers(widths[700:], shift[700:], **thr)
    from collections import Counter
    print("tier distribution (test):", dict(Counter(tiers)))
    print("thresholds (train-only):", {k: round(v,3) for k,v in thr.items()})
