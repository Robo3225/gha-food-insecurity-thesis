"""
======================================================================

harness_fews.py  (M10.1 — FEWS NET projection benchmark, D24)

Harness-side join of the FEWS NET projection benchmark from the raw
World Bank source. Frozen panels untouched (D24: projection is
evaluation metadata, neither feature nor label).

D24 (option B) — exact same-target-period comparison:
  A FEWS projection ISSUED at assessment period t_a is a forecast FOR a
  future assessment period. It is scored against the observed fews_ipc
  for that SAME future period — the identical quantity the model
  predicts at the corresponding horizon. Alignment is on ACTUAL
  per-country assessment periods (not a fixed 4-month offset), so
  Somalia's two missing crisis-period rounds (2022-06, 2023-02) are
  handled by construction.

Unadjusted only: fews_proj_near / fews_proj_med. The _ha/_adjusted
variants encode humanitarian assistance (the D3 confounder) and are
excluded — the benchmark must be like-for-like with the unadjusted
target.

Horizon mapping (D24, tri-annual cadence verified):
  - assessments are ~4 months apart (Feb/Jun/Oct).
  - fews_proj_near  -> the NEXT assessment (~4 months ahead) -> the
    model's 3-MONTH horizon (closest month-horizon to the assessment
    interval; the exact target period is matched, see below).
  - fews_proj_med   -> the assessment after next (~8 months ahead) ->
    reported only if/where a model is run at that horizon (not in the
    current 1/3-month set).
  - the model's 1-MONTH horizon has NO FEWS analogue (FEWS does not
    forecast at 1-month granularity) -> persistence only (D24).

NO MODELLING.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from harness_data import ROOT, UNIT_COL

RAW = ROOT / "data" / "raw" / "world_bank" / \
    "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"

COUNTRY_NAME_TO_SLUG = {
    "Kenya": "kenya", "Ethiopia": "ethiopia", "Somalia": "somalia",
}
PHASE3_THRESHOLD = 3


def load_fews_projections() -> pd.DataFrame:
    """
    Load the raw FEWS projections for the three study countries. Returns
    long frame: [slug, admin_code, year_month, ym_int, fews_proj_near,
    fews_proj_med] — UNADJUSTED only. year_month restricted to rows where
    a projection was actually issued (an assessment period).
    """
    raw = pd.read_csv(RAW)
    raw = raw[raw["country"].isin(COUNTRY_NAME_TO_SLUG)].copy()
    raw["slug"] = raw["country"].map(COUNTRY_NAME_TO_SLUG)
    raw["ym_int"] = raw["year"].astype(int) * 12 + (raw["month"].astype(int) - 1)

    keep = ["slug", "admin_code", "year_month", "ym_int",
            "fews_proj_near", "fews_proj_med"]
    proj = raw[keep].copy()
    # a projection row exists only where FEWS issued one (assessment period);
    # rows with both projections null are not assessment-issue periods.
    proj = proj[proj[["fews_proj_near", "fews_proj_med"]].notna().any(axis=1)]
    return proj


def build_fews_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ASSESSMENT-ISSUE period t_a in the raw projections, produce a
    FEWS baseline record aligned to the target period it forecasts:

      fews_proj_near issued at t_a  -> targets the NEXT assessment period
                                        t_next (the actual next period in
                                        THIS country's assessment sequence,
                                        so Somalia's 8-month gaps map to the
                                        real next period, not t_a+4).
      the prediction is (fews_proj_near >= 3); the truth is the observed
      phase3plus at t_next (from the modelling frame df).

    Returns rows: [slug, admin_code, issue_ym_int, target_ym_int,
    fews_pred_binary] — the model is then compared against these at the
    matching (unit, target period).
    """
    proj = load_fews_projections()

    rows = []
    # per country, per unit: order assessment periods, map each near-proj to
    # the actual next assessment period in that unit's sequence.
    for (slug, ac), g in proj.groupby(["slug", "admin_code"]):
        g = g.sort_values("ym_int")
        periods = g["ym_int"].tolist()
        near = g["fews_proj_near"].tolist()
        med = g["fews_proj_med"].tolist()
        for i, t_a in enumerate(periods):
            # NEXT actual assessment period (handles gaps: real next, not +4)
            if i + 1 < len(periods):
                t_next = periods[i + 1]
                pn = near[i]
                if pd.notna(pn):
                    rows.append({
                        "slug": slug, "admin_code": ac,
                        "issue_ym_int": t_a, "target_ym_int": t_next,
                        "fews_pred_binary": float(pn >= PHASE3_THRESHOLD),
                        "proj_type": "near",
                    })
            # assessment AFTER next for med-term (t+2 in the sequence)
            if i + 2 < len(periods):
                t_med = periods[i + 2]
                pm = med[i]
                if pd.notna(pm):
                    rows.append({
                        "slug": slug, "admin_code": ac,
                        "issue_ym_int": t_a, "target_ym_int": t_med,
                        "fews_pred_binary": float(pm >= PHASE3_THRESHOLD),
                        "proj_type": "med",
                    })
    return pd.DataFrame(rows)


def fews_baseline_for_split(df: pd.DataFrame, fews_rows: pd.DataFrame,
                            split: dict, proj_type: str = "near") -> dict | None:
    """
    Assemble the FEWS baseline for one split's test rows, at exactly the
    target periods FEWS forecasts for. Only the model rows whose
    (unit, predicted-target-period) match a FEWS projection of the given
    type are compared — exact same-target-period alignment (D24 B).

    Returns None if no matching FEWS rows exist for this split (e.g. the
    1-month horizon, where near-term FEWS does not align — persistence
    only there, D24).
    """
    from harness_eval import evaluate_binary
    from harness_data import TARGET_BINARY, IS_OBSERVED

    h = split["horizon"]

    # D24: FEWS does not forecast at 1-month granularity, so the 1-month
    # horizon has NO fair FEWS comparator (a target-period match there would
    # compare FEWS's ~4-month-lead projection against a 1-month-lead model
    # prediction — different lead times). The 1-month horizon is evaluated
    # against persistence only. Suppress FEWS at h=1 BY DESIGN, not by
    # incidental row counts.
    if h == 1:
        return None
    if proj_type == "near" and h != 3:
        return None   # near-projection aligns to the 3-month horizon only

    te = df.loc[split["test_idx"]].copy()
    # Model prediction targets the ABSOLUTE month t+h. FEWS rows carry
    # target_ym_int as absolute (year*12 + month-1) of the period they
    # forecast. One consistent definition, so the merge aligns exact periods.
    te["target_ym_int"] = (
        te["year"].astype(int) * 12 + (te["month"].astype(int) - 1) + h
    )

    fr = fews_rows[fews_rows["proj_type"] == proj_type][
        ["slug", "admin_code", "target_ym_int", "fews_pred_binary"]]

    # match model test rows to FEWS predictions on (country, unit, target period)
    merged = te.merge(
        fr,
        left_on=["country", UNIT_COL, "target_ym_int"],
        right_on=["slug", "admin_code", "target_ym_int"],
        how="inner",
    )
    if len(merged) == 0:
        return None  # no aligned FEWS comparison at this horizon (e.g. h=1)

    ycol = f"y_binary_h{h}"
    merged = merged[merged[ycol].notna()]
    if len(merged) == 0:
        return None

    # D25: persistence input (phase at feature-time t) must be non-NaN too,
    # so persistence and FEWS score the IDENTICAL rows. Drop NaN-persistence
    # rows from the aligned set before scoring either.
    merged = merged[merged[TARGET_BINARY].notna()]
    if len(merged) == 0:
        return None

    y_true = merged[ycol].to_numpy()
    fews_pred = merged["fews_pred_binary"].to_numpy()
    persist_pred = merged[TARGET_BINARY].to_numpy().astype(float)  # phase at t
    obs = merged[IS_OBSERVED].to_numpy().astype(bool) if IS_OBSERVED in merged else None

    # BOTH scored on the identical aligned row-set (D25) -> exact comparison
    res = {
        "fews": evaluate_binary(y_true, fews_pred, observed_mask=obs),
        "persistence_on_fews_rows": evaluate_binary(y_true, persist_pred, observed_mask=obs),
        "n_aligned": len(merged),
        "proj_type": proj_type,
    }
    return res


if __name__ == "__main__":
    # build_fews_baseline_rows reads the raw source directly; df arg unused.
    fr = build_fews_baseline_rows(None)
    print(f"FEWS baseline rows (period-matched): {len(fr):,}")
    print(fr.groupby(["slug", "proj_type"]).size().to_string())
