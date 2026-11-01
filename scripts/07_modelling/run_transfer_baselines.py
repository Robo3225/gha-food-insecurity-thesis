"""
======================================================================

run_transfer_baselines.py  (P1 Somalia-transfer D25 verification)

Diagnostic re-score. Emits, for the C3
train-KE+ET / test-Somalia transfer, the SAME four D25 columns the
in-pool run_baselines.py reports, so the transfer table can be verified
identically to C1 Table 1:

    n_test, fews_n, n_dropped, persist_on_fews

WHY THIS IS NEEDED
------------------
c3_transfer.py computes so_fews via harness_fews.fews_baseline_for_split,
which ALREADY scores FEWS and persistence on the identical aligned rows
(the returned "persistence_on_fews_rows" and "n_aligned"). But
c3_transfer.py DISCARDS those two — it keeps only fb["fews"] and scores
so_persistence separately on the FULL Somalia test set. Result: in
c3_transfer.csv, so_persistence is on the full set and so_fews is on the
aligned subset. Not a like-for-like FEWS-vs-persistence comparison.

WHAT THIS DOES (and does NOT do)
--------------------------------
- Rebuilds the transfer test rows EXACTLY as c3_transfer.py does:
  same TEST_WINDOWS, same predicted-time (t+h) windowing, same
  add_horizon_targets / add_phase_history, same load_all frozen panels.
- Reports, per window (h=3, the only horizon with a FEWS comparator):
    * n_test        : the full Somalia persistence-set size (D25 set 1:
                      non-NaN target AND non-NaN persistence input)
    * fews_n        : the FEWS-aligned subset size (D25 set 2)
    * n_dropped     : full test rows dropped for NaN target / NaN persist
    * persist_on_fews : Somalia persistence RE-SCORED on the fews-aligned
                        subset (the missing like-for-like floor)
  and, for context, the aligned FEWS PR-AUC and full-set persistence.
- NO MODELLING. Baselines only (persistence + FEWS period-matched join).
  It does not touch any panel, any model, or any settled result. It reads
  the frozen v2 panels read-only through the harness, exactly as the
  banked baseline run did.

This is the transfer analogue of run_baselines.py, restricted to the
Somalia test country, using the harness's own aligned-row machinery so
the row-set is defined identically to the in-pool table.

Author
------
Robert Odingo MSc Thesis
======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import load_all, TARGET_BINARY, IS_OBSERVED
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_phase_history import add_phase_history, assert_phase_history_safe
from harness_eval import evaluate_binary
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split

TEST_COUNTRY = "somalia"
# identical to c3_transfer.py / harness_splits.py
TEST_WINDOWS = {2021: (36, 47), 2022: (48, 59), 2023: (60, 71)}


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def somalia_persistence_full(te_win, ycol_bin):
    """
    Somalia's OWN persistence on the FULL D25 persistence-set: rows with
    (a) non-NaN horizon target AND (b) non-NaN persistence input (phase at
    feature-time t). This mirrors evaluate_baselines' n_scorable definition
    and c3_transfer.py's somalia_persistence, but reports the drop counts.
    """
    n_before = len(te_win)
    # drop NaN horizon target
    sub = te_win[te_win[ycol_bin].notna()]
    n_no_target = n_before - len(sub)
    # drop NaN persistence input (phase at feature-time t) — D25 / D6 gap
    n_pre = len(sub)
    sub = sub[sub[TARGET_BINARY].notna()]
    n_no_persist = n_pre - len(sub)

    if len(sub) == 0:
        return None

    y_true = sub[ycol_bin].to_numpy()
    persist = sub[TARGET_BINARY].to_numpy().astype(float)
    obs = sub[IS_OBSERVED].to_numpy().astype(bool) if IS_OBSERVED in sub else None
    res = evaluate_binary(y_true, persist, observed_mask=obs)
    return {
        "n_scorable": len(sub),
        "n_dropped_no_target": n_no_target,
        "n_dropped_no_persist": n_no_persist,
        "persistence_full": res["all"]["pr_auc"],
        "prevalence": res["all"]["prevalence"],
    }


def main():
    section("P1 — SOMALIA TRANSFER BASELINES (D25 aligned row-sets)")

    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)
    df = add_phase_history(df)
    assert_phase_history_safe(df)

    fews_rows = build_fews_baseline_rows(df)
    print(f"period-matched FEWS rows: {len(fews_rows):,}")

    test_all = df[df["country"] == TEST_COUNTRY].copy()
    print(f"Somalia rows total: {len(test_all):,}")

    section("PER-WINDOW (h=3): full persistence-set vs FEWS-aligned subset")
    print(f"{'win':<7}{'n_test':>7}{'fews_n':>7}{'n_drop':>7}"
          f"{'prev':>7}{'persist(full)':>14}{'persist@fews':>14}{'FEWS(aligned)':>14}")
    print("-" * 83)

    rows = []
    h = 3  # only horizon with a FEWS comparator (D24)
    ycol_bin = f"y_binary_h{h}"

    for yr, (t_lo, t_hi) in TEST_WINDOWS.items():
        # Somalia test rows whose PREDICTED period (t+h) falls in the window
        # — identical windowing to c3_transfer.py lines 143-145.
        pred_t = test_all["t_index"] + h
        te_win = test_all[(pred_t >= t_lo) & (pred_t <= t_hi)].copy()
        if len(te_win) == 0:
            continue

        # --- D25 set 1: full Somalia persistence-set ---
        pf = somalia_persistence_full(te_win, ycol_bin)
        if pf is None:
            continue

        # --- D25 set 2: FEWS-aligned subset (harness's own machinery) ---
        # Build a split-like dict exactly as c3_transfer.py does (lines 170-172),
        # so fews_baseline_for_split defines the aligned rows identically.
        split_like = {"test_idx": te_win.index, "horizon": h,
                      "t_lo": t_lo, "t_hi": t_hi, "test_year": yr}
        fb = fews_baseline_for_split(df, fews_rows, split_like, proj_type="near")

        if fb is not None:
            fews_pr = fb["fews"]["all"]["pr_auc"]
            # THE missing number: Somalia persistence RE-SCORED on the SAME
            # aligned rows FEWS was scored on. Already computed by the harness;
            # c3_transfer.py just never read it.
            persist_on_fews = fb["persistence_on_fews_rows"]["all"]["pr_auc"]
            fews_n = fb["n_aligned"]
        else:
            fews_pr, persist_on_fews, fews_n = np.nan, np.nan, 0

        n_drop = pf["n_dropped_no_target"] + pf["n_dropped_no_persist"]

        print(f"{yr:<7}{pf['n_scorable']:>7}{fews_n:>7}{n_drop:>7}"
              f"{pf['prevalence']:>7.3f}{pf['persistence_full']:>14.4f}"
              f"{persist_on_fews:>14.4f}{fews_pr:>14.4f}")

        rows.append({
            "year": yr, "horizon": h,
            "n_test": pf["n_scorable"],
            "fews_n": fews_n,
            "n_dropped": n_drop,
            "n_dropped_no_target": pf["n_dropped_no_target"],
            "n_dropped_no_persist": pf["n_dropped_no_persist"],
            "prevalence": pf["prevalence"],
            "so_persistence_full": pf["persistence_full"],
            "so_persistence_on_fews": persist_on_fews,
            "so_fews_aligned": fews_pr,
            "fews_beats_persist_aligned": (
                "y" if (not np.isnan(fews_pr) and not np.isnan(persist_on_fews)
                        and fews_pr > persist_on_fews) else "n"),
        })

    section("READING")
    print("- so_persistence_full   = Somalia persistence on the FULL set (what")
    print("  c3_transfer.csv's so_persistence column reports).")
    print("- so_persistence_on_fews= Somalia persistence RE-SCORED on the SAME")
    print("  FEWS-aligned rows -> the ONLY valid persistence-vs-FEWS floor.")
    print("- so_fews_aligned       = FEWS on the aligned subset (c3_transfer.csv")
    print("  so_fews column). Valid FEWS-vs-persistence = last two columns.")
    print("- fews_n small relative to n_test => transfer FEWS comparison rests")
    print("  on few rows; report the count.")
    print("- B (transferred model) is NOT re-touched here; its FEWS/persistence")
    print("  comparison should be READ against the aligned floor/ceiling below,")
    print("  and which row-set each B number sits on is documented.")

    out = pd.DataFrame(rows)
    dest = "outputs/diagnostics/c3_transfer_baselines.csv"
    out.to_csv(dest, index=False)
    print(f"\nsaved: {dest}")


if __name__ == "__main__":
    main()
