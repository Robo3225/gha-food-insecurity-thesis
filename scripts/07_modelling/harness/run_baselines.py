"""
======================================================================

run_baselines.py  (M10.1 capstone — real floor / ceiling / prevalence)

Loads the real frozen v2 panels through the harness, builds the
horizon-shifted targets, generates the leakage-audited walk-forward
splits, and reports for every split:
  - class prevalence (the class-weighting input)
  - PERSISTENCE baseline PR-AUC (the validity floor)
  - FEWS NET projection baseline PR-AUC at h=3 (the significance ceiling)

No models. This is the last M10.1 output before XGBoost — the numbers
the models must beat, reported against both baselines as built (D24).

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness_data import load_all
from harness_target import add_horizon_targets, assert_target_not_leaking
from harness_splits import all_splits
from harness_eval import evaluate_baselines
from harness_fews import build_fews_baseline_rows, fews_baseline_for_split


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    section("M10.1 BASELINES — real floor / ceiling / prevalence")

    df = load_all()
    df = add_horizon_targets(df)
    for h in (1, 3):
        assert_target_not_leaking(df, h)

    section("WALK-FORWARD SPLITS (leakage-audited)")
    splits = all_splits(df, verbose=True)

    section("FEWS projection rows (period-matched, D24)")
    fews_rows = build_fews_baseline_rows(df)
    print(f"total period-matched FEWS rows: {len(fews_rows):,}")

    section("PER-SPLIT: prevalence, persistence floor, FEWS ceiling")
    print(f"{'split':<12}{'n_test':>7}{'prev':>7}{'persist(all)':>13}"
          f"{'fews_n':>7}{'persist@fews':>13}{'FEWS':>7}")
    print("-" * 70)

    rows = []
    for s in splits:
        h = s["horizon"]; yr = s["test_year"]
        base = evaluate_baselines(df, s)
        persist_pr = base["persistence"]["all"].get("pr_auc", np.nan)
        prev = base["persistence"]["all"].get("prevalence", np.nan)
        n_test = base.get("n_scorable", np.nan)
        n_drop = base.get("n_dropped_no_target", 0) + base.get("n_dropped_no_persist", 0)

        # FEWS ceiling: h=3 near only (D24); None at h=1. Returns FEWS AND
        # persistence scored on the IDENTICAL aligned rows (D25).
        fews = fews_baseline_for_split(df, fews_rows, s, proj_type="near")
        if fews is not None:
            fews_pr = fews["fews"]["all"].get("pr_auc", np.nan)
            persist_on_fews = fews["persistence_on_fews_rows"]["all"].get("pr_auc", np.nan)
            fews_n = fews.get("n_aligned", 0)
        else:
            fews_pr, persist_on_fews, fews_n = np.nan, np.nan, 0

        label = f"{yr}/h{h}"
        pf = persist_on_fews if not np.isnan(persist_on_fews) else float("nan")
        fp = fews_pr if not np.isnan(fews_pr) else float("nan")
        print(f"{label:<12}{n_test:>7}{prev:>7.2f}{persist_pr:>13.3f}"
              f"{fews_n:>7}{pf:>13.3f}{fp:>7.3f}")
        rows.append({"split": label, "horizon": h, "year": yr,
                     "n_test": n_test, "prevalence": prev,
                     "persistence_pr_auc": persist_pr,
                     "fews_pr_auc": fews_pr, "persist_on_fews_pr_auc": persist_on_fews,
                     "fews_n": fews_n, "n_dropped": n_drop})

    section("PREVALENCE SUMMARY (class-weighting input)")
    summ = pd.DataFrame(rows)
    # overall prevalence per country per horizon (observed positives)
    for h in (1, 3):
        sub = summ[summ.horizon == h]
        print(f"h={h}: prevalence range {sub.prevalence.min():.3f}"
              f"–{sub.prevalence.max():.3f} across windows")

    section("READING")
    print("- persistence PR-AUC = the FLOOR. A model beating this is not just")
    print("  learning forward-fill autocorrelation.")
    print("- persist@fews vs FEWS: the FAIR floor/ceiling comparison (D25) -")
    print("  BOTH scored on the identical FEWS-aligned rows. This is the")
    print("  valid 'does FEWS beat persistence' test.")
    print("- prevalence = class-weighting input for XGBoost (M10.2).")
    print("- h=1 has no FEWS row by design (D24: FEWS has no 1-month forecast).")

    summ.to_csv("outputs/diagnostics/m10_baselines.csv", index=False)
    print("\nsaved: outputs/diagnostics/m10_baselines.csv")


if __name__ == "__main__":
    main()
