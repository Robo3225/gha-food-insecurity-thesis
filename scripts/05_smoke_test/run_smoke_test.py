"""
======================================================================

run_smoke_test.py  (scripts/05_smoke_test/ — DISPOSABLE)

Milestone 7.5 smoke test runner.

Answers: can the pipeline run end to end, and does the model learn
anything beyond label autocorrelation?

Sequence:
  1. Load frozen v1 model panel
  2. Take a stratified unit subset (crisis-active + stable)
  3. XGBoost control (walk-forward, per test year) - disambiguates
     'TFT problem' from 'data problem'
  4. TFT five-diagnostic smoke test
  5. Print verdicts

Run on the frozen dataset only:
    data/processed/v1/kenya_model_panel.csv

======================================================================
"""

from pathlib import Path
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import smoke_common as SC
from smoke_xgboost import run_xgboost_control
from smoke_tft import run_tft_smoke, print_verdicts, pick_device

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "processed" / "v1" / "kenya_model_panel.csv"

N_UNITS = 60          # small subset for the M4 plumbing test
TEST_YEAR = 2022      # the drought window; the informative test


def main():
    print("=" * 70)
    print("MILESTONE 7.5 SMOKE TEST")
    print("=" * 70)
    print(f"device: {pick_device()}")

    df = SC.load_panel(PANEL)
    print(f"Full panel: {len(df):,} rows, {df[SC.GROUP].nunique()} units")

    sub = SC.stratified_subset(df, n_units=N_UNITS, seed=0)
    print(f"Subset: {len(sub):,} rows, {sub[SC.GROUP].nunique()} units "
          f"(stratified, smoke-test only)")

    # --- XGBoost control first: fast, proves the panel carries signal ---
    print("\n" + "=" * 70)
    print("XGBOOST CONTROL")
    print("=" * 70)
    for yr in [2021, TEST_YEAR, 2023]:
        r = run_xgboost_control(sub, yr)
        x, p = r["xgboost"], r["persistence"]
        flag = "beats persistence" if r["beats_persistence"] else "DOES NOT beat persistence"
        print(f"{yr}: XGB PR-AUC={x['pr_auc']:.3f}  "
              f"persistence={p['pr_auc']:.3f}  -> {flag}")

    # --- TFT smoke test on the drought window ---
    print("\n" + "=" * 70)
    print(f"TFT SMOKE TEST (test year {TEST_YEAR})")
    print("=" * 70)
    train = sub[sub["year"] < TEST_YEAR]
    val = sub[sub["year"] == TEST_YEAR]
    results = run_tft_smoke(train, val, val)
    print_verdicts(results)

    # --- overall read ---
    print("\n" + "=" * 70)
    print("READ")
    print("=" * 70)
    if results.get("d5_beats_persistence"):
        print("TFT beats persistence -> pipeline works, TFT viable. Proceed.")
    elif results.get("d1_dataset_constructs") and results.get("d2_trains"):
        print("TFT runs but does NOT beat persistence.")
        print("Check: did XGBoost beat persistence above?")
        print("  - XGB beats, TFT doesn't -> TFT-specific (config/convergence).")
        print("    Try encoder length 6 or 18; check variable roles.")
        print("  - Neither beats -> data/target problem, not the model.")
        print("    This is the LightGBM kill-switch signal.")
    else:
        print("TFT failed to construct or train - see diagnostics above.")


if __name__ == "__main__":
    main()
