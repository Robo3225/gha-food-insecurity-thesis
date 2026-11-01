"""
check_lst_baseline_skew.py

Settle whether the window-negative LST anomaly (Ethiopia -0.35, Kenya
-0.32 mean) is a GENUINE climate signal or a standardization artifact.

Logic: the anomaly is (window_lst - baseline_mean) / baseline_sd, per
unit per calendar month. A negative mean anomaly means window LST <
baseline LST on average. That is EITHER:
  (a) genuine - 2018-2023 really was cooler in raw Celsius than the
      2000-2020 baseline (honest anomaly), OR
  (b) artifact - the baseline_mean is biased warm relative to the
      window for reasons unrelated to real temperature (miscalibration).

Test: compare RAW Celsius directly.
  - baseline mean raw LST (from the climatology file's clim_mean)
  - window mean raw LST (from the feature panel's lst_celsius)
  matched per unit per calendar month.
If window_raw < baseline_raw by ~the same magnitude the anomaly implies,
the anomaly is HONEST (genuine cooling). If raw temps are ~equal but the
anomaly is negative, the standardization is the problem.

Country-parameterised.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from config.countries import get_country

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    args = ap.parse_args()
    slug = get_country(args.country)["name"].lower()

    panel = pd.read_csv(ROOT / "data" / "processed" / f"{slug}_lst_panel.csv")
    clim = pd.read_csv(ROOT / "data" / "climatology" / f"{slug}_lst_climatology.csv")

    print("=" * 70)
    print(f"{slug.upper()} LST BASELINE-SKEW CHECK")
    print("=" * 70)

    # climatology columns: FNID, month, clim_mean, clim_sd, n_years (typical)
    print(f"climatology columns: {list(clim.columns)}")
    mean_col = "clim_mean" if "clim_mean" in clim.columns else None
    if mean_col is None:
        # try to find the baseline-mean column
        cand = [c for c in clim.columns if "mean" in c.lower()]
        mean_col = cand[0] if cand else None
        print(f"using baseline-mean column: {mean_col}")

    # 1. Direct raw comparison: window mean lst_celsius vs baseline clim_mean,
    #    matched per unit per calendar month.
    win = (panel.groupby(["FNID", "month"])["lst_celsius"]
           .mean().reset_index().rename(columns={"lst_celsius": "window_raw"}))
    base = clim[["FNID", "month", mean_col]].rename(columns={mean_col: "baseline_raw"})
    m = win.merge(base, on=["FNID", "month"], how="inner")
    m["raw_diff"] = m["window_raw"] - m["baseline_raw"]

    print(f"\nmatched unit-months: {len(m):,}")
    print(f"\nRAW LST (Celsius) comparison:")
    print(f"  window mean raw   : {m['window_raw'].mean():.2f} C")
    print(f"  baseline mean raw : {m['baseline_raw'].mean():.2f} C")
    print(f"  mean raw diff     : {m['raw_diff'].mean():+.2f} C  "
          f"(window minus baseline)")
    print(f"  median raw diff   : {m['raw_diff'].median():+.2f} C")

    # 2. What anomaly would that raw diff imply, given typical sd?
    if "clim_sd" in clim.columns:
        typ_sd = clim["clim_sd"].median()
        implied = m["raw_diff"].mean() / typ_sd
        print(f"\n  median baseline sd: {typ_sd:.2f} C")
        print(f"  implied anomaly from raw diff: {implied:+.2f} SD")
        print(f"  (compare to observed panel anomaly mean ~ -0.35 SD)")

    # 3. Verdict logic
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    raw_diff = m["raw_diff"].mean()
    if raw_diff < -0.5:
        print(f"GENUINE: window is {raw_diff:+.2f} C cooler than baseline in RAW")
        print("Celsius. The negative anomaly is honest - 2018-2023 really was")
        print("cooler than 2000-2020 for this country. LST-weak-signal is a real")
        print("finding, not a calibration artifact.")
    elif raw_diff > 0.5:
        print(f"ARTIFACT SUSPECTED: window is {raw_diff:+.2f} C WARMER than baseline")
        print("in raw Celsius, yet the anomaly is negative. The standardization is")
        print("inconsistent with the raw temperatures - investigate apply_anomaly /")
        print("climatology matching.")
    else:
        print(f"NEAR-NEUTRAL raw diff ({raw_diff:+.2f} C). The window and baseline")
        print("are close in raw Celsius, so a -0.35 SD anomaly mean would be")
        print("LARGER than the raw diff justifies - possible mild standardization")
        print("skew. Check the implied-anomaly line above against observed.")

    # 4. Per-year raw diff — is the cooling concentrated or spread?
    print("\nper-year window raw LST vs baseline (by calendar-month match):")
    py = panel.merge(base, on=["FNID", "month"], how="inner")
    py["raw_diff"] = py["lst_celsius"] - py["baseline_raw"]
    print(py.groupby("year")["raw_diff"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
