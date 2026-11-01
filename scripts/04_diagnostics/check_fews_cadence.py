"""
check_fews_cadence.py

Pre-application gate for D24: does FEWS assess on a
regular tri-annual (~4-month) cadence across 2018-2023 for all three
countries? Option (B) exact-period alignment depends on it.

An assessment period is a year_month where fews_ipc is genuinely present
(an actual assessment, not a forward-filled or projected value). The raw
World Bank source carries the real assessment months (before any fill).

Reads the RAW source. Reports, per country, the distinct assessment
months in 2018-2023 and the month-gaps between consecutive ones.
  - gaps mostly 4  -> regular tri-annual, D24 mapping holds
  - mixed 3/4/5    -> irregular, join must align on ACTUAL periods
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw/world_bank/FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv")

COUNTRY_MAP = {"KE": "kenya", "ET": "ethiopia", "SO": "somalia",
               "Kenya":"kenya","Ethiopia":"ethiopia","Somalia":"somalia"}

def main():
    raw = pd.read_csv(RAW)
    print("country values:", sorted(raw["country"].astype(str).unique())[:10])

    raw["ym_int"] = raw["year"].astype(int) * 12 + (raw["month"].astype(int) - 1)

    for country in sorted(raw["country"].astype(str).unique()):
        sub = raw[(raw["country"].astype(str) == country) & raw["fews_ipc"].notna()]
        # distinct assessment months in the study window
        months = sub[(sub["year"] >= 2018) & (sub["year"] <= 2023)]
        periods = sorted(months["ym_int"].unique())
        if not periods:
            print(f"\n{country}: no 2018-2023 assessments found")
            continue
        gaps = [periods[i+1] - periods[i] for i in range(len(periods) - 1)]
        # readable year_month labels
        labels = [f"{p//12}-{p%12+1:02d}" for p in periods]
        print(f"\n{country}: {len(periods)} assessments 2018-2023")
        print(f"  months: {labels}")
        print(f"  gaps (months): {gaps}")
        from collections import Counter
        print(f"  gap distribution: {dict(sorted(Counter(gaps).items()))}")
        n4 = sum(1 for g in gaps if g == 4)
        print(f"  -> {'REGULAR tri-annual' if n4/max(len(gaps),1) >= 0.7 else 'IRREGULAR'} "
              f"({n4}/{len(gaps)} gaps == 4 months)")

    print("\n" + "="*66)
    print("Read: if gaps are mostly 4, D24's near->3mo / med->longer mapping")
    print("holds. If mixed, the FEWS join must align on the ACTUAL assessment")
    print("and projection periods present in the data, not an assumed 4-mo step.")

if __name__ == "__main__":
    main()
