"""
======================================================================
Inspect Humanitarian Assistance Adjustment

Purpose
-------
Investigate whether the fews_ipc_adjusted column is associated with
humanitarian assistance (fews_ha).

Questions answered
------------------
1. Are adjustments concentrated where humanitarian assistance exists?
2. Does humanitarian assistance almost perfectly predict adjustment?
3. Should fews_ipc_adjusted be interpreted as an assistance-adjusted
   IPC classification?

Author:
Robert Odingo MSc Thesis
======================================================================
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

print("=" * 70)
print("INSPECT HUMANITARIAN ASSISTANCE ADJUSTMENT")
print("=" * 70)

raw = pd.read_csv(INPUT_FILE)

for country in ["Kenya", "Ethiopia", "Somalia"]:

    print("\n" + "=" * 70)
    print(country.upper())
    print("=" * 70)

    df = raw[
        raw["country"].str.lower() == country.lower()
    ].copy()

    required = [
        "fews_ipc",
        "fews_ipc_adjusted",
        "fews_ha",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"Missing columns: {missing}")
        continue

    both = df.dropna(
        subset=["fews_ipc", "fews_ipc_adjusted"]
    ).copy()

    both["changed"] = (
        both["fews_ipc"] != both["fews_ipc_adjusted"]
    )

    print(f"\nRows analysed : {len(both):,}")

    print("\nAdjustment vs Humanitarian Assistance")
    print("--------------------------------------")

    table = pd.crosstab(
        both["changed"],
        both["fews_ha"],
        margins=True
    )

    print(table)

    print("\nRow percentages")
    print("----------------")

    pct = pd.crosstab(
        both["changed"],
        both["fews_ha"],
        normalize="index"
    ).round(3)

    print(pct)

print("\n" + "=" * 70)
print("Diagnostic complete.")
print("=" * 70)