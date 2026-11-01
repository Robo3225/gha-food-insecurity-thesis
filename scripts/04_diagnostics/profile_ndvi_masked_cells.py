"""
======================================================================

profile_ndvi_masked_cells.py

Purpose
-------
Profile the fully masked NDVI unit-months identified during the
composite aggregation step.

This script summarizes:

1. Temporal distribution of masked cells
2. Spatial clustering using FNID prefixes
3. Whether every masked cell was successfully interpolated

Inputs
------
data/processed/kenya_ndvi_masked_cells.csv

Outputs
-------
Console summary
data/processed/kenya_ndvi_masked_profile_by_month.csv
data/processed/kenya_ndvi_masked_profile_by_area.csv

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED = PROJECT_ROOT / "data" / "processed"

MASKED_FILE = PROCESSED / "kenya_ndvi_masked_cells.csv"

MONTH_OUT = PROCESSED / "kenya_ndvi_masked_profile_by_month.csv"
AREA_OUT = PROCESSED / "kenya_ndvi_masked_profile_by_area.csv"

# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

masked = pd.read_csv(MASKED_FILE)

print("\n==============================")
print("NDVI MASKED CELL PROFILE")
print("==============================\n")

print(f"Total originally masked cells : {len(masked)}")

print(f"Successfully interpolated     : {masked['interpolated'].sum()}")

print(f"Still missing                 : {masked['still_missing'].sum()}")

# ---------------------------------------------------------------------
# Temporal profile
# ---------------------------------------------------------------------

print("\n------------------------------")
print("Masked cells by year/month")
print("------------------------------")

by_month = (
    masked
    .groupby(["year", "month"])
    .size()
    .rename("masked_cells")
    .reset_index()
    .sort_values(["year", "month"])
)

print(by_month.to_string(index=False))

by_month.to_csv(MONTH_OUT, index=False)

# ---------------------------------------------------------------------
# Spatial profile
# ---------------------------------------------------------------------

print("\n------------------------------")
print("Masked cells by FNID prefix")
print("------------------------------")

# First 9 characters groups neighbouring administrative areas
masked["admin_prefix"] = masked["FNID"].str[:9]

by_area = (
    masked
    .groupby("admin_prefix")
    .size()
    .rename("masked_cells")
    .reset_index()
    .sort_values("masked_cells", ascending=False)
)

print(by_area.to_string(index=False))

by_area.to_csv(AREA_OUT, index=False)

# ---------------------------------------------------------------------
# FNIDs
# ---------------------------------------------------------------------

print("\n------------------------------")
print("Affected FNIDs")
print("------------------------------")

print(
    masked[
        [
            "FNID",
            "year",
            "month",
            "interpolated",
            "still_missing",
        ]
    ]
    .sort_values(["year", "month", "FNID"])
    .to_string(index=False)
)

print("\n==============================")
print("Outputs written")
print("==============================")

print(MONTH_OUT)
print(AREA_OUT)