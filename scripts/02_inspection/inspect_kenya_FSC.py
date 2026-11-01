"""
======================================================================

inspect_kenya_FSC.py

Purpose
-------
Inspect the Kenya FSC shapefile before validation.

This script reports:

- Number of polygons
- Coordinate Reference System (CRS)
- Available columns
- Number of unique values per attribute
- First 20 records
- Potential hierarchy fields used in the crosswalk

======================================================================
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

# =====================================================================
# Configuration
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kenya"
    / "KENYAFSC"
    / "KE_Admin3_LHZ_2016.2.shp"
)


FEWS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)


# =====================================================================
# Load shapefile
# =====================================================================

gdf = gpd.read_file(SHAPEFILE)

print("\nAdministrative hierarchy with livelihood zone:")
print(
    gdf[
        [
            "ADMIN1",
            "ADMIN2",
            "ADMIN3",
            "LZNAME",
        ]
    ].head(20)
)


# ---------------------------------------------------------------------
# TEMPORARY DEBUGGING - CHECK 2
# ---------------------------------------------------------------------

key = ["ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"]

print("\nFSC uniqueness check")
print("--------------------")
print("Total rows:", len(gdf))
print(
    "Unique (ADMIN1, ADMIN2, ADMIN3, LZNAME):",
    len(gdf.drop_duplicates(subset=key)),
)

# ---------------------------------------------------------------------
# END TEMPORARY DEBUGGING
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# CHECK 3 - Compare FEWS and FSC livelihood-zone names
# ---------------------------------------------------------------------

print("\nCHECK 3 - Inspect 5-component FEWS structure")
print("--------------------------------------------")

# Load FEWS
fews = pd.read_csv(FEWS_FILE)

# Kenya only
fews = fews[
    fews["country"].str.lower() == "kenya"
].copy()

# Number of comma-separated components
comp = fews["admin_name"].str.split(",").apply(len)

# Split into columns
parts = fews["admin_name"].str.split(",", expand=True)

# Show distinct 5-component structures
five = (
    parts.loc[comp == 5]
    .drop_duplicates()
    .sort_values(0)
)

print("\nUnique values in first component of 5-part rows:")
print("-----------------------------------------------")

print(
    sorted(
        parts.loc[comp == 5, 0]
             .str.strip()
             .drop_duplicates()
             .tolist()
    )
)

# ----------------------------------------------------------
# Check whether "Southeastern Medium Potential" is split
# ----------------------------------------------------------

mask = parts[0].str.strip() == "Southeastern Medium Potential"

print("\nRows for 'Southeastern Medium Potential':")
print("-----------------------------------------")

print(
    parts.loc[mask]
         .drop_duplicates()
         .to_string(index=False)
)

#----------------------------------------------------------------------
# END CHECK 3
#----------------------------------------------------------------------

print(
    gdf[
        [
            "ADMIN1",
            "ADMIN2",
            "ADMIN3",
            "LZCODE",
            "LZNAME",
        ]
    ].head(30)
)

print(gdf["LZCODE"].unique())

# =====================================================================
# Basic information
# =====================================================================

print("=" * 70)
print("KENYA FSC SHAPEFILE")
print("=" * 70)

print(f"\nFile:\n{SHAPEFILE}")

print("\nNumber of polygons:")
print(len(gdf))

print("\nCoordinate Reference System (CRS):")
print(gdf.crs)

# =====================================================================
# Columns
# =====================================================================

print("\nColumns:")

for column in gdf.columns:
    print(f"- {column}")

# =====================================================================
# Unique values
# =====================================================================

print("\nUnique values per column:")

for column in gdf.columns:

    if column == "geometry":
        continue

    try:
        print(f"{column}: {gdf[column].nunique()}")

    except Exception:
        pass

# =====================================================================
# Preview
# =====================================================================

print("\nFirst 20 rows:")

preview_columns = [
    column
    for column in gdf.columns
    if column != "geometry"
]

print(gdf[preview_columns].head(20))

# =====================================================================
# Potential hierarchy fields
# =====================================================================

print("\nPotential hierarchy fields:")

for column in gdf.columns:

    upper = column.upper()

    if (
        "ADMIN" in upper
        or "FNID" in upper
        or "LHZ" in upper
        or "ZONE" in upper
    ):
        print(f"- {column}")

