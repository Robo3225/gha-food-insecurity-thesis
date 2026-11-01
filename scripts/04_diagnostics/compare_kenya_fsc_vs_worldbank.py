import pandas as pd
import geopandas as gpd

# ==========================================================
# FILES
# ==========================================================

TABULAR = "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"

SHAPEFILE = "KENYAFSC/KE_Admin3_LHZ_2016.2.shp"

# ==========================================================
# LOAD
# ==========================================================

print("Loading World Bank dataset...")

df = pd.read_csv(TABULAR)

print("Loading Kenya FSC shapefile...")

gdf = gpd.read_file(SHAPEFILE)

# ==========================================================
# WORLD BANK
# ==========================================================

wb = (
    df[df["country"] == "Kenya"]
    [["admin_code", "admin_name"]]
    .drop_duplicates()
    .copy()
)

# Split into components

parts = wb["admin_name"].str.split(", ", expand=True)

print("\nNumber of components in admin_name:")

print(parts.shape[1])

# Kenya should have 5 components

wb["REGION"] = parts[0]

wb["LZNAME"] = parts[1]

wb["ADMIN3"] = parts[2]

wb["ADMIN1"] = parts[3]

# ==========================================================
# FSC
# ==========================================================

fsc = gdf[[
    "FNID",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
    "LZNAME"
]].copy()

# ==========================================================
# COMPARE
# ==========================================================

comparison = wb.merge(

    fsc,

    on=["ADMIN1", "ADMIN3", "LZNAME"],

    how="outer",

    indicator=True

)

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(comparison["_merge"].value_counts())

print("\nOnly in World Bank:")

wb_only = comparison[
    comparison["_merge"] == "left_only"
]

print(wb_only[[
    "admin_code",
    "LZNAME",
    "ADMIN3",
    "ADMIN1"
]])

print("\nOnly in FSC:")

fsc_only = comparison[
    comparison["_merge"] == "right_only"
]

print(fsc_only[[
    "FNID",
    "LZNAME",
    "ADMIN3",
    "ADMIN1"
]])

print("\nSaving CSVs...")

wb_only.to_csv(
    "kenya_worldbank_only.csv",
    index=False
)

fsc_only.to_csv(
    "kenya_fsc_only.csv",
    index=False
)

print("Done.")