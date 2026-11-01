import geopandas as gpd
import pandas as pd

shapefile = "Ethiopia/ET_Admin3_2021.shp"

gdf = gpd.read_file(shapefile)

print("=" * 70)
print("SHAPEFILE VERIFICATION")
print("=" * 70)

print(f"\nNumber of polygons: {len(gdf)}")

print("\nColumns:")
for col in gdf.columns:
    print(f" - {col}")

print("\nPotential join fields:")
for col in gdf.columns:
    if any(key in col.lower() for key in ["admin", "fnid", "id", "code", "name"]):
        print(f" - {col}")

print("\nFirst five records (without geometry):")
print(gdf.drop(columns="geometry").head())

print("\nUnique polygon identifiers:")
for col in gdf.columns:
    if col.lower() != "geometry":
        print(f"{col}: {gdf[col].nunique()} unique values")

print(gdf[["FNID","ADMIN3"]].head(20))

# Load tabular data
df = pd.read_csv("FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv")

# Ethiopia only
eth = (
    df[df["country"] == "Ethiopia"]
    [["admin_code", "admin_name"]]
    .drop_duplicates()
)

print("Unique Ethiopia reporting units:", len(eth))

# Search for Aba-Korow
print("\nAba-Korow in tabular dataset:")
print(
    eth[eth["admin_name"].str.contains("Aba-Korow", case=False, na=False)]
)

parts = eth["admin_name"].str.split(", ", expand=True)

print(parts.head(20))

print("\n" + "=" * 70)
print("CHECK FOR DUPLICATES IN THE TABULAR DATA")
print("=" * 70)

import pandas as pd

# Load the tabular dataset
df = pd.read_csv("FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv")

# Ethiopia only
eth = (
    df[df["country"] == "Ethiopia"]
    [["admin_code", "admin_name"]]
    .drop_duplicates()
)

print(f"\nUnique admin_codes: {eth['admin_code'].nunique()}")
print(f"Unique admin_names: {eth['admin_name'].nunique()}")

# --------------------------------------------------
# 1. One admin_code with multiple names?
# --------------------------------------------------

print("\nAdmin codes linked to multiple names:")

codes = (
    eth.groupby("admin_code")["admin_name"]
       .nunique()
)

multi_names = codes[codes > 1]

if len(multi_names) == 0:
    print("None")
else:
    print(multi_names)

    print("\nDetails:")
    print(
        eth[
            eth["admin_code"].isin(multi_names.index)
        ].sort_values("admin_code")
    )

# --------------------------------------------------
# 2. One admin_name with multiple codes?
# --------------------------------------------------

print("\nAdmin names linked to multiple codes:")

names = (
    eth.groupby("admin_name")["admin_code"]
       .nunique()
)

multi_codes = names[names > 1]

if len(multi_codes) == 0:
    print("None")
else:
    print(multi_codes)

    print("\nDetails:")
    print(
        eth[
            eth["admin_name"].isin(multi_codes.index)
        ].sort_values("admin_name")
    )