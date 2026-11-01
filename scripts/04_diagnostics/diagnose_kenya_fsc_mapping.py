import pandas as pd
import geopandas as gpd

# =====================================================
# FILES
# =====================================================

TABULAR = "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
SHAPEFILE = "ETHIOPIAFSC2/ET_Admin3C_2021_2.shp"

# =====================================================
# LOAD DATA
# =====================================================

print("Loading World Bank dataset...")
df = pd.read_csv(TABULAR)

print("Loading Ethiopia FSC shapefile...")
gdf = gpd.read_file(SHAPEFILE)

# =====================================================
# WORLD BANK DATA
# =====================================================

wb = (
    df[df["country"] == "Ethiopia"]
    [["admin_code", "admin_name"]]
    .drop_duplicates()
    .copy()
)

print(f"\nWorld Bank reporting units : {len(wb)}")
print(f"FSC polygons              : {len(gdf)}")

# =====================================================
# SPLIT ADMIN NAME
#
# Format:
# Aba-Korow, Shabelle, Somali, Ethiopia
#
# ADMIN3, ADMIN2, ADMIN1, COUNTRY
# =====================================================

parts = wb["admin_name"].str.split(", ", expand=True)

wb["ADMIN3"] = parts[0]
wb["ADMIN2"] = parts[1]
wb["ADMIN1"] = parts[2]

# =====================================================
# CLEAN FUNCTION
# =====================================================

def clean(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = (
        text.replace("/", " ")
            .replace("-", " ")
            .replace(",", " ")
            .replace("'", "")
            .replace("\r", " ")
            .replace("\n", " ")
    )

    text = " ".join(text.split())

    return text

# =====================================================
# NORMALIZE WORLD BANK
# =====================================================

wb["ADMIN1"] = wb["ADMIN1"].apply(clean)
wb["ADMIN2"] = wb["ADMIN2"].apply(clean)
wb["ADMIN3"] = wb["ADMIN3"].apply(clean)

# =====================================================
# NORMALIZE FSC
# =====================================================

gdf["ADMIN1"] = gdf["ADMIN1"].fillna("").apply(clean)
gdf["ADMIN2"] = gdf["ADMIN2"].fillna("").apply(clean)
gdf["ADMIN3"] = gdf["ADMIN3"].fillna("").apply(clean)

# =====================================================
# COMPARE
# =====================================================

matches = []
unmatched = []

for _, row in gdf.iterrows():

    found = wb[
        (wb["ADMIN1"] == row["ADMIN1"]) &
        (wb["ADMIN2"] == row["ADMIN2"]) &
        (wb["ADMIN3"] == row["ADMIN3"])
    ]

    if len(found) > 0:

        matches.append({
            "FNID": row["FNID"],
            "admin_code": found.iloc[0]["admin_code"],
            "admin_name": found.iloc[0]["admin_name"]
        })

    else:

        unmatched.append({
            "FNID": row["FNID"],
            "ADMIN1": row["ADMIN1"],
            "ADMIN2": row["ADMIN2"],
            "ADMIN3": row["ADMIN3"]
        })

# =====================================================
# RESULTS
# =====================================================

matches = pd.DataFrame(matches)
unmatched = pd.DataFrame(unmatched)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nMatched    : {len(matches)}")
print(f"Unmatched  : {len(unmatched)}")

print(f"\nSuccess rate: {100 * len(matches) / len(gdf):.2f}%")

# =====================================================
# SHOW SAMPLE MATCHES
# =====================================================

print("\nSample matches:\n")

print(matches.head(20))

# =====================================================
# SHOW SAMPLE UNMATCHED
# =====================================================

print("\nFirst unmatched polygons:\n")

if len(unmatched):

    print(unmatched.head(20))

else:

    print("None")

# =====================================================
# SAVE
# =====================================================

matches.to_csv(
    "ethiopia_matches.csv",
    index=False
)

unmatched.to_csv(
    "ethiopia_unmatched.csv",
    index=False
)

print("\nSaved:")

print("ethiopia_matches.csv")
print("ethiopia_unmatched.csv")