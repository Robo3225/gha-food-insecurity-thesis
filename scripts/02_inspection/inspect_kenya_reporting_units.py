import pandas as pd

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv("FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv")

kenya = df[df["country"] == "Kenya"].copy()

print("=" * 70)
print("KENYA REPORTING UNIT INSPECTION")
print("=" * 70)

# ------------------------------------------------------
# BASIC COUNTS
# ------------------------------------------------------

print("\nRows:")
print(len(kenya))

print("\nUnique reporting units (admin_code):")
print(kenya["admin_code"].nunique())

print("\nUnique admin_name:")
print(kenya["admin_name"].nunique())

# ------------------------------------------------------
# SAMPLE NAMES
# ------------------------------------------------------

print("\nFirst 40 unique reporting-unit names:\n")

for name in sorted(kenya["admin_name"].drop_duplicates())[:40]:
    print(name)

# ------------------------------------------------------
# LOOK FOR LIVELIHOOD KEYWORDS
# ------------------------------------------------------

keywords = [
    "Pastoral",
    "Agropastoral",
    "Agricultural",
    "Mixed",
    "Marginal",
    "Fishing",
    "High Potential",
    "Low Potential",
    "Irrigated",
    "Urban",
    "Coastal",
    "Ranching",
    "Livestock",
    "Riverine",
    "Forest"
]

print("\n" + "=" * 70)
print("KEYWORD SEARCH")
print("=" * 70)

found = False

for word in keywords:

    matches = kenya[
        kenya["admin_name"].str.contains(
            word,
            case=False,
            na=False
        )
    ]["admin_name"].drop_duplicates()

    if len(matches):

        found = True

        print(f"\n{word} ({len(matches)} matches)")

        for m in sorted(matches)[:10]:
            print("  ", m)

if not found:
    print("No livelihood keywords detected.")

# ------------------------------------------------------
# HOW MANY REPORTING UNITS PER DISTRICT?
# ------------------------------------------------------

print("\n" + "=" * 70)
print("REPORTING UNITS PER DISTRICT")
print("=" * 70)

parts = kenya["admin_name"].str.split(", ", expand=True)

if parts.shape[1] >= 4:

    kenya["ZONE"] = parts[0]
    kenya["DISTRICT"] = parts[1]
    kenya["COUNTY"] = parts[2]

    district_counts = (
        kenya.groupby("DISTRICT")["admin_code"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(district_counts.head(25))

else:
    print("admin_name structure is different than expected.")

print("\nInspection complete.")