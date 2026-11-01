import pandas as pd

# ==========================================================
# FEWS Dataset Verification Script
# Thesis: IPC Phase 3+ Forecasting
# ==========================================================

CSV_FILE = "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"

df = pd.read_csv(CSV_FILE)

print("=" * 70)
print("FEWS DATASET VERIFICATION REPORT")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ----------------------------------------------------------
# 1. ADMIN CODE COVERAGE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK 1: ADMINISTRATIVE COVERAGE")
print("=" * 70)

countries = ["Kenya", "Ethiopia", "Somalia"]

coverage = []

for country in countries:

    subset = df[df["country"] == country]

    coverage.append({
        "Country": country,
        "Rows": len(subset),
        "Distinct admin_code": subset["admin_code"].nunique(),
        "Distinct admin_name": subset["admin_name"].nunique(),
        "Start": subset["year_month"].min(),
        "End": subset["year_month"].max()
    })

coverage_df = pd.DataFrame(coverage)

print(coverage_df.to_string(index=False))

# ----------------------------------------------------------
# 2. STABILITY OF ADMIN CODE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK 2: ADMIN CODE STABILITY")
print("=" * 70)

district = input(
    "\nEnter district to test (e.g. Turkana or Gedo): "
).strip()

subset = df[
    df["admin_name"].str.contains(
        district,
        case=False,
        na=False
    )
]

if len(subset) == 0:

    print("\nDistrict not found.")

else:

    codes = subset["admin_code"].unique()

    print(f"\nRows: {len(subset)}")
    print(f"Distinct admin codes: {len(codes)}")
    print("Codes:", sorted(codes))

    print("\nUnique admin_name ↔ admin_code combinations:")
    print(
        subset[["admin_name", "admin_code"]]
        .drop_duplicates()
        .sort_values("admin_code")
        .to_string(index=False)
    )

    print("\nTimeline:")
    print(
        subset[["year_month", "admin_name", "admin_code"]]
        .drop_duplicates()
        .sort_values(["admin_name", "year_month"])
        .to_string(index=False)
    )
# ----------------------------------------------------------
# 3. IPC SCENARIO STRUCTURE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK 3: IPC STRUCTURE")
print("=" * 70)

print("\nCurrent assessment column:")
print("   fews_ipc")

print("\nProjection columns:")
print("   fews_proj_near")
print("   fews_proj_med")

print("\nMissing values")

for col in [
    "fews_ipc",
    "fews_proj_near",
    "fews_proj_med"
]:

    missing = df[col].isna().sum()

    pct = missing / len(df) * 100

    print(f"{col:<22} {missing:>8} ({pct:.2f}%)")

# ----------------------------------------------------------
# 4. DATE RANGE
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK 4: DATE RANGE")
print("=" * 70)

for country in countries:

    subset = df[df["country"] == country]

    print(f"\n{country}")

    print("Start:", subset["year_month"].min())

    print("End  :", subset["year_month"].max())

# ----------------------------------------------------------
# 5. SOMALIA MISSINGNESS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("CHECK 5: SOMALIA MISSINGNESS")
print("=" * 70)

somalia = df[df["country"] == "Somalia"]

summary = (
    somalia
    .groupby("year")["fews_ipc"]
    .agg(
        total="size",
        missing=lambda x: x.isna().sum()
    )
)

summary["percent_missing"] = (
    summary["missing"] /
    summary["total"] *
    100
).round(2)

print(summary)

print("\nOverall Somalia missingness:")

overall_missing = somalia["fews_ipc"].isna().sum()

overall_pct = (
    overall_missing /
    len(somalia) *
    100
)

print(f"{overall_missing} missing values")
print(f"{overall_pct:.2f}% missing")

# ----------------------------------------------------------
# 6. EXTRA DIAGNOSTICS
# ----------------------------------------------------------

print("\n" + "=" * 70)
print("EXTRA DIAGNOSTICS")
print("=" * 70)

print("\nAdministrative units per year:")

for country in countries:

    subset = df[df["country"] == country]

    counts = (
        subset
        .groupby("year")["admin_code"]
        .nunique()
    )

    print(f"\n{country}")

    print(counts.describe())

print("\nSample administrative names")

for country in countries:

    subset = df[df["country"] == country]

    print(f"\n{country}")

    print(
        sorted(
            subset["admin_name"]
            .dropna()
            .unique()
        )[:20]
    )

print("\nVerification complete.")
print("=" * 70)