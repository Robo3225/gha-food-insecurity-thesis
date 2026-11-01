import pandas as pd

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv")

print("\n========== COLUMNS ==========")
print(df.columns.tolist())

# ----------------------------------------------------
# Automatically identify important columns
# ----------------------------------------------------
def find_column(possible_names):
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                return col
    return None

country_col = find_column(["country"])
admin_col   = find_column(["admin"])
fnid_col    = find_column(["fnid"])
date_col    = find_column(["date"])
phase_col   = find_column(["phase","ipc"])
scenario_col= find_column(["scenario","period","ml","cs"])
unit_col    = find_column(["district","admin2","name"])

print("\nDetected columns")
print("Country :", country_col)
print("Admin   :", admin_col)
print("FNID    :", fnid_col)
print("Date    :", date_col)
print("Phase   :", phase_col)
print("Scenario:", scenario_col)
print("Unit    :", unit_col)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ====================================================
# CHECK 1
# ====================================================
print("\n==============================")
print("CHECK 1 - ADMIN2 COVERAGE")
print("==============================")

for country in ["Kenya","Ethiopia","Somalia"]:
    subset = df[df[country_col].str.contains(country, case=False, na=False)]

    print(f"\n{country}")
    print("Rows:", len(subset))

    if unit_col:
        print("Distinct reporting units:",
              subset[unit_col].nunique())

    if admin_col:
        print("\nAdministrative levels:")
        print(subset[admin_col].value_counts())

# ====================================================
# CHECK 2
# ====================================================
print("\n==============================")
print("CHECK 2 - FNID STABILITY")
print("==============================")

district = input(
    "\nEnter district (e.g. Turkana): "
)

subset = df[df[unit_col].astype(str).str.contains(
    district,
    case=False,
    na=False
)]

print("\nRows:", len(subset))

if fnid_col:
    print("Distinct FNIDs:",
          subset[fnid_col].nunique())

    print(subset[fnid_col].unique())

# ====================================================
# CHECK 3
# ====================================================
print("\n==============================")
print("CHECK 3 - TEMPORAL STRUCTURE")
print("==============================")

if fnid_col:
    print("\nObservations per FNID")
    print(df.groupby(fnid_col).size().describe())

if scenario_col:
    print("\nScenario values:")
    print(df[scenario_col].value_counts(dropna=False))

# ====================================================
# CHECK 4
# ====================================================
print("\n==============================")
print("CHECK 4 - DATE RANGE")
print("==============================")

if date_col:
    for country in ["Kenya","Ethiopia","Somalia"]:
        subset = df[df[country_col].str.contains(country,
                                                 case=False,
                                                 na=False)]

        print(f"\n{country}")
        print("Start:", subset[date_col].min())
        print("End  :", subset[date_col].max())

# ====================================================
# CHECK 5
# ====================================================
print("\n==============================")
print("CHECK 5 - SOMALIA MISSINGNESS")
print("==============================")

somalia = df[df[country_col].str.contains("Somalia",
                                          case=False,
                                          na=False)].copy()

if date_col:
    somalia["year"] = somalia[date_col].dt.year

if phase_col:
    missing = (
        somalia
        .groupby([fnid_col,"year"])[phase_col]
        .apply(lambda x: x.isna().sum())
        .reset_index(name="missing_values")
    )

    print(missing)

    print("\nSummary")
    print(missing["missing_values"].describe())
