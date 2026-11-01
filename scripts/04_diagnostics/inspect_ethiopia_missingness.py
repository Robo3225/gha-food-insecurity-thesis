from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

panel = pd.read_csv(
    PROJECT_ROOT /
    "data/processed/ethiopia_label_panel.csv"
)

print("=" * 70)
print("ETHIOPIA LABEL MISSINGNESS")
print("=" * 70)

# ----------------------------------------------------------
# Missingness by reporting unit
# ----------------------------------------------------------

unit_missing = (
    panel.groupby("canonical_admin_code")["ipc_filled"]
         .apply(lambda s: s.isna().mean())
)

print("\nReporting units")
print("--------------------------")
print(f"Total units             : {len(unit_missing):,}")
print(f"> 25% missing           : {(unit_missing > 0.25).sum()}")
print(f"> 50% missing           : {(unit_missing > 0.50).sum()}")
print(f"> 75% missing           : {(unit_missing > 0.75).sum()}")
print(f"100% missing           : {(unit_missing == 1.0).sum()}")

print("\nWorst 10 reporting units")
print("--------------------------")
print(
    unit_missing.sort_values(ascending=False)
                .head(10)
)

# ----------------------------------------------------------
# Missingness through time
# ----------------------------------------------------------

missing_by_month = (
    panel[panel["ipc_filled"].isna()]
    .groupby("year_month")
    .size()
    .sort_values(ascending=False)
)

print("\nMonths with most missing labels")
print("--------------------------")
print(missing_by_month.head(15))

# ----------------------------------------------------------
# Overall
# ----------------------------------------------------------

print("\nOverall")
print("--------------------------")

overall = panel["ipc_filled"].isna().mean()

print(f"Overall missing rate    : {overall:.2%}")