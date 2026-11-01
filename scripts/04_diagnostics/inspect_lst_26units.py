"""
======================================================================
INSPECT LST EMPTY-BASELINE UNITS
======================================================================

Purpose
-------
Investigate reporting units that have ZERO valid years in the
LST climatology.

Checks:
1. Which FNIDs are affected?
2. Which reporting units do they correspond to?
3. Which hierarchy fields are available?
4. Do these units ever experience IPC Phase 3+?
5. Are they likely to matter for modelling?

Author
------
Robert Odingo MSc Thesis
======================================================================
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

clim = pd.read_csv(ROOT / "data/climatology/kenya_lst_climatology.csv")
units = pd.read_csv(ROOT / "data/interim/kenya_fews_reporting_units.csv")
crosswalk = pd.read_csv(
    ROOT / "data/crosswalks/kenya_fnid_admin_code_crosswalk.csv"
)
panel = pd.read_csv(ROOT / "data/processed/kenya_model_panel.csv")

print("=" * 70)
print("LST EMPTY-BASELINE DIAGNOSTIC")
print("=" * 70)

# ---------------------------------------------------------------------
# Units with at least one climatology cell having zero valid years
# ---------------------------------------------------------------------

empty_units = (
    clim.loc[clim["n_years"] == 0, "FNID"]
    .drop_duplicates()
    .sort_values()
)

print(f"\nReporting units with >=1 empty climatology month: {len(empty_units)}")

# ---------------------------------------------------------------------
# Join FNID -> admin_code -> reporting-unit lookup
# ---------------------------------------------------------------------

named = (
    crosswalk[crosswalk["FNID"].isin(empty_units)]
    .merge(units, on="admin_code", how="left")
)

print("\nAvailable columns after merge:")
print(list(named.columns))

# ---------------------------------------------------------------------
# Choose hierarchy columns automatically
# ---------------------------------------------------------------------

preferred = [
    "FNID",
    "admin_code",
    "ADMIN1",
    "ADMIN2",
    "ADMIN3",
    "admin1",
    "admin2",
    "admin3",
    "LZNAME",
    "lzname",
]

display_cols = [c for c in preferred if c in named.columns]

print("\nAffected reporting units")
print("-" * 70)
print(named[display_cols].sort_values("FNID").to_string(index=False))

# ---------------------------------------------------------------------
# Number of empty months per FNID
# ---------------------------------------------------------------------

print("\nEmpty climatology months per FNID")
print("-" * 70)

per_unit = (
    clim[clim["n_years"] == 0]
    .groupby("FNID")
    .size()
    .rename("empty_months")
    .reset_index()
)

print(per_unit.sort_values("empty_months", ascending=False).to_string(index=False))

# ---------------------------------------------------------------------
# Crisis history
# ---------------------------------------------------------------------

codes = crosswalk.loc[
    crosswalk["FNID"].isin(empty_units),
    "admin_code"
]

crisis = (
    panel[panel["canonical_admin_code"].isin(codes)]
    .groupby("canonical_admin_code")
    .agg(
        phase3plus_total=("phase3plus", "sum"),
        observed=("is_observed", "sum")
    )
    .reset_index()
)

print("\nPhase 3+ history")
print("-" * 70)
print(crisis.to_string(index=False))

print("\nSummary")
print("-" * 70)
print(f"Units with any Phase 3+ history : {(crisis['phase3plus_total'] > 0).sum()}")
print(f"Units with NO Phase 3+ history  : {(crisis['phase3plus_total'] == 0).sum()}")

# ---------------------------------------------------------------------
# Save diagnostics
# ---------------------------------------------------------------------

out = ROOT / "outputs/diagnostics"

out.mkdir(parents=True, exist_ok=True)

named.to_csv(out / "lst_empty_units.csv", index=False)
per_unit.to_csv(out / "lst_empty_month_counts.csv", index=False)
crisis.to_csv(out / "lst_empty_units_phase3_history.csv", index=False)

print("\nOutputs written")
print("-" * 70)
print(out / "lst_empty_units.csv")
print(out / "lst_empty_month_counts.csv")
print(out / "lst_empty_units_phase3_history.csv")
print("\nDiagnostic complete.")