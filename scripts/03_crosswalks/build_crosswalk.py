"""
======================================================================

build_crosswalk.py

Purpose
-------
Build and persist the FNID <-> admin_code crosswalk for a country.

This is the join table that links the two identifier systems:
  - FNID      : FEWS NET FSC geometry id (carried by feature panels)
  - admin_code: World Bank harmonized id (carried by label panels)

Neither the feature panels nor the label panels share a direct key;
the crosswalk is the bridge, keyed on the canonical administrative
hierarchy that both sides were standardized to. validate_fsc_mapping.py
proved the hierarchies match; this script persists the pairing as a
first-class artifact so every downstream join routes through one
audited table rather than re-deriving the link.

Output
------
data/crosswalks/<country>_fnid_admin_code_crosswalk.csv
    columns: admin_code, FNID, ADMIN1, ADMIN2, ADMIN3[, LZNAME]

Guarantees
----------
- Every LABELLED reporting unit (admin_code side) is matched to exactly
  one FNID. A labelled unit with no FNID is a hard failure - it could
  not be given features.
- FSC polygons with no label (Kenya: Mvita; Somalia: Zeylac) are
  reported and excluded, consistent with the standardization step.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

import argparse
import sys

from pathlib import Path as _Path

from canonicalize import canonicalize_hierarchy, normalize_admin_names, assert_canonical
from crosswalk_config import get_country_config

# Ensure the project root is importable for the shared country config,
# regardless of how this script is invoked (the sibling imports above
# are path-relative to 03_crosswalks; this one is absolute).
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config.countries import get_country


# Resolve --country up front. get_country() accepts name/code
# (ethiopia/et/...) and returns the canonical capitalized name that
# crosswalk_config keys on ("Ethiopia"). Unknown country fails loud.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
COUNTRY = get_country(_country_arg.country)["name"]

config = get_country_config(COUNTRY)
join_key = config["join_key"]
expected_units = config["expected_units"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNITS_FILE = (
    PROJECT_ROOT / "data" / "interim"
    / f"{COUNTRY.lower()}_fews_reporting_units.csv"
)

FSC_FILE = PROJECT_ROOT.joinpath(*config["fsc_relative_path"])

OUT_DIR = PROJECT_ROOT / "data" / "crosswalks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"{COUNTRY.lower()}_fnid_admin_code_crosswalk.csv"

DIAG_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
FSC_ONLY_FILE = DIAG_DIR / f"{COUNTRY.lower()}_crosswalk_fsc_only.csv"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# =====================================================================
# Load both sides
# =====================================================================

section(f"BUILD {COUNTRY.upper()} FNID <-> admin_code CROSSWALK")

units = pd.read_csv(UNITS_FILE)
print(f"Labelled reporting units : {len(units):,}")
if len(units) != expected_units:
    raise RuntimeError(
        f"Reporting-unit lookup has {len(units):,} rows, expected "
        f"{expected_units:,}."
    )

fsc = gpd.read_file(FSC_FILE)
print(f"FSC polygons             : {len(fsc):,}")


# =====================================================================
# Canonicalize both sides on the join key
# =====================================================================

section("CANONICALIZE")

# The lookup was written already-canonical, but re-canonicalizing is
# idempotent and guards against drift.
for col in join_key:
    if col not in units.columns:
        raise RuntimeError(
            f"Reporting-unit lookup missing join column '{col}'. "
            f"Columns: {list(units.columns)}"
        )
units = canonicalize_hierarchy(units, join_key)
units = normalize_admin_names(units)
assert_canonical(units)

fsc = fsc[[c for c in config["fsc_columns"]]].copy()
for raw, canon in config.get("fsc_rename", {}).items():
    fsc = fsc.rename(columns={raw: canon})
fsc = canonicalize_hierarchy(fsc, join_key)
fsc = normalize_admin_names(fsc)
assert_canonical(fsc)

print("\u2713 Both sides canonicalized.")


# =====================================================================
# Merge on the hierarchy
# =====================================================================

section("MERGE")

merged = units.merge(
    fsc[["FNID", *join_key]],
    on=join_key,
    how="outer",
    indicator=True,
    validate="one_to_one",
)

both = merged[merged["_merge"] == "both"]
label_only = merged[merged["_merge"] == "left_only"]     # labelled, no FNID
fsc_only = merged[merged["_merge"] == "right_only"]       # FNID, no label

print(f"Matched                  : {len(both):,}")
print(f"Labelled w/o FNID        : {len(label_only)}")
print(f"FSC w/o label            : {len(fsc_only)}")

# Labelled units MUST all match - they need features.
if len(label_only):
    print("\nLabelled units with no FSC polygon:")
    print(label_only[["admin_code", *join_key]].to_string(index=False))
    raise RuntimeError(
        f"{len(label_only)} labelled units have no FNID. They cannot be "
        f"given features. Investigate hierarchy mismatch."
    )

# FSC-only polygons are expected (Mvita, Zeylac) - log and exclude.
if len(fsc_only):
    fsc_only[["FNID", *join_key]].to_csv(FSC_ONLY_FILE, index=False)
    print(f"\nFSC-only polygons (excluded) logged to:\n{FSC_ONLY_FILE}")


# =====================================================================
# Validate and write
# =====================================================================

section("VALIDATE")

crosswalk = both[["admin_code", "FNID", *join_key]].copy()

# Every admin_code -> exactly one FNID and vice versa.
if crosswalk["admin_code"].duplicated().any():
    raise RuntimeError("An admin_code maps to multiple FNIDs.")
if crosswalk["FNID"].duplicated().any():
    raise RuntimeError("An FNID maps to multiple admin_codes.")

if len(crosswalk) != expected_units:
    raise RuntimeError(
        f"Crosswalk has {len(crosswalk):,} rows, expected "
        f"{expected_units:,}."
    )

print(f"Crosswalk rows           : {len(crosswalk):,}")
print(f"Unique admin_codes       : {crosswalk['admin_code'].nunique():,}")
print(f"Unique FNIDs             : {crosswalk['FNID'].nunique():,}")
print("\u2713 One-to-one crosswalk validated.")

crosswalk.to_csv(OUT_FILE, index=False)

section("DONE")
print(f"Crosswalk : {OUT_FILE}")
