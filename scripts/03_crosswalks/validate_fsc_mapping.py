"""
=====================================================================

validate_fsc_mapping.py

Purpose
-------
Validate that the FEWS reporting units and the configured country's FSC
reporting units are structurally compatible before building the
production FNID <-> admin_code crosswalk.

This script performs:

- Load standardized FEWS reporting units
- Load configured country's FSC reporting units
- Validate schemas
- Validate uniqueness
- Validate administrative hierarchies
- Perform exact one-to-one hierarchy validation

Canonicalization note
---------------------
BOTH sides are canonicalized through the shared canonicalize module.
The FEWS side is already canonical (it comes from the standardizer),
but re-canonicalizing here is idempotent and makes this validator
robust to any upstream regression rather than silently propagating
it. Canonicalizing only one side is exactly the failure mode that
produces a silent 0% match.

Author:
Robert Odingo MSc Thesis

=====================================================================
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from canonicalize import (
    canonicalize_hierarchy,
    assert_canonical,
    HIERARCHY,
)

# =====================================================================
# Configuration
# =====================================================================

COUNTRY = "Kenya"

PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT.name != "gee_project":
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate project root directory.")
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABULAR = (
    DATA_DIR
    / "interim"
    / "kenya_fews_reporting_units.csv"
)

SHAPEFILE = (
    DATA_DIR
    / "raw"
    / "kenya"
    / "KENYAFSC"
    / "KE_Admin3_LHZ_2016.2.shp"
)

# =====================================================================
# Helper Functions
# =====================================================================

def section(title: str) -> None:
    """Print a formatted section heading."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def validate_columns(df, required_columns, dataset_name):
    """
    Ensure a dataset contains all required columns.
    """

    missing = sorted(
        set(required_columns) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing}"
        )

# =====================================================================
# Load datasets
# =====================================================================

section(f"{COUNTRY.upper()} FSC VALIDATION")

print("Loading FEWS dataset...")

fews = pd.read_csv(TABULAR)

print(f"Loading {COUNTRY} FSC reporting units...")

fsc = gpd.read_file(SHAPEFILE)
# Keep only the columns needed for validation
fsc = fsc[
    [
        "FNID",
        "ADMIN1",
        "ADMIN2",
        "ADMIN3",
    ]
].copy()

# =====================================================================
# Validate schemas
# =====================================================================

validate_columns(
    fews,
    [
        "admin_code",
        "ADMIN1",
        "ADMIN2",
        "ADMIN3",
    ],
    "Standardized FEWS reporting units",
)

validate_columns(
    fsc,
    [
        "FNID",
        "ADMIN1",
        "ADMIN2",
        "ADMIN3",
    ],
    "FSC shapefile",
)

print("\u2713 Required columns verified.")

print(f"\nFEWS reporting units : {len(fews)}")
print(f"FSC reporting units  : {len(fsc)}")

# Administrative hierarchy columns
hierarchy = HIERARCHY

# ---------------------------------------------------------------------
# Canonicalize BOTH sides through the shared module.
#
# The FEWS side arrives already canonical from the standardizer;
# re-canonicalizing is idempotent and guards against an upstream
# regression. The FSC side is canonicalized from raw shapefile
# attributes here. Both must pass the canonical-form invariant
# before any merge is attempted.
# ---------------------------------------------------------------------

fsc = canonicalize_hierarchy(fsc)
fews = canonicalize_hierarchy(fews)

assert_canonical(fsc)
assert_canonical(fews)

print("\u2713 Both hierarchies canonicalized.")

print("\nFirst 10 FEWS hierarchies:")
print(fews[hierarchy].head(10))

print("\nFirst 10 FSC hierarchies:")
print(fsc[hierarchy].head(10))

print("\u2713 Dataset preparation complete.")

# =====================================================================
# Validate duplicate identifiers
# =====================================================================

section("VALIDATING IDENTIFIERS")

duplicate_admin_codes = fews[
    fews["admin_code"].duplicated(keep=False)
].sort_values("admin_code")

if duplicate_admin_codes.empty:
    print("\u2713 No duplicate admin_codes found.")
else:
    print(
        f"\u2717 Found {duplicate_admin_codes['admin_code'].nunique()} duplicate admin_codes."
    )

    print("\nDuplicate admin_codes:")

    print(
        duplicate_admin_codes[
            [
            "admin_code",
            "ADMIN1",
            "ADMIN2",
            "ADMIN3",
        ]
    ]
)

duplicate_fnids = fsc[
    fsc["FNID"].duplicated(keep=False)
].sort_values("FNID")

if duplicate_fnids.empty:
    print("\u2713 No duplicate FNIDs found.")
else:
    print(
        f"\u2717 Found {duplicate_fnids['FNID'].nunique()} duplicate FNIDs."
    )

    print("\nDuplicate FNIDs:")
    print(
        duplicate_fnids[
            [
                "FNID",
                "ADMIN1",
                "ADMIN2",
                "ADMIN3",
            ]
        ]
    )
# =====================================================================
# Validate duplicate administrative hierarchies
# =====================================================================

section("VALIDATING ADMINISTRATIVE HIERARCHY")

duplicate_fews = fews[
    fews.duplicated(subset=hierarchy, keep=False)
].sort_values(hierarchy)

if duplicate_fews.empty:
    print("\u2713 FEWS hierarchy is unique.")
else:
    print(
        f"\u2717 Found {len(duplicate_fews)} duplicated FEWS hierarchy records."
    )

    print("\nDuplicate FEWS hierarchy:")
    print(
        duplicate_fews[
            [
                "admin_code",
                "admin_name",
                "ADMIN1",
                "ADMIN2",
                "ADMIN3",
            ]
        ]
    )

duplicate_fsc = fsc[
    fsc.duplicated(subset=hierarchy, keep=False)
].sort_values(hierarchy)

if duplicate_fsc.empty:
    print("\u2713 FSC hierarchy is unique.")
else:
    print(
        f"\u2717 Found {len(duplicate_fsc)} duplicated FSC hierarchy records."
    )

    print("\nDuplicate FSC hierarchy:")
    print(
        duplicate_fsc[
            [
                "FNID",
                "ADMIN1",
                "ADMIN2",
                "ADMIN3",
            ]
        ]
    )
# =====================================================================
# Perform exact hierarchy matching
# =====================================================================

section("MERGE READINESS")

print(
    f"FEWS unique hierarchies : "
    f"{len(fews[hierarchy].drop_duplicates())}"
)

print(
    f"FSC unique hierarchies  : "
    f"{len(fsc[hierarchy].drop_duplicates())}"
)

section("EXACT HIERARCHY MATCHING")

comparison = fews.merge(
    fsc[
        [
            "FNID",
            "ADMIN1",
            "ADMIN2",
            "ADMIN3",
        ]
    ],
    on=hierarchy,
    how="outer",
    indicator=True,
    validate="one_to_one",
)

matched = comparison[
    comparison["_merge"] == "both"
].copy()

fews_only = comparison[
    comparison["_merge"] == "left_only"
].copy()

fsc_only = comparison[
    comparison["_merge"] == "right_only"
].copy()

# =====================================================================
# Validation summary
# =====================================================================

section("VALIDATION SUMMARY")

total_fews = len(fews)
total_fsc = len(fsc)

matched_count = len(matched)

success_rate = (
    matched_count / total_fews * 100
    if total_fews > 0
    else 0
)

print(f"FEWS reporting units : {total_fews}")
print(f"FSC reporting units  : {total_fsc}")
print()

print(f"Exact matches        : {matched_count}")
print(f"FEWS only            : {len(fews_only)}")
print(f"FSC only             : {len(fsc_only)}")
print(f"Success rate         : {success_rate:.2f}%")

if success_rate >= 95:
    print("\n\u2713 Excellent structural agreement.")
elif success_rate >= 90:
    print("\n\u2713 Good agreement. Investigate remaining unmatched records.")
else:
    print("\n\u26a0 Low agreement. Investigate hierarchy before building the crosswalk.")

# =====================================================================
# Save validation diagnostics
# =====================================================================

section("SAVING DIAGNOSTICS")

matched_file = OUTPUT_DIR / f"{COUNTRY.lower()}_matched.csv"
fews_only_file = OUTPUT_DIR / f"{COUNTRY.lower()}_fews_only.csv"
fsc_only_file = OUTPUT_DIR / f"{COUNTRY.lower()}_fsc_only.csv"
summary_file = OUTPUT_DIR / f"{COUNTRY.lower()}_validation_summary.txt"

matched.to_csv(
    matched_file,
    index=False,
)

fews_only.to_csv(
    fews_only_file,
    index=False,
)

fsc_only.to_csv(
    fsc_only_file,
    index=False,
)

# =====================================================================
# Write validation summary
# =====================================================================

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("=" * 70 + "\n")
    file.write(f"{COUNTRY.upper()} FSC VALIDATION SUMMARY\n")
    file.write("=" * 70 + "\n\n")

    file.write(f"Country: {COUNTRY}\n\n")

    file.write(f"FEWS reporting units : {total_fews}\n")
    file.write(f"FSC reporting units  : {total_fsc}\n\n")

    file.write(f"Exact matches        : {matched_count}\n")
    file.write(f"FEWS only            : {len(fews_only)}\n")
    file.write(f"FSC only             : {len(fsc_only)}\n")
    file.write(f"Success rate         : {success_rate:.2f}%\n\n")

    file.write("Duplicate Checks\n")
    file.write("------------------------------\n")
    file.write(
        f"Duplicate admin_codes : "
        f"{duplicate_admin_codes['admin_code'].nunique()}\n"
    )
    file.write(
        f"Duplicate FNIDs       : "
        f"{duplicate_fnids['FNID'].nunique()}\n"
    )
    file.write(
        f"Duplicate FEWS hierarchy : "
        f"{len(duplicate_fews)}\n"
    )
    file.write(
        f"Duplicate FSC hierarchy  : "
        f"{len(duplicate_fsc)}\n"
    )

print("\u2713 Matched records saved.")
print("\u2713 FEWS-only records saved.")
print("\u2713 FSC-only records saved.")
print("\u2713 Validation summary saved.")

# =====================================================================
# Finished
# =====================================================================

section(f"{COUNTRY.upper()} VALIDATION COMPLETE")

print("Validation finished successfully.")

print(f"\nMatched records : {matched_file.name}")
print(f"FEWS only       : {fews_only_file.name}")
print(f"FSC only        : {fsc_only_file.name}")
print(f"Summary         : {summary_file.name}")

print("\nThe datasets are now ready for crosswalk construction.")
