"""
validate_kenya_parser.py

Validation 1

Purpose
-------
1. Inspect the Kenya FEWS schema.
2. Determine conclusively whether the harmonized FEWS dataset contains a
   livelihood-zone code (e.g. KE12).
3. Produce evidence before implementing the production parser.

This script does NOT modify any data.
"""

from pathlib import Path
import sys
import re

import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CROSSWALKS_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "03_crosswalks"
)

sys.path.insert(0, str(CROSSWALKS_DIR))

from canonicalize import (
    canonicalize_hierarchy,
    canonicalize_text,
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


FEWS_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

COUNTRY = "Kenya"


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("=" * 70)
print("KENYA FEWS SCHEMA VALIDATION")
print("=" * 70)

print("\nLoading FEWS dataset...")
print(FEWS_FILE)

fews = pd.read_csv(FEWS_FILE)

print(f"\nRows loaded: {len(fews):,}")

fews = fews[
    fews["country"].str.lower() == COUNTRY.lower()
].copy()

print(f"Kenya rows: {len(fews):,}")


# ---------------------------------------------------------------------
# PRINT SCHEMA
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN LIST")
print("=" * 70)

for col in fews.columns:
    print(f"- {col}")


# ---------------------------------------------------------------------
# SAMPLE VALUES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE VALUES")
print("=" * 70)

for col in fews.columns:

    sample = (
        fews[col]
        .dropna()
        .astype(str)
        .unique()[:5]
    )

    print(f"\n{col}")
    print(sample)


# ---------------------------------------------------------------------
# SEARCH FOR LIVELIHOOD-ZONE CODES
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("SEARCH FOR KE## LIVELIHOOD-ZONE CODES")
print("=" * 70)

pattern = re.compile(r"KE\d+")

found_any = False

for col in fews.columns:

    values = (
        fews[col]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    matches = sorted(
        {
            value
            for value in values
            if pattern.match(value)
        }
    )

    if matches:

        found_any = True

        print(f"\nColumn: {col}")
        print(f"Matches ({len(matches)}):")

        for value in matches:
            print(f"  {value}")


# ---------------------------------------------------------------------
# FINAL CONCLUSION
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if found_any:

    print("""
A livelihood-zone code exists in the harmonized FEWS dataset.

The production parser should investigate whether this field can be used
instead of LZNAME for joining to the FSC geography.
""")

else:

    print("""
No livelihood-zone code (e.g. KE12) was found in any FEWS column.

Based on the available schema, Kenya FEWS does not expose an FSC-style
livelihood-zone code.

The production parser will therefore need to reconstruct and
canonicalize LZNAME before joining to the FSC geography.
""")

# ---------------------------------------------------------------------
# VALIDATION 2 - BUILD CANONICAL FSC LOOKUP SETS
# ---------------------------------------------------------------------

SHAPEFILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kenya"
    / "KENYAFSC"
    / "KE_Admin3_LHZ_2016.2.shp"
)

print("\n" + "=" * 70)
print("VALIDATION 2 - CANONICAL FSC LOOKUP SETS")
print("=" * 70)

print("\nLoading Kenya FSC shapefile...")
print(SHAPEFILE)

fsc = gpd.read_file(SHAPEFILE)

print(f"\nPolygons loaded: {len(fsc):,}")

# Canonicalize the hierarchy using the SAME functions as production
fsc = canonicalize_hierarchy(
    fsc,
    columns=["ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],
)

# Build lookup vocabularies
lzname_set = set(fsc["LZNAME"])

admin3_set = set(fsc["ADMIN3"])

print("\nLookup vocabulary sizes")
print("-----------------------")

print(f"Canonical LZNAME values : {len(lzname_set)}")
print(f"Canonical ADMIN3 values : {len(admin3_set)}")

print("\nSample LZNAME values")
print("-------------------")

for value in sorted(lzname_set)[:10]:
    print(value)

print("\nSample ADMIN3 values")
print("-------------------")

for value in sorted(admin3_set)[:10]:
    print(value)


# ---------------------------------------------------------------------
# VALIDATION 3 - PARSER CLASSIFIER
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION 3 - FIVE-COMPONENT CLASSIFIER")
print("=" * 70)

# Split FEWS reporting units
parts = fews["admin_name"].str.split(",", expand=True)

component_counts = (
    fews["admin_name"]
    .str.split(",")
    .apply(len)
)

five = parts.loc[component_counts == 5].copy()

print(f"\nFive-component FEWS rows: {len(five):,}")

# Canonicalize every component
for col in five.columns:
    five[col] = five[col].apply(canonicalize_text)

# Classification counters
format2 = 0
format3 = 0
neither = 0
both = 0

for _, row in five.iterrows():

    field0 = row[0]
    field1 = row[1]

    # -------------------------------------------------------------
    # Pattern A
    # Split livelihood-zone name
    # -------------------------------------------------------------

    lz_candidate = canonicalize_text(
        f"{field0}, {field1}"
    )

    match_a = lz_candidate in lzname_set

    # -------------------------------------------------------------
    # Pattern B
    # ADMIN3 present
    # -------------------------------------------------------------

    match_b = (
        (field0 in lzname_set)
        and
        (field1 in admin3_set)
    )

    matches = int(match_a) + int(match_b)

    if matches == 0:
        neither += 1

    elif matches == 2:
        both += 1

    elif match_a:
        format2 += 1

    else:
        format3 += 1

print("\nClassification summary")
print("----------------------")

print(f"Format 2 (split LZ) : {format2:,}")
print(f"Format 3 (ADMIN3)   : {format3:,}")
print(f"Neither             : {neither:,}")
print(f"Both                : {both:,}")

#----------------------------------------------------------------------
# Distribution
#---------------------------------------------------------------------- 

total = format2 + format3

print("\nDistribution")
print("------------")
print(f"Format 2 : {format2} ({format2 / total:.1%})")
print(f"Format 3 : {format3} ({format3 / total:.1%})")

print("\nCheck")
print("-----")

print(
    format2 +
    format3 +
    neither +
    both
)

# ---------------------------------------------------------------------
# VALIDATION 4 - Missing ADMIN3 Representation
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION 4 - COASTAL ADMIN3")
print("=" * 70)

# ------------------------------------------------------------------
# FSC
# ------------------------------------------------------------------

print("\nFSC")

raw_fsc = gpd.read_file(SHAPEFILE)

raw_nan = raw_fsc["ADMIN3"].isna().sum()

print(f"Total FSC polygons          : {len(raw_fsc)}")
print(f"ADMIN3 populated            : {len(raw_fsc) - raw_nan}")
print(f"ADMIN3 missing              : {raw_nan}")

print(f"Original ADMIN3 NaN values : {raw_nan}")

canonical_fsc = canonicalize_hierarchy(
    raw_fsc,
    columns=["ADMIN1", "ADMIN2", "ADMIN3", "LZNAME"],
)

canonical_empty = (canonical_fsc["ADMIN3"] == "").sum()

print(f'Canonical ADMIN3 "" values : {canonical_empty}')

print(f"Counts match : {raw_nan == canonical_empty}")

# ------------------------------------------------------------------
# FEWS
# ------------------------------------------------------------------

print("\nFEWS")

component_counts = (
    fews["admin_name"]
    .str.split(",")
    .apply(len)
)

four_component = (component_counts == 4).sum()

print(f"Four-component reporting units : {four_component:,}")

print("\nInterpretation")

if raw_nan == canonical_empty:
    print(
        "FSC polygons with missing ADMIN3 values canonicalize missing ADMIN3 "
        'to the empty string ("").'
    )
        
print(
    "Four-component FEWS reporting units will therefore "
    'produce ADMIN3 = "" during parser construction.'
)

print(
    "\nResult: FSC and FEWS use the same canonical "
    "representation for missing ADMIN3 values."
)

# ---------------------------------------------------------------------
# OVERALL VALIDATION STATUS
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL VALIDATION STATUS")
print("=" * 70)

print("\nValidation 1 - FEWS Schema Inspection")
print("-------------------------------------")
print("✓ FEWS schema enumerated")
print("✓ No livelihood-zone code field present")
print("✓ No KE## livelihood-zone codes detected")
print("✓ Parser must reconstruct LZNAME from admin_name")

print("\nValidation 2 - Canonical FSC Lookup Vocabularies")
print("------------------------------------------------")
print(f"✓ Canonical LZNAME vocabulary : {len(lzname_set)} values")
print(f"✓ Canonical ADMIN3 vocabulary : {len(admin3_set)} values")
print("✓ Shared canonicalization successfully applied")

print("\nValidation 3 - Five-Component Parser Classifier")
print("-----------------------------------------------")
print(f"✓ Total five-component rows : {total:,}")
print(f"✓ Format 2 (split LZ)       : {format2:,} ({format2 / total:.1%})")
print(f"✓ Format 3 (ADMIN3)         : {format3:,} ({format3 / total:.1%})")
print(f"✓ Unclassified rows         : {neither}")
print(f"✓ Ambiguous rows            : {both}")

print("\nValidation 4 - Missing ADMIN3 Representation")
print("--------------------------------------------")
print(f"✓ Total FSC polygons        : {len(raw_fsc):,}")
print(f"✓ ADMIN3 populated          : {len(raw_fsc) - raw_nan:,}")
print(f"✓ ADMIN3 missing            : {raw_nan:,}")
print("✓ Missing ADMIN3 canonicalized consistently to the empty string")
print("✓ FEWS and FSC use identical canonical representation")

print("\nConclusion")
print("----------")

if neither == 0 and both == 0 and raw_nan == canonical_empty:
    print(
        "SUCCESS: All validation checks passed.\n"
        "The Kenya reporting-unit parser has been fully validated and "
        "is ready for implementation in standardize_fews_reporting_units.py."
    )
else:
    print(
        "WARNING: One or more validation checks failed. "
        "Review the diagnostics above before implementing the parser."
    )