"""
======================================================================

standardize_fews_reporting_units.py

Purpose
-------
Construct a standardized FEWS reporting-unit lookup table for the
configured country. This lookup is the geographic backbone of the
FNID <-> admin_code crosswalk and of the later label join.

Workflow
--------
1. Load the World Bank harmonized FEWS dataset.
2. Extract the configured country's rows.
3. Load and canonicalize the FEWS NET FSC reference geography.
4. Parse admin_name into hierarchy fields (country-specific formats,
   with vocabulary-based disambiguation where formats are ambiguous).
5. Deduplicate unit-months down to unique reporting units.
6. Resolve reporting units carrying multiple admin_codes.
7. Compare against the FSC reference and report discrepancies.
8. Validate the unit count and write the lookup table.

Design notes
------------
Deduplication is on the JOIN KEY alone, not on (admin_code + join key).
A reporting unit is defined by its geography. Including admin_code in
the dedup subset would let a single unit survive as several rows if the
harmonized dataset assigned it more than one code - which is exactly
what happens for one Kenya unit.

Where a unit does carry several admin_codes, the LOWEST is adopted as
the canonical identifier and an ALIAS TABLE is written recording every
code that maps to it. The alias table matters: label resolution is a
downstream concern (Milestone 7.5), and silently discarding the extra
codes here would drop those label rows without trace. The alias table
carries the ambiguity forward so it can be resolved deliberately at
join time rather than lost.

Known Kenya anomalies (Feb-2026 snapshot), both documented findings:
  - turkana/turkana central/kerio delta/northwestern pastoral zone
    carries admin_codes 2995 and 2996. Both span the full series with
    identical admin_name; IPC values disagree in 3 of 55 periods, two
    of which straddle the Phase 3+ threshold. Resolution deferred to
    label join via the alias table.
  - mombasa/mvita/coastal medium potential farming zone exists in the
    FSC geography but has no rows in the harmonized dataset. It is
    reported as FSC-only and excluded: a unit with features but no
    labels cannot contribute to training or evaluation.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from canonicalize import (
    canonicalize_hierarchy,
    normalize_admin_names,
    assert_canonical,
)

from crosswalk_config import get_country_config
from parse_admin_name import parse_fews_admin_name


# =====================================================================
# PHASE 1 - CONFIGURATION
# =====================================================================

COUNTRY = "Somalia"  # Change this to the country you want to process

config = get_country_config(COUNTRY)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / f"{COUNTRY.lower()}_fews_reporting_units.csv"
ALIAS_FILE = OUTPUT_DIR / f"{COUNTRY.lower()}_admin_code_aliases.csv"

DIAGNOSTIC_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

FSC_ONLY_FILE = DIAGNOSTIC_DIR / f"{COUNTRY.lower()}_fsc_only_units.csv"


# =====================================================================
# Helper functions
# =====================================================================

def section(title: str) -> None:
    """Print a section heading."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> None:
    """Ensure a dataset contains every required column."""

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{dataset_name} missing required columns: {missing}"
        )


def prepare_fsc_reference(
    fsc: gpd.GeoDataFrame,
    config: dict,
) -> gpd.GeoDataFrame:
    """Canonicalize and validate the FSC reference geography."""

    fsc = canonicalize_hierarchy(fsc, config["join_key"])
    fsc = normalize_admin_names(fsc)
    assert_canonical(fsc)

    return fsc


# =====================================================================
# PHASE 1 - LOAD FEWS DATASET
# =====================================================================

section(f"STANDARDIZE {COUNTRY.upper()} FEWS REPORTING UNITS")

print("Loading FEWS dataset...")
print(f"Source:\n{INPUT_FILE}")

fews = pd.read_csv(INPUT_FILE)

print(f"\nTotal FEWS rows : {len(fews):,}")

require_columns(
    fews,
    ["country", "admin_code", "admin_name"],
    "FEWS dataset",
)

print("\u2713 Required FEWS columns verified.")


# ---------------------------------------------------------------------
# Extract configured country
# ---------------------------------------------------------------------

print(f"\nExtracting {COUNTRY} reporting units...")

fews = (
    fews.loc[
        fews["country"].str.lower() == COUNTRY.lower(),
        ["admin_code", "admin_name"],
    ]
    .copy()
)

print(f"{COUNTRY} reporting rows : {len(fews):,}")

if fews.empty:
    raise RuntimeError(
        f"No rows found for country '{COUNTRY}'. "
        f"Check the country spelling against the dataset."
    )


# =====================================================================
# PHASE 1 - LOAD FSC REFERENCE GEOGRAPHY
# =====================================================================

section(f"LOAD {COUNTRY.upper()} FSC GEOGRAPHY")

fsc_file = PROJECT_ROOT.joinpath(*config["fsc_relative_path"])

print("Loading FSC reference geography...")
print(f"Source:\n{fsc_file}")

fsc = gpd.read_file(fsc_file)

print(f"\nFSC polygons : {len(fsc):,}")

require_columns(fsc, config["fsc_columns"], "FSC reference geography")

print("\u2713 Required FSC columns verified.")


section("PREPARE FSC REFERENCE")

fsc = prepare_fsc_reference(fsc, config)

print("\u2713 FSC reference prepared.")


# =====================================================================
# PHASE 2 - PARSE AND CANONICALIZE
# =====================================================================

section("PARSE FEWS REPORTING UNITS")

# The parser needs the canonicalized FSC frame: for countries whose
# admin_name formats are positionally ambiguous (Kenya, Somalia), it
# disambiguates by testing field content against FSC vocabularies.


fews = parse_fews_admin_name(
    fews=fews,
    country_config=config,
    fsc_canonical=fsc,
)

print("\u2713 Reporting units parsed.")


section("CANONICALIZE PARSED FEWS")

fews = normalize_admin_names(fews)
assert_canonical(fews)

print("\u2713 Parsed hierarchy canonicalized.")


# =====================================================================
# PHASE 3 - DEDUPLICATE TO REPORTING UNITS
# =====================================================================

section("DEDUPLICATE REPORTING UNITS")

join_columns = config["join_key"]

print(f"Unit-month rows          : {len(fews):,}")

# --- identify units carrying more than one admin_code ---------------

alias_map = (
    fews.groupby(join_columns)["admin_code"]
        .apply(lambda s: sorted(pd.unique(s)))
        .reset_index(name="admin_codes")
)

alias_map["n_codes"] = alias_map["admin_codes"].apply(len)
alias_map["canonical_admin_code"] = alias_map["admin_codes"].apply(min)

multi_code = alias_map[alias_map["n_codes"] > 1]

if multi_code.empty:
    print("\u2713 Every reporting unit has exactly one admin_code.")
else:
    print(
        f"\u26a0 {len(multi_code)} reporting unit(s) carry multiple "
        f"admin_codes:"
    )
    print(
        multi_code[[*join_columns, "admin_codes", "canonical_admin_code"]]
        .to_string(index=False)
    )
    print(
        "\n  The lowest code is adopted as canonical. All codes are\n"
        "  recorded in the alias table so that label resolution can be\n"
        "  handled deliberately at join time (see module docstring)."
    )

# --- deduplicate on the JOIN KEY alone ------------------------------
#
# A reporting unit is defined by its geography. Deduplicating on
# (admin_code + join key) would preserve one row per code and inflate
# the unit count.

reporting_units = (
    fews.sort_values(["admin_code", *join_columns])
        .drop_duplicates(subset=join_columns, keep="first")
        .sort_values(join_columns)
        .reset_index(drop=True)
)

print(f"\nReporting units discovered : {len(reporting_units):,}")

# --- invariant: one row per join key --------------------------------

if reporting_units.duplicated(subset=join_columns).any():
    raise RuntimeError(
        "Deduplication failed: duplicate join keys remain."
    )

print("\u2713 One row per reporting unit confirmed.")


# =====================================================================
# PHASE 3 - COMPARE AGAINST FSC REFERENCE
# =====================================================================

section("COMPARE WITH FSC REFERENCE")

fews_keys = set(map(tuple, reporting_units[join_columns].fillna("").values))
fsc_keys = set(map(tuple, fsc[join_columns].fillna("").values))

only_in_fews = fews_keys - fsc_keys
only_in_fsc = fsc_keys - fews_keys
matched = fews_keys & fsc_keys

print(f"Reporting-unit keys (FEWS) : {len(fews_keys):,}")
print(f"Reporting-unit keys (FSC)  : {len(fsc_keys):,}")
print(f"\nMatched      : {len(matched):,}")
print(f"Only in FEWS : {len(only_in_fews)}")
print(f"Only in FSC  : {len(only_in_fsc)}")

match_rate = len(matched) / len(fews_keys) * 100 if fews_keys else 0
print(f"Match rate   : {match_rate:.2f}% of FEWS units")

# --- FEWS-only units are a hard failure ------------------------------
#
# A FEWS unit with no FSC polygon cannot be given satellite features.

if only_in_fews:
    print("\nReporting units found only in FEWS")
    print("----------------------------------")
    print(
        reporting_units[
            reporting_units[join_columns].apply(tuple, axis=1).isin(only_in_fews)
        ][["admin_code", "admin_name", *join_columns]]
        .sort_values(join_columns)
        .to_string(index=False)
    )
    raise RuntimeError(
        f"{len(only_in_fews)} FEWS reporting unit(s) have no matching FSC "
        f"polygon. These units cannot be assigned satellite features. "
        f"Investigate before proceeding."
    )

# --- FSC-only polygons are excluded and logged -----------------------
#
# An FSC polygon with no FEWS rows has no labels, so it cannot be used
# for training or evaluation. Excluded, but recorded.

if only_in_fsc:
    fsc_only = (
        fsc[fsc[join_columns].apply(tuple, axis=1).isin(only_in_fsc)]
        [[c for c in fsc.columns if c in ["FNID", *join_columns]]]
        .sort_values(join_columns)
    )

    print("\nFSC polygons with no FEWS labels (EXCLUDED)")
    print("-------------------------------------------")
    print(fsc_only.to_string(index=False))
    print(
        "\n  These polygons exist in the reference geography but have no\n"
        "  rows in the harmonized dataset, so no IPC labels exist for\n"
        "  them. They are excluded from the lookup and logged."
    )

    fsc_only.to_csv(FSC_ONLY_FILE, index=False)
    print(f"\n  Logged to: {FSC_ONLY_FILE}")
else:
    print("\n\u2713 Every FSC polygon has corresponding FEWS rows.")


# =====================================================================
# PHASE 3 - VALIDATE UNIT COUNT
# =====================================================================

section("VALIDATE REPORTING UNIT COUNT")

expected_units = config["expected_units"]

print(f"Expected reporting units   : {expected_units:,}")
print(f"Observed reporting units   : {len(reporting_units):,}")

if len(reporting_units) != expected_units:
    raise RuntimeError(
        f"Expected {expected_units:,} reporting units but found "
        f"{len(reporting_units):,}. If this change is intentional "
        f"(e.g. a dataset update), update 'expected_units' in "
        f"crosswalk_config.py and record the reason."
    )

print("\u2713 Reporting-unit count validated.")


# =====================================================================
# PHASE 3 - WRITE OUTPUTS
# =====================================================================

section("SAVE OUTPUTS")

output_columns = ["admin_code", "admin_name", *join_columns]

reporting_units.loc[:, output_columns].to_csv(OUTPUT_FILE, index=False)
print(f"Reporting-unit lookup : {OUTPUT_FILE}")

# Alias table: every admin_code -> canonical admin_code, one row per code.
alias_rows = alias_map.explode("admin_codes").rename(
    columns={"admin_codes": "admin_code"}
)
alias_rows = alias_rows[
    ["admin_code", "canonical_admin_code", *join_columns, "n_codes"]
].sort_values(["canonical_admin_code", "admin_code"])

alias_rows.to_csv(ALIAS_FILE, index=False)
print(f"admin_code alias table : {ALIAS_FILE}")


# =====================================================================
# SUMMARY
# =====================================================================

section("SUMMARY")

print(f"Country                  : {COUNTRY}")
print(f"Unit-month rows parsed   : {len(fews):,}")
print(f"Reporting units          : {len(reporting_units):,}")
print(f"Matched to FSC           : {len(matched):,} ({match_rate:.2f}%)")
print(f"FSC polygons excluded    : {len(only_in_fsc)}")
print(f"Units with alias codes   : {len(multi_code)}")

print("\n\u2713 Standardization complete.")