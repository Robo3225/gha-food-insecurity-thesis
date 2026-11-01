"""
======================================================================
Inspect Adjusted FEWS Target Columns

Purpose
-------
Compare fews_ipc against fews_ipc_adjusted before selecting the target
variable for modelling.

Questions answered
------------------
1. Does the adjusted column differ from the raw column?
2. Does it improve coverage?
3. Does it change the IPC Phase 3+ binary target?
4. Where do differences occur?

Author:
Robert Odingo MSc Thesis
======================================================================
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

print("=" * 70)
print("INSPECT ADJUSTED FEWS TARGET COLUMNS")
print("=" * 70)
print(f"\nSource:\n{INPUT_FILE}")

raw = pd.read_csv(INPUT_FILE)

print(f"\nDataset rows : {len(raw):,}")

# ---------------------------------------------------------------------
# Compare countries
# ---------------------------------------------------------------------

for country in ["Kenya", "Ethiopia", "Somalia"]:

    print("\n" + "=" * 70)
    print(country.upper())
    print("=" * 70)

    df = raw[
        raw["country"].str.lower() == country.lower()
    ].copy()

    print(f"Rows : {len(df):,}")

    # --------------------------------------------------------------
    # Does the adjusted column exist?
    # --------------------------------------------------------------

    if "fews_ipc_adjusted" not in df.columns:
        print("\nNo fews_ipc_adjusted column present.")
        print("Decision: fews_ipc must be used.")
        continue

    # --------------------------------------------------------------
    # Rows where both values are present
    # --------------------------------------------------------------

    both = df[
        ["fews_ipc", "fews_ipc_adjusted"]
    ].dropna()

    print(f"\nRows with both present : {len(both):,}")

    differing = (
        both["fews_ipc"] != both["fews_ipc_adjusted"]
    ).sum()

    print(f"Differing values       : {differing:,}")

    # --------------------------------------------------------------
    # Coverage
    # --------------------------------------------------------------

    print("\nNull counts")
    print("----------------------------")
    print(f"fews_ipc              : {df['fews_ipc'].isna().sum():,}")
    print(f"fews_ipc_adjusted     : {df['fews_ipc_adjusted'].isna().sum():,}")

    # --------------------------------------------------------------
    # Binary target comparison
    # --------------------------------------------------------------

    raw_binary = (
        pd.to_numeric(df["fews_ipc"], errors="coerce") >= 3
    )

    adj_binary = (
        pd.to_numeric(df["fews_ipc_adjusted"], errors="coerce") >= 3
    )

    raw_positive = raw_binary.sum()
    adj_positive = adj_binary.sum()

    binary_changes = (raw_binary != adj_binary).sum()

    print("\nIPC Phase 3+")
    print("----------------------------")
    print(f"Raw column            : {raw_positive:,}")
    print(f"Adjusted column       : {adj_positive:,}")
    print(f"Binary label changes  : {binary_changes:,}")

    # --------------------------------------------------------------
    # Crosstab
    # --------------------------------------------------------------

    if differing > 0:

        print("\nRaw vs Adjusted Crosstab")
        print("----------------------------")

        table = pd.crosstab(
            both["fews_ipc"],
            both["fews_ipc_adjusted"]
        )

        print(table)

print("\n" + "=" * 70)
print("Diagnostic complete.")
print("=" * 70)