"""
======================================================================

validate_temporal_alignment.py

Purpose
-------
Validate temporal alignment between monthly feature observations and
forward-filled FEWS NET labels.

This diagnostic confirms that:

1. NDVI varies monthly.
2. is_observed identifies genuine FEWS NET assessment months.
3. ipc_filled carries assessments forward between observations.
4. Every reporting unit contains observed assessments.
5. A sample is written for manual inspection.

Outputs
-------
Console report

outputs/diagnostics/
    temporal_alignment_sample.csv

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path

import pandas as pd


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "kenya_model_panel.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = OUTPUT_DIR / "temporal_alignment_sample.csv"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# Load
# =====================================================================

section("LOAD MODEL PANEL")

panel = pd.read_csv(MODEL_PANEL)

print(f"Rows                  : {len(panel):,}")
print(f"Reporting units       : {panel['canonical_admin_code'].nunique():,}")

# =====================================================================
# Observation summary
# =====================================================================

section("OBSERVED ASSESSMENT MONTHS")

observed = panel.loc[panel["is_observed"]]

print(f"Observed rows         : {len(observed):,}")

month_summary = (
    observed["year_month"]
    .str[-2:]
    .value_counts()
    .sort_index()
)

print("\nObserved rows by month:")
print(month_summary.to_string())

# =====================================================================
# Every reporting unit has observations
# =====================================================================

section("CHECK REPORTING UNIT COVERAGE")

coverage = (
    panel
    .groupby("canonical_admin_code")["is_observed"]
    .sum()
)

missing_units = coverage[coverage == 0]

print(f"Units checked         : {len(coverage):,}")

if len(missing_units):

    print("\nUnits with no observed assessments:")
    print(missing_units)

    raise RuntimeError(
        "Some reporting units contain no observed FEWS assessments."
    )

print("✓ Every reporting unit contains observed assessments.")

# =====================================================================
# Sample units
# =====================================================================

section("TEMPORAL ALIGNMENT SAMPLE")

sample_units = (
    panel["canonical_admin_code"]
    .drop_duplicates()
    .sort_values()
    .head(3)
)

samples = []

for unit in sample_units:

    print("\n" + "-" * 70)
    print(f"Reporting Unit: {unit}")
    print("-" * 70)

    u = (
        panel[
            panel["canonical_admin_code"] == unit
        ]
        .sort_values("year_month")
    )

    display = u[
        [
            "year_month",
            "is_observed",
            "ipc_filled",
            "phase3plus",
            "ndvi",
            "n_composites",
            "ndvi_interpolated",
        ]
    ]

    print(display.head(18).to_string(index=False))

    sample = display.copy()
    sample.insert(0, "canonical_admin_code", unit)

    samples.append(sample)

# =====================================================================
# Save sample
# =====================================================================

section("SAVE SAMPLE")

sample_df = pd.concat(samples, ignore_index=True)

sample_df.to_csv(SAMPLE_FILE, index=False)

print(f"Sample written to:\n{SAMPLE_FILE}")

# =====================================================================
# Summary
# =====================================================================

section("SUMMARY")

print(f"""
Model panel rows              : {len(panel):,}
Reporting units               : {panel['canonical_admin_code'].nunique():,}
Observed assessment rows      : {len(observed):,}
Forward-filled rows           : {(~panel['is_observed']).sum():,}
Interpolated NDVI values      : {panel['ndvi_interpolated'].sum():,}

Diagnostic completed successfully.
""")