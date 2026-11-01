"""
======================================================================

build_model_panel.py

Purpose
-------
Join feature panel(s) to the label panel to produce the modelling
panel a country's smoke test consumes.

Join design
-----------
Feature panels key on FNID. Label panels key on canonical_admin_code.
The two identifier systems are bridged by the persisted crosswalk
(build_crosswalk.py). The join therefore routes:

    feature[FNID] --crosswalk--> admin_code == label[canonical_admin_code]

at monthly cadence. The crosswalk is one-to-one and covers every
labelled unit, so:
  - FSC-only polygons (Mvita, Zeylac) have no crosswalk row and drop
    out when features are mapped to admin_code.
  - Every label row must receive its feature; a miss is a hard failure.

Temporal-alignment note
-----------------------
Labels are forward-filled tri-annual assessments; features are genuine
monthly observations. The join pairs each month's real feature with the
label in force that month. `is_observed` distinguishes real from
forward-filled labels and is preserved. Because forward-filling makes
labels heavily autocorrelated, the persistence baseline is the decisive
benchmark, not raw accuracy.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path

import pandas as pd

from crosswalk_config import get_country_config


COUNTRY = "Kenya"

config = get_country_config(COUNTRY)
expected_units = config["expected_units"]

PANEL_START, PANEL_END = "2018-01", "2023-12"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
CROSSWALK_FILE = (
    PROJECT_ROOT / "data" / "crosswalks"
    / f"{COUNTRY.lower()}_fnid_admin_code_crosswalk.csv"
)

LABEL_FILE = PROCESSED / f"{COUNTRY.lower()}_label_panel.csv"
FEATURE_FILES = {
    "ndvi": PROCESSED / f"{COUNTRY.lower()}_ndvi_panel.csv",
}
OUTPUT_FILE = PROCESSED / f"{COUNTRY.lower()}_model_panel.csv"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# =====================================================================
# Load crosswalk and labels
# =====================================================================

section(f"BUILD {COUNTRY.upper()} MODEL PANEL")

if not CROSSWALK_FILE.exists():
    raise RuntimeError(
        f"Crosswalk not found: {CROSSWALK_FILE}\n"
        f"Run build_crosswalk.py for {COUNTRY} first."
    )

crosswalk = pd.read_csv(CROSSWALK_FILE)[["admin_code", "FNID"]]
print(f"Crosswalk rows         : {len(crosswalk):,}")

labels = pd.read_csv(LABEL_FILE)
print(f"Label rows             : {len(labels):,}")

n_label_units = labels["canonical_admin_code"].nunique()
if n_label_units != expected_units:
    raise RuntimeError(
        f"Label panel has {n_label_units:,} units, expected "
        f"{expected_units:,}."
    )


# =====================================================================
# Join each feature panel via the crosswalk
# =====================================================================

panel = labels.copy()

for name, path in FEATURE_FILES.items():

    section(f"JOIN FEATURE: {name.upper()}")

    if not path.exists():
        raise RuntimeError(f"Feature panel not found: {path}")

    feat = pd.read_csv(path)
    print(f"{name} rows            : {len(feat):,}")
    print(f"{name} units (FNID)    : {feat['FNID'].nunique():,}")

    # Map feature FNID -> admin_code via the crosswalk. FSC-only
    # polygons (no crosswalk row) drop here.
    feat = feat.merge(crosswalk, on="FNID", how="inner")

    feat["year_month"] = (
        feat["year"].astype(int).astype(str) + "_"
        + feat["month"].astype(int).astype(str).str.zfill(2)
    )

    value_cols = [
        c for c in feat.columns
        if c not in ("FNID", "admin_code", "year", "month", "year_month",
                     "ADMIN1", "ADMIN2", "ADMIN3", "LZNAME")
    ]

    before = len(panel)
    panel = panel.merge(
        feat[["admin_code", "year_month", *value_cols]],
        left_on=["canonical_admin_code", "year_month"],
        right_on=["admin_code", "year_month"],
        how="left",
        validate="one_to_one",
    ).drop(columns=["admin_code"])

    if len(panel) != before:
        raise RuntimeError(
            f"Join changed row count {before:,} -> {len(panel):,}."
        )

    primary = "ndvi" if "ndvi" in value_cols else value_cols[0]
    n_missing = panel[primary].isna().sum()
    print(f"Label rows matched     : {(len(panel) - n_missing):,}")
    print(f"Label rows UNMATCHED   : {n_missing:,}")

    if n_missing:
        sample = panel.loc[
            panel[primary].isna(), ["canonical_admin_code", "year_month"]
        ].head(10)
        raise RuntimeError(
            f"{n_missing:,} label rows found no {name} feature. First "
            f"rows:\n{sample}"
        )
    print(f"\u2713 All label rows matched to {name}.")


# =====================================================================
# Validate and save
# =====================================================================

section("VALIDATE MODEL PANEL")

months = pd.period_range(PANEL_START, PANEL_END, freq="M")
expected_rows = expected_units * len(months)

print(f"Units                  : {panel['canonical_admin_code'].nunique():,}")
print(f"Rows                   : {len(panel):,}")
print(f"Expected rows          : {expected_rows:,}")

if len(panel) != expected_rows:
    raise RuntimeError(
        f"Model panel has {len(panel):,} rows, expected {expected_rows:,}."
    )

for target in ["phase3plus", "transition"]:
    if target in panel.columns:
        n = panel[target].notna().sum()
        pos = int(pd.to_numeric(panel[target], errors="coerce").sum())
        print(f"{target:12s} labelled : {n:,}  positives: {pos:,}")

print("\u2713 Model panel validated.")

section("SAVE")
panel.to_csv(OUTPUT_FILE, index=False)
print(f"Model panel : {OUTPUT_FILE}")
print(f"Rows        : {len(panel):,}  Columns: {len(panel.columns)}")
print(f"\nColumns: {list(panel.columns)}")
print("\n\u2713 Model panel complete.")
