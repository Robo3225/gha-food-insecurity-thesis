"""
======================================================================

build_label_panel.py

Purpose
-------
Construct the monthly IPC label panel for a country: join FEWS NET
IPC assessments to the standardized reporting units, forward-fill the
tri-annual cadence to monthly, and derive both LEVEL and TRANSITION
binary targets.

Output is the label side of the modelling panel. Features (NDVI,
rainfall, LST) are joined to it separately.

Decisions encoded here
----------------------
TARGET COLUMN = `fews_ipc`, not `fews_ipc_adjusted`.
    Verified (Feb-2026 snapshot): the adjusted column is the phase that
    would apply WITHOUT humanitarian assistance. The relationship is
    deterministic - `fews_ha == 1` implies adjustment, always, in all
    three countries, with the phase revised upward by exactly one.
    `fews_ipc` is used because (a) the C1 benchmark comparator
    `fews_proj_near` has no adjusted variant, so training on the
    adjusted column would compare two different quantities, and (b) the
    operational early-warning signal is what FEWS NET will classify a
    unit as, not the counterfactual absent aid.

    CONSEQUENCE TO DOCUMENT: the target already reflects humanitarian
    response, so attributions may partly capture where assistance was
    delivered rather than solely where conditions deteriorated.

    `fews_ha` is deliberately EXCLUDED as a feature - it encodes
    assistance presence and would leak.

FORWARD-FILL is forward-only and capped at FFILL_LIMIT months.
    Backward-filling would use a future assessment to label a past
    month, which is leakage. The cap prevents a gap in the series from
    propagating a stale label indefinitely.

BOTH TARGETS are emitted:
    `phase3plus`  - LEVEL:      is the unit in Phase 3+ this month?
    `transition`  - TRANSITION: did it ENTER Phase 3+ this month?
    Level on forward-filled labels is heavily autocorrelated and may be
    near-trivial; transitions are far rarer. Class balance for both is
    reported so the modelling target can be chosen on evidence.

ALIAS RESOLUTION: where one reporting unit carries several admin_codes
    (Kenya has one such unit), rows are folded onto the canonical code
    taking the MAX IPC per period. Rationale: in a humanitarian
    early-warning context under-predicting crisis is the costlier
    error, so the conservative reading is taken. Affects 3 periods.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from crosswalk_config import get_country_config


# =====================================================================
# Configuration
# =====================================================================

COUNTRY = "Somalia"  # Change this to the country you want to process

# Panel window. 72 months (2018-01..2023-12) gives ~18 tri-annual label
# observations per unit vs ~15 at 60, and covers the 2022 Horn of Africa
# drought. To reach back to the 2016-17 drought as well, set START to
# "2017-01" (84 months) - but extend the feature extraction to match.
START = "2018-01"
END = "2023-12"

# Max months a single assessment may be carried forward. Tri-annual
# cadence is ~4 months, so 4 covers a normal interval without letting
# a genuine gap propagate stale labels.
FFILL_LIMIT = 4

TARGET_COLUMN = "fews_ipc"
CRISIS_THRESHOLD = 3          # IPC Phase 3+ = crisis

config = get_country_config(COUNTRY)
join_key = config["join_key"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEWS_FILE = (
    PROJECT_ROOT / "data" / "raw" / "world_bank"
    / "FEWS_February_2026_Update_TrueBoundaries_03-15-26.csv"
)

INTERIM = PROJECT_ROOT / "data" / "interim"
UNITS_FILE = INTERIM / f"{COUNTRY.lower()}_fews_reporting_units.csv"
ALIAS_FILE = INTERIM / f"{COUNTRY.lower()}_admin_code_aliases.csv"

PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED / f"{COUNTRY.lower()}_label_panel.csv"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# Load reporting units and alias table
# =====================================================================

section(f"BUILD {COUNTRY.upper()} LABEL PANEL")

if not UNITS_FILE.exists():
    raise RuntimeError(
        f"Reporting-unit lookup not found: {UNITS_FILE}\n"
        f"Run standardize_fews_reporting_units.py for {COUNTRY} first."
    )

units = pd.read_csv(UNITS_FILE)
print(f"Reporting units        : {len(units):,}")

expected = config["expected_units"]
if len(units) != expected:
    raise RuntimeError(
        f"Reporting-unit lookup has {len(units):,} rows, expected "
        f"{expected:,}. Re-run standardization."
    )

alias = pd.read_csv(ALIAS_FILE)
n_alias = (alias["n_codes"] > 1).sum() if "n_codes" in alias.columns else 0
print(f"admin_code aliases     : {len(alias):,} "
      f"({n_alias} belonging to multi-code units)")


# =====================================================================
# Load labels
# =====================================================================

section("LOAD LABELS")

raw = pd.read_csv(FEWS_FILE)

labels = raw.loc[
    raw["country"].str.lower() == COUNTRY.lower(),
    ["admin_code", "year_month", TARGET_COLUMN, "fews_ha"],
].copy()

print(f"Label rows ({COUNTRY})  : {len(labels):,}")

labels["ym"] = pd.to_datetime(
    labels["year_month"], format="%Y_%m", errors="coerce"
)

if labels["ym"].isna().any():
    bad = labels.loc[labels["ym"].isna(), "year_month"].unique()[:5]
    raise RuntimeError(f"Unparseable year_month values: {list(bad)}")

labels[TARGET_COLUMN] = pd.to_numeric(labels[TARGET_COLUMN], errors="coerce")

print(f"Null {TARGET_COLUMN:<17}: {labels[TARGET_COLUMN].isna().sum():,} "
      f"({labels[TARGET_COLUMN].isna().mean():.2%})")


# =====================================================================
# Resolve admin_code aliases
# =====================================================================

section("RESOLVE ALIASES")

labels = labels.merge(
    alias[["admin_code", "canonical_admin_code"]],
    on="admin_code",
    how="left",
)

unmapped = labels["canonical_admin_code"].isna().sum()
if unmapped:
    raise RuntimeError(
        f"{unmapped:,} label rows have an admin_code absent from the "
        f"alias table. The lookup and the label file disagree."
    )

before = len(labels)

# MAX per (canonical unit, month): conservative resolution of the rare
# case where two codes for one unit disagree.
labels = (
    labels.groupby(["canonical_admin_code", "ym"], as_index=False)
          .agg(**{TARGET_COLUMN: (TARGET_COLUMN, "max"),
                  "fews_ha": ("fews_ha", "max")})
)

collapsed = before - len(labels)
print(f"Rows before            : {before:,}")
print(f"Rows after alias merge : {len(labels):,}")
print(f"Collapsed              : {collapsed:,}")


# =====================================================================
# Build monthly spine and forward-fill
# =====================================================================

section("FORWARD-FILL TO MONTHLY")

months = pd.date_range(f"{START}-01", f"{END}-01", freq="MS")
print(f"Panel window           : {START} to {END} ({len(months)} months)")

units_keyed = units.rename(
    columns={"admin_code": "canonical_admin_code"}
)[["canonical_admin_code", *join_key]].drop_duplicates(
    subset="canonical_admin_code"
)

# Full unit x month grid
spine = (
    units_keyed.assign(_k=1)
    .merge(pd.DataFrame({"ym": months, "_k": 1}), on="_k")
    .drop(columns="_k")
)

panel = spine.merge(
    labels[["canonical_admin_code", "ym", TARGET_COLUMN, "fews_ha"]],
    on=["canonical_admin_code", "ym"],
    how="left",
)

panel = panel.sort_values(["canonical_admin_code", "ym"]).reset_index(drop=True)

panel["is_observed"] = panel[TARGET_COLUMN].notna()

# Forward-fill ONLY (no bfill: future labels must not label the past).
panel["ipc_filled"] = (
    panel.groupby("canonical_admin_code")[TARGET_COLUMN]
         .ffill(limit=FFILL_LIMIT)
)

obs_rate = panel["is_observed"].mean()
fill_rate = panel["ipc_filled"].notna().mean()

print(f"Observed unit-months   : {panel['is_observed'].sum():,} "
      f"({obs_rate:.1%})")
print(f"After forward-fill     : {panel['ipc_filled'].notna().sum():,} "
      f"({fill_rate:.1%})")
print(f"Unlabelled             : {panel['ipc_filled'].isna().sum():,} "
      f"({1 - fill_rate:.1%})")
print(f"\n  Unlabelled months are before a unit's first assessment or in\n"
      f"  gaps longer than the {FFILL_LIMIT}-month fill limit.")


# =====================================================================
# Derive targets
# =====================================================================

section("DERIVE TARGETS")

# LEVEL
panel["phase3plus"] = (panel["ipc_filled"] >= CRISIS_THRESHOLD).astype("Int64")
panel.loc[panel["ipc_filled"].isna(), "phase3plus"] = pd.NA

# TRANSITION: entering crisis from non-crisis
panel["prev_phase3plus"] = panel.groupby("canonical_admin_code")["phase3plus"].shift(1)
panel["transition"] = (
    (panel["prev_phase3plus"] == 0) & (panel["phase3plus"] == 1)
).astype("Int64")
panel.loc[
    panel["phase3plus"].isna() | panel["prev_phase3plus"].isna(),
    "transition",
] = pd.NA

# --- class balance -------------------------------------------------

lvl = panel["phase3plus"].dropna()
trn = panel["transition"].dropna()

print("LEVEL  (in Phase 3+ this month)")
print(f"  labelled unit-months : {len(lvl):,}")
print(f"  positives            : {int(lvl.sum()):,} ({lvl.mean():.2%})")

print("\nTRANSITION  (entered Phase 3+ this month)")
print(f"  labelled unit-months : {len(trn):,}")
print(f"  positives            : {int(trn.sum()):,} ({trn.mean():.2%})")

# Absolute counts matter more than rates when positives are scarce.
if int(trn.sum()) < 200:
    print(f"\n  WARNING: only {int(trn.sum()):,} transition positives. "
          f"This may be too few\n  to train on at country level - "
          f"pooling countries may be required.")

# Observed-only balance: what the labels look like before forward-fill
# inflates autocorrelation.
obs_only = panel.loc[panel["is_observed"], "phase3plus"].dropna()
print(f"\nOBSERVED-ONLY (no forward-fill)")
print(f"  labelled unit-months : {len(obs_only):,}")
print(f"  positives            : {int(obs_only.sum()):,} ({obs_only.mean():.2%})")


# =====================================================================
# Persistence baseline reference
# =====================================================================

section("PERSISTENCE BASELINE")

# "Predict the same as last month." Forward-filling creates heavy
# autocorrelation, so a model must beat this to have learned anything.
valid = panel.dropna(subset=["phase3plus", "prev_phase3plus"])
persistence_acc = (valid["phase3plus"] == valid["prev_phase3plus"]).mean()

print(f"Persistence accuracy   : {persistence_acc:.2%}")
print(f"  (predicting last month's level unchanged)")
print(f"\n  Any model scoring at or below this has learned nothing\n"
      f"  beyond label autocorrelation. This is the decisive\n"
      f"  diagnostic for the TFT smoke test.")


# =====================================================================
# Write panel
# =====================================================================

section("SAVE PANEL")

out = panel[[
    "canonical_admin_code", *join_key, "ym",
    TARGET_COLUMN, "ipc_filled", "is_observed",
    "phase3plus", "transition", "fews_ha",
]].copy()

out["year_month"] = out["ym"].dt.strftime("%Y_%m")

out.to_csv(OUTPUT_FILE, index=False)

print(f"Rows written           : {len(out):,}")
print(f"Output                 : {OUTPUT_FILE}")


# =====================================================================
# Summary
# =====================================================================

section("SUMMARY")

print(f"Country                : {COUNTRY}")
print(f"Target column          : {TARGET_COLUMN}")
print(f"Window                 : {START} to {END} ({len(months)} months)")
print(f"Reporting units        : {len(units_keyed):,}")
print(f"Unit-months            : {len(panel):,}")
print(f"Labelled (post-ffill)  : {panel['ipc_filled'].notna().sum():,} "
      f"({fill_rate:.1%})")
print(f"Level positives        : {int(lvl.sum()):,} ({lvl.mean():.2%})")
print(f"Transition positives   : {int(trn.sum()):,} ({trn.mean():.2%})")
print(f"Persistence accuracy   : {persistence_acc:.2%}")

print("\n\u2713 Label panel complete.")
