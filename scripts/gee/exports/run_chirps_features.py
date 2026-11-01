"""
======================================================================

run_chirps_features.py

CHIRPS modelling-window SPI feature extraction (2018-2023).

REVISED for D18 SPI (2026-08-06). Extracts monthly rainfall with a
2-month lead-in (2017-11), accumulates SPI-3, applies the frozen SPI
reference (run_chirps_baseline.py), then TRIMS to the 72-month
modelling window (2018-01 .. 2023-12).

Produces:
  - rainfall  : monthly total precipitation (mm), unit mean
  - spi3      : Standardized Precipitation Index, 3-month
  - spi3_empirical : bool, True where the empirical fallback was used

Two-mode:
  --submit-only  : submit the export (2017-11 .. 2023-12), exit
  --process-only : accumulate SPI-3, apply reference, trim, write panel

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

import ee

from config.config import PROJECT_ID
from config.countries import get_country

from scripts.gee.extractors.gee_utils import initialize, load_feature_collection
from scripts.gee.features.chirps import FEATURE
from scripts.gee.exports.export_monthly import build_monthly_export
from scripts.gee.exports.build_spi import accumulate, apply_spi


# Resolve --country up front so the country-stamped paths below build
# correctly. Unknown country fails loud (get_country raises).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]

# Modelling window (D7) and its 2-month SPI-3 lead-in.
MODEL_START = (2018, 1)
MODEL_END = (2023, 12)
EXTRACT_START = (2017, 11)     # lead-in for SPI-3 at 2018-01
EXTRACT_END = (2023, 12)
SPI_MONTHS = 3

DRIVE_FOLDER = "gee_exports"
EXPORT_NAME = f"chirps_features_{COUNTRY.lower()}_{EXTRACT_START[0]}{EXTRACT_START[1]:02d}_{EXTRACT_END[0]}"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "gee_exports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_CSV = DOWNLOAD_DIR / f"{EXPORT_NAME}.csv"

CLIM_DIR = PROJECT_ROOT / "data" / "climatology"
REF_FILE = CLIM_DIR / f"{COUNTRY.lower()}_chirps_spi_reference.csv"
EMP_FILE = CLIM_DIR / f"{COUNTRY.lower()}_chirps_spi_empirical.pkl"

PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
PANEL_FILE = PROCESSED / f"{COUNTRY.lower()}_chirps_panel.csv"

POLL_SECONDS = 45


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def submit():
    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])
    task = build_monthly_export(
        polygons=polygons, feature=FEATURE,
        start_year=EXTRACT_START[0], end_year=EXTRACT_END[0],
        id_property="FNID", description=EXPORT_NAME, folder=DRIVE_FOLDER,
    )
    task.start()
    print(f"\u2713 Submitted: {EXPORT_NAME}  (task {task.id})")
    print(f"  extraction {EXTRACT_START[0]}-{EXTRACT_END[0]}, "
          f"trim to {MODEL_START[0]}-{MODEL_END[0]}")
    return task


def wait(task):
    section("WAIT FOR EXPORT")
    t0 = time.time()
    while True:
        s = task.status(); state = s["state"]
        print(f"  [{int(time.time()-t0):>5}s] {state}")
        if state == "COMPLETED":
            print("\u2713 Export complete."); return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Export {state}: {s.get('error_message')}")
        time.sleep(POLL_SECONDS)


def process():
    section("BUILD CHIRPS SPI FEATURE PANEL")

    for f in (EXPECTED_CSV, REF_FILE, EMP_FILE):
        if not Path(f).exists():
            raise RuntimeError(f"Required input missing: {f}")

    raw = pd.read_csv(EXPECTED_CSV)
    reducer_col = FEATURE["reducer_output"]
    raw["rainfall"] = raw[reducer_col] * FEATURE.get("multiplier", 1.0) + FEATURE.get("offset", 0.0)

    # trim any pre-lead-in months a whole-year export included
    raw = raw[~((raw["year"] == EXTRACT_START[0]) & (raw["month"] < EXTRACT_START[1]))]
    print(f"Extracted rows (incl 2-mo lead-in): {len(raw):,}")

    # accumulate SPI-3 across the full extracted span (lead-in seeds 2018-01)
    raw = accumulate(raw, SPI_MONTHS, "rainfall")

    # apply the frozen reference
    ref = pd.read_csv(REF_FILE)
    with open(EMP_FILE, "rb") as f:
        emp = pickle.load(f)

    panel = apply_spi(raw, f"acc{SPI_MONTHS}", ref, emp,
                      out_col="spi3", unit_col="FNID")

    # --- TRIM to the 72-month modelling window ---
    section("TRIM TO MODELLING WINDOW")
    before = len(panel)
    panel = panel[
        (panel["year"] * 100 + panel["month"] >= MODEL_START[0] * 100 + MODEL_START[1])
        & (panel["year"] * 100 + panel["month"] <= MODEL_END[0] * 100 + MODEL_END[1])
    ].reset_index(drop=True)

    n_units = panel["FNID"].nunique()
    n_months = 72
    print(f"Rows before trim : {before:,}  after: {len(panel):,}")
    print(f"Grid             : {n_units} units x {n_months} months = {n_units*n_months:,}")
    if len(panel) != n_units * n_months:
        raise RuntimeError(
            f"Panel has {len(panel):,} rows, expected {n_units*n_months:,}."
        )

    # SPI-3 must be complete after trim (lead-in guarantees 2018-01)
    n_nan = panel["spi3"].isna().sum()
    if n_nan:
        raise RuntimeError(
            f"{n_nan} SPI-3 values are NaN after trim - lead-in failed to "
            f"seed the panel head."
        )
    print("\u2713 SPI-3 complete for all 72 months (lead-in worked).")

    # --- report ---
    section("SPI-3 SUMMARY")
    print(f"SPI-3 range      : {panel['spi3'].min():.2f} .. {panel['spi3'].max():.2f}")
    print(f"SPI-3 mean / sd  : {panel['spi3'].mean():.3f} / {panel['spi3'].std():.3f}")
    emp_used = int(panel["spi3_empirical"].sum())
    print(f"Empirical-fallback rows: {emp_used:,} ({emp_used/len(panel):.1%})")
    for yr in [2020, 2021, 2022, 2023]:
        m = panel[panel["year"] == yr]["spi3"].mean()
        print(f"  {yr} mean SPI-3 : {m:+.2f}")

    out = panel[["FNID", "year", "month", "rainfall", "spi3", "spi3_empirical"]]
    out.to_csv(PANEL_FILE, index=False)

    section("DONE")
    print(f"CHIRPS SPI panel : {PANEL_FILE}")
    print(f"Rows             : {len(out):,}")


if __name__ == "__main__":
    if "--process-only" in sys.argv:
        process()
    elif "--submit-only" in sys.argv:
        submit()
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        t = submit(); wait(t)
        print("\nDownload the CSV, then --process-only.")
