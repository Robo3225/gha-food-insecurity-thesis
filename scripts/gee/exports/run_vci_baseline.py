"""
======================================================================

run_vci_baseline.py

Build the NDVI percentile climatology for VCI (Vegetation Condition
Index) from the MODIS 2000-2020 baseline.

VCI = (NDVI - NDVI_p5) / (NDVI_p95 - NDVI_p5), clipped to [0,1], per
unit per calendar month (D18). Uses the 5th/95th percentiles - robust
to outliers / residual cloud - not raw min/max.

This is the BASELINE stage: it extracts 21 years of monthly NDVI and
reduces to a per-unit per-month percentile lookup. The FEATURE stage
(run_vci_features.py) applies this lookup to the EXISTING 2018-2023
NDVI panel from Milestone 7.5 - no new modelling-window export needed.

Uses NDVI's own mask (SummaryQA <= 1), UNCHANGED by D20 (D20 is LST-
only). MODIS begins 2000-02-18, so baseline start is 2000-03 (via
baseline_periods) and export_monthly's empty-month guard covers any
gap.

Two-mode: --submit-only / --process-only.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse

import sys
import time
from pathlib import Path

import pandas as pd

import ee

from config.config import PROJECT_ID
from config.countries import get_country

from scripts.gee.extractors.gee_utils import initialize, load_feature_collection
from scripts.gee.features.ndvi import FEATURE
from scripts.gee.exports.export_monthly import build_monthly_export
from scripts.gee.exports.build_climatology import build_vci_climatology
from scripts.gee.exports.baseline_periods import get_baseline


# Resolve --country up front so the country-stamped paths below build
# correctly. Unknown country fails loud (get_country raises).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]

_b = get_baseline("ndvi")                      # 2000-2020, start_month 3
BASELINE_START, BASELINE_END = _b["start"], _b["end"]
BASELINE_LABEL = _b["label"]

DRIVE_FOLDER = "gee_exports"
EXPORT_NAME = f"vci_baseline_{COUNTRY.lower()}_{BASELINE_START}_{BASELINE_END}"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "gee_exports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_CSV = DOWNLOAD_DIR / f"{EXPORT_NAME}.csv"

CLIM_DIR = PROJECT_ROOT / "data" / "climatology"
CLIM_DIR.mkdir(parents=True, exist_ok=True)
CLIM_FILE = CLIM_DIR / f"{COUNTRY.lower()}_vci_climatology.csv"

POLL_SECONDS = 60


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def submit():
    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])
    n_months = (BASELINE_END - BASELINE_START + 1) * 12
    print(f"NDVI (VCI) baseline : {BASELINE_START}-{BASELINE_END} "
          f"({n_months} months, MODIS effective start 2000-03)")
    print(f"Units               : {_CFG['fsc_polygons']}")

    task = build_monthly_export(
        polygons=polygons, feature=FEATURE,
        start_year=BASELINE_START, end_year=BASELINE_END,
        id_property="FNID", description=EXPORT_NAME, folder=DRIVE_FOLDER,
    )
    task.start()
    print(f"\u2713 Submitted: {EXPORT_NAME}  (task {task.id})")
    return task


def wait(task):
    section("WAIT FOR EXPORT")
    t0 = time.time()
    while True:
        s = task.status(); state = s["state"]
        print(f"  [{int(time.time()-t0):>5}s] {state}")
        if state == "COMPLETED":
            print("\u2713 COMPLETED."); return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Export {state}: {s.get('error_message')}")
        time.sleep(POLL_SECONDS)


def process():
    section("BUILD VCI CLIMATOLOGY (5th/95th percentile)")

    if not EXPECTED_CSV.exists():
        raise RuntimeError(f"CSV not found: {EXPECTED_CSV}")

    raw = pd.read_csv(EXPECTED_CSV)
    reducer_col = FEATURE["reducer_output"]
    raw["ndvi"] = raw[reducer_col] * FEATURE["multiplier"] + FEATURE.get("offset", 0.0)
    print(f"Baseline rows   : {len(raw):,}")

    v = raw["ndvi"].dropna()
    print(f"NDVI range      : {v.min():.3f} .. {v.max():.3f}")

    clim = build_vci_climatology(
        raw, value_col="ndvi", unit_col="FNID",
        lo_pct=5.0, hi_pct=95.0, baseline_label=BASELINE_LABEL,
    )

    section("CLIMATOLOGY COVERAGE")
    n_units = clim["FNID"].nunique()
    print(f"Units           : {n_units:,}")
    print(f"Climatology rows: {len(clim):,} (expect units x 12 = {n_units*12:,})")
    print("\nn_years per cell:")
    print(clim["n_years"].value_counts().sort_index().to_string())

    # degenerate range check: p95==p5 means no VCI dynamic range
    degenerate = clim[(clim["ndvi_hi"] - clim["ndvi_lo"]).abs() < 1e-6]
    if len(degenerate):
        print(f"\n\u26a0 {len(degenerate)} cells have p5==p95 (no NDVI range); "
              f"VCI there is undefined and will map to neutral 0.5 - flag.")

    if len(clim) != n_units * 12:
        raise RuntimeError(
            f"Climatology has {len(clim):,} rows, expected {n_units*12:,}."
        )

    clim.to_csv(CLIM_FILE, index=False)
    section("DONE")
    print(f"VCI climatology : {CLIM_FILE}")


if __name__ == "__main__":
    if "--process-only" in sys.argv:
        process()
    elif "--submit-only" in sys.argv:
        submit()
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        t = submit(); wait(t)
        print("\nDownload the CSV, then --process-only.")
