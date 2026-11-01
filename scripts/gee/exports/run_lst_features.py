"""
======================================================================

run_lst_features.py

MODIS LST modelling-window feature extraction (2018-2023).

Extracts monthly QC-masked daytime LST over the 72-month window, then
computes the z-score anomaly against the frozen 2000-2020 climatology
(D18; z-score retained for temperature, not SPI).

Produces:
  - lst_celsius : monthly mean daytime LST (Celsius), unit mean
  - lst_anomaly : (lst - clim_mean) / clim_sd, per unit per calendar
                  month, against the 2000-2020 baseline

No lead-in / accumulation (unlike SPI-3): the anomaly is a per-month
standardization, so the extraction window equals the modelling window.

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
from scripts.gee.features.lst import FEATURE
from scripts.gee.exports.export_monthly import build_monthly_export
from scripts.gee.exports.build_climatology import apply_anomaly


# Resolve --country up front so the country-stamped paths below build
# correctly. Unknown country fails loud (get_country raises).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]
START_YEAR, END_YEAR = 2018, 2023

DRIVE_FOLDER = "gee_exports"
EXPORT_NAME = f"lst_features_{COUNTRY.lower()}_{START_YEAR}_{END_YEAR}"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "gee_exports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_CSV = DOWNLOAD_DIR / f"{EXPORT_NAME}.csv"

CLIM_FILE = (
    PROJECT_ROOT / "data" / "climatology"
    / f"{COUNTRY.lower()}_lst_climatology.csv"
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
PANEL_FILE = PROCESSED / f"{COUNTRY.lower()}_lst_panel.csv"

POLL_SECONDS = 45


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def submit():
    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])
    print(f"Modelling window : {START_YEAR}-{END_YEAR}")
    task = build_monthly_export(
        polygons=polygons, feature=FEATURE,
        start_year=START_YEAR, end_year=END_YEAR,
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
            print("\u2713 Export complete."); return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Export {state}: {s.get('error_message')}")
        time.sleep(POLL_SECONDS)


def process():
    section("BUILD LST FEATURE PANEL")

    for f in (EXPECTED_CSV, CLIM_FILE):
        if not Path(f).exists():
            raise RuntimeError(f"Required input missing: {f}")

    raw = pd.read_csv(EXPECTED_CSV)
    reducer_col = FEATURE["reducer_output"]
    raw["lst_celsius"] = (
        raw[reducer_col] * FEATURE["multiplier"] + FEATURE.get("offset", 0.0)
    )
    print(f"Modelling-window rows : {len(raw):,}")

    # completeness grid
    n_months = (END_YEAR - START_YEAR + 1) * 12
    n_units = raw["FNID"].nunique()
    if len(raw) != n_units * n_months:
        raise RuntimeError(
            f"Export has {len(raw):,} rows, expected {n_units*n_months:,} "
            f"({n_units} x {n_months}). Missing unit-months."
        )
    print(f"Grid complete         : {n_units} x {n_months} = {len(raw):,}")

    # NOTE on cloud gaps: LST is QC-masked, so some unit-months may be
    # null (fully cloudy). apply_anomaly requires a climatology entry for
    # every row, which exists (12 months per unit); nulls in the VALUE
    # propagate to a null anomaly and are reported, not filled here -
    # left for the temporal-feature / interpolation step to handle
    # consistently with the NDVI cloud-gap discipline (D11).
    n_null = raw["lst_celsius"].isna().sum()
    if n_null:
        print(f"\u26a0 {n_null} unit-months are QC-masked null (cloudy). "
              f"Anomaly will be null there; handled downstream (D11-style).")

    section("APPLY LST ANOMALY (z-score vs 2000-2020 climatology)")
    clim = pd.read_csv(CLIM_FILE)
    panel = apply_anomaly(
        raw, clim, value_col="lst_celsius", out_col="lst_anomaly",
        unit_col="FNID",
    )

    a = panel["lst_anomaly"].dropna()
    print(f"lst_celsius range : {panel['lst_celsius'].min():.1f} .. "
          f"{panel['lst_celsius'].max():.1f} C")
    print(f"anomaly range     : {a.min():.2f} .. {a.max():.2f} SD")
    print(f"anomaly mean / sd : {a.mean():.3f} / {a.std():.3f}")
    # Drought years should show POSITIVE LST anomaly (hotter surface).
    for yr in [2020, 2021, 2022, 2023]:
        m = panel[panel["year"] == yr]["lst_anomaly"].mean()
        print(f"  {yr} mean LST anomaly: {m:+.2f} "
              f"(drought years expected warm / positive)")

    out = panel[["FNID", "year", "month", "lst_celsius", "lst_anomaly"]]
    out.to_csv(PANEL_FILE, index=False)
    section("DONE")
    print(f"LST feature panel : {PANEL_FILE}")
    print(f"Rows              : {len(out):,}")


if __name__ == "__main__":
    if "--process-only" in sys.argv:
        process()
    elif "--submit-only" in sys.argv:
        submit()
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        t = submit(); wait(t)
        print("\nDownload the CSV, then --process-only.")
