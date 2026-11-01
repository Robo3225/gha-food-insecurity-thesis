"""
======================================================================

run_lst_baseline.py

D18 stage-1: build the MODIS LST z-score climatology for Kenya from the
2000-2020 baseline.

LST uses the z-score anomaly (mean/SD), NOT SPI - D18 retains the
z-score for temperature because LST is approximately Gaussian and
well-behaved, unlike zero-inflated rainfall. So this reuses the
build_climatology mean/SD path, not build_spi.

No SPI-3 accumulation and therefore NO lead-in is needed: the LST
anomaly is a per-month standardization, not a rolling accumulation. The
baseline is simply 2000-2020 (D18, via baseline_periods).

Two-mode:
  --submit-only  : submit the baseline export (2000-2020), exit
  --process-only : build the mean/SD climatology, save it

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
from scripts.gee.exports.build_climatology import build_anomaly_climatology
from scripts.gee.exports.baseline_periods import get_baseline


# Resolve --country up front so the country-stamped paths below build
# correctly. Unknown country fails loud (get_country raises).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]

_b = get_baseline("lst")                       # 2000-2020 (guarded)
BASELINE_START, BASELINE_END = _b["start"], _b["end"]
BASELINE_LABEL = _b["label"]

DRIVE_FOLDER = "gee_exports"
EXPORT_NAME = f"lst_baseline_{COUNTRY.lower()}_{BASELINE_START}_{BASELINE_END}"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "gee_exports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_CSV = DOWNLOAD_DIR / f"{EXPORT_NAME}.csv"

CLIM_DIR = PROJECT_ROOT / "data" / "climatology"
CLIM_DIR.mkdir(parents=True, exist_ok=True)
CLIM_FILE = CLIM_DIR / f"{COUNTRY.lower()}_lst_climatology.csv"

POLL_SECONDS = 60


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def submit():
    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])
    n_months = (BASELINE_END - BASELINE_START + 1) * 12
    print(f"LST baseline    : {BASELINE_START}-{BASELINE_END} "
          f"({n_months} months)")
    print(f"Units           : {_CFG['fsc_polygons']}")

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
    section("BUILD LST CLIMATOLOGY (z-score, mean/SD)")

    if not EXPECTED_CSV.exists():
        raise RuntimeError(f"CSV not found: {EXPECTED_CSV}")

    raw = pd.read_csv(EXPECTED_CSV)
    print(f"Baseline rows   : {len(raw):,}")

    reducer_col = FEATURE["reducer_output"]
    if reducer_col not in raw.columns:
        raise RuntimeError(
            f"Reducer column '{reducer_col}' absent. Columns: {list(raw.columns)}"
        )

    # LST scaling: Kelvin -> Celsius via multiplier 0.02 + offset -273.15
    raw["lst_celsius"] = (
        raw[reducer_col] * FEATURE["multiplier"] + FEATURE.get("offset", 0.0)
    )

    # sanity: Celsius LST for Kenya should be roughly 0-55 C at surface
    v = raw["lst_celsius"].dropna()
    print(f"LST range       : {v.min():.1f} .. {v.max():.1f} C")
    if not (-20 <= v.min() and v.max() <= 75):
        print("\u26a0 LST outside plausible surface range - check QC mask / scaling.")

    clim = build_anomaly_climatology(
        raw, value_col="lst_celsius", unit_col="FNID",
        baseline_label=BASELINE_LABEL,
    )

    section("CLIMATOLOGY COVERAGE")
    n_units = clim["FNID"].nunique()
    print(f"Units           : {n_units:,}")
    print(f"Climatology rows: {len(clim):,} (expect units x 12 = {n_units*12:,})")
    print(f"\nn_years per cell:")
    print(clim["n_years"].value_counts().sort_index().to_string())

    expected_years = BASELINE_END - BASELINE_START + 1
    thin = clim[clim["n_years"] < expected_years - 1]
    if len(thin):
        print(f"\n\u26a0 {len(thin)} cells have < {expected_years-1} years "
              f"(LST is QC-masked; some thinning expected - methodology note).")
    else:
        print(f"\n\u2713 All cells ~{expected_years} years.")

    # --- D20 per-country recovery figure (C3 evidence) --------------
    # Under the relaxed mask (bits <= 1), how many units still have ANY
    # fully-empty calendar month? Compare to the strict-mask baseline,
    # where 29 crisis-active arid units were empty across all 12 months.
    # This per-country count is a transferability finding (D17/D20): LST
    # retrieval quality degrades over bright arid surfaces, worst in the
    # most arid (and most food-insecure) contexts.
    section("D20 LST RECOVERY (C3 EVIDENCE)")
    empty_now = clim[clim["n_years"] == 0]
    units_fully_empty = (
        empty_now.groupby("FNID").size().pipe(lambda s: (s == 12).sum())
    )
    print(f"Country               : {COUNTRY}")
    print(f"Mask                  : QC_Day bits <= 1 (D20, relaxed)")
    print(f"Units still fully empty (all 12 mo): {units_fully_empty}")
    print(f"Cells with n_years==0 : {len(empty_now)}")
    print("  -> Record this alongside the strict-mask count (Kenya strict: "
          "29 units) as the per-country recovery figure. Repeat for "
          "Ethiopia and Somalia when their baselines run; Somalia (most "
          "arid) is expected to show the largest strict->relaxed recovery, "
          "which is the C3 data-structure-axis finding.")

    if len(clim) != n_units * 12:
        raise RuntimeError(
            f"Climatology has {len(clim):,} rows, expected {n_units*12:,}. "
            f"Some unit-months have no baseline LST at all (persistent "
            f"cloud) - investigate before proceeding."
        )

    clim.to_csv(CLIM_FILE, index=False)
    section("DONE")
    print(f"LST climatology : {CLIM_FILE}")


if __name__ == "__main__":
    if "--process-only" in sys.argv:
        process()
    elif "--submit-only" in sys.argv:
        submit()
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        t = submit(); wait(t)
        print("\nDownload the CSV, then --process-only.")
