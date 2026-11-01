"""
======================================================================

run_ndvi_extraction.py

Purpose
-------
End-to-end runner for composite-resolution NDVI extraction:

    1. initialize GEE
    2. load reporting-unit polygons
    3. submit a batch export of 16-day composites to Drive
    4. poll the task to completion
    5. read the exported CSV
    6. apply scaling, aggregate to a monthly panel
    7. assert the panel is a complete units x months grid
    8. write the monthly panel and a temporal-support report

Default window is a SMOKE TEST (2020 only, ~23 composites) so the
export shape and composite_date tagging can be verified cheaply before
committing to the full 72-month job. Set SMOKE_TEST = False for the
full 2018-2023 window.

Two-stage discipline: run the smoke window first, confirm the panel is
639 x 12 with sane NDVI ranges, THEN switch to the full window.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import argparse

import ee

from config.config import PROJECT_ID
from config.countries import get_country

from scripts.gee.extractors.gee_utils import (
    initialize,
    load_feature_collection,
)
from scripts.gee.features.ndvi import FEATURE, apply_quality_mask
from scripts.gee.exports.export_composites import build_composite_export
from scripts.gee.exports.aggregate_composites import (
    aggregate_to_monthly,
    report_temporal_support,
    assert_panel_complete,
    interpolate_masked,
    assert_no_nulls,
)


# =====================================================================
# Configuration
# =====================================================================

# Resolve --country up front so the country-stamped paths below build
# correctly. Unknown country fails loud (get_country raises).
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--country", required=True)
_country_arg, _ = _ap.parse_known_args()
_CFG = get_country(_country_arg.country)
COUNTRY = _CFG["name"]
EXPECTED_UNITS = _CFG["fsc_polygons"]


SMOKE_TEST = False   # 2020 only. Set False for full 2018-2023.

if SMOKE_TEST:
    START, END = "2020-01-01", "2021-01-01"     # end exclusive
    PANEL_START, PANEL_END = "2020-01", "2020-12"
else:
    START, END = "2018-01-01", "2024-01-01"
    PANEL_START, PANEL_END = "2018-01", "2023-12"

DRIVE_FOLDER = "gee_exports"
EXPORT_NAME = f"ndvi_composites_{COUNTRY.lower()}_{PANEL_START}_{PANEL_END}"

PROJECT_ROOT = Path(__file__).resolve().parents[2]  
DOWNLOAD_DIR = PROJECT_ROOT / "data" / "raw" / "gee_exports"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# The runner does NOT auto-download from Drive (that needs Drive API
# credentials). It waits for the task, then expects the CSV to be
# placed here manually or via a Drive sync. This keeps credentials out
# of the pipeline and makes the handoff explicit.
EXPECTED_CSV = DOWNLOAD_DIR / f"{EXPORT_NAME}.csv"

PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
PANEL_FILE = PROCESSED / f"{COUNTRY.lower()}_ndvi_panel.csv"
SUPPORT_FILE = PROCESSED / f"{COUNTRY.lower()}_ndvi_temporal_support.csv"
MASKED_FILE = PROCESSED / f"{COUNTRY.lower()}_ndvi_masked_cells.csv"

POLL_SECONDS = 30


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# =====================================================================
# Stage 1-3: submit export
# =====================================================================

def submit_export() -> "ee.batch.Task":
    initialize(PROJECT_ID)
    polygons = load_feature_collection(_CFG["asset"])

    task = build_composite_export(
        polygons=polygons,
        feature=FEATURE,
        start=START,
        end=END,
        mask_fn=apply_quality_mask,
        id_property="FNID",
        description=EXPORT_NAME,
        folder=DRIVE_FOLDER,
    )
    task.start()
    print(f"\u2713 Export submitted: {EXPORT_NAME}")
    print(f"  window     : {START} .. {END} (end exclusive)")
    print(f"  drive folder: {DRIVE_FOLDER}")
    print(f"  task id    : {task.id}")
    return task


def wait_for(task) -> None:
    section("WAIT FOR EXPORT")
    while True:
        status = task.status()
        state = status["state"]
        print(f"  {state}")
        if state in ("COMPLETED",):
            print("\u2713 Export complete.")
            return
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"Export {state}: {status.get('error_message', 'no message')}"
            )
        time.sleep(POLL_SECONDS)


# =====================================================================
# Stage 5-8: local processing
# =====================================================================

def process_local() -> None:
    section("READ EXPORT")

    if not EXPECTED_CSV.exists():
        raise RuntimeError(
            f"Exported CSV not found at:\n{EXPECTED_CSV}\n"
            f"Download '{EXPORT_NAME}.csv' from Drive folder "
            f"'{DRIVE_FOLDER}' to that path, then re-run with "
            f"--process-only."
        )

    composites = pd.read_csv(EXPECTED_CSV)
    print(f"Composite rows : {len(composites):,}")

    reducer_col = FEATURE["reducer_output"]
    if reducer_col not in composites.columns:
        raise RuntimeError(
            f"Reducer column '{reducer_col}' absent from export. "
            f"Columns: {list(composites.columns)}"
        )

    # --- apply scaling/offset locally (single source of truth) ------
    offset = FEATURE.get("offset", 0.0)
    composites[FEATURE["output"]] = (
        composites[reducer_col] * FEATURE["multiplier"] + offset
    )

    # --- aggregate composites -> monthly ----------------------------
    section("AGGREGATE TO MONTHLY")
    monthly = aggregate_to_monthly(
        composites,
        value_col=FEATURE["output"],
        date_col="composite_date",
        unit_col="FNID",
        agg="mean",
    )
    print(f"Monthly rows   : {len(monthly):,}")

    # --- temporal support report ------------------------------------
    support = report_temporal_support(monthly)
    print("\nComposites-per-month distribution:")
    print(support.to_string(index=False))
    support.to_csv(SUPPORT_FILE, index=False)

    # --- sanity: NDVI range -----------------------------------------
    v = monthly[FEATURE["output"]].dropna()
    print(f"\n{FEATURE['output']} range : {v.min():.3f} .. {v.max():.3f}")
    if not (-0.2 <= v.min() and v.max() <= 1.0):
        print("\u26a0 NDVI outside expected [-0.2, 1.0]. Check scaling.")

    # --- completeness assertion -------------------------------------
    section("VALIDATE PANEL")
    assert_panel_complete(
        monthly,
        expected_units=EXPECTED_UNITS,
        start=PANEL_START,
        end=PANEL_END,
        unit_col="FNID",
    )
    print(f"\u2713 Complete grid: {EXPECTED_UNITS} units x months.")

    # --- fill fully-masked unit-months (within-unit interpolation) ---
    section("INTERPOLATE MASKED UNIT-MONTHS")
    monthly, masked_report = interpolate_masked(
        monthly, value_col=FEATURE["output"], unit_col="FNID", limit=1,
    )
    n_masked = len(masked_report)
    n_filled = int(masked_report["interpolated"].sum())
    print(f"Originally masked   : {n_masked}")
    print(f"Interpolated        : {n_filled}")
    print(f"Still missing       : {n_masked - n_filled}")
    if n_masked:
        masked_report.to_csv(MASKED_FILE, index=False)
        print(f"Masked-cell report  : {MASKED_FILE}")

    # D11 permits a SMALL flagged residual of unfillable gaps: multi-
    # month or series-edge masked cells that single-month linear
    # interpolation cannot defensibly fill (D11 reversal condition).
    # These remain NaN and propagate as legitimate value-nulls (like
    # LST cloudy cells), distinguished downstream from join-failures by
    # the assembler's merge indicator. Guard against a LARGE residual,
    # which would signal a structural problem (a heavily-masked unit or
    # a bad export) rather than scattered cloud.
    still_null = monthly[monthly[FEATURE["output"]].isna()]
    n_still = len(still_null)
    null_frac = n_still / len(monthly) if len(monthly) else 0.0
    NULL_TOLERANCE = 0.005          # 0.5% ceiling; scattered cloud only

    # Flag the unfillable cells so they are auditable in the panel.
    monthly["ndvi_unfillable"] = monthly[FEATURE["output"]].isna()

    if n_still:
        print(f"\u26a0 {n_still} unit-months remain null after interpolation "
              f"({null_frac:.2%}) - multi-month/edge gaps (D11). Flagged "
              f"'ndvi_unfillable', kept NaN for downstream handling:")
        print(still_null[["FNID", "year", "month"]].to_string(index=False))

    if null_frac > NULL_TOLERANCE:
        raise RuntimeError(
            f"{n_still} nulls ({null_frac:.2%}) exceed the "
            f"{NULL_TOLERANCE:.1%} tolerance. Too many to be scattered "
            f"cloud - investigate for a heavily-masked unit or a bad "
            f"export before proceeding."
        )
    print(f"\u2713 Null residual within tolerance ({n_still} cells, "
          f"{null_frac:.2%} <= {NULL_TOLERANCE:.1%}); flagged and "
          f"permitted per D11.")

    monthly.to_csv(PANEL_FILE, index=False)
    section("DONE")
    print(f"Panel   : {PANEL_FILE}")
    print(f"Support : {SUPPORT_FILE}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import sys

    if "--process-only" in sys.argv:
        # Skip submission; CSV already downloaded.
        process_local()
    elif "--submit-only" in sys.argv:
        submit_export()
        print("\nExport running. Download the CSV when COMPLETED, then "
              "re-run with --process-only.")
    else:
        task = submit_export()
        wait_for(task)
        print("\nExport complete on GEE's side. It writes to Drive "
              "asynchronously; allow a moment, then download "
              f"'{EXPORT_NAME}.csv' to:\n{EXPECTED_CSV}\n"
              "and re-run with --process-only.")
