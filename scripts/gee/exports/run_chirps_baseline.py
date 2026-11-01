"""
======================================================================

run_chirps_baseline.py

D18 stage-1: build the CHIRPS SPI reference for a country from the
1991-2020 baseline (zero-adjusted mixed gamma, per-unit per-calendar-
month). Parameterised by --country (KE/ET/SO); no hardcoded country.

Two-mode:
  --submit-only   : submit the baseline export (1990-11..2020-12), exit
  --process-only  : fit the SPI reference from the downloaded CSV

Usage:
  python -m scripts.gee.exports.run_chirps_baseline --country ethiopia --submit-only
  python -m scripts.gee.exports.run_chirps_baseline --country ethiopia --process-only

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import ee

from config.config import PROJECT_ID
from config.countries import get_country

from scripts.gee.extractors.gee_utils import initialize, load_feature_collection
from scripts.gee.features.chirps import FEATURE
from scripts.gee.exports.export_monthly import build_monthly_export
from scripts.gee.exports.build_spi import accumulate, fit_spi_reference, build_empirical_samples
from scripts.gee.exports.baseline_periods import get_baseline


# Baseline span with 2-month lead-in for symmetric SPI-3 across the
# 1991-2020 reference (extraction 1990-11 -> 2020-12).
_b = get_baseline("chirps")
BASELINE_START, BASELINE_END = _b["start"], _b["end"]
BASELINE_LABEL = _b["label"]
LEADIN_START = (BASELINE_START - 1, 11)   # 1990-11
SPI_MONTHS = 3

POLL_SECONDS = 60


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def paths_for(country):
    """All country-specific paths/names derived from the config."""
    c = get_country(country)
    slug = c["name"].lower()
    root = Path(__file__).resolve().parents[3]
    return {
        "cfg": c,
        "slug": slug,
        "export_name": f"chirps_baseline_{slug}_{LEADIN_START[0]}_{BASELINE_END}",
        "download": root / "data" / "raw" / "gee_exports"
                    / f"chirps_baseline_{slug}_{LEADIN_START[0]}_{BASELINE_END}.csv",
        "ref_out": root / "data" / "climatology" / f"{slug}_chirps_spi_reference.csv",
        "emp_out": root / "data" / "climatology" / f"{slug}_chirps_spi_empirical.pkl",
    }


def submit(P):
    initialize(PROJECT_ID)
    polygons = load_feature_collection(P["cfg"]["asset"])
    print(f"Country      : {P['cfg']['name']} ({P['cfg']['fsc_polygons']} FSC polygons)")
    print(f"Extraction   : {LEADIN_START[0]}-{LEADIN_START[1]:02d} .. {BASELINE_END}-12 "
          f"(reference fit on {BASELINE_START}-{BASELINE_END})")
    # build_monthly_export iterates whole years. The 2-month SPI-3 lead-in
    # is achieved by extracting the whole lead-in year (1990) and letting
    # the SPI accumulation/trim handle the month boundaries downstream -
    # matching the working Kenya baseline run (no start_month arg exists).
    task = build_monthly_export(
        polygons=polygons, feature=FEATURE,
        start_year=LEADIN_START[0], end_year=BASELINE_END,
        id_property="FNID", description=P["export_name"], folder="gee_exports",
    )
    task.start()
    print(f"\u2713 Submitted: {P['export_name']}  (task {task.id})")
    return task


def process(P):
    section(f"BUILD {P['cfg']['name'].upper()} CHIRPS SPI REFERENCE")
    if not P["download"].exists():
        raise RuntimeError(f"CSV not found: {P['download']}")

    import pickle

    raw = pd.read_csv(P["download"])
    reducer_col = FEATURE["reducer_output"]
    raw["rainfall"] = (raw[reducer_col] * FEATURE.get("multiplier", 1.0)
                       + FEATURE.get("offset", 0.0))
    print(f"Baseline rows (incl 2-mo lead-in): {len(raw):,}")

    # accumulate SPI-3 (3-month sums) BEFORE fitting the reference; the
    # reference is fit on the accumulation column, not raw rainfall.
    raw = accumulate(raw, SPI_MONTHS, "rainfall")
    acc_col = f"acc{SPI_MONTHS}"

    # fit_spi_reference returns a SINGLE DataFrame (the gamma/q reference);
    # build_empirical_samples then builds the empirical fallback dict.
    ref = fit_spi_reference(raw, acc_col, unit_col="FNID")
    emp = build_empirical_samples(raw, acc_col, ref, unit_col="FNID")

    section("SPI REFERENCE COVERAGE")
    n_units = ref["FNID"].nunique()
    print(f"Units            : {n_units:,}")
    print(f"Reference cells  : {len(ref):,} (expect units x 12 = {n_units*12:,})")
    if "method" in ref.columns:
        print(f"\nfit method per cell:\n{ref['method'].value_counts().to_string()}")
        n_emp = (ref["method"] == "empirical").sum()
        print(f"\nEmpirical-fallback cells: {n_emp} ({n_emp/len(ref)*100:.1f}%)")

    if len(ref) != n_units * 12:
        raise RuntimeError(f"reference has {len(ref)} rows, expected {n_units*12}")

    ref.to_csv(P["ref_out"], index=False)
    with open(P["emp_out"], "wb") as fh:
        pickle.dump(emp, fh)
    section("DONE")
    print(f"SPI reference    : {P['ref_out']}")
    print(f"Empirical samples: {P['emp_out']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True,
                    help="Country name or code (kenya/ethiopia/somalia, KE/ET/SO)")
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--process-only", action="store_true")
    args = ap.parse_args()

    P = paths_for(args.country)   # resolves + validates country up front

    if args.process_only:
        process(P)
    elif args.submit_only:
        submit(P)
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        t = submit(P)
        print("\nSubmitted. Download when COMPLETED, then --process-only.")


if __name__ == "__main__":
    main()
