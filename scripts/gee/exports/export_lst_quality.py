"""
======================================================================

export_lst_quality.py

D20-amendment C3 instrumentation: measure the per-unit MODIS LST
"other-quality" retrieval fraction over the baseline period.

This is a STANDALONE measurement export, deliberately separate from the
LST feature path (lst.py / export_monthly.py). It does NOT touch the
settled mask or the frozen feature panels - the D20 mask decision is
unchanged; this is purely additive C3 evidence.

What it measures
----------------
MOD11A2 QC_Day mandatory bits 0-1:
    0 = good quality
    1 = other quality (bright/arid surfaces get flagged here)
    2,3 = not produced (cloud / other) -> excluded from the denominator

For each unit, over the baseline window, the fraction:

    other_quality_fraction = count(bits==1) / count(bits in {0,1})

i.e. among PRODUCED retrievals (good or other), what share were
"other quality". High fraction = retrieval quality systematically
degraded for that unit (the arid-surface effect). Stored PER UNIT so
the analysis can test whether degradation concentrates in the arid
crisis units, not just report a country aggregate (D20 amendment).

Denominator note: bits 2/3 (not-produced: cloud, other) are excluded -
they are absence of retrieval, not low-quality retrieval. The fraction
is "of the retrievals we got, how many were lower-tier", which is the
retrieval-quality question C3 asks. Cloud-driven absence is a separate
axis (NDVI masking / label coverage).

Two-mode: --submit-only / --process-only. Country-parameterised.

Usage:
  python -m scripts.gee.exports.export_lst_quality --country somalia --submit-only
  python -m scripts.gee.exports.export_lst_quality --country somalia --process-only

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
from scripts.gee.exports.baseline_periods import get_baseline


COLLECTION = "MODIS/061/MOD11A2"
QC_BAND = "QC_Day"
LST_BAND = "LST_Day_1km"
SCALE = 1000
TILE_SCALE = 4
DRIVE_FOLDER = "gee_exports"


def section(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def _paths(country):
    cfg = get_country(country)
    slug = cfg["name"].lower()
    b = get_baseline("lst")               # 2000-2020, start_month 3
    root = Path(__file__).resolve().parents[3]
    name = f"lst_quality_{slug}_{b['start']}_{b['end']}"
    return {
        "cfg": cfg, "slug": slug, "b": b, "name": name,
        "download": root / "data" / "raw" / "gee_exports" / f"{name}.csv",
        "out": root / "data" / "climatology" / f"{slug}_lst_quality.csv",
    }


def _mask_counts(image):
    """
    For one image, produce two 0/1 bands over PRODUCED pixels:
      good  = 1 where bits==0
      other = 1 where bits==1
    Pixels with bits 2/3 (not produced) are masked out of BOTH, so they
    don't enter either count (correct denominator = produced only).
    """
    qc = image.select(QC_BAND)
    bits = qc.bitwiseAnd(0b11)
    produced = bits.lte(1)                      # good or other only
    good = bits.eq(0).updateMask(produced).rename("good_cnt")
    other = bits.eq(1).updateMask(produced).rename("other_cnt")
    return good.addBands(other)


def submit(P):
    initialize(PROJECT_ID)
    polygons = load_feature_collection(P["cfg"]["asset"])
    b = P["b"]
    # baseline window (respect MODIS start_month via a start date)
    start = ee.Date.fromYMD(b["start"], b.get("start_month", 1), 1)
    end = ee.Date.fromYMD(b["end"], 12, 31).advance(1, "day")

    col = (ee.ImageCollection(COLLECTION)
           .filterDate(start, end)
           .select([QC_BAND]))

    # sum good & other counts across the whole baseline, per pixel, then
    # zonally SUM over each unit -> total good & other retrievals per unit.
    counts = col.map(_mask_counts).sum()       # per-pixel totals

    reduced = counts.reduceRegions(
        collection=polygons,
        reducer=ee.Reducer.sum(),
        scale=SCALE,
        tileScale=TILE_SCALE,
    )

    def tag(f):
        return ee.Feature(None, {
            "FNID": f.get("FNID"),
            "good_cnt": f.get("good_cnt"),
            "other_cnt": f.get("other_cnt"),
        })

    out = ee.FeatureCollection(reduced).map(tag)

    task = ee.batch.Export.table.toDrive(
        collection=out, description=P["name"], folder=DRIVE_FOLDER,
        fileFormat="CSV", selectors=["FNID", "good_cnt", "other_cnt"],
    )
    task.start()
    print(f"Country      : {P['cfg']['name']} ({P['cfg']['fsc_polygons']} polygons)")
    print(f"Baseline     : {b['start']}-{b.get('start_month',1):02d} .. {b['end']}")
    print(f"\u2713 Submitted: {P['name']}  (task {task.id})")
    return task


def process(P):
    section(f"BUILD {P['cfg']['name'].upper()} LST OTHER-QUALITY FRACTION")
    if not P["download"].exists():
        raise RuntimeError(f"CSV not found: {P['download']}")

    df = pd.read_csv(P["download"])
    df["good_cnt"] = pd.to_numeric(df["good_cnt"], errors="coerce").fillna(0)
    df["other_cnt"] = pd.to_numeric(df["other_cnt"], errors="coerce").fillna(0)
    df["produced"] = df["good_cnt"] + df["other_cnt"]

    # per-unit other-quality fraction; guard divide-by-zero (a unit with
    # zero produced retrievals over the whole baseline -> undefined, NaN)
    df["other_quality_fraction"] = df["other_cnt"] / df["produced"].where(df["produced"] > 0)

    n_units = df["FNID"].nunique()
    n_zero = int((df["produced"] == 0).sum())
    frac = df["other_quality_fraction"].dropna()

    print(f"Units                    : {n_units:,}")
    print(f"Units w/ 0 produced (NaN): {n_zero}")
    print(f"other-quality fraction   : "
          f"min {frac.min():.3f}  median {frac.median():.3f}  "
          f"mean {frac.mean():.3f}  max {frac.max():.3f}")

    out = df[["FNID", "good_cnt", "other_cnt", "produced",
              "other_quality_fraction"]]
    out.to_csv(P["out"], index=False)
    section("DONE")
    print(f"Per-unit fraction : {P['out']}")
    print(f"Country mean other-quality fraction: {frac.mean():.4f}")
    print("  -> record as the C3 data-quality-gradient entry for this country")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--process-only", action="store_true")
    args = ap.parse_args()
    P = _paths(args.country)
    if args.process_only:
        process(P)
    elif args.submit_only:
        submit(P)
        print("\nDownload when COMPLETED, then --process-only.")
    else:
        submit(P)
        print("\nSubmitted; download when COMPLETED, then --process-only.")


if __name__ == "__main__":
    main()
