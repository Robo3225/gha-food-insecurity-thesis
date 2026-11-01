"""
======================================================================

probe_lst_qc_mask.py

Confirm the cause of the 29 empty LST units. Hypothesis: the strict
QC_Day mask (bits 0-1 == 0, "good quality only") erases bright-surface
arid retrievals that are flagged "other quality" (bits 0-1 == 01),
leaving these desert units with no LST at all.

Method: for a few known-empty arid units, extract monthly-mean LST
under THREE mask conditions over several months:
  - none   : no QC mask (does the raw retrieval exist at all?)
  - relaxed: bits 0-1 <= 1 (good OR other-quality) - mirrors NDVI's
             SummaryQA <= 1
  - strict : bits 0-1 == 0 (good only) - the current mask

If 'none' and 'relaxed' return values but 'strict' returns null, the
strict QC mask is confirmed as the cause, and relaxing it is the fix.

Read-only probe. Extracts a handful of unit-months; trivial EECU cost.

Author
------
Robert Odingo MSc Thesis

======================================================================
"""

from __future__ import annotations


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ee

from config.config import PROJECT_ID
from config.countries import KENYA
from scripts.gee.extractors.gee_utils import (
    initialize,
    load_feature_collection,
)

# A few of the confirmed-empty arid units (from the diagnostic).
PROBE_FNIDS = [
    "KE2016C308010207",
    "KE2016C309030209",
    "KE2016C309040509",
]

# Months to probe (dry-season and wet-season, different years).
PROBE_MONTHS = [(2015, 2), (2015, 6), (2018, 9), (2020, 3)]

COLLECTION = "MODIS/061/MOD11A2"
SCALE = 1000


def strict_mask(img):
    qc = img.select("QC_Day")
    return img.updateMask(qc.bitwiseAnd(0b11).eq(0))


def relaxed_mask(img):
    qc = img.select("QC_Day")
    return img.updateMask(qc.bitwiseAnd(0b11).lte(1))


def monthly_mean(col, mask_fn):
    if mask_fn is not None:
        col = col.map(mask_fn)
    return col.select("LST_Day_1km").mean()


def probe():
    initialize(PROJECT_ID)
    polygons = load_feature_collection(KENYA["asset"])

    print("=" * 70)
    print("LST QC-MASK PROBE")
    print("=" * 70)
    print(f"{'FNID':<18}{'month':<9}{'none':>10}{'relaxed':>10}{'strict':>10}")
    print("-" * 70)

    for fnid in PROBE_FNIDS:
        unit = polygons.filter(ee.Filter.eq("FNID", fnid)).first()
        geom = ee.Feature(unit).geometry()

        for (y, m) in PROBE_MONTHS:
            start = ee.Date.fromYMD(y, m, 1)
            end = start.advance(1, "month")
            col = (ee.ImageCollection(COLLECTION)
                   .filterDate(start, end)
                   .select(["LST_Day_1km", "QC_Day"]))

            n_imgs = col.size().getInfo()

            def mean_over(mask_fn):
                img = monthly_mean(col, mask_fn)
                # Convert to Celsius for readability
                cel = img.multiply(0.02).subtract(273.15)
                val = cel.reduceRegion(
                    reducer=ee.Reducer.mean(), geometry=geom,
                    scale=SCALE, tileScale=4, bestEffort=True,
                ).get("LST_Day_1km")
                v = val.getInfo()
                return f"{v:.1f}" if v is not None else "null"

            none_v = mean_over(None)
            relax_v = mean_over(relaxed_mask)
            strict_v = mean_over(strict_mask)

            print(f"{fnid:<18}{y}-{m:02d}  "
                  f"{none_v:>10}{relax_v:>10}{strict_v:>10}"
                  f"   ({n_imgs} imgs)")

    print("-" * 70)
    print("Read: if 'none'/'relaxed' show values but 'strict' is null,")
    print("the strict QC mask erases these arid units -> relaxing to <=1")
    print("(good + other-quality) recovers them.")


if __name__ == "__main__":
    probe()
