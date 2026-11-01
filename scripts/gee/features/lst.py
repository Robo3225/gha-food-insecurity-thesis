"""
============================================================

lst.py

Feature definition for MODIS land surface temperature.

This feature extracts monthly mean daytime land surface
temperature from the MODIS MOD11A2 Version 6.1 product.

Scaling note
------------
MOD11A2 LST is stored as a 16-bit integer in Kelvin with
a scale factor of 0.02. Conversion to degrees Celsius
therefore requires BOTH a multiplicative scale factor
and an additive offset:

    Celsius = DN * 0.02 - 273.15

This is the reason the feature schema supports an
optional "offset" key in addition to "multiplier".

Quality note
------------
MOD11A2 carries a QC_Day bitmask rather than the simple
SummaryQA enumeration used by MOD13Q1. Bits 0-1 encode
mandatory QA: 0 indicates good quality. The mask below
retains only pixels where those bits are zero.

Author:
Robert Odingo MSc Thesis

============================================================
"""

from __future__ import annotations

import ee


# ----------------------------------------------------------
# Quality Mask
# ----------------------------------------------------------

def apply_quality_mask(image: ee.Image) -> ee.Image:
    """
    Apply MOD11A2 QC_Day mandatory quality mask.

    QC_Day bits 0-1 (mandatory QA flags)

    0 = LST produced, good quality
    1 = LST produced, other quality
    2 = LST not produced, cloud effects
    3 = LST not produced, other reasons

    We retain bits 0-1 <= 1: BOTH "good" and "other quality"
    retrievals (D20). The stricter "good only" (== 0) mask erased 29
    crisis-active arid units across the entire baseline, because MODIS
    flags bright, hot, low-emissivity desert surfaces as "other
    quality" rather than "good" - a probe confirmed the retrievals
    exist and are physically plausible (38-46 C), just labelled
    lower-tier. <= 1 recovers them and matches the NDVI mask's
    SummaryQA <= 1 (good + marginal), aligning both MODIS features to
    the same quality threshold. Tradeoff: "other quality" LST is
    noisier; the alternative, however, is no LST at all in the arid
    crisis core, which is worse for a food-insecurity model whose
    signal concentrates there.
    """

    qc = image.select("QC_Day")

    mask = qc.bitwiseAnd(0b11).lte(1)

    return image.updateMask(mask)


# ----------------------------------------------------------
# Monthly Aggregation
# ----------------------------------------------------------

def aggregate_monthly(
    collection: ee.ImageCollection,
) -> ee.Image:
    """
    Apply QA masking before computing the monthly mean
    daytime land surface temperature.
    """

    collection = collection.map(apply_quality_mask)

    return collection.select("LST_Day_1km").mean()


# ----------------------------------------------------------
# Feature Definition
# ----------------------------------------------------------

FEATURE = {

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    "name": "lst",

    "description": (
        "MODIS daytime land surface temperature "
        "(degrees Celsius)"
    ),

    # ------------------------------------------------------
    # Earth Engine
    # ------------------------------------------------------

    "collection": "MODIS/061/MOD11A2",

    "bands": [
        "LST_Day_1km",
        "QC_Day",
    ],

    # ------------------------------------------------------
    # Spatial resolution
    # ------------------------------------------------------

    "scale": 1000,

    # ------------------------------------------------------
    # Scaling: Kelvin -> Celsius
    # ------------------------------------------------------

    "multiplier": 0.02,

    "offset": -273.15,

    # ------------------------------------------------------
    # Monthly aggregation
    # ------------------------------------------------------

    "aggregation": aggregate_monthly,

    # ------------------------------------------------------
    # Spatial reducer
    # ------------------------------------------------------

    "reducer": lambda:ee.Reducer.mean(),

    "reducer_output": "mean",

    # ------------------------------------------------------
    # Output column
    # ------------------------------------------------------

    "output": "lst_celsius",

}