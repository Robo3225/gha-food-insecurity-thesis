"""
============================================================

ndvi.py

Feature definition for MODIS NDVI.

This feature extracts monthly NDVI from the
MODIS MOD13Q1 Version 6.1 product.

Quality masking is applied using the SummaryQA
band before monthly aggregation.

Methodological note
-------------------
MOD13Q1 is a 16-day composite product, not a daily
product. Filtering by calendar month therefore selects
composites whose START date falls within the month,
typically yielding two images per month. Monthly NDVI
is consequently the mean of the 16-day composites
beginning in that month, and a composite may span the
month boundary. This convention is consistent with
common practice in the food security remote sensing
literature and is documented in the methodology.

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
    Apply MODIS SummaryQA mask.

    SummaryQA values

    0 = Good quality
    1 = Marginal quality
    2 = Snow / Ice
    3 = Cloud

    We retain only good and marginal observations.
    """

    qa = image.select("SummaryQA")

    mask = qa.lte(1)

    return image.updateMask(mask)


# ----------------------------------------------------------
# Monthly Aggregation
# ----------------------------------------------------------

def aggregate_monthly(
    collection: ee.ImageCollection,
) -> ee.Image:
    """
    Apply QA masking before computing
    the monthly mean NDVI.

    The QA band is required for masking and is dropped
    after the mask is applied, so that only the NDVI
    band reaches the zonal reduction.
    """

    collection = collection.map(apply_quality_mask)

    return collection.select("NDVI").mean()


# ----------------------------------------------------------
# Feature Definition
# ----------------------------------------------------------

FEATURE = {

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    "name": "ndvi",

    "description": (
        "MODIS Normalized Difference Vegetation Index"
    ),

    # ------------------------------------------------------
    # Earth Engine
    # ------------------------------------------------------

    "collection": "MODIS/061/MOD13Q1",

    # NDVI is the signal band.
    # SummaryQA is required for quality masking and is
    # dropped inside aggregate_monthly().
    "bands": [
        "NDVI",
        "SummaryQA",
    ],

    # ------------------------------------------------------
    # Spatial resolution
    # ------------------------------------------------------

    "scale": 250,

    # ------------------------------------------------------
    # MODIS scale factor
    # ------------------------------------------------------

    "multiplier": 0.0001,

    # No additive offset required for NDVI.

    # ------------------------------------------------------
    # Monthly aggregation
    # ------------------------------------------------------

    "aggregation": aggregate_monthly,

    # ------------------------------------------------------
    # Spatial reducer
    # ------------------------------------------------------

     # Lazy reducer construction.
      # The reducer is created only after Earth Engine has been
     # initialized by the extraction pipeline.
     "reducer": lambda: ee.Reducer.mean(),

    # Property name produced by reduceRegions for this
    # reducer. Declared explicitly so the engine does not
    # assume a mean reduction.
    "reducer_output": "mean",

    # ------------------------------------------------------
    # Output column
    # ------------------------------------------------------

    "output": "ndvi",

}