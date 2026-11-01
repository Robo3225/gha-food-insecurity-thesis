"""
============================================================

chirps.py

Feature definition for CHIRPS monthly precipitation.

This feature extracts total monthly rainfall from the
CHIRPS Daily Version 2.0 product.

Version note
------------
CHIRPS v2.0 is used rather than v3 to preserve direct
comparability with the closest published benchmark
studies (Busker et al., 2024; Westerveld et al., 2021),
both of which use v2.0. Using a different precipitation
product would mean that any performance difference
against those benchmarks is partly attributable to input
data rather than model architecture. CHIRPS v3 may be
evaluated as an appendix robustness check.

Aggregation note
----------------
Monthly rainfall is a TEMPORAL SUM of daily values,
followed by a SPATIAL MEAN over the reporting unit.
The temporal sum is performed in aggregate_monthly();
the spatial mean is performed by the zonal reducer.
Reducing daily images with a mean would produce mean
daily rainfall, which differs from monthly total by
roughly a factor of thirty.

Author:
Robert Odingo MSc Thesis

============================================================
"""

from __future__ import annotations

import ee


# ----------------------------------------------------------
# Monthly Aggregation
# ----------------------------------------------------------

def aggregate_monthly(
    collection: ee.ImageCollection,
) -> ee.Image:
    """
    Sum daily precipitation to a monthly total.

    CHIRPS carries no quality assurance band, so no
    masking is applied.
    """

    return collection.select("precipitation").sum()


# ----------------------------------------------------------
# Feature Definition
# ----------------------------------------------------------

FEATURE = {

    # ------------------------------------------------------
    # General
    # ------------------------------------------------------

    "name": "rainfall",

    "description": (
        "CHIRPS v2.0 total monthly precipitation (mm)"
    ),

    # ------------------------------------------------------
    # Earth Engine
    # ------------------------------------------------------

    "collection": "UCSB-CHG/CHIRPS/DAILY",

    "bands": [
        "precipitation",
    ],

    # ------------------------------------------------------
    # Spatial resolution
    # ------------------------------------------------------
    #
    # CHIRPS native resolution is approximately 5566 m.

    "scale": 5566,

    # ------------------------------------------------------
    # Scaling
    # ------------------------------------------------------
    #
    # CHIRPS precipitation is already in millimetres.

    "multiplier": 1.0,

    # ------------------------------------------------------
    # Monthly aggregation (temporal sum)
    # ------------------------------------------------------

    "aggregation": aggregate_monthly,

    # ------------------------------------------------------
    # Spatial reducer (spatial mean of the monthly total)
    # ------------------------------------------------------

    "reducer": lambda: ee.Reducer.mean(),

    "reducer_output": "mean",

    # ------------------------------------------------------
    # Output column
    # ------------------------------------------------------

    "output": "rainfall_mm",

}