"""
============================================================

gee_utils.py

Common utility functions for the MSc Thesis
Google Earth Engine feature extraction pipeline.

These utilities are shared by every environmental
variable (NDVI, rainfall, LST, soil moisture, etc.)

Author:
Robert Odingo MSc Thesis

============================================================
"""

from __future__ import annotations

import warnings

import ee


# ----------------------------------------------------------
# Earth Engine
# ----------------------------------------------------------

def initialize(project_id: str) -> None:
    """
    Initialize Google Earth Engine.

    Parameters
    ----------
    project_id : str
        Google Cloud Project ID.

    Raises
    ------
    ValueError
        If project_id is empty.
    """

    if not project_id:
        raise ValueError("Project ID cannot be empty.")

    try:
        ee.Initialize(project=project_id)

    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    print("✓ Earth Engine initialized")


# ----------------------------------------------------------
# FeatureCollection
# ----------------------------------------------------------

def load_feature_collection(asset_path: str) -> ee.FeatureCollection:
    """
    Load an Earth Engine FeatureCollection.

    Parameters
    ----------
    asset_path : str
        Earth Engine asset path.

    Returns
    -------
    ee.FeatureCollection
    """

    if not asset_path:
        raise ValueError("Asset path cannot be empty.")

    fc = ee.FeatureCollection(asset_path)

    count = fc.size().getInfo()

    if count == 0:
        raise RuntimeError(
            "FeatureCollection contains no features."
        )

    print(f"✓ Loaded FeatureCollection ({count} features)")

    return fc


# ----------------------------------------------------------
# Dates
# ----------------------------------------------------------

def monthly_dates(
    year: int,
    month: int,
) -> tuple[ee.Date, ee.Date]:
    """
    Return monthly start and end dates.

    End date is exclusive.
    """

    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    return start, end


# ----------------------------------------------------------
# Generic zonal reduction
# ----------------------------------------------------------

def zonal_reduce(
    image: ee.Image,
    polygons: ee.FeatureCollection,
    reducer: ee.Reducer,
    scale: int,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """
    Reduce an image over polygons.

    Parameters
    ----------
    image : ee.Image

    polygons : ee.FeatureCollection

    reducer : ee.Reducer
        Spatial reducer. The name of the output property
        follows the reducer used, e.g. ee.Reducer.mean()
        produces "mean".

    scale : int
        Nominal scale in metres.

    tile_scale : int, default=4
        Subdivides the computation to avoid exceeding the
        Earth Engine per-tile memory limit. Raise to 8 or
        16 if "User memory limit exceeded" errors occur on
        large polygons at fine scales.

    Returns
    -------
    ee.FeatureCollection
    """

    print("✓ Running zonal reduction...")

    return image.reduceRegions(
        collection=polygons,
        reducer=reducer,
        scale=scale,
        tileScale=tile_scale,
    )


# ----------------------------------------------------------
# Backwards compatibility
# (Temporary wrapper - remove after Sprint 2)
# ----------------------------------------------------------

def zonal_mean(
    image: ee.Image,
    polygons: ee.FeatureCollection,
    scale: int = 250,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    """
    Deprecated wrapper around zonal_reduce().

    Will be removed after Sprint 2. Use zonal_reduce()
    with an explicit reducer instead.
    """

    warnings.warn(
        "zonal_mean() is deprecated and will be removed "
        "after Sprint 2. Use zonal_reduce() with an "
        "explicit reducer.",
        DeprecationWarning,
        stacklevel=2,
    )

    return zonal_reduce(
        image=image,
        polygons=polygons,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tile_scale=tile_scale,
    )


# ----------------------------------------------------------
# FeatureCollection → Python list
# ----------------------------------------------------------

def featurecollection_to_list(
    fc: ee.FeatureCollection,
) -> list[dict]:
    """
    Convert a FeatureCollection to a list of
    feature properties.

    Notes
    -----
    Uses getInfo(), which transfers the entire collection
    to the client in a single synchronous request. Earth
    Engine caps this payload at approximately 5000
    features and imposes a computation timeout.

    This is appropriate for single-country, single-month
    pilot extraction (Kenya: 640 units). For large-scale
    extraction across multiple countries and many months,
    migrate to asynchronous exports via
    ee.batch.Export.table.toDrive() or toCloudStorage().
    """

    features = fc.getInfo()["features"]

    rows = [
        feature["properties"]
        for feature in features
    ]

    return rows